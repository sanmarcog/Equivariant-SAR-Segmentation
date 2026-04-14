"""
src/data/preprocess.py

Scene-wide preprocessing pipeline. Applied once per scene before patching.
Produces a [12, H, W] float32 array in the following channel order:

  0  VH_post   — post-event VH backscatter, LIA-normalised, Lee-filtered, clipped [−25,−5] dB
  1  VV_post   — post-event VV backscatter, LIA-normalised, Lee-filtered, clipped [−25,−5] dB
  2  slope     — terrain slope (degrees)
  3  sin_asp   — sin(aspect)
  4  cos_asp   — cos(aspect)
  5  LIA       — local incidence angle (degrees)
  6  VH_pre    — pre-event VH, same processing as VH_post
  7  VV_pre    — pre-event VV, same processing as VV_post
  8  log_ratio_VH — log(VH_post / VH_pre)   multiplicative speckle suppression
  9  log_ratio_VV — log(VV_post / VV_pre)
 10  xpol_post — VH_post / VV_post           cross-pol ratio, post
 11  xpol_pre  — VH_pre  / VV_pre            cross-pol ratio, pre

Processing order:
  1. Refined Lee 5×5 speckle filter on VH and VV (pre and post)
  2. LIA normalisation on all SAR bands (multiply by cos(ref_angle) / cos(LIA))
  3. Convert to dB   if not already (values are assumed to be σ⁰ linear or dB — see _to_db)
  4. Compute log-ratio and cross-pol ratio channels  (in linear domain BEFORE dB clipping)
  5. Clip SAR (VH/VV, pre+post) to [−25, −5] dB
  6. Normalise all 12 channels with train-split statistics (applied by dataset.py, not here)

NaN / nodata handling: every NaN → 0.0 before returning.

Usage:
    from src.data.preprocess import preprocess_scene, CHANNEL_NAMES

    arr12 = preprocess_scene(scene_dir)   # np.ndarray [12, H, W] float32
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import uniform_filter, generic_filter

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAR_MIN_DB: float = -25.0
SAR_MAX_DB: float = -5.0

# Reference angle for LIA normalisation (standard cosine correction)
# cos(ref) / cos(LIA) — ref = 30° matches Sentinel-1 IW nominal incidence
LIA_REF_DEG: float = 30.0

CHANNEL_NAMES = [
    "VH_post", "VV_post", "slope", "sin_asp", "cos_asp", "LIA",
    "VH_pre",  "VV_pre",
    "log_ratio_VH", "log_ratio_VV",
    "xpol_post", "xpol_pre",
]

# Expected file suffixes (matches AvalCD naming convention)
_SUFFIXES = {
    "postVH": "postVH.tif",
    "postVV": "postVV.tif",
    "preVH":  "preVH.tif",
    "preVV":  "preVV.tif",
    "SLP":    "SLP.tif",
    "ASP":    "ASP.tif",
    "LIA":    "LIA.tif",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_band(path: Path) -> np.ndarray:
    """Read first band of a GeoTIFF as float32. NaN for nodata pixels."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
    return arr


def _find_scene_file(scene_dir: Path, suffix: str) -> Path:
    """Locate a file whose name ends with *suffix inside scene_dir."""
    candidates = list(scene_dir.glob(f"*{suffix}"))
    if not candidates:
        raise FileNotFoundError(
            f"No file matching '*{suffix}' found in {scene_dir}"
        )
    return candidates[0]


def _refined_lee_5x5(arr: np.ndarray) -> np.ndarray:
    """
    Refined Lee speckle filter (5×5 kernel, single-look approximation).

    This is a simplified Refined Lee filter:
      - Estimate local mean and variance via uniform filtering (5×5).
      - Weight: w = var_signal / (var_signal + var_noise)
        where var_noise ≈ mean² / ENL  (ENL=1 for single-look GRD)
      - Filtered = mean + w * (pixel − mean)

    Operates in LINEAR power domain. Input values that look like dB
    (typically < 0) are left as-is; the caller is responsible for
    passing linear-scale SAR values if using the physical Refined Lee.
    For the AvalCD data (already dB from SNAP), we apply the filter
    directly in dB — a common practical approximation.

    NaN pixels are preserved (fill with local mean).
    """
    window = 5
    # Replace NaN with local mean before filtering to avoid NaN propagation
    nan_mask = np.isnan(arr)
    arr_clean = arr.copy()
    if nan_mask.any():
        local_mean_fill = uniform_filter(np.where(nan_mask, 0.0, arr), size=window)
        count_fill = uniform_filter((~nan_mask).astype(np.float32), size=window)
        count_fill = np.maximum(count_fill, 1e-8)
        arr_clean[nan_mask] = (local_mean_fill / count_fill)[nan_mask]

    local_mean = uniform_filter(arr_clean, size=window)
    local_sq   = uniform_filter(arr_clean ** 2, size=window)
    local_var  = np.maximum(local_sq - local_mean ** 2, 0.0)

    # Noise variance: σ²_noise = μ² / ENL  (ENL=1 for single-look GRD)
    noise_var  = local_mean ** 2

    weight     = local_var / np.maximum(local_var + noise_var, 1e-10)
    filtered   = local_mean + weight * (arr_clean - local_mean)

    # Restore NaN positions
    filtered[nan_mask] = np.nan
    return filtered


def _lia_normalise(
    sar_db: np.ndarray,
    lia_deg: np.ndarray,
    ref_deg: float = LIA_REF_DEG,
) -> np.ndarray:
    """
    Cosine LIA normalisation in linear domain:
        σ⁰_norm = σ⁰_linear × cos(ref) / cos(LIA)

    Applied in linear, then converted back to dB.
    NaN LIA pixels → no correction (identity).
    """
    # Convert dB → linear
    linear = 10.0 ** (sar_db / 10.0)

    cos_ref = np.cos(np.deg2rad(ref_deg))
    cos_lia = np.cos(np.deg2rad(lia_deg))

    # Avoid divide-by-zero at extreme LIA (near 90°)
    cos_lia_safe = np.where(np.abs(cos_lia) < 1e-3, np.nan, cos_lia)
    ratio = cos_ref / cos_lia_safe

    # Cap ratio to avoid wild corrections at invalid angles
    ratio = np.clip(ratio, 0.1, 10.0)

    # Where LIA is NaN (nodata), keep original
    ratio = np.where(np.isnan(ratio), 1.0, ratio)

    linear_norm = linear * ratio
    # Back to dB, guard log(0)
    db_norm = 10.0 * np.log10(np.maximum(linear_norm, 1e-10))
    # Restore original NaN mask from sar_db
    db_norm[np.isnan(sar_db)] = np.nan
    return db_norm.astype(np.float32)


def _log_ratio(sar_post_db: np.ndarray, sar_pre_db: np.ndarray) -> np.ndarray:
    """
    Log-ratio change channel:
        log_ratio = (SAR_post_dB − SAR_pre_dB) / 10
    which equals log10(σ⁰_post_linear / σ⁰_pre_linear).

    Division in log domain → subtraction. Divided by 10 to keep in
    a ≈[−2, +2] range rather than [−20, +20] dB.
    NaN → 0.
    """
    ratio = (sar_post_db - sar_pre_db) / 10.0
    return np.nan_to_num(ratio, nan=0.0).astype(np.float32)


def _xpol_ratio(vh_db: np.ndarray, vv_db: np.ndarray) -> np.ndarray:
    """
    Cross-pol ratio in dB: VH − VV  (dB subtraction = linear division).
    Values centred around negative (VH typically < VV for snow/ice).
    NaN → 0.
    """
    ratio = vh_db - vv_db
    return np.nan_to_num(ratio, nan=0.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def preprocess_scene(scene_dir: str | Path) -> np.ndarray:
    """
    Preprocess one AvalCD scene into a 12-channel float32 array [12, H, W].

    Args:
        scene_dir: Path to a scene directory containing *postVH.tif, etc.

    Returns:
        arr: np.ndarray [12, H, W] float32, channel order per CHANNEL_NAMES.
             All NaN → 0.0.  Normalisation NOT applied here (done in dataset).
    """
    scene_dir = Path(scene_dir)
    log.info("Preprocessing scene: %s", scene_dir.name)

    # ── 1. Load raw rasters ────────────────────────────────────────────────
    vh_post_raw = _read_band(_find_scene_file(scene_dir, _SUFFIXES["postVH"]))
    vv_post_raw = _read_band(_find_scene_file(scene_dir, _SUFFIXES["postVV"]))
    vh_pre_raw  = _read_band(_find_scene_file(scene_dir, _SUFFIXES["preVH"]))
    vv_pre_raw  = _read_band(_find_scene_file(scene_dir, _SUFFIXES["preVV"]))
    slope_raw   = _read_band(_find_scene_file(scene_dir, _SUFFIXES["SLP"]))
    asp_raw     = _read_band(_find_scene_file(scene_dir, _SUFFIXES["ASP"]))
    lia_raw     = _read_band(_find_scene_file(scene_dir, _SUFFIXES["LIA"]))

    # ── 2. Refined Lee 5×5 (applied in dB domain — practical approximation) ─
    vh_post = _refined_lee_5x5(vh_post_raw)
    vv_post = _refined_lee_5x5(vv_post_raw)
    vh_pre  = _refined_lee_5x5(vh_pre_raw)
    vv_pre  = _refined_lee_5x5(vv_pre_raw)

    # ── 3. LIA normalisation ───────────────────────────────────────────────
    # Only where LIA is valid (non-NaN)
    lia_valid = np.nan_to_num(lia_raw, nan=LIA_REF_DEG)  # fallback: identity
    vh_post = _lia_normalise(vh_post, lia_valid)
    vv_post = _lia_normalise(vv_post, lia_valid)
    vh_pre  = _lia_normalise(vh_pre,  lia_valid)
    vv_pre  = _lia_normalise(vv_pre,  lia_valid)

    # ── 4. Engineered change channels (before dB clipping) ─────────────────
    log_ratio_vh = _log_ratio(vh_post, vh_pre)
    log_ratio_vv = _log_ratio(vv_post, vv_pre)
    xpol_post    = _xpol_ratio(vh_post, vv_post)
    xpol_pre     = _xpol_ratio(vh_pre,  vv_pre)

    # ── 5. Clip SAR to [−25, −5] dB ───────────────────────────────────────
    vh_post = np.clip(vh_post, SAR_MIN_DB, SAR_MAX_DB)
    vv_post = np.clip(vv_post, SAR_MIN_DB, SAR_MAX_DB)
    vh_pre  = np.clip(vh_pre,  SAR_MIN_DB, SAR_MAX_DB)
    vv_pre  = np.clip(vv_pre,  SAR_MIN_DB, SAR_MAX_DB)

    # ── 6. Terrain channels ────────────────────────────────────────────────
    slope   = slope_raw
    sin_asp = np.sin(np.deg2rad(asp_raw)).astype(np.float32)
    cos_asp = np.cos(np.deg2rad(asp_raw)).astype(np.float32)
    lia     = lia_raw

    # ── 7. Stack → [12, H, W] ─────────────────────────────────────────────
    # SAR channels (0,1,6,7): NaN border pixels filled with SAR_MIN_DB
    # so they stay within the valid dB range and don't contaminate z-scores.
    # Non-SAR channels: NaN → 0.0 (reasonable neutral value post-normalisation).
    def _fill(arr: np.ndarray, fill: float) -> np.ndarray:
        return np.nan_to_num(arr, nan=fill, posinf=fill, neginf=fill)

    channels = [
        _fill(vh_post,       SAR_MIN_DB),   # 0  VH_post
        _fill(vv_post,       SAR_MIN_DB),   # 1  VV_post
        _fill(slope_raw,     0.0),          # 2  slope
        _fill(sin_asp,       0.0),          # 3  sin_asp
        _fill(cos_asp,       0.0),          # 4  cos_asp
        _fill(lia,           LIA_REF_DEG),  # 5  LIA  (fill with ref angle → ratio=1)
        _fill(vh_pre,        SAR_MIN_DB),   # 6  VH_pre
        _fill(vv_pre,        SAR_MIN_DB),   # 7  VV_pre
        _fill(log_ratio_vh,  0.0),          # 8  log_ratio_VH  (no change → 0)
        _fill(log_ratio_vv,  0.0),          # 9  log_ratio_VV
        _fill(xpol_post,     0.0),          # 10 xpol_post
        _fill(xpol_pre,      0.0),          # 11 xpol_pre
    ]
    arr = np.stack(channels, axis=0)
    return arr.astype(np.float32)


def load_gt_mask(scene_dir: str | Path) -> np.ndarray:
    """Load binary GT mask [H, W] uint8 (1=deposit, 0=background)."""
    scene_dir = Path(scene_dir)
    gt_path   = _find_scene_file(scene_dir, "GT.tif")
    with rasterio.open(gt_path) as src:
        mask = src.read(1).astype(np.uint8)
    return mask


def load_scene_meta(scene_dir: str | Path) -> dict:
    """Return CRS, transform, and shape from the postVH raster."""
    scene_dir = Path(scene_dir)
    with rasterio.open(_find_scene_file(scene_dir, "postVH.tif")) as src:
        return {
            "crs":       src.crs,
            "transform": src.transform,
            "shape":     (src.height, src.width),
        }
