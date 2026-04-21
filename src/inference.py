"""
src/inference.py

Patch-based inference with sliding window, TTA, and threshold sweep.

Algorithm:
  1. Preprocess scene → [12, H, W] float32
  2. Slide 64×64 window over scene with stride 16 (75% overlap)
  3. For each patch: run 4-fold TTA (identity, hflip, vflip, both)
  4. Average logits across TTA variants
  5. Stitch patch logits into full-scene probability map via average pooling
     (each pixel accumulates the sum of logit values from all overlapping patches,
     divided by patch count)
  6. Apply threshold → binary prediction map
  7. Connected components → polygon extraction

Usage (one scene):
    from src.inference import predict_scene

    prob_map = predict_scene(
        model, scene_dir, stats, device,
        patch_size=64, stride=16, tta=True,
    )   # [H, W] float32, values in [0, 1]

    # Threshold sweep for F1/F2 operating points (on val set):
    from src.evaluate import sweep_thresholds
    results = sweep_thresholds(prob_map.ravel(), gt_mask.ravel())
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.data.preprocess import preprocess_scene, load_gt_mask, load_scene_meta

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TTA helpers
# ---------------------------------------------------------------------------

def _tta_transforms(patch: torch.Tensor) -> list[torch.Tensor]:
    """
    4-variant TTA: identity, hflip, vflip, both-flips.
    Input/output: [1, 12, 64, 64]
    """
    return [
        patch,
        torch.flip(patch, dims=[-1]),         # horizontal flip
        torch.flip(patch, dims=[-2]),          # vertical flip
        torch.flip(patch, dims=[-1, -2]),      # both
    ]


def _tta_inverse(logits: list[torch.Tensor]) -> torch.Tensor:
    """Undo flip transforms and average. Input: list of [1, 1, H, W]."""
    inv = [
        logits[0],
        torch.flip(logits[1], dims=[-1]),
        torch.flip(logits[2], dims=[-2]),
        torch.flip(logits[3], dims=[-1, -2]),
    ]
    return torch.stack(inv, dim=0).mean(dim=0)


# ---------------------------------------------------------------------------
# Scene-level inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_scene(
    model:      nn.Module,
    scene_dir:  str | Path,
    stats:      dict,
    device:     torch.device,
    patch_size: int = 64,
    stride:     int = 16,
    tta:        bool = True,
    batch_size: int = 64,
    blending:   str = "mean",
    gaussian_sigma_frac: float = 0.25,
) -> np.ndarray:
    """
    Run inference on a full AvalCD scene.

    Args:
        model:      Trained D4SegNet (eval mode).
        scene_dir:  Path to AvalCD scene directory.
        stats:      Norm stats dict (from norm_stats_12ch.json).
        device:     Torch device.
        patch_size: Patch side length in pixels.
        stride:     Sliding window stride.
        tta:        If True, apply 4-fold TTA (redundant for D4 equivariance).
        batch_size: Number of patches per forward pass.
        blending:   How to combine overlapping tile predictions:
                    'mean'     — average logits across tiles covering each pixel (our default)
                    'max'      — max probability across tiles (Gatti's F2-opt best)
                    'gaussian' — Gaussian-weighted average centered on each tile (Gatti's F1-opt best)
        gaussian_sigma_frac: Sigma of the Gaussian kernel as fraction of patch_size
                             (only used when blending='gaussian'; default 0.25 → σ = P/4)

    Returns:
        prob_map: [H, W] float32 array with predicted probabilities in [0, 1].
    """
    assert blending in ("mean", "max", "gaussian", "center_crop"), f"unknown blending={blending}"
    scene_dir = Path(scene_dir)
    model.eval()

    # ── Preprocess scene ──────────────────────────────────────────────
    arr12 = preprocess_scene(scene_dir)   # [12, H, W]
    mean  = np.array(stats["mean"], dtype=np.float32).reshape(-1, 1, 1)
    std   = np.array(stats["std"],  dtype=np.float32).reshape(-1, 1, 1)
    arr12 = (arr12 - mean) / std

    C, H, W = arr12.shape
    P = patch_size

    # ── Build sliding-window patch coordinates ─────────────────────────
    row_starts = list(range(0, H - P + 1, stride))
    col_starts = list(range(0, W - P + 1, stride))
    # Ensure final patch covers right/bottom edge
    if row_starts[-1] + P < H:
        row_starts.append(H - P)
    if col_starts[-1] + P < W:
        col_starts.append(W - P)

    coords = [(i, j) for i in row_starts for j in col_starts]

    # ── Prepare blending accumulators ──────────────────────────────────
    if blending == "mean":
        logit_sum = np.zeros((H, W), dtype=np.float64)
        count_map = np.zeros((H, W), dtype=np.float64)
    elif blending == "max":
        # Initialize to -inf so first write wins; operate on probability space.
        prob_acc = np.full((H, W), -1.0, dtype=np.float32)
    elif blending == "gaussian":
        # Gaussian-weighted average (Gatti's F1-opt best blending)
        sigma = max(1.0, gaussian_sigma_frac * P)
        yy, xx = np.mgrid[0:P, 0:P].astype(np.float32)
        cy = (P - 1) / 2.0
        kernel = np.exp(-((yy - cy) ** 2 + (xx - cy) ** 2) / (2.0 * sigma ** 2))
        kernel = kernel.astype(np.float64)
        weighted_logit_sum = np.zeros((H, W), dtype=np.float64)
        weight_sum         = np.zeros((H, W), dtype=np.float64)
    elif blending == "center_crop":
        # Take only center region of each tile (stride × stride pixels)
        cc_margin = (P - stride) // 2
        prob_map_cc = np.zeros((H, W), dtype=np.float32)
        written_cc  = np.zeros((H, W), dtype=bool)

    arr_t = torch.from_numpy(arr12)

    # Process in batches of patches
    for batch_start in range(0, len(coords), batch_size):
        batch_coords = coords[batch_start : batch_start + batch_size]
        patches = [arr_t[:, i:i+P, j:j+P] for i, j in batch_coords]
        batch_t = torch.stack(patches).to(device)   # [BS, 12, P, P]

        if tta:
            all_logits = []
            for flip_dims in [None, [-1], [-2], [-1, -2]]:
                x = batch_t
                if flip_dims:
                    x = torch.flip(x, dims=flip_dims)
                out = model(x)
                logit = out["logit"]
                if flip_dims:
                    logit = torch.flip(logit, dims=flip_dims)
                all_logits.append(logit)
            batch_logit = torch.stack(all_logits, dim=0).mean(dim=0)
        else:
            out = model(batch_t)
            batch_logit = out["logit"]

        batch_logit_np = batch_logit.squeeze(1).cpu().float().numpy()  # [BS, P, P]

        for k, (i, j) in enumerate(batch_coords):
            tile_logit = batch_logit_np[k]
            if blending == "mean":
                logit_sum[i:i+P, j:j+P] += tile_logit
                count_map[i:i+P, j:j+P] += 1.0
            elif blending == "max":
                tile_prob = 1.0 / (1.0 + np.exp(-tile_logit.astype(np.float32)))
                np.maximum(prob_acc[i:i+P, j:j+P], tile_prob, out=prob_acc[i:i+P, j:j+P])
            elif blending == "gaussian":
                weighted_logit_sum[i:i+P, j:j+P] += tile_logit.astype(np.float64) * kernel
                weight_sum[i:i+P, j:j+P]         += kernel
            elif blending == "center_crop":
                tile_prob = 1.0 / (1.0 + np.exp(-tile_logit.astype(np.float32)))
                cr0, cr1 = cc_margin, P - cc_margin
                cc0, cc1 = cc_margin, P - cc_margin
                si, sj = i + cr0, j + cc0
                eh = min(cr1 - cr0, H - si)
                ew = min(cc1 - cc0, W - sj)
                if eh > 0 and ew > 0 and not written_cc[si:si+eh, sj:sj+ew].all():
                    mask = ~written_cc[si:si+eh, sj:sj+ew]
                    prob_map_cc[si:si+eh, sj:sj+ew][mask] = tile_prob[cr0:cr0+eh, cc0:cc0+ew][mask]
                    written_cc[si:si+eh, sj:sj+ew] = True

    # ── Finalize ───────────────────────────────────────────────────────
    if blending == "mean":
        count_map = np.maximum(count_map, 1.0)
        avg_logit = logit_sum / count_map
        prob_map  = 1.0 / (1.0 + np.exp(-avg_logit))
    elif blending == "max":
        # Border pixels never touched remain -1 → clamp to 0
        prob_map = np.where(prob_acc < 0.0, 0.0, prob_acc)
    elif blending == "gaussian":
        weight_sum = np.maximum(weight_sum, 1e-10)
        avg_logit  = weighted_logit_sum / weight_sum
        prob_map   = 1.0 / (1.0 + np.exp(-avg_logit))
    elif blending == "center_crop":
        prob_map = prob_map_cc

    return prob_map.astype(np.float32)


# ---------------------------------------------------------------------------
# Batch inference over multiple scenes
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_scenes(
    model:      nn.Module,
    scene_dirs: list[Path],
    stats:      dict,
    device:     torch.device,
    **kwargs,
) -> dict[str, np.ndarray]:
    """
    Run predict_scene on a list of scenes.

    Returns:
        dict mapping scene_name → prob_map [H, W]
    """
    results = {}
    for scene_dir in scene_dirs:
        scene_dir = Path(scene_dir)
        log.info("Inference on %s ...", scene_dir.name)
        prob_map = predict_scene(model, scene_dir, stats, device, **kwargs)
        results[scene_dir.name] = prob_map
    return results


# ---------------------------------------------------------------------------
# Polygon extraction from binary prediction map
# ---------------------------------------------------------------------------

def extract_polygons(
    binary_map: np.ndarray,    # [H, W] uint8 (0/1)
    meta:       dict,          # from load_scene_meta
    min_area_px: int = 4,      # minimum connected component size
) -> list[dict]:
    """
    Extract connected components and convert to polygons.

    Returns list of dicts:
        { geometry: shapely.Polygon, area_m2: float, centroid_xy: tuple }
    """
    from scipy.ndimage import label as ndi_label
    import shapely.geometry as sg

    labeled, n_comps = ndi_label(binary_map)
    polygons = []
    transform = meta["transform"]

    for comp_id in range(1, n_comps + 1):
        comp = (labeled == comp_id).astype(np.uint8)
        area_px = comp.sum()
        if area_px < min_area_px:
            continue

        # Bounding box centroid in pixel space
        rows, cols = np.where(comp)
        cy_px = rows.mean()
        cx_px = cols.mean()

        # Convert centroid to CRS coordinates
        cx_crs, cy_crs = rasterio_xy(transform, cy_px, cx_px)
        area_m2 = float(area_px) * 100.0   # 10m × 10m pixels

        # Build convex hull of pixel centroids as proxy polygon
        if len(rows) >= 3:
            pts = [(float(transform.c + (c + 0.5) * transform.a),
                    float(transform.f + (r + 0.5) * transform.e))
                   for r, c in zip(rows, cols)]
            try:
                geom = sg.MultiPoint(pts).convex_hull
            except Exception:
                geom = sg.Point(cx_crs, cy_crs)
        else:
            geom = sg.Point(cx_crs, cy_crs)

        polygons.append({
            "geometry":    geom,
            "area_m2":     area_m2,
            "centroid_xy": (cx_crs, cy_crs),
        })

    return polygons


def rasterio_xy(transform, row: float, col: float) -> tuple[float, float]:
    """Convert row/col pixel coords to CRS x/y using an Affine transform."""
    x = transform.c + (col + 0.5) * transform.a
    y = transform.f + (row + 0.5) * transform.e
    return x, y
