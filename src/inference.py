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
) -> np.ndarray:
    """
    Run inference on a full AvalCD scene.

    Args:
        model:      Trained D4SegNet (eval mode).
        scene_dir:  Path to AvalCD scene directory.
        stats:      Norm stats dict (from norm_stats_12ch.json).
        device:     Torch device.
        patch_size: Patch side length in pixels (default 64).
        stride:     Sliding window stride (default 16, 75% overlap).
        tta:        If True, apply 4-fold TTA (default True).
        batch_size: Number of patches per forward pass.

    Returns:
        prob_map: [H, W] float32 array with predicted probabilities in [0, 1].
    """
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

    # ── Accumulate logit sum and count ────────────────────────────────
    logit_sum = np.zeros((H, W), dtype=np.float64)
    count_map = np.zeros((H, W), dtype=np.float64)

    arr_t = torch.from_numpy(arr12)

    # Process in batches of patches
    for batch_start in range(0, len(coords), batch_size):
        batch_coords = coords[batch_start : batch_start + batch_size]
        patches = [arr_t[:, i:i+P, j:j+P] for i, j in batch_coords]
        batch_t = torch.stack(patches).to(device)   # [BS, 12, 64, 64]

        if tta:
            # Run 4 TTA variants
            all_logits = []
            for flip_dims in [None, [-1], [-2], [-1, -2]]:
                x = batch_t
                if flip_dims:
                    x = torch.flip(x, dims=flip_dims)
                out = model(x)
                logit = out["logit"]  # [BS, 1, 64, 64]
                if flip_dims:
                    logit = torch.flip(logit, dims=flip_dims)
                all_logits.append(logit)
            batch_logit = torch.stack(all_logits, dim=0).mean(dim=0)  # [BS, 1, 64, 64]
        else:
            out = model(batch_t)
            batch_logit = out["logit"]   # [BS, 1, 64, 64]

        batch_logit_np = batch_logit.squeeze(1).cpu().float().numpy()  # [BS, 64, 64]

        for k, (i, j) in enumerate(batch_coords):
            logit_sum[i:i+P, j:j+P] += batch_logit_np[k]
            count_map[i:i+P, j:j+P] += 1.0

    # Avoid division by zero in uncovered border pixels
    count_map = np.maximum(count_map, 1.0)
    avg_logit = logit_sum / count_map
    prob_map  = 1.0 / (1.0 + np.exp(-avg_logit))   # sigmoid

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
