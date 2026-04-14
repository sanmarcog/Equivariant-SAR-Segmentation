"""
src/evaluate.py

Evaluation metrics for Phase 2, matching Gatti et al. 2026 protocol exactly.

Primary (pixel-level):
  - Sweep all unique predicted probabilities as thresholds on val set
  - Report best F1 at F1-optimised threshold
  - Report best F2 at F2-optimised threshold
  - These are two operating points on the same trained model

Supplementary (polygon-level):
  - Threshold → connected components → match GT polygons by IoU ≥ 0.1
  - Polygon hit rate = fraction of GT polygons with ≥1 predicted positive pixel
  - Per-D-scale breakdown (D1/D2/D3/D4) for Tromsø OOD test set

Usage:
    from src.evaluate import sweep_thresholds, evaluate_scene, polygon_metrics

    # Val pixel metrics
    results = sweep_thresholds(prob_flat, gt_flat)
    # → {'best_f1': 0.82, 'thr_f1': 0.43, 'best_f2': 0.86, 'thr_f2': 0.31}

    # Tromsø OOD evaluation
    metrics = evaluate_scene(prob_map, gt_mask)

    # Polygon-level (supplementary)
    poly_m = polygon_metrics(pred_polygons, gt_gdf, iou_thresh=0.1)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pixel-level metrics (primary)
# ---------------------------------------------------------------------------

def _f_beta(precision: float, recall: float, beta: float) -> float:
    """F-beta score (β=1 for F1, β=2 for F2)."""
    denom = (beta ** 2) * precision + recall
    if denom < 1e-10:
        return 0.0
    return (1.0 + beta ** 2) * precision * recall / denom


def sweep_thresholds(
    prob: np.ndarray,    # 1-D float array, values in [0, 1]
    gt:   np.ndarray,    # 1-D binary array (0/1)
) -> dict[str, float]:
    """
    Sweep all unique predicted probabilities as thresholds.
    Returns best F1 and best F2 operating points.

    Matches Gatti et al. 2026 evaluation protocol:
    "We sweep all unique predicted probabilities on the validation set
    and report the best F1 and the best F2 separately."

    Args:
        prob: Predicted probability for each pixel, shape [N].
        gt:   Ground-truth binary label for each pixel, shape [N].

    Returns:
        dict with keys: best_f1, thr_f1, best_f2, thr_f2,
                        precision_f1, recall_f1, precision_f2, recall_f2
    """
    gt = gt.astype(np.float32)
    n_pos = gt.sum()
    if n_pos == 0:
        return {k: 0.0 for k in ["best_f1", "thr_f1", "best_f2", "thr_f2",
                                  "precision_f1", "recall_f1", "precision_f2", "recall_f2"]}

    # Sort by decreasing probability so we can compute TP/FP incrementally
    sort_idx = np.argsort(-prob)
    prob_sorted = prob[sort_idx]
    gt_sorted   = gt[sort_idx]

    # Unique thresholds (each predicted probability value)
    thresholds = np.unique(prob_sorted)[::-1]   # decreasing order

    # For each threshold: pred = (prob >= thr)
    # Compute TP, FP, FN using cumulative sums
    cum_tp = np.cumsum(gt_sorted)              # TP at each position
    cum_fp = np.cumsum(1.0 - gt_sorted)        # FP at each position

    best_f1 = 0.0; thr_f1 = 0.5; pr_f1 = 0.0; rc_f1 = 0.0
    best_f2 = 0.0; thr_f2 = 0.5; pr_f2 = 0.0; rc_f2 = 0.0

    for thr in thresholds:
        # Position of last probability >= thr
        k = int(np.searchsorted(-prob_sorted, -thr, side="right"))
        if k == 0:
            continue
        tp = cum_tp[k - 1]
        fp = cum_fp[k - 1]
        fn = n_pos - tp

        prec = tp / (tp + fp + 1e-10)
        rec  = tp / (tp + fn + 1e-10)

        f1 = _f_beta(prec, rec, beta=1.0)
        f2 = _f_beta(prec, rec, beta=2.0)

        if f1 > best_f1:
            best_f1, thr_f1, pr_f1, rc_f1 = f1, thr, prec, rec
        if f2 > best_f2:
            best_f2, thr_f2, pr_f2, rc_f2 = f2, thr, prec, rec

    return {
        "best_f1":      float(best_f1),
        "thr_f1":       float(thr_f1),
        "precision_f1": float(pr_f1),
        "recall_f1":    float(rc_f1),
        "best_f2":      float(best_f2),
        "thr_f2":       float(thr_f2),
        "precision_f2": float(pr_f2),
        "recall_f2":    float(rc_f2),
    }


def evaluate_scene(
    prob_map: np.ndarray,    # [H, W] float32
    gt_mask:  np.ndarray,    # [H, W] uint8 (0/1)
) -> dict[str, float]:
    """
    Full pixel-level evaluation on a single scene.

    Returns sweep_thresholds results over all scene pixels.
    """
    return sweep_thresholds(prob_map.ravel(), gt_mask.ravel().astype(np.float32))


# ---------------------------------------------------------------------------
# Polygon-level metrics (supplementary)
# ---------------------------------------------------------------------------

def _iou_poly(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """IoU between two binary masks."""
    inter = (pred_mask & gt_mask).sum()
    union = (pred_mask | gt_mask).sum()
    return inter / union if union > 0 else 0.0


def polygon_metrics(
    prob_map:    np.ndarray,     # [H, W] float32
    gt_gdf,                      # GeoDataFrame with 'geometry', 'size' columns (Tromsø GT)
    meta:        dict,           # scene meta from load_scene_meta
    threshold:   float,
    iou_thresh:  float = 0.1,    # IoU threshold for polygon matching (supplementary)
    min_area_px: int   = 4,
) -> dict:
    """
    Polygon-level F1/F2 and hit rate on Tromsø OOD test scene.

    A GT polygon is "hit" if ≥1 predicted positive pixel overlaps it.
    A GT polygon is "matched" if a predicted polygon has IoU ≥ iou_thresh.

    Returns:
        dict with overall and per-D-scale metrics:
            hit_rate_overall, hit_rate_D1, ..., hit_rate_D4
            poly_f1_overall,  poly_f2_overall
            poly_f1_D2, ...
    """
    from scipy.ndimage import label as ndi_label
    import rasterio.features as riof
    import rasterio

    pred_binary = (prob_map >= threshold).astype(np.uint8)

    # Rasterise GT polygons using rasterio
    transform  = meta["transform"]
    H, W       = meta["shape"]
    gt_raster  = np.zeros((H, W), dtype=np.uint8)   # each polygon gets its own ID

    # Build per-polygon masks
    poly_masks = []
    for idx, row in gt_gdf.iterrows():
        geom = row.geometry
        size = int(row.get("size", 0)) if "size" in gt_gdf.columns else 0
        try:
            burned = riof.rasterize(
                [(geom, 1)],
                out_shape=(H, W),
                transform=transform,
                fill=0,
                dtype=np.uint8,
            )
        except Exception:
            burned = np.zeros((H, W), dtype=np.uint8)
        poly_masks.append({"mask": burned.astype(bool), "size": size})

    # Extract predicted connected components
    labeled, n_pred = ndi_label(pred_binary)

    # Per-GT-polygon: hit rate and best IoU
    hits_by_dscale = {1: [], 2: [], 3: [], 4: []}
    matched_by_dscale = {1: [], 2: [], 3: [], 4: []}

    for pm in poly_masks:
        gt_px  = pm["mask"]
        dscale = pm["size"]
        if dscale not in hits_by_dscale:
            hits_by_dscale[dscale]   = []
            matched_by_dscale[dscale] = []

        # Hit: any predicted positive pixel overlaps GT polygon
        hit = bool((pred_binary.astype(bool) & gt_px).sum() > 0)
        hits_by_dscale[dscale].append(int(hit))

        # Match: any predicted CC has IoU ≥ iou_thresh with GT polygon
        best_iou = 0.0
        if n_pred > 0:
            # Only check CCs that overlap GT bbox
            rows, cols = np.where(gt_px)
            if len(rows) > 0:
                r0, r1, c0, c1 = rows.min(), rows.max(), cols.min(), cols.max()
                for cc_id in np.unique(labeled[r0:r1+1, c0:c1+1]):
                    if cc_id == 0:
                        continue
                    cc_mask = (labeled == cc_id).astype(bool)
                    iou = _iou_poly(cc_mask, gt_px)
                    if iou > best_iou:
                        best_iou = iou
        matched_by_dscale[dscale].append(int(best_iou >= iou_thresh))

    # Aggregate metrics
    def _f_beta_from_lists(tp, fp, fn, beta):
        p = tp / (tp + fp + 1e-10)
        r = tp / (tp + fn + 1e-10)
        return _f_beta(p, r, beta)

    results = {}
    all_hits = []; all_matched = []

    for d in [1, 2, 3, 4]:
        h = hits_by_dscale.get(d, [])
        m = matched_by_dscale.get(d, [])
        n = len(h)
        results[f"n_D{d}"]         = n
        results[f"hit_rate_D{d}"]  = float(np.mean(h)) if n > 0 else 0.0
        results[f"matched_D{d}"]   = int(np.sum(m))
        all_hits.extend(h)
        all_matched.extend(m)

    results["hit_rate_overall"] = float(np.mean(all_hits)) if all_hits else 0.0
    results["n_gt_total"] = len(all_hits)

    # Polygon-level precision/recall/F1/F2:
    # True positive = GT polygon matched by ≥1 predicted polygon
    # False negative = GT polygon not matched
    # False positive = predicted CC not matched to any GT polygon (approximated)
    n_tp = sum(all_matched)
    n_fn = len(all_matched) - n_tp
    # Count predicted CCs not matched to any GT
    matched_cc_ids = set()
    for pm in poly_masks:
        gt_px = pm["mask"]
        rows, cols = np.where(gt_px)
        if len(rows) == 0:
            continue
        r0, r1, c0, c1 = rows.min(), rows.max(), cols.min(), cols.max()
        for cc_id in np.unique(labeled[r0:r1+1, c0:c1+1]):
            if cc_id == 0:
                continue
            cc_mask = (labeled == cc_id).astype(bool)
            if _iou_poly(cc_mask, gt_px) >= iou_thresh:
                matched_cc_ids.add(cc_id)
    n_fp  = max(n_pred - len(matched_cc_ids), 0)

    prec  = n_tp / (n_tp + n_fp + 1e-10)
    rec   = n_tp / (n_tp + n_fn + 1e-10)
    results["poly_precision"] = float(prec)
    results["poly_recall"]    = float(rec)
    results["poly_f1"]        = float(_f_beta(prec, rec, beta=1.0))
    results["poly_f2"]        = float(_f_beta(prec, rec, beta=2.0))

    return results


# ---------------------------------------------------------------------------
# Multi-seed aggregation
# ---------------------------------------------------------------------------

def aggregate_seeds(results_list: list[dict]) -> dict[str, dict]:
    """
    Aggregate metric dicts across seeds (mean ± std).

    Args:
        results_list: List of metric dicts, one per seed.

    Returns:
        dict mapping metric_name → {'mean': ..., 'std': ...}
    """
    keys = results_list[0].keys()
    out  = {}
    for k in keys:
        vals = [r[k] for r in results_list if isinstance(r.get(k), (int, float))]
        if vals:
            out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return out


# ---------------------------------------------------------------------------
# CLI — evaluate a saved checkpoint on val or test
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse, json, torch
    from src.data.norm_stats import load_stats
    from src.models.segnet import build_model
    from src.inference import predict_scene
    from src.data.preprocess import load_gt_mask

    p = argparse.ArgumentParser(description="Evaluate saved checkpoint")
    p.add_argument("--ckpt",       required=True, type=Path)
    p.add_argument("--data-dir",   required=True, type=Path)
    p.add_argument("--stats",      required=True, type=Path)
    p.add_argument("--split",      default="val", choices=["val", "test"])
    p.add_argument("--out",        required=True, type=Path)
    p.add_argument("--thr-f1",     type=float, default=None)
    p.add_argument("--thr-f2",     type=float, default=None)
    p.add_argument("--no-tta",     action="store_true")
    args = p.parse_args()

    from src.data.dataset import VAL_SCENES, TEST_SCENES
    scene_names = VAL_SCENES if args.split == "val" else TEST_SCENES

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats  = load_stats(args.stats)

    ckpt = torch.load(args.ckpt, map_location=device)
    use_skip = ckpt.get("cfg", {}).get("use_skip", True)
    model = build_model(use_skip=use_skip).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    all_probs = []; all_gts = []
    scene_results = {}

    for scene_name in scene_names:
        scene_dir = args.data_dir / scene_name
        gt_mask   = load_gt_mask(scene_dir)
        prob_map  = predict_scene(
            model, scene_dir, stats, device,
            tta=not args.no_tta,
        )
        metrics = evaluate_scene(prob_map, gt_mask)
        scene_results[scene_name] = metrics
        all_probs.append(prob_map.ravel())
        all_gts.append(gt_mask.ravel().astype(np.float32))

        log.info(
            "%s: F1=%.4f (thr=%.3f)  F2=%.4f (thr=%.3f)",
            scene_name, metrics["best_f1"], metrics["thr_f1"],
            metrics["best_f2"], metrics["thr_f2"],
        )

    # Overall across all scenes
    all_prob_cat = np.concatenate(all_probs)
    all_gt_cat   = np.concatenate(all_gts)
    overall = sweep_thresholds(all_prob_cat, all_gt_cat)
    log.info(
        "Overall: F1=%.4f (thr=%.3f)  F2=%.4f (thr=%.3f)",
        overall["best_f1"], overall["thr_f1"],
        overall["best_f2"], overall["thr_f2"],
    )

    out = {"scene_results": scene_results, "overall": overall}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    log.info("Results saved → %s", args.out)


if __name__ == "__main__":
    main()
