"""
src/evaluate.py

Evaluation metrics for Phase 2, matching Gatti et al. 2026 protocol exactly.

Primary (pixel-level):
  - Sweep all unique predicted probabilities as thresholds on val set
  - Report best F1 at F1-optimised threshold, best F2 at F2-optimised threshold
  - AUPRC used for model selection (early stopping) and grid-search criterion

Supplementary (polygon-level):
  - Threshold → connected components → match GT polygons by IoU ≥ 0.1
  - Polygon hit rate = fraction of GT polygons with ≥1 predicted positive pixel
  - Per-D-scale breakdown (D1/D2/D3/D4) for Tromsø OOD test set

Statistical validity:
  - Bootstrap 95% CI on per-D-scale pixel F2 (10K draws, resamples polygons per class)
  - Permutation test for D2: p-value that D2 detection > chance (10K permutations)

Dual ablation tables (same runs, no extra compute):
  - Table A: overall pixel F1/F2 across all D-scales
  - Table B: D2-only pixel F2 with bootstrap 95% CIs

Usage:
    from src.evaluate import (
        sweep_thresholds, auprc,
        build_polygon_masks, dscale_pixel_f2,
        bootstrap_dscale_ci, permutation_test_d2,
        polygon_metrics, aggregate_seeds,
        format_ablation_tables,
    )

    # Val — model selection
    results = sweep_thresholds(prob_flat, gt_flat)
    auc     = auprc(prob_flat, gt_flat)

    # Tromsø OOD
    poly_masks = build_polygon_masks(gt_gdf, meta)
    overall    = sweep_thresholds(prob_map.ravel(), gt_mask.ravel())
    d2_f2      = dscale_pixel_f2(prob_map, poly_masks, thr=results['thr_f2'])[2]
    ci         = bootstrap_dscale_ci(prob_map, poly_masks, thr=results['thr_f2'])
    pval       = permutation_test_d2(prob_map, poly_masks, observed_d2_f2=d2_f2, thr=results['thr_f2'])
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pixel-level F1/F2 threshold sweep (primary — Gatti protocol)
# ---------------------------------------------------------------------------

def _f_beta(precision: float, recall: float, beta: float) -> float:
    denom = (beta ** 2) * precision + recall
    if denom < 1e-10:
        return 0.0
    return (1.0 + beta ** 2) * precision * recall / denom


def sweep_thresholds(
    prob: np.ndarray,   # [N] float, values in [0,1]
    gt:   np.ndarray,   # [N] binary (0/1)
) -> dict[str, float]:
    """
    Sweep all unique predicted probabilities as thresholds.
    Report best F1 and best F2 operating points (two separate thresholds).

    Matches Gatti et al. 2026 protocol exactly.

    Returns dict: best_f1, thr_f1, precision_f1, recall_f1,
                  best_f2, thr_f2, precision_f2, recall_f2
    """
    gt = gt.astype(np.float32)
    n_pos = gt.sum()
    if n_pos == 0:
        return {k: 0.0 for k in [
            "best_f1", "thr_f1", "precision_f1", "recall_f1",
            "best_f2", "thr_f2", "precision_f2", "recall_f2",
        ]}

    # Sort by probability descending, compute cumulative TP at each rank.
    # At rank k (1-indexed): TP = cum_tp[k-1], FP = k - TP, FN = n_pos - TP.
    # Fully vectorized — no Python loop over thresholds.
    sort_idx    = np.argsort(-prob)
    prob_sorted = prob[sort_idx].astype(np.float32, copy=False)
    gt_sorted   = gt[sort_idx]

    cum_tp = np.cumsum(gt_sorted, dtype=np.float64)
    positions = np.arange(1, len(prob_sorted) + 1, dtype=np.float64)

    prec = cum_tp / positions
    rec  = cum_tp / float(n_pos)

    denom_f1 = prec + rec
    denom_f2 = 4.0 * prec + rec
    f1 = np.where(denom_f1 > 0, 2.0 * prec * rec / (denom_f1 + 1e-10), 0.0)
    f2 = np.where(denom_f2 > 0, 5.0 * prec * rec / (denom_f2 + 1e-10), 0.0)

    k1 = int(np.argmax(f1))
    k2 = int(np.argmax(f2))

    return {
        "best_f1":      float(f1[k1]),
        "thr_f1":       float(prob_sorted[k1]),
        "precision_f1": float(prec[k1]),
        "recall_f1":    float(rec[k1]),
        "best_f2":      float(f2[k2]),
        "thr_f2":       float(prob_sorted[k2]),
        "precision_f2": float(prec[k2]),
        "recall_f2":    float(rec[k2]),
    }


def auprc(prob: np.ndarray, gt: np.ndarray) -> float:
    """
    Area under the precision-recall curve.
    Used for: (a) early-stopping criterion, (b) grid-search model selection.
    Matches Gatti et al. 2026 protocol (they use torchmetrics AUPRC).
    """
    from sklearn.metrics import average_precision_score
    gt = gt.astype(np.float32)
    if gt.sum() == 0 or gt.sum() == len(gt):
        return 0.0
    return float(average_precision_score(gt, prob))


def evaluate_scene(
    prob_map: np.ndarray,   # [H, W]
    gt_mask:  np.ndarray,   # [H, W] uint8 (0/1)
) -> dict[str, float]:
    """Pixel-level threshold sweep on a full scene."""
    return sweep_thresholds(prob_map.ravel(), gt_mask.ravel().astype(np.float32))


# ---------------------------------------------------------------------------
# Per-polygon mask builder (reused by polygon_metrics, bootstrap, permutation)
# ---------------------------------------------------------------------------

def build_polygon_masks(gt_gdf, meta: dict) -> list[dict]:
    """
    Rasterize each GT polygon from a GeoDataFrame into a binary pixel mask.

    Args:
        gt_gdf: GeoDataFrame with 'geometry' and 'size' (D-scale int) columns.
        meta:   Scene metadata dict from load_scene_meta (keys: transform, shape).

    Returns:
        list of dicts:  { mask: bool [H,W], size: int, area_px: int }
        Ordered the same as gt_gdf rows.
    """
    import rasterio.features as riof

    transform = meta["transform"]
    H, W      = meta["shape"]

    poly_masks = []
    for _, row in gt_gdf.iterrows():
        geom = row.geometry
        size = int(row.get("size", 0)) if "size" in gt_gdf.columns else 0
        try:
            burned = riof.rasterize(
                [(geom, 1)],
                out_shape=(H, W),
                transform=transform,
                fill=0,
                dtype=np.uint8,
            ).astype(bool)
        except Exception:
            burned = np.zeros((H, W), dtype=bool)
        poly_masks.append({
            "mask":     burned,
            "size":     size,
            "area_px":  int(burned.sum()),
        })
    return poly_masks


# ---------------------------------------------------------------------------
# Per-D-scale pixel F2 (at a fixed threshold)
# ---------------------------------------------------------------------------

def dscale_pixel_f2(
    prob_map:   np.ndarray,        # [H, W] float
    poly_masks: list[dict],        # from build_polygon_masks
    thr:        float,
    poly_subset: list[int] | None = None,   # indices into poly_masks; None = all
) -> dict[int, float]:
    """
    Compute pixel F2 treating each D-scale class as the positive class.

    For D-scale d:
      - Positives = pixels in D-scale-d polygons (from poly_subset)
      - TP = predicted positive AND in a D-scale-d polygon
      - FP = predicted positive AND NOT in any D-scale-d polygon
      - FN = in a D-scale-d polygon AND predicted negative

    Returns:
        {1: f2_d1, 2: f2_d2, 3: f2_d3, 4: f2_d4}
        d-scales with zero polygons return 0.0.
    """
    pred_binary = (prob_map >= thr)
    H, W = prob_map.shape

    # Subset of polygon masks to use
    masks = [poly_masks[i] for i in poly_subset] if poly_subset is not None else poly_masks

    # Group masks by D-scale
    by_dscale: dict[int, list[np.ndarray]] = {}
    for pm in masks:
        d = pm["size"]
        by_dscale.setdefault(d, []).append(pm["mask"])

    results = {}
    for d in [1, 2, 3, 4]:
        if d not in by_dscale or not by_dscale[d]:
            results[d] = 0.0
            continue

        # Union of all D-scale-d polygon masks
        gt_d = np.zeros((H, W), dtype=bool)
        for m in by_dscale[d]:
            gt_d |= m

        tp = (pred_binary & gt_d).sum()
        fp = (pred_binary & ~gt_d).sum()
        fn = (~pred_binary & gt_d).sum()

        prec = tp / (tp + fp + 1e-10)
        rec  = tp / (tp + fn + 1e-10)
        results[d] = float(_f_beta(prec, rec, beta=2.0))

    return results


# ---------------------------------------------------------------------------
# Bootstrap 95% CI on per-D-scale pixel F2
# ---------------------------------------------------------------------------

def bootstrap_dscale_ci(
    prob_map:    np.ndarray,   # [H, W]
    poly_masks:  list[dict],   # from build_polygon_masks
    thr:         float,
    n_bootstrap: int   = 10_000,
    seed:        int   = 42,
    confidence:  float = 0.95,
) -> dict[int, dict[str, float]]:
    """
    Bootstrap confidence intervals for per-D-scale pixel F2.

    For each D-scale d (with n_d polygons):
      1. Draw n_d polygons with replacement from the d-class polygons.
      2. Build D-scale mask from the resampled polygons.
      3. Compute pixel F2 against the full prediction map.
      4. Repeat n_bootstrap times → distribution of F2 values.
      5. Report 2.5th / 97.5th percentile as 95% CI.

    Critical for D2 (n=25): 3-seed mean ± std masks small-sample uncertainty.

    Returns:
        {d: {'observed': float, 'ci_lower': float, 'ci_upper': float, 'n': int}}
    """
    rng = np.random.default_rng(seed)

    # Group polygon indices by D-scale
    by_dscale: dict[int, list[int]] = {}
    for i, pm in enumerate(poly_masks):
        by_dscale.setdefault(pm["size"], []).append(i)

    # Observed F2 per D-scale (using all polygons)
    observed = dscale_pixel_f2(prob_map, poly_masks, thr)

    results = {}
    for d in [1, 2, 3, 4]:
        idxs = by_dscale.get(d, [])
        n_d  = len(idxs)
        if n_d == 0:
            results[d] = {"observed": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "n": 0}
            continue

        boot_f2 = np.empty(n_bootstrap, dtype=np.float32)
        for b in range(n_bootstrap):
            # Resample n_d polygon indices with replacement
            sampled = rng.choice(idxs, size=n_d, replace=True)
            boot_f2[b] = dscale_pixel_f2(prob_map, poly_masks, thr, poly_subset=sampled.tolist())[d]

        alpha = 1.0 - confidence
        lo = float(np.percentile(boot_f2, 100 * alpha / 2))
        hi = float(np.percentile(boot_f2, 100 * (1 - alpha / 2)))

        results[d] = {
            "observed":  float(observed[d]),
            "ci_lower":  lo,
            "ci_upper":  hi,
            "n":         n_d,
            "boot_mean": float(boot_f2.mean()),
            "boot_std":  float(boot_f2.std()),
        }

    return results


# ---------------------------------------------------------------------------
# Permutation test for D2
# ---------------------------------------------------------------------------

def permutation_test_d2(
    prob_map:        np.ndarray,   # [H, W]
    poly_masks:      list[dict],   # from build_polygon_masks; 117 Tromsø polygons
    observed_d2_f2:  float,        # observed D2 pixel F2 at threshold thr
    thr:             float,
    n_perm:          int  = 10_000,
    seed:            int  = 42,
) -> dict[str, float]:
    """
    Permutation test: is D2 detection significantly above chance?

    Null hypothesis: the set of polygons labelled 'D2' is a random subset
    of the 117 Tromsø polygons (i.e., our model doesn't specifically detect
    D2 signatures — it just detects whatever 25 polygons are assigned the D2 label).

    Procedure:
      1. Randomly select n_D2 polygon indices from all 117 (without replacement).
      2. Build 'D2' mask from those randomly selected polygons.
      3. Compute pixel F2 treating this shuffled 'D2' set as positives.
      4. Repeat n_perm times.
      5. p-value = fraction of permutations where permuted_f2 >= observed_d2_f2.

    Correctly handles the constraint that each permutation has exactly n_D2
    polygons (same class size as observed) — unlike shuffling labels freely.

    Returns:
        {'p_value': float, 'observed': float, 'n_perm': int,
         'null_mean': float, 'null_std': float, 'null_max': float}
    """
    rng = np.random.default_rng(seed)

    n_total = len(poly_masks)
    n_d2 = sum(1 for pm in poly_masks if pm["size"] == 2)

    if n_d2 == 0:
        return {"p_value": 1.0, "observed": observed_d2_f2,
                "n_perm": 0, "null_mean": 0.0, "null_std": 0.0, "null_max": 0.0}

    # Build a fake "D2" poly_masks for the resampled subset
    # We reuse the mask data but relabel the size to 2
    all_indices = list(range(n_total))

    pred_binary = (prob_map >= thr)
    H, W = prob_map.shape

    null_f2 = np.empty(n_perm, dtype=np.float32)

    for p in range(n_perm):
        sampled_idx = rng.choice(all_indices, size=n_d2, replace=False)

        # Union of selected polygon masks
        gt_perm = np.zeros((H, W), dtype=bool)
        for i in sampled_idx:
            gt_perm |= poly_masks[i]["mask"]

        tp = (pred_binary & gt_perm).sum()
        fp = (pred_binary & ~gt_perm).sum()
        fn = (~pred_binary & gt_perm).sum()
        prec = tp / (tp + fp + 1e-10)
        rec  = tp / (tp + fn + 1e-10)
        null_f2[p] = _f_beta(prec, rec, beta=2.0)

    p_value = float((null_f2 >= observed_d2_f2).mean())

    return {
        "p_value":   p_value,
        "observed":  observed_d2_f2,
        "n_perm":    n_perm,
        "n_d2":      n_d2,
        "null_mean": float(null_f2.mean()),
        "null_std":  float(null_f2.std()),
        "null_max":  float(null_f2.max()),
    }


# ---------------------------------------------------------------------------
# Polygon-level metrics (supplementary)
# ---------------------------------------------------------------------------

def _iou_poly(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    return inter / union if union > 0 else 0.0


def polygon_metrics(
    prob_map:   np.ndarray,   # [H, W] float32
    poly_masks: list[dict],   # from build_polygon_masks
    threshold:  float,
    iou_thresh: float = 0.1,
) -> dict:
    """
    Polygon-level F1/F2 and hit rate on Tromsø OOD test scene.

    A GT polygon is "hit"    if ≥1 predicted positive pixel overlaps it.
    A GT polygon is "matched" if a predicted CC has IoU ≥ iou_thresh with it.

    Returns:
        dict with overall and per-D-scale metrics.
    """
    from scipy.ndimage import label as ndi_label

    pred_binary = (prob_map >= threshold).astype(np.uint8)
    labeled, n_pred = ndi_label(pred_binary)

    hits_by_d    = {1: [], 2: [], 3: [], 4: []}
    matched_by_d = {1: [], 2: [], 3: [], 4: []}

    for pm in poly_masks:
        gt_px  = pm["mask"]
        d      = pm["size"]
        hits_by_d.setdefault(d, [])
        matched_by_d.setdefault(d, [])

        hit = bool((pred_binary.astype(bool) & gt_px).any())
        hits_by_d[d].append(int(hit))

        best_iou = 0.0
        if n_pred > 0:
            rows, cols = np.where(gt_px)
            if len(rows) > 0:
                r0, r1, c0, c1 = rows.min(), rows.max(), cols.min(), cols.max()
                for cc_id in np.unique(labeled[r0:r1+1, c0:c1+1]):
                    if cc_id == 0:
                        continue
                    cc_mask = (labeled == cc_id).astype(bool)
                    best_iou = max(best_iou, _iou_poly(cc_mask, gt_px))
        matched_by_d[d].append(int(best_iou >= iou_thresh))

    results = {}
    all_hits = []; all_matched = []
    for d in [1, 2, 3, 4]:
        h = hits_by_d.get(d, [])
        m = matched_by_d.get(d, [])
        results[f"n_D{d}"]        = len(h)
        results[f"hit_rate_D{d}"] = float(np.mean(h)) if h else 0.0
        results[f"matched_D{d}"]  = int(np.sum(m))
        all_hits.extend(h)
        all_matched.extend(m)

    results["hit_rate_overall"] = float(np.mean(all_hits)) if all_hits else 0.0
    results["n_gt_total"] = len(all_hits)

    n_tp = sum(all_matched)
    n_fn = len(all_matched) - n_tp

    # Count predicted CCs not matched to any GT polygon
    matched_cc_ids: set[int] = set()
    for pm in poly_masks:
        gt_px = pm["mask"]
        rows, cols = np.where(gt_px)
        if len(rows) == 0:
            continue
        r0, r1, c0, c1 = rows.min(), rows.max(), cols.min(), cols.max()
        for cc_id in np.unique(labeled[r0:r1+1, c0:c1+1]):
            if cc_id == 0:
                continue
            if _iou_poly((labeled == cc_id).astype(bool), gt_px) >= iou_thresh:
                matched_cc_ids.add(cc_id)
    n_fp = max(n_pred - len(matched_cc_ids), 0)

    prec = n_tp / (n_tp + n_fp + 1e-10)
    rec  = n_tp / (n_tp + n_fn + 1e-10)
    results["poly_precision"] = float(prec)
    results["poly_recall"]    = float(rec)
    results["poly_f1"]        = float(_f_beta(prec, rec, beta=1.0))
    results["poly_f2"]        = float(_f_beta(prec, rec, beta=2.0))

    return results


# ---------------------------------------------------------------------------
# Multi-seed aggregation
# ---------------------------------------------------------------------------

def aggregate_seeds(results_list: list[dict]) -> dict[str, dict]:
    """Aggregate metric dicts across seeds → mean ± std."""
    keys = results_list[0].keys()
    out  = {}
    for k in keys:
        vals = [r[k] for r in results_list if isinstance(r.get(k), (int, float))]
        if vals:
            out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return out


# ---------------------------------------------------------------------------
# Dual ablation tables
# ---------------------------------------------------------------------------

def format_ablation_tables(
    condition_results: dict[int, list[dict]],
) -> dict[str, object]:
    """
    Format Table A (overall pixel F1/F2) and Table B (D2-only pixel F2 + bootstrap CIs)
    from per-condition, per-seed results.

    Args:
        condition_results: {condition_int: [seed0_dict, seed1_dict, seed2_dict]}
            Each seed_dict must contain at least:
              'best_f1', 'best_f2'  — from sweep_thresholds on test set
              'dscale_f2'           — {d: f2} from dscale_pixel_f2
              'bootstrap_d2'        — from bootstrap_dscale_ci[2]

    Returns:
        {'table_A': {...}, 'table_B': {...}}
    """
    condition_names = {
        1: "Baseline (BCE, random sample, no skip)",
        2: "+ biased sampling",
        3: "+ Focal+Tversky loss",
        4: "+ U-Net skip connections",
        5: "+ copy-paste (full system)",
    }

    table_a = []
    table_b = []

    for cond in sorted(condition_results.keys()):
        seeds = condition_results[cond]

        # Table A: overall F1/F2
        f1_vals = [s["best_f1"] for s in seeds]
        f2_vals = [s["best_f2"] for s in seeds]
        table_a.append({
            "condition": cond,
            "name":      condition_names.get(cond, str(cond)),
            "f1_mean":   float(np.mean(f1_vals)),
            "f1_std":    float(np.std(f1_vals)),
            "f2_mean":   float(np.mean(f2_vals)),
            "f2_std":    float(np.std(f2_vals)),
        })

        # Table B: D2-only F2 + bootstrap CI
        # Use mean of observed D2-F2 across seeds, and mean of CIs
        d2_obs   = [s["dscale_f2"][2]      for s in seeds if "dscale_f2"   in s]
        d2_lo    = [s["bootstrap_d2"]["ci_lower"] for s in seeds if "bootstrap_d2" in s]
        d2_hi    = [s["bootstrap_d2"]["ci_upper"] for s in seeds if "bootstrap_d2" in s]
        n_d2     = seeds[0].get("bootstrap_d2", {}).get("n", "?")
        pval     = np.mean([s.get("perm_p_value", 1.0) for s in seeds]) if seeds else 1.0

        if d2_obs:
            table_b.append({
                "condition":  cond,
                "name":       condition_names.get(cond, str(cond)),
                "d2_f2_mean": float(np.mean(d2_obs)),
                "d2_f2_std":  float(np.std(d2_obs)),
                "d2_ci_lower_mean": float(np.mean(d2_lo)),
                "d2_ci_upper_mean": float(np.mean(d2_hi)),
                "n_d2":       n_d2,
                "perm_p_value_mean": float(pval),
            })

    return {"table_A": table_a, "table_B": table_b}


# ---------------------------------------------------------------------------
# CLI — evaluate a checkpoint on val or test, output full metrics JSON
# ---------------------------------------------------------------------------

def _evaluate_checkpoint(
    ckpt_path:  Path,
    data_dir:   Path,
    stats_path: Path,
    split:      str,
    out_path:   Path,
    use_tta:    bool = True,
    iou_thresh: float = 0.1,
    n_bootstrap: int = 10_000,
    n_perm:     int  = 10_000,
    morph_closing: bool = False,
) -> dict:
    import json
    import torch
    from src.data.norm_stats import load_stats
    from src.data.dataset import VAL_SCENES, TEST_SCENES
    from src.data.preprocess import load_gt_mask, load_scene_meta
    from src.models.segnet import build_model
    from src.inference import predict_scene

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats  = load_stats(stats_path)

    ckpt  = torch.load(ckpt_path, map_location=device)
    model = build_model(use_skip=ckpt.get("cfg", {}).get("use_skip", True)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    scene_names = VAL_SCENES if split == "val" else TEST_SCENES
    is_test = split == "test"

    # Load GT GeoDataFrame (Tromsø only — for bootstrap/permutation)
    gt_gdf = None
    if is_test:
        try:
            import geopandas as gpd
            gt_gdf = gpd.read_file(data_dir / scene_names[0] / f"{scene_names[0]}_GT.gpkg")
        except Exception as e:
            log.warning("Could not load GT GeoDataFrame: %s", e)

    all_prob = []; all_gt_flat = []
    scene_results = {}

    for scene_name in scene_names:
        scene_dir = data_dir / scene_name
        gt_mask   = load_gt_mask(scene_dir)
        meta      = load_scene_meta(scene_dir)
        prob_map  = predict_scene(model, scene_dir, stats, device, tta=use_tta)

        pixel_metrics = evaluate_scene(prob_map, gt_mask)
        all_prob.append(prob_map.ravel())
        all_gt_flat.append(gt_mask.ravel().astype(np.float32))

        scene_out: dict = {"pixel": pixel_metrics}

        # Optional morphological closing on F1- and F2-thresholded binary masks
        # (matches Gatti et al. 2026 inference: kernel=3, iterations=1)
        if morph_closing:
            from scipy.ndimage import binary_closing
            struct = np.ones((3, 3), dtype=bool)
            for tag, thr_key in [("f1", "thr_f1"), ("f2", "thr_f2")]:
                thr = pixel_metrics[thr_key]
                binary = (prob_map >= thr).astype(np.uint8)
                closed = binary_closing(binary, structure=struct, iterations=1).astype(np.float32)
                gt_f = gt_mask.astype(np.float32)
                tp = float((closed * gt_f).sum())
                fp = float((closed * (1 - gt_f)).sum())
                fn = float(((1 - closed) * gt_f).sum())
                prec = tp / (tp + fp + 1e-10)
                rec  = tp / (tp + fn + 1e-10)
                if tag == "f1":
                    f = 2 * prec * rec / (prec + rec + 1e-10)
                else:
                    f = 5 * prec * rec / (4 * prec + rec + 1e-10)
                scene_out[f"pixel_morph_{tag}"] = {
                    "precision": prec, "recall": rec,
                    f"f{tag[-1]}": float(f), "thr": float(thr),
                }

        # Supplementary for Tromsø test scene
        if is_test and gt_gdf is not None:
            thr_f2 = pixel_metrics["thr_f2"]
            poly_masks = build_polygon_masks(gt_gdf, meta)

            # Per-D-scale pixel F2
            d2_f2_vals = dscale_pixel_f2(prob_map, poly_masks, thr=thr_f2)
            scene_out["dscale_f2"] = {int(d): float(v) for d, v in d2_f2_vals.items()}

            # Bootstrap CIs
            boot_ci = bootstrap_dscale_ci(
                prob_map, poly_masks, thr=thr_f2,
                n_bootstrap=n_bootstrap,
            )
            scene_out["bootstrap_ci"] = {
                int(d): {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                         for k, v in ci.items()}
                for d, ci in boot_ci.items()
            }

            # Permutation test for D2
            perm = permutation_test_d2(
                prob_map, poly_masks,
                observed_d2_f2=d2_f2_vals[2],
                thr=thr_f2,
                n_perm=n_perm,
            )
            scene_out["perm_d2"] = perm

            # Polygon-level metrics
            scene_out["polygon"] = polygon_metrics(
                prob_map, poly_masks, threshold=thr_f2, iou_thresh=iou_thresh
            )

        scene_results[scene_name] = scene_out
        log.info(
            "%s: F1=%.4f (thr=%.3f)  F2=%.4f (thr=%.3f)",
            scene_name,
            pixel_metrics["best_f1"], pixel_metrics["thr_f1"],
            pixel_metrics["best_f2"], pixel_metrics["thr_f2"],
        )
        if is_test and gt_gdf is not None:
            log.info(
                "  D2 pixel F2=%.4f  95%%CI=[%.4f,%.4f]  perm p=%.4f",
                d2_f2_vals[2],
                boot_ci[2]["ci_lower"], boot_ci[2]["ci_upper"],
                perm["p_value"],
            )

    # Overall across all scenes
    overall_prob = np.concatenate(all_prob)
    overall_gt   = np.concatenate(all_gt_flat)
    overall = sweep_thresholds(overall_prob, overall_gt)
    overall["auprc"] = auprc(overall_prob, overall_gt)

    output = {
        "ckpt":          str(ckpt_path),
        "split":         split,
        "scene_results": scene_results,
        "overall":       overall,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info("Results saved → %s", out_path)
    return output


def main() -> None:
    import argparse, logging, sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        stream=sys.stderr,
    )

    p = argparse.ArgumentParser(description="Evaluate a Phase 2 checkpoint")
    p.add_argument("--ckpt",       required=True, type=Path)
    p.add_argument("--data-dir",   required=True, type=Path)
    p.add_argument("--stats",      required=True, type=Path)
    p.add_argument("--split",      default="val", choices=["val", "test"])
    p.add_argument("--out",        required=True, type=Path)
    p.add_argument("--no-tta",     action="store_true")
    p.add_argument("--iou-thresh", default=0.1,   type=float)
    p.add_argument("--n-bootstrap",default=10000, type=int)
    p.add_argument("--n-perm",     default=10000, type=int)
    p.add_argument("--morph-closing", action="store_true",
                   help="Apply binary_closing(3x3, iter=1) at F1/F2 thresholds (Gatti-style postproc)")
    args = p.parse_args()

    _evaluate_checkpoint(
        ckpt_path=args.ckpt,
        data_dir=args.data_dir,
        stats_path=args.stats,
        split=args.split,
        out_path=args.out,
        use_tta=not args.no_tta,
        iou_thresh=args.iou_thresh,
        n_bootstrap=args.n_bootstrap,
        n_perm=args.n_perm,
        morph_closing=args.morph_closing,
    )


if __name__ == "__main__":
    main()
