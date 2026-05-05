"""Per-deposit metadata for the click-to-inspect feature.

The labeled GT array and per-component static stats (area, bbox, centroid,
prob distributions) are computed once and cached. IoU is computed live on
click, using the user's current threshold + morph setting.
"""
from __future__ import annotations

from typing import TypedDict

import numpy as np
import streamlit as st
from scipy import ndimage

# 10 m Sentinel-1 GRD pixel ≈ 100 m²
PIXEL_AREA_M2 = 100


class Deposit(TypedDict):
    id: int
    size_px: int
    area_m2: int
    bbox: tuple[int, int, int, int]   # (y0, y1, x0, x1)
    centroid: tuple[int, int]          # (cy, cx)
    our_max_prob: float
    our_mean_prob: float
    gatti_max_prob: float
    gatti_mean_prob: float


@st.cache_data(show_spinner="Indexing deposits...")
def index_deposits(
    gt: np.ndarray,
    prob_ours: np.ndarray,
    prob_gatti: np.ndarray,
) -> tuple[np.ndarray, list[Deposit]]:
    """Label connected components in GT and gather static per-deposit stats."""
    lab, n = ndimage.label(gt > 0.5)
    deposits: list[Deposit] = []
    for cid in range(1, n + 1):
        mask = lab == cid
        ys, xs = np.where(mask)
        if len(ys) == 0:
            continue
        size = int(mask.sum())
        deposits.append({
            "id":              cid,
            "size_px":         size,
            "area_m2":         size * PIXEL_AREA_M2,
            "bbox":            (int(ys.min()), int(ys.max()),
                                int(xs.min()), int(xs.max())),
            "centroid":        (int(ys.mean()), int(xs.mean())),
            "our_max_prob":    float(prob_ours[mask].max()),
            "our_mean_prob":   float(prob_ours[mask].mean()),
            "gatti_max_prob":  float(prob_gatti[mask].max()),
            "gatti_mean_prob": float(prob_gatti[mask].mean()),
        })
    return lab, deposits


def best_iou_against(gt_mask: np.ndarray, pred_lab: np.ndarray) -> float:
    """Best IoU between gt_mask and any predicted connected component."""
    overlapping = np.unique(pred_lab[gt_mask])
    overlapping = overlapping[overlapping > 0]
    best = 0.0
    for pid in overlapping:
        pmask = pred_lab == pid
        inter = np.logical_and(gt_mask, pmask).sum()
        union = np.logical_or(gt_mask, pmask).sum()
        if union > 0:
            iou = float(inter) / float(union)
            if iou > best:
                best = iou
    return best


def instance_metrics(
    gt_lab: np.ndarray,
    pred_mask: np.ndarray,
    iou_thr: float = 0.3,
) -> dict:
    """Instance-level F1, precision, recall via greedy 1-1 IoU matching.

    Returns {f1, precision, recall, tp, fp, fn, n_pred, n_gt}.

    Pred components matching no GT count as FP; GT components matching no
    pred count as FN. This penalises the inflated low-threshold masks Gatti
    produces, which a pure recall count would not.
    """
    pred_lab, n_pred = ndimage.label(pred_mask)
    n_gt = int(gt_lab.max())
    if n_gt == 0:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0,
                "tp": 0, "fp": int(n_pred), "fn": 0,
                "n_pred": int(n_pred), "n_gt": 0}
    if n_pred == 0:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0,
                "tp": 0, "fp": 0, "fn": n_gt,
                "n_pred": 0, "n_gt": n_gt}

    # Full intersection matrix via flat bincount
    flat_g = gt_lab.ravel().astype(np.int64)
    flat_p = pred_lab.ravel().astype(np.int64)
    combined = flat_g * (n_pred + 1) + flat_p
    counts = np.bincount(combined, minlength=(n_gt + 1) * (n_pred + 1))
    intersect = counts.reshape((n_gt + 1, n_pred + 1))[1:, 1:]   # drop bg row/col

    gt_sizes   = intersect.sum(axis=1)
    pred_sizes = intersect.sum(axis=0)

    # IoU[g, p]
    union = gt_sizes[:, None] + pred_sizes[None, :] - intersect
    union = np.maximum(union, 1)
    iou = intersect / union

    # Greedy 1-1 matching by descending IoU
    iou_work = iou.copy()
    tp = 0
    while True:
        m = iou_work.max()
        if m < iou_thr or m == 0:
            break
        g, p = np.unravel_index(iou_work.argmax(), iou_work.shape)
        tp += 1
        iou_work[g, :] = 0
        iou_work[:, p] = 0

    fp = int(n_pred) - tp
    fn = int(n_gt)   - tp
    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)
    return {"f1": float(f1), "precision": float(precision),
            "recall": float(recall),
            "tp": int(tp), "fp": int(fp), "fn": int(fn),
            "n_pred": int(n_pred), "n_gt": int(n_gt)}
