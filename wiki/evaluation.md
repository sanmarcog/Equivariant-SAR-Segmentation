# Evaluation — Phase 2

**Primary metric: pixel-level F1/F2 matching Gatti et al. 2026's protocol exactly. Supplementary: polygon-level metrics and per-D-scale breakdown.**

---

## Pixel-level evaluation (primary — for Gatti comparison)

> ✓ DECIDED: Gatti et al. 2026 report **pixel-level** F1/F2 using torchmetrics BinaryF1Score. Their repo contains no polygon matching code. To compare fairly, pixel-level F1/F2 is our primary metric.

### Protocol

1. Threshold predicted pixel mask at probability `t` → binary mask
2. Compute pixel-level precision, recall, F1, F2, IoU against GT raster
3. Report two operating points:
   - **F1-optimized**: threshold `t` that maximizes pixel F1 on validation set
   - **F2-optimized**: threshold `t` that maximizes pixel F2 on validation set

This matches Gatti's protocol exactly: sweep all unique predicted probabilities on val, select the threshold maximizing the target metric.

### Targets

| Metric | Gatti et al. 2026 | Our target |
|--------|-------------------|------------|
| Pixel F1 (F1-optimized threshold) | 0.8061 | ≥ 0.806 |
| Pixel F2 (F2-optimized threshold) | 0.8414 | ≥ 0.841 |

### F1 and F2 formulas

```
Precision = TP / (TP + FP)       (pixel-level)
Recall    = TP / (TP + FN)       (pixel-level)
F1 = 2 × P × R / (P + R)        — equal weight
F2 = 5 × P × R / (4P + R)       — recall weighted 2× more than precision
```

F2 is the primary reporting metric for hazard applications: missing a real avalanche (FN) is more costly than a false alarm (FP).

---

## Polygon-level evaluation (supplementary — our additional contribution)

> ✓ DECIDED: polygon-level metrics are reported as a supplementary analysis, NOT as the primary comparison with Gatti.

### Protocol

1. Threshold predicted pixel mask → binary mask
2. Extract connected components → predicted polygons
3. For each GT polygon, find predicted polygon with highest IoU
4. **Match** if IoU ≥ threshold; **miss** otherwise
5. Predicted polygons with no GT match = false positives
6. Compute polygon-level precision, recall, F1, F2

This is an object-level metric, not pixel-level. One GT polygon = one detection event.

> ⚠ OPEN: IoU threshold for polygon matching. Gatti does not use polygon matching, so there is no external reference. Choose a standard value (0.1 or 0.25) and report sensitivity to the threshold.

### Polygon hit rate

Additionally report polygon hit rate (= fraction of GT polygons containing ≥1 predicted positive pixel) for comparison with Gatti's reported 80.36%.

---

## Per-size-class breakdown

Report pixel F1/F2 AND polygon metrics separately for D1, D2, D3, D4 on Tromsø GT. This is the key diagnostic for the "D2 detection advantage" question.

> Note: Gatti et al. do not report per-D-scale breakdown. Our per-D-scale analysis is an independent contribution.

---

## Secondary metrics

| Metric | Purpose |
|--------|---------|
| Pixel-level IoU (Jaccard) | Standard segmentation diagnostic |
| AUPRC | Model selection metric (matches Gatti's protocol) |
| Area estimation MAE (m²) | Size estimation experiment |
| D-scale classification accuracy | Supplementary snowpack fusion experiment |

---

## Statistical validity

> ✓ DECIDED: bootstrap CIs + permutation test for D2. Details below.

### Training variance

3 seeds per configuration → report mean ± std across seeds. This captures model initialization variance.

### Bootstrap confidence intervals (per-D-scale)

For each trained model, bootstrap-resample the test polygons within each D-scale class
(10,000 draws with replacement), recompute per-class pixel F2 each draw, report 95% CI.

This is critical for D2 (n=25): 3-seed mean ± std hides the small-sample problem.
Bootstrap CIs give honest uncertainty bounds that reflect the actual number of test objects.
Costs seconds on CPU — no additional GPU time.

Report both: across-seed variance ("how stable is training?") and bootstrap CI
("how confident are we given n=25?").

### Permutation test for D2

Shuffle D-scale labels across all 117 Tromsø polygons, recompute "D2" pixel F2 on the
shuffled set, repeat 10,000 times. Report p-value.

This answers: "is our D2 detection significantly better than chance, or are we just
picking up D3 edges?" Cheap insurance against reviewer skepticism on small-n results.

---

## Success criteria (updated)

- **Full win**: pixel F1 ≥ 0.806 AND meaningful D2 recall (bootstrap CI above chance, permutation p < 0.05)
- **Partial win**: pixel F1 ≥ 0.806 but D2 recall weak — report which regularization techniques helped and which didn't
- **Informative loss**: document what the capacity/resolution gap costs for small deposits; motivates Phase 3

---

## What we are NOT reporting as primary

- AUC-ROC — Phase 1 metric, not comparable to Gatti
- Patch hit-rate — non-standard, Phase 1 only
- Polygon-level F1/F2 as the primary Gatti comparison (they don't use polygon matching)
