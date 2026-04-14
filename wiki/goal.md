# Goal — Phase 2

**Phase 2 moves from patch-level binary classification to pixel-level segmentation, enabling direct metric comparison with Gatti et al. 2026 and deposit area estimation for avalanche size classification.**

---

## What changes from Phase 1

| Dimension | Phase 1 | Phase 2 |
|-----------|---------|---------|
| Output | Patch probability (scalar) | Pixel mask (64×64) |
| Evaluation | AUC-ROC / patch hit-rate | Pixel-level F1/F2 (primary) + polygon-level (supplementary) |
| Size | Not addressed | Deposit area → D-scale proxy |
| Preprocessing | None | Speckle filtering (Lee/Refined Lee 5×5) |
| Calibration | Temperature scaling | Label smoothing (fixes T≈50 logit collapse) |
| Reproducibility | Single seed | 3-seed training for all conditions |
| Inference | Stride=32 (50% overlap) | Stride=16 (75% overlap) for D2 detection |

---

## Primary goals

1. **Match or beat Gatti et al. 2026** on pixel-level F1 ≥ 0.806 / F2 ≥ 0.841 on Tromsø OOD. Gatti uses SwinV2-Tiny (vision transformer, ~2.39M params); we use D4-equivariant CNN (~500–600K params).
2. **D2 detection**: demonstrate meaningful recall on D2-class deposits (25 polygons, ~600–5000 m²). This is the hard target — just above the speckle floor at 10m GRD. Everything in Phase 2 (75% overlap, biased sampling, Focal+Tversky loss, skip connections, copy-paste augmentation) is ultimately in service of not losing D2. D1 (n=5) is likely below the detection floor and not expected to be recoverable.

> The paper's central claim is that an equivariant CNN can match a vision transformer on overall metrics AND detect small avalanches that are inherently hard at this resolution. Parameter efficiency (~4× fewer) is supporting evidence, not the headline.

## Secondary goals

3. **Deposit area pipeline**: pixel mask → georeferenced polygon → area in m² → D-scale proxy label.
4. **D2 detection advantage**: test whether 64×64 patches at 75% overlap outperform 128×128 at 50% overlap for small deposits.

## Supplementary goal

**Snowpack fusion**: SAR deposit area + SeNorge ΔSD₇₂ + bulk density → D-scale classification on Tromsø. Novel contribution — no published paper has done this. Proof of concept only (single scene, not generalizable).

---

## Success criteria

> ✓ DECIDED: primary metric is **pixel-level** F1/F2 matching Gatti's protocol exactly (threshold sweep on val, report best F1 at F1-optimized threshold and best F2 at F2-optimized threshold). Polygon-level metrics reported as supplementary.

- **Full win**: pixel F1 ≥ 0.806 AND meaningful D2 recall (per-D-scale F2_D2 reported)
- **Partial win**: pixel F1 ≥ 0.806 but D2 recall weak — report what regularization helped and what didn't
- **Informative loss**: document what the capacity/resolution gap costs for small deposits; motivates Phase 3

See [evaluation.md](evaluation.md) for metric details and [baselines.md](baselines.md) for Gatti protocol.
