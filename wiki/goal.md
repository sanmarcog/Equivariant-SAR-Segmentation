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

## Primary goal

1. **Match or beat Gatti et al. 2026** on pixel-level F1 ≥ 0.806 / F2 ≥ 0.841 on Tromsø OOD, with ~4× fewer parameters. Gatti uses SwinV2-Tiny (vision transformer, ~2.39M params); we use D4-equivariant CNN (~625K params). Current result: F2=0.79 (5pp gap). Augmentation expected to close most of this.

> The paper's central claim is that an equivariant CNN can match a vision transformer on overall segmentation metrics with 4× fewer parameters, and that D4-equivariance provides a strong inductive bias that eliminates the need for geometric augmentation while maintaining competitive performance.

## D2 detection — reframed as failure analysis

2. **D2 detection failed** (F2=0.06, permutation p=1.0). Despite biased sampling, Focal+Tversky loss, skip connections, and copy-paste augmentation, the model has zero D2-specific capability. The paper presents this as a structured negative result: what was tried, why it failed, what it implies for small-deposit detection at 10m GRD.

## Secondary goals

3. **Deposit area pipeline**: pixel mask → georeferenced polygon → area in m² → D-scale proxy label.
4. **Investigate D2 failure causes**: is it patch size (64×64 insufficient context), resolution (10m GRD), or capacity? Informs Phase 3 direction.

## Supplementary goal

**Snowpack fusion**: SAR deposit area + SeNorge ΔSD₇₂ + bulk density → D-scale classification on Tromsø. Novel contribution — no published paper has done this. Proof of concept only (single scene, not generalizable).

---

## Success criteria

> ✓ DECIDED: primary metric is **pixel-level** F1/F2 matching Gatti's protocol exactly (threshold sweep on val, report best F1 at F1-optimized threshold and best F2 at F2-optimized threshold). Polygon-level metrics reported as supplementary.

- **Full win**: pixel F2 ≥ 0.841 with ~4× fewer parameters + honest D2 failure analysis with diagnostic evidence
- **Partial win** (current trajectory): pixel F2 ≈ 0.79–0.84, gap partially closed by augmentation, D2 failure documented
- **Informative loss**: document what the capacity/resolution gap costs; motivates Phase 3

See [evaluation.md](evaluation.md) for metric details and [baselines.md](baselines.md) for Gatti protocol.
