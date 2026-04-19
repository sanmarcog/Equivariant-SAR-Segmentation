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

1. **Approach Gatti et al. 2026** on pixel-level F2 on Tromsø OOD, with ~4× fewer parameters. Gatti uses SwinV2-Tiny (vision transformer, ~2.39M params); we use D4-equivariant CNN (~625K params). Current best deployment-honest result: **F2=0.774 at thr=0.5** (cond 2, biased sampling + BCE), **AUPRC=0.825**, vs Gatti F2=0.841. Gap is 7pp at standard threshold, with 4× fewer parameters.

> The paper's central claim is that an equivariant CNN approaches a vision transformer on overall segmentation metrics with 4× fewer parameters, using only BCE loss + biased sampling. D4-equivariance eliminates geometric augmentation (4/6 transforms redundant) and TTA (4× eval speedup). Frozen-threshold evaluation reveals that sweep-mode F2 can severely overstate unconventional-threshold architectures — a methodological contribution.

## D2 detection — bimodal, environmentally determined (2026-04-16)

2. **D2 detection is real and bimodal.** The strict F2=0.06 was a metric artifact (D3/D4 TPs counted as D2 FPs). Per-polygon analysis shows 15/25 D2 polygons detected at high confidence (mean prob >0.7), 7/25 clearly missed (<0.4). Size does NOT predict detection success — same-size polygons have opposite outcomes (#6 at 2519 m²=0.99 vs #7 at 2520 m²=0.30). The discriminator is environmental (SAR viewing geometry, terrain context). Paper claim: "D2 detection floor is set by SAR physics, not model architecture. 60% of D2 deposits are confidently detected; ~28% are plausibly SAR-invisible." Vs-bg F2 numbers from reeval will quantify this formally.

## Secondary goals

3. **Deposit area pipeline**: pixel mask → georeferenced polygon → area in m² → D-scale proxy label.
4. **Investigate D2 failure causes**: is it patch size (64×64 insufficient context), resolution (10m GRD), or capacity? Informs Phase 3 direction.

## Supplementary goal

**Snowpack fusion**: SAR deposit area + SeNorge ΔSD₇₂ + bulk density → D-scale classification on Tromsø. Novel contribution — no published paper has done this. Proof of concept only (single scene, not generalizable).

---

## Success criteria

> ✓ DECIDED: primary metric is **pixel-level** F1/F2 matching Gatti's protocol exactly (threshold sweep on val, report best F1 at F1-optimized threshold and best F2 at F2-optimized threshold). Polygon-level metrics reported as supplementary.

- **Full win**: pixel F2 ≥ 0.841 — NOT achieved. Gap is 7pp at deployment-honest threshold.
- **Strong partial win** (current result): F2=0.774@thr=0.5 (92% of Gatti, 4× fewer params) + D2 67% recall with sensor-visibility explanation + frozen-threshold methodological finding + clean ablation (biased sampling only positive technique). This is a publishable result with 5+ distinct contributions.
- **Informative loss**: document what the capacity/resolution gap costs; motivates Phase 3

See [evaluation.md](evaluation.md) for metric details and [baselines.md](baselines.md) for Gatti protocol.
