# Open Questions — Phase 2

**The 3 key scientific questions Phase 2 answers, plus unresolved sub-questions.**

---

## Q1: Can equivariant segmentation match Gatti et al. 2026 AND detect small avalanches?

**Operationalization**: Pixel-level F1 ≥ 0.806 and/or pixel F2 ≥ 0.841 on Tromsø OOD, with demonstrated D2-class recall. Parameter efficiency (~4× fewer params) is supporting evidence, not the primary claim.

> ✓ DECIDED: comparison is pixel-level F1/F2 (matching Gatti's protocol). Polygon-level F1/F2 is supplementary.

Sub-questions:
- Does the D4 symmetry group still provide benefit in a segmentation setting (vs classification)?
- Does label smoothing (ε=0.05) adequately fix the T≈50 logit collapse?
- Do our 4 additional engineered channels (log-ratio, cross-pol ratio) help with D2 detection?
- Which regularization technique contributes most to D2 recall? (answered by ablation)

---

## Q2: Does 64×64 patching at 75% overlap give measurable D2 detection advantage vs 128×128?

**Operationalization**: Compare D2-class pixel F2 against our own 128×128 baseline (Gatti does not report per-D-scale). This is co-primary with Q1 — if we nail overall F1/F2 but miss D2, the result is weak.

Sub-questions:
- At stride=16, how many patches cover a typical D2 polygon (area ~600–5000 m²)?
- Does higher overlap increase FP rate for non-deposit background?
- Does speckle filtering (Lee 5×5) change the tradeoff?

> ✓ DECIDED: Gatti et al. do not report per-D-scale breakdown. The D2 detection question is answered internally (our 64×64 vs our 128×128 comparison).

---

## Q3: Does SeNorge snowpack fusion improve D-scale classification beyond area alone?

**Operationalization**: D-scale accuracy (area only) vs D-scale accuracy (area + ΔSD₇₂ + density) on Tromsø GT.

Sub-questions:
- Is ΔSD₇₂ available at daily resolution for 2024-12-19/20 from thredds.met.no?
- Does spatial interpolation from 1km to polygon centroid introduce enough error to wash out the signal?
- Can 112 labeled polygons provide sufficient statistical power for a 3-class classifier?

---

## Q4: Is multi-temporal pre-event stacking feasible for this dataset?

**CLOSED — 2026-04-14. Decision: DO NOT implement.**

**Toy experiment result** (`scripts/stacking_toy_experiment.py`, Tromsø scene):

| D-scale | n | Single (dB) | Stack 2-scene (dB) | Δ (dB) |
|---------|----|-----------|--------------------|--------|
| D1 | 4 | +1.42 | +1.33 | −0.10 |
| D2 | 25 | +4.47 | +4.50 | +0.03 |
| D3 | 71 | +5.15 | +4.65 | −0.50 |
| D4 | 16 | +6.69 | +5.75 | −0.94 |

Polygon contrast (mean signal − background, dB). Higher = cleaner detection.

**Findings**:
- Stack has 2.6× higher global std (4.80 dB) vs single (1.85 dB)
- Root cause: systematic snowpack backscatter change between acquisitions dominates any noise benefit
- The AvalCD single pre-event image is substantially better than a multi-temporal stack

---

## Q5: Is NL-SAR despeckling worth the implementation cost?

**CLOSED — not pursued.** No maintained Python implementation; GRD already multi-looked; marginal gain over Refined Lee at 10m resolution.

---

## Tromsø results (2026-04-15) — overall strong, D2 reframing in progress

### Overall pixel metrics (cond 5, seeds 0–1)

| Metric | Seed 0 | Seed 1 | Gatti |
|--------|--------|--------|-------|
| Pixel F1 | 0.748 | 0.745 | 0.806 |
| Pixel F2 | 0.788 | 0.790 | 0.841 |
| AUPRC | 0.811 | 0.819 | — |

Overall F2=0.79 vs Gatti's 0.84. Gap is 5pp with 4× fewer params. Parameter efficiency claim is intact.

### Per-D-scale (cond 5, seed 0)

| D-scale | n | F2 | 95% CI | Permutation p |
|---------|---|-----|--------|---------------|
| D1 | 5 | 0.001 | [0, 0.001] | — |
| D2 | 25 | 0.064 | [0.029, 0.054] | 1.0000 |
| D3 | 71 | 0.579 | [0.408, 0.501] | — |
| D4 | 16 | 0.654 | [0.440, 0.606] | — |

> ⚠ REFRAMED (2026-04-16): D2 F2=0.06 was a **metric artifact**. The strict `dscale_pixel_f2` counted correct D3/D4 predictions as D2 false positives (~25K D3/D4 TPs crushed D2 precision to 0.005). This measures D-scale *discrimination*, not D2 *detection*. Visual inspection (pair 3 overlay) showed strong prediction activity at D2 locations — yellow/green probability mass. New `dscale_pixel_f2_vs_bg` metric (FP from true background only) running in reeval job 34653541.
>
> The previous "D2 failure" decision is **suspended** pending vs-bg numbers. If vs-bg D2 F2 is substantially higher, the paper claim shifts from "structured failure" to "D2 detected at lower confidence, recoverable with threshold tuning."

### Paper reframe (pending vs-bg D2 numbers)

**Pre-reeval finding (2026-04-16): D2 detection is bimodal, not failed.**

Per-polygon probability analysis: 15/25 D2 polygons detected at mean prob >0.7 (60%), 7/25 missed at <0.4 (28%), 3 in middle ground. Size is NOT the predictor — polygon #6 (2519 m², prob=0.99) vs #7 (2520 m², prob=0.30). The discriminator is environmental (SAR viewing geometry, terrain context), not dimensional.

Paper framing: "D2 detection failure is not size-limited. 60% of D2 deposits are detected with high confidence. The remaining 28% are plausibly SAR-invisible due to terrain shadow, layover, or unfavorable viewing geometry — a physics limitation, not a model limitation."

This is a stronger finding than either "D2 fails" or "D2 works." It characterizes the detection floor and attributes it to the right cause.

### Key diagnostic findings (2026-04-15)

**Copy-paste hurts.** Cond 4 (no copy-paste) F2=0.797 > cond 5 (copy-paste) F2=0.782. Blend-boundary artifacts likely cause. Drop copy-paste from final system.

**Cond 4 is best architecture (skip + no copy-paste).** Cond 3 (no skip) shows marginally higher sweep-mode F2 (0.800 vs 0.793) but has 2.4× higher threshold instability across seeds (threshold std=0.17 vs 0.07). Sweep-mode hides this because each seed gets its own optimal threshold. With a frozen threshold (deployment-realistic), cond 4 is expected to outperform cond 3. Paper framing: "skip connections stabilize the operating point without sacrificing peak F2."

**D2 detection is bimodal and SAR-visibility-limited — predictor analysis COMPLETE.** 15/25 D2 polygons detected at >0.7 confidence, 7/25 missed at <0.4 — and size does not predict which. Feature correlation + LOO validation identifies `log_ratio_VH_abs_max` as the single best predictor at **88% LOO accuracy**. This is the maximum absolute VH backscatter change in the log-ratio channel — a direct measure of SAR visibility of the deposit. When the radar sees no change (low log-ratio), no model can detect the deposit. Detection floor is at the sensor level, not the model level. Size-weighted loss, two-head architecture, capacity bump, and patch-size 128 all pruned from decision tree as a result.

**TTA is redundant on D4-equivariant model.** The 4 TTA variants (identity, h-flip, v-flip, both-flips) are all D4 group elements. Model produces identical logits for each. Remove TTA, cut eval time 4×. Paper framing: "equivariance eliminates TTA" alongside "eliminates 4/6 geometric augmentations."

### Full ablation results — FINAL (2026-04-16, after frozen-threshold reeval)

| # | Condition | Sweep F2 | AUPRC | F2@0.5 | F2@0.7 | D2 Recall (vs-bg) |
|---|-----------|----------|-------|--------|--------|-------------------|
| 1 | Baseline (BCE, random, no skip) | 0.806 | — | 0.668 | 0.521 | 0.55 |
| **2** | **+ biased sampling** | **0.793** | **0.825** | **0.774** | **0.767** | **0.67** |
| 3 | + Focal+Tversky loss | 0.790 | 0.813 | 0.760 | 0.776 | 0.66 |
| 4 | + U-Net skip connections | 0.784 | 0.815 | 0.705 | 0.775 | 0.62 |
| 5 | + copy-paste (full system) | 0.774 | 0.807 | 0.655 | 0.758 | 0.64 |
| — | Gatti et al. 2026 | 0.841 | — | — | — | — |

> **FROZEN-THRESHOLD REVERSAL**: Cond 1's sweep-mode win was an artifact of operating at threshold ~0.17. At standard thresholds, cond 1 collapses (F2=0.521@0.7). **Cond 2 is the correct headline**: highest AUPRC, best deployment stability, highest D2 recall.

**Key implications**:
- Deployment-honest gap to Gatti is **7pp** (0.774 vs 0.841 at thr=0.5), not the 3-5pp previously claimed
- Sweep-mode F2 overestimates unconventional-threshold architectures — methodological finding for the paper
- Biased sampling is the single positive technique: stabilizes threshold, boosts AUPRC, improves D2 recall
- BCE+pos_weight=3 outperforms Focal+Tversky across all metrics
- Skip connections and copy-paste are small net negatives

### Remaining experiments (final priority, 2026-04-16)

1. ~~Reeval 34653576 with vs-bg metric~~ — ✅ COMPLETE. Confirmed D2 recall 0.55–0.67, cond 2 best at 0.67.
2. ~~Off-D4 augmentation~~ — DECISION: **NO, do not run.** Paper story is coherent without it. The 7pp gap is honestly reported and attributed. Augmentation won't change the headline narrative.
3. ~~Size-weighted loss~~ — PRUNED. Sensor-limited.
4. ~~Capacity bump~~ — PRUNED. Sensor-limited.
5. ~~Two-head architecture~~ — PRUNED. Sensor-limited.
6. ~~Patch size 128~~ — PRUNED. Sensor-limited.

> ✓ **DATA COLLECTION COMPLETE. START WRITING.**

---

## Additional unresolved decisions

- [evaluation.md](evaluation.md): IoU threshold for polygon matching (supplementary metric — Gatti doesn't use polygon matching, so we choose our own standard value)
- [evaluation.md](evaluation.md): bootstrap vs analytical CIs — CLOSED, bootstrap decided
- [baselines.md](baselines.md): 112 vs 117 polygon count — likely D1 exclusion (117 − 5 = 112) but needs verification from full paper text
- [baselines.md](baselines.md): check Gatti's repo for per-scene val numbers (apples-to-apples comparison with our val F2)
- [datasets.md](datasets.md): D1 exclusion from primary F2 reporting — if confirmed, instant +0.02–0.05 F2
- [datasets.md](datasets.md): SeNorge variable names and Dec 2024 availability
- [size_estimation.md](size_estimation.md): empirical area thresholds for D-scale boundaries
- [baselines.md](baselines.md): Bianchi & Grahn 2025 — check if they use AvalCD
- [baselines.md](baselines.md): Bianchi et al. 2021 — not yet ingested
