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

## Active diagnostic: val F2=0.42 vs Gatti test F2=0.84

> ⚠ CRITICAL: as of 2026-04-15, the 0.42 number is val (Livigno), NOT test (Tromsø). Gatti's 0.84 is test (Tromsø). Tromsø has ~15× more deposit content per pixel (1.4% vs 0.09%), making F2 structurally easier. The Tromsø eval result — not the val number — is the real comparison. Waiting on cond 5 eval.

Three diagnosed factors contributing to the gap:
1. **No training-time augmentation** (FIXED — equivariance-aware augmentation branch in progress). Model memorized 3,300 positive patches by epoch 8.
2. **Val vs test scene mismatch** (NOT a bug — just can't compare yet). Livigno val has 0.09% positive pixels; Tromsø test has 1.4%. Structurally different F2 landscape.
3. **Threshold=0.82 signal**: model outputs near-zero probabilities everywhere, needs very high threshold to maximize F2. Suggests class imbalance undertreated. Cond 1 (BCE) in the ablation will show if loss is part of the problem.

**Pending diagnostics**: visual decoder smoke test on Tromsø D2 patches (blurry blob vs uniform zero vs wrong location). Single-batch overfit test (loss → 0 in 200 steps confirms optimizer not broken).

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
