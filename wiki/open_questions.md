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

## Tromsø results (2026-04-15) — overall strong, D2 failed

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

> ✓ DECIDED: D2 detection has failed (F2=0.06, permutation p=1.0). Model has zero D2-specific capability — D2 detections are spillover from nearby D3+ deposits. D2 as co-primary goal is dropped. Reframed as structured failure analysis in the paper.

### Paper reframe

The story is now: equivariant CNNs are parameter-efficient for SAR avalanche segmentation (positive claim) + small deposits remain below the effective detection floor at 64×64/10m even with targeted regularization (honest limitation with diagnostic evidence). The permutation test, per-D-scale ablation, and D2 visualization support this framing.

### Remaining diagnostics

- Augmentation branch: likely helps overall F2 (0.79→0.82–0.84), probably won't fix D2. Worth running.
- Patch size 128: could help D2 via more terrain context. Worth investigating if viz shows "blob near D3" pattern.
- Cond 4 vs cond 5: tells us if copy-paste augmentation contributed anything.
- Viz on D2 polygons: pending — determines whether 128×128 is worth trying.

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
