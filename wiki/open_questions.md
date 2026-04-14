# Open Questions — Phase 2

**The 3 key scientific questions Phase 2 answers, plus unresolved sub-questions.**

---

## Q1: Can equivariant segmentation match or beat Gatti et al. 2026 with fewer parameters?

**Operationalization**: Pixel-level F1 ≥ 0.806 and/or pixel F2 ≥ 0.841 on Tromsø OOD, with ~500–600K parameters (~4× fewer than Gatti's 2.39M SwinV2-Tiny).

> ✓ DECIDED: comparison is pixel-level F1/F2 (matching Gatti's protocol). Polygon-level F1/F2 is supplementary.

Sub-questions:
- Does the D4 symmetry group still provide benefit in a segmentation setting (vs classification)?
- Does label smoothing (ε=0.05) adequately fix the T≈50 logit collapse?
- Do our 4 additional engineered channels (log-ratio, cross-pol ratio) compensate for fewer parameters?

---

## Q2: Does 64×64 patching at 75% overlap give measurable D2 detection advantage vs 128×128?

**Operationalization**: Compare D2-class pixel F2 against our own 128×128 baseline (Gatti does not report per-D-scale).

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

## Additional unresolved decisions

- [evaluation.md](evaluation.md): IoU threshold for polygon matching (supplementary metric — Gatti doesn't use polygon matching, so we choose our own standard value)
- [evaluation.md](evaluation.md): bootstrap vs analytical CIs
- [baselines.md](baselines.md): 112 vs 117 polygon count — likely D1 exclusion (117 − 5 = 112) but needs verification from full paper text
- [datasets.md](datasets.md): D1 exclusion from size classifier, SeNorge variable names
- [size_estimation.md](size_estimation.md): empirical area thresholds for D-scale boundaries
- [baselines.md](baselines.md): Bianchi & Grahn 2025 — check if they use AvalCD
- [baselines.md](baselines.md): Bianchi et al. 2021 — not yet ingested
