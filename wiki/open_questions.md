# Open Questions — Phase 2

**The 3 key scientific questions Phase 2 answers, plus unresolved sub-questions.**

---

## Q1: Can equivariant segmentation match or beat Gattimgatti 2026 with 6× fewer parameters?

**Operationalization**: F1 ≥ 0.806 and/or F2 ≥ 0.841 on Tromsø OOD, with ≤391K parameters.

Sub-questions:
- Does the D4 symmetry group still provide benefit in a segmentation setting (vs classification)?
- Does freezing the Phase 1 backbone hurt performance relative to fine-tuning end-to-end?
- Does label smoothing (ε=0.05) adequately fix the T≈50 logit collapse?

---

## Q2: Does 64×64 patching at 75% overlap give measurable D2 detection advantage vs 128×128?

**Operationalization**: Compare D2-class recall (F2_D2) against Gattimgatti's equivalent metric if reported.

Sub-questions:
- At stride=16, how many patches cover a typical D2 polygon (area ~600–5000 m²)?
- Does higher overlap increase FP rate for non-deposit background?
- Does speckle filtering (Lee 5×5) change the tradeoff?

> ⚠ OPEN: Gattimgatti 2026 may not break down results by D-scale. If they don't, this question can only be answered internally (our model at 64×64 vs 128×128).

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

| D-scale | n  | Single (dB) | Stack 2-scene (dB) | Δ (dB) |
|---------|----|-------------|-------------------|--------|
| D1      | 4  | +1.42       | +1.33             | −0.10  |
| D2      | 25 | +4.47       | +4.50             | +0.03  |
| D3      | 71 | +5.15       | +4.65             | −0.50  |
| D4      | 16 | +6.69       | +5.75             | −0.94  |

Polygon contrast (mean signal − background, dB). Higher = cleaner detection.

**Findings**:
- Stack has 2.6× higher global std (4.80 dB) vs single (1.85 dB) — the stack change image is
  dominated by large-scale structured variability, not reduced noise.
- Visually: single shows clean, spatially localized red spots at deposit locations; stack shows
  broad terrain-following patterns that completely mask the deposit signal.
- Root cause: in winter Arctic terrain, snowpack backscatter changes systematically between
  acquisitions (Nov 21 + Dec 3 vs the actual AvalCD pre-event). A median of 2 scenes
  from 2–4 weeks earlier does NOT represent the actual surface state before the event.
- With only 2 valid descending scenes available (Dec 8 zip was corrupt), median = mean,
  providing no robustness against temporal outliers.

**Conclusion**: The AvalCD single pre-event image (close in time, same orbit, calibrated
to the same processing baseline) is substantially better than a multi-temporal stack.
Multi-temporal stacking is not worth implementing for Phase 2. Close Q4.

---

## Q5: Is NL-SAR despeckling worth the implementation cost?

**Hypothesis**: Non-local SAR despeckling preserves deposit edges better than Refined Lee,
improving D2 boundary delineation in the segmentation mask.

Feasibility questions:
- Is there a maintained Python implementation of NL-SAR compatible with our pipeline
  (rasterio/numpy inputs)?
- What is the runtime cost per scene vs Refined Lee? (NL-SAR is O(n²) in patch comparisons)
- Is the edge-preservation gain measurable on AvalCD GRD data, or is it marginal at 10m
  resolution where speckle is already partially averaged by multi-looking?

**Decision gate**: benchmark Refined Lee vs NL-SAR on one Tromsø scene before committing.
If runtime is acceptable and D2 boundary IoU improves, include. Otherwise Refined Lee is
sufficient.

---

## Additional unresolved decisions

See `> ⚠ OPEN` callouts in:
- [architecture.md](architecture.md): freeze vs fine-tune, skip connections, speckle filter timing, Dice vs BCE
- [evaluation.md](evaluation.md): exact IoU threshold used by Gattimgatti, bootstrap vs analytical CIs
- [datasets.md](datasets.md): D1 exclusion from size classifier, SeNorge variable names
- [size_estimation.md](size_estimation.md): empirical area thresholds for D-scale boundaries
- [baselines.md](baselines.md): Gattimgatti architecture details (need to read paper)
