# Planned Figures — Phase 2

Tracked here so nothing gets lost between sessions. Add to paper/README when results are in.

---

## Confirmed figures

1. **12-channel visualization on D2 deposits.** Show log-ratio VH/VV vs raw VH/VV on 2–3 Tromsø D2 polygons (medium and small). Demonstrates that engineered channels carry clear signal where raw SAR is ambiguous. Already generated during pre-flight check (2026-04-15).

2. **Ablation Table A — overall pixel F1/F2.** 5 conditions × 3 seeds, mean ± std. Standard ablation presentation.

3. **Ablation Table B — D2-only pixel F2 with bootstrap 95% CIs.** Same 5 conditions, D2-class only. Answers "which regularization technique helps small deposits?"

4. **Threshold vs D2 F2 diagnostic curve.** Plot D2 pixel F2 as a function of threshold alongside overall F2 curve. Shows whether overall-optimal threshold is also good for D2, or if small deposits want a different operating point. (Loose end from design review — not yet in wiki evaluation protocol, but planned.)

5. **Per-polygon D2 detection map.** For the full system (condition 5), show which of the 25 D2 polygons are consistently detected across 3 seeds vs consistently missed. More informative than a single F2_D2 number.

---

## Possible figures (pending results)

6. **Cross-pol channel utility.** If ablation or feature importance shows cross-pol channels contribute minimally, include a note/figure showing they're land-cover signal not change signal (observed during pre-flight check #2).

7. **Deposit area scatter plot.** Predicted area vs GT area for Tromsø polygons, colored by D-scale. Shows area pipeline accuracy.

8. **SeNorge fusion confusion matrix.** D-scale classification with and without snowpack features. Only if the supplementary experiment gets run.
