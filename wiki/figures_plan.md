# Planned Figures — Phase 2

Tracked here so nothing gets lost between sessions. Add to paper/README when results are in.

---

## Confirmed figures

1. **12-channel visualization on D2 deposits.** Show log-ratio VH/VV vs raw VH/VV on 2–3 Tromsø D2 polygons (medium and small). Demonstrates that engineered channels carry clear signal where raw SAR is ambiguous. Already generated during pre-flight check (2026-04-15).

2. **Ablation Table A — overall pixel F1/F2.** 5 conditions × 3 seeds, mean ± std. Standard ablation presentation.

3. **Ablation Table B — D2 pixel F2 (both strict and vs-bg metrics).** Same 5 conditions, D2-class only. Now includes both metrics to show the metric artifact and the honest detection rate. Bootstrap CIs on vs-bg metric.

4. **Threshold vs D2 F2 diagnostic curve.** Plot D2 pixel F2 (vs-bg) as a function of threshold alongside overall F2 curve. Shows the detection–precision tradeoff for small deposits.

5. **Per-polygon D2 detection map.** For cond 1 (best system), show which of the 25 D2 polygons are consistently detected across 3 seeds vs consistently missed. More informative than a single F2_D2 number.

5b. **D2 bimodal detection grid (HEADLINE FIGURE).** Grid of all 25 D2 polygons showing SAR patch + probability overlay + GT boundary, sorted by mean predicted probability. Highlights the bimodal split: 15 high-confidence detections vs 7 clear misses. Key comparison pair: #6 (2519 m², prob=0.99) vs #7 (2520 m², prob=0.30) — same size, opposite outcomes. Generated at `~/Desktop/sar_viz/d2_grid.png`. This single figure carries the "detection is environmental, not dimensional" narrative.

5c. **Threshold vs D2 F2 curve (vs-bg metric).** Plot D2 vs-bg F2 as a function of threshold, showing that at threshold 0.5 recall jumps to ~0.60 from the strict metric's 0.06. Include the overall F2 curve for comparison. Shows the tradeoff between D2 recovery and overall precision.

---

## Possible figures (pending results)

6. **Cross-pol channel utility.** If ablation or feature importance shows cross-pol channels contribute minimally, include a note/figure showing they're land-cover signal not change signal (observed during pre-flight check #2).

7. **Deposit area scatter plot.** Predicted area vs GT area for Tromsø polygons, colored by D-scale. Shows area pipeline accuracy.

8. **SeNorge fusion confusion matrix.** D-scale classification with and without snowpack features. Only if the supplementary experiment gets run.
