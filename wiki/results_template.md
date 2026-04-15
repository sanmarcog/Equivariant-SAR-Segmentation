# Phase 2 Results — Template

> Fill in numbers when ablation tables land at `results/ablation_tables.json`.

---

## Headline claim

D4-equivariant CNN (~625K params) **{matches / approaches / falls short of}** SwinV2-Tiny (Gatti et al. 2026, 2.39M params, pixel F1=0.806, F2=0.841) on the Tromsø OOD test set, while {detecting / failing to detect} small (D2-class) avalanche deposits at meaningfully better recall.

---

## Table A — Overall pixel F1 / F2 on Tromsø (OOD test)

| # | Condition | F1 (mean ± std) | F2 (mean ± std) |
|---|---|---|---|
| 1 | Baseline (BCE, random sample, no skip) | TBD | TBD |
| 2 | + biased sampling | TBD | TBD |
| 3 | + Focal+Tversky loss | TBD | TBD |
| 4 | + U-Net skip connections | TBD | TBD |
| 5 | + copy-paste (full system) | TBD | TBD |
| — | Gatti et al. 2026 (SwinV2-Tiny) | 0.806 | 0.841 |

**Selected hparams from grid search (cond 5, 18 combos × 20 epochs):**
- γ (focal): {1, 2, 3} → winner: **TBD**
- α/β (Tversky): {0.3/0.7, 0.2/0.8} → winner: **TBD**
- pos_fraction: {0.4, 0.5, 0.6} → winner: **TBD**
- pos_frac=0.6 leads at every (γ, α/β) (4-for-4 monotonic improvement in 14-combo intermediate)

---

## Table B — D2-only pixel F2 (n=25 polygons, bootstrap 95% CI)

| # | Condition | D2 F2 mean ± std | 95% CI | perm p-value |
|---|---|---|---|---|
| 1 | Baseline | TBD | TBD | TBD |
| 2 | + biased sampling | TBD | TBD | TBD |
| 3 | + Focal+Tversky | TBD | TBD | TBD |
| 4 | + skip connections | TBD | TBD | TBD |
| 5 | + copy-paste | TBD | TBD | TBD |

**Statistical validity** (n_D2 = 25; below the threshold for naive 3-seed mean ± std):
- Bootstrap: 1,000 resamples with replacement per D-scale, 95% percentile interval
- Permutation: 1,000 shuffles of D-scale labels across all 117 polygons, p-value vs observed D2 F2

---

## Per-D-scale breakdown (supplementary)

Pixel F2 by D-scale class on best condition (cond 5, mean across 3 seeds):
| D-scale | n | F2 |
|---|---|---|
| D1 | 5 | TBD |
| D2 | 25 | TBD |
| D3 | 71 | TBD |
| D4 | 16 | TBD |

---

## Findings (interpretation — fill once results land)

### Did the equivariant CNN match the vision transformer?
- TBD comparison vs Gatti

### Did D2 detection improve materially?
- TBD vs random/baseline; bootstrap CI overlap with chance

### Which ablation step contributed most to D2 detection?
- TBD per-step delta from Table B

### Where does the model still fail?
- TBD: smallest D2 (<800 m²)? Mountain shadow regions? Look at false-negative breakdown.

---

## Methods notes for paper

- **Channel set (12-ch)**: VH/VV post + pre (6) + slope/sin(asp)/cos(asp)/LIA (4 terrain) + log_ratio_VH/VV (2 change) + xpol_post/pre (2 cross-pol). Validated qualitatively: log-ratio channels show distinct bright/dark signal at D2 polygons in Tromsø, raw VH/VV does not.
- **Biased sampling**: 5.9× boost in positive patches per batch (random=2.69 → biased@0.5=16.0); deposit-size distribution preserved (median seen-deposit-size 133 → 141 px), so small-deposit exposure is not suppressed.
- **Statistical validity**: bootstrap CIs and permutation p-values over Tromsø polygons (n_D2=25 too small for naive seed-only stats).
- **Compute**: All training on a single A40 GPU via Hyak ckpt partition, account `demo` (`MaxJobs=1` so all 15 train+eval runs are sequential in one job, ~15h total).
