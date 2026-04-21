# Gatti-Mirror Results: D4-EquiCNN vs Swin-UNet

> **Note:** This page documents the **Gatti-mirror experiment** — one specific
> ablation where we trained our model using Gatti's exact recipe (128×128 patches,
> BCE pw=3.0, biased sampler). This is NOT the final model. The final model
> (condition 1, 64×64 patches, BCE pw=1.0, no sampler) achieves F1 = 0.794 —
> see the [main README](../README.md) for headline numbers.

> **Gatti-mirror protocol**: identical training recipe (patch 128×128, stride 64,
> BCE pos_weight=3.0, balanced sampler, 110 epochs, AdamW LR=1e-4,
> 10-epoch warmup + cosine, morph closing, no TTA). Only difference: architecture.

**Test set**: Tromsø 2024-12-20 (OOD, 117 polygons: D1=5, D2=25, D3=71, D4=16)

---

## Table 5 Slot-In (F1-opt, architecture comparison)

All baselines from Gatti et al. 2026, Table 5 (Max blending, F1-opt threshold).

| Model                          | Params (M) | F1     | IoU    |
|--------------------------------|------------|--------|--------|
| LFG-Net (Waldner et al.)      | 31.55      | 0.2814 | 0.1635 |
| DeepLabV3 (Chen et al.)       | 39.64      | 0.4765 | 0.3130 |
| U-Net (Ronneberger et al.)    | 31.04      | 0.4866 | 0.3213 |
| FCN8 (Long et al.)            | 134.27     | 0.6019 | 0.4304 |
| **D4-EquiCNN (Ours)**         | **0.63**   | **0.749** | **0.599** |
| RUNet (Weber)                 | 7.76       | 0.7667 | 0.6219 |
| A-BT-UNet (Guo et al.)       | 12.43      | 0.7927 | 0.6572 |
| Swin-UNet (Gatti et al.)      | 2.39       | 0.8027 | 0.6608 |

> Our model ranks **5th / 8** — behind the top 3 but with **3.8× fewer params**
> than the leader. The F1 gap to Swin-UNet is **−5.4 pp** (0.749 vs 0.803).

---

## Full Operating-Point Comparison (Gatti Table 4 format)

| Config                  | Recall | Precision | F1     | F2     | IoU    | Hit % (n=112) |
|-------------------------|--------|-----------|--------|--------|--------|----------------|
| **Gatti — Gauss F1-opt** | 0.793  | 0.820     | **0.806** | 0.799  | 0.675  | 65.2% (73)     |
| **Gatti — Max F2-opt**   | 0.877  | 0.702     | 0.780  | **0.841** | 0.639  | **80.4% (90)** |
| Ours — Gauss F1-opt     | 0.790  | 0.712     | 0.749  | 0.801  | 0.599  | 82.1% (92)     |
| Ours — Max F2-opt       | 0.878  | 0.614     | 0.758  | 0.809  | 0.611  | 83.0% (93)     |

> **Pixel metrics**: we trail Gatti by 5.7 pp F1 and 3.2 pp F2.
>
> **Polygon detection**: we **match or exceed** Gatti's hit rate at both operating
> points (82.1% vs 65.2% at F1-opt, 83.0% vs 80.4% at F2-opt), despite lower
> pixel scores. This is the pattern from Phase 1: our equivariant model finds
> avalanches well but segments boundaries less precisely.

---

## Per-D-Scale Polygon Hit Rate (Gatti Table 9 format)

Strict hit rate (≥50% polygon area covered), at F2-opt threshold.

| Config              | D2 (n=25) | D3 (n=71) | D4 (n=16) | All excl D1 (n=112) |
|---------------------|-----------|-----------|-----------|---------------------|
| Gatti — Max F2-opt  | **64.0%** (16) | 81.7% (58) | 100% (16) | 80.4% (90)          |
| Ours — Max F2-opt   | 56.0% (14)     | **88.7%** (63) | 100% (16) | **83.0%** (93)      |
| Δ (Ours − Gatti)    | −8.0 pp   | **+7.0 pp** | 0.0 pp    | **+2.6 pp**         |

At F1-opt threshold:

| Config              | D2 (n=25) | D3 (n=71) | D4 (n=16) | All excl D1 (n=112) |
|---------------------|-----------|-----------|-----------|---------------------|
| Gatti — Gauss F1-opt | 28.0% (7) | 70.4% (50) | 100% (16) | 65.2% (73)         |
| Ours — Gauss F1-opt  | **56.0%** (14) | **87.3%** (62) | 100% (16) | **82.1%** (92) |
| Δ (Ours − Gatti)     | **+28.0 pp** | **+16.9 pp** | 0.0 pp    | **+16.9 pp**    |

> **Key finding at F2-opt**: We lose 8 pp on D2 (small avalanches, hardest) but
> gain 7 pp on D3 (large, most numerous class). Net: +2.6 pp overall hit rate.
>
> **Key finding at F1-opt**: Massive advantage (+17 pp overall). Gatti's higher
> F1-opt threshold kills recall on smaller features; our model is more robust at
> this operating point.

---

## D2 Statistical Significance

| Metric                 | Gaussian F1-opt | Max F2-opt |
|------------------------|-----------------|------------|
| D2 pixel F2            | 0.057           | 0.055      |
| Bootstrap 95% CI       | [0.025, 0.049]  | [0.024, 0.047] |
| Permutation test p     | 1.000           | 1.000      |
| Null mean (random)     | 0.403           | 0.403      |

> D2 pixel-level F2 is not significantly above random (p=1.0). Both models struggle
> with D2 — these are 10²–10³ m³ deposits spanning only ~10–100 pixels at 10 m
> resolution.

---

## Summary

| Dimension         | Gatti (2.39M) | Ours (0.63M) | Winner |
|-------------------|---------------|--------------|--------|
| Params            | 2.39M         | **0.63M** (3.8×) | Ours   |
| F1 (pixel)        | **0.806**     | 0.749 (−5.7pp) | Gatti  |
| F2 (pixel)        | **0.841**     | 0.809 (−3.2pp) | Gatti  |
| Hit % (F2-opt)    | 80.4%         | **83.0%** (+2.6pp) | Ours   |
| Hit % (F1-opt)    | 65.2%         | **82.1%** (+16.9pp) | Ours   |
| D2 hit (F2-opt)   | **64.0%**     | 56.0% (−8pp)  | Gatti  |
| D3 hit (F2-opt)   | 81.7%         | **88.7%** (+7pp) | Ours   |
| D4 hit            | 100%          | 100%          | Tie    |

> **Story**: With 3.8× fewer parameters and built-in D4 equivariance, our model
> finds more avalanches overall than Gatti's Swin-UNet, but segments their
> boundaries less precisely — the precision-recall gap accounts for the lower
> pixel-level F-scores. The detection vs segmentation split is the core thesis
> tension.
