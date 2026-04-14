# Architecture — Phase 2

**D4-BT backbone + equivariant segmentation decoder + area regression head. Retrained from scratch.**

---

## Backbone: D4BiTemporalCNN

**Input split** (from the 12-channel scene array):
- Post branch (6ch): VH_post, VV_post, slope, sin(aspect), cos(aspect), LIA — indices 0–5
- Pre  branch (6ch): VH_pre, VV_pre, slope, sin(aspect), cos(aspect), LIA — indices [6,7,2,3,4,5]
- Extra channels (4ch): log_ratio_VH, log_ratio_VV, xpol_post, xpol_pre — indices 8–11

The extra 4 channels are not passed through the shared equivariant encoder (which requires identical input structure for both branches). They are injected directly into the decoder — see below.

**Encoder**: 5 blocks, shared weights, D4-equivariant, n_reg = [8, 16, 32, 32, 32]:
```
Block 1 (no pool): trivial_6 → reg×8   [B, n1·8, 64, 64]
Block 2 (pool×2):  reg×8  → reg×16  [B, n2·8, 32, 32]
Block 3 (pool×2):  reg×16 → reg×32  [B, n3·8, 16, 16]
Block 4 (pool×2):  reg×32 → reg×32  [B, n4·8,  8,  8]
Block 5 (pool×2):  reg×32 → reg×32  [B, n5·8,  4,  4]
```

**Change feature — multi-scale GroupPooling (intentional extension for segmentation)**:

The Phase 1 spec said `GroupPooling → global avg pool → [B, 256]` for the classification head. For a segmentation decoder this is insufficient — we need spatially-resolved change maps at each scale to localise deposit boundaries. So GroupPooling is applied at all 5 encoder scales, not just the bottleneck:

```
diff_i   = feat_post_i.tensor − feat_pre_i.tensor   (equivariant: g·diff = g·post − g·pre)
skip_i   = GroupPooling(diff_i).tensor               [B, n_i, H_i, W_i]  (invariant)
bottleneck = skip_5                                  [B, 32, 4, 4]
```

This gives the decoder multi-scale change evidence that varies across the 64×64 patch, which is critical for D2 boundary localisation. Global avg pool is NOT applied — spatial dims are preserved.

Dropout(0.3) applied to the bottleneck `[B, 32, 4, 4]` before decoding.

- Retrained from scratch with 12-channel input — Phase 1 checkpoint not reused

> Note: Gatti et al. 2026 use SwinV2-Tiny (vision transformer) with ~2.39M params and 8 input channels (no log-ratio or cross-pol ratio). Our comparison is **equivariant CNN vs vision transformer**, not equivariant vs standard CNN. Our 12-channel input adds 4 engineered features they don't use.

---

## Segmentation decoder

4-stage U-Net decoder (standard Conv2d, invariant features). At each stage, the grouped change skip `skip_i` and the 4 extra channels (avg-pooled to the stage's spatial size) are concatenated before the refinement conv.

```
Bottleneck: [B, 32, 4, 4]  (n5 change channels + dropout)

Stage 1:  cat([bot,   e_4 ])       → Conv2d(32+4→128) + BN + ELU  → [B, 128, 4, 4]
          ConvTranspose2d×2         →                                  [B, 128, 8, 8]
          cat([x, skip4, e_8 ])    → Conv2d(128+32+4→128) + BN + ELU → [B, 128, 8, 8]

Stage 2:  cat([x,   e_8 ])        → Conv2d(128+4→64) + BN + ELU   → [B,  64, 8, 8]
          ConvTranspose2d×2         →                                  [B,  64,16,16]
          cat([x, skip3, e_16])   → Conv2d(64+32+4→64)  + BN + ELU → [B,  64,16,16]

Stage 3:  cat([x,   e_16])        → Conv2d(64+4→32)  + BN + ELU   → [B,  32,16,16]
          ConvTranspose2d×2         →                                  [B,  32,32,32]
          cat([x, skip2, e_32])   → Conv2d(32+16+4→32) + BN + ELU  → [B,  32,32,32]

Stage 4:  cat([x,   e_32])        → Conv2d(32+4→16)  + BN + ELU   → [B,  16,32,32]
          ConvTranspose2d×2         →                                  [B,  16,64,64]
          cat([x, skip1, e_64])   → Conv2d(16+8+4→16)  + BN + ELU  → [B,  16,64,64]

Final:    Conv2d(16→1, k=1)       →                                  [B,   1,64,64]  logit
```

where `e_sz = AdaptiveAvgPool(extra_4ch, sz×sz)`.

**Why inject the 4 extra channels at every decoder stage?** The log-ratio and cross-pol channels are explicit change signals. Feeding them at each decoder scale gives the network direct access to the change evidence at every resolution — the equivariant encoder's implicit difference is supplemented by the explicit engineered features, particularly important at the finest (64×64) scale for small deposit outlines.

Skip connections preserve fine-scale spatial detail for small (D2) deposit boundaries.
Parameter count: **~625K** (verified; ~4× fewer than Gatti et al.'s 2.39M).

---

## Area regression head

- Input: soft mask (sigmoid of logit) for training; hard mask (threshold 0.5) for reporting
- `area_m2 = soft_mask.sum() × (10.0)²`
- D-scale proxy: log-linear mapping from area_m2 → {D1, D2, D3, D4} using Tromsø GT thresholds
- Supplementary only — not in the main segmentation loss

---

## Loss function

```
L = L_seg + λ_area × L_area
```

- `L_seg`: Focal loss (γ=2) + Tversky loss (α=0.3, β=0.7), equal weight
  - Focal: down-weights easy background pixels, forces focus on hard small-deposit pixels
  - Tversky: FN penalized 2.3× more than FP — optimizes recall on rare deposits
- `L_area`: L1 on log(area_m2) vs log(GT_area_m2), Tromsø samples only
- `λ_area = 0.1` (default; tune on val set)

> For reference, Gatti et al. 2026 use BCE with positive weight 3.0 (and also tested BCE+Dice and Focal Tversky with α=0.7, β=0.3, γ=1.33 — note their α/β are swapped relative to our convention where α=FP weight, β=FN weight).

---

## Training

**Patch sampling**: biased — 50% of each batch are patches containing ≥1 deposit pixel,
with extra weight on D1/D2-sized deposits. Prevents model from ignoring rare small deposits.
Remaining 50% sampled randomly (maintains background representation).

**Copy-paste augmentation**: paste D2/D1 deposit patches onto background regions within
the same region only (Livigno→Livigno, Nuuk→Nuuk, Pish→Pish — no cross-region pasting).
Apply Gaussian edge blending at paste boundary. Cap at 20–30% of positive patches per batch.
Monitor val precision as early warning for artifact learning.

**Dropout**: 0.3 on the bottleneck ([B, 32, 4, 4]).

**Regularization**: weight decay 1e-4 (L2).

**Hyperparameters to tune on val F2** (grid search before full training run):
- γ (focal loss): {1, 2, 3}
- α/β (Tversky): {0.3/0.7, 0.2/0.8}
- positive patch fraction: {0.4, 0.5, 0.6}

**Fixed hyperparameters**:
- Label smoothing: ε=0.05
- LR scheduler: cosine decay
- Seeds: 3 per configuration; report mean ± std

---

## Ablation plan

Run in this order; each condition uses 3 seeds:

| # | Condition | Purpose |
|---|-----------|---------|
| 1 | Baseline: 12ch input, random sampling, BCE, no skip connections | Phase 2 starting point |
| 2 | + biased patch sampling | Isolate sampling effect |
| 3 | + Focal + Tversky loss | Isolate loss effect |
| 4 | + U-Net skip connections | Isolate architecture effect |
| 5 | + copy-paste augmentation | Full system |

Report two ablation tables from the same runs (no extra compute):

**Table A — Overall**: pixel-level F1/F2 (primary) and polygon-level F1/F2 (supplementary) across all D-scales. Standard ablation presentation.

**Table B — D2-only**: pixel-level F2 for D2-class polygons only (n=25), with bootstrap 95% CIs. Directly answers "which technique actually helped small deposit detection?" If biased sampling and skip connections help D2 but copy-paste doesn't, that's a useful finding even if the overall table shows monotonic improvement.

D1 (n=5, below detection floor) and D3/D4 (easy targets) are not tabulated separately — report in supplementary per-D-scale breakdown only.

Conditions 1–5 are the paper's ablation tables.

---

## Inference

- Patch size: 64×64, stride: 16 (75% overlap)
- Patch logits stitched via average pooling in overlap regions
- Threshold at 0.5 → binary → connected components → polygon extraction
- TTA: horizontal + vertical flip (4 variants averaged)

---

## Preprocessing (applied scene-wide before patching)

Input channels (12 total — see [datasets.md](datasets.md) for full spec):

1. Refined Lee 5×5 on VH and VV (preserves edges; applied first)
2. LIA normalization on VH/VV
3. Compute log-ratio: `log(VH_post/VH_pre)`, `log(VV_post/VV_pre)`
4. Compute cross-pol ratio: `VH_post/VV_post`, `VH_pre/VV_pre`
5. Clip VH/VV to [−25, −5] dB
6. Normalize all channels with train-split stats (must recompute — Phase 1 stats invalid)

---

## Rejected approaches

- **Multi-temporal pre-event stacking**: toy experiment (2026-04-14) showed 2-scene stack
  has 2.6× higher std and worse polygon contrast than single AvalCD pre-event at all D-scales.
  Systematic snowpack change between acquisition dates dominates any noise benefit. Closed.
- **NL-SAR despeckling**: no maintained Python lib; marginal gain at 10m GRD. Closed.
