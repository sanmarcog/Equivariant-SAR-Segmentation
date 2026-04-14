# Architecture — Phase 2

**D4-BT backbone + equivariant segmentation decoder + area regression head. Retrained from scratch.**

---

## Backbone: D4BiTemporalCNN

- Shared-weight D4-equivariant encoder applied to post and pre patches separately
- Change feature: `GroupPooling(feat_post − feat_pre)` → global avg pool → [B, 256]
- Retrained from scratch with 12-channel input — Phase 1 checkpoint not reused

---

## Segmentation decoder

4-stage equivariant upsampling with U-Net skip connections from encoder:

```
Encoder features [B, 256, 4, 4]
  → R2Conv (128ch) + BN + ELU + ConvTranspose2d ×2  → [B, 128, 8, 8]  ← skip from enc stage 4
  → R2Conv (64ch)  + BN + ELU + ConvTranspose2d ×2  → [B, 64, 16, 16] ← skip from enc stage 3
  → R2Conv (32ch)  + BN + ELU + ConvTranspose2d ×2  → [B, 32, 32, 32] ← skip from enc stage 2
  → R2Conv (16ch)  + BN + ELU + ConvTranspose2d ×2  → [B, 16, 64, 64] ← skip from enc stage 1
  → Conv2d 1×1 (trivial repr → scalar)              → [B, 1, 64, 64]  (logit mask)
```

Skip connections preserve fine-scale spatial detail for small (D2) deposit boundaries.
Parameter count: ~500–600K (up from ~391K; still ~4× fewer than Gattimgatti's 2.39M).

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

---

## Training

**Patch sampling**: biased — 50% of each batch are patches containing ≥1 deposit pixel,
with extra weight on D1/D2-sized deposits. Prevents model from ignoring rare small deposits.
Remaining 50% sampled randomly (maintains background representation).

**Hyperparameters to tune on val F2** (grid search before full training run):
- γ (focal loss): {1, 2, 3}
- α/β (Tversky): {0.3/0.7, 0.2/0.8}
- positive patch fraction: {0.4, 0.5, 0.6}

**Fixed hyperparameters**:
- Label smoothing: ε=0.05
- Weight decay: 1e-4 (L2 regularization)
- LR scheduler: cosine decay
- Seeds: 3 per configuration; report mean ± std

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
