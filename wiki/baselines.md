# Baselines — Phase 2

**Primary comparison: Gatti et al. 2026 (arXiv:2603.22658).**

> ✓ LINT 2026-04-16: all numbers below verified against the PDF (pages 1-14).
> Prior wiki had several errors (parameter count, missing Table 9, missing Hit%
> definition) — corrected in this revision. See bottom of file for audit trail.

---

## Gatti et al. 2026 — verified fact sheet

### Metadata

| Property | Value |
|---|---|
| arXiv | 2603.22658v1, 24 Mar 2026 |
| Title | Large-Scale Avalanche Mapping from SAR Images with Deep Learning-based Change Detection |
| Authors | Mattia Gatti, Alberto Mariani, Ignazio Gallo, Fabiano Monti |
| Corresponding author | mgatti3@uninsubria.it |
| Code | [github.com/mattiagatti/avalanche-deep-change-detection](https://github.com/mattiagatti/avalanche-deep-change-detection) |
| Data | [zenodo.15863589](https://doi.org/10.5281/zenodo.15863589) |

### Dataset (AvalCD)

- 4 regions: Livigno (Italy), Nuuk (Greenland), Pish (Tajikistan), Tromsø (Norway)
- 7 events total, 796 train+val polygons + 117 test polygons (Tromsø_20241220)
- Pixel spacing: 10×10 m for all regions except Nuuk (5×5 m due to polar latitude)
- Tromsø held out entirely (no train, no val) as OOD test

### Tromsø test composition

- **117 polygons total** by D-scale: D2=25, D3=71, D4=16, plus D1=5 (5 not printed in Gatti's Table 1, but 117−112=5 per Gatti's exclusion logic)
- **112 evaluated** — Gatti excludes D1 (size 1) because "their spatial extent can fall below the effective ground resolution of Sentinel-1 imagery"
- EAWS D-scale volume (Table 6): D1 <10² m³, D2 10²-10³, D3 10³-10⁴, D4 10⁴-10⁵, D5 >10⁵

### Architecture

| Component | Details |
|---|---|
| Encoder | **Swin Transformer V2 Tiny** (vision transformer) with 4 stages |
| Branches | 2 weight-shared SAR encoders (pre/post) + 1 separate AUX encoder (LIA + slope) |
| Fusion | Element-wise difference of deepest SAR features, concat with AUX features |
| Decoder | Hierarchical Swin Transformer with skip connections to encoder stages |
| **Params (multimodal, with AUX)** | **70.58M** (Section 3.3 text, Figure 4) |
| **Params (unimodal, SAR only)** | **2.39M** (Table 5) — **this is the one they report as their main result** |
| Input channels | 8: pre VV, pre VH, post VV, post VH + aux (DEM-derived slope + LIA) |
| Patch size (best) | **128×128** (Tables 3, 4) |

> ⚠ **Crucial clarification**: Gatti's Section 5.2 ablation showed AUX inputs
> did NOT help ("virtually identical, and in some cases slightly improved,
> performance" without them). Their headline results (F1=0.8061, F2=0.8414)
> use the **unimodal 2.39M variant**. When comparing parameter efficiency,
> use 2.39M as the reference, not 70.58M.

### Training

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| LR schedule | 10-epoch linear warmup + 100-epoch cosine anneal = 110 total |
| Loss | **BCE with pos_weight=3.0** (Eq. 2: L = −w_p·y·log(ŷ) − (1−y)·log(1−ŷ)) |
| Sampling | Balanced sampler: equal positive (≥1 avalanche pixel) and negative per epoch, event-aware across 6 training events |
| Patch sizes tested | 32, 64, 128 with strides 16, 32, 64 |

### Augmentations (Section 4.2)

Synchronized across pre-event, post-event, aux, mask:
- Random horizontal flip
- Random 90° rotation (k ∈ {0,1,2,3})
- Mild affine (rotation, translation, scale, shear)

Radiometric (SAR channels only):
- Gaussian noise addition
- Random gain perturbation (intensity scaling)

### Inference

| Property | Value |
|---|---|
| Patch size | 128×128, stride 64 (50% overlap) |
| Blending strategies tested | None, Min, Max, Mean, Gaussian, Center Crop |
| Best for F1 | **Gaussian** blending |
| Best for F2 | **Max** blending |
| Post-processing | Morphological closing (kernel=3, iter=1) |
| Threshold | Tuned on val to maximize F1 or F2 — **numeric values NOT reported in paper** |
| Hardware | NVIDIA A100 GPU, ~87 km²/s inference |

### Pixel-level metrics (Table 8, final comparison)

| Config | Recall | Precision | F1 | F2 | IoU | Hit % (n=112) |
|---|---|---|---|---|---|---|
| Max F1-opt | 0.8136 | 0.7924 | 0.8029 | 0.8076 | 0.6707 | 68.75% (77) |
| Gauss F1-opt | 0.7928 | 0.8199 | **0.8061** | 0.7991 | 0.6752 | 65.18% (73) |
| **Max F2-opt** | **0.8771** | 0.7021 | 0.7799 | **0.8414** | 0.6392 | **80.36% (90)** |
| Gauss F2-opt | 0.8633 | 0.7345 | 0.7937 | 0.8324 | 0.6579 | 78.57% (88) |

**Abstract's headline numbers:**
- F1 = **0.8061** → Gaussian blending + F1-opt threshold
- F2 = **0.8414** → Max blending + F2-opt threshold (and the 80.36% hit rate)

### Polygon "Hit %" definition

> ✓ VERIFIED from Table 7 caption (page 11):
> **"A reference polygon is detected when at least 50% of its area is predicted as avalanche."**
>
> This is STRICTER than a "≥1 pixel overlap" definition. Our own `polygon_metrics`
> in `src/evaluate.py` uses ≥1 pixel (permissive). Any cross-paper comparison of
> hit-rate numbers requires we use Gatti's stricter definition.

### Per-D-scale polygon hit rates (Table 9)

All at ≥50% area coverage, 128×128 patches:

| Config | D2 Hit (n=25) | D3 Hit (n=71) | D4 Hit (n=16) |
|---|---|---|---|
| Max F1-opt | 32.00% (8) | 74.65% (53) | 100% (16) |
| Gauss F1-opt | 28.00% (7) | 70.42% (50) | 100% |
| **Max F2-opt** | **64.00% (16)** | 81.69% (58) | 100% |
| Gauss F2-opt | 60.00% (15) | 80.28% (57) | 100% |

> 🔑 **The per-D-scale numbers we wanted are in this table.** No separate email
> to Gatti needed for these.

### Baselines tested by Gatti (Table 5, F1-opt, F1 + IoU only)

| Model | Params (M) | F1 | IoU |
|---|---|---|---|
| SiamUnet-diff (Daudt 2018) | 1.35 | 0.7370 | 0.5836 |
| SiamUnet-conc (Daudt 2018) | 1.35 | 0.7498 | 0.5997 |
| STANet (Chen & Shi 2020) | 2.42 | 0.6460 | 0.4771 |
| BIT (Chen 2021) | 2.97 | 0.6999 | 0.5383 |
| SNUNet-CD (Fang 2021) | 12.03 | 0.7546 | 0.6059 |
| ChangeFormer (Bandara 2022) | 55.26 | 0.7720 | 0.6197 |
| TinyCD (Codegoni 2023) | 0.29 | 0.7660 | 0.6208 |
| STNet (Ma 2023) | 14.60 | 0.7777 | 0.6362 |
| **Swin-UNet (this work)** | **2.39** | **0.8027** | **0.6608** |

---

## Differences to document when writing the paper

1. **Parameter count**: Gatti 2.39M vs ours 625K → **~4× fewer** (not 113×)
2. **Patch size**: Gatti 128×128 vs ours 64×64 — we did NOT test 128 on our model
3. **Augmentation**: Gatti uses flips, 90° rotations, affine, radiometric; our sar-seq has NONE of these. The `aug-equivariant` branch implements only the off-D4 subset (since our model is D4-equivariant by construction, flips/90° are redundant).
4. **Inference blending**: Gatti uses Max (for F2) or Gaussian (for F1); ours uses averaging.
5. **Input channels**: Gatti 8 channels (SAR + LIA + slope); ours 12 (SAR + LIA + slope + aspect_sin/cos + log_ratio_VH/VV + xpol_post/pre). Note: Gatti's ablation shows AUX doesn't help — relevant for our claim about channel expansion.
6. **Loss**: Gatti BCE+pos_weight=3 (their best); we tested Focal+Tversky (cond 3-5) and BCE+pos_weight (cond 1-2). **Cond 2 matches Gatti's loss.**
7. **Polygon metric**: Gatti ≥50% area; our `src/evaluate.py::polygon_metrics` uses ≥1 pixel. Need to add a strict version before making any hit-rate comparison.
8. **D1 handling**: Gatti excludes D1 from object-level eval (112 polygons); our evaluate.py includes all 117.
9. **Threshold values**: neither paper reports exact numeric thresholds. Threshold tuning protocol: both sweep on val, pick best F1 or F2.

---

## Internal Phase 1 baselines (carried forward for context)

See [phase1_results.md](phase1_results.md).

| Model | Best AUC (OOD Tromsø) | Params |
|---|---|---|
| D4-BT (bi-temporal equivariant) | 0.912 @ 50% data | ~391K |
| CNN-BT (bi-temporal plain CNN) | 0.789 @ 50% data | ~391K |
| D4 single-image | 0.814 @ 100% data | ~391K |
| ResNet-18 | 0.823 @ 100% data | 11.2M |

Phase 1 numbers are patch-level AUC, not directly comparable to Phase 2 pixel-level F1/F2.

---

## Other relevant papers

### Bianchi & Grahn 2025 — arXiv:2502.18157
*Monitoring Snow Avalanches from SAR Data with Deep Learning.* Reports FPN+Xception as best of 10+ tested architectures. Key finding: rotation TTA at inference improves all non-equivariant models. Reference for input representation conventions (VH/VV, dB clip [−25, −5], DEM-derived slope/aspect).

### Weiler & Cesa 2019 — arXiv:1911.08251
*General E(2)-Equivariant Steerable CNNs.* NeurIPS 2019. Theoretical foundation for D4, C8, SO(2) groups via escnn library.

### Cesa et al. 2022 — ICLR 2022
*A Program to Build E(N)-Equivariant Steerable CNNs.* The escnn library.

### Han et al. 2021 — arXiv:2103.07733
*ReDet: A Rotation-Equivariant Detector for Aerial Object Detection.* CVPR 2021. Prior work applying equivariant CNNs to aerial/satellite imagery.

---

## Audit trail (2026-04-16)

Corrections to prior wiki content:

| Claim in prior wiki | Reality (verified from PDF) |
|---|---|
| "Swin-UNet ~2.39M params" | ✓ Correct (Table 5) |
| "8 input channels" | ✓ Correct (SAR + LIA + slope) |
| "No per-D-scale breakdown in paper" | ✗ WRONG. Table 9 has per-D-scale hit rates for D2/D3/D4. |
| "Polygon hit rate = fraction of polygons with ≥1 predicted pos pixel" | ✗ WRONG. Gatti uses ≥50% area (Table 7 caption). |
| "Morphological closing kernel=3, iter=1" | ✓ Correct |
| "BCE with positive weight 3.0" | ✓ Correct |
| "Focal Tversky α=0.7, β=0.3, γ=1.33" | ✗ UNVERIFIED in this read. Paper doesn't mention Focal Tversky as a variant they tested. |
| "112 vs 117 polygon count reason: D1 excluded" | ✓ Correct, confirmed in Section 5 text |
| "Parameter ratio our 625K vs Gatti 2.39M → ~4×" | ✓ Correct |

Lessons for future:
- Always extract full tables from baseline papers directly, not summaries.
- Metric definitions (especially polygon-level) must be verified character-by-character before cross-paper comparison.
- Ablation tables (Section 5.2 in Gatti) often contain crucial negative findings that change what counts as the "main" comparison point.
