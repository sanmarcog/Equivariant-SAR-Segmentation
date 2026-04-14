# Baselines — Phase 2

**Primary comparison: Gatti et al. 2026 (arXiv:2603.22658).**

---

## Gatti et al. 2026

> ✓ DECIDED: citation corrected to "Gatti et al. 2026" — first author is Mattia Gatti (University of Insubria). Previous wiki sessions used "Gattimgatti" in error.

| Property | Value |
|----------|-------|
| arXiv | 2603.22658 |
| Title | Large-Scale Avalanche Mapping from SAR Images with Deep Learning-based Change Detection |
| Authors | Mattia Gatti, Alberto Mariani, Ignazio Gallo, Fabiano Monti |
| Repo | [github.com/mattiagatti/avalanche-deep-change-detection](https://github.com/mattiagatti/avalanche-deep-change-detection) |
| Task | Pixel-level SAR avalanche segmentation via bi-temporal change detection |
| Dataset | AvalCD (same as ours — Zenodo doi:10.5281/zenodo.15863589) |
| Geographic split | Same — Tromsø held out as OOD test |
| Test polygons | 112 (vs our 117 — see note below) |

### Architecture

| Component | Details |
|-----------|---------|
| Encoder | **SwinV2-Tiny** (Swin Transformer V2) — vision transformer, NOT a CNN |
| Channel progression | Stage 0: 96ch → Stage 1: 192ch → Stage 2: 384ch → Stage 3: 768ch |
| Decoder | Custom `TransformerDecoder` with `BasicLayerUp` (SwinTransformerV2 blocks) + `FinalPatchExpandXN` |
| Fusion | **Difference-based** (default): concatenate `feat_post − feat_pre` with aux → Conv→BN→ReLU→Conv→BN |
| Other fusion tested | AGMF (attention gating), Cross-Attention (windowed multi-head) |
| Parameters | ~2.39M |
| Input channels | 8: pre (VH, VV), post (VH, VV), aux (DEM, slope, aspect, LIA) |
| Patch size | 128×128 |

> Note: SwinV2-Tiny is a vision transformer. Our comparison framing is **equivariant CNN vs vision transformer**, not equivariant vs standard CNN.

### Training

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Scheduler | Linear warmup (10 epochs, start_factor=0.1) → Cosine annealing (100 epochs) |
| Total epochs | 110 |
| Batch size | 32 |
| Early stopping | Patience 20 (warmup-aware) |
| Model selection | AUPRC on validation set |
| Loss (default) | BCE with positive weight 3.0 |
| Loss (also tested) | BCE+Dice (equal weight), Focal Tversky (α=0.7, β=0.3, γ=1.33) |

### Inference

| Property | Value |
|----------|-------|
| Patch size | 128×128 |
| Stride | 64 (50% overlap) |
| Stitching | Blending (averaging in overlap regions) |
| Post-processing | Morphological closing (kernel=3, iterations=1) |
| TTA | Not evident in code |

### Metrics (pixel-level)

> ✓ DECIDED: Gatti reports **pixel-level** F1/F2, NOT polygon-level IoU-based F1/F2. Their `test.py` uses `torchmetrics` BinaryF1Score, BinaryPrecision, BinaryRecall, BinaryJaccardIndex at the pixel level. No polygon matching code exists in the repo.

**Two operating points** on the same trained model, using different probability thresholds:

| Config | Threshold | F1 | F2 | Polygon hit rate |
|--------|-----------|-----|-----|------------------|
| F1-optimized (conservative) | Selected to max F1 on val | **0.8061** | — | — |
| F2-optimized (recall-oriented) | Selected to max F2 on val | — | **0.8414** | 80.36% |

The threshold is selected by sweeping all unique predicted probabilities on the validation set and choosing the one that maximizes the target metric (F1 or F2).

Polygon hit rate = fraction of GT polygons that contain ≥1 predicted positive pixel. Separate metric, not derived from F1/F2.

### Baselines tested by Gatti et al.

8 change detection architectures compared: BIT, ChangeFormer, SiamUNet-conc, SiamUNet-diff, SNUNet, STANet, STNet, TinyCD. Swin-UNet is their proposed method.

### 112 vs 117 polygon count

> ⚠ OPEN: our Tromsø GT has 117 polygons (5×D1, 25×D2, 71×D3, 16×D4). Gatti reports 112. Likely explanation: 117 − 5 D1 = 112 (D1 excluded as below detection floor). Must verify from full paper text. If confirmed, consider whether we also exclude D1 from pixel-level F1/F2 or report them separately.

### Known differences to document

1. **112 vs 117 GT polygons** — likely D1 exclusion; see above
2. **128×128 vs 64×64 patch size** — we argue smaller patches give D2 detection advantage
3. **~2.39M vs ~500–600K parameters** — our primary efficiency claim (~4× fewer)
4. **Vision transformer vs equivariant CNN** — different architectural paradigm
5. **8 vs 12 input channels** — we add log-ratio, cross-pol ratio (4 engineered features they don't use)
6. **Pixel-level evaluation** — both report pixel-level F1/F2; we additionally report polygon-level

---

## Internal Phase 1 baselines (carried forward for context)

See [phase1_results.md](phase1_results.md) for full numbers.

| Model | Best AUC (OOD Tromsø) | Params |
|-------|----------------------|--------|
| D4-BT (bi-temporal equivariant) | 0.912 @ 50% data | ~391K |
| CNN-BT (bi-temporal plain CNN) | 0.789 @ 50% data | ~391K |
| D4 single-image | 0.814 @ 100% data | ~391K |
| ResNet-18 | 0.823 @ 100% data | 11.2M |

These are patch-level AUC numbers — not directly comparable to Phase 2 pixel-level F1/F2. The D4-BT backbone is the starting point for Phase 2.

---

## Other relevant papers

### Bianchi & Grahn 2025 — arXiv:2502.18157

*Monitoring Snow Avalanches from SAR Data with Deep Learning.*

| Property | Value |
|----------|-------|
| Task | SAR avalanche segmentation benchmark |
| Architectures tested | 10+, including FPN+Xception (best) |
| Key finding | Rotation TTA at inference improves all models |
| Relevance to Phase 2 | Defines input representation used here: VH/VV channels, dB clipping [−25, −5], DEM-derived slope/aspect. Equivariant architecture removes the need for TTA by construction. |

> ⚠ OPEN: check whether Bianchi & Grahn use the same AvalCD dataset or a different SAR corpus. If same, their FPN+Xception numbers are an additional comparison point.

---

### Bianchi et al. 2021 — arXiv:1910.05411

*Snow Avalanche Segmentation in SAR Images With FCNNs.* IEEE JSTARS.

| Property | Value |
|----------|-------|
| Task | SAR avalanche segmentation (earlier work) |
| Relevance to Phase 2 | Foundational paper for FCNNs on SAR avalanche data; establishes the pixel-level segmentation framing this project builds on. |

> ⚠ OPEN: not yet ingested. Read to check metrics, dataset, and architectural choices for related work section.

---

### Weiler & Cesa 2019 — arXiv:1911.08251

*General E(2)-Equivariant Steerable CNNs.* NeurIPS 2019.

| Property | Value |
|----------|-------|
| Relevance to Phase 2 | Theoretical foundation for D4, C8, SO(2) groups via escnn. |

---

### Cesa et al. 2022 — ICLR 2022

*A Program to Build E(N)-Equivariant Steerable CNNs.* ([escnn](https://github.com/QUVA-Lab/escnn))

| Property | Value |
|----------|-------|
| Relevance to Phase 2 | escnn library implements all equivariant layers including decoder R2Conv. |

---

### Han et al. 2021 — arXiv:2103.07733

*ReDet: A Rotation-Equivariant Detector for Aerial Object Detection.* CVPR 2021.

| Property | Value |
|----------|-------|
| Relevance to Phase 2 | Closest prior work applying equivariant CNNs to aerial/satellite imagery. Supports novelty claim. |

> ⚠ OPEN: audit flag noted thin related work coverage. Continue adding papers from ISPRS/TGRS/Remote Sensing as encountered during Phase 2.
