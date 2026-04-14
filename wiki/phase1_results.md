# Phase 1 Results — Carried Forward

**Summary: D4-BT dominates all single-image models at every data fraction and beats CNN-BT by +0.123 to +0.191 AUC. Both bi-temporal signal AND equivariant constraint are necessary.**

---

## OOD test set AUC-ROC (Tromsø_20241220)

| Model | 10% | 25% | 50% | 100% |
|-------|-----|-----|-----|------|
| D4-BT (bi-temporal equivariant) | **0.871** | **0.906** | **0.912** | **0.894** |
| CNN-BT (bi-temporal plain CNN) | 0.683 | 0.724 | 0.789 | 0.703 |
| ResNet-18 | 0.555 | 0.786 | 0.743 | 0.803 |
| D4 (single-image equivariant) | 0.717 | 0.789 | 0.778 | 0.769 |
| C8 (single-image equivariant) | 0.675 | 0.676 | 0.745 | 0.737 |
| SO(2) (single-image equivariant) | 0.645 | 0.660 | 0.672 | 0.724 |
| CNN baseline (no aug) | 0.499 | 0.677 | 0.783 | 0.723 |
| CNN + rotation augmentation | 0.523 | 0.622 | 0.744 | 0.705 |

*Single seed. Metric: AUC-ROC.*

---

## Key findings carried into Phase 2

1. **D4-BT backbone is the starting point.** Best model at every fraction; 10% data already reaches 0.871.
2. **Bi-temporal change signal is the larger contributor.** D4-BT vs D4 single-image: +0.154 / +0.117 / +0.134 / +0.125 AUC. Signal from change detection dominates equivariance.
3. **Equivariance is an independent contribution.** D4-BT vs CNN-BT: +0.188 / +0.182 / +0.123 / +0.191 AUC. Equivariant constraint provides regularization beyond what Dropout+BN achieves.
4. **Data efficiency.** D4-BT at 10% (0.871) already exceeds ResNet-18 at 100% (0.803).
5. **Logit collapse.** Temperature scaling fit T≈50 — indicates near-certain logit outputs. Fix in Phase 2: label smoothing ε=0.05.
6. **Single seed.** All Phase 1 numbers are single-seed. Error bars unknown; 3-seed training is a Phase 2 requirement for all conditions.

---

## Known limitations Phase 2 must address

- AUC/patch hit-rate metrics are not comparable to Gatti et al.'s pixel-level F1/F2
- No pixel-level output — cannot evaluate deposit geometry
- Logit collapse (T≈50) degrades calibration
- Single seed — no variance estimates

---

## Phase 1 repo

`github.com/sanmarcog/Equivariant-CNN-SAR` — branch `main`, last commit `7a50f79`.
