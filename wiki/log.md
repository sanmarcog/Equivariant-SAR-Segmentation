# Activity Log

Append-only. Format: `TYPE | YYYY-MM-DD | description | pages touched`

---

INGEST | 2026-04-14 | Phase 2 scope defined by user (sanmarco) at session start | goal.md, architecture.md, datasets.md, evaluation.md, baselines.md, phase1_results.md, size_estimation.md, open_questions.md, out_of_scope.md — all seeded from scratch
INGEST | 2026-04-14 | Phase 1 README references extracted | baselines.md — added Bianchi & Grahn 2025 (arXiv:2502.18157), Bianchi et al. 2021 (arXiv:1910.05411), Weiler & Cesa 2019 (arXiv:1911.08251), Cesa et al. 2022 (escnn), Han et al. 2021 (arXiv:2103.07733)
LINT | 2026-04-14 | AUC table had wrong numbers (val AUC used instead of OOD test AUC) | phase1_results.md — corrected from Phase 1 README; added C8 and SO(2) rows; fixed ResNet-18 10% (0.555), D4, CNN baseline, CNN+aug; updated delta claims in findings
INGEST | 2026-04-14 | Small deposit detection analysis — 4 preprocessing improvements confirmed, 2 flagged for feasibility investigation | architecture.md — Refined Lee (not standard Lee), log-ratio change image, LIA normalization, VH/VV ratio channel added to plan; multi-temporal stacking and NL-SAR added as feasibility questions. datasets.md — input channels expanded from 7 to 12, norm stats invalidated, backbone channel mismatch flagged. open_questions.md — Q4 (multi-temporal stacking) and Q5 (NL-SAR) added.
INGEST | 2026-04-14 | Backbone retraining decision | architecture.md, datasets.md — retrain from scratch with 12-channel input; Phase 1 checkpoint not reused; freeze-vs-fine-tune question closed.
INGEST | 2026-04-14 | D2 detection improvements added; wiki cleaned | architecture.md — added U-Net skip connections (DECIDED), Focal+Tversky loss (DECIDED, replaces BCE OPEN), biased patch sampling, TTA, hyperparameter tuning protocol, weight decay. open_questions.md — closed Q5 (NL-SAR), removed stale freeze/fine-tune sub-question from Q1, updated parameter count to ~500–600K, cleaned unresolved decisions list.
INGEST | 2026-04-14 | Copy-paste augmentation, dropout, ablation plan added | architecture.md — copy-paste (within-region only, Gaussian blending, 20-30% cap), dropout 0.3 on bottleneck, 5-condition ablation table.
