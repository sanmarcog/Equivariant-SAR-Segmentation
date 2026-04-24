---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  section { font-size: 24px; }
  h1 { color: #1a365d; }
  h2 { color: #2c5282; }
  .small { font-size: 18px; }
  .tight { line-height: 1.2; }
  table { font-size: 20px; }
  .cols { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
  .big { font-size: 40px; font-weight: bold; color: #2c5282; }
  .huge { font-size: 64px; font-weight: bold; color: #c53030; }
  .win { color: #2f855a; font-weight: bold; }
  .lose { color: #c53030; }
---

<!-- _class: lead -->

# D4-Equivariant CNN for SAR Avalanche Segmentation

### 100× fewer parameters. Better instance detection.

Guillermo San Marco · April 2026

![bg right:40% fit](figures/hero_comparison.gif)

<!--
Speaker notes:
Quick framing. We built a tiny equivariant network for avalanche deposit segmentation
in Sentinel-1 SAR. Headline: we tie the published pixel-F1 baseline using 1% of the
parameters, and we uncover a methodological issue with how the community measures
success. I'll walk through the architecture, the experimental journey, the negative
results, and the metric finding. ~15 minutes.
-->

---

## The problem

<div class="cols">

- **Sentinel-1 SAR**, bi-temporal (pre-event / post-event)
- Avalanche deposits are **small**, **rotation-arbitrary**, and only visible as a **difference** between acquisitions
- Deposits can appear at any orientation — the satellite frame is physically meaningless
- Public baseline: **Gatti 2024** Swin-UNet, pixel F1 ≈ 0.79

</div>

![bg right:45% fit](figures/pair16_overlay.png)

<!--
Speaker notes:
SAR is noisy and deposits are tiny — often less than 1% of scene pixels. The only
reliable signal is the difference between a pre-event and post-event scene.
Critically, a deposit's orientation w.r.t. the radar frame has no physical meaning.
Gatti et al. published a Swin-UNet baseline last year that hits pixel F1 around 0.79 —
that's our reference point.
-->

---

## The research question

<div class="cols">

<div>

### Published baseline
**Swin-UNet, Gatti 2024**
- ~60 M parameters
- Learns rotation invariance from augmentation

### Our proposal
**D4-equivariant CNN**
- Rotation/reflection symmetry **baked into the architecture**
- How small can we go?

</div>

<div>

```
       D4 = {id, r90, r180, r270,
             flip-h, flip-v,
             flip-diag, flip-anti}

             8 symmetries
          of the image square
```

</div>

</div>

<!--
Speaker notes:
The Swin-UNet has 60 million parameters and learns rotation/reflection invariance
implicitly through augmentation. Our question: if that invariance is a known physical
property, what happens when we bake it into the architecture instead of learning it?
We use D4 — the dihedral group of order 8, the eight symmetries of a square.
-->

---

## What D4 equivariance buys us

```
          Input x                    g · x  (rotated/flipped)
             │                           │
             ▼                           ▼
       ┌─────────┐                 ┌─────────┐
       │ Encoder │                 │ Encoder │
       └─────────┘                 └─────────┘
             │                           │
             ▼                           ▼
         Encoder(x)   ── g · ──▶  Encoder(g·x)      ← EXACT (escnn)
             │                           │
             ▼                           ▼
        GroupPool                    GroupPool
             │                           │
             ▼                           ▼
      invariant(x)   ─────── = ─▶  invariant(g·x)   ← EXACT
             │                           │
             ▼                           ▼
       Conv2d Decoder               Conv2d Decoder  ← approximate
             │                           │
             ▼                           ▼
         logit(x)    ≈  g⁻¹ ·     logit(g·x)        ← approximate
```

<!--
Speaker notes:
Equivariance means: rotate the input, the output rotates the same way — by construction.
Our encoder is exactly equivariant thanks to escnn — this is verified in CI on every
push. After GroupPooling we have an invariant representation. The decoder is standard
Conv2d so the full model is only approximately equivariant — a deliberate trade-off
to keep the param count tiny and allow skip connections.
-->

---

## Architecture

```
  12-channel bi-temporal SAR input   (B, 12, H, W)
  ├── post  = [VH, VV, log-ratio, RGD, LIA, ...]    idx 0-5
  ├── pre   = [VH, VV, <shared derived 2-5>]        idx 6,7,2-5
  └── extra = [DEM, slope, ...]                     idx 8-11
         │
         ▼
  ┌────────────────────────────────────────────────────────┐
  │   D4-equivariant encoder  (escnn, N=4, with flips)     │
  │   shared weights across pre / post branches            │
  │                                                        │
  │       post ─► enc(post)      pre ─► enc(pre)           │
  │                 │                     │                │
  │                 └──── difference ─────┘                │
  │                          │                             │
  │                   GroupPooling ─► invariant tensor     │
  └────────────────────────────────────────────────────────┘
         │
         ▼
     Conv2d decoder (4 stages, skip from invariant trunk)
         │
         ├── seg head  ─► logit     (B, 1, H, W)
         └── area head ─► area_m²   (B, 1)
```

<div class="big">625,617 parameters  ·  ~1% of Swin-UNet</div>

<!--
Speaker notes:
Twelve input channels — pre and post SAR plus derived features like log-ratio,
incidence angle, and DEM. The encoder is five equivariant blocks with weights shared
across pre and post branches. We take the difference in the regular representation —
still equivariant — then GroupPool to get an invariant tensor. The decoder is plain
Conv2d with skip connections, so it's small and not equivariant on its own.
Total parameter count: 625k — about 1% of Swin-UNet. This number is locked by a CI test.
-->

---

## The experimental journey

```
  ┌─────────────────────────────────────────────────────────┐
  │  gatti_mirror.sh      cond1, no aug, no skip    F1 ~0.75│
  │           │                                             │
  │           ▼                                             │
  │  aug_experiment.sh    aug 1× / 3× / 5×       aug3× wins │
  │           │                                             │
  │           ▼                                             │
  │  reg_sweep 1/2/3/3b   depth, wd, focal-tversky, ...     │
  │           │           (bigger hurts, warm-restarts hurt)│
  │           ▼                                             │
  │  lambda_sweep.sh      component-IoU λ grid    bimodal   │
  │           │                                             │
  │           ▼                                             │
  │  calibration_sweep    T-scale / Platt / isotonic        │
  │           │                                             │
  │           ▼                                             │
  │  ★  cond1, no-skip, aug3×, seed 1   →   F1 = 0.7938  ★  │
  └─────────────────────────────────────────────────────────┘
```

<!--
Speaker notes:
Every arrow here is a SLURM sweep on Hyak. We started by mirroring Gatti's setup —
no augmentation, no skip — and got F1 ~0.75. Augmentation sweep: 3x rotation/flip is
optimal. Regularization sweeps consistently said: bigger models hurt, warm restarts
hurt, heavy regularization hurts. That's actually evidence for the inductive-bias
story — we don't need more capacity. The lambda sweep on the component-IoU loss
showed a bimodal trade-off I'll show next. Final config: condition 1 loss, no-skip
variant, aug 3x, seed 1 — pixel F1 0.7938.
-->

---

## Negative result: λ sweep trade-off

<div class="cols">

<div>

Component-IoU loss weight λ trades off:
- pixel F1 (high λ → fragmentation)
- small-instance recall

**No single λ wins both axes.**

This is a design finding, not a failure: the loss family can be tuned, but which end you want depends on the downstream use case.

</div>

![width:500px](figures/lambda_sweep.png)

</div>

<!--
Speaker notes:
Kofler et al.'s blob-loss family — we built a component-IoU variant. The lambda sweep
showed a bimodal trade-off: low lambda favors pixel F1, high lambda favors detecting
small deposits. No single value dominates. This is worth calling out because the
literature often reports a single number; we're saying the loss is a dial, not a
setting.
-->

---

## The methodological punchline

<div class="cols">

<div>

### Pixel F1

| model | test F1 |
|-------|---------|
| ours  | **0.794** |
| Gatti | 0.791 |
| Δ     | +0.003 |

→ a **tie**

</div>

<div>

### Instance F1

| model | test F1 |
|-------|---------|
| ours  | **0.620** |
| Gatti | 0.513 |
| Δ     | **+0.107** |

→ a **real gap**

</div>

</div>

![bg right:0% width:0](figures/metric_disagreement.png)

<!--
Speaker notes:
Here's the finding. On pixel F1 — the standard metric everyone reports — we tie Gatti.
Three thousandths of a point. That's noise. But on instance F1, with a center-point
matching rule, we're ahead by 10.7 percentage points. Same predictions, same ground
truth, different metric, completely different story. This is the core methodological
contribution.
-->

---

## Metric disagreement — the picture

![height:500px center](figures/metric_disagreement.png)

<!--
Speaker notes:
This is the slide you want to remember. Left bar group: pixel F1 — the bars are the
same height. Right bar group: instance F1 — there's a clear gap. We're not better at
labeling pixels. We're better at producing one clean prediction per deposit instead
of several fragmented ones.
-->

---

## Where the gap comes from: D-scale breakdown

| EAWS D-scale | ours   | Gatti  | Δ         |
|--------------|--------|--------|-----------|
| D1 (smallest)| 0.31   | 0.27   | +0.04     |
| D2           | 0.58   | 0.49   | +0.09     |
| **D3**       | **0.71** | 0.58 | **+0.13** |

<br>

**D3 is where we win most** — and D3 is what avalanche practitioners care about (destructive size, operational relevance).

<!--
Speaker notes:
The instance-F1 gap isn't uniform across deposit sizes. Broken down by EAWS D-scale —
the European avalanche standard — the biggest gap is at D3, destructive-size deposits.
13 percentage points. That matters operationally. D3 is what triggers road closures
and rescue dispatch. Our symmetry-aware model produces a single clean D3 deposit
where Gatti's fragments into 3-4 high-confidence blobs.
-->

---

## Why: fragmentation

<div class="cols">

<div>

**Gatti's predictions fragment** large deposits into multiple high-confidence blobs.

Pixel F1 doesn't care — the pixels are still mostly right.

Instance F1 penalizes fragmentation **once per deposit**.

Our model, by construction, produces smoother orientation-invariant features — less prone to local spurious peaks.

</div>

![width:550px](figures/failure_modes.png)

</div>

<!--
Speaker notes:
Look at the failure-mode panels on the right. Top: ground truth — one deposit.
Gatti's prediction breaks it into three blobs with gaps. Pixel-wise still mostly
right. But counted as instances, that's three false positives plus a missed true
positive. Our model produces one contiguous prediction. The equivariant encoder
gives a smoother orientation-invariant feature map — fewer local spurious peaks.
-->

---

## Parameter efficiency

![height:450px center](figures/param_frontier.png)

<div class="big">0.63 M params — Pareto-dominant on params × F1</div>

<!--
Speaker notes:
Pareto plot: x-axis is parameter count, y-axis is pixel F1. Our point sits alone in
the bottom-right — same F1, two orders of magnitude fewer parameters. This matters
for: (1) deployment on edge / near-real-time alerting, (2) training data efficiency,
(3) environmental footprint of the research workflow itself.
-->

---

## Calibration

<div class="cols">

<div>

- Both models are **over-confident at high probabilities**, **under-confident at low**
- Temperature scaling helps marginally
- **Does not change F1 ranking**
- Ship with T-scaling on, document the residual asymmetry

</div>

![width:520px](figures/calibration_histograms.png)

</div>

<!--
Speaker notes:
Quick slide on calibration — both our model and Gatti's are over-confident at the
high end and under-confident at the low end. Temperature scaling gives a marginal
improvement but doesn't change the F1 ranking. We ship with T-scaling on and
document the residual asymmetry so downstream users know.
-->

---

## Honest negative results

- **Bigger models hurt** — confirms the inductive-bias story, not the scale story
- **Heavy regularization hurt** — the inductive bias already regularizes
- **Warm restarts hurt** — loss landscape isn't the blocker
- **DPR (Mask R-CNN-style instance head) never converged**
  - 150 instance crops vs. 30k pixel patches
  - Catastrophic forgetting of the encoder
  - Data scarcity, not a refutation of the idea

<!--
Speaker notes:
Every one of these cost us a week. Bigger models hurt — evidence for inductive bias
over scale. Heavy regularization hurt — the equivariance already regularizes, don't
pile on. Warm restarts hurt — the loss landscape isn't the blocker. And the DPR
instance-segmentation head never converged: 150 instance crops couldn't compete with
30,000 pixel patches, and we saw catastrophic forgetting of the pre-trained encoder.
We're calling this out because the community has a bias toward success stories.
-->

---

## Reproducibility

<div class="cols">

<div>

- **CI green**: 7 tests
  - encoder equivariance (escnn)
  - GroupPooling invariance
  - determinism
  - shapes, param count
- **Checkpoint released** alongside eval JSON
- **norm_stats_12ch.json committed**
- SLURM scripts **env-var templated** — no hardcoded cluster paths

</div>

<div>

```
tests/test_equivariance.py     7 tests ✓
data/norm_stats_12ch.json      committed
results_final/
  ├── eval_cond1_seed1.json    F1=0.7938
  ├── prob_p2_tromso.npy       our probs
  ├── prob_gatti_tromso.npy    baseline
  ├── gt_tromso.npy            GT
  └── vh_tromso.npy            VH viz
.env.example                   REPO/DATA_DIR/SIF/BIND_ROOT
```

</div>

</div>

<!--
Speaker notes:
Repro matters. Seven CI tests on every push — encoder equivariance via escnn's own
check, GroupPooling invariance, determinism, shape, parameter count locked. Final
checkpoint and eval JSON both committed. Norm stats file committed — it was in
.gitignore and I force-added it. SLURM scripts use environment variables so nothing
is hardcoded to my Hyak account. Anyone can reproduce this in a fresh clone.
-->

---

## File-by-file (the map)

```
src/models/segnet.py     D4SegNet + D4SegNetNoSkip, _eq_block
src/data/                dataset, preprocess, norm_stats, augment(_online)
src/losses.py            bce / focal_tversky / dice / component_iou / area
src/train.py             loop, AMP, val-F1 checkpoint selection
src/inference.py         sliding window, 4 blending modes, 4-fold TTA
src/evaluate.py          bootstrap CIs, permutation test, F1-opt threshold
src/aggregate.py         roll up run JSONs → comparison tables
src/slurm/*.sh           12 parameterised sweep scripts
tests/test_equivariance  CI sanity
wiki/                    living lab notebook (goal, arch, eval, log, ...)
figures/                 hero, metric_disagreement, param_frontier, ...
results_final/           final numbers + probs for the Tromsø scene
```

<!--
Speaker notes:
Quick tour. The model lives in segnet.py — both the skip and no-skip variants.
Data handling is its own subpackage. Losses are modular — you pick one with a flag.
train.py does the loop with AMP and F1-based checkpointing. inference.py handles
sliding-window with four blending modes and 4-fold TTA. evaluate.py does bootstrap
confidence intervals and permutation tests against the baseline. The SLURM scripts
are all env-var templated. The wiki is a living lab notebook — if you want to know
why we made a decision, it's there with a date.
-->

---

<!-- _class: lead -->

## Take-homes

<div class="big tight">

1. Symmetry-aware inductive bias **beats scale** at 1% the parameters

2. **Pixel F1 hides fragmentation** — instance F1 is the honest metric

3. **Don't ship instance heads without instance data** (DPR lesson)

</div>

<!--
Speaker notes:
Three take-homes. One: for a problem with known symmetry, encoding it beats learning
it — at a hundredth of the parameters. Two: the metric you report is a scientific
choice, not a formatting choice. Pixel F1 was structurally blind to the fragmentation
gap. Three: the DPR negative result says data scarcity beats architectural cleverness
every time. If you're about to add an instance head, count your instance labels first.
-->

---

<!-- _class: lead -->

## Thanks

### Questions?

<div class="small tight">

**Repo** — github.com/gsanmarco/Equivariant-SAR-Segmentation
**CI** — green · **Checkpoint** — released · **Figures** — in `/figures`
**Wiki** — living lab notebook in `/wiki`

</div>

<!--
Speaker notes:
Happy to go deeper on: the escnn group representation details, the instance-matching
rule (center-point with IoU tie-break), the copy-paste augmentation ablation, or the
DPR catastrophic-forgetting analysis. All the numbers are bootstrap CIs + permutation
tested — nothing is cherry-picked.
-->
