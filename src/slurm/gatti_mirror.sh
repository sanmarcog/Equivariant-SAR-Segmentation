#!/bin/bash
#SBATCH --job-name=sar-gatti-mirror
#SBATCH --account=demo
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --exclude=z3005,z3006
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Gatti-mirror experiment: match every Gatti training/inference detail, swap only the architecture.
#   - Patch 128×128, stride 64 (50% overlap) for training AND inference
#   - Balanced sampler pos_frac=0.5
#   - Full off-D4 augmentation (affine + Gaussian noise + intensity)
#   - BCE with pos_weight=3.0
#   - 110 epochs FLAT (no early stopping — Gatti trains fixed schedule)
#   - AdamW LR=1e-4, 10-epoch warmup + 100-epoch cosine
#   - Morph closing at inference, no TTA
#   - Evaluated at F1-opt (Gaussian blending) and F2-opt (Max blending)
#   - 1 seed (matching Gatti)
#
# Our one deviation: architecture = D4-equivariant CNN (625K params) instead of SwinV2-Tiny (2.39M).

set -uo pipefail
REPO=${REPO:?Set REPO to repo root}
DATA_DIR=${DATA_DIR:?Set DATA_DIR}
STATS=$REPO/data/norm_stats_12ch.json
SIF=${SIF:?Set SIF to container path}
SEED=${SEED:-0}
OUT_DIR=$REPO/checkpoints_gatti_mirror_full
EVAL_DIR=$REPO/results_gatti_mirror_full
mkdir -p "$OUT_DIR" "$EVAL_DIR"

CKPT=$OUT_DIR/best_cond2_seed${SEED}.pt

# ── 1. TRAIN ──────────────────────────────────────────────────────────────────
if [[ -f "$CKPT" ]]; then
    echo "Checkpoint exists ($CKPT) — skipping training"
else
    echo "=== TRAIN (Gatti-mirror, seed=$SEED) ==="
    apptainer exec --nv --bind "${BIND_ROOT:?Set BIND_ROOT}" "$SIF" /bin/bash -c "
        source $REPO/.venv/bin/activate
        cd $REPO
        python -m src.train \
            --data-dir  $DATA_DIR \
            --stats     $STATS \
            --out-dir   $OUT_DIR \
            --condition 2 \
            --seed      $SEED \
            --pos-frac  0.5 \
            --pos-weight 3.0 \
            --patch-size 128 \
            --train-stride 64 \
            --online-aug \
            --epochs 110 \
            --patience 999 \
            --warmup-epochs 10 \
            --batch-size 32 \
            --lr 1e-4 \
            --wd 1e-4 \
            --num-workers 8 \
            --no-wandb
    "
fi

# ── 2. EVAL at F1-opt operating point (Gaussian blending, Gatti's F1-opt best) ─
EVAL_F1=$EVAL_DIR/eval_seed${SEED}_gaussian_f1opt.json
if [[ ! -f "$EVAL_F1" ]]; then
    echo "=== EVAL F1-opt (Gaussian blending) ==="
    apptainer exec --nv --bind "${BIND_ROOT:?Set BIND_ROOT}" "$SIF" /bin/bash -c "
        source $REPO/.venv/bin/activate
        cd $REPO
        python -m src.evaluate \
            --ckpt     $CKPT \
            --data-dir $DATA_DIR \
            --stats    $STATS \
            --split    test \
            --out      $EVAL_F1 \
            --no-tta \
            --patch-size 128 \
            --stride 64 \
            --blending gaussian \
            --morph-closing \
            --multi-threshold \
            --frozen-thresholds 0.3 0.5 0.7 \
            --n-bootstrap 1000 \
            --n-perm      1000
    "
fi

# ── 3. EVAL at F2-opt operating point (Max blending, Gatti's F2-opt best) ────
EVAL_F2=$EVAL_DIR/eval_seed${SEED}_max_f2opt.json
if [[ ! -f "$EVAL_F2" ]]; then
    echo "=== EVAL F2-opt (Max blending) ==="
    apptainer exec --nv --bind "${BIND_ROOT:?Set BIND_ROOT}" "$SIF" /bin/bash -c "
        source $REPO/.venv/bin/activate
        cd $REPO
        python -m src.evaluate \
            --ckpt     $CKPT \
            --data-dir $DATA_DIR \
            --stats    $STATS \
            --split    test \
            --out      $EVAL_F2 \
            --no-tta \
            --patch-size 128 \
            --stride 64 \
            --blending max \
            --morph-closing \
            --multi-threshold \
            --frozen-thresholds 0.3 0.5 0.7 \
            --n-bootstrap 1000 \
            --n-perm      1000
    "
fi

echo "DONE. Results in $EVAL_DIR"
