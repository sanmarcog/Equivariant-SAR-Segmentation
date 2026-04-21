#!/bin/bash
#SBATCH --job-name=sar-calib
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
# Calibration sweep: fix under-confident predictions (thr=0.22).
# Hypothesis: double class-balance (sampler + pos_weight=3) makes model diffuse.
# Try: pos_weight=1, Dice loss, BCE+Dice. All with aug5x.

set -uo pipefail
REPO=${REPO:?Set REPO to repo root}
DATA_DIR=${DATA_DIR:?Set DATA_DIR}
STATS=$REPO/data/norm_stats_12ch.json
SIF=${SIF:?Set SIF to container path}
SEED=0

run_eval() {
    local TAG=$1 CKPT=$2
    local EVAL_DIR=$REPO/results_calib/${TAG}
    mkdir -p "$EVAL_DIR"
    local EVAL_F1=$EVAL_DIR/eval_s16_tta.json
    if [[ ! -f "$EVAL_F1" ]]; then
        echo "=== EVAL [$TAG] stride16+TTA ==="
        apptainer exec --nv --bind "${BIND_ROOT:?Set BIND_ROOT}" "$SIF" /bin/bash -c "
            source $REPO/.venv/bin/activate && cd $REPO && \
            python -m src.evaluate \
                --ckpt $CKPT --data-dir $DATA_DIR --stats $STATS \
                --split test --out $EVAL_F1 \
                --patch-size 128 --stride 16 --blending gaussian \
                --morph-closing --multi-threshold --frozen-thresholds 0.3 0.5 0.7 \
                --n-bootstrap 1000 --n-perm 1000
        "
    fi
    echo "[$TAG] results:"
    grep '"best_f1"\|"thr_f1"' $EVAL_F1 2>/dev/null | head -2
}

COMMON="--data-dir $DATA_DIR --stats $STATS --seed $SEED --pos-frac 0.5 --patch-size 128 --train-stride 64 --online-aug --aug-strength 5.0 --epochs 110 --warmup-epochs 10 --batch-size 32 --lr 1e-4 --wd 1e-4 --num-workers 8 --no-wandb --patience 30"

# ── N: BCE with pos_weight=1.0 (remove double class-balance correction) ──
TAG=bce_pw1
OUT=$REPO/checkpoints_calib/$TAG
CKPT=$OUT/best_cond2_seed${SEED}.pt
mkdir -p "$OUT"
if [[ ! -f "$CKPT" ]]; then
    echo "=== TRAIN [$TAG] ==="
    apptainer exec --nv --bind "${BIND_ROOT:?Set BIND_ROOT}" "$SIF" /bin/bash -c "
        source $REPO/.venv/bin/activate && cd $REPO && \
        python -m src.train $COMMON \
            --out-dir $OUT --condition 2 --pos-weight 1.0
    "
fi
run_eval "$TAG" "$CKPT"

# ── O: Dice loss (directly optimizes F1) ──────────────────────────────────
TAG=dice
OUT=$REPO/checkpoints_calib/$TAG
CKPT=$OUT/best_cond2_seed${SEED}.pt
mkdir -p "$OUT"
if [[ ! -f "$CKPT" ]]; then
    echo "=== TRAIN [$TAG] ==="
    apptainer exec --nv --bind "${BIND_ROOT:?Set BIND_ROOT}" "$SIF" /bin/bash -c "
        source $REPO/.venv/bin/activate && cd $REPO && \
        python -m src.train $COMMON \
            --out-dir $OUT --condition 2 --loss-mode dice --pos-weight 1.0
    "
fi
run_eval "$TAG" "$CKPT"

# ── P: BCE + Dice combined (calibrated + sharp) ──────────────────────────
TAG=bce_dice
OUT=$REPO/checkpoints_calib/$TAG
CKPT=$OUT/best_cond2_seed${SEED}.pt
mkdir -p "$OUT"
if [[ ! -f "$CKPT" ]]; then
    echo "=== TRAIN [$TAG] ==="
    apptainer exec --nv --bind "${BIND_ROOT:?Set BIND_ROOT}" "$SIF" /bin/bash -c "
        source $REPO/.venv/bin/activate && cd $REPO && \
        python -m src.train $COMMON \
            --out-dir $OUT --condition 2 --loss-mode bce_dice --pos-weight 1.0
    "
fi
run_eval "$TAG" "$CKPT"

echo "=== ALL DONE ==="
