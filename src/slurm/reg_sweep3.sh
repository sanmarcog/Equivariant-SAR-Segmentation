#!/bin/bash
#SBATCH --job-name=sar-reg-sweep3
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
# Sweep 3: bigger model, better loss, warm restarts.
# Best so far: aug3x → test F1=0.757 (baseline 0.749, Gatti 0.806).

set -uo pipefail
REPO=${REPO:?Set REPO to repo root}
DATA_DIR=${DATA_DIR:?Set DATA_DIR}
STATS=$REPO/data/norm_stats_12ch.json
SIF=${SIF:?Set SIF to container path}
SEED=0

run_eval() {
    local TAG=$1 CKPT=$2
    local EVAL_DIR=$REPO/results_reg3/${TAG}
    mkdir -p "$EVAL_DIR"
    local EVAL_F1=$EVAL_DIR/eval_gaussian_f1opt.json
    if [[ ! -f "$EVAL_F1" ]]; then
        echo "=== EVAL [$TAG] ==="
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
    echo "[$TAG] test F1:"
    grep '"best_f1"' "$EVAL_F1" 2>/dev/null | head -1
}

COMMON="--data-dir $DATA_DIR --stats $STATS --seed $SEED --pos-frac 0.5 --pos-weight 3.0 --patch-size 128 --train-stride 64 --online-aug --epochs 110 --warmup-epochs 10 --batch-size 32 --num-workers 8 --no-wandb"

# ── G: Bigger model (1.4M params) + aug3x ─────────────────────────────────
TAG=big_aug3x
OUT=$REPO/checkpoints_reg3/$TAG
CKPT=$OUT/best_cond2_seed${SEED}.pt
mkdir -p "$OUT"
if [[ ! -f "$CKPT" ]]; then
    echo "=== TRAIN [$TAG] ==="
    apptainer exec --nv --bind "${BIND_ROOT:?Set BIND_ROOT}" "$SIF" /bin/bash -c "
        source $REPO/.venv/bin/activate && cd $REPO && \
        python -m src.train $COMMON \
            --out-dir $OUT --condition 2 \
            --lr 1e-4 --wd 1e-4 --aug-strength 3.0 --patience 30 \
            --n-reg 12,24,48,48,48
    "
fi
run_eval "$TAG" "$CKPT"

# ── H: Focal+Tversky loss (condition 5) + aug3x ──────────────────────────
TAG=cond5_aug3x
OUT=$REPO/checkpoints_reg3/$TAG
CKPT=$OUT/best_cond5_seed${SEED}.pt
mkdir -p "$OUT"
if [[ ! -f "$CKPT" ]]; then
    echo "=== TRAIN [$TAG] ==="
    apptainer exec --nv --bind "${BIND_ROOT:?Set BIND_ROOT}" "$SIF" /bin/bash -c "
        source $REPO/.venv/bin/activate && cd $REPO && \
        python -m src.train $COMMON \
            --out-dir $OUT --condition 5 \
            --gamma 2.0 --alpha 0.3 --beta 0.7 \
            --lr 1e-4 --wd 1e-4 --aug-strength 3.0 --patience 30
    "
fi
run_eval "$TAG" "$CKPT"

# ── I: aug3x + 3 warm restarts ───────────────────────────────────────────
TAG=aug3x_wr3
OUT=$REPO/checkpoints_reg3/$TAG
CKPT=$OUT/best_cond2_seed${SEED}.pt
mkdir -p "$OUT"
if [[ ! -f "$CKPT" ]]; then
    echo "=== TRAIN [$TAG] ==="
    apptainer exec --nv --bind "${BIND_ROOT:?Set BIND_ROOT}" "$SIF" /bin/bash -c "
        source $REPO/.venv/bin/activate && cd $REPO && \
        python -m src.train $COMMON \
            --out-dir $OUT --condition 2 \
            --lr 1e-4 --wd 1e-4 --aug-strength 3.0 --patience 999 \
            --warm-restarts 3
    "
fi
run_eval "$TAG" "$CKPT"

# ── J: Bigger model + focal+tversky + aug3x ──────────────────────────────
TAG=big_cond5_aug3x
OUT=$REPO/checkpoints_reg3/$TAG
CKPT=$OUT/best_cond5_seed${SEED}.pt
mkdir -p "$OUT"
if [[ ! -f "$CKPT" ]]; then
    echo "=== TRAIN [$TAG] ==="
    apptainer exec --nv --bind "${BIND_ROOT:?Set BIND_ROOT}" "$SIF" /bin/bash -c "
        source $REPO/.venv/bin/activate && cd $REPO && \
        python -m src.train $COMMON \
            --out-dir $OUT --condition 5 \
            --gamma 2.0 --alpha 0.3 --beta 0.7 \
            --lr 1e-4 --wd 1e-4 --aug-strength 3.0 --patience 30 \
            --n-reg 12,24,48,48,48
    "
fi
run_eval "$TAG" "$CKPT"

echo "=== ALL CONFIGS DONE ==="
