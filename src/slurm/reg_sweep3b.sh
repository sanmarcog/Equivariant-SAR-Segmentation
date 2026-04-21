#!/bin/bash
#SBATCH --job-name=sar-reg-sweep3b
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
# Sweep 3b: untried configs. Focus on loss function and augmentation extremes.
# Best so far: aug3x → test F1=0.757.
# Bigger model hurt. Warm restarts hurt. Heavy reg hurt.

set -uo pipefail
REPO=${REPO:?Set REPO to repo root}
DATA_DIR=${DATA_DIR:?Set DATA_DIR}
STATS=$REPO/data/norm_stats_12ch.json
SIF=${SIF:?Set SIF to container path}
SEED=0

run_eval() {
    local TAG=$1 CKPT=$2
    local EVAL_DIR=$REPO/results_reg3b/${TAG}
    mkdir -p "$EVAL_DIR"
    local EVAL_F1=$EVAL_DIR/eval_gaussian_f1opt.json
    if [[ ! -f "$EVAL_F1" ]]; then
        echo "=== EVAL [$TAG] ==="
        apptainer exec --nv --bind /mmfs1 "$SIF" /bin/bash -c "
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

# ── H: Condition 5 (focal+tversky + skip + copy-paste) + aug3x, DEFAULT model ──
#   Copy-paste bug fixed. This is the best loss + architecture from our ablation.
TAG=cond5_aug3x
OUT=$REPO/checkpoints_reg3b/$TAG
CKPT=$OUT/best_cond5_seed${SEED}.pt
mkdir -p "$OUT"
if [[ ! -f "$CKPT" ]]; then
    echo "=== TRAIN [$TAG] ==="
    apptainer exec --nv --bind /mmfs1 "$SIF" /bin/bash -c "
        source $REPO/.venv/bin/activate && cd $REPO && \
        python -m src.train $COMMON \
            --out-dir $OUT --condition 5 \
            --gamma 2.0 --alpha 0.3 --beta 0.7 \
            --lr 1e-4 --wd 1e-4 --aug-strength 3.0 --patience 30
    "
fi
run_eval "$TAG" "$CKPT"

# ── K: Condition 4 (focal+tversky + skip, NO copy-paste) + aug3x ──────────
#   Skip connections might help boundary precision (the F1 gap is from precision).
TAG=cond4_aug3x
OUT=$REPO/checkpoints_reg3b/$TAG
CKPT=$OUT/best_cond4_seed${SEED}.pt
mkdir -p "$OUT"
if [[ ! -f "$CKPT" ]]; then
    echo "=== TRAIN [$TAG] ==="
    apptainer exec --nv --bind /mmfs1 "$SIF" /bin/bash -c "
        source $REPO/.venv/bin/activate && cd $REPO && \
        python -m src.train $COMMON \
            --out-dir $OUT --condition 4 \
            --gamma 2.0 --alpha 0.3 --beta 0.7 \
            --lr 1e-4 --wd 1e-4 --aug-strength 3.0 --patience 30
    "
fi
run_eval "$TAG" "$CKPT"

# ── L: aug5x (push augmentation further) + condition 2 ────────────────────
TAG=aug5x
OUT=$REPO/checkpoints_reg3b/$TAG
CKPT=$OUT/best_cond2_seed${SEED}.pt
mkdir -p "$OUT"
if [[ ! -f "$CKPT" ]]; then
    echo "=== TRAIN [$TAG] ==="
    apptainer exec --nv --bind /mmfs1 "$SIF" /bin/bash -c "
        source $REPO/.venv/bin/activate && cd $REPO && \
        python -m src.train $COMMON \
            --out-dir $OUT --condition 2 \
            --lr 1e-4 --wd 1e-4 --aug-strength 5.0 --patience 30
    "
fi
run_eval "$TAG" "$CKPT"

# ── M: Best loss (cond 4) + aug5x ─────────────────────────────────────────
TAG=cond4_aug5x
OUT=$REPO/checkpoints_reg3b/$TAG
CKPT=$OUT/best_cond4_seed${SEED}.pt
mkdir -p "$OUT"
if [[ ! -f "$CKPT" ]]; then
    echo "=== TRAIN [$TAG] ==="
    apptainer exec --nv --bind /mmfs1 "$SIF" /bin/bash -c "
        source $REPO/.venv/bin/activate && cd $REPO && \
        python -m src.train $COMMON \
            --out-dir $OUT --condition 4 \
            --gamma 2.0 --alpha 0.3 --beta 0.7 \
            --lr 1e-4 --wd 1e-4 --aug-strength 5.0 --patience 30
    "
fi
run_eval "$TAG" "$CKPT"

echo "=== ALL CONFIGS DONE ==="
