#!/bin/bash
#SBATCH --job-name=sar-lam-sweep
#SBATCH --account=demo
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --exclude=z3005,z3006
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -uo pipefail
REPO=${REPO:?Set REPO to repo root}
SIF=${SIF:?Set SIF to container path}
SEED=1

run_eval() {
    local TAG=$1 CKPT=$2
    local EVAL_DIR=$REPO/results_lam/${TAG}
    mkdir -p "$EVAL_DIR"
    local EVAL=$EVAL_DIR/eval_s16_tta.json
    if [[ ! -f "$EVAL" ]]; then
        echo "=== EVAL [$TAG] ==="
        apptainer exec --nv --bind "${BIND_ROOT:?Set BIND_ROOT}" "$SIF" /bin/bash -c "
            source $REPO/.venv/bin/activate && cd $REPO && \
            python -m src.evaluate \
                --ckpt $CKPT \
                --data-dir ${DATA_DIR:?Set DATA_DIR} \
                --stats $REPO/data/norm_stats_12ch.json \
                --split test --out $EVAL \
                --patch-size 64 --stride 16 --blending gaussian \
                --morph-closing --multi-threshold --frozen-thresholds 0.3 0.5 0.7 \
                --n-bootstrap 1000 --n-perm 1000
        "
    fi
    echo "[$TAG]:"
    grep '"best_f1"\|"thr_f1"\|"best_f2"' $EVAL 2>/dev/null | head -3
}

COMMON="--data-dir ${DATA_DIR:?Set DATA_DIR} --stats $REPO/data/norm_stats_12ch.json --condition 1 --seed $SEED --patch-size 64 --train-stride 32 --epochs 110 --patience 30 --warmup-epochs 10 --batch-size 32 --lr 1e-4 --wd 1e-4 --pos-weight 1.0 --num-workers 8 --no-wandb"

# --- lambda=0.5 (most informative single point) ---
TAG=lam0.5
OUT=$REPO/checkpoints_lam/$TAG
CKPT=$OUT/best_cond1_seed${SEED}.pt
mkdir -p "$OUT"
if [[ ! -f "$CKPT" ]]; then
    echo "=== TRAIN [$TAG] ==="
    apptainer exec --nv --bind "${BIND_ROOT:?Set BIND_ROOT}" "$SIF" /bin/bash -c "
        source $REPO/.venv/bin/activate && cd $REPO && \
        python -m src.train $COMMON \
            --out-dir $OUT --loss-mode component_iou --balance-weight 0.5
    "
fi
run_eval "$TAG" "$CKPT"

# --- lambda=0.25 ---
TAG=lam0.25
OUT=$REPO/checkpoints_lam/$TAG
CKPT=$OUT/best_cond1_seed${SEED}.pt
mkdir -p "$OUT"
if [[ ! -f "$CKPT" ]]; then
    echo "=== TRAIN [$TAG] ==="
    apptainer exec --nv --bind "${BIND_ROOT:?Set BIND_ROOT}" "$SIF" /bin/bash -c "
        source $REPO/.venv/bin/activate && cd $REPO && \
        python -m src.train $COMMON \
            --out-dir $OUT --loss-mode component_iou --balance-weight 0.25
    "
fi
run_eval "$TAG" "$CKPT"

# --- lambda=2.0 ---
TAG=lam2.0
OUT=$REPO/checkpoints_lam/$TAG
CKPT=$OUT/best_cond1_seed${SEED}.pt
mkdir -p "$OUT"
if [[ ! -f "$CKPT" ]]; then
    echo "=== TRAIN [$TAG] ==="
    apptainer exec --nv --bind "${BIND_ROOT:?Set BIND_ROOT}" "$SIF" /bin/bash -c "
        source $REPO/.venv/bin/activate && cd $REPO && \
        python -m src.train $COMMON \
            --out-dir $OUT --loss-mode component_iou --balance-weight 2.0
    "
fi
run_eval "$TAG" "$CKPT"

# --- lambda=1 + Dice global (Kofler reference formulation) ---
TAG=lam1_dice
OUT=$REPO/checkpoints_lam/$TAG
CKPT=$OUT/best_cond1_seed${SEED}.pt
mkdir -p "$OUT"
if [[ ! -f "$CKPT" ]]; then
    echo "=== TRAIN [$TAG] ==="
    apptainer exec --nv --bind "${BIND_ROOT:?Set BIND_ROOT}" "$SIF" /bin/bash -c "
        source $REPO/.venv/bin/activate && cd $REPO && \
        python -m src.train $COMMON \
            --out-dir $OUT --loss-mode component_iou_dice --balance-weight 1.0
    "
fi
run_eval "$TAG" "$CKPT"

echo "=== ALL DONE ==="
