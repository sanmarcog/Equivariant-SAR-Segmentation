#!/bin/bash
#SBATCH --job-name=sar-reg-sweep
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
# Regularization sweep: fix overfitting (model peaks at epoch 8/110).
# Run 3 configs sequentially inside one job.
# Each uses early stopping (patience=20) so bad configs die fast.

set -uo pipefail
REPO=${REPO:?Set REPO to repo root}
DATA_DIR=${DATA_DIR:?Set DATA_DIR}
STATS=$REPO/data/norm_stats_12ch.json
SIF=${SIF:?Set SIF to container path}
SEED=0

run_config() {
    local TAG=$1 LR=$2 WD=$3 DEC_DROP=$4

    OUT_DIR=$REPO/checkpoints_reg/${TAG}
    EVAL_DIR=$REPO/results_reg/${TAG}
    mkdir -p "$OUT_DIR" "$EVAL_DIR"
    CKPT=$OUT_DIR/best_cond2_seed${SEED}.pt

    if [[ -f "$CKPT" ]]; then
        echo "[$TAG] Checkpoint exists — skipping training"
    else
        echo "=== TRAIN [$TAG] lr=$LR wd=$WD dec_drop=$DEC_DROP ==="
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
                --patience 20 \
                --warmup-epochs 10 \
                --batch-size 32 \
                --lr $LR \
                --wd $WD \
                --dec-dropout $DEC_DROP \
                --num-workers 8 \
                --no-wandb
        "
    fi

    # Eval at F1-opt (Gaussian blending)
    EVAL_F1=$EVAL_DIR/eval_gaussian_f1opt.json
    if [[ ! -f "$EVAL_F1" ]]; then
        echo "=== EVAL [$TAG] Gaussian F1-opt ==="
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

    # Eval at F2-opt (Max blending)
    EVAL_F2=$EVAL_DIR/eval_max_f2opt.json
    if [[ ! -f "$EVAL_F2" ]]; then
        echo "=== EVAL [$TAG] Max F2-opt ==="
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

    echo "[$TAG] DONE"
}

# ── Config A: Lower LR + decoder dropout + higher WD ──────────────────────
run_config "lr3e5_drop02_wd1e3"  3e-5  1e-3  0.2

# ── Config B: Even lower LR + stronger dropout ────────────────────────────
run_config "lr1e5_drop03_wd1e3"  1e-5  1e-3  0.3

# ── Config C: Middle ground ───────────────────────────────────────────────
run_config "lr3e5_drop03_wd5e4"  3e-5  5e-4  0.3

echo "=== ALL CONFIGS DONE ==="
