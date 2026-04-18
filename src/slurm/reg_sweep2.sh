#!/bin/bash
#SBATCH --job-name=sar-reg-sweep2
#SBATCH --account=demo
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --exclude=z3005,z3006
#SBATCH --output=/mmfs1/gscratch/scrubbed/sanmarco/equivariant-sar-seg/logs/reg_sweep2_%j.out
#SBATCH --error=/mmfs1/gscratch/scrubbed/sanmarco/equivariant-sar-seg/logs/reg_sweep2_%j.err
#
# Sweep 2: augmentation strength + light regularization.
# Lesson from sweep 1: heavy LR/WD/dropout suppresses performance.
# New hypothesis: keep LR=1e-4, add data diversity via stronger augmentation.

set -uo pipefail
REPO=/mmfs1/gscratch/scrubbed/sanmarco/equivariant-sar-seg
DATA_DIR=/mmfs1/gscratch/scrubbed/sanmarco/equivariant-sar/data/raw
STATS=$REPO/data/norm_stats_12ch.json
SIF=/mmfs1/gscratch/scrubbed/sanmarco/pytorch_24.12-py3.sif
SEED=0

run_config() {
    local TAG=$1 LR=$2 WD=$3 DEC_DROP=$4 AUG_STR=$5

    OUT_DIR=$REPO/checkpoints_reg2/${TAG}
    EVAL_DIR=$REPO/results_reg2/${TAG}
    mkdir -p "$OUT_DIR" "$EVAL_DIR"
    CKPT=$OUT_DIR/best_cond2_seed${SEED}.pt

    if [[ -f "$CKPT" ]]; then
        echo "[$TAG] Checkpoint exists — skipping training"
    else
        echo "=== TRAIN [$TAG] lr=$LR wd=$WD dec_drop=$DEC_DROP aug=$AUG_STR ==="
        apptainer exec --nv --bind /mmfs1 "$SIF" /bin/bash -c "
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
                --aug-strength $AUG_STR \
                --epochs 110 \
                --patience 30 \
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

    echo "[$TAG] Gaussian test F1:"
    grep '"best_f1"' $EVAL_F1 | head -1

    echo "[$TAG] DONE"
}

# ── Config D: Original LR + 2x augmentation (main hypothesis) ────────────
run_config "aug2x"           1e-4  1e-4  0.0  2.0

# ── Config E: Original LR + 3x augmentation ──────────────────────────────
run_config "aug3x"           1e-4  1e-4  0.0  3.0

# ── Config F: Original LR + 2x aug + light decoder dropout ───────────────
run_config "aug2x_drop01"    1e-4  1e-4  0.1  2.0

echo "=== ALL CONFIGS DONE ==="
