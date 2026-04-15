#!/bin/bash
#SBATCH --job-name=sar-seg-train
#SBATCH --account=demo
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_cond${CONDITION}_seed${SEED}_%j.out
#SBATCH --error=logs/train_cond${CONDITION}_seed${SEED}_%j.err

# Environment variables (pass via --export or set before sbatch):
#   CONDITION  — ablation condition 1-5  (required)
#   SEED       — random seed 0,1,2       (required)
#   GAMMA      — focal gamma (default: 2)
#   ALPHA      — Tversky FP weight (default: 0.3)
#   BETA       — Tversky FN weight (default: 0.7)
#   POS_FRAC   — biased sampler pos fraction (default: 0.5)
#   EPOCHS     — training epochs (default: 110)
#
# Example usage (3 seeds for condition 5):
#   for seed in 0 1 2; do
#     CONDITION=5 SEED=$seed sbatch src/slurm/train.slurm
#   done

set -euo pipefail

: "${CONDITION:?CONDITION env var required}"
: "${SEED:?SEED env var required}"
GAMMA=${GAMMA:-2}
ALPHA=${ALPHA:-0.3}
BETA=${BETA:-0.7}
POS_FRAC=${POS_FRAC:-0.5}
EPOCHS=${EPOCHS:-110}

REPO=/mmfs1/gscratch/scrubbed/sanmarco/equivariant-sar-seg
DATA_DIR=/mmfs1/gscratch/scrubbed/sanmarco/equivariant-sar/data/raw
STATS=$REPO/data/norm_stats_12ch.json
OUT_DIR=$REPO/checkpoints/cond${CONDITION}

mkdir -p "$REPO/logs" "$OUT_DIR"

echo "Training: condition=$CONDITION seed=$SEED gamma=$GAMMA alpha=$ALPHA beta=$BETA pos_frac=$POS_FRAC"

apptainer exec --nv --bind /mmfs1 \
    /mmfs1/gscratch/scrubbed/sanmarco/pytorch_24.12-py3.sif \
    bash -c "
        source $REPO/.venv/bin/activate
        cd $REPO
        python -m src.train \
            --data-dir  $DATA_DIR \
            --stats     $STATS \
            --out-dir   $OUT_DIR \
            --condition $CONDITION \
            --seed      $SEED \
            --gamma     $GAMMA \
            --alpha     $ALPHA \
            --beta      $BETA \
            --pos-frac  $POS_FRAC \
            --epochs    $EPOCHS \
            --batch-size 32 \
            --lr 1e-4 \
            --wd 1e-4 \
            --warmup-epochs 10 \
            --num-workers 8 \
            --wandb
    "

echo "Training complete: condition=$CONDITION seed=$SEED"
