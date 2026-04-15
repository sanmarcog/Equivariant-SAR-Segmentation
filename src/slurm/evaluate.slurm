#!/bin/bash
#SBATCH --job-name=sar-eval
#SBATCH --account=demo
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output=logs/eval_cond${CONDITION}_seed${SEED}_%j.out
#SBATCH --error=logs/eval_cond${CONDITION}_seed${SEED}_%j.err

# Evaluate a saved checkpoint on val or OOD test.
# Usage:
#   CONDITION=5 SEED=0 SPLIT=test sbatch src/slurm/evaluate.slurm

: "${CONDITION:?CONDITION env var required}"
: "${SEED:?SEED env var required}"
SPLIT=${SPLIT:-test}

REPO=/mmfs1/gscratch/scrubbed/sanmarco/equivariant-sar-seg
DATA_DIR=/mmfs1/gscratch/scrubbed/sanmarco/equivariant-sar/data/raw
STATS=$REPO/data/norm_stats_12ch.json
CKPT=$REPO/checkpoints/cond${CONDITION}/best_cond${CONDITION}_seed${SEED}.pt
OUT=$REPO/results/eval_cond${CONDITION}_seed${SEED}_${SPLIT}.json

mkdir -p "$REPO/logs" "$REPO/results"

apptainer exec --nv --bind /mmfs1 \
    /mmfs1/gscratch/scrubbed/sanmarco/pytorch_24.12-py3.sif \
    bash -c "
        source $REPO/.venv/bin/activate
        cd $REPO
        python -m src.evaluate \
            --ckpt     $CKPT \
            --data-dir $DATA_DIR \
            --stats    $STATS \
            --split    $SPLIT \
            --out      $OUT
    "

echo "Evaluation complete: $OUT"
