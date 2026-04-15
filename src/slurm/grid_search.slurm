#!/bin/bash
#SBATCH --job-name=sar-grid
#SBATCH --account=demo
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --output=logs/grid_cond${CONDITION}_%j.out
#SBATCH --error=logs/grid_cond${CONDITION}_%j.err

# Hyperparameter grid search (18 combinations × 20 epochs).
# Run this BEFORE full 110-epoch training to select γ, α/β, pos_frac.
#
# Usage:
#   CONDITION=5 sbatch src/slurm/grid_search.slurm

: "${CONDITION:?CONDITION env var required}"

REPO=/mmfs1/gscratch/scrubbed/sanmarco/equivariant-sar-seg
DATA_DIR=/mmfs1/gscratch/scrubbed/sanmarco/equivariant-sar/data/raw
STATS=$REPO/data/norm_stats_12ch.json
OUT_DIR=$REPO/checkpoints/grid/cond${CONDITION}

mkdir -p "$REPO/logs" "$OUT_DIR"

apptainer exec --nv --bind /mmfs1 \
    /mmfs1/gscratch/scrubbed/sanmarco/pytorch_24.12-py3.sif \
    bash -c "
        source $REPO/.venv/bin/activate
        cd $REPO
        python -m src.train \
            --data-dir    $DATA_DIR \
            --stats       $STATS \
            --out-dir     $OUT_DIR \
            --condition   $CONDITION \
            --seed        0 \
            --grid-search \
            --grid-epochs 20 \
            --batch-size  32 \
            --num-workers 8
    "

echo "Grid search complete for condition $CONDITION"
echo "Results: $OUT_DIR/grid_results.json"
