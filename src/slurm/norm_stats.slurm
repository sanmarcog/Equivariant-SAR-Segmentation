#!/bin/bash
#SBATCH --job-name=sar-norm-stats
#SBATCH --account=demo
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:0
#SBATCH --time=1:00:00
#SBATCH --output=logs/norm_stats_%j.out
#SBATCH --error=logs/norm_stats_%j.err

# Prerequisites:
#   1. sync-sar2 (repo is at $REPO)
#   2. AvalCD data at $DATA_DIR
#   3. Venv at $REPO/.venv (run setup_venv.sh once)
#
# Submits from login node:
#   sbatch src/slurm/norm_stats.slurm

set -euo pipefail
REPO=/mmfs1/gscratch/scrubbed/sanmarco/equivariant-sar-seg
DATA_DIR=/mmfs1/gscratch/scrubbed/sanmarco/equivariant-sar/data/raw
STATS_OUT=$REPO/data/norm_stats_12ch.json

mkdir -p "$REPO/logs" "$REPO/data"

apptainer exec --nv --bind /mmfs1 \
    /mmfs1/gscratch/scrubbed/sanmarco/pytorch_24.12-py3.sif \
    bash -c "
        source $REPO/.venv/bin/activate
        cd $REPO
        python -m src.data.norm_stats \
            --data-dir $DATA_DIR \
            --out $STATS_OUT
    "

echo "Norm stats written to $STATS_OUT"
