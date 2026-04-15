#!/bin/bash
# src/slurm/launch_all.sh
#
# Master launcher — submits the full Phase 2 pipeline with SLURM job dependencies.
# Run from the Hyak login node after syncing the repo.
#
# Pipeline:
#   1. norm_stats      (CPU, ~30 min)
#   2. grid_search     (GPU, ~8 hr)   — depends on 1
#   3. dispatch        (CPU, ~5 min)  — depends on 2; reads grid results, submits:
#       3a. 15 training jobs          — conds 1-5 × seeds 0-2
#       3b. 15 eval jobs              — each depends on its paired training job
#       3c.  1 aggregate job          — depends on all 15 eval jobs
#
# Usage:
#   bash src/slurm/launch_all.sh
#
# To skip norm_stats (already done):
#   SKIP_NORM=1 bash src/slurm/launch_all.sh
#
# To skip grid_search (already done):
#   SKIP_NORM=1 SKIP_GRID=1 bash src/slurm/launch_all.sh

set -euo pipefail

REPO=/mmfs1/gscratch/scrubbed/sanmarco/equivariant-sar-seg
SKIP_NORM=${SKIP_NORM:-0}
SKIP_GRID=${SKIP_GRID:-0}

echo "=== Phase 2 pipeline launcher ==="
echo "Repo: $REPO"
echo ""

# ── Step 1: norm_stats ────────────────────────────────────────────────────────
if [[ $SKIP_NORM -eq 1 ]]; then
    echo "[1/3] norm_stats — SKIPPED"
    NORM_DEP=""
else
    NORM_JOB=$(sbatch --parsable "$REPO/src/slurm/norm_stats.slurm")
    echo "[1/3] norm_stats submitted → job $NORM_JOB"
    NORM_DEP="--dependency=afterok:$NORM_JOB"
fi

# ── Step 2: grid_search (condition 5, seed 0) ─────────────────────────────────
if [[ $SKIP_GRID -eq 1 ]]; then
    echo "[2/3] grid_search — SKIPPED"
    GRID_DEP=""
else
    GRID_JOB=$(
        CONDITION=5 \
        sbatch --parsable $NORM_DEP "$REPO/src/slurm/grid_search.slurm"
    )
    echo "[2/3] grid_search submitted → job $GRID_JOB ${NORM_DEP:+(after $NORM_JOB)}"
    GRID_DEP="--dependency=afterok:$GRID_JOB"
fi

# ── Step 3: dispatch (reads grid results, submits training + eval + aggregate) ──
DISPATCH_JOB=$(
    sbatch --parsable \
        $GRID_DEP \
        --job-name=sar-dispatch \
        --account=demo \
        --partition=ckpt \
        --nodes=1 \
        --ntasks-per-node=1 \
        --cpus-per-task=2 \
        --mem=4G \
        --time=0:15:00 \
        --output="$REPO/logs/dispatch_%j.out" \
        --error="$REPO/logs/dispatch_%j.err" \
        --wrap="bash $REPO/src/slurm/dispatch_training.sh"
)
echo "[3/3] dispatch submitted → job $DISPATCH_JOB ${GRID_DEP:+(after $GRID_JOB)}"

echo ""
echo "Full pipeline queued. Monitor with:"
echo "  squeue -u sanmarco"
echo "  tail -f $REPO/logs/dispatch_${DISPATCH_JOB}.out"
