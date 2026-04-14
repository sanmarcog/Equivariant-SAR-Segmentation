#!/bin/bash
# src/slurm/dispatch_training.sh
#
# Reads grid_results.json produced by grid_search, then submits:
#   - 15 training jobs (conditions 1-5 × seeds 0,1,2) with correct hparams
#   - 15 eval jobs, each depending on its paired training job
#
# Called automatically by launch_all.sh as a SLURM job step after grid_search.
# Can also be called manually:
#   bash src/slurm/dispatch_training.sh

set -euo pipefail

REPO=/gscratch/scrubbed/sanmarco/equivariant-sar-seg
GRID_RESULTS=$REPO/checkpoints/grid/cond5/grid_results.json

# ── Read best hparams from grid results ──────────────────────────────────────
if [[ ! -f "$GRID_RESULTS" ]]; then
    echo "ERROR: Grid results not found at $GRID_RESULTS" >&2
    exit 1
fi

# Parse best hparams with Python (avoid jq dependency)
read -r BEST_GAMMA BEST_ALPHA BEST_BETA BEST_POS_FRAC < <(python3 - <<EOF
import json
with open("$GRID_RESULTS") as f:
    results = json.load(f)
best = results[0]  # already sorted by F2 descending
print(best["gamma"], best["alpha"], best["beta"], best["pos_frac"])
EOF
)

echo "Best hparams from grid search:"
echo "  gamma=$BEST_GAMMA  alpha=$BEST_ALPHA  beta=$BEST_BETA  pos_frac=$BEST_POS_FRAC"

# ── Submit training jobs ──────────────────────────────────────────────────────
# Hparam propagation rules (from wiki):
#   pos_frac → conditions 2–5  (cond 1 uses random sampling, pos_frac ignored by code)
#   gamma/alpha/beta → conditions 3–5  (conds 1–2 use BCE mode, focal params unused)
declare -a TRAIN_JOB_IDS
declare -a EVAL_JOB_IDS

for CONDITION in 1 2 3 4 5; do
    for SEED in 0 1 2; do

        # Set hparams per condition
        if   [[ $CONDITION -le 2 ]]; then
            # BCE mode: focal/tversky params irrelevant; pos_frac matters for cond 2
            GAMMA=2.0; ALPHA=0.3; BETA=0.7
            POS_FRAC=$BEST_POS_FRAC
        else
            # Focal+Tversky: use all grid-search winners
            GAMMA=$BEST_GAMMA; ALPHA=$BEST_ALPHA; BETA=$BEST_BETA
            POS_FRAC=$BEST_POS_FRAC
        fi

        JOB_ID=$(
            CONDITION=$CONDITION SEED=$SEED \
            GAMMA=$GAMMA ALPHA=$ALPHA BETA=$BETA POS_FRAC=$POS_FRAC \
            sbatch --parsable \
                "$REPO/src/slurm/train.slurm"
        )
        echo "Submitted training  cond=$CONDITION seed=$SEED → job $JOB_ID"
        TRAIN_JOB_IDS+=("$JOB_ID")

        # Submit matching eval job with dependency
        EVAL_JOB_ID=$(
            CONDITION=$CONDITION SEED=$SEED SPLIT=test \
            sbatch --parsable \
                --dependency=afterok:$JOB_ID \
                "$REPO/src/slurm/evaluate.slurm"
        )
        echo "Submitted eval      cond=$CONDITION seed=$SEED → job $EVAL_JOB_ID (after $JOB_ID)"
        EVAL_JOB_IDS+=("$EVAL_JOB_ID")

    done
done

# ── Submit aggregate job after all evals ──────────────────────────────────────
ALL_EVAL_IDS=$(IFS=:; echo "${EVAL_JOB_IDS[*]}")

AGG_JOB_ID=$(
    sbatch --parsable \
        --dependency=afterok:$ALL_EVAL_IDS \
        --job-name=sar-aggregate \
        --account=stf \
        --partition=cpu-g2 \
        --nodes=1 \
        --ntasks-per-node=1 \
        --cpus-per-task=2 \
        --mem=8G \
        --time=0:30:00 \
        --output="$REPO/logs/aggregate_%j.out" \
        --error="$REPO/logs/aggregate_%j.err" \
        --wrap="
            apptainer exec \
                /gscratch/scrubbed/sanmarco/pytorch_24.12-py3.sif \
                bash -c '
                    source $REPO/.venv/bin/activate
                    cd $REPO
                    python -m src.aggregate \
                        --results-dir $REPO/results \
                        --split test \
                        --out $REPO/results/ablation_tables.json
                '
        "
)

echo ""
echo "Pipeline submitted:"
printf '  Training jobs:  %s\n' "${TRAIN_JOB_IDS[@]}"
printf '  Eval jobs:      %s\n' "${EVAL_JOB_IDS[@]}"
echo "  Aggregate job:  $AGG_JOB_ID"
echo ""
echo "Monitor with: squeue -u sanmarco"
echo "Tables will appear in: $REPO/results/ablation_tables.json"
echo "Aggregate log:          $REPO/logs/aggregate_${AGG_JOB_ID}.out"
