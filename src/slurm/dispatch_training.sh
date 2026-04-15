#!/bin/bash
#SBATCH --job-name=sar-seq
#SBATCH --account=demo
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=/mmfs1/gscratch/scrubbed/sanmarco/equivariant-sar-seg/logs/seq_%j.out
#SBATCH --error=/mmfs1/gscratch/scrubbed/sanmarco/equivariant-sar-seg/logs/seq_%j.err
#
# Sequential pipeline: reads grid_results.json, then runs all 15 training
# runs and all 15 eval runs back-to-back in a single job, then aggregates.
# Single job avoids MaxSubmit=1 / MaxJobs=1 account limits.
#
# Submitted automatically by launch_all.sh after grid_search completes.
# Can also be submitted manually:
#   sbatch src/slurm/dispatch_training.sh

set -euo pipefail

REPO=/mmfs1/gscratch/scrubbed/sanmarco/equivariant-sar-seg
DATA_DIR=/mmfs1/gscratch/scrubbed/sanmarco/equivariant-sar/data/raw
STATS=$REPO/data/norm_stats_12ch.json
SIF=/mmfs1/gscratch/scrubbed/sanmarco/pytorch_24.12-py3.sif
GRID_RESULTS=$REPO/checkpoints/grid/cond5/grid_results.json

mkdir -p "$REPO/logs" "$REPO/results"

# ── Read best hparams from grid results ──────────────────────────────────────
if [[ ! -f "$GRID_RESULTS" ]]; then
    echo "ERROR: Grid results not found at $GRID_RESULTS" >&2
    exit 1
fi

read -r BEST_GAMMA BEST_ALPHA BEST_BETA BEST_POS_FRAC < <(python3 - <<EOF
import json
with open("$GRID_RESULTS") as f:
    results = json.load(f)
best = results[0]  # sorted by val pixel F2 descending
print(best["gamma"], best["alpha"], best["beta"], best["pos_frac"])
EOF
)

echo "===== Phase 2 sequential pipeline ====="
echo "Best hparams: gamma=$BEST_GAMMA  alpha=$BEST_ALPHA  beta=$BEST_BETA  pos_frac=$BEST_POS_FRAC"
echo ""

# ── Sequential training + eval loop ──────────────────────────────────────────
# Hparam propagation:
#   pos_frac → conds 2-5  (cond 1 uses random sampling)
#   gamma/alpha/beta → conds 3-5  (conds 1-2 use BCE, focal params unused)

for CONDITION in 1 2 3 4 5; do
    for SEED in 0 1 2; do

        if [[ $CONDITION -le 2 ]]; then
            GAMMA=2.0; ALPHA=0.3; BETA=0.7
            POS_FRAC=$BEST_POS_FRAC
        else
            GAMMA=$BEST_GAMMA; ALPHA=$BEST_ALPHA; BETA=$BEST_BETA
            POS_FRAC=$BEST_POS_FRAC
        fi

        OUT_DIR=$REPO/checkpoints/cond${CONDITION}
        CKPT=$OUT_DIR/best_cond${CONDITION}_seed${SEED}.pt
        EVAL_OUT=$REPO/results/eval_cond${CONDITION}_seed${SEED}_test.json

        echo "------------------------------------------------------------"
        echo "TRAIN  condition=$CONDITION  seed=$SEED  gamma=$GAMMA  alpha=$ALPHA  beta=$BETA  pos_frac=$POS_FRAC"
        echo "------------------------------------------------------------"

        apptainer exec --nv --bind /mmfs1 "$SIF" /bin/bash -c "
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
                --epochs    110 \
                --batch-size 32 \
                --lr 1e-4 \
                --wd 1e-4 \
                --warmup-epochs 10 \
                --num-workers 8 \
                --no-wandb
        "

        echo "------------------------------------------------------------"
        echo "EVAL   condition=$CONDITION  seed=$SEED"
        echo "------------------------------------------------------------"

        apptainer exec --nv --bind /mmfs1 "$SIF" /bin/bash -c "
            source $REPO/.venv/bin/activate
            cd $REPO
            python -m src.evaluate \
                --ckpt     $CKPT \
                --data-dir $DATA_DIR \
                --stats    $STATS \
                --split    test \
                --out      $EVAL_OUT \
                --n-bootstrap 10000 \
                --n-perm      10000
        "

        echo "Done: cond=$CONDITION seed=$SEED -> $EVAL_OUT"
        echo ""

    done
done

# ── Aggregate → ablation tables ───────────────────────────────────────────────
echo "============================================================"
echo "AGGREGATE -> results/ablation_tables.json"
echo "============================================================"

apptainer exec --bind /mmfs1 "$SIF" /bin/bash -c "
    source $REPO/.venv/bin/activate
    cd $REPO
    python -m src.aggregate \
        --results-dir $REPO/results \
        --split test \
        --out $REPO/results/ablation_tables.json
"

echo ""
echo "Pipeline complete. Results: $REPO/results/ablation_tables.json"
