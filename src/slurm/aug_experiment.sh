#!/bin/bash
#SBATCH --job-name=sar-aug
#SBATCH --account=demo
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Augmentation experiment: train cond 4 + 5 with --online-aug, 3 seeds each.
# Picks the same hparams as the main ablation. Eval with --morph-closing --no-tta.
# Cond 4 (no copy-paste) included to disentangle online-aug from copy-paste.
#
# REQUIRES: aug-equivariant branch must be merged or files synced to Hyak.
# Check before running:
#   ssh klone "grep -q OnlineAugment ${REPO:?Set REPO to repo root}/src/train.py && echo OK || echo NEEDS_AUG_BRANCH"

set -uo pipefail

REPO=${REPO:?Set REPO to repo root}
DATA_DIR=${DATA_DIR:?Set DATA_DIR}
STATS=$REPO/data/norm_stats_12ch.json
SIF=${SIF:?Set SIF to container path}
GRID_RESULTS=$REPO/checkpoints/grid/cond5/grid_results.json

# Read winning hparams from grid
read -r BEST_GAMMA BEST_ALPHA BEST_BETA BEST_POS_FRAC < <(python3 - <<EOF
import json
with open("$GRID_RESULTS") as f: results = json.load(f)
b = results[0]
print(b["gamma"], b["alpha"], b["beta"], b["pos_frac"])
EOF
)

OUT_BASE=$REPO/checkpoints_aug
RESULTS=$REPO/results_aug
mkdir -p "$OUT_BASE" "$RESULTS"

for CONDITION in 5 4; do
    for SEED in 0 1 2; do
        OUT_DIR=$OUT_BASE/cond${CONDITION}
        CKPT=$OUT_DIR/best_cond${CONDITION}_seed${SEED}.pt
        EVAL_OUT=$RESULTS/eval_cond${CONDITION}_seed${SEED}_test.json

        echo "=== TRAIN cond=$CONDITION seed=$SEED (with online-aug) ==="
        if [[ -f "$CKPT" ]]; then
            echo "checkpoint exists, skipping training: $CKPT"
        else
            apptainer exec --nv --bind "${BIND_ROOT:?Set BIND_ROOT}" "$SIF" /bin/bash -c "
                source $REPO/.venv/bin/activate
                cd $REPO
                python -m src.train \
                    --data-dir  $DATA_DIR \
                    --stats     $STATS \
                    --out-dir   $OUT_DIR \
                    --condition $CONDITION \
                    --seed      $SEED \
                    --gamma     $BEST_GAMMA \
                    --alpha     $BEST_ALPHA \
                    --beta      $BEST_BETA \
                    --pos-frac  $BEST_POS_FRAC \
                    --epochs    110 \
                    --batch-size 32 \
                    --lr 1e-4 \
                    --wd 1e-4 \
                    --warmup-epochs 10 \
                    --num-workers 8 \
                    --no-wandb \
                    --online-aug
            "
        fi

        echo "=== EVAL cond=$CONDITION seed=$SEED ==="
        if [[ -f "$EVAL_OUT" ]]; then
            echo "eval exists, skipping: $EVAL_OUT"
        else
            apptainer exec --nv --bind "${BIND_ROOT:?Set BIND_ROOT}" "$SIF" /bin/bash -c "
                source $REPO/.venv/bin/activate
                cd $REPO
                python -m src.evaluate \
                    --ckpt     $CKPT \
                    --data-dir $DATA_DIR \
                    --stats    $STATS \
                    --split    test \
                    --out      $EVAL_OUT \
                    --no-tta \
                    --morph-closing \
                    --n-bootstrap 1000 \
                    --n-perm      1000
            "
        fi
    done
done

echo "=== AGGREGATE (aug) ==="
apptainer exec --bind "${BIND_ROOT:?Set BIND_ROOT}" "$SIF" /bin/bash -c "
    source $REPO/.venv/bin/activate
    cd $REPO
    python -m src.aggregate \
        --conditions 4 5 \
        --results-dir $RESULTS \
        --split test \
        --out $RESULTS/ablation_tables.json
"
echo "DONE"
