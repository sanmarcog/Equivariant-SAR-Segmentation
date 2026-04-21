#!/bin/bash
#SBATCH --job-name=sar-reeval
#SBATCH --account=demo
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Re-evaluate all 15 sar-seq checkpoints with --morph-closing --no-tta.
# Writes results into a parallel results_v2/ dir so the original ablation
# numbers are preserved for comparison.
#
# Run after sar-seq finishes:  sbatch src/slurm/reeval_morph_notta.sh

set -uo pipefail

REPO=${REPO:?Set REPO to repo root}
DATA_DIR=${DATA_DIR:?Set DATA_DIR}
STATS=$REPO/data/norm_stats_12ch.json
SIF=${SIF:?Set SIF to container path}
OUT_DIR=${OUT_DIR:-$REPO/results_v2}

mkdir -p "$OUT_DIR"

for COND in 1 2 3 4 5; do
    for SEED in 0 1 2; do
        CKPT=$REPO/checkpoints/cond${COND}/best_cond${COND}_seed${SEED}.pt
        EVAL_OUT=$OUT_DIR/eval_cond${COND}_seed${SEED}_test.json

        if [[ ! -f "$CKPT" ]]; then
            echo "Missing checkpoint: $CKPT — skipping"
            continue
        fi
        if [[ -f "$EVAL_OUT" ]]; then
            echo "Already done: $EVAL_OUT — skipping"
            continue
        fi

        echo "=== EVAL cond=$COND seed=$SEED ==="
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
                --multi-threshold \
                --frozen-thresholds 0.3 0.5 0.7 \
                --n-bootstrap 1000 \
                --n-perm      1000
        "
    done
done

echo "=== AGGREGATE ==="
apptainer exec --bind "${BIND_ROOT:?Set BIND_ROOT}" "$SIF" /bin/bash -c "
    source $REPO/.venv/bin/activate
    cd $REPO
    python -m src.aggregate \
        --results-dir $OUT_DIR \
        --split test \
        --out $OUT_DIR/ablation_tables.json
"
echo "DONE: results in $OUT_DIR/"
