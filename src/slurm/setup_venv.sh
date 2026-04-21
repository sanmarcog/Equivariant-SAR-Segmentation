#!/bin/bash
# src/slurm/setup_venv.sh
#
# One-time setup: create the Python venv inside the Apptainer container.
# Run this ONCE from the login node before submitting training jobs.
#
# Usage:
#   apptainer exec --bind /mmfs1 --nv --bind /mmfs1 ${SIF:?Set SIF to container path} \
#       bash src/slurm/setup_venv.sh

set -euo pipefail

REPO=${REPO:?Set REPO to repo root}
VENV=$REPO/.venv

echo "Creating venv at $VENV ..."
python -m venv --system-site-packages "$VENV"
source "$VENV/bin/activate"

echo "Installing requirements ..."
pip install --upgrade pip
pip install -r "$REPO/requirements.txt"

echo "Done. Venv at $VENV"
