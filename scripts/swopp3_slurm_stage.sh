#!/bin/bash
#SBATCH --job-name=swopp3_stg
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:0
#SBATCH --time=00:30:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# ── Stage repo + ERA5 data onto local SSD (/scratch) ──

set -euo pipefail

SCRATCH="/scratch/${USER}/routetools"
mkdir -p "$SCRATCH"

echo "Staging $HOME/routetools → $SCRATCH ..."
rsync -a --delete \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'output' \
    "$HOME/routetools/" "$SCRATCH/"

echo "Staged to $SCRATCH ($(du -sh "$SCRATCH" | cut -f1)) at $(date)"
