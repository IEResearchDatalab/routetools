#!/bin/bash
#SBATCH --job-name=swopp3_stg
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# ── Stage repo + ERA5 data onto local SSD (/scratch) ──
#
# First run:  clones the repo and copies data + venv (~19 GB).
# Later runs: pulls latest code; rsync --update skips unchanged data.

set -euo pipefail

SCRATCH="/scratch/${USER}/routetools"
BRANCH="$(git -C "$HOME/routetools" rev-parse --abbrev-ref HEAD)"

if [ -d "$SCRATCH/.git" ]; then
    echo "Repo already on scratch – pulling latest…"
    git -C "$SCRATCH" fetch origin "$BRANCH"
    git -C "$SCRATCH" checkout "$BRANCH"
    git -C "$SCRATCH" reset --hard "origin/$BRANCH"
elif [ -d "$SCRATCH" ]; then
    echo "Directory exists without .git – initialising repo in place…"
    git -C "$SCRATCH" init
    git -C "$SCRATCH" remote add origin "$HOME/routetools"
    git -C "$SCRATCH" fetch origin "$BRANCH"
    git -C "$SCRATCH" checkout -f "$BRANCH"
else
    echo "First run – cloning repo to $SCRATCH …"
    git clone --branch "$BRANCH" --single-branch \
        "$HOME/routetools" "$SCRATCH"
fi

# Sync non-git artefacts (data + venv) – --update skips unchanged files
rsync -a --update \
    "$HOME/routetools/data/" "$SCRATCH/data/"
rsync -a --update \
    "$HOME/routetools/.venv/" "$SCRATCH/.venv/"

echo "Staged to $SCRATCH ($(du -sh "$SCRATCH" | cut -f1)) at $(date)"
echo "HEAD: $(git -C "$SCRATCH" log --oneline -1)"
