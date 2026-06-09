#!/bin/bash
#SBATCH --job-name=sweep_atl
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0-12:00:00
#SBATCH --output=slurm_sweep_atl_%j.out
#SBATCH --error=slurm_sweep_atl_%j.err

# ── Atlantic parameter sweep: generalisation test ──
#
# 60 configs × 20 departures = 1200 CMA-ES runs (CPU only).
# Estimated ~40 min at ~2 s/run.

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"

SCRATCH="/scratch/${USER}/routetools"
cd "$SCRATCH"
source .venv/bin/activate

export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

DATA="data/era5"
OUTDIR="output"
mkdir -p "$OUTDIR"

echo "======================================"
echo "Atlantic sweep on $(hostname)"
echo "Date:   $(date)"
echo "CPUs:   ${SLURM_CPUS_PER_TASK}"
echo "Output: ${OUTDIR}/sweep_atlantic.csv"
echo "======================================"

python scripts/sweep_atlantic.py \
    --wind-path "${DATA}/era5_wind_atlantic_2024.nc" \
    --wave-path "${DATA}/era5_waves_atlantic_2024.nc" \
    --output "${OUTDIR}/sweep_atlantic.csv"

# ── Copy results back to /home ──
HOME_OUTDIR="$HOME/routetools/output"
mkdir -p "$HOME_OUTDIR"
cp -v "$OUTDIR/sweep_atlantic.csv" "$HOME_OUTDIR/"

echo ""
echo "Atlantic sweep completed at $(date)"
