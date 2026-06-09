#!/bin/bash
#SBATCH --job-name=sweep_b
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0-08:00:00
#SBATCH --output=slurm_sweep_b_%j.out
#SBATCH --error=slurm_sweep_b_%j.err

# ── Stage B parameter sweep: unstick GC-stuck Pacific departures ──
#
# 24 configs × 20 departures = 480 CMA-ES runs (CPU only).
# Some configs use maxfevals=50000, so budget extra time.

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
echo "Stage B sweep on $(hostname)"
echo "Date:   $(date)"
echo "CPUs:   ${SLURM_CPUS_PER_TASK}"
echo "Output: ${OUTDIR}/sweep_stage_b.csv"
echo "======================================"

python scripts/sweep_stage_b.py \
    --wind-path "${DATA}/era5_wind_pacific_2024.nc" \
    --wave-path "${DATA}/era5_waves_pacific_2024.nc" \
    --output "${OUTDIR}/sweep_stage_b.csv"

# ── Copy results back to /home ──
HOME_OUTDIR="$HOME/routetools/output"
mkdir -p "$HOME_OUTDIR"
cp -v "$OUTDIR/sweep_stage_b.csv" "$HOME_OUTDIR/"

echo ""
echo "Stage B sweep completed at $(date)"
