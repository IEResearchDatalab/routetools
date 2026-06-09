#!/bin/bash
#SBATCH --job-name=sweep_a
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0-06:00:00
#SBATCH --output=slurm_sweep_a_%j.out
#SBATCH --error=slurm_sweep_a_%j.err

# ── Stage A parameter sweep: catastrophic Pacific detours ──
#
# 30 configs × 10 departures = 300 CMA-ES runs (CPU only).

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
echo "Stage A sweep on $(hostname)"
echo "Date:   $(date)"
echo "CPUs:   ${SLURM_CPUS_PER_TASK}"
echo "Output: ${OUTDIR}/sweep_stage_a.csv"
echo "======================================"

python scripts/sweep_stage_a.py \
    --wind-path "${DATA}/era5_wind_pacific_2024.nc" \
    --wave-path "${DATA}/era5_waves_pacific_2024.nc" \
    --output "${OUTDIR}/sweep_stage_a.csv"

# ── Copy results back to /home ──
HOME_OUTDIR="$HOME/routetools/output"
mkdir -p "$HOME_OUTDIR"
cp -v "$OUTDIR/sweep_stage_a.csv" "$HOME_OUTDIR/"

echo ""
echo "Stage A sweep completed at $(date)"
