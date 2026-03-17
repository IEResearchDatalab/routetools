#!/bin/bash
#SBATCH --job-name=swopp3_npts
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# ── N-points resample comparison (Pacific, GPU) ──
#
# Resamples existing optimized tracks to L = 50, 100, 200 and
# re-evaluates energy. No re-optimization, just resampling.
# Tests best, median, and worst departures.

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"

SCRATCH="/scratch/${USER}/routetools"
cd "$SCRATCH"
source .venv/bin/activate

export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

DATA="data/era5"

echo "======================================"
echo "N-points resample comparison on $(hostname)"
echo "Date:  $(date)"
echo "GPU:   $(nvidia-smi -L 2>/dev/null | head -1 || echo 'unknown')"
echo "======================================"

# WPS
python scripts/swopp3_npoints_comparison.py \
    --corridor pacific \
    --n-points 50 100 200 \
    --wind-path "${DATA}/era5_wind_pacific_2024.nc" \
    --wave-path "${DATA}/era5_waves_pacific_2024.nc" \
    --output-dir output/npoints_comparison

# noWPS
python scripts/swopp3_npoints_comparison.py \
    --corridor pacific \
    --n-points 50 100 200 \
    --no-wps \
    --wind-path "${DATA}/era5_wind_pacific_2024.nc" \
    --wave-path "${DATA}/era5_waves_pacific_2024.nc" \
    --output-dir output/npoints_comparison

# Copy results back to /home
HOME_OUTDIR="$HOME/routetools/output/npoints_comparison"
mkdir -p "$HOME_OUTDIR"
cp -rv output/npoints_comparison/* "$HOME_OUTDIR/"

echo ""
echo "N-points resample comparison completed at $(date)"
