#!/bin/bash
#SBATCH --job-name=swopp3_atl
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# ── SWOPP3 Atlantic corridor (hourly ERA5, GPU) ──
#
# n-points=355 gives dt = 354h / 354 ≈ 1.0h per segment,
# matching the hourly ERA5 temporal resolution.

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"

# Work from local SSD (staged by swopp3_slurm_stage.sh)
SCRATCH="/scratch/${USER}/routetools"
cd "$SCRATCH"
source .venv/bin/activate

export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

DATA="data/era5"
OUTDIR="output/swopp3_gpu"
mkdir -p "$OUTDIR"

echo "======================================"
echo "SWOPP3 Atlantic on $(hostname)"
echo "Date:     $(date)"
echo "GPU:      $(nvidia-smi -L 2>/dev/null | head -1 || echo 'unknown')"
echo "n-points: 355  (dt ≈ 1.0h)"
echo "Workdir:  $SCRATCH"
echo "Output:   ${OUTDIR}"
echo "======================================"

python scripts/swopp3_run.py \
    --cases AO_WPS --cases AO_noWPS --cases AGC_WPS --cases AGC_noWPS \
    --wind-path-atlantic "${DATA}/era5_wind_atlantic_2024.nc" \
    --wave-path-atlantic "${DATA}/era5_waves_atlantic_2024.nc" \
    --output-dir "$OUTDIR" \
    --n-points 355

# ── Copy results back to /home ──
HOME_OUTDIR="$HOME/routetools/output/swopp3_gpu"
mkdir -p "$HOME_OUTDIR"
cp -v "$OUTDIR"/* "$HOME_OUTDIR/"

echo ""
echo "Atlantic completed at $(date)"
