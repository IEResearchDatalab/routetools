#!/bin/bash
#SBATCH --job-name=swopp3_pac
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# ── SWOPP3 Pacific corridor (hourly ERA5, GPU) ──
#
# n-points=584 gives dt = 583h / 583 ≈ 1.0h per segment,
# matching the hourly ERA5 temporal resolution.

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/routetools"
source .venv/bin/activate

export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

DATA="data/era5"
OUTDIR="output/swopp3_gpu"
mkdir -p "$OUTDIR"

echo "======================================"
echo "SWOPP3 Pacific on $(hostname)"
echo "Date:     $(date)"
echo "GPU:      $(nvidia-smi -L 2>/dev/null | head -1 || echo 'unknown')"
echo "n-points: 584  (dt ≈ 1.0h)"
echo "Output:   ${OUTDIR}"
echo "======================================"

python scripts/swopp3_run.py \
    --cases PO_WPS --cases PO_noWPS --cases PGC_WPS --cases PGC_noWPS \
    --wind-path-pacific "${DATA}/era5_wind_pacific_2024.nc" \
    --wave-path-pacific "${DATA}/era5_waves_pacific_2024.nc" \
    --output-dir "$OUTDIR" \
    --n-points 584

echo ""
echo "Pacific completed at $(date)"
