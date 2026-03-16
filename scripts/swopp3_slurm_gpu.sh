#!/bin/bash
#SBATCH --job-name=swopp3_gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# ── SWOPP3 full run on rust-HPC (hourly ERA5 data, GPU mode) ──
#
# This variant uses a single RTX 6000 Ada (48 GB).
# Key tuning:
#   - XLA_PYTHON_CLIENT_PREALLOCATE=false: don't pre-reserve GPU memory,
#     allow on-demand allocation so compilation peaks don't OOM.
#   - XLA_PYTHON_CLIENT_MEM_FRACTION=0.95: allow up to 95% of GPU memory.
#
# Submit:  sbatch scripts/swopp3_slurm_gpu.sh
# Monitor: squeue -u $USER

set -euo pipefail

# ── Environment ──
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/routetools"
source .venv/bin/activate

# GPU memory management: don't preallocate, allow peak usage
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

# CPU threads for data loading/preprocessing
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# ── Paths ──
# Hourly ERA5 data (download with: uv run scripts/download_era5.py --output-dir data/era5_1h)
DATA="data/era5_1h"
OUTDIR="output/swopp3_1h_gpu"

mkdir -p "$OUTDIR"

echo "======================================"
echo "SWOPP3 GPU run on $(hostname)"
echo "Date:     $(date)"
echo "CPUs:     ${SLURM_CPUS_PER_TASK}"
echo "GPU:      $(nvidia-smi -L 2>/dev/null | head -1 || echo 'unknown')"
echo "JAX:      CUDA mode"
echo "Data:     hourly ERA5"
echo "Output:   ${OUTDIR}"
echo "======================================"

# Verify data
for f in \
    "${DATA}/era5_wind_atlantic_2024.nc" \
    "${DATA}/era5_waves_atlantic_2024.nc" \
    "${DATA}/era5_wind_pacific_2024.nc" \
    "${DATA}/era5_waves_pacific_2024.nc"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: Missing data file: $f" >&2
        exit 1
    fi
done
echo "All data files present."

# JAX sanity check + GPU info
python -c "
import jax
print(f'JAX {jax.__version__}, devices: {jax.devices()}')
for d in jax.devices():
    if hasattr(d, 'memory_stats'):
        stats = d.memory_stats()
        if stats:
            total = stats.get('bytes_limit', 0) / 1e9
            print(f'  {d}: {total:.1f} GB total')
"

echo ""
echo "Starting SWOPP3 run at $(date)"
echo ""

python scripts/swopp3_run.py \
    --wind-path-atlantic "${DATA}/era5_wind_atlantic_2024.nc" \
    --wave-path-atlantic "${DATA}/era5_waves_atlantic_2024.nc" \
    --wind-path-pacific  "${DATA}/era5_wind_pacific_2024.nc"  \
    --wave-path-pacific  "${DATA}/era5_waves_pacific_2024.nc"  \
    --output-dir "$OUTDIR"

echo ""
echo "======================================"
echo "SWOPP3 GPU run completed at $(date)"
echo "======================================"
echo ""
echo "Output files:"
ls -lh "$OUTDIR"/*.csv 2>/dev/null || echo "(no CSV files found)"
