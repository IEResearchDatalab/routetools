#!/bin/bash
#SBATCH --job-name=swopp3_gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output=output/swopp3_0125_gpu/slurm_%j.out
#SBATCH --error=output/swopp3_0125_gpu/slurm_%j.err

# NOTE: The SLURM log directory (output/swopp3_0125_gpu/) must exist
# before submission.  Run: mkdir -p output/swopp3_0125_gpu

# ── SWOPP3 full run on rust-HPC (0.125° ERA5 data, GPU mode) ──
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
DATA_0125="data/era5_0125"
OUTDIR="output/swopp3_0125_gpu"

mkdir -p "$OUTDIR"

echo "======================================"
echo "SWOPP3 GPU run on $(hostname)"
echo "Date:     $(date)"
echo "CPUs:     ${SLURM_CPUS_PER_TASK}"
echo "GPU:      $(nvidia-smi -L 2>/dev/null | head -1 || echo 'unknown')"
echo "JAX:      CUDA mode"
echo "Data:     0.125°"
echo "Output:   ${OUTDIR}"
echo "======================================"

# Verify data
for f in \
    "${DATA_0125}/era5_wind_atlantic_2024.nc" \
    "${DATA_0125}/era5_waves_atlantic_2024.nc" \
    "${DATA_0125}/era5_wind_pacific_2024.nc" \
    "${DATA_0125}/era5_waves_pacific_2024.nc"; do
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
    --wind-path-atlantic "${DATA_0125}/era5_wind_atlantic_2024.nc" \
    --wave-path-atlantic "${DATA_0125}/era5_waves_atlantic_2024.nc" \
    --wind-path-pacific  "${DATA_0125}/era5_wind_pacific_2024.nc"  \
    --wave-path-pacific  "${DATA_0125}/era5_waves_pacific_2024.nc"  \
    --output-dir "$OUTDIR"

echo ""
echo "======================================"
echo "SWOPP3 GPU run completed at $(date)"
echo "======================================"
echo ""
echo "Output files:"
ls -lh "$OUTDIR"/*.csv 2>/dev/null || echo "(no CSV files found)"
