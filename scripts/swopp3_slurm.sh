#!/bin/bash
#SBATCH --job-name=swopp3_1h
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# ── SWOPP3 full run on rust-HPC (hourly ERA5 data, CPU mode) ──
#
# Submit:  sbatch scripts/swopp3_slurm.sh
# Monitor: squeue -u $USER
# Cancel:  scancel <jobid>

set -euo pipefail

# ── Environment ──
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/routetools"
source .venv/bin/activate

# Force JAX to use CPU (avoid GPU OOM with large 0.125° grids)
export JAX_PLATFORMS=cpu

# Use all allocated CPUs for XLA parallelism (preserve existing XLA_FLAGS)
export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_cpu_multi_thread_eigen=true --xla_force_host_platform_device_count=${SLURM_CPUS_PER_TASK}"

# ── Paths ──
# Hourly ERA5 data (download with: uv run scripts/download_era5.py --output-dir data/era5_1h)
DATA="data/era5_1h"
OUTDIR="output/swopp3_1h_cpu"

mkdir -p "$OUTDIR"

echo "======================================"
echo "SWOPP3 run on $(hostname)"
echo "Date:     $(date)"
echo "CPUs:     ${SLURM_CPUS_PER_TASK}"
echo "Memory:   ${SLURM_MEM_PER_NODE}MB"
echo "JAX:      CPU mode"
echo "Data:     hourly ERA5"
echo "Output:   ${OUTDIR}"
echo "======================================"

# Verify data is present
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

# Quick JAX sanity check
python -c "import jax; print(f'JAX {jax.__version__}, devices: {jax.devices()}')"

# ── Run all 8 cases with 0.125° data ──
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
echo "SWOPP3 run completed at $(date)"
echo "======================================"

# ── Summary ──
echo ""
echo "Output files:"
ls -lh "$OUTDIR"/*.csv 2>/dev/null || echo "(no CSV files found)"
echo ""
echo "Done."
