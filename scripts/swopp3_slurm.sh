#!/bin/bash
#SBATCH --job-name=swopp3_0125
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=output/swopp3_0125_rust/slurm_%j.out
#SBATCH --error=output/swopp3_0125_rust/slurm_%j.err

# NOTE: The SLURM log directory (output/swopp3_0125_rust/) must exist
# before submission.  Run: mkdir -p output/swopp3_0125_rust

# ── SWOPP3 full run on rust-HPC (0.125° ERA5 data, CPU mode) ──
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

# Use all allocated CPUs for XLA parallelism (append to preserve existing flags)
export XLA_FLAGS="${XLA_FLAGS:-} --xla_cpu_multi_thread_eigen=true --xla_force_host_platform_device_count=${SLURM_CPUS_PER_TASK}"

# ── Paths ──
DATA_025="data/era5"
DATA_0125="data/era5_0125"
OUTDIR="output/swopp3_0125_rust"

mkdir -p "$OUTDIR"

echo "======================================"
echo "SWOPP3 run on $(hostname)"
echo "Date:     $(date)"
echo "CPUs:     ${SLURM_CPUS_PER_TASK}"
echo "Memory:   ${SLURM_MEM_PER_NODE}MB"
echo "JAX:      CPU mode"
echo "Data:     0.125°"
echo "Output:   ${OUTDIR}"
echo "======================================"

# Verify data is present
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

# Quick JAX sanity check
python -c "import jax; print(f'JAX {jax.__version__}, devices: {jax.devices()}')"

# ── Run all 8 cases with 0.125° data ──
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
echo "SWOPP3 run completed at $(date)"
echo "======================================"

# ── Summary ──
echo ""
echo "Output files:"
ls -lh "$OUTDIR"/*.csv 2>/dev/null || echo "(no CSV files found)"
echo ""
echo "Done."
