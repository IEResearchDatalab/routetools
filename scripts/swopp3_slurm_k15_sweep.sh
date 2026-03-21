#!/bin/bash
#SBATCH --job-name=swopp3_k15
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm_k15_%A_%a.out
#SBATCH --error=slurm_k15_%A_%a.err
#SBATCH --array=0-1

# ── SWOPP3 K=15 experiment (normalized mean penalties) ──
#
# Tests whether increasing K from 10 to 15 lets the optimizer better
# avoid weather with smooth penalties.  Two weights: w200 and w500.
#
# Submit:  sbatch scripts/swopp3_slurm_k15_sweep.sh
# Monitor: squeue -u $USER

set -euo pipefail

# ── Sweep configurations ──
CONFIGS=(
    "200.0 200.0 10.0 k15_w200"
    "500.0 500.0 10.0 k15_w500"
)

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
read -r WIND_PW WAVE_PW DIST_PW LABEL <<< "$CONFIG"

# ── Environment ──
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/routetools"
source .venv/bin/activate

export JAX_PLATFORMS=cpu
export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_cpu_multi_thread_eigen=true --xla_force_host_platform_device_count=${SLURM_CPUS_PER_TASK}"

# ── Paths ──
DATA="/scratch/fjsuarez/routetools/data/era5"
OUTDIR="output/swopp3_${LABEL}"
mkdir -p "$OUTDIR"

K=15
DT_EVAL=30  # Δt₂ = 30 min evaluation grid

echo "======================================"
echo "SWOPP3 K=${K} experiment: ${LABEL}"
echo "Date:     $(date)"
echo "Host:     $(hostname)"
echo "Job:      ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "CPUs:     ${SLURM_CPUS_PER_TASK}"
echo "Output:   ${OUTDIR}"
echo "K:        ${K}"
echo "Wind PW:  ${WIND_PW}"
echo "Wave PW:  ${WAVE_PW}"
echo "Dist PW:  ${DIST_PW}"
echo "dt_eval:  ${DT_EVAL} min"
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

python -c "import jax; print(f'JAX {jax.__version__}, devices: {jax.devices()}')"

# ── Atlantic cases (354 h → n_points=178 for Δt₁=2 h) ──
echo ""
echo "=== Atlantic cases (n_points=178) ==="
echo ""

python scripts/swopp3_run.py \
    --cases AO_WPS --cases AO_noWPS --cases AGC_WPS --cases AGC_noWPS \
    --wind-path-atlantic "${DATA}/era5_wind_atlantic_2024.nc" \
    --wave-path-atlantic "${DATA}/era5_waves_atlantic_2024.nc" \
    --output-dir "$OUTDIR" \
    --n-points 178 \
    --dt-eval-minutes "$DT_EVAL" \
    --cmaes-k "$K" \
    --wind-penalty-weight "$WIND_PW" \
    --wave-penalty-weight "$WAVE_PW" \
    --distance-penalty-weight "$DIST_PW"

# ── Pacific cases (583 h → n_points=293 for Δt₁=2 h) ──
echo ""
echo "=== Pacific cases (n_points=293) ==="
echo ""

python scripts/swopp3_run.py \
    --cases PO_WPS --cases PO_noWPS --cases PGC_WPS --cases PGC_noWPS \
    --wind-path-pacific "${DATA}/era5_wind_pacific_2024.nc" \
    --wave-path-pacific "${DATA}/era5_waves_pacific_2024.nc" \
    --output-dir "$OUTDIR" \
    --n-points 293 \
    --dt-eval-minutes "$DT_EVAL" \
    --cmaes-k "$K" \
    --wind-penalty-weight "$WIND_PW" \
    --wave-penalty-weight "$WAVE_PW" \
    --distance-penalty-weight "$DIST_PW"

echo ""
echo "=== K=${K} ${LABEL} complete: $(date) ==="
echo ""
