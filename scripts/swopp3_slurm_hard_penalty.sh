#!/bin/bash
#SBATCH --job-name=swopp3_hard
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm_hard_%j.out
#SBATCH --error=slurm_hard_%j.err

# ── SWOPP3 full run: hard (step) weather penalty ──
#
# Uses the combined --weather-penalty-weight which counts boolean
# violations (TWS > limit or Hs > limit) per segment and multiplies
# by the weight.  No smooth ramp — pure count × weight.
#
# Submit:  sbatch scripts/swopp3_slurm_hard_penalty.sh
# Monitor: squeue -u $USER

set -euo pipefail

# ── Environment ──
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/routetools"
source .venv/bin/activate

export JAX_PLATFORMS=cpu
export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_cpu_multi_thread_eigen=true --xla_force_host_platform_device_count=${SLURM_CPUS_PER_TASK}"

# ── Paths ──
DATA="/scratch/fjsuarez/routetools/data/era5"
OUTDIR="output/swopp3_hard_penalty"
mkdir -p "$OUTDIR"

# ── Penalty configuration ──
# Hard penalty: count of violating segments × weight
# With ~177 segments (Atlantic) or ~292 (Pacific), a single violation
# adds WEATHER_PW to the cost.  Energy is ~100-300 MWh, so weight=10
# means each violating segment adds 10 to cost.
WEATHER_PW=10.0
DIST_PW=10.0
DT_EVAL=30  # Δt₂ = 30 min evaluation grid

echo "======================================"
echo "SWOPP3 hard-penalty run on $(hostname)"
echo "Date:       $(date)"
echo "CPUs:       ${SLURM_CPUS_PER_TASK}"
echo "JAX:        CPU mode"
echo "Data:       ${DATA}"
echo "Output:     ${OUTDIR}"
echo "Weather PW: ${WEATHER_PW} (hard count)"
echo "Dist PW:    ${DIST_PW}"
echo "dt_eval:    ${DT_EVAL} min"
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
    --weather-penalty-weight "$WEATHER_PW" \
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
    --weather-penalty-weight "$WEATHER_PW" \
    --distance-penalty-weight "$DIST_PW"

echo ""
echo "=== Hard-penalty run complete: $(date) ==="
echo ""
