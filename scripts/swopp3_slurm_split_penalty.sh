#!/bin/bash
#SBATCH --job-name=swopp3_split
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm_split_%j.out
#SBATCH --error=slurm_split_%j.err

# ── SWOPP3 full run: split wind+wave penalties (Δt₁ = 2 h) ──
#
# Uses separate --wind-penalty-weight and --wave-penalty-weight instead
# of the old combined --weather-penalty-weight.
# n-points per corridor: Atlantic=178 (354h/2h+1), Pacific=293 (583h/2h+1).
#
# Submit:  sbatch scripts/swopp3_slurm_split_penalty.sh
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
OUTDIR="output/swopp3_split_penalty"
mkdir -p "$OUTDIR"

# ── Penalty configuration ──
WIND_PW=100.0
WAVE_PW=100.0
DIST_PW=10.0
DT_EVAL=30   # Δt₂ = 30 min evaluation grid

echo "======================================"
echo "SWOPP3 split-penalty run on $(hostname)"
echo "Date:     $(date)"
echo "CPUs:     ${SLURM_CPUS_PER_TASK}"
echo "JAX:      CPU mode"
echo "Data:     ${DATA}"
echo "Output:   ${OUTDIR}"
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
    --wind-penalty-weight "$WIND_PW" \
    --wave-penalty-weight "$WAVE_PW" \
    --distance-penalty-weight "$DIST_PW"

echo ""
echo "======================================"
echo "SWOPP3 split-penalty run completed at $(date)"
echo "======================================"
echo ""
echo "Output files:"
ls -lh "$OUTDIR"/*.csv 2>/dev/null || echo "(no CSV files found)"
