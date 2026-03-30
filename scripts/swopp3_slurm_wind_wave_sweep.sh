#!/bin/bash
#SBATCH --job-name=swopp3_ww
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm_ww_%A_%a.out
#SBATCH --error=slurm_ww_%A_%a.err
#SBATCH --array=0-35

# ── SWOPP3 2-D wind × wave penalty weight sweep ──
#
# Explores a 6×6 grid of (wind_penalty_weight, wave_penalty_weight) with
# fixed CMA-ES parameters (K=15, σ₀=0.3, popsize=200, dt_eval=30 min).
# Each SLURM array task handles one (wind, wave) combination and runs
# both Atlantic and Pacific corridors.
#
# Grid:  wind_pw ∈ {0, 50, 100, 200, 500, 1000}
#        wave_pw ∈ {0, 50, 100, 200, 500, 1000}
#        = 36 combinations (array 0-35)
#
# Submit:  sbatch scripts/swopp3_slurm_wind_wave_sweep.sh
# Monitor: squeue -u $USER
# Cancel:  scancel <jobid>

set -euo pipefail

# ── 2-D grid definition ──
WIND_PWS=(0.0 50.0 100.0 200.0 500.0 1000.0)
WAVE_PWS=(0.0 50.0 100.0 200.0 500.0 1000.0)

N_WAVE=${#WAVE_PWS[@]}
WIND_IDX=$(( SLURM_ARRAY_TASK_ID / N_WAVE ))
WAVE_IDX=$(( SLURM_ARRAY_TASK_ID % N_WAVE ))

WIND_PW=${WIND_PWS[$WIND_IDX]}
WAVE_PW=${WAVE_PWS[$WAVE_IDX]}
LABEL="ww_w${WIND_PW%.*}_v${WAVE_PW%.*}"

# ── Fixed CMA-ES parameters ──
K=15
SIGMA0=0.3
DIST_PW=10.0
DT_EVAL=30   # Δt₂ = 30 min evaluation grid

# ── Environment ──
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/routetools"
source .venv/bin/activate

export JAX_PLATFORMS=cpu
export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_cpu_multi_thread_eigen=true --xla_force_host_platform_device_count=${SLURM_CPUS_PER_TASK}"

# ── Paths ──
DATA="/data/fjsuarez/era5"
OUTDIR="output/swopp3_${LABEL}"
mkdir -p "$OUTDIR"

echo "======================================"
echo "SWOPP3 wind×wave sweep: ${LABEL}"
echo "Date:     $(date)"
echo "Host:     $(hostname)"
echo "Job:      ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "CPUs:     ${SLURM_CPUS_PER_TASK}"
echo "Output:   ${OUTDIR}"
echo "K:        ${K}"
echo "σ₀:       ${SIGMA0}"
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
    --sigma0 "$SIGMA0" \
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
    --sigma0 "$SIGMA0" \
    --wind-penalty-weight "$WIND_PW" \
    --wave-penalty-weight "$WAVE_PW" \
    --distance-penalty-weight "$DIST_PW"

echo ""
echo "======================================"
echo "SWOPP3 ${LABEL} complete: $(date)"
echo "======================================"
echo ""
echo "Output files:"
ls -lh "$OUTDIR"/*.csv 2>/dev/null || echo "(no CSV files found)"
