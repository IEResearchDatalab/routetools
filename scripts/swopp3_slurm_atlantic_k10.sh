#!/bin/bash
#SBATCH --job-name=swopp3_atl_k10
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_atl_k10_%j.out
#SBATCH --error=slurm_atl_k10_%j.err

# ── SWOPP3 Atlantic-only run: K=10, popsize=400, maxfevals=50000 ──
#
# Experiment to test whether more CMA-ES budget / smaller K resolves
# the pathological energy spikes observed in the Atlantic corridor with
# wind×wave penalties (w=50, v=50).
#
# Submit:  sbatch scripts/swopp3_slurm_atlantic_k10.sh
# Monitor: squeue -u $USER

set -euo pipefail

# ── Parameters ──
K=10
SIGMA0=0.3
POPSIZE=400
MAXFEVALS=50000
WIND_PW=50.0
WAVE_PW=50.0
DIST_PW=10.0
DT_EVAL=30   # Δt₂ = 30 min evaluation grid

LABEL="atlantic_k10_p400_w50_v50"

# ── Environment ──
ROOTDIR="/home/fjsuarez/routetools"
export PATH="/home/fjsuarez/.local/bin:$PATH"
cd "$ROOTDIR"
source .venv/bin/activate

export JAX_PLATFORMS=cpu
export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_cpu_multi_thread_eigen=true --xla_force_host_platform_device_count=${SLURM_CPUS_PER_TASK}"

# ── Paths ──
DATA="/data/fjsuarez/era5"
OUTDIR="output/swopp3_${LABEL}"
mkdir -p "$OUTDIR"

echo "======================================"
echo "SWOPP3 Atlantic K=10 experiment"
echo "Date:      $(date)"
echo "Host:      $(hostname)"
echo "Job:       ${SLURM_JOB_ID}"
echo "CPUs:      ${SLURM_CPUS_PER_TASK}"
echo "Output:    ${OUTDIR}"
echo "K:         ${K}"
echo "σ₀:        ${SIGMA0}"
echo "popsize:   ${POPSIZE}"
echo "maxfevals: ${MAXFEVALS}"
echo "Wind PW:   ${WIND_PW}"
echo "Wave PW:   ${WAVE_PW}"
echo "Dist PW:   ${DIST_PW}"
echo "dt_eval:   ${DT_EVAL} min"
echo "======================================"

# Verify data
for f in \
    "${DATA}/era5_wind_atlantic_2024.nc" \
    "${DATA}/era5_waves_atlantic_2024.nc"; do
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
    --popsize "$POPSIZE" \
    --maxfevals "$MAXFEVALS" \
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
