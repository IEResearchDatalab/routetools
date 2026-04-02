#!/bin/bash
#SBATCH --job-name=swopp3_pac_k15
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_pac_k15_%j.out
#SBATCH --error=slurm_pac_k15_%j.err

# ── SWOPP3 Pacific-only run: K=15, popsize=400, maxfevals=50000 ──
#
# Experiment to confirm w=50, v=200 is the best penalty config for
# Pacific, using double the CMA-ES budget (50k fevals, popsize=400).
#
# Submit:  sbatch scripts/swopp3_slurm_pacific_k15_p400.sh
# Monitor: squeue -u $USER

set -euo pipefail

# ── Parameters ──
K=15
SIGMA0=0.3
POPSIZE=400
MAXFEVALS=50000
WIND_PW=50.0
WAVE_PW=200.0
DIST_PW=10.0
DT_EVAL=30   # Δt₂ = 30 min evaluation grid

LABEL="pacific_k15_p400_w50_v200"

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
echo "SWOPP3 Pacific K=15 p=400 experiment"
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
    "${DATA}/era5_wind_pacific_2024.nc" \
    "${DATA}/era5_waves_pacific_2024.nc"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: Missing data file: $f" >&2
        exit 1
    fi
done
echo "All data files present."

python -c "import jax; print(f'JAX {jax.__version__}, devices: {jax.devices()}')"

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
