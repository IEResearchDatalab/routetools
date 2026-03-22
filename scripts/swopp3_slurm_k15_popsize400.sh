#!/bin/bash
#SBATCH --job-name=swopp3_k15p400
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm_k15p400_%A_%a.out
#SBATCH --error=slurm_k15p400_%A_%a.err
#SBATCH --array=0-1

# ── SWOPP3 "kitchen sink": K=15, popsize=400, w=1000 ──
#
# Array task 0 = Atlantic cases, task 1 = Pacific cases (parallel).
#
# Submit:  sbatch scripts/swopp3_slurm_k15_popsize400.sh
# Monitor: squeue -u $USER

set -euo pipefail

# ── Environment ──
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/routetools"
source .venv/bin/activate

export JAX_PLATFORMS=cpu
export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_cpu_multi_thread_eigen=true --xla_force_host_platform_device_count=${SLURM_CPUS_PER_TASK}"

# ── Parameters ──
K=15
POPSIZE=400
WIND_PW=1000.0
WAVE_PW=1000.0
DIST_PW=10.0
MAXFEVALS=50000
DT_EVAL=30

DATA="/scratch/fjsuarez/routetools/data/era5"
OUTDIR="output/swopp3_k15_p400_w1000"
mkdir -p "$OUTDIR"

echo "======================================"
echo "SWOPP3 kitchen sink: K=${K} pop=${POPSIZE} w=${WIND_PW}"
echo "Date:     $(date)"
echo "Host:     $(hostname)"
echo "Job:      ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "CPUs:     ${SLURM_CPUS_PER_TASK}"
echo "Output:   ${OUTDIR}"
echo "K:        ${K}"
echo "Popsize:  ${POPSIZE}"
echo "Wind PW:  ${WIND_PW}"
echo "Wave PW:  ${WAVE_PW}"
echo "Dist PW:  ${DIST_PW}"
echo "MaxFEval: ${MAXFEVALS}"
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

if [[ "${SLURM_ARRAY_TASK_ID}" == "0" ]]; then
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
        --popsize "$POPSIZE" \
        --maxfevals "$MAXFEVALS" \
        --cmaes-verbose \
        --wind-penalty-weight "$WIND_PW" \
        --wave-penalty-weight "$WAVE_PW" \
        --distance-penalty-weight "$DIST_PW"

    echo ""
    echo "=== Atlantic complete: $(date) ==="

else
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
        --popsize "$POPSIZE" \
        --maxfevals "$MAXFEVALS" \
        --cmaes-verbose \
        --wind-penalty-weight "$WIND_PW" \
        --wave-penalty-weight "$WAVE_PW" \
        --distance-penalty-weight "$DIST_PW"

    echo ""
    echo "=== Pacific complete: $(date) ==="
fi

echo ""
echo "=== K=${K} pop=${POPSIZE} w=${WIND_PW} task ${SLURM_ARRAY_TASK_ID} done: $(date) ==="
echo ""
