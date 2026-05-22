#!/bin/bash
#SBATCH --job-name=swopp3_k15p400
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm_k15p400_%A_%a.out
#SBATCH --error=slurm_k15p400_%A_%a.err
#SBATCH --array=0-3

# ── SWOPP3 "kitchen sink": K=15, popsize=400, w=1000 ──
#
# 4 array tasks, one per optimised case (GC cases skipped — deterministic):
#   0=AO_WPS  1=AO_noWPS  (Atlantic, n=178)
#   2=PO_WPS  3=PO_noWPS  (Pacific,  n=293)
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

# ── Per-task configuration ──
CASES=(AO_WPS AO_noWPS PO_WPS PO_noWPS)
NPOINTS=(178 178 293 293)
# Atlantic tasks (0-1) use atlantic data, Pacific tasks (2-3) use pacific data
OCEAN=()
for i in 0 1; do OCEAN[$i]="atlantic"; done
for i in 2 3; do OCEAN[$i]="pacific"; done

CASE="${CASES[$SLURM_ARRAY_TASK_ID]}"
NP="${NPOINTS[$SLURM_ARRAY_TASK_ID]}"
OCN="${OCEAN[$SLURM_ARRAY_TASK_ID]}"
WIND_PATH="${DATA}/era5_wind_${OCN}_2024.nc"
WAVE_PATH="${DATA}/era5_waves_${OCN}_2024.nc"

echo "======================================"
echo "SWOPP3 kitchen sink: K=${K} pop=${POPSIZE} w=${WIND_PW}"
echo "Date:     $(date)"
echo "Host:     $(hostname)"
echo "Job:      ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "CPUs:     ${SLURM_CPUS_PER_TASK}"
echo "Case:     ${CASE}"
echo "Ocean:    ${OCN}"
echo "n_points: ${NP}"
echo "Output:   ${OUTDIR}"
echo "======================================"

# Verify data
for f in "$WIND_PATH" "$WAVE_PATH"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: Missing data file: $f" >&2
        exit 1
    fi
done
echo "All data files present."

python -c "import jax; print(f'JAX {jax.__version__}, devices: {jax.devices()}')"

echo ""
echo "=== Running ${CASE} (n_points=${NP}) ==="
echo ""

python scripts/swopp3_run.py \
    --cases "$CASE" \
    --wind-path-${OCN} "$WIND_PATH" \
    --wave-path-${OCN} "$WAVE_PATH" \
    --output-dir "$OUTDIR" \
    --n-points "$NP" \
    --dt-eval-minutes "$DT_EVAL" \
    --cmaes-k "$K" \
    --popsize "$POPSIZE" \
    --maxfevals "$MAXFEVALS" \
    --strategy optimised \
    --cmaes-verbose \
    --wind-penalty-weight "$WIND_PW" \
    --wave-penalty-weight "$WAVE_PW" \
    --distance-penalty-weight "$DIST_PW"

echo ""
echo "=== ${CASE} complete: $(date) ==="
echo ""
