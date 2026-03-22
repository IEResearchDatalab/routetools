#!/bin/bash
#SBATCH --job-name=swopp3_maxsweep
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm_maxsweep_%A_%a.out
#SBATCH --error=slurm_maxsweep_%A_%a.err
#SBATCH --array=0-19

# ── SWOPP3 penalty weight sweep (max-based penalties) ──
#
# 5 weights × 4 optimised cases = 20 array tasks, all parallel.
# GC cases skipped — deterministic, unaffected by penalty weights.
#
# Layout (task = weight_idx * 4 + case_idx):
#   weight_idx:  0=w5  1=w10  2=w25  3=w50  4=w100
#   case_idx:    0=AO_WPS  1=AO_noWPS  2=PO_WPS  3=PO_noWPS
#
# Submit:  sbatch scripts/swopp3_slurm_max_penalty_sweep.sh
# Monitor: squeue -u $USER

set -euo pipefail

# ── Per-axis configuration ──
WEIGHTS=(5.0 10.0 25.0 50.0 100.0)
WEIGHT_LABELS=(w5 w10 w25 w50 w100)

CASES=(AO_WPS AO_noWPS PO_WPS PO_noWPS)
NPOINTS=(178 178 293 293)
OCEANS=(atlantic atlantic pacific pacific)

# Decode task ID
WEIGHT_IDX=$(( SLURM_ARRAY_TASK_ID / 4 ))
CASE_IDX=$(( SLURM_ARRAY_TASK_ID % 4 ))

WIND_PW="${WEIGHTS[$WEIGHT_IDX]}"
WAVE_PW="${WEIGHTS[$WEIGHT_IDX]}"
DIST_PW=10.0
LABEL="${WEIGHT_LABELS[$WEIGHT_IDX]}"

CASE="${CASES[$CASE_IDX]}"
NP="${NPOINTS[$CASE_IDX]}"
OCN="${OCEANS[$CASE_IDX]}"

# ── Environment ──
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/routetools"
source .venv/bin/activate

export JAX_PLATFORMS=cpu
export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_cpu_multi_thread_eigen=true --xla_force_host_platform_device_count=${SLURM_CPUS_PER_TASK}"

# ── Paths ──
DATA="/scratch/fjsuarez/routetools/data/era5"
OUTDIR="output/swopp3_max_sweep_${LABEL}"
mkdir -p "$OUTDIR"

WIND_PATH="${DATA}/era5_wind_${OCN}_2024.nc"
WAVE_PATH="${DATA}/era5_waves_${OCN}_2024.nc"

echo "======================================"
echo "SWOPP3 max-penalty sweep: ${LABEL} / ${CASE}"
echo "Date:     $(date)"
echo "Host:     $(hostname)"
echo "Job:      ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "CPUs:     ${SLURM_CPUS_PER_TASK}"
echo "Weight:   ${WIND_PW}"
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
echo "=== Running ${CASE} (n_points=${NP}, w=${WIND_PW}) ==="
echo ""

python scripts/swopp3_run.py \
    --cases "$CASE" \
    --wind-path-${OCN} "$WIND_PATH" \
    --wave-path-${OCN} "$WAVE_PATH" \
    --output-dir "$OUTDIR" \
    --n-points "$NP" \
    --dt-eval-minutes 30 \
    --strategy optimised \
    --wind-penalty-weight "$WIND_PW" \
    --wave-penalty-weight "$WAVE_PW" \
    --distance-penalty-weight "$DIST_PW"

echo ""
echo "=== ${CASE} w=${WIND_PW} complete: $(date) ==="
echo ""
