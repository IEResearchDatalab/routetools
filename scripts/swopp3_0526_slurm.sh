#!/bin/bash
#SBATCH --job-name=swopp3_0526
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=output/logs/slurm_swopp3_0526_%j.out
#SBATCH --error=output/logs/slurm_swopp3_0526_%j.err

# Complete 0526 pipeline under SLURM (single job, sequential steps).
#
# This mirrors scripts/run_swopp3_0526.sh but runs inside one isolated SLURM job
# to avoid accidental overlaps.
#
# Submit:
#   sbatch scripts/swopp3_0526_slurm.sh
#
# Monitor:
#   squeue -u "$USER"
#   tail -f output/logs/slurm_swopp3_0526_<JOBID>.out
#
# Cancel:
#   scancel <JOBID>

set -euo pipefail

ROOTDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOTDIR"

mkdir -p output/logs

# Optional: activate project venv if it exists.
if [[ -f ".venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

# ERA5 data paths
WIND_ATL="data/era5/era5_wind_atlantic_2024.nc"
WAVE_ATL="data/era5/era5_waves_atlantic_2024.nc"
WIND_PAC="data/era5/era5_wind_pacific_2024.nc"
WAVE_PAC="data/era5/era5_waves_pacific_2024.nc"

# 0526 CMA-ES parameters
CMAES_K=10
SIGMA0=0.5
POPSIZE=200
MAXFEVALS=25000
DT_EVAL=30
DATALOAD_LIMIT=2

# Fixed penalty configuration (single run)
WIND_PENALTY=50.0
WAVE_PENALTY=50.0
DISTANCE_PENALTY=10.0

# Basic data validation
for f in "$WIND_ATL" "$WAVE_ATL" "$WIND_PAC" "$WAVE_PAC"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: Missing data file: $f" >&2
        exit 1
    fi
done

if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: 'uv' not found in PATH. Activate your environment first." >&2
    exit 1
fi

# Use CPU by default for stability with large constant captures in JAX.
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
if [[ "${JAX_PLATFORMS}" == "cpu" ]]; then
    export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_cpu_multi_thread_eigen=true --xla_force_host_platform_device_count=${SLURM_CPUS_PER_TASK:-32}"
fi
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

echo "============================================"
echo "0526 SLURM pipeline started"
echo "Date:         $(date)"
echo "Host:         $(hostname)"
echo "Job ID:       ${SLURM_JOB_ID:-n/a}"
echo "Partition:    ${SLURM_JOB_PARTITION:-n/a}"
echo "CPUs:         ${SLURM_CPUS_PER_TASK:-n/a}"
echo "JAX platform: ${JAX_PLATFORMS}"
echo "CMAES_K:      $CMAES_K"
echo "SIGMA0:       $SIGMA0"
echo "POPSIZE:      $POPSIZE"
echo "MAXFEVALS:    $MAXFEVALS"
echo "DT_EVAL:      ${DT_EVAL} min"
echo "DATALOAD_LIMIT: $DATALOAD_LIMIT"
echo "WIND_PENALTY: $WIND_PENALTY"
echo "WAVE_PENALTY: $WAVE_PENALTY"
echo "DIST_PENALTY: $DISTANCE_PENALTY"
echo "============================================"

SWEEP_OUT="output/swopp3_0526"
FMS_OUT="${SWEEP_OUT}_fms"

echo ""
echo "============================================"
echo "Config:      fixed"
echo "Sweep out:   $SWEEP_OUT"
echo "FMS out:     $FMS_OUT"
echo "Wind PW:     $WIND_PENALTY"
echo "Wave PW:     $WAVE_PENALTY"
echo "Dist PW:     $DISTANCE_PENALTY"
echo "============================================"

mkdir -p "$SWEEP_OUT" "$FMS_OUT"

# Pacific cases (583 h -> n_points=293 for dt1=2 h)
uv run scripts/swopp3_run.py \
    --cases PO_WPS --cases PO_noWPS --cases PGC_WPS --cases PGC_noWPS \
    --wind-path-pacific "$WIND_PAC" \
    --wave-path-pacific "$WAVE_PAC" \
    --output-dir "$SWEEP_OUT" \
    --n-points 293 \
    --dt-eval-minutes "$DT_EVAL" \
    --cmaes-k "$CMAES_K" \
    --sigma0 "$SIGMA0" \
    --popsize "$POPSIZE" \
    --maxfevals "$MAXFEVALS" \
    --dataload-limit "$DATALOAD_LIMIT" \
    --wind-penalty-weight "$WIND_PENALTY" \
    --wave-penalty-weight "$WAVE_PENALTY" \
    --distance-penalty-weight "$DISTANCE_PENALTY"

# Atlantic cases (354 h -> n_points=178 for dt1=2 h)
uv run scripts/swopp3_run.py \
    --cases AO_WPS --cases AO_noWPS --cases AGC_WPS --cases AGC_noWPS \
    --wind-path-atlantic "$WIND_ATL" \
    --wave-path-atlantic "$WAVE_ATL" \
    --output-dir "$SWEEP_OUT" \
    --n-points 178 \
    --dt-eval-minutes "$DT_EVAL" \
    --cmaes-k "$CMAES_K" \
    --sigma0 "$SIGMA0" \
    --popsize "$POPSIZE" \
    --maxfevals "$MAXFEVALS" \
    --dataload-limit "$DATALOAD_LIMIT" \
    --wind-penalty-weight "$WIND_PENALTY" \
    --wave-penalty-weight "$WAVE_PENALTY" \
    --distance-penalty-weight "$DISTANCE_PENALTY"

# FMS refinement over full sweep output for this config
uv run scripts/swopp3_apply_fms.py \
    "$SWEEP_OUT" \
    --output-dir "$FMS_OUT" \
    --wind-path-atlantic "$WIND_ATL" \
    --wave-path-atlantic "$WAVE_ATL" \
    --wind-path-pacific "$WIND_PAC" \
    --wave-path-pacific "$WAVE_PAC" \
    --wind-penalty-weight "$WIND_PENALTY" \
    --wave-penalty-weight "$WAVE_PENALTY"

echo "Completed fixed config at $(date)"

echo ""
echo "============================================"
echo "0526 SLURM pipeline finished: $(date)"
echo "============================================"
