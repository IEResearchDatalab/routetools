#!/usr/bin/env bash
# Run the final BERS revision analyses without rerunning any routes.
#
# Usage:
#   bash revision/run_revision_postprocessing.sh RUN_ROOT [ANALYSIS_DIR]
#
# RUN_ROOT must contain the completed server run's `bers/` and `cmaes/`
# directories. ANALYSIS_DIR defaults to RUN_ROOT/revision_analysis.

set -Eeuo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 RUN_ROOT [ANALYSIS_DIR]" >&2
    exit 2
fi

script_path="$(realpath "${BASH_SOURCE[0]}")"
repo_root="$(cd "$(dirname "$script_path")/.." && pwd)"
run_root="$(realpath "$1")"
analysis_dir="$(realpath -m "${2:-$run_root/revision_analysis}")"
bers_dir="$run_root/bers"
cmaes_dir="$run_root/cmaes"

for required in "$bers_dir" "$cmaes_dir" "$bers_dir/experiment_manifest.json"; do
    if [[ ! -e "$required" ]]; then
        echo "Missing required final-run artifact: $required" >&2
        exit 1
    fi
done
if [[ ! -f "$run_root/COMPLETED" ]]; then
    echo "Final-run marker is absent: $run_root/COMPLETED" >&2
    exit 1
fi

mkdir -p "$analysis_dir"
log_path="$analysis_dir/revision_postprocessing.log"
exec > >(tee -a "$log_path") 2>&1

failed_marker="$analysis_dir/FAILED"
completed_marker="$analysis_dir/COMPLETED"
rm -f "$failed_marker" "$completed_marker"
trap 'status=$?; if [[ $status -ne 0 ]]; then date -Is > "$failed_marker"; fi' EXIT

cd "$repo_root"
export JAX_PLATFORMS=cpu
export PYTHONUNBUFFERED=1

printf '%s\n' \
    "BERS final revision post-processing" \
    "Started: $(date -Is)" \
    "Host: $(hostname)" \
    "Repository: $repo_root" \
    "Git commit: $(git rev-parse HEAD)" \
    "Final run: $run_root" \
    "Analysis output: $analysis_dir" \
    "JAX platform: $JAX_PLATFORMS" \
    > "$analysis_dir/provenance.txt"
cp "$script_path" "$analysis_dir/run_revision_postprocessing.sh"

echo "[1/5] Segment-speed distribution"
uv run python -u revision/task1_speed_distribution.py \
    --real-ocean-dir "$bers_dir" \
    --output-dir "$analysis_dir"

echo "[2/5] Paired GC / CMA-ES / BERS improvements"
uv run python -u revision/task3_paired_improvements.py \
    --bers-dir "$bers_dir" \
    --cmaes-dir "$cmaes_dir" \
    --output-dir "$analysis_dir"

echo "[3/5] Wind and wave distributions and exceedances"
uv run python -u revision/task4_weather_violations.py \
    --real-ocean-dir "$bers_dir" \
    --output-dir "$analysis_dir"

echo "[4/5] Exact real-ocean coastline intersection"
uv run python -u revision/task7_land_verification.py \
    --real-ocean-dir "$bers_dir" \
    --output-dir "$analysis_dir" \
    --no-include-synthetic

echo "[5/5] Route-by-route local-optimality audit"
uv run python -u revision/task8_local_optimality.py \
    --real-ocean-dir "$bers_dir" \
    --output-dir "$analysis_dir" \
    --checkpoint-dir "$analysis_dir/task8_checkpoints" \
    --land-verification-csv "$analysis_dir/task7_route_results.csv"

declare -A expected_rows=(
    [task3_paired_metrics.csv]=1464
    [task4_route_metrics.csv]=2928
    [task7_route_results.csv]=1464
    [task8_local_optimality_routes.csv]=1464
)
for filename in "${!expected_rows[@]}"; do
    actual_rows=$(( $(wc -l < "$analysis_dir/$filename") - 1 ))
    if [[ $actual_rows -ne ${expected_rows[$filename]} ]]; then
        echo "Unexpected row count for $filename: $actual_rows" >&2
        exit 1
    fi
done

date -Is > "$completed_marker"
printf '\nCompleted all post-processing: %s\n' "$(date -Is)"
printf 'Results: %s\n' "$analysis_dir"
