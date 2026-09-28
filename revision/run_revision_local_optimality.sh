#!/usr/bin/env bash
# Local-optimality evidence for the BERS revision (Reviewer 1, comment 2).
#
# Usage:
#   bash revision/run_revision_local_optimality.sh [OUTPUT_ROOT]
#
# Steps
#   Task 9  operating-envelope Hessian of the performance model (no data needed)
#   Task 10 synthetic-field route audit + coupled Newton polishing (no data)
#   Task 11 real-ocean coupled Newton polishing (needs ERA5 + final routes)
#
# Tasks 9 and 10 run in a few minutes on a laptop CPU. Task 11 needs the
# ERA5 2024 files under data/era5/ and the final route folder used in the
# paper (output/sweep_combined_fms_strict); set SKIP_TASK11=1 to skip it.

set -Eeuo pipefail

script_path="$(realpath "${BASH_SOURCE[0]}")"
repo_root="$(cd "$(dirname "$script_path")/.." && pwd)"
out_root="$(realpath -m "${1:-$repo_root/output/revision_local_optimality}")"
mkdir -p "$out_root"
cd "$repo_root"
export JAX_PLATFORMS=cpu
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=ignore

printf '%s\n' "Started: $(date -Is)" "Git commit: $(git rev-parse HEAD)" \
    > "$out_root/provenance.txt"

echo "[1/3] Task 9: operating-envelope Hessian"
uv run python revision/task9_envelope_hessian.py \
    --output-dir "$out_root/task9_envelope_hessian"

echo "[2/3] Task 10: synthetic local-optimality audit"
uv run python revision/task10_synthetic_local_optimality.py \
    --output-dir "$out_root/task10_synthetic_local_opt"

if [[ "${SKIP_TASK11:-0}" != "1" ]]; then
    echo "[3/3] Task 11: real-ocean coupled Newton polishing"
    uv run python revision/task11_real_ocean_newton_polish.py \
        --real-ocean-dir output/sweep_combined_fms_strict \
        --experiment-manifest revision/task8_sweep_combined_fms_strict_manifest.json \
        --tws-limit 19.9 --hs-limit 6.9 --land-distance-weight 0 \
        --output-dir "$out_root/task11_newton_polish" \
        --checkpoint-dir "$out_root/task11_newton_polish/checkpoints"
fi

date -Is > "$out_root/COMPLETED"
echo "Results: $out_root"
