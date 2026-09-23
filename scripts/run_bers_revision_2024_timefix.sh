#!/usr/bin/env bash
# Launch the full corrected 2024 BERS real-ocean experiment in the background.
#
# Usage (from anywhere inside the repository):
#   bash scripts/run_bers_revision_2024_timefix.sh
#
# Optional custom run name:
#   bash scripts/run_bers_revision_2024_timefix.sh my_run_name

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TIMESTAMP_FIX_COMMIT="57d82bd5564c160c93af5d066a5b7a78e1c829a6"

run_worker() {
    local output_dir="$1"
    local config_path="$2"

    cd "$ROOT_DIR"

    echo "============================================================"
    echo "BERS corrected 2024 real-ocean experiment"
    echo "Started:       $(date --iso-8601=seconds)"
    echo "Host:          $(hostname)"
    echo "Repository:    $ROOT_DIR"
    echo "Git commit:    $(git rev-parse HEAD)"
    echo "Output:        $output_dir"
    echo "Configuration: $config_path"
    echo "============================================================"

    free -h || true
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi \
            --query-gpu=name,memory.total,memory.free,driver_version \
            --format=csv,noheader || true
    fi

    export JAX_PLATFORMS=cuda
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export XLA_PYTHON_CLIENT_ALLOCATOR=platform

    if ! uv run --extra cuda python -c \
        'import jax; print(f"JAX {jax.__version__}; devices={jax.devices()}")'; then
        printf 'exit_code=1\nfinished=%s\n' \
            "$(date --iso-8601=seconds)" > "$output_dir/FAILED"
        echo "ERROR: the CUDA/JAX environment check failed." >&2
        return 1
    fi

    if uv run --extra cuda python -u scripts/swopp3_run.py bers_revision_final \
        --config-path "$config_path"; then
        printf '%s\n' "$(date --iso-8601=seconds)" > "$output_dir/COMPLETED"
        echo "============================================================"
        echo "Completed successfully: $(date --iso-8601=seconds)"
        echo "Output: $output_dir"
        echo "============================================================"
        return 0
    else
        exit_code=$?
        printf 'exit_code=%s\nfinished=%s\n' \
            "$exit_code" "$(date --iso-8601=seconds)" > "$output_dir/FAILED"
        echo "ERROR: experiment exited with code $exit_code" >&2
        return "$exit_code"
    fi
}

if [[ "${1:-}" == "--worker" ]]; then
    if (( $# != 3 )); then
        echo "Internal usage: $0 --worker OUTPUT_DIR CONFIG_PATH" >&2
        exit 2
    fi
    run_worker "$2" "$3"
    exit $?
fi

cd "$ROOT_DIR"

if (( $# > 1 )); then
    echo "Usage: $0 [RUN_NAME]" >&2
    exit 2
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv is not installed or is not available on PATH." >&2
    exit 1
fi

if ! git merge-base --is-ancestor "$TIMESTAMP_FIX_COMMIT" HEAD; then
    echo "ERROR: this checkout does not contain the weather timestamp fix." >&2
    echo "Required commit: $TIMESTAMP_FIX_COMMIT" >&2
    exit 1
fi

DATA_FILES=(
    "data/era5/era5_wind_atlantic_2024.nc"
    "data/era5/era5_waves_atlantic_2024.nc"
    "data/era5/era5_wind_pacific_2024.nc"
    "data/era5/era5_waves_pacific_2024.nc"
)

for data_file in "${DATA_FILES[@]}"; do
    if [[ ! -s "$data_file" ]]; then
        echo "ERROR: missing or empty ERA5 input: $data_file" >&2
        exit 1
    fi
done

run_name="${1:-bers_revision_2024_final_$(date +%Y%m%d)}"
if [[ ! "$run_name" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "ERROR: run name may contain only letters, digits, dots, underscores, and hyphens." >&2
    exit 1
fi

output_dir="$ROOT_DIR/output/$run_name"
bers_output_dir="$output_dir/bers"
cmaes_output_dir="$output_dir/cmaes"
config_path="$output_dir/resolved_config.toml"
log_path="$output_dir/run.log"
pid_path="$output_dir/run.pid"

if [[ -e "$output_dir" ]]; then
    echo "ERROR: output directory already exists: $output_dir" >&2
    echo "Use a different run name; existing results will not be overwritten." >&2
    exit 1
fi

mkdir -p "$output_dir"

# Make a run-specific copy of the final profile with immutable stage paths.
awk -v bers_out="$bers_output_dir" -v cmaes_out="$cmaes_output_dir" '
    $0 == "[swopp3.experiments.bers_revision_final]" { profile=1 }
    profile && /^output_dir[[:space:]]*=/ {
        print "output_dir = \"" bers_out "\""
        profile=0
        next
    }
    $0 == "[swopp3.experiments.bers_revision_final.defaults]" { defaults=1 }
    defaults && /^cmaes_output_dir[[:space:]]*=/ {
        print "cmaes_output_dir = \"" cmaes_out "\""
        defaults=0
        next
    }
    { print }
' config.toml > "$config_path"

if ! grep -Fqx "output_dir = \"$bers_output_dir\"" "$config_path"; then
    echo "ERROR: could not resolve the final BERS output directory." >&2
    exit 1
fi
if ! grep -Fqx "cmaes_output_dir = \"$cmaes_output_dir\"" "$config_path"; then
    echo "ERROR: could not resolve the CMA-ES ablation output directory." >&2
    exit 1
fi

# Record sufficient provenance to identify the exact code and configuration.
git rev-parse HEAD > "$output_dir/git_commit.txt"
git status --short > "$output_dir/git_status.txt"
git diff HEAD > "$output_dir/uncommitted.patch"
uv --version > "$output_dir/uv_version.txt"
cp "${BASH_SOURCE[0]}" "$output_dir/launcher_snapshot.sh"
stat --printf='%n\t%s bytes\t%y\n' "${DATA_FILES[@]}" \
    > "$output_dir/era5_inputs.txt"

nohup bash "$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")" \
    --worker "$output_dir" "$config_path" \
    > "$log_path" 2>&1 &

run_pid=$!
printf '%s\n' "$run_pid" > "$pid_path"

echo "Started the complete BERS route experiment."
echo "Run name: $run_name"
echo "PID:      $run_pid"
echo "Run root: $output_dir"
echo "BERS:     $bers_output_dir"
echo "CMA-ES:   $cmaes_output_dir"
echo
echo "Monitor:"
echo "  tail -f '$log_path'"
echo
echo "Check status:"
echo "  ps -fp \"\$(cat '$pid_path')\""
echo
echo "Completion marker:"
echo "  test -f '$output_dir/COMPLETED' && echo complete"
