#!/bin/bash
# ── FMS refinement for sweep_combined output ──
#
# Applies FMS to output/sweep_combined and writes output/sweep_combined_fms.
# Automatically restarts if the process exits unexpectedly (non-zero exit code).
# Resume is built into swopp3_apply_fms.py: completed routes are skipped.
#
# Run in the background with:
#   nohup bash scripts/run_fms_sweep_combined.sh > output/sweep_combined_fms.log 2>&1 &
#
# Monitor:
#   tail -f output/sweep_combined_fms.log
#   cat output/fms_sweep_combined.pid

set -euo pipefail  # strict, but we trap errors below

ROOTDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOTDIR"

LOGFILE="output/sweep_combined_fms.log"
PIDFILE="output/fms_sweep_combined.pid"
INPUT_DIR="output/sweep_combined"
OUTPUT_DIR="output/sweep_combined_fms"

# FMS parameters (matching RISE cost used in CMA-ES sweep)
# Both corridors used wind_pw=50, wave_pw=50
WIND_PW=50
WAVE_PW=50

# ERA5 data paths
WIND_ATL="data/era5/era5_wind_atlantic_2024.nc"
WAVE_ATL="data/era5/era5_waves_atlantic_2024.nc"
WIND_PAC="data/era5/era5_wind_pacific_2024.nc"
WAVE_PAC="data/era5/era5_waves_pacific_2024.nc"

# Auto-restart configuration
MAX_RETRIES=20
RETRY_DELAY=60   # seconds between retries

echo "$BASHPID" > "$PIDFILE"

echo "============================================"
echo "FMS sweep_combined runner started"
echo "Date:         $(date)"
echo "Host:         $(hostname)"
echo "PID:          $BASHPID"
echo "PID file:     $PIDFILE"
echo "Input:        $INPUT_DIR"
echo "Output:       $OUTPUT_DIR"
echo "Wind PW:      $WIND_PW"
echo "Wave PW:      $WAVE_PW"
echo "Max retries:  $MAX_RETRIES"
echo "============================================"

# Validate data files
for f in "$WIND_ATL" "$WAVE_ATL" "$WIND_PAC" "$WAVE_PAC"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: Missing data file: $f" >&2
        exit 1
    fi
done
echo "All ERA5 data files present."

mkdir -p "$OUTPUT_DIR"

attempt=0
while (( attempt < MAX_RETRIES )); do
    attempt=$(( attempt + 1 ))
    echo ""
    echo "--- Attempt ${attempt}/${MAX_RETRIES}: $(date) ---"

    # Run FMS; allow non-zero exit so we can retry
    set +e
    uv run scripts/swopp3_apply_fms.py \
        "$INPUT_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --wind-path-atlantic "$WIND_ATL" \
        --wave-path-atlantic "$WAVE_ATL" \
        --wind-path-pacific  "$WIND_PAC" \
        --wave-path-pacific  "$WAVE_PAC" \
        --wind-penalty-weight "$WIND_PW" \
        --wave-penalty-weight "$WAVE_PW"
    exit_code=$?
    set -e

    if (( exit_code == 0 )); then
        echo ""
        echo "============================================"
        echo "FMS completed successfully: $(date)"
        echo "Output: $OUTPUT_DIR"
        echo "============================================"
        rm -f "$PIDFILE"
        exit 0
    fi

    echo ""
    echo "WARNING: FMS exited with code ${exit_code} at $(date)"
    if (( attempt < MAX_RETRIES )); then
        echo "Retrying in ${RETRY_DELAY}s (attempt ${attempt}/${MAX_RETRIES})..."
        sleep "$RETRY_DELAY"
    fi
done

echo ""
echo "ERROR: FMS failed after ${MAX_RETRIES} attempts. Giving up."
rm -f "$PIDFILE"
exit 1
