#!/bin/bash

# Fail loudly in logs
export PYTHONUNBUFFERED=1
export CUDA_LAUNCH_BLOCKING=1

LOGFILE="$HOME/SHARED/routetools/run_results.log"

echo "======================================" >> "$LOGFILE"
echo "Starting results.py at $(date)" >> "$LOGFILE"

while true; do
    uv run python -X faulthandler scripts/realworld/results.py >> "$LOGFILE" 2>&1
    EXIT_CODE=$?

    echo "--------------------------------------" >> "$LOGFILE"
    echo "Process exited with code $EXIT_CODE at $(date)" >> "$LOGFILE"
    echo "Restarting in 10 seconds..." >> "$LOGFILE"
    sleep 10
done

uv run python -X faulthandler scripts/realworld/figures.py >> "$LOGFILE" 2>&1
uv run python -X faulthandler scripts/realworld/tables.py >> "$LOGFILE" 2>&1
