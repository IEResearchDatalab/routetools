#!/bin/bash
# ── Pacific no-constraints run on vangelis (standalone GPU) ──
#
# Runs Pacific corridor cases (PO_noWPS, PGC_noWPS) with
# weather_penalty_weight=0, disabling the operational constraints
# (TWS < 20 m/s, Hs < 7 m) so the optimizer is free to route through
# severe weather if it reduces energy consumption.
#
# Usage:
#   ssh vangelis
#   cd ~/routetools
#   nohup bash scripts/vangelis_pacific_noconstraint.sh > run_noconstraint.log 2>&1 &
#
# Prerequisites:
#   - ERA5 hourly data in data/era5/ (see copy commands below)
#   - Python venv with routetools installed: uv sync

set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

# ── GPU config ──
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

DATA="data/era5"
OUTDIR="output/pacific_noconstraint"
mkdir -p "$OUTDIR"

echo "======================================"
echo "Pacific NO-CONSTRAINT run on $(hostname)"
echo "Date:     $(date)"
echo "GPU:      $(nvidia-smi -L 2>/dev/null | head -1 || echo 'unknown')"
echo "n-points: 584 (dt ≈ 1.0h)"
echo "wpw:      0 (no operational constraints)"
echo "Output:   ${OUTDIR}"
echo "======================================"

# Verify data files exist
for f in \
    "${DATA}/era5_wind_pacific_2024.nc" \
    "${DATA}/era5_waves_pacific_2024.nc"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: Missing data file: $f" >&2
        echo "Copy from rust-hpc with:" >&2
        echo "  rsync -avP rust-hpc:~/routetools/data/era5/era5_*_pacific_2024*.nc data/era5/" >&2
        exit 1
    fi
done
echo "All data files present."

# JAX sanity check
python -c "
import jax
print(f'JAX {jax.__version__}, devices: {jax.devices()}')
for d in jax.devices():
    if hasattr(d, 'memory_stats'):
        stats = d.memory_stats()
        if stats:
            total = stats.get('bytes_limit', 0) / 1e9
            print(f'  {d}: {total:.1f} GB total')
"

echo ""
echo "Starting run at $(date)"
echo ""

python scripts/swopp3_run.py \
    --cases PO_noWPS --cases PGC_noWPS \
    --wind-path-pacific "${DATA}/era5_wind_pacific_2024.nc" \
    --wave-path-pacific "${DATA}/era5_waves_pacific_2024.nc" \
    --output-dir "$OUTDIR" \
    --n-points 584 \
    --weather-penalty-weight 0

echo ""
echo "======================================"
echo "Completed at $(date)"
echo "======================================"
echo ""
echo "Output files:"
ls -lh "$OUTDIR"/*.csv 2>/dev/null || echo "(no CSV files found)"
