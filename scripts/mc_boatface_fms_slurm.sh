#!/bin/bash
#SBATCH --job-name=mc_boatface_fms
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0-12:00:00
#SBATCH --output=slurm_fms_%j.out
#SBATCH --error=slurm_fms_%j.err

# ── MC Boatface FMS Optimization on rust-HPC ──
#
# Applies FMS optimization to resampled routes with full ERA5 datasets loaded.
#
# Submit:  sbatch scripts/mc_boatface_fms_slurm.sh
# Monitor: squeue -u $USER

set -euo pipefail

# ── Environment ──
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/routetools"
source .venv/bin/activate

export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# ── Paths ──
DATA="data/era5"
INPUT_DIR="output/mc_boatface/resampled"
OUTPUT_DIR="output/mc_boatface"

mkdir -p "$OUTPUT_DIR"

echo "======================================"
echo "MC Boatface FMS optimization on $(hostname)"
echo "Date:     $(date)"
echo "CPUs:     ${SLURM_CPUS_PER_TASK}"
echo "Memory:   ${SLURM_MEM_PER_NODE}M"
echo "Input:    ${INPUT_DIR}"
echo "Output:   ${OUTPUT_DIR}"
echo "======================================"

# Verify input folder
if [[ ! -d "$INPUT_DIR" ]]; then
    echo "ERROR: Input directory not found: $INPUT_DIR" >&2
    exit 1
fi
echo "Input directory found."

# Verify data files
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
echo "All ERA5 data files present."

echo ""
echo "Starting FMS optimization at $(date)"
echo ""

# Run the FMS script
python scripts/mc_boatface_fms_hpc.py

echo ""
echo "======================================"
echo "FMS optimization completed at $(date)"
echo "======================================"
echo ""
echo "Output summary:"
if [[ -f "${OUTPUT_DIR}/energy_comparison.csv" ]]; then
    echo "Energy comparison:"
    wc -l "${OUTPUT_DIR}/energy_comparison.csv"
    echo ""
    echo "Sample (first 10 rows):"
    head -10 "${OUTPUT_DIR}/energy_comparison.csv"
else
    echo "No energy_comparison.csv found"
fi
