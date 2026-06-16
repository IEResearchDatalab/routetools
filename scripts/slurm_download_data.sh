#!/bin/bash
#SBATCH --job-name=era5_download
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=era5_download_%j.out
#SBATCH --error=era5_download_%j.err

set -euo pipefail

echo "======================================"
echo "ERA5 Download Job"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "======================================"

# Load stable Python module
module load Python/3.11.5-GCCcore-13.2.0

# Go to project directory
cd $HOME/routetools

# Activate virtual environment
source .venv/bin/activate

# Ensure JAX stays on CPU
export JAX_PLATFORMS=cpu

# Create output directory if missing
mkdir -p data/era5

# Run ERA5 downloader (default: 2024, atlantic + pacific)
uv run scripts/download_era5.py

echo ""
echo "======================================"
echo "Download completed at $(date)"
echo "======================================"
