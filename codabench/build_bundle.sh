#!/usr/bin/env bash
# Build CodaBench competition bundle zip files.
#
# Usage:
#     cd codabench && bash build_bundle.sh
#
# Creates:
#     scoring_program.zip   — upload as Scoring Program
#     starting_kit.zip      — upload as Starting Kit
#     reference_data.zip    — upload as Reference Data
#     competition_bundle.zip — full bundle (alternative upload method)

set -euo pipefail
cd "$(dirname "$0")"

echo "Building CodaBench bundles..."

# Scoring program
echo "  → scoring_program.zip"
(cd scoring_program && zip -r ../scoring_program.zip . -x '__pycache__/*' '*.pyc')

# Starting kit
echo "  → starting_kit.zip"
(cd starting_kit && zip -r ../starting_kit.zip . -x '__pycache__/*' '*.pyc')

# Reference data — ERA5 NetCDF files + Natural Earth land shapefile
# These files must be placed in reference_data/ before building.
# See the list below for required files.
echo "  → reference_data.zip"
mkdir -p reference_data

REQUIRED_FILES=(
    # ERA5 weather data (2024)
    "era5_wind_atlantic_2024.nc"
    "era5_waves_atlantic_2024.nc"
    "era5_wind_pacific_2024.nc"
    "era5_waves_pacific_2024.nc"
    # ERA5 weather data (January 2025, for late-2024 departures)
    "era5_wind_atlantic_2025_01.nc"
    "era5_waves_atlantic_2025_01.nc"
    "era5_wind_pacific_2025_01.nc"
    "era5_waves_pacific_2025_01.nc"
    # Natural Earth land shapefile (for land crossing checks)
    "ne_10m_land.shp"
    "ne_10m_land.shx"
    "ne_10m_land.dbf"
    "ne_10m_land.prj"
)

MISSING=0
for f in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "reference_data/$f" ]]; then
        echo "    ⚠ Missing: reference_data/$f"
        MISSING=1
    fi
done
if [[ $MISSING -eq 1 ]]; then
    echo ""
    echo "    Some reference data files are missing."
    echo "    Place them in codabench/reference_data/ and re-run."
    echo "    The bundle will be built without them (scoring will use self-reported values)."
    echo ""
fi

(cd reference_data && zip -r ../reference_data.zip . -x '__pycache__/*' '*.pyc')

# Lightweight bundle (no ERA5 .nc files — upload reference_data.zip separately)
# CodaBench v2 expects directories, not nested zips.
echo "  → competition_bundle.zip (lightweight, no ERA5 data)"
rm -f competition_bundle.zip
zip -r competition_bundle.zip \
    competition.yaml \
    logo.png \
    scoring_program/ \
    starting_kit/ \
    reference_data/ \
    pages/ \
    -x '__pycache__/*' '*.pyc' '*.nc'

echo ""
echo "Done!"
echo ""
echo "Lightweight bundle (for CodaBench upload wizard):"
echo "  competition_bundle.zip  — everything except ERA5 .nc files"
echo ""
echo "After the competition is created, upload the full reference data:"
echo "  reference_data.zip      → Edit Competition → Tasks tab → Reference Data"
echo ""
echo "Individual zips (for manual upload via competition editor):"
echo "  scoring_program.zip     → Tasks tab → Scoring Program"
echo "  reference_data.zip      → Tasks tab → Reference Data"
echo "  starting_kit.zip        → Participation tab → Starting Kit"
