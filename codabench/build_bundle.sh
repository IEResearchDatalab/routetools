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
(cd scoring_program && zip -r ../scoring_program.zip . -x '*__pycache__/*' '*__pycache__' '*.pyc')

# Starting kit
echo "  → starting_kit.zip"
(cd starting_kit && zip -r ../starting_kit.zip . -x '*__pycache__/*' '*__pycache__' '*.pyc')

# Reference data — metadata only.
# The ~19.6 GB hourly ERA5 release and the Natural Earth shapefile are NOT
# bundled: the compute worker serves them from /codabench/data, mounted into
# each submission container as /app/data. Only the manifest travels here so the
# scorer can confirm which release it scored against.
echo "  → reference_data.zip (metadata only)"
mkdir -p reference_data

REQUIRED_FILES=(
    "config.json"
    "SHA256SUMS"
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
    echo "    Reference metadata is incomplete — the scorer cannot report which"
    echo "    weather release it used. Restore the files and re-run."
    echo ""
fi

# If a full data directory happens to be staged here, verify it matches the
# frozen Benchmark 001 release before it can be published anywhere.
if [[ -f "reference_data/SHA256SUMS" ]] && compgen -G "reference_data/*.nc" > /dev/null; then
    echo "    Verifying Benchmark 001 checksums..."
    (cd reference_data && sha256sum --check SHA256SUMS)
fi

(cd reference_data && zip -r ../reference_data.zip . -x '__pycache__/*' '*.pyc')

# CodaBench v2 expects directories, not nested zips.
# Stale *.html page exports are excluded: competition.yaml serves the Markdown
# pages, and the old exports still describe superseded conditions.
echo "  → competition_bundle.zip"
rm -f competition_bundle.zip
zip -r competition_bundle.zip \
    competition.yaml \
    logo.png \
    scoring_program/ \
    starting_kit/ \
    reference_data/ \
    pages/ \
    -x '*__pycache__/*' '*__pycache__' '*.pyc' '*.nc' 'pages/*.html'

echo ""
echo "Done!"
echo ""
echo "Upload to CodaBench:"
echo "  competition_bundle.zip  → Benchmarks → Create → upload bundle"
echo ""
echo "The hourly ERA5 release is NOT in the bundle. It is already on the compute"
echo "worker at /codabench/data. At 19.57 GB it also exceeds the 15 GB CodaBench"
echo "account quota, so publish the participant copy externally and link it from"
echo "the Data page, alongside reference_data/SHA256SUMS."
echo ""
echo "Individual zips (for manual upload via the competition editor):"
echo "  scoring_program.zip     → Tasks tab → Scoring Program"
echo "  reference_data.zip      → Tasks tab → Reference Data (metadata only)"
echo "  starting_kit.zip        → Participation tab → Starting Kit"
