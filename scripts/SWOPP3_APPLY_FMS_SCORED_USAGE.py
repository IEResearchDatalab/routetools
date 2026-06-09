#!/usr/bin/env python
r"""USAGE EXAMPLES FOR swopp3_apply_fms_to_scored_submissions.py.

This script applies FMS (Finite-difference route refinement) post-processing to
scored SWOPP3 submission archives and generates before/after energy comparisons.

The FMS-refined submissions are stored back in swopp3_submissions_score with
a "_fms" suffix, allowing them to be automatically discovered as additional
competitors by swopp3_submission_compare.py.

BASIC USAGE
===========

Default behavior (store FMS results in swopp3_submissions_score):
    python scripts/swopp3_apply_fms_to_scored_submissions.py \
        output/swopp3_submissions_score/*boatface*.zip \
        output/swopp3_submissions_score/*ohy*.zip

    Creates:
    - output/swopp3_submissions_score/mc boatface_fms/
    - output/swopp3_submissions_score/ohy123_fms/

Single submission with custom output directory:
    python scripts/swopp3_apply_fms_to_scored_submissions.py \
        output/swopp3_submissions_score/submission_1.zip \
        --output-dir output/swopp3_fms_custom

Multiple archives with custom base directory:
    python scripts/swopp3_apply_fms_to_scored_submissions.py \
        output/swopp3_submissions_score/*boatface*.zip \
        output/swopp3_submissions_score/*ohy*.zip \
        --output-base output/swopp3_fms_archive

WHAT THE SCRIPT DOES
====================

1. **Extracts submission**: Reads the scored archive and extracts the
   embedded resampled_tracks.zip containing the original routes.

2. **Processes non-GC routes**: Applies FMS refinement to optimised cases only:
   - AO_WPS (Atlantic Optimised with WPS)
   - AO_noWPS (Atlantic Optimised without WPS)
   - PO_WPS (Pacific Optimised with WPS)
   - PO_noWPS (Pacific Optimised without WPS)

   Great-circle cases (AGC_WPS, AGC_noWPS, PGC_WPS, PGC_noWPS) are not refined.

3. **Generates outputs**: For each non-GC case, creates:
   - IEUniversity-{CASE_ID}.csv - Summary with FMS-refined routes and energy
   - tracks/ folder - FMS-refined track CSV files
   - fms_comparison_{CASE_ID}.csv - Detailed before/after comparison table
   - fms_comparison_summary.json - Aggregate statistics by case

OUTPUT STRUCTURE
================

Default output (stored in swopp3_submissions_score):

output/swopp3_submissions_score/
├── [original submission zips]
├── mc boatface_fms/
│   ├── tracks/
│   │   ├── details_swopp_wps_atlantic_8kn_2024-01-01T12.csv
│   │   └── ... (366 FMS-refined tracks per case)
│   ├── IEUniversity-AO_WPS.csv
│   ├── IEUniversity-AO_noWPS.csv
│   ├── IEUniversity-PO_WPS.csv
│   ├── IEUniversity-PO_noWPS.csv
│   ├── fms_comparison_AO_WPS.csv
│   ├── fms_comparison_AO_noWPS.csv
│   ├── fms_comparison_PO_WPS.csv
│   ├── fms_comparison_PO_noWPS.csv
│   └── fms_comparison_summary.json
└── ohy123_fms/
    └── ... (same structure)

These folders are automatically discovered as additional competitors when
running swopp3_submission_compare.py on the same input-root directory.

COMPARISON METRICS
==================

Each comparison table includes:
- departure_utc: Departure timestamp
- original_energy_mwh: Energy before FMS
- fms_energy_mwh: Energy after FMS
- energy_delta_mwh: Absolute change (negative = improvement)
- energy_pct_change: Percentage change
- original_max_tws_mps: Original max wind speed
- fms_max_tws_mps: FMS max wind speed
- original_max_hs_m: Original max wave height
- fms_max_hs_m: FMS max wave height
- original_land_violations: Land violations before FMS
- fms_land_violations: Land violations after FMS

COMMAND-LINE OPTIONS
====================

Input/Output:
  --output-dir DIR        Explicit output directory (for single archive only).
                          If not specified, defaults to storing in the same
                          directory as the archive with _fms suffix.
  --output-base DIR       Base folder for multiple archives. Creates subfolders
                          with participant name + _fms suffix. Useful for
                          organizing FMS results separately from originals.

FMS Parameters:
  --fms-patience N        Early-stopping patience (default: 200)
  --fms-damping FLOAT     Damping factor (default: 0.95)
  --fms-maxfevals N       Max iterations per route (default: 10000)

ERA5 Data:
  --wind-path PATH        Wind NetCDF for all corridors
  --wave-path PATH        Wave NetCDF for all corridors
  --wind-path-atlantic    Atlantic-specific wind NetCDF
  --wave-path-atlantic    Atlantic-specific wave NetCDF
  --wind-path-pacific     Pacific-specific wind NetCDF
  --wave-path-pacific     Pacific-specific wave NetCDF

Constraints:
  --tws-limit FLOAT       Max true wind speed (default: 20.0 m/s)
  --hs-limit FLOAT        Max significant wave height (default: 7.0 m)
  --wind-penalty-weight FLOAT     Wind violation penalty (default: 1000)
  --wave-penalty-weight FLOAT     Wave violation penalty (default: 1000)
  --enforce-weather-limits        Reject updates that new violate limits

Control:
  --quiet / -q            Suppress progress output
  --era5-batch-days FLOAT Reload interval (default: 183 days)

PRACTICAL EXAMPLE
=================

1. Refine submissions and store in swopp3_submissions_score:

  python scripts/swopp3_apply_fms_to_scored_submissions.py \
      output/swopp3_submissions_score/*boatface*.zip \
      output/swopp3_submissions_score/*ohy*.zip

2. Compare original vs FMS-refined submissions:

  python scripts/swopp3_submission_compare.py \
      --input-root output/swopp3_submissions_score \
      --output-dir output/swopp3_comparison_with_fms

  This will automatically include:
  - Original submissions (mc_boatface, ohy123, etc.)
  - FMS-refined variants (mc_boatface_fms, ohy123_fms, etc.)

3. View FMS-specific summary:
  cat output/swopp3_submissions_score/mc\\ boatface_fms/
      fms_comparison_summary.json | jq .

4. Calculate energy improvements by case:
  python -c "
import json
for participant in ['mc boatface_fms', 'ohy123_fms']:
    path = f'output/swopp3_submissions_score/{participant}/fms_comparison_summary.json'
    with open(path) as f:
        summary = json.load(f)
        print(f'\n{participant}:')
        for case, stats in sorted(summary.items()):
            delta = stats['total_energy_delta_mwh']
            pct = 100 * delta / stats['total_original_energy_mwh']
            print(f'  {case}: {delta:+.1f} MWh ({pct:+.1f}%)')
  "

Notes
-----
- Processing time: ~15-30 minutes per case (depends on hardware and FMS params)
- Memory: ~8GB per corridor (ERA5 data loaded in rolling windows)
- The script creates temporary directories that are cleaned up automatically
- JAX/FMS caches are cleared between cases to manage memory
- Paths to ERA5 datasets are required (default paths assume data/era5/)
  Download ERA5 data: uv run scripts/download_era5.py

PERFORMANCE TIPS
================

1. Process submissions in parallel using separate processes
2. Use --era5-batch-days to tune memory vs I/O tradeoff
3. Reduce --fms-maxfevals for faster (less refined) results
4. Use --quiet to reduce I/O overhead
"""

if __name__ == "__main__":
    print(__doc__)
