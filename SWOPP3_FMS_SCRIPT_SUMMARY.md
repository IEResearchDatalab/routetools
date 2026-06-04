# SWOPP3 FMS Refinement Script - Summary

## What Was Created

A new script **`scripts/swopp3_apply_fms_to_scored_submissions.py`** that applies FMS (Finite-difference route refinement) post-processing to scored SWOPP3 submission archives.

## Key Features

1. **Scored Archive Processing**

   - Extracts submissions from competition-scored zip archives
   - Handles embedded resampled_tracks.zip structure
   - Properly maps case IDs from ZIP folder paths

2. **Selective Route Refinement**

   - Applies FMS only to optimised (non-GC) routes:
     - AO_WPS, AO_noWPS (Atlantic routes)
     - PO_WPS, PO_noWPS (Pacific routes)
   - Skips great-circle cases (AGC, PGC)

3. **Integrated with Submission Comparison**

   - FMS-refined submissions stored in `swopp3_submissions_score/{participant}_fms/`
   - Automatically discovered as additional competitors by `swopp3_submission_compare.py`
   - Enables direct before/after FMS comparison using existing analysis pipeline

4. **Before/After Comparisons**

   - Energy consumption (MWh)
   - Max wind speed (m/s)
   - Max wave height (m)
   - Land violation counts
   - Percentage improvements

5. **Comprehensive Output**
   - SWOPP3-format summary CSVs (IEUniversity-{CASE_ID}.csv)
   - Refined track files in original folder structure
   - Per-case detailed comparison CSV
   - JSON summary with aggregate statistics

## Current Execution

Running on two submissions:

- `mc_boatface` (704564*mc_boatface_PhaseId25571*\*.zip)
- `ohy123` (756343*ohy123_PhaseId25571*\*.zip)

**Status**: Currently processing first case (AO_WPS) - expected to complete in several hours

Output directories:

- `output/swopp3_submissions_score/mc boatface_fms/`
- `output/swopp3_submissions_score/ohy123_fms/`

These will be automatically discovered as additional competitors when running `swopp3_submission_compare.py`.

## Example Command

```bash
# Default: store FMS results in swopp3_submissions_score with _fms suffix
python scripts/swopp3_apply_fms_to_scored_submissions.py \
    output/swopp3_submissions_score/*boatface*.zip \
    output/swopp3_submissions_score/*ohy*.zip

# Custom output base directory
python scripts/swopp3_apply_fms_to_scored_submissions.py \
    output/swopp3_submissions_score/*boatface*.zip \
    output/swopp3_submissions_score/*ohy*.zip \
    --output-base output/swopp3_fms_archive
```

## How to Use the Generated Comparisons

### View Summary Statistics

```bash
cat output/swopp3_fms_scored/mc\ boatface/fms_comparison_summary.json | python -m json.tool
```

### Calculate Total Energy Savings

```bash
python -c "
import json
with open('output/swopp3_fms_scored/mc boatface/fms_comparison_summary.json') as f:
    summary = json.load(f)
    total_original = sum(s['total_original_energy_mwh'] for s in summary.values())
    total_fms = sum(s['total_fms_energy_mwh'] for s in summary.values())
    print(f'Total original: {total_original:.1f} MWh')
    print(f'Total FMS: {total_fms:.1f} MWh')
    print(f'Savings: {total_original - total_fms:.1f} MWh ({100*(total_original-total_fms)/total_original:.1f}%)')
"
```

### Find Best and Worst Improvements

```python
import pandas as pd

# For each case
for case in ['AO_WPS', 'AO_noWPS', 'PO_WPS', 'PO_noWPS']:
    df = pd.read_csv(f'output/swopp3_fms_scored/mc boatface/fms_comparison_{case}.csv')
    best = df.loc[df['energy_pct_change'].idxmin()]
    worst = df.loc[df['energy_pct_change'].idxmax()]

    print(f"\n{case}:")
    print(f"  Best:  {best['departure_utc']} → {best['energy_pct_change']:.2f}%")
    print(f"  Worst: {worst['departure_utc']} → {worst['energy_pct_change']:.2f}%")
    print(f"  Mean:  {df['energy_pct_change'].mean():.2f}%")
```

## Technical Details

### Column Name Handling

The script automatically handles both:

- Original SWOPP3 format: `lat_deg`, `lon_deg`
- Resampled format: `lat`, `lon`

### Memory Management

- ERA5 data loaded in rolling windows (183 days default)
- FMS/JAX caches cleared between cases
- Temporary files automatically cleaned up

### Performance

- Current run: 134% CPU, 26.7GB RAM used
- Processing time: ~15-30 minutes per case
- Total expected: 2-4 hours for 4 cases × 366 departures each

## Files Generated

For each participant, a new folder is created in `swopp3_submissions_score/` with `_fms` suffix:

```
output/swopp3_submissions_score/
├── 704564_mc_boatface_PhaseId25571_*.zip  (original submission)
├── 756343_ohy123_PhaseId25571_*.zip       (original submission)
├── mc boatface_fms/                       (FMS-refined version)
│   ├── tracks/
│   │   └── [366 refined route CSVs, one per departure]
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
    └── [same structure]
```

These can be directly discovered and compared using `swopp3_submission_compare.py` without any additional arguments.

## Next Steps

Once processing completes:

1. **Automatically compare with original submissions**

   ```bash
   python scripts/swopp3_submission_compare.py \
       --input-root output/swopp3_submissions_score \
       --output-dir output/swopp3_submissions_compare_with_fms
   ```

   The `{participant}_fms` folders will be automatically discovered as additional competitors.

2. **Review aggregate statistics** in fms_comparison_summary.json for each FMS variant

3. **Analyze per-departure improvements** in fms*comparison*\*.csv files

4. **Compare submission leaderboard** between original and FMS-refined variants

5. **Generate visualization** of energy distributions before/after FMS

6. **Export refined routes** for further analysis or submission

## Notes

- Script handles multiple archive inputs gracefully
- Pre-commit hooks ensure code quality
- Proper error handling with temp directory cleanup
- Supports all SWOPP3 cases (8 total) but only refines non-GC cases
