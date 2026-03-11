#!/usr/bin/env python
"""Compare ERA5 data downloaded from GCS vs CDS.

Downloads a small subset from the public GCS archive and compares it
against the existing CDS-downloaded NetCDF files in ``data/era5/``.

Usage
-----
Compare Atlantic wind for January 2024::

    uv run scripts/compare_era5_sources.py

Compare a specific corridor/month::

    uv run scripts/compare_era5_sources.py --corridor pacific --months 6

Compare already-downloaded files without re-downloading::

    uv run scripts/compare_era5_sources.py --gcs-dir data/era5_gcs --skip-download
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import xarray as xr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Variable name mapping: GCS (long) → CDS (short)
VAR_MAP = {
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "significant_height_of_combined_wind_waves_and_swell": "swh",
    "mean_wave_direction": "mwd",
}


def _normalise_ds(ds: xr.Dataset) -> xr.Dataset:
    """Normalise coordinate and variable names for comparison.

    - Renames ``time`` → ``valid_time`` if needed.
    - Renames long GCS variable names to short CDS names.
    - Drops extra CDS coordinates (``number``, ``expver``).
    - Ensures ascending latitude.
    """
    # Time dimension
    if "time" in ds.dims and "valid_time" not in ds.dims:
        ds = ds.rename({"time": "valid_time"})

    # Variable names
    rename = {k: v for k, v in VAR_MAP.items() if k in ds.data_vars}
    if rename:
        ds = ds.rename(rename)

    # Drop extra coords that CDS adds
    for coord in ("number", "expver"):
        if coord in ds.coords:
            ds = ds.drop_vars(coord)

    # Ensure ascending latitude
    lat_name = "latitude" if "latitude" in ds.dims else "lat"
    if ds[lat_name].values[0] > ds[lat_name].values[-1]:
        ds = ds.isel({lat_name: slice(None, None, -1)})

    return ds


def _compare_field(
    gcs_ds: xr.Dataset,
    cds_ds: xr.Dataset,
    var_name: str,
    field_label: str,
) -> dict:
    """Compare a single variable between GCS and CDS datasets.

    Returns a dict with comparison statistics.
    """
    gcs_vals = gcs_ds[var_name].values
    cds_vals = cds_ds[var_name].values

    result: dict = {"variable": var_name, "field": field_label}

    # Shape check
    if gcs_vals.shape != cds_vals.shape:
        result["match"] = False
        result["error"] = (
            f"Shape mismatch: GCS {gcs_vals.shape} vs CDS {cds_vals.shape}"
        )
        return result

    # NaN pattern
    gcs_nan = np.isnan(gcs_vals)
    cds_nan = np.isnan(cds_vals)
    nan_match = np.array_equal(gcs_nan, cds_nan)
    result["nan_pattern_match"] = nan_match
    result["gcs_nan_count"] = int(np.sum(gcs_nan))
    result["cds_nan_count"] = int(np.sum(cds_nan))

    # Value comparison (ignoring NaN positions)
    valid = ~(gcs_nan | cds_nan)
    n_valid = int(np.sum(valid))
    result["n_valid_points"] = n_valid

    if n_valid == 0:
        result["match"] = nan_match
        return result

    gcs_valid = gcs_vals[valid]
    cds_valid = cds_vals[valid]

    # Exact match
    exact = np.array_equal(gcs_valid, cds_valid)
    result["exact_match"] = exact

    # Numerical differences
    diff = np.abs(gcs_valid - cds_valid)
    result["max_abs_diff"] = float(np.max(diff))
    result["mean_abs_diff"] = float(np.mean(diff))
    result["median_abs_diff"] = float(np.median(diff))

    # Relative differences (avoid division by zero)
    denom = np.maximum(np.abs(cds_valid), 1e-10)
    rel_diff = diff / denom
    result["max_rel_diff"] = float(np.max(rel_diff))
    result["mean_rel_diff"] = float(np.mean(rel_diff))

    # Tolerance checks
    result["allclose_atol1e-5"] = bool(np.allclose(gcs_valid, cds_valid, atol=1e-5))
    result["allclose_atol1e-3"] = bool(np.allclose(gcs_valid, cds_valid, atol=1e-3))

    result["match"] = exact
    return result


def compare_datasets(
    gcs_path: Path,
    cds_path: Path,
    field: str,
    month_slice: slice | None = None,
) -> list[dict]:
    """Load and compare GCS vs CDS NetCDF files.

    Parameters
    ----------
    gcs_path, cds_path : Path
        Paths to the GCS and CDS NetCDF files.
    field : str
        ``"wind"`` or ``"waves"``.
    month_slice : slice, optional
        Time slice to apply to the CDS file (which may cover a full year).

    Returns
    -------
    list[dict]
        Per-variable comparison results.
    """
    logger.info("Loading GCS: %s", gcs_path)
    gcs_ds = _normalise_ds(xr.open_dataset(gcs_path))

    logger.info("Loading CDS: %s", cds_path)
    cds_ds = _normalise_ds(xr.open_dataset(cds_path))

    # Align time range: slice CDS to match GCS time range
    time_name = "valid_time"
    gcs_times = gcs_ds[time_name].values
    t_start, t_end = gcs_times[0], gcs_times[-1]
    logger.info(
        "GCS time range: %s → %s (%d steps)",
        np.datetime_as_string(t_start, unit="h"),
        np.datetime_as_string(t_end, unit="h"),
        len(gcs_times),
    )

    cds_ds = cds_ds.sel({time_name: slice(t_start, t_end)})
    cds_times = cds_ds[time_name].values
    logger.info(
        "CDS time range (sliced): %s → %s (%d steps)",
        np.datetime_as_string(cds_times[0], unit="h"),
        np.datetime_as_string(cds_times[-1], unit="h"),
        len(cds_times),
    )

    # Report grid comparison
    for dim in ("latitude", "longitude"):
        gcs_v = gcs_ds[dim].values
        cds_v = cds_ds[dim].values
        if np.array_equal(gcs_v, cds_v):
            logger.info("  %s: EXACT match (%d points)", dim, len(gcs_v))
        else:
            logger.warning(
                "  %s: MISMATCH — GCS %d pts [%.4f, %.4f] vs CDS %d pts [%.4f, %.4f]",
                dim,
                len(gcs_v),
                gcs_v[0],
                gcs_v[-1],
                len(cds_v),
                cds_v[0],
                cds_v[-1],
            )

    # Time comparison
    if np.array_equal(gcs_times, cds_times):
        logger.info("  valid_time: EXACT match (%d steps)", len(gcs_times))
    else:
        logger.warning(
            "  valid_time: MISMATCH — GCS %d steps vs CDS %d steps",
            len(gcs_times),
            len(cds_times),
        )

    # Compare variables
    if field == "wind":
        var_pairs = [("u10", "u-wind"), ("v10", "v-wind")]
    else:
        var_pairs = [("swh", "wave height"), ("mwd", "wave direction")]

    results = []
    for var_name, label in var_pairs:
        if var_name not in gcs_ds.data_vars:
            logger.error("Variable %s not found in GCS dataset", var_name)
            continue
        if var_name not in cds_ds.data_vars:
            logger.error("Variable %s not found in CDS dataset", var_name)
            continue

        r = _compare_field(gcs_ds, cds_ds, var_name, label)
        results.append(r)

    gcs_ds.close()
    cds_ds.close()
    return results


def print_results(results: list[dict]) -> None:
    """Print comparison results in a readable table."""
    print("\n" + "=" * 72)
    print("ERA5 GCS vs CDS Comparison Results")
    print("=" * 72)

    all_match = True
    for r in results:
        print(f"\n--- {r['field']} ({r['variable']}) ---")

        if "error" in r:
            print(f"  ERROR: {r['error']}")
            all_match = False
            continue

        print(f"  Valid points:      {r['n_valid_points']:,}")
        print(f"  NaN pattern match: {r['nan_pattern_match']}")
        print(f"    GCS NaN count:   {r['gcs_nan_count']:,}")
        print(f"    CDS NaN count:   {r['cds_nan_count']:,}")
        print(f"  Exact match:       {r['exact_match']}")
        print(f"  Max abs diff:      {r['max_abs_diff']:.2e}")
        print(f"  Mean abs diff:     {r['mean_abs_diff']:.2e}")
        print(f"  Median abs diff:   {r['median_abs_diff']:.2e}")
        print(f"  Max rel diff:      {r['max_rel_diff']:.2e}")
        print(f"  Mean rel diff:     {r['mean_rel_diff']:.2e}")
        print(f"  allclose(atol=1e-5): {r['allclose_atol1e-5']}")
        print(f"  allclose(atol=1e-3): {r['allclose_atol1e-3']}")

        if not r["match"]:
            all_match = False

    print("\n" + "=" * 72)
    if all_match:
        print("RESULT: All variables are EXACT MATCHES between GCS and CDS.")
    else:
        print("RESULT: Differences found between GCS and CDS (see above).")
    print("=" * 72 + "\n")


def main() -> None:
    """Download from GCS and compare against existing CDS files."""
    parser = argparse.ArgumentParser(
        description="Compare ERA5 data from GCS vs CDS.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--corridor",
        choices=("atlantic", "pacific"),
        default="atlantic",
        help="Corridor to compare (default: atlantic).",
    )
    parser.add_argument(
        "--months",
        type=int,
        nargs="+",
        default=[1],
        help="Month(s) to download from GCS (default: 1 = January).",
    )
    parser.add_argument(
        "--year",
        default="2024",
        help="Year (default: 2024).",
    )
    parser.add_argument(
        "--cds-dir",
        default="data/era5",
        help="Directory with existing CDS files (default: data/era5).",
    )
    parser.add_argument(
        "--gcs-dir",
        default="data/era5_gcs_compare",
        help="Directory for GCS downloads (default: data/era5_gcs_compare).",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip GCS download, use existing files in --gcs-dir.",
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        choices=("wind", "waves"),
        default=["wind", "waves"],
        help="Fields to compare (default: both).",
    )
    args = parser.parse_args()

    cds_dir = Path(args.cds_dir)
    gcs_dir = Path(args.gcs_dir)

    # Check CDS files exist
    for field in args.fields:
        cds_file = cds_dir / f"era5_{field}_{args.corridor}_{args.year}.nc"
        if not cds_file.exists():
            print(
                f"CDS file not found: {cds_file}\n"
                f"Download it first with: uv run scripts/download_era5.py "
                f"--backend cds --corridor {args.corridor} --year {args.year}",
                file=sys.stderr,
            )
            sys.exit(1)

    # Download from GCS
    if not args.skip_download:
        from routetools.era5.download_gcs import (
            download_era5_waves_gcs,
            download_era5_wind_gcs,
        )

        gcs_dir.mkdir(parents=True, exist_ok=True)

        for field in args.fields:
            logger.info(
                "Downloading %s/%s months=%s from GCS ...",
                args.corridor,
                field,
                args.months,
            )
            if field == "wind":
                download_era5_wind_gcs(
                    output_dir=gcs_dir,
                    corridor=args.corridor,
                    year=args.year,
                    months=args.months,
                    time_step=6,
                )
            else:
                download_era5_waves_gcs(
                    output_dir=gcs_dir,
                    corridor=args.corridor,
                    year=args.year,
                    months=args.months,
                    time_step=6,
                )

    # Compare each field
    from routetools.era5.download_gcs import _output_filename

    all_results: list[dict] = []
    for field in args.fields:
        gcs_file = _output_filename(
            gcs_dir, field, args.corridor, args.year, args.months
        )
        cds_file = cds_dir / f"era5_{field}_{args.corridor}_{args.year}.nc"

        if not gcs_file.exists():
            logger.error("GCS file not found: %s", gcs_file)
            continue

        results = compare_datasets(gcs_file, cds_file, field)
        all_results.extend(results)

    print_results(all_results)

    # Exit with non-zero if any mismatch
    if not all(r.get("match", False) for r in all_results):
        sys.exit(1)


if __name__ == "__main__":
    main()
