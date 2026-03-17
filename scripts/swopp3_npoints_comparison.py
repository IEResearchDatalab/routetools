#!/usr/bin/env python
"""Resample existing optimized tracks and compare energy at different resolutions.

Takes the already-optimized 584-point (Pacific) or 355-point (Atlantic)
tracks, resamples them to fewer waypoints (e.g. 50, 100, 200), and
re-evaluates energy to see how reporting resolution affects results.

Automatically selects the best, median, and worst departures by energy.

Usage
-----
::

    uv run scripts/swopp3_npoints_comparison.py --corridor pacific
    uv run scripts/swopp3_npoints_comparison.py --corridor pacific --case PO_noWPS
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def resample_curve(curve: np.ndarray, n_out: int) -> np.ndarray:
    """Resample a (L, 2) curve to n_out points by arc-length interpolation.

    Uses cumulative haversine distances to place n_out evenly-spaced
    points along the original route, preserving the exact start and end.
    """
    from routetools._cost.haversine import haversine_distance_from_curve
    import jax.numpy as jnp

    seg_dist = np.asarray(
        haversine_distance_from_curve(jnp.array(curve)), dtype=np.float64
    )
    cum_dist = np.concatenate(([0.0], np.cumsum(seg_dist)))
    total = cum_dist[-1]

    # Target arc-lengths for the resampled points
    target = np.linspace(0, total, n_out)

    # Interpolate lon and lat independently along the arc
    lon_out = np.interp(target, cum_dist, curve[:, 0])
    lat_out = np.interp(target, cum_dist, curve[:, 1])
    return np.column_stack([lon_out, lat_out])


def load_track(path: Path) -> tuple[np.ndarray, datetime]:
    """Load a track CSV (time_utc, lat_deg, lon_deg) -> (L,2) lon/lat array + departure."""
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    lons = np.array([float(r["lon_deg"]) for r in rows])
    lats = np.array([float(r["lat_deg"]) for r in rows])
    dep = datetime.strptime(rows[0]["time_utc"], "%Y-%m-%d %H:%M:%S").replace(
        tzinfo=timezone.utc
    )
    return np.column_stack([lons, lats]), dep


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resample existing tracks and compare energy at different resolutions.",
    )
    parser.add_argument(
        "--corridor",
        choices=["atlantic", "pacific"],
        default="pacific",
        help="Corridor (default: pacific).",
    )
    parser.add_argument(
        "--case",
        default=None,
        help="Case ID (e.g. PO_WPS). Default: auto from corridor + wps.",
    )
    parser.add_argument(
        "--n-points",
        type=int,
        nargs="+",
        default=[50, 100, 200],
        help="Resampling resolutions to test. Native (584/355) always included.",
    )
    parser.add_argument(
        "--wind-path",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--wave-path",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--tracks-dir",
        type=Path,
        default=Path("output/swopp3_gpu/tracks"),
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=None,
        help="File-A CSV with energy values. Default: auto-detected.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/npoints_comparison"),
    )
    parser.add_argument(
        "--no-wps",
        action="store_true",
        help="Use noWPS case instead of WPS.",
    )
    args = parser.parse_args()

    wps = not args.no_wps
    case_id = args.case or (
        f"{'P' if args.corridor == 'pacific' else 'A'}O_{'WPS' if wps else 'noWPS'}"
    )
    native_npoints = {"pacific": 584, "atlantic": 355}
    native = native_npoints[args.corridor]
    npoints_list = sorted(set(args.n_points) | {native})

    # ---- Find file-A CSV to pick best/median/worst ----
    results_csv = args.results_csv or (
        args.tracks_dir.parent / f"IEUniversity-1-{case_id}.csv"
    )
    if not results_csv.exists():
        print(f"ERROR: Results CSV not found: {results_csv}", file=sys.stderr)
        sys.exit(1)

    # Parse energies per departure
    with open(results_csv) as f:
        reader = csv.DictReader(f)
        file_a = list(reader)

    energies = [(r["departure_time_utc"], float(r["energy_cons_mwh"])) for r in file_a]
    energies.sort(key=lambda x: x[1])

    best = energies[0]
    worst = energies[-1]
    median = energies[len(energies) // 2]
    selected = [
        ("best", best[0], best[1]),
        ("median", median[0], median[1]),
        ("worst", worst[0], worst[1]),
    ]

    print("=" * 75)
    print(f"Resample comparison — {case_id}")
    print(f"  Native L:  {native}")
    print(f"  Test L:    {npoints_list}")
    print(f"  Selected departures:")
    for label, dep_str, orig_e in selected:
        print(f"    {label:>6}: {dep_str}  ({orig_e:.1f} MWh original)")
    print("=" * 75)

    # ---- Data paths ----
    data_dir = Path("data/era5")
    wind_path = args.wind_path or data_dir / f"era5_wind_{args.corridor}_2024.nc"
    wave_path = args.wave_path or data_dir / f"era5_waves_{args.corridor}_2024.nc"
    for p in (wind_path, wave_path):
        if not p.exists():
            print(f"ERROR: Missing data file: {p}", file=sys.stderr)
            sys.exit(1)

    # ---- Heavy imports ----
    import jax.numpy as jnp

    from routetools.cost import evaluate_route_energy
    from routetools.era5.loader import (
        load_dataset_epoch,
        load_era5_wavefield,
        load_era5_windfield,
    )
    from routetools.swopp3 import SWOPP3_CASES

    case = SWOPP3_CASES[case_id]
    passage_hours = float(case["passage_hours"])

    print("\nLoading ERA5 fields...")
    t0 = time.time()
    windfield = load_era5_windfield(wind_path)
    epoch = load_dataset_epoch(wind_path)
    wavefield = load_era5_wavefield(wave_path)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # ---- Evaluate ----
    results: list[dict] = []

    for label, dep_str, orig_energy in selected:
        dep_dt = datetime.strptime(dep_str, "%Y-%m-%d %H:%M:%S").replace(
            tzinfo=timezone.utc
        )
        dep_date = dep_dt.strftime("%Y%m%d")
        dep_offset_h = (dep_dt - epoch).total_seconds() / 3600.0

        # Load the original track
        track_name = f"IEUniversity-1-{case_id}-{dep_date}.csv"
        track_path = args.tracks_dir / track_name
        if not track_path.exists():
            print(f"  WARNING: Track file not found: {track_path}, skipping")
            continue

        curve_orig, _ = load_track(track_path)
        actual_L = curve_orig.shape[0]

        print(f"\n{'─' * 65}")
        print(f"{label.upper()} departure: {dep_str}  "
              f"(original: {orig_energy:.1f} MWh, L={actual_L})")
        print(f"  {'L':>6}  {'Energy MWh':>12}  {'Dist nm':>10}  "
              f"{'max TWS':>8}  {'max Hs':>8}  {'Δ energy':>10}")

        for npts in npoints_list:
            if npts == native or npts == actual_L:
                curve = jnp.array(curve_orig)
                tag = " (native)"
            else:
                curve = jnp.array(resample_curve(curve_orig, npts))
                tag = ""

            energy_mwh, max_tws, max_hs = evaluate_route_energy(
                curve,
                passage_hours,
                wps=wps,
                windfield=windfield,
                wavefield=wavefield,
                departure_offset_h=dep_offset_h,
            )

            from routetools.swopp3_output import sailed_distance_nm
            dist_nm = sailed_distance_nm(curve)

            pct = ((energy_mwh - orig_energy) / orig_energy) * 100

            print(
                f"  {npts:>6}{tag:<9} {energy_mwh:>10.2f}  {dist_nm:>10.1f}  "
                f"{max_tws:>8.2f}  {max_hs:>8.2f}  {pct:>+9.2f}%"
            )

            results.append({
                "label": label,
                "departure": dep_str,
                "original_mwh": orig_energy,
                "n_points": npts,
                "energy_mwh": float(energy_mwh),
                "distance_nm": float(dist_nm),
                "max_tws_mps": float(max_tws),
                "max_hs_m": float(max_hs),
                "delta_pct": float(pct),
            })

    # ---- Summary ----
    print(f"\n{'=' * 75}")
    print("SUMMARY — mean absolute Δ energy vs native")
    print(f"{'=' * 75}")
    for npts in npoints_list:
        rows = [r for r in results if r["n_points"] == npts]
        if not rows:
            continue
        mean_delta = np.mean([abs(r["delta_pct"]) for r in rows])
        tag = " (native)" if npts == native else ""
        print(f"  L={npts:>4}{tag:<10}  mean |Δ| = {mean_delta:.2f}%")

    # ---- Save CSV ----
    if results:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = args.output_dir / f"resample_{case_id}.csv"
        with open(csv_path, "w") as f:
            cols = list(results[0].keys())
            f.write(",".join(cols) + "\n")
            for row in results:
                f.write(",".join(str(row[c]) for c in cols) + "\n")
        print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
