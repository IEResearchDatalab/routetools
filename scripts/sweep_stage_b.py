#!/usr/bin/env python3
r"""Stage B parameter sweep — unstick GC-stuck Pacific departures.

Runs a grid search over exploration/convergence parameters on a
representative sample of Pacific noWPS departures where the current
optimizer returns bit-for-bit identical energy to the great-circle
baseline (zero-delta departures).

Target departures (0-indexed, sampled 20 of 137 zero-delta):
    44, 45, 46, 111, 113, 116, 118, 125, 143, 159,
    160, 169, 174, 177, 185, 186, 232, 233, 239, 274

Grid:
    sigma0   : 0.1, 0.3, 0.5
    popsize  : 200, 400
    maxfevals: 25000, 50000
    K        : 10, 15

Results are written to a single CSV for easy comparison.

Usage
-----
Run from the repo root with the virtual-env activated::

    python scripts/sweep_stage_b.py \
        --wind-path data/era5/era5_wind_pacific_2024.nc \
        --wave-path data/era5/era5_waves_pacific_2024.nc

On SLURM / HPC, wrap in ``srun`` or submit via ``sbatch``.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Target departure indices (0-based day-of-year)
# Representative sample of 20 zero-delta Pacific noWPS departures.
# ---------------------------------------------------------------------------
STAGE_B_DEPARTURES = [
    44,
    45,
    46,
    111,
    113,
    116,
    118,
    125,
    143,
    159,
    160,
    169,
    174,
    177,
    185,
    186,
    232,
    233,
    239,
    274,
]

# Parameter grid — exploration / convergence knobs
SIGMA0_VALUES = [0.1, 0.3, 0.5]
POPSIZE_VALUES = [200, 400]
MAXFEVALS_VALUES = [25_000, 50_000]
K_VALUES = [10, 15]

CASE_ID = "PO_noWPS"
GC_CASE_ID = "PGC_noWPS"


def main() -> None:
    """Run the Stage B Pacific exploration sweep and write the results CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wind-path",
        type=Path,
        default=Path("data/era5/era5_wind_pacific_2024.nc"),
    )
    parser.add_argument(
        "--wave-path",
        type=Path,
        default=Path("data/era5/era5_waves_pacific_2024.nc"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/sweep_stage_b.csv"),
        help="Output CSV path (default: output/sweep_stage_b.csv)",
    )
    parser.add_argument("--n-points", type=int, default=100, help="Route waypoints (L)")
    args = parser.parse_args()

    for p in [args.wind_path, args.wave_path]:
        if not p.exists():
            sys.exit(f"Missing data file: {p}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # ---- Lazy imports (heavy) ----
    import xarray as xr

    from routetools.era5.loader import (
        load_dataset_epoch,
        load_era5_wavefield,
        load_era5_windfield,
        load_natural_earth_land_mask,
    )
    from routetools.swopp3 import departures_2024
    from routetools.swopp3_runner import (
        run_gc_departure,
        run_optimised_departure,
    )

    # ---- Load fields once ----
    print("Loading ERA5 wind …")
    windfield = load_era5_windfield(args.wind_path)
    wind_epoch = load_dataset_epoch(args.wind_path)
    print("Loading ERA5 waves …")
    wavefield = load_era5_wavefield(args.wave_path)
    vectorfield = windfield  # same closure

    # Land mask
    with xr.open_dataset(args.wind_path) as ds:
        for cname in ("longitude", "lon"):
            if cname in ds.coords:
                lons = ds[cname].values
                break
        for cname in ("latitude", "lat"):
            if cname in ds.coords:
                lats = ds[cname].values
                break
    lon_range = (float(lons.min()), float(lons.max()))
    lat_range = (float(lats.min()), float(lats.max()))
    print("Building land mask …")
    land = load_natural_earth_land_mask(lon_range, lat_range)

    # ---- Departures ----
    all_departures = departures_2024()
    target_deps = [(idx, all_departures[idx]) for idx in STAGE_B_DEPARTURES]

    # ---- Compute GC baseline for comparison ----
    print(f"\nComputing GC baseline for {len(target_deps)} departures …")
    gc_energies: dict[int, float] = {}
    for idx, dep in target_deps:
        dep_naive = dep.replace(tzinfo=None) if dep.tzinfo else dep
        epoch_naive = wind_epoch.replace(tzinfo=None)
        offset_h = (dep_naive - epoch_naive).total_seconds() / 3600.0
        gc_result = run_gc_departure(
            GC_CASE_ID,
            dep,
            windfield=windfield,
            wavefield=wavefield,
            departure_offset_h=offset_h,
            n_points=args.n_points,
        )
        gc_energies[idx] = gc_result.energy_mwh
        print(f"  dep {idx} GC: {gc_result.energy_mwh:.2f} MWh")

    # ---- Parameter grid ----
    grid = list(
        itertools.product(SIGMA0_VALUES, POPSIZE_VALUES, MAXFEVALS_VALUES, K_VALUES)
    )
    total_runs = len(grid) * len(target_deps)
    print(
        f"\nSweeping {len(grid)} configs × {len(target_deps)} departures "
        f"= {total_runs} runs\n"
    )

    # ---- Run sweep ----
    fieldnames = [
        "sigma0",
        "popsize",
        "maxfevals",
        "K",
        "dep_index",
        "dep_date",
        "energy_opt_mwh",
        "energy_gc_mwh",
        "delta_pct",
        "distance_nm",
        "max_tws_mps",
        "max_hs_m",
        "comp_time_s",
    ]

    with open(args.output, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        run_count = 0
        t_total_start = time.time()

        for sig, pop, mfe, k in grid:
            print(f"--- sigma0={sig}, popsize={pop}, maxfevals={mfe}, K={k} ---")
            for idx, dep in target_deps:
                run_count += 1
                dep_naive = dep.replace(tzinfo=None) if dep.tzinfo else dep
                epoch_naive = wind_epoch.replace(tzinfo=None)
                offset_h = (dep_naive - epoch_naive).total_seconds() / 3600.0

                result = run_optimised_departure(
                    CASE_ID,
                    dep,
                    vectorfield=vectorfield,
                    windfield=windfield,
                    wavefield=wavefield,
                    land=land,
                    departure_offset_h=offset_h,
                    n_points=args.n_points,
                    sigma0=sig,
                    popsize=pop,
                    maxfevals=mfe,
                    K=k,
                )

                e_gc = gc_energies[idx]
                delta = (result.energy_mwh - e_gc) / e_gc * 100 if e_gc else 0.0

                row = {
                    "sigma0": sig,
                    "popsize": pop,
                    "maxfevals": mfe,
                    "K": k,
                    "dep_index": idx,
                    "dep_date": dep.strftime("%Y-%m-%d"),
                    "energy_opt_mwh": f"{result.energy_mwh:.4f}",
                    "energy_gc_mwh": f"{e_gc:.4f}",
                    "delta_pct": f"{delta:.2f}",
                    "distance_nm": f"{result.distance_nm:.1f}",
                    "max_tws_mps": f"{result.max_tws_mps:.2f}",
                    "max_hs_m": f"{result.max_hs_m:.2f}",
                    "comp_time_s": f"{result.comp_time_s:.1f}",
                }
                writer.writerow(row)
                csvfile.flush()

                status = "BETTER" if delta < -1 else ("WORSE" if delta > 1 else "~SAME")
                print(
                    f"  [{run_count}/{total_runs}] dep {idx} "
                    f"E={result.energy_mwh:.1f} ({delta:+.1f}%) "
                    f"d={result.distance_nm:.0f}nm  {status}  "
                    f"t={result.comp_time_s:.1f}s"
                )

        elapsed = time.time() - t_total_start
        print(f"\nDone. {total_runs} runs in {elapsed:.0f}s")
        print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
