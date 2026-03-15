#!/usr/bin/env python
"""SWOPP3 Starting Kit — Great Circle Baseline.

This script generates a valid submission using the simplest possible
strategy: great-circle routes at constant speed. It demonstrates how to:

1. Load ERA5 weather data via routetools.
2. Evaluate routes using the RISE performance model.
3. Write File A and File B CSVs in the required format.

This baseline can be used as a template. Replace the great-circle route
with your optimizer's output to improve energy efficiency.

Usage
-----
::

    pip install routetools
    python starting_kit.py --data-dir data/era5 --output-dir my_submission

Then zip the output directory and upload to CodaBench.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, UTC
from pathlib import Path

import jax.numpy as jnp

from routetools.swopp3 import (
    SWOPP3_CASES,
    departures_2024,
    case_endpoints,
    great_circle_route,
)
from routetools.cost import cost_function_rise
from routetools.era5 import load_era5_windfield, load_era5_wavefield

# ─── Configuration ───────────────────────────────────────────────────

TEAM = "MyTeam"
SUBMISSION = 1
N_WAYPOINTS = 100  # Number of waypoints in the route

# ERA5 file paths (relative to data_dir)
ERA5_FILES = {
    "atlantic": {
        "wind": "era5_wind_atlantic_2024.nc",
        "waves": "era5_waves_atlantic_2024.nc",
    },
    "pacific": {
        "wind": "era5_wind_pacific_2024.nc",
        "waves": "era5_waves_pacific_2024.nc",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="SWOPP3 baseline: great circle.")
    parser.add_argument("--data-dir", default="data/era5", help="ERA5 data directory.")
    parser.add_argument("--output-dir", default="submission", help="Output directory.")
    parser.add_argument("--team", default=TEAM, help="Team name.")
    parser.add_argument("--submission", type=int, default=SUBMISSION, help="Submission #.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    tracks_dir = output_dir / "tracks"
    tracks_dir.mkdir(parents=True, exist_ok=True)

    departures = departures_2024()
    dtfmt = "%Y-%m-%d %H:%M:%S"

    for case_id, case in SWOPP3_CASES.items():
        print(f"\n{'='*60}")
        print(f"Case: {case_id} ({case['label']})")
        print(f"{'='*60}")

        route_id = case["route"]
        passage_hours = case["passage_hours"]
        wps = case["wps"]

        # Build great-circle route
        src, dst = case_endpoints(case_id)
        gc_route = great_circle_route(src, dst, n_points=N_WAYPOINTS)

        # Prepare File A rows
        file_a_rows = []

        for dep_idx, departure in enumerate(departures):
            dep_str = departure.strftime("%Y%m%d")
            dep_time_str = departure.strftime(dtfmt)

            # Load weather for this departure
            wind_file = data_dir / ERA5_FILES[route_id]["wind"]
            wave_file = data_dir / ERA5_FILES[route_id]["waves"]

            if not wind_file.exists() or not wave_file.exists():
                raise FileNotFoundError(
                    f"Missing ERA5 files. Download with:\n"
                    f"  python -m routetools.era5.download --corridor {route_id} --year 2024"
                )

            windfield = load_era5_windfield(
                str(wind_file),
                departure_time=departure.strftime("%Y-%m-%dT%H:%M:%S"),
            )
            wavefield = load_era5_wavefield(
                str(wave_file),
                departure_time=departure.strftime("%Y-%m-%dT%H:%M:%S"),
            )

            # Evaluate energy using RISE model
            curve_batch = gc_route[None, :, :]  # (1, L, 2)
            energy_mwh = float(
                cost_function_rise(
                    windfield=windfield,
                    curve=curve_batch,
                    travel_time=float(passage_hours),
                    wavefield=wavefield,
                    wps=wps,
                    time_offset=0.0,
                )[0]
            )

            # Compute max weather along route
            mid_lon = (gc_route[:-1, 0] + gc_route[1:, 0]) / 2
            mid_lat = (gc_route[:-1, 1] + gc_route[1:, 1]) / 2
            n_seg = len(mid_lon)
            dt_h = passage_hours / n_seg
            seg_times = jnp.array([(i + 0.5) * dt_h for i in range(n_seg)])

            u10, v10 = windfield(mid_lon, mid_lat, seg_times)
            tws = jnp.sqrt(u10**2 + v10**2)
            max_wind = float(jnp.max(tws))

            hs, _ = wavefield(mid_lon, mid_lat, seg_times)
            max_hs = float(jnp.max(hs))

            # Sailed distance (nm)
            from routetools.cost import haversine_distance_from_curve
            dist_m = float(jnp.sum(haversine_distance_from_curve(gc_route)))
            dist_nm = dist_m / 1852.0

            # Arrival time
            arrival = departure + timedelta(hours=passage_hours)

            # File B name
            fb_name = f"{args.team}-{args.submission}-{case_id}-{dep_str}.csv"

            file_a_rows.append({
                "departure_time_utc": dep_time_str,
                "arrival_time_utc": arrival.strftime(dtfmt),
                "energy_cons_mwh": f"{energy_mwh:.6f}",
                "max_wind_mps": f"{max_wind:.4f}",
                "max_hs_m": f"{max_hs:.4f}",
                "sailed_distance_nm": f"{dist_nm:.4f}",
                "details_filename": fb_name,
            })

            # Write File B (track waypoints)
            fb_path = tracks_dir / fb_name
            total_seconds = passage_hours * 3600
            with fb_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["time_utc", "lat_deg", "lon_deg"])
                writer.writeheader()
                for wp_idx in range(N_WAYPOINTS):
                    t = departure + timedelta(
                        seconds=total_seconds * wp_idx / (N_WAYPOINTS - 1)
                    )
                    writer.writerow({
                        "time_utc": t.strftime(dtfmt),
                        "lat_deg": f"{float(gc_route[wp_idx, 1]):.6f}",
                        "lon_deg": f"{float(gc_route[wp_idx, 0]):.6f}",
                    })

            if (dep_idx + 1) % 30 == 0:
                print(f"  Processed {dep_idx + 1}/366 departures "
                      f"(energy={energy_mwh:.1f} MWh)")

        # Write File A
        fa_name = f"{args.team}-{args.submission}-{case_id}.csv"
        fa_path = output_dir / fa_name
        columns = [
            "departure_time_utc", "arrival_time_utc", "energy_cons_mwh",
            "max_wind_mps", "max_hs_m", "sailed_distance_nm", "details_filename",
        ]
        with fa_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(file_a_rows)

        total_case_energy = sum(float(r["energy_cons_mwh"]) for r in file_a_rows)
        print(f"  → {fa_name}: total energy = {total_case_energy:.1f} MWh")

    print(f"\nDone! Submission files written to {output_dir}/")
    print(f"Zip this directory and upload to CodaBench.")


if __name__ == "__main__":
    main()
