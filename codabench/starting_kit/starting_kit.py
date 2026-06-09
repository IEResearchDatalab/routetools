#!/usr/bin/env python
"""SWOPP3 Starting Kit — Great Circle Baseline.

This script generates a valid submission using the simplest possible
strategy: great-circle routes at constant speed. No external libraries
required beyond the Python standard library.

Replace the great-circle route with your optimizer's output to improve
energy efficiency. Energy values here are placeholders (0.0) — the
CodaBench scorer will re-evaluate all routes using the official RISE
model and ERA5 data.

Usage
-----
::

    python starting_kit.py --output-dir my_submission

Then zip the output directory and upload to CodaBench.
"""

from __future__ import annotations

import argparse
import csv
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

# ─── Configuration ───────────────────────────────────────────────────

TEAM = "MyTeam"
SUBMISSION = 1
N_WAYPOINTS = 100  # Number of waypoints in the route

# Case definitions: (src_lat, src_lon), (dst_lat, dst_lon), passage_hours
CASES = {
    "AO_WPS": {
        "src": (43.6, -4.0),
        "dst": (40.6, -69.0),
        "passage_h": 354,
        "wps": True,
    },
    "AO_noWPS": {
        "src": (43.6, -4.0),
        "dst": (40.6, -69.0),
        "passage_h": 354,
        "wps": False,
    },
    "AGC_WPS": {
        "src": (43.6, -4.0),
        "dst": (40.6, -69.0),
        "passage_h": 354,
        "wps": True,
    },
    "AGC_noWPS": {
        "src": (43.6, -4.0),
        "dst": (40.6, -69.0),
        "passage_h": 354,
        "wps": False,
    },
    "PO_WPS": {
        "src": (34.8, 140.0),
        "dst": (34.4, -121.0),
        "passage_h": 583,
        "wps": True,
    },
    "PO_noWPS": {
        "src": (34.8, 140.0),
        "dst": (34.4, -121.0),
        "passage_h": 583,
        "wps": False,
    },
    "PGC_WPS": {
        "src": (34.8, 140.0),
        "dst": (34.4, -121.0),
        "passage_h": 583,
        "wps": True,
    },
    "PGC_noWPS": {
        "src": (34.8, 140.0),
        "dst": (34.4, -121.0),
        "passage_h": 583,
        "wps": False,
    },
}


def great_circle_waypoints(
    src_lat: float,
    src_lon: float,
    dst_lat: float,
    dst_lon: float,
    n_points: int,
) -> list[tuple[float, float]]:
    """Generate waypoints along a great-circle path.

    Returns list of (lat_deg, lon_deg) tuples.
    """
    lat1 = math.radians(src_lat)
    lon1 = math.radians(src_lon)
    lat2 = math.radians(dst_lat)
    lon2 = math.radians(dst_lon)

    d = math.acos(
        math.sin(lat1) * math.sin(lat2)
        + math.cos(lat1) * math.cos(lat2) * math.cos(lon2 - lon1)
    )

    waypoints = []
    for i in range(n_points):
        f = i / (n_points - 1) if n_points > 1 else 0.0
        if d < 1e-10:
            waypoints.append((src_lat, src_lon))
            continue
        a = math.sin((1 - f) * d) / math.sin(d)
        b = math.sin(f * d) / math.sin(d)
        x = a * math.cos(lat1) * math.cos(lon1) + b * math.cos(lat2) * math.cos(lon2)
        y = a * math.cos(lat1) * math.sin(lon1) + b * math.cos(lat2) * math.sin(lon2)
        z = a * math.sin(lat1) + b * math.sin(lat2)
        lat = math.degrees(math.atan2(z, math.sqrt(x**2 + y**2)))
        lon = math.degrees(math.atan2(y, x))
        waypoints.append((lat, lon))

    return waypoints


def haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance in nautical miles."""
    R_NM = 3440.065  # Earth radius in nm
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return R_NM * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def main() -> None:
    """Generate a great-circle baseline submission for all SWOPP3 cases."""
    parser = argparse.ArgumentParser(description="SWOPP3 baseline: great circle.")
    parser.add_argument("--output-dir", default="submission", help="Output directory.")
    parser.add_argument("--team", default=TEAM, help="Team name.")
    parser.add_argument(
        "--submission", type=int, default=SUBMISSION, help="Submission #."
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    tracks_dir = output_dir / "tracks"
    tracks_dir.mkdir(parents=True, exist_ok=True)

    # 366 daily departures at noon UTC throughout 2024
    departures = [
        datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC) + timedelta(days=d)
        for d in range(366)
    ]
    dtfmt = "%Y-%m-%d %H:%M:%S"

    for case_id, case in CASES.items():
        print(f"\nCase: {case_id}")

        src_lat, src_lon = case["src"]
        dst_lat, dst_lon = case["dst"]
        passage_h = case["passage_h"]

        gc = great_circle_waypoints(src_lat, src_lon, dst_lat, dst_lon, N_WAYPOINTS)

        # Total sailed distance (nm)
        dist_nm = sum(
            haversine_nm(gc[i][0], gc[i][1], gc[i + 1][0], gc[i + 1][1])
            for i in range(len(gc) - 1)
        )

        file_a_rows = []
        for dep_idx, departure in enumerate(departures):
            dep_str = departure.strftime("%Y%m%d")
            arrival = departure + timedelta(hours=passage_h)
            fb_name = f"{args.team}-{args.submission}-{case_id}-{dep_str}.csv"

            file_a_rows.append(
                {
                    "departure_time_utc": departure.strftime(dtfmt),
                    "arrival_time_utc": arrival.strftime(dtfmt),
                    "energy_cons_mwh": "0.000000",  # placeholder — scorer re-evaluates
                    "max_wind_mps": "0.0000",
                    "max_hs_m": "0.0000",
                    "sailed_distance_nm": f"{dist_nm:.4f}",
                    "details_filename": fb_name,
                }
            )

            # Write File B (track waypoints)
            fb_path = tracks_dir / fb_name
            total_seconds = passage_h * 3600
            with fb_path.open("w", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["time_utc", "lat_deg", "lon_deg"]
                )
                writer.writeheader()
                for wp_idx, (wlat, wlon) in enumerate(gc):
                    t = departure + timedelta(
                        seconds=total_seconds * wp_idx / (N_WAYPOINTS - 1)
                    )
                    writer.writerow(
                        {
                            "time_utc": t.strftime(dtfmt),
                            "lat_deg": f"{wlat:.6f}",
                            "lon_deg": f"{wlon:.6f}",
                        }
                    )

            if (dep_idx + 1) % 100 == 0:
                print(f"  {dep_idx + 1}/366 departures done")

        # Write File A
        fa_name = f"{args.team}-{args.submission}-{case_id}.csv"
        fa_path = output_dir / fa_name
        columns = [
            "departure_time_utc",
            "arrival_time_utc",
            "energy_cons_mwh",
            "max_wind_mps",
            "max_hs_m",
            "sailed_distance_nm",
            "details_filename",
        ]
        with fa_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(file_a_rows)

        print(f"  → {fa_name} written ({dist_nm:.0f} nm)")

    print(f"\nDone! Submission in {output_dir}/")
    print("Zip this directory and upload to CodaBench.")
    print("Energy values are placeholders — the scorer will re-evaluate all routes.")


if __name__ == "__main__":
    main()
