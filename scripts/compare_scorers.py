#!/usr/bin/env python
"""Compare routetools JAX scorer vs self-contained numpy scorer on one route.

Usage:
    python scripts/compare_scorers.py [--case AO_noWPS] [--dep 2024-01-01]

Loads Atlantic ERA5 data with both routetools (JAX) and the numpy scorer,
evaluates one route from the output tracks, and prints intermediate values
side-by-side to find the source of divergence.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from routetools.cost import cost_function_rise
from routetools.era5 import load_era5_wavefield, load_era5_windfield

BASE_DIR = Path(__file__).resolve().parent.parent
SCORING_PROGRAM_DIR = BASE_DIR / "codabench" / "scoring_program"
DATA_DIR = BASE_DIR / "data" / "era5"
CODABENCH_REF = BASE_DIR / "codabench" / "reference_data"
TRACKS_DIR = BASE_DIR / "output" / "swopp3_0125_rust" / "tracks"

# Import RISE constants and helpers from the scoring program
sys.path.insert(0, str(SCORING_PROGRAM_DIR))
from scoring import (  # noqa: E402
    CASE_DEFS,
    _forward_bearing_deg,
    _haversine_m,
    _interp_era5,
    _load_era5_grid,
    _rise_power,
)

DTFMT = "%Y-%m-%d %H:%M:%S"


def load_track(case_id: str, dep_date: str) -> list[tuple[datetime, float, float]]:
    """Load waypoints from a track file."""
    dep_str = dep_date.replace("-", "")
    fname = f"IEUniversity-1-{case_id}-{dep_str}.csv"
    path = TRACKS_DIR / fname
    if not path.exists():
        raise FileNotFoundError(f"Track file not found: {path}")
    waypoints = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = datetime.strptime(row["time_utc"], DTFMT)
            lat = float(row["lat_deg"])
            lon = float(row["lon_deg"])
            waypoints.append((t, lat, lon))
    return waypoints


def main():
    """Compare one routetools route evaluation against the numpy scorer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="AO_noWPS", help="Case ID")
    parser.add_argument("--dep", default="2024-01-01", help="Departure date YYYY-MM-DD")
    args = parser.parse_args()

    case_id = args.case
    dep_date = args.dep
    case_def = CASE_DEFS[case_id]
    corridor = case_def["route"]
    passage_h = float(case_def["passage_h"])
    wps = case_def["wps"]

    print(f"Case: {case_id}, Corridor: {corridor}, Passage: {passage_h}h, WPS: {wps}")
    print(f"Departure: {dep_date}")
    print()

    # Load track
    waypoints = load_track(case_id, dep_date)
    n_wp = len(waypoints)
    n_seg = n_wp - 1
    dt_h = passage_h / n_seg
    print(f"Track: {n_wp} waypoints, {n_seg} segments, dt_h={dt_h:.4f}")

    lats = np.array([wp[1] for wp in waypoints])
    lons = np.array([wp[2] for wp in waypoints])
    dep_dt = waypoints[0][0]
    dep_str = dep_dt.strftime(DTFMT)

    # ================================================================
    # ROUTETOOLS JAX evaluation
    # ================================================================
    print("\n" + "=" * 70)
    print("ROUTETOOLS (JAX) EVALUATION")
    print("=" * 70)

    # Find ERA5 files (prefer 6-hourly from codabench/reference_data)
    wind_files = []
    wave_files = []
    for d in (CODABENCH_REF, DATA_DIR):
        wf = d / f"era5_wind_{corridor}_2024.nc"
        vf = d / f"era5_waves_{corridor}_2024.nc"
        if wf.exists() and vf.exists():
            wind_files = [str(wf)]
            wave_files = [str(vf)]
            for suf in (
                f"era5_wind_{corridor}_2025_01.nc",
                f"era5_wind_{corridor}_2025.nc",
            ):
                p = d / suf
                if p.exists():
                    wind_files.append(str(p))
                    break
            for suf in (
                f"era5_waves_{corridor}_2025_01.nc",
                f"era5_waves_{corridor}_2025.nc",
            ):
                p = d / suf
                if p.exists():
                    wave_files.append(str(p))
                    break
            print(f"Using ERA5 data from: {d}")
            break
    else:
        print("ERROR: No ERA5 data found!")
        return

    # Build curve array: shape (1, L, 2) with (lon, lat) per routetools convention
    curve_jax = jnp.array(
        [[lon, lat] for lat, lon in zip(lats, lons, strict=False)], dtype=jnp.float32
    )[None, :, :]  # (1, L, 2)

    # Load fields with departure_time
    windfield = load_era5_windfield(wind_files, departure_time=dep_str)
    wavefield = load_era5_wavefield(wave_files, departure_time=dep_str)

    # Full energy via cost_function_rise
    energy_jax = float(
        cost_function_rise(
            windfield=windfield,
            curve=curve_jax,
            travel_time=passage_h,
            wavefield=wavefield,
            wps=wps,
            time_offset=0.0,
        )[0]
    )
    print(f"\nJAX energy: {energy_jax:.6f} MWh")

    # Now compute intermediate values for comparison
    mid_lon_jax = (curve_jax[0, :-1, 0] + curve_jax[0, 1:, 0]) / 2
    mid_lat_jax = (curve_jax[0, :-1, 1] + curve_jax[0, 1:, 1]) / 2
    seg_times_jax = (jnp.arange(n_seg) + 0.5) * dt_h  # relative to departure

    # Interpolate weather
    u10_jax, v10_jax = windfield(mid_lon_jax, mid_lat_jax, seg_times_jax)
    hs_jax, mwd_jax = wavefield(mid_lon_jax, mid_lat_jax, seg_times_jax)

    # Bearing
    lon1 = jnp.radians(curve_jax[0, :-1, 0])
    lon2 = jnp.radians(curve_jax[0, 1:, 0])
    lat1 = jnp.radians(curve_jax[0, :-1, 1])
    lat2 = jnp.radians(curve_jax[0, 1:, 1])
    dlon = lon2 - lon1
    x_b = jnp.sin(dlon) * jnp.cos(lat2)
    y_b = jnp.cos(lat1) * jnp.sin(lat2) - jnp.sin(lat1) * jnp.cos(lat2) * jnp.cos(dlon)
    bearing_jax = jnp.mod(jnp.degrees(jnp.arctan2(x_b, y_b)), 360.0)

    # Wind angles
    tws_jax = jnp.sqrt(u10_jax**2 + v10_jax**2)
    wind_from_jax = jnp.mod(180.0 + jnp.degrees(jnp.arctan2(u10_jax, v10_jax)), 360.0)
    twa_jax = jnp.mod(wind_from_jax - bearing_jax, 360.0)
    mwa_jax = jnp.mod(mwd_jax - bearing_jax, 360.0)

    # Distance and speed (routetools uses equirectangular approx)
    from routetools._cost.haversine import haversine_meters_components

    dx_m, dy_m = haversine_meters_components(
        curve_jax[0, :-1, 1],
        curve_jax[0, :-1, 0],
        curve_jax[0, 1:, 1],
        curve_jax[0, 1:, 0],
    )
    dist_jax = jnp.sqrt(dx_m**2 + dy_m**2)
    v_mps_jax = dist_jax / (dt_h * 3600.0)

    # RISE power
    from routetools.performance import predict_power_jax

    power_jax = predict_power_jax(tws_jax, twa_jax, hs_jax, mwa_jax, v_mps_jax, wps=wps)

    # ================================================================
    # NUMPY SCORER evaluation
    # ================================================================
    print("\n" + "=" * 70)
    print("NUMPY SCORER EVALUATION")
    print("=" * 70)

    wind_grid = _load_era5_grid(wind_files)
    wave_grid = _load_era5_grid(wave_files)

    mid_lat_np = (lats[:-1] + lats[1:]) / 2
    mid_lon_np = (lons[:-1] + lons[1:]) / 2

    dep_dt64 = np.datetime64(dep_str)
    dep_offset_h = float((dep_dt64 - wind_grid["t0"]) / np.timedelta64(1, "h"))
    seg_times_np = dep_offset_h + (np.arange(n_seg) + 0.5) * dt_h

    u10_np = _interp_era5(wind_grid, "u10", mid_lat_np, mid_lon_np, seg_times_np)
    v10_np = _interp_era5(wind_grid, "v10", mid_lat_np, mid_lon_np, seg_times_np)
    swh_np = _interp_era5(wave_grid, "swh", mid_lat_np, mid_lon_np, seg_times_np)
    mwd_np = _interp_era5(wave_grid, "mwd", mid_lat_np, mid_lon_np, seg_times_np)
    np.nan_to_num(swh_np, copy=False, nan=0.0)
    np.nan_to_num(mwd_np, copy=False, nan=0.0)

    dist_np = _haversine_m(lats[:-1], lons[:-1], lats[1:], lons[1:])
    v_mps_np = dist_np / (dt_h * 3600.0)

    bearing_np = _forward_bearing_deg(lats[:-1], lons[:-1], lats[1:], lons[1:])

    tws_np = np.sqrt(u10_np**2 + v10_np**2)
    wind_from_np = np.mod(180.0 + np.degrees(np.arctan2(u10_np, v10_np)), 360.0)
    twa_np = np.mod(wind_from_np - bearing_np, 360.0)
    mwa_np = np.mod(mwd_np - bearing_np, 360.0)

    power_np = _rise_power(tws_np, twa_np, swh_np, mwa_np, v_mps_np, wps)
    energy_np = float(np.sum(power_np) * dt_h / 1000.0)

    print(f"\nNumpy energy: {energy_np:.6f} MWh")

    # ================================================================
    # COMPARISON
    # ================================================================
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    print(
        f"\nEnergy — JAX: {energy_jax:.6f}  Numpy: {energy_np:.6f}  "
        f"diff: {abs(energy_jax - energy_np):.6f} MWh "
        f"({100 * abs(energy_jax - energy_np) / energy_jax:.2f}%)"
    )

    # Show first 5 and last 5 segments
    segments_to_show = list(range(min(5, n_seg))) + list(
        range(max(0, n_seg - 5), n_seg)
    )
    segments_to_show = sorted(set(segments_to_show))

    def pct(a, b):
        if a == 0:
            return 0.0
        return 100 * abs(a - b) / abs(a)

    print(
        f"\n{'seg':>4} | {'u10_jax':>10} {'u10_np':>10} {'err%':>6} | "
        f"{'v10_jax':>10} {'v10_np':>10} {'err%':>6}"
    )
    print("-" * 80)
    for i in segments_to_show:
        print(
            f"{i:4d} | {float(u10_jax[i]):10.4f} {float(u10_np[i]):10.4f} "
            f"{pct(float(u10_jax[i]), float(u10_np[i])):6.2f} | "
            f"{float(v10_jax[i]):10.4f} {float(v10_np[i]):10.4f} "
            f"{pct(float(v10_jax[i]), float(v10_np[i])):6.2f}"
        )

    print(
        f"\n{'seg':>4} | {'bear_jax':>10} {'bear_np':>10} {'err%':>6} | "
        f"{'dist_jax':>12} {'dist_np':>12} {'err%':>6}"
    )
    print("-" * 80)
    for i in segments_to_show:
        print(
            f"{i:4d} | {float(bearing_jax[i]):10.4f} {float(bearing_np[i]):10.4f} "
            f"{pct(float(bearing_jax[i]), float(bearing_np[i])):6.2f} | "
            f"{float(dist_jax[i]):12.2f} {float(dist_np[i]):12.2f} "
            f"{pct(float(dist_jax[i]), float(dist_np[i])):6.2f}"
        )

    print(
        f"\n{'seg':>4} | {'tws_jax':>10} {'tws_np':>10} {'err%':>6} | "
        f"{'twa_jax':>10} {'twa_np':>10} {'err%':>6}"
    )
    print("-" * 80)
    for i in segments_to_show:
        print(
            f"{i:4d} | {float(tws_jax[i]):10.4f} {float(tws_np[i]):10.4f} "
            f"{pct(float(tws_jax[i]), float(tws_np[i])):6.2f} | "
            f"{float(twa_jax[i]):10.4f} {float(twa_np[i]):10.4f} "
            f"{pct(float(twa_jax[i]), float(twa_np[i])):6.2f}"
        )

    print(
        f"\n{'seg':>4} | {'hs_jax':>10} {'hs_np':>10} {'err%':>6} | "
        f"{'mwa_jax':>10} {'mwa_np':>10} {'err%':>6}"
    )
    print("-" * 80)
    for i in segments_to_show:
        print(
            f"{i:4d} | {float(hs_jax[i]):10.4f} {float(swh_np[i]):10.4f} "
            f"{pct(float(hs_jax[i]), float(swh_np[i])):6.2f} | "
            f"{float(mwa_jax[i]):10.4f} {float(mwa_np[i]):10.4f} "
            f"{pct(float(mwa_jax[i]), float(mwa_np[i])):6.2f}"
        )

    print(
        f"\n{'seg':>4} | {'v_jax':>10} {'v_np':>10} {'err%':>6} | "
        f"{'pow_jax':>12} {'pow_np':>12} {'err%':>6}"
    )
    print("-" * 80)
    for i in segments_to_show:
        print(
            f"{i:4d} | {float(v_mps_jax[i]):10.4f} {float(v_mps_np[i]):10.4f} "
            f"{pct(float(v_mps_jax[i]), float(v_mps_np[i])):6.2f} | "
            f"{float(power_jax[i]):12.4f} {float(power_np[i]):12.4f} "
            f"{pct(float(power_jax[i]), float(power_np[i])):6.2f}"
        )

    # Aggregate comparison
    print("\nAggregate diffs (RMS):")
    print(f"  u10:     {np.sqrt(np.mean((np.array(u10_jax) - u10_np) ** 2)):.6f}")
    print(f"  v10:     {np.sqrt(np.mean((np.array(v10_jax) - v10_np) ** 2)):.6f}")
    print(f"  swh:     {np.sqrt(np.mean((np.array(hs_jax) - swh_np) ** 2)):.6f}")
    print(f"  mwd:     {np.sqrt(np.mean((np.array(mwd_jax) - mwd_np) ** 2)):.6f}")
    print(
        f"  bearing: {np.sqrt(np.mean((np.array(bearing_jax) - bearing_np) ** 2)):.6f}"
    )
    print(f"  dist:    {np.sqrt(np.mean((np.array(dist_jax) - dist_np) ** 2)):.6f}")
    print(f"  v_mps:   {np.sqrt(np.mean((np.array(v_mps_jax) - v_mps_np) ** 2)):.6f}")
    print(f"  tws:     {np.sqrt(np.mean((np.array(tws_jax) - tws_np) ** 2)):.6f}")
    print(f"  twa:     {np.sqrt(np.mean((np.array(twa_jax) - twa_np) ** 2)):.6f}")
    print(f"  mwa:     {np.sqrt(np.mean((np.array(mwa_jax) - mwa_np) ** 2)):.6f}")
    print(f"  power:   {np.sqrt(np.mean((np.array(power_jax) - power_np) ** 2)):.6f}")


if __name__ == "__main__":
    main()
