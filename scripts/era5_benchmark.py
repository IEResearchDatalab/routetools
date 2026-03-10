#!/usr/bin/env python
"""Real-world benchmark: USNYC → DEHAM using ERA5 weather data.

This script demonstrates the full ERA5-based weather routing pipeline:

1. Load ERA5 wind and wave fields from downloaded NetCDF files.
2. Create a Natural Earth high-resolution land mask.
3. Optimize a route from New York (USNYC) to Hamburg (DEHAM) departing
   on 2023-01-08 00:00 UTC, at 12 knots through water.
4. Refine the route with the FMS smoother.
5. Plot and save the result.

Prerequisites
-------------
Download ERA5 data for the Atlantic corridor first::

    uv run scripts/download_era5.py --corridor atlantic --year 2023

This creates ``data/era5/era5_wind_atlantic_2023.nc`` and
``data/era5/era5_waves_atlantic_2023.nc``.

Usage
-----
::

    uv run scripts/era5_benchmark.py

To change the departure date::

    uv run scripts/era5_benchmark.py --departure 2023-02-15

"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt

from routetools._ports import DICT_PORTS
from routetools.cmaes import optimize
from routetools.cost import haversine_distance_from_curve
from routetools.era5 import (
    load_era5_wavefield,
    load_era5_windfield,
    load_natural_earth_land_mask,
)
from routetools.fms import optimize_fms
from routetools.plot import plot_curve


def main() -> None:
    """Run an end-to-end ERA5 benchmark from USNYC to DEHAM."""
    parser = argparse.ArgumentParser(
        description="ERA5-based benchmark: USNYC → DEHAM.",
    )
    parser.add_argument(
        "--departure",
        default="2023-01-08T00:00:00",
        help="Departure datetime (default: 2023-01-08T00:00:00).",
    )
    parser.add_argument(
        "--vel-ship",
        type=float,
        default=6.0,
        help="Speed through water in m/s (≈12 kn). Default: 6.0.",
    )
    parser.add_argument(
        "--data-dir",
        default="data/era5",
        help="Directory containing ERA5 NetCDF files.",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory for output plots.",
    )
    parser.add_argument(
        "--K", type=int, default=10, help="Free Bézier control points (default: 10)."
    )
    parser.add_argument(
        "--L", type=int, default=320, help="Curve discretisation points (default: 320)."
    )
    parser.add_argument(
        "--popsize", type=int, default=500, help="CMA-ES population (default: 500)."
    )
    parser.add_argument(
        "--maxfevals",
        type=int,
        default=int(1e8),
        help="Max CMA-ES evaluations (default: 1e8).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wind_file = data_dir / "era5_wind_atlantic_2023.nc"
    wave_file = data_dir / "era5_waves_atlantic_2023.nc"

    for f in (wind_file, wave_file):
        if not f.exists():
            raise FileNotFoundError(
                f"Missing {f}. Download first with:\n"
                "  uv run scripts/download_era5.py --corridor atlantic --year 2023"
            )

    # ── Load ERA5 fields ──────────────────────────────────────────────
    print(f"[ERA5] Loading wind from {wind_file}")
    windfield = load_era5_windfield(wind_file, departure_time=args.departure)

    print(f"[ERA5] Loading waves from {wave_file}")
    wavefield = load_era5_wavefield(wave_file, departure_time=args.departure)

    # No ocean current data — use a zero vectorfield (SWOPP uses wind
    # via the performance model, not as current in the cost function).
    def vectorfield(lon, lat, t):
        return jnp.zeros_like(lon), jnp.zeros_like(lon)

    # ── Build land mask ───────────────────────────────────────────────
    # Atlantic corridor bounds (must cover Hamburg at ~54°N going around UK)
    lon_range = (-80.0, 10.0)
    lat_range = (25.0, 60.0)

    print("[ERA5] Building Natural Earth land mask...")
    land = load_natural_earth_land_mask(lon_range=lon_range, lat_range=lat_range)

    # ── Source / destination ──────────────────────────────────────────
    src = jnp.array([DICT_PORTS["USNYC"]["lon"], DICT_PORTS["USNYC"]["lat"]])
    dst = jnp.array([DICT_PORTS["DEHAM"]["lon"], DICT_PORTS["DEHAM"]["lat"]])

    # ── CMA-ES optimisation ──────────────────────────────────────────
    print("[CMA-ES] Optimising route USNYC → DEHAM ...")
    curve_cmaes, dict_cmaes = optimize(
        vectorfield=vectorfield,
        src=src,
        dst=dst,
        land=land,
        wavefield=wavefield,
        travel_stw=args.vel_ship,
        travel_time=None,
        penalty=1e10,
        K=args.K,
        L=args.L,
        num_pieces=1,
        popsize=args.popsize,
        sigma0=1.0,
        tolfun=60.0,
        damping=1.0,
        maxfevals=args.maxfevals,
        weight_l1=1.0,
        weight_l2=0.0,
        spherical_correction=True,
        keep_top=0.002,
        seed=args.seed,
        verbose=True,
    )
    print(f"[CMA-ES] Done — cost = {dict_cmaes['cost'] / 3600:.1f} hours")

    # ── FMS refinement ───────────────────────────────────────────────
    print("[FMS] Refining route ...")
    curve_fms, dict_fms = optimize_fms(
        vectorfield=vectorfield,
        curve=curve_cmaes,
        land=land,
        travel_stw=args.vel_ship,
        travel_time=None,
        patience=100,
        damping=0.9,
        maxfevals=int(1e6),
        weight_l1=1.0,
        weight_l2=0.0,
        spherical_correction=True,
        seed=args.seed,
        verbose=True,
    )
    print(f"[FMS] Done — cost = {sum(dict_fms['cost']) / 3600:.1f} hours")

    # ── Metrics ──────────────────────────────────────────────────────
    dist_cmaes = float(jnp.sum(haversine_distance_from_curve(curve_cmaes))) / 1000
    dist_fms = float(jnp.sum(haversine_distance_from_curve(curve_fms[0]))) / 1000

    print(f"\n{'Route':<12} {'Cost (h)':>10} {'Distance (km)':>14}")
    print("-" * 38)
    print(f"{'CMA-ES':<12} {dict_cmaes['cost'] / 3600:>10.1f} {dist_cmaes:>14.0f}")
    print(f"{'FMS':<12} {sum(dict_fms['cost']) / 3600:>10.1f} {dist_fms:>14.0f}")

    # ── Plot ─────────────────────────────────────────────────────────
    fig, ax = plot_curve(
        vectorfield=vectorfield,
        ls_curve=[curve_cmaes, curve_fms[0]],
        ls_name=[
            f"CMA-ES ({dict_cmaes['cost'] / 3600:.0f} h, {dist_cmaes:.0f} km)",
            f"FMS ({sum(dict_fms['cost']) / 3600:.0f} h, {dist_fms:.0f} km)",
        ],
        land=land,
        gridstep=1 / 12,
        figsize=(10, 6),
        xlim=(lon_range[0], lon_range[1]),
        ylim=(lat_range[0], lat_range[1]),
        color_currents=True,
    )
    ax.set_title(f"USNYC → DEHAM | {args.departure} | {2 * args.vel_ship:.0f} kn")
    fig.tight_layout()

    outfile = output_dir / "era5_benchmark_usnyc_deham.jpg"
    fig.savefig(outfile, dpi=300)
    plt.close()
    print(f"\n[SAVED] {outfile}")


if __name__ == "__main__":
    main()
