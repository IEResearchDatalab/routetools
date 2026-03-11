#!/usr/bin/env python
"""Demo script for reroute_around_land on the DEHAM -> USNYC corridor.

The script:
1. Builds a great-circle route (route_gc).
2. Reroutes it with ``reroute_around_land`` (route_land).
3. Plots both routes and highlights waypoints that touch land.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import typer

from routetools._cost.haversine import (
    great_circle_route,
    haversine_meters_components,
)
from routetools._ports import DICT_PORTS
from routetools.era5 import load_natural_earth_land_mask
from routetools.fms import optimize_fms
from routetools.land import Land, reroute_around_land
from routetools.vectorfield import vectorfield_zero


def _land_touch_mask(route: np.ndarray, land: Land) -> np.ndarray:
    """Return boolean mask of waypoints that lie on land."""
    return np.asarray(land._check_nointerp(jnp.asarray(route)), dtype=bool).reshape(-1)


def _route_distance_km(route: np.ndarray) -> float:
    """Return total route length in kilometers using haversine segments."""
    route_jnp = jnp.asarray(route)
    if route_jnp.shape[0] < 2:
        return 0.0

    lats = route_jnp[:, 1]
    lons = route_jnp[:, 0]
    dx, dy = haversine_meters_components(
        lats[1:],
        lons[1:],
        lats[:-1],
        lons[:-1],
    )
    dists_m = jnp.sqrt(dx**2 + dy**2)
    return float(jnp.sum(dists_m) / 1000.0)


def _run_fms_with_endpoint_land_handling(
    route: np.ndarray,
    land: Land,
    patience: int,
    damping: float,
    maxfevals: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Run FMS and optionally strip/restore boundary points that are on land."""
    route_in = np.asarray(route, dtype=float)
    touch = _land_touch_mask(route_in, land)
    land_idx = np.flatnonzero(touch)

    drop_start = False
    drop_end = False
    if land_idx.size > 0 and np.all(np.isin(land_idx, [0, len(route_in) - 1])):
        drop_start = bool(touch[0])
        drop_end = bool(touch[-1])
        if drop_start:
            route_in = route_in[1:]
        if drop_end:
            route_in = route_in[:-1]

    if route_in.shape[0] < 3:
        return np.asarray(route, dtype=float), {
            "status": "skipped",
            "reason": "too_few_points_after_endpoint_strip",
        }

    try:
        curve_fms, info = optimize_fms(
            vectorfield_zero,
            curve=jnp.asarray(route_in),
            land=land,
            travel_stw=1.0,
            patience=patience,
            damping=damping,
            maxfevals=maxfevals,
            spherical_correction=True,
            verbose=False,
        )
        route_fms = np.asarray(curve_fms[0], dtype=float)
    except ValueError as exc:
        return np.asarray(route, dtype=float), {
            "status": "skipped",
            "reason": str(exc),
        }

    if drop_start:
        route_fms = np.vstack([route[:1], route_fms])
    if drop_end:
        route_fms = np.vstack([route_fms, route[-1:]])

    return route_fms, {
        "status": "ok",
        "drop_start": drop_start,
        "drop_end": drop_end,
        "niter": info.get("niter", None),
    }


def main(
    output: Path = Path("output/demo_land_avoidance.png"),
    n_points: int = 220,
    astar_resolution_scale: int = 2,
    land_resolution: float = 0.01,
    ne_resolution: str = "50m",
    margin_deg: float = 6.0,
    fms_patience: int = 1000,
    fms_damping: float = 0.5,
    fms_maxfevals: int = 100000,
    show: bool = False,
) -> None:
    """Generate and plot DEHAM -> USNYC land-avoidance routes."""
    src = np.array([DICT_PORTS["DEHAM"]["lon"], DICT_PORTS["DEHAM"]["lat"]])
    dst = np.array([DICT_PORTS["USNYC"]["lon"], DICT_PORTS["USNYC"]["lat"]])

    route_gc = np.asarray(great_circle_route(src, dst, n_points=n_points), dtype=float)

    lon_min = float(np.min(route_gc[:, 0]) - margin_deg)
    lon_max = float(np.max(route_gc[:, 0]) + margin_deg)
    lat_min = float(max(-90.0, np.min(route_gc[:, 1]) - margin_deg))
    lat_max = float(min(90.0, np.max(route_gc[:, 1]) + margin_deg))

    land = load_natural_earth_land_mask(
        lon_range=(lon_min, lon_max),
        lat_range=(lat_min, lat_max),
        resolution=land_resolution,
        ne_resolution=ne_resolution,
        interpolate=0,
    )

    route_land = reroute_around_land(
        route_gc,
        land=land,
        astar_resolution_scale=astar_resolution_scale,
    )

    route_fms, fms_info = _run_fms_with_endpoint_land_handling(
        route_land,
        land,
        patience=fms_patience,
        damping=fms_damping,
        maxfevals=fms_maxfevals,
    )

    touch_gc = _land_touch_mask(route_gc, land)
    touch_land = _land_touch_mask(route_land, land)
    touch_fms = _land_touch_mask(route_fms, land)
    dist_gc_km = _route_distance_km(route_gc)
    dist_land_km = _route_distance_km(route_land)
    dist_fms_km = _route_distance_km(route_fms)

    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.contourf(
        np.asarray(land.x),
        np.asarray(land.y),
        np.asarray(land.array).T,
        levels=[0.0, 0.5, 1.0],
        colors=["#dcefff", "#8f8f8f"],
        alpha=0.9,
        zorder=0,
    )

    (line_gc,) = ax.plot(
        route_gc[:, 0],
        route_gc[:, 1],
        color="#1f77b4",
        lw=2.0,
        label=f"route_gc ({dist_gc_km:.1f} km)",
        zorder=2,
    )
    (line_land,) = ax.plot(
        route_land[:, 0],
        route_land[:, 1],
        color="#d62728",
        lw=2.0,
        label=f"route_land ({dist_land_km:.1f} km)",
        zorder=3,
    )
    (line_fms,) = ax.plot(
        route_fms[:, 0],
        route_fms[:, 1],
        color="#2ca02c",
        lw=2.0,
        label=f"route_fms ({dist_fms_km:.1f} km)",
        zorder=4,
    )

    if touch_gc.any():
        ax.scatter(
            route_gc[touch_gc, 0],
            route_gc[touch_gc, 1],
            s=50,
            marker="o",
            facecolors="none",
            edgecolors=line_gc.get_color(),
            linewidths=1.5,
            zorder=4,
        )

    if touch_land.any():
        ax.scatter(
            route_land[touch_land, 0],
            route_land[touch_land, 1],
            s=50,
            marker="o",
            facecolors="none",
            edgecolors=line_land.get_color(),
            linewidths=1.5,
            zorder=5,
        )

    if touch_fms.any():
        ax.scatter(
            route_fms[touch_fms, 0],
            route_fms[touch_fms, 1],
            s=50,
            marker="o",
            facecolors="none",
            edgecolors=line_fms.get_color(),
            linewidths=1.5,
            zorder=6,
        )

    ax.scatter(src[0], src[1], color="black", s=40, zorder=7)
    ax.scatter(dst[0], dst[1], color="black", s=40, zorder=7)
    ax.text(src[0], src[1], " DEHAM", va="bottom", ha="left")
    ax.text(dst[0], dst[1], " USNYC", va="bottom", ha="left")

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    ax.set_title("Land avoidance demo: DEHAM -> USNYC")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()

    fig.savefig(output, dpi=180)
    if show:
        plt.show()
    plt.close(fig)

    print(f"Saved plot to: {output}")
    print(f"route_gc distance: {dist_gc_km:.1f} km")
    print(f"route_land distance: {dist_land_km:.1f} km")
    print(f"route_fms distance: {dist_fms_km:.1f} km")
    print(f"route_gc land-touch points: {int(touch_gc.sum())}")
    print(f"route_land land-touch points: {int(touch_land.sum())}")
    print(f"route_fms land-touch points: {int(touch_fms.sum())}")
    print(f"fms status: {fms_info.get('status')}")
    if fms_info.get("status") == "ok":
        print(
            "fms endpoint handling: "
            f"drop_start={fms_info.get('drop_start')}, "
            f"drop_end={fms_info.get('drop_end')}"
        )
        print(f"fms iterations: {fms_info.get('niter')}")
    else:
        print(f"fms reason: {fms_info.get('reason')}")


if __name__ == "__main__":
    typer.run(main)
