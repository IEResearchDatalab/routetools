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

from routetools._cost.haversine import great_circle_route
from routetools._ports import DICT_PORTS
from routetools.era5 import load_natural_earth_land_mask
from routetools.land import Land, reroute_around_land


def _land_touch_mask(route: np.ndarray, land: Land) -> np.ndarray:
    """Return boolean mask of waypoints that lie on land."""
    return np.asarray(land._check_nointerp(jnp.asarray(route)), dtype=bool).reshape(-1)


def main(
    output: Path = Path("output/demo_land_avoidance.png"),
    n_points: int = 220,
    astar_resolution_scale: int = 2,
    land_resolution: float = 0.08,
    ne_resolution: str = "50m",
    margin_deg: float = 6.0,
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

    touch_gc = _land_touch_mask(route_gc, land)
    touch_land = _land_touch_mask(route_land, land)

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
        label=f"route_gc ({int(touch_gc.sum())} land-touch points)",
        zorder=2,
    )
    (line_land,) = ax.plot(
        route_land[:, 0],
        route_land[:, 1],
        color="#d62728",
        lw=2.0,
        label=f"route_land ({int(touch_land.sum())} land-touch points)",
        zorder=3,
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

    ax.scatter(src[0], src[1], color="black", s=40, zorder=6)
    ax.scatter(dst[0], dst[1], color="black", s=40, zorder=6)
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
    print(f"route_gc land-touch points: {int(touch_gc.sum())}")
    print(f"route_land land-touch points: {int(touch_land.sum())}")


if __name__ == "__main__":
    typer.run(main)
