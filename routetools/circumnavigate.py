from __future__ import annotations

import heapq
from collections.abc import Iterable
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as _plt
import typer
from h3.api import basic_int as h3

from routetools.cost import cost_function
from routetools.fms import optimize_fms
from routetools.land import Land
from routetools.plot import plot_curve
from routetools.vectorfield import vectorfield_zero


def get_h3_cells_from_land(
    land: Land, res: int = 5, land_dilation: int = 1
) -> set[int]:
    """Return navigable H3 cells inside the Land bounding limits.

    Algorithm:
    - Build the set of H3 cells that cover the rectangular bounds defined by
      `land.x` and `land.y` by mapping the grid centers to H3 cells.
    - Remove cells whose centroid falls on land according to the provided
      `Land` instance.

    This approach inverts the previous sampling strategy (sample water cells
    only) and ensures we start from the full coverage of the domain then
    subtract land cells.
    """
    # Derive coverage entirely through H3 operations.
    # Start from the domain center and expand a disk until the set of cell
    # centroids covers the bounding box of the Land instance.
    xmin, xmax = land.xmin, land.xmax
    ymin, ymax = land.ymin, land.ymax

    center_lat = 0.5 * (ymin + ymax)
    center_lon = 0.5 * (xmin + xmax)
    center_cell = h3.latlng_to_cell(center_lat, center_lon, res)

    all_cells: set[int] = set()
    radius = 0
    max_radius = 200  # safety limit to avoid infinite loops
    while radius <= max_radius:
        disk = set(h3.grid_disk(center_cell, radius))
        all_cells.update(disk)

        # compute centroid bounds
        lons = []
        lats = []
        for c in all_cells:
            clat, clon = _cell_centroid(c)
            lats.append(clat)
            lons.append(clon)
        if not lats or not lons:
            radius += 1
            continue
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)

        # Check whether centroid coverage spans the land bbox
        if (
            (min_lon <= xmin)
            and (max_lon >= xmax)
            and (min_lat <= ymin)
            and (max_lat >= ymax)
        ):
            break
        radius += 1

    # determine which of these cells are land (by testing the centroid)
    land_cells: set[int] = set()
    for c in all_cells:
        clat, clon = _cell_centroid(c)
        # Land expects coordinates as (x, y) -> (lon, lat)
        is_land = bool(land(jnp.array([clon, clat])))
        if is_land:
            land_cells.add(c)

    navigable = all_cells - land_cells

    if land_dilation > 0:
        navigable = _remove_border_cells(navigable, land_dilation)

    return navigable


def _remove_border_cells(cells: set[int], num_cells: int = 1) -> set[int]:
    updated = set()
    for cell in cells:
        neigh = h3.grid_disk(cell, num_cells)
        for n in neigh:
            if n not in cells:
                break
        else:
            updated.add(cell)
    return updated


def _cell_centroid(cell: int) -> tuple[float, float]:
    """Return (lat, lon) centroid of an h3 cell."""
    lat, lon = h3.cell_to_latlng(cell)
    return lat, lon


def _neighbors(
    cell: int,
    cells: set[int] | None,
    neighbour_disk: int = 1,
) -> Iterable[int]:
    """Return neighbor cells for `cell` that are present in `cells`.

    Uses `grid_ring` for immediate neighbors when neighbour_disk==1, otherwise
    uses `grid_disk` and excludes the center cell.
    """
    # When `cells` is None, allow any neighbour from the H3 grid. Otherwise
    # restrict to neighbours that are members of `cells`.
    if neighbour_disk == 1:
        # Use grid_disk and exclude the center cell; grid_ring can return an
        # empty list for some cells (e.g. pentagons) which would prevent the
        # search from expanding. grid_disk is more robust.
        disk = h3.grid_disk(cell, 1) or []
        for n in disk:
            if n != cell and (cells is None or n in cells):
                yield n
    else:
        disk = h3.grid_disk(cell, neighbour_disk)
        for n in disk:
            if n != cell and (cells is None or n in cells):
                yield n


def _snap(lat: float, lon: float, cells: set[int] | None = None, res: int = 5) -> int:
    # If no land/cells provided, snap directly to the cell containing the
    # coordinate.
    c = h3.latlng_to_cell(lat, lon, res)

    if cells is None:
        return c
    if c in cells:
        return c
    # find nearest by centroid distance among provided cells
    best = None
    best_d = float("inf")
    for cell in cells:
        clat, clon = _cell_centroid(cell)
        # Use cost_function with zero vectorfield and STW=1.0 as a proxy
        # distance/cost between the point and the cell centroid.
        # cost_function expects coordinates in [lon, lat] order.
        curve = jnp.array([[[lon, lat], [clon, clat]]])
        d = float(cost_function(vectorfield_zero, curve, travel_stw=1.0)[0])
        if d < best_d:
            best_d = d
            best = cell
    return best


def circumnavigate(
    src: tuple[float, float],
    dst: tuple[float, float],
    land: Land | None = None,
    land_dilation: int = 1,
    res: int = 5,
    neighbour_disk: int = 1,
    heuristic_weight: float = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Run A* on the h3 cell graph and return a list of (lat, lon) points.

    If start or end are not inside any cell in `cells`, we snap them to the
    nearest available cell centroid.
    It then refines the resulting route using FMS optimization.

    Parameters
    ----------
    start_lat : float
        Latitude of the start point.
    start_lon : float
        Longitude of the start point.
    end_lat : float
        Latitude of the end point.
    end_lon : float
        Longitude of the end point.
    land : Land | None, optional
        Land instance to derive navigable cells from. If None, assumes no land
        constraints, by default None.
    land_dilation : int, optional
        Number of dilation steps to apply to land cells when deriving the
        navigable H3 cells, by default 1.
    res : int, optional
        H3 resolution to use for the grid, by default 5.
    neighbour_disk : int, optional
        Neighbourhood disk size to consider when expanding nodes in A*,
        by default 1.
    heuristic_weight : float, optional
        Weighting factor for the heuristic function, by default 1.0.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        Tuple containing:
        - The refined route as an array of (lon, lat) points.
        - The initial A* route as an array of (lon, lat) points.
    """
    if land is None:
        cells = None
    else:
        cells = get_h3_cells_from_land(land, res=res, land_dilation=land_dilation)

    # Snap start and end to nearest available cells
    start_lat, start_lon = src
    end_lat, end_lon = dst
    start_cell = _snap(start_lat, start_lon, cells=cells, res=res)
    end_cell = _snap(end_lat, end_lon, cells=cells, res=res)

    # Initialize an empty route
    curve = jnp.array([])

    # A* structures
    open_heap: list[tuple[float, int]] = []  # (f, cell)
    # initialize with heuristic of start
    heapq.heappush(open_heap, (0.0 + 0.0, start_cell))
    came_from: dict[int, int | None] = {start_cell: None}
    gscore: dict[int, float] = {start_cell: 0.0}

    def heuristic(c: int) -> float:
        clat, clon = _cell_centroid(c)
        # Heuristic estimated using the cost function with zero currents and
        # constant speed-through-water of 1. This mirrors the CMA-ES style.
        # cost_function expects [lon, lat]
        curve = jnp.array([[[clon, clat], [end_lon, end_lat]]])
        # cost_function returns a scalar per curve; take the first element
        est = float(cost_function(vectorfield_zero, curve, travel_stw=1.0)[0])
        return heuristic_weight * est

    visited = set()

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current in visited:
            continue
        if current == end_cell:
            # reconstruct path
            path_cells = []
            node = current
            while node is not None:
                path_cells.append(node)
                node = came_from.get(node)
            path_cells.reverse()
            # convert to lat/lon centroids
            # Return coordinates in [lon, lat] order (cost_function expects this)
            coords: list[tuple[float, float]] = []
            for c in path_cells:
                clat, clon = _cell_centroid(c)
                coords.append((clon, clat))
            curve = jnp.array(coords)
            break

        visited.add(current)

        for nb in _neighbors(current, cells, neighbour_disk=neighbour_disk):
            # compute cost/time for the single-segment curve between centroids
            clat, clon = _cell_centroid(current)
            nlat, nlon = _cell_centroid(nb)
            # cost_function expects [lon, lat]
            seg_curve = jnp.array([[[clon, clat], [nlon, nlat]]])
            seg_cost = float(
                cost_function(vectorfield_zero, seg_curve, travel_stw=1.0)[0]
            )
            tentative_g = gscore[current] + seg_cost
            if (nb not in gscore) or tentative_g < gscore[nb]:
                came_from[nb] = current
                gscore[nb] = tentative_g
                f = tentative_g + heuristic(nb)
                heapq.heappush(open_heap, (f, nb))

    if len(curve) == 0:
        return curve, curve  # no path found

    # Refine the route using FMS optimization
    curve_refined, _ = optimize_fms(
        vectorfield_zero,
        curve=curve,
        land=land,
        travel_stw=1.0,
        tolfun=1e-8,
        maxfevals=50000,
        damping=0.9,
    )

    # no path found
    return (curve_refined[0], curve)


def main(
    src_lat: float = 0.0,
    src_lon: float = 0.0,
    dst_lat: float = 10.0,
    dst_lon: float = 10.0,
    res: int = 5,
    neighbour_disk: int = 1,
    land_dilation: int = 1,
    out: str = "output",
) -> None:
    """Compute a circumnavigation route using A* on an H3 grid.

    The script builds an H3 grid from a provided `Land` instance (if any).
    Populated H3 cells are considered navigable; if `land` is omitted the
    search assumes no land constraints.
    """
    land = Land(
        (-1, 11),
        (-1, 11),
        water_level=0.7,
        resolution=6,
        random_seed=1,
        outbounds_is_land=True,
    )

    curve_refined, curve = circumnavigate(
        src=(src_lat, src_lon),
        dst=(dst_lat, dst_lon),
        land=land,
        land_dilation=land_dilation,
        res=res,
        neighbour_disk=neighbour_disk,
    )

    if len(curve) == 0:
        typer.echo("No route found")

    ls_curve = [curve, curve_refined]
    ls_name = ["A* route", "FMS refined route"]

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    typer.echo(f"Computed route with {len(curve)} waypoints")
    try:
        fig, ax = plot_curve(
            vectorfield_zero,
            ls_curve=ls_curve,
            ls_name=ls_name,
            land=land,
            gridstep=0.5,
            figsize=(6, 6),
        )
        jpg_path = out / "circumnavigation.jpg"
        fig.savefig(jpg_path, dpi=300, bbox_inches="tight")
        typer.echo(f"Route plot written to {jpg_path}")
        # close the figure

        _plt.close(fig)
    except Exception as e:  # pragma: no cover - plotting best-effort
        typer.echo(f"Failed to save route plot: {e}")


if __name__ == "__main__":
    typer.run(main)
