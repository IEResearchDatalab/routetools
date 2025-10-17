from __future__ import annotations

import heapq
import json
from collections.abc import Iterable
from pathlib import Path

import jax.numpy as jnp
import typer
from h3.api import basic_int as h3
from shapely.geometry import MultiPolygon, Polygon, shape

from routetools.cost import cost_function
from routetools.vectorfield import vectorfield_zero


def load_multipolygon(path: str | Path) -> Polygon | MultiPolygon:
    """Load a GeoJSON file and return a shapely Polygon or MultiPolygon."""
    with open(path, encoding="utf-8") as fh:
        gj = json.load(fh)
    features = gj.get("features", [gj])
    geom = features[0].get("geometry", gj.get("geometry", None))
    if geom is None:
        raise ValueError("No geometry found in GeoJSON")

    return shape(geom)


def get_h3_cells_from_multipolygon(
    multipolygon: Polygon | MultiPolygon, res: int = 5, land_dilation: int = 1
) -> set[int]:
    """Convert a shapely multipolygon into a set of h3 cell indices.

    Cells that are adjacent to land (border cells) are removed using
    `land_dilation`.
    """
    polygons = []
    if isinstance(multipolygon, Polygon):
        polygons.append(multipolygon)
    else:
        polygons.extend(
            [geom for geom in multipolygon.geoms if isinstance(geom, Polygon)]
        )

    cells: set[int] = set()
    for polygon in polygons:
        # Use GeoJSON-style coordinates (lon, lat) for compatibility with
        # different h3 versions. polygon.exterior.coords yields (lon, lat).
        exterior_coords = [list(coord) for coord in polygon.exterior.coords]
        interior_coords = [
            [list(coord) for coord in interior.coords[:]]
            for interior in polygon.interiors
        ]
        geo = {"type": "Polygon", "coordinates": [exterior_coords] + interior_coords}
        cells_polygon = h3.polygon_to_cells(geo, res)
        cells.update(cells_polygon)

    if land_dilation > 0:
        cells = _remove_border_cells(cells, land_dilation)
    return cells


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


def _neighbors(cell: int, cells: set[int], neighbour_disk: int = 1) -> Iterable[int]:
    """Return neighbor cells for `cell` that are present in `cells`.

    Uses `grid_ring` for immediate neighbors when neighbour_disk==1, otherwise
    uses `grid_disk` and excludes the center cell.
    """
    if neighbour_disk == 1:
        ring = h3.grid_ring(cell, 1) or []
        for n in ring:
            if n in cells:
                yield n
    else:
        disk = h3.grid_disk(cell, neighbour_disk)
        for n in disk:
            if n != cell and n in cells:
                yield n


def astar_search(
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    cells: set[int],
    res: int = 5,
    neighbour_disk: int = 1,
    heuristic_weight: float = 1.0,
) -> list[tuple[float, float]]:
    """Run A* on the h3 cell graph and return a list of (lat, lon) points.

    If start or end are not inside any cell in `cells`, we snap them to the
    nearest available cell centroid.
    """

    # helper to snap latlon to cell in set
    def _snap(lat: float, lon: float) -> int:
        c = h3.latlng_to_cell(lat, lon, res)
        if c in cells:
            return c
        # find nearest by centroid distance
        best = None
        best_d = float("inf")
        for cell in cells:
            clat, clon = _cell_centroid(cell)
            # Use cost_function with zero vectorfield and STW=1.0 as a proxy
            # distance/cost between the point and the cell centroid.
            curve = jnp.array([[[lat, lon], [clat, clon]]])
            d = float(cost_function(vectorfield_zero, curve, travel_stw=1.0)[0])
            if d < best_d:
                best_d = d
                best = cell
        return best

    start_cell = _snap(start_lat, start_lon)
    end_cell = _snap(end_lat, end_lon)

    # A* structures
    open_heap: list[tuple[float, int]] = []  # (f, cell)
    heapq.heappush(open_heap, (0.0, start_cell))
    came_from: dict[int, int | None] = {start_cell: None}
    gscore: dict[int, float] = {start_cell: 0.0}

    def heuristic(c: int) -> float:
        clat, clon = _cell_centroid(c)
        # Heuristic estimated using the cost function with zero currents and
        # constant speed-through-water of 1. This mirrors the CMA-ES style.
        curve = jnp.array([[[clat, clon], [end_lat, end_lon]]])
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
            return [_cell_centroid(c) for c in path_cells]

        visited.add(current)

        for nb in _neighbors(current, cells, neighbour_disk=neighbour_disk):
            # compute cost/time for the single-segment curve between centroids
            clat, clon = _cell_centroid(current)
            nlat, nlon = _cell_centroid(nb)
            seg_curve = jnp.array([[[clat, clon], [nlat, nlon]]])
            seg_cost = float(
                cost_function(vectorfield_zero, seg_curve, travel_stw=1.0)[0]
            )
            tentative_g = gscore[current] + seg_cost
            if (nb not in gscore) or tentative_g < gscore[nb]:
                came_from[nb] = current
                gscore[nb] = tentative_g
                f = tentative_g + heuristic(nb)
                heapq.heappush(open_heap, (f, nb))

    # no path found
    return []


def main(
    src_lat: float = 0,
    src_lon: float = 0,
    dst_lat: float = 1,
    dst_lon: float = 1,
    land_geojson: Path | None = None,
    res: int = 5,
    neighbour_disk: int = 1,
    land_dilation: int = 1,
    out: Path = Path("output/astar_route.json"),
) -> None:
    """Compute a circumnavigation route using A* on an H3 grid.

    The script builds an H3 grid from the provided geojson (the geojson should
    contain the ocean polygon(s) — populated cells will be considered navigable).
    """
    if land_geojson is None:
        raise typer.BadParameter(
            "land_geojson is required (provide ocean polygon geojson)"
        )

    multipoly = load_multipolygon(land_geojson)
    cells = get_h3_cells_from_multipolygon(
        multipoly,
        res=res,
        land_dilation=land_dilation,
    )

    route = astar_search(
        start_lat=src_lat,
        start_lon=src_lon,
        end_lat=dst_lat,
        end_lon=dst_lon,
        cells=cells,
        res=res,
        neighbour_disk=neighbour_disk,
    )

    if not route:
        typer.echo("No route found")
        raise typer.Exit(code=1)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump({"route": [[float(lat), float(lon)] for lat, lon in route]}, fh)

    typer.echo(f"Route with {len(route)} waypoints written to {out}")


if __name__ == "__main__":
    typer.run(main)
