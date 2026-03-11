from collections.abc import Iterable

import h3.api.basic_int as h3
import numpy as np
from h3 import Polygon as H3Polygon
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)
from shapely.ops import substring, unary_union


def invert_polygon(
    polygon: Polygon | MultiPolygon,
    bounding_box: Iterable[float],
) -> Polygon | MultiPolygon:
    """
    Invert a Polygon or MultiPolygon with a bounding box.

    Parameters
    ----------
    polygon : Union[Polygon, MultiPolygon]
        The input geometry.
    bounding_box : Iterable[float]
        The bounding box as a list of four floats: [min_lat, min_lon, max_lat, max_lon].

    Returns
    -------
    Union[Polygon, MultiPolygon]
        The inverted geometry.
    """
    bounding_polygon = Polygon(
        [
            (bounding_box[1], bounding_box[0]),
            (bounding_box[3], bounding_box[0]),
            (bounding_box[3], bounding_box[2]),
            (bounding_box[1], bounding_box[2]),
        ]
    )
    try:
        polygon = bounding_polygon.difference(polygon)
    except Exception as e:
        raise Exception(
            f"The ocean geometry is not valid. Please check the geojson file.{polygon}"
        ) from e

    return polygon


def crop_polygon(
    multipolygon: MultiPolygon, bounding_box: Iterable[float]
) -> MultiPolygon:
    """
    Crop a MultiPolygon with a bounding box.

    Parameters
    ----------
    multipolygon : MultiPolygon
        The input geometries as a MultiPolygon.
    bounding_box : Iterable[float]
        The bounding box as a list of four floats: [min_lat, min_lon, max_lat, max_lon].

    Returns
    -------
    MultiPolygon
        The cropped MultiPolygon.
    """
    # Crear el polígono de la bounding box
    bounding_polygon = Polygon(
        [
            (bounding_box[1], bounding_box[0]),
            (bounding_box[3], bounding_box[0]),
            (bounding_box[3], bounding_box[2]),
            (bounding_box[1], bounding_box[2]),
        ]
    )

    return multipolygon.intersection(bounding_polygon)


def relative_to_latlon(
    y: float, x: float, height: int, width: int, bounding_box: Iterable[float]
) -> Iterable[float]:
    """
    Transform the relative coordinates of a pixel to latitude and longitude.

    Parameters
    ----------
    y : float
        The relative y coordinate.
    x : float
        The relative x coordinate.
    height : int
        The height of the image.
    width : int
        The width of the image.
    bounding_box : Iterable[float]
        The bounding box as a list of four floats: [min_lat, min_lon, max_lat, max_lon].

    Returns
    -------
    Iterable[float]
        The latitude and longitude of the pixel.
    """
    lat_min, lon_min, lat_max, lon_max = bounding_box
    lat = lat_min + (lat_max - lat_min) * (y / height)
    lon = lon_min + (lon_max - lon_min) * (x / width)

    return lon, lat


def get_h3_cells(polygons: list[dict], res: int = 5) -> set[int]:
    """
    Get the h3 cells from a list of polygons.

    Parameters
    ----------
    polygons : List[dict]
        A list of polygons with the coordinates of the geometry.
    res : int, optional
        The resolution of the h3 cells, by default 5.

    Returns
    -------
    set[int]
        A set of h3 cells.
    """
    cells = set()

    for polygon in polygons:
        exterior = [(lat, lon) for lon, lat in polygon.exterior.coords]
        interior = [
            [(lat, lon) for lon, lat in interior.coords[:]]
            for interior in polygon.interiors
        ]
        polygon_h3 = H3Polygon(exterior, *interior)
        cells_polygon = h3.polygon_to_cells(polygon_h3, res)

        # TODO: compact_cells is not working due to a bug from the library.
        # Once it is fixed, we can use it to reduce the number of cells
        # cells_compacted = h3.compact_cells(cells_polygon)

        cells.update(cells_polygon)

    return cells


def remove_border_cells(cells: set[int], num_cells: int = 1) -> set[int]:
    """
    Remove the cells that are next to land.

    Parameters
    ----------
    cells : set[int]
        A set of h3 cells.
    num_cells : int, optional
        Distance to border in cells, by default 1

    Returns
    -------
    set[int]
        A set of h3 cells.
    """
    updated_cells = set()

    for cell in cells:
        neighbours = h3.grid_disk(cell, num_cells)

        for n in neighbours:
            if n not in cells:
                # When a neighbor is not in the cell set, means that neighbor is land
                # We then consider this node as land and not
                # included it in the new cell set
                break
        else:
            updated_cells.add(cell)

    return updated_cells


def multipolygon_to_h3_cells(
    multipolygon: Polygon | MultiPolygon,
    res: int = 5,
    land_dilation: int = 1,
) -> set[int]:
    """
    Get the h3 cells from a geojson file.

    Parameters
    ----------
    multipolygon : Union[Polygon, MultiPolygon]
        A shapely polygon or multipolygon.
    res : int, optional
        The resolution of the h3 cells, by default 5.
    land_dilation : int, optional
        Distance to land from cells border, by default 1

    Returns
    -------
    set[int]
        A set of h3 cells.
    """
    polygons = []

    if isinstance(multipolygon, Polygon):
        polygons.append(multipolygon)
    else:
        # If it is a multipolygon, we need to get only the polygons from it
        for geom in multipolygon.geoms:
            if isinstance(geom, Polygon):
                polygons.append(geom)

    cells = get_h3_cells(polygons, res=res)

    if land_dilation > 0:
        cells = remove_border_cells(cells, land_dilation)

    return cells


def _extract_points(geometry) -> list[Point]:
    """Extract points from a shapely intersection result."""
    if geometry.is_empty:
        return []

    if isinstance(geometry, Point):
        return [geometry]

    if isinstance(geometry, MultiPoint):
        return list(geometry.geoms)

    if isinstance(geometry, LineString):
        coords = list(geometry.coords)
        return [Point(coords[0]), Point(coords[-1])]

    if isinstance(geometry, MultiLineString):
        points = []
        for line in geometry.geoms:
            coords = list(line.coords)
            points.append(Point(coords[0]))
            points.append(Point(coords[-1]))
        return points

    if isinstance(geometry, GeometryCollection):
        points = []
        for subgeom in geometry.geoms:
            points.extend(_extract_points(subgeom))
        return points

    return []


def _dedupe_coords(coords: list[tuple[float, float]], tol: float = 1e-10):
    """Remove consecutive duplicate/near-duplicate coordinates."""
    if not coords:
        return coords

    out = [coords[0]]
    for coord in coords[1:]:
        if np.hypot(coord[0] - out[-1][0], coord[1] - out[-1][1]) > tol:
            out.append(coord)
    return out


def _arc_between(
    ring_line: LineString,
    start_dist: float,
    end_dist: float,
) -> LineString:
    """Return the ring arc from start to end following ring orientation."""
    ring_length = ring_line.length
    if start_dist <= end_dist:
        return substring(ring_line, start_dist, end_dist)

    first = substring(ring_line, start_dist, ring_length)
    second = substring(ring_line, 0.0, end_dist)
    merged = list(first.coords) + list(second.coords)[1:]
    return LineString(_dedupe_coords(merged))


def _line_crosses_land(line: LineString, land: Polygon | MultiPolygon) -> bool:
    """Return True when line intersects land interior (not only boundary touch)."""
    return line.intersects(land) and not line.touches(land)


def _bypass_via_land_boundary(
    a: np.ndarray,
    b: np.ndarray,
    land: Polygon | MultiPolygon,
    n_points: int,
    clearance: float,
) -> np.ndarray:
    """Generate *n_points* waypoints from *a* to *b* that hug the land boundary.

    Parameters
    ----------
    a, b : np.ndarray
        Anchor waypoints in (lon, lat) order, shape (2,).
    land : Polygon | MultiPolygon
        Obstacle geometry (already dilated).  Coordinates in (lon, lat) order.
    n_points : int
        Number of interior bypass waypoints to produce (>= 1).

    Returns
    -------
    np.ndarray
        Shape (n_points, 2) array of (lon, lat) bypass waypoints.
        Falls back to a straight-line interpolation if no suitable arc is found.
    """
    if n_points <= 0:
        return np.empty((0, 2), dtype=float)

    path_ab = LineString([tuple(a), tuple(b)])

    # Select obstacle geom(s) that are hit by the direct anchor segment.
    if isinstance(land, Polygon):
        obstacle_geoms = [land] if path_ab.intersects(land) else []
    else:
        obstacle_geoms = [g for g in land.geoms if path_ab.intersects(g)]

    if not obstacle_geoms:
        # No blocking geometry found; straight interpolation is fine.
        return np.linspace(a, b, n_points + 2)[1:-1]

    # Merge and pick the dominant blocking polygon.
    obstacle = unary_union(obstacle_geoms)
    if isinstance(obstacle, MultiPolygon):
        obstacle = obstacle.convex_hull

    # Expand the guidance polygon progressively until the discretized replacement
    # is crossing-free (important when only a few waypoints can be inserted).
    guide_clearance = max(clearance, 1e-2)
    for _ in range(12):
        guide = obstacle.buffer(guide_clearance, join_style=2)
        if isinstance(guide, MultiPolygon):
            guide = max(guide.geoms, key=lambda g: g.area)

        ring_line = LineString(guide.exterior.coords)

        boundary_hits = _extract_points(path_ab.intersection(guide.boundary))
        if len(boundary_hits) >= 2:
            boundary_hits = sorted(boundary_hits, key=path_ab.project)
            unique_hits: list[Point] = []
            for point in boundary_hits:
                if not unique_hits or point.distance(unique_hits[-1]) > 1e-9:
                    unique_hits.append(point)
            p_entry = unique_hits[0]
            p_exit = unique_hits[-1]
        else:
            p_entry = ring_line.interpolate(ring_line.project(Point(tuple(a))))
            p_exit = ring_line.interpolate(ring_line.project(Point(tuple(b))))

        d_entry = ring_line.project(p_entry)
        d_exit = ring_line.project(p_exit)

        arc_fwd = _arc_between(ring_line, d_entry, d_exit)
        arc_bwd_raw = _arc_between(ring_line, d_exit, d_entry)
        arc_bwd = LineString(list(arc_bwd_raw.coords)[::-1])

        best_sampled: np.ndarray | None = None
        best_length = np.inf
        for arc in (arc_fwd, arc_bwd):
            if arc.length <= 0:
                continue

            if n_points == 1:
                # Brute-force search for a single apex waypoint that yields a
                # crossing-free two-segment detour.
                sampled = None
                sampled_length = np.inf
                for d in np.linspace(0.0, arc.length, 181):
                    candidate = np.array(
                        [[arc.interpolate(d).x, arc.interpolate(d).y]],
                        dtype=float,
                    )
                    discrete_line = LineString(
                        [tuple(a), tuple(candidate[0]), tuple(b)]
                    )
                    if _line_crosses_land(discrete_line, land):
                        continue

                    length = discrete_line.length
                    if length < sampled_length:
                        sampled = candidate
                        sampled_length = length

                if sampled is None:
                    continue
            else:
                arc_dist = np.linspace(0.0, arc.length, n_points + 2)[1:-1]
                sampled = np.array(
                    [[arc.interpolate(d).x, arc.interpolate(d).y] for d in arc_dist],
                    dtype=float,
                )

            discrete_line = LineString([tuple(a), *map(tuple, sampled), tuple(b)])
            if _line_crosses_land(discrete_line, land):
                continue

            length = discrete_line.length
            if length < best_length:
                best_sampled = sampled
                best_length = length

        if best_sampled is not None:
            return best_sampled

        guide_clearance *= 2.0

    return np.linspace(a, b, n_points + 2)[1:-1]


def reroute_around_land(
    route: np.ndarray,
    land: Polygon | MultiPolygon,
    dilation: float = 0.1,
) -> np.ndarray:
    """Reroute a path to avoid land by tracing along the land boundary.

    The algorithm works in five steps:

    1. **Dilate** the land geometry by *dilation* degrees to create a safety margin.
    2. **Detect** every route segment that crosses the dilated land.
    3. **Group** the crossing-point indices into contiguous runs.  For each run the
       two anchor points (last water point before, first water point after) are kept.
    4. **Bypass** each run: replace the removed interior points with the same number
       of waypoints sampled along the exterior boundary of the blocking land polygon.
    5. Return the modified route with **the same number of points** as the input.

    Parameters
    ----------
    route : np.ndarray
        Waypoints as an array of shape (N, 2) with columns ``[lon, lat]``.
    land : Polygon | MultiPolygon
        Shapely geometry representing land obstacles, in (lon, lat) coordinate
        order.  Typically ``ocean.shapely_ocean`` from
        :class:`routetools.wrr_bench.ocean.Ocean`.
    dilation : float, optional
        Buffer distance applied to *land* before crossing detection, in degrees.
        Defaults to 0.1.

    Returns
    -------
    np.ndarray
        Rerouted waypoints, shape (N, 2).  The total number of points is
        identical to the input.  If the crossing run touches a route endpoint
        no bypass is inserted for that run (the endpoint cannot be moved).

    Notes
    -----
    The route is expected to have its first and last points in water.  Interior
    points on land are replaced by points on the dilated-land boundary.  When
    multiple obstacles block the direct path, the dominant one (longest
    intersection) is used to trace the boundary arc.
    """
    route = np.asarray(route, dtype=float)
    if route.ndim != 2 or route.shape[1] != 2:
        raise ValueError(f"route must have shape (N, 2), got {route.shape}")

    n_points = len(route)
    if n_points < 2:
        return route.copy()

    if land.is_empty:
        return route.copy()

    # Step 1 – dilate land used for safety analysis.
    dilated = land.buffer(dilation, join_style=2) if dilation > 0 else land
    clearance = max(1e-2, dilation * 0.25)

    result = route.copy()

    # Iterate a few rounds in case one local fix reveals a new crossing nearby.
    for _ in range(4):
        # Step 2 – locate all crossing segments.
        crossing_seg = np.zeros(n_points - 1, dtype=bool)
        for i in range(n_points - 1):
            seg = LineString([tuple(result[i]), tuple(result[i + 1])])
            crossing_seg[i] = _line_crosses_land(seg, dilated)

        crossing_count = int(crossing_seg.sum())
        if crossing_count == 0:
            break

        # Step 3 – group crossing segments into contiguous runs [i_start, i_end].
        runs: list[tuple[int, int]] = []
        i = 0
        while i < len(crossing_seg):
            if crossing_seg[i]:
                j = i
                while j + 1 < len(crossing_seg) and crossing_seg[j + 1]:
                    j += 1
                runs.append((i, j))
                i = j + 1
            else:
                i += 1

        # Step 4 – for each run, replace only interior points and keep anchors.
        for seg_start, seg_end in runs:
            anchor_a_idx = seg_start
            anchor_b_idx = seg_end + 1

            # Number of points removed/replaced between anchors.
            n_removed = anchor_b_idx - anchor_a_idx - 1
            if n_removed <= 0:
                continue

            a = result[anchor_a_idx]
            b = result[anchor_b_idx]
            bypass = _bypass_via_land_boundary(a, b, dilated, n_removed, clearance)
            result[anchor_a_idx + 1 : anchor_b_idx] = bypass

    return result
