from __future__ import annotations

import heapq

import h3.api.basic_int as h3
import numpy as np
from h3 import Polygon as H3Polygon
from shapely.geometry import MultiPolygon, Polygon

from routetools.wrr_bench import Ocean
from routetools.wrr_bench.polygons import invert_polygon


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


class Node:
    """A node class for A* Pathfinding."""

    def __init__(
        self, hex_id: int, parent: Node | None = None, vel_ship: float | None = None
    ):
        self.parent = parent
        self.hex_id = hex_id
        self.vel_ship = vel_ship

        self.g = 0
        self.h = 0
        self.f = 0
        self.dt = 0

    def __eq__(self, other: Node):
        """Compare nodes for equality."""
        # Skip vel comparison if one of the nodes has "None"
        if self.vel_ship is None or other.vel_ship is None:
            eq_vel = True
        else:
            eq_vel = self.vel_ship == other.vel_ship
        # Compare hex_id and velocity
        return (self.hex_id == other.hex_id) and eq_vel

    def __lt__(self, other: Node):
        """Compare nodes by f-value for priority queue ordering."""
        return self.f < other.f

    def __repr__(self):
        """Return a short string representation of the node."""
        return str(self.hex_id) + "-" + str(self.vel_ship)


def _get_route(last_node: Node) -> list[tuple]:
    route: list[tuple] = []
    node_now = last_node
    while node_now is not None:
        route.append(h3.cell_to_latlng(node_now.hex_id))
        node_now = node_now.parent
    return route[::-1]


def _get_neighbours(
    node: Node, ocean_cells: set[int], neighbour_disk_size: int = 3
) -> list[Node]:
    neighbours: list[Node] = []
    for neighbour in h3.grid_disk(node.hex_id, neighbour_disk_size):
        if neighbour != node.hex_id and neighbour in ocean_cells:
            neighbours.append(Node(neighbour))
    return neighbours


def circumnavigate(
    lat_start: float,
    lon_start: float,
    lat_end: float,
    lon_end: float,
    data: Ocean,
    grid_resolution: int = 4,
    neighbour_disk_size: int = 3,
    land_dilation: int = 0,
    check_land_edges: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Optimize the route using the A* algorithm.

    The ocean data is replaced by one with all zero currents.

    Parameters
    ----------
    lat_start : float
        The latitude of the starting point.
    lon_start : float
        The longitude of the starting point.
    lat_end : float
        The latitude of the ending point.
    lon_end : float
        The longitude of the ending point.
    data : Ocean
        The ocean data to use for the optimization.
    grid_resolution : int, optional
        The resolution of the h3 grid used for the A* algorithm, by default 4
    neighbour_disk_size : int, optional
        The size of the disk used to get the neighbors of a node, by default 3
    land_dilation : int, optional
        The number of cells to dilate the land cells, by default 0
    check_land_edges : bool, optional
        Whether to check if the edge between two nodes is crossing land,
        by default True

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The optimized route with a fine reparametrization,
        as arrays of latitudes and longitudes.
    """
    # Replace the ocean data with one with all zero currents to compute the route
    ocean_zero = Ocean(
        bounding_box=data.bounding_box,
        land_file=data.land_file,
        interp_method=data.interp_method,
        radius=data.radius,
    )

    hex_start = h3.latlng_to_cell(lat_start, lon_start, grid_resolution)
    hex_end = h3.latlng_to_cell(lat_end, lon_end, grid_resolution)

    node_start = Node(hex_start)
    node_end = Node(hex_end)

    polygon = invert_polygon(ocean_zero.shapely_ocean, ocean_zero.bounding_box)
    ocean_cells = multipolygon_to_h3_cells(
        polygon,
        res=grid_resolution,
        land_dilation=land_dilation,
    )

    ocean_cells.add(node_start.hex_id)
    ocean_cells.add(node_end.hex_id)

    # Create priority queue and add start node
    open_list: list[Node] = []
    heapq.heapify(open_list)  # PriorityQueue
    heapq.heappush(open_list, node_start)

    closed_list: list[Node] = []

    nodes_route = None
    while len(open_list) > 0:
        node_now = heapq.heappop(open_list)

        if node_now == node_end:
            nodes_route = _get_route(node_now)
            break

        for neighbour in _get_neighbours(node_now, ocean_cells, neighbour_disk_size):
            # Check if the node is in the closed list
            if neighbour in closed_list:
                continue

            # Calculate the cost of the neighbour
            neighbour.parent = node_now
            if node_now == node_start:
                node_now_lat = lat_start
                node_now_lon = lon_start
            else:
                node_now_lat, node_now_lon = h3.cell_to_latlng(node_now.hex_id)

            if neighbour == node_end:
                neighbour_lat = lat_end
                neighbour_lon = lon_end
            else:
                neighbour_lat, neighbour_lon = h3.cell_to_latlng(neighbour.hex_id)

            if (
                check_land_edges
                and data.get_land_edge(
                    np.array([node_now_lat, neighbour_lat]),
                    np.array([node_now_lon, neighbour_lon]),
                ).all()
            ):
                continue

            # Push or update the neighbour in the priority queue
            if neighbour not in open_list:
                heapq.heappush(open_list, neighbour)
            else:
                node_old = open_list[open_list.index(neighbour)]
                if neighbour.g < node_old.g:
                    open_list.remove(node_old)
                    heapq.heappush(open_list, neighbour)

        closed_list.append(node_now)

    if nodes_route is None:
        raise ValueError("The route is not possible with the given parameters.")

    nodes_route[0] = (lat_start, lon_start)
    nodes_route[-1] = (lat_end, lon_end)

    latitudes, longitudes = list(zip(*nodes_route, strict=False))
    latitudes, longitudes = np.array(latitudes), np.array(longitudes)

    # Return latitude and longitude arrays for the circumnavigated route.
    latitudes = np.array(latitudes)
    longitudes = np.array(longitudes)
    return latitudes, longitudes
