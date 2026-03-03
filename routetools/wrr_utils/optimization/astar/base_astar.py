from __future__ import annotations

import heapq
import json
import time
from collections.abc import Iterable

import numpy as np
import pandas as pd
from h3.api import basic_int as h3

from routetools.wrr_bench.ocean import Ocean
from routetools.wrr_utils.optimization.astar.hexagonal_grid import (
    multipolygon_to_h3_cells,
)
from routetools.wrr_utils.optimization.base_optimizer import BaseOptimizer
from routetools.wrr_utils.route import Route
from routetools.wrr_utils.simulation import split_route_segments
from routetools.wrr_utils.utils.polygons import invert_polygon


class Node:
    """A node class for A* Pathfinding."""

    def __init__(self, hex_id: int, parent: Node = None, vel_ship: float | None = None):
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


class BaseAstar(BaseOptimizer):
    """Base class for A* optimizers over an H3 hexagonal grid."""

    def __init__(
        self,
        grid_resolution: int = 4,
        neighbour_disk_size: int = 3,
        land_dilation: int = 0,
        weighted_heuristic: float = 1.1,
        edge_resolution: float | None = None,
        refine_route: bool = False,
        check_land_edges: bool = True,
        closed_nodes_path: str = None,
        consistency_data_path: str = None,
    ):
        """
        Optimizer that uses the A* algorithm to find the shortest route.

        Parameters
        ----------
        grid_resolution : int, optional
            Resolution of hexagonal grid, by default 4.
        neighbour_disk_size : int, optional
            Size of the disk to search for neighbours, by default 3.
        land_dilation : int, optional
            Distance to land from cells border, by default 0
        weighted_heuristic : float, optional
            Weight of the heuristic, by default 1.1.
        edge_resolution : float, optional
            When computing an edge, this is the minimum distance (in meters)
            between intermediate points. By default None.
        refine_route : bool, optional
            When the computation is finished, whether the route is refined to get
            hexagon by hexagon.
        check_land_edges : bool, optional
            Whether to check if the edge is land, by default True.
        closed_nodes_path : str, optional
            Path to save the closed nodes of the last optimization, by default None.
        consistency_data_path : str, optional
            Path to save the consistency data of the last optimization, by
            default None. It will store heuristic of current node
            (`h_n`), cost from current to neighbour (`cost_np`), and
            heuristic of neighbour (`h_p`). This will work differently than
            `last_optimization_summary` because consistency data is too heavy.
        """
        self.grid_resolution = grid_resolution
        self.neighbour_disk_size = neighbour_disk_size
        self.land_dilation = land_dilation
        self.weighted_heuristic = weighted_heuristic
        self.edge_resolution = edge_resolution
        self.refine_route = refine_route
        self.check_land_edges = check_land_edges
        self.closed_nodes_path = closed_nodes_path
        self.total_closed_nodes = None
        self.total_expanded_nodes = None
        self.comp_time = None
        self.consistency_data_path = consistency_data_path

    def _get_route(self, last_node: Node) -> list[tuple]:
        route: list[tuple] = []
        node_now = last_node
        while node_now is not None:
            parent = node_now.parent

            if (
                self.refine_route
                and parent is not None
                and not h3.are_neighbor_cells(parent.hex_id, node_now.hex_id)
            ):
                sub_route = h3.grid_path_cells(node_now.hex_id, parent.hex_id)
                route.extend([h3.cell_to_latlng(hex_id) for hex_id in sub_route][:-1])
            else:
                route.append(h3.cell_to_latlng(node_now.hex_id))
            node_now = node_now.parent
        return route[::-1]

    def _get_neighbours(self, node: Node, ocean_cells: set[int]) -> list[Node]:
        neighbours: list[Node] = []
        for neighbour in h3.grid_disk(node.hex_id, self.neighbour_disk_size):
            if neighbour != node.hex_id and neighbour in ocean_cells:
                neighbours.append(Node(neighbour))
        return neighbours

    def _heuristic(
        self,
        node: Node,
        end_node: Node,
        **kwargs: dict,
    ) -> float:
        """
        Compulsory abstract method to implement in the child class.

        It must define the heuristic used by the algorithm.

        Parameters
        ----------
        node : Node
            Current being expanded node.
        end_node : Node
            Objective node to be reached.

        Returns
        -------
        float
            Value of the heuristic.
        """
        pass

    def _g_delta(
        self,
        node: Node,
        neighbour: Node,
        **kwargs: dict,
    ) -> float:
        """
        Compulsory abstract method to implement in the child class.

        It must define the real cost of the movement between two nodes.

        Parameters
        ----------
        node : Node
            Current node.
        neighbour : Node
            Neighbour node.

        Returns
        -------
        float
            Cost of the movement.
        """
        pass

    def _update_neighbour_cost(
        self,
        node_now: Node,
        neighbour: Node,
        node_end: Node,
        data: Ocean,
        date_start: np.datetime64,
        date_end: np.datetime64,
        vel_ship: float,
    ) -> float:
        g_delta = self._g_delta(
            node_now,
            neighbour,
            data=data,
            date_start=date_start,
            date_end=date_end,
            vel_ship=vel_ship,
        )
        neighbour.g = node_now.g + g_delta
        neighbour.h = self._heuristic(
            neighbour,
            node_end,
            data=data,
            date_start=date_start,
            date_end=date_end,
            vel_ship=vel_ship,
        )
        neighbour.f = neighbour.g + self.weighted_heuristic * neighbour.h
        return g_delta

    def optimize(
        self,
        lat_start: float,
        lon_start: float,
        lat_end: float,
        lon_end: float,
        data: Ocean,
        date_start: np.datetime64,
        vel_ship: float | None = None,
        date_end: np.datetime64 | None = None,
        bounding_box: Iterable[float] | None = None,
        **kwargs,
    ) -> Route:
        """
        Calculate the fastest route between two points.

        Parameters
        ----------
        lat_start : float
            Latitude of the starting point.
        lon_start : float
            Longitude of the starting point.
        lat_end : float
            Latitude of the ending point.
        lon_end : float
            Longitude of the ending point.
        data : Ocean
            Object that contains methods to get the ocean data.
        date_start : dt.datetime
            Date of the starting point.
        vel_ship : float
            Ship velocity.
        bounding_box : Iterable[float], optional
            Bounding box of the computation, by default None. The order of
            the coordinates is (lat_bottom_left, lon_bottom_left,
            lat_up_right, lon_up_right).

        Returns
        -------
        Route
            Optimal route.
        """
        self.comp_time = time.time()
        hex_start = h3.latlng_to_cell(lat_start, lon_start, self.grid_resolution)
        hex_end = h3.latlng_to_cell(lat_end, lon_end, self.grid_resolution)

        node_start = Node(hex_start)
        node_end = Node(hex_end)

        # Include JIT time if needed
        if date_end is not None:
            dt = (date_end - date_start).astype("timedelta64[us]").astype(float) / 1e6
            node_end.dt = dt

        ocean_cells = multipolygon_to_h3_cells(
            invert_polygon(data.shapely_ocean, bounding_box),
            res=self.grid_resolution,
            land_dilation=self.land_dilation,
        )

        ocean_cells.add(node_start.hex_id)
        ocean_cells.add(node_end.hex_id)

        # Create priority queue and add start node
        open_list: list[Node] = []
        heapq.heapify(open_list)  # PriorityQueue
        heapq.heappush(open_list, node_start)

        closed_list: list[Node] = []

        consistency_data = []

        nodes_route = None
        while len(open_list) > 0:
            node_now = heapq.heappop(open_list)

            if node_now == node_end:
                nodes_route = self._get_route(node_now)
                break

            for neighbour in self._get_neighbours(node_now, ocean_cells):
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
                    self.check_land_edges
                    and data.get_land_edge(
                        np.array([node_now_lat, neighbour_lat]),
                        np.array([node_now_lon, neighbour_lon]),
                    ).all()
                ):
                    # Since start and end can be in cells outside the graph, the cell
                    # center may lie on land even if the point itself is in the ocean.
                    # In that case, outputs and arrivals to these nodes are not
                    # affected by this land intersection condition.
                    continue

                g_delta = self._update_neighbour_cost(
                    node_now=node_now,
                    neighbour=neighbour,
                    node_end=node_end,
                    data=data,
                    date_start=date_start,
                    date_end=date_end,
                    vel_ship=vel_ship,
                )
                if self.consistency_data_path is not None:
                    consistency_data.append(
                        {
                            "id_n": node_now.hex_id,
                            "id_p": neighbour.hex_id,
                            "h_n": node_now.h,
                            "h_p": neighbour.h,
                            "cost_np": g_delta,
                        }
                    )

                # Push or update the neighbour in the priority queue
                if neighbour not in open_list:
                    heapq.heappush(open_list, neighbour)
                else:
                    node_old = open_list[open_list.index(neighbour)]
                    if neighbour.g < node_old.g:
                        open_list.remove(node_old)
                        heapq.heappush(open_list, neighbour)

            closed_list.append(node_now)

        self.total_closed_nodes = len(closed_list)
        self.total_expanded_nodes = len(closed_list) + len(open_list)
        if self.closed_nodes_path is not None:
            with open(self.closed_nodes_path, "w") as f:
                json.dump(
                    [
                        {"hex_id": node.hex_id, "g": node.g, "h": node.h, "f": node.f}
                        for node in closed_list
                    ],
                    f,
                )

        if self.consistency_data_path is not None:
            pd.DataFrame(consistency_data).to_csv(self.consistency_data_path)

        if nodes_route is None:
            raise ValueError("The route is not possible with the given parameters.")
        else:
            nodes_route[0] = (lat_start, lon_start)
            nodes_route[-1] = (lat_end, lon_end)

            latitudes, longitudes = list(zip(*nodes_route, strict=False))
            latitudes, longitudes = np.array(latitudes), np.array(longitudes)
            if self.edge_resolution:
                latitudes, longitudes = split_route_segments(
                    latitudes,
                    longitudes,
                    threshold=self.edge_resolution,
                    ocean_data=data,
                )

            # Compute the time of the route
            route = Route.from_start_time(
                latitudes,
                longitudes,
                date_start,
                ocean_data=data,
                vel_ship=vel_ship,
            )

        self.comp_time = time.time() - self.comp_time

        return route

    def last_optimization_summary(self) -> dict:
        """
        Return a summary of the last optimization.

        Returns
        -------
        dict
            Summary of the last optimization containing:
            - weighted_heuristic
            - grid_resolution
            - neighbour_disk_size
            - land_dilation
            - edge_resolution
            - refine_route
            - total_closed_nodes
            - total_expanded_nodes
            - comp_time
        """
        # Number of nodes used and closed
        total_closed = self.total_closed_nodes
        # Number of nodes expanded
        total_expanded = self.total_expanded_nodes
        # Computation time
        comp_time = self.comp_time

        return {
            "weighted_heuristic": self.weighted_heuristic,
            "grid_resolution": self.grid_resolution,
            "neighbour_disk_size": self.neighbour_disk_size,
            "land_dilation": self.land_dilation,
            "edge_resolution": self.edge_resolution,
            "refine_route": self.refine_route,
            "total_closed_nodes": total_closed,
            "total_expanded_nodes": total_expanded,
            "comp_time": comp_time,
        }
