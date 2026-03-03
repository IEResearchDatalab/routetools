import h3.api.basic_int as h3
import typer

from routetools.wrr_bench.ocean import Ocean
from routetools.wrr_utils.benchmark import load
from routetools.wrr_utils.optimization.astar.base_astar import BaseAstar, Node
from routetools.wrr_utils.optimization.dnj import DNJ
from routetools.wrr_utils.route import Route


class Circumnavigate(BaseAstar):
    """Optimizer that circumnavigates using A* and DNJ smoothing."""

    def __init__(
        self,
        num_iter: int = 200,
        damping: float = 0.5,
        threshold: float = 0.1,
        early_stop: int = 5,
        **kwargs: dict,
    ):
        """Initialize the circumnavigate optimizer.

        Parameters
        ----------
        num_iter : int, optional
            Number of DNJ iterations, by default 200
        damping : float, optional
            Similar to a 'learning rate' controls how strongly the points
            are moved towards the optimal solution, by default 0.5
        threshold : float, optional
            Maximum value that a point can be moved in a single iteration,
            in degrees, by default 0.1
        kwargs : dict
            All the other A* parameters.
        """
        super().__init__(**kwargs)
        self.num_iter = num_iter
        self.damping = damping
        self.threshold = threshold
        self.early_stop = early_stop

    def _heuristic(self, n1: Node, n2: Node, **kwargs: dict):
        node1_lat, node1_lon = h3.cell_to_latlng(n1.hex_id)
        node2_lat, node2_lon = h3.cell_to_latlng(n2.hex_id)
        return h3.great_circle_distance(
            (node1_lat, node1_lon), (node2_lat, node2_lon), unit="m"
        )

    def _g_delta(
        self,
        node_now: Node,
        neighbour: Node,
        **kwargs: dict,
    ) -> float:
        """
        Calculate the distance between two nodes.

        Parameters
        ----------
        node_now : Node
            Current node.
        neighbour : Node
            Neighbour node.

        Returns
        -------
        float
            Distance between the two nodes.
        """
        node_now_lat, node_now_lon = h3.cell_to_latlng(node_now.hex_id)
        neighbour_lat, neighbour_lon = h3.cell_to_latlng(neighbour.hex_id)

        return h3.great_circle_distance(
            (node_now_lat, node_now_lon),
            (neighbour_lat, neighbour_lon),
            unit="m",
        )

    def optimize(self, **kwargs):
        """
        Optimize the route using the A* algorithm and DNJ smoothing.

        The ocean data is replaced by one with all zero currents.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        Route
            The optimized route with a fine reparametrization.
        """
        # Remove the kwarg "data" and replace it with zero currents
        data: Ocean = kwargs.pop("data")
        ocean_zero = Ocean(
            bounding_box=data.bounding_box,
            land_file=data.land_file,
            interp_method=data.interp_method,
            radius=data.radius,
        )
        route = super().optimize(data=ocean_zero, **kwargs)
        # If the A* cannot find a route, it will return None
        if route is None:
            return None

        if self.num_iter > 0:
            # Apply DNJ smoothing
            dnj = DNJ(
                num_iter=self.num_iter,
                damping=self.damping,
                threshold=self.threshold,
                early_stop=self.early_stop,
            )
            route = dnj.optimize(route)

        # Return the original ocean data
        route = Route.from_start_time(
            lats=route.lats,
            lons=route.lons,
            time_start=route.time_stamps[0],
            vel_ship=route.vel_ship,
            ocean_data=data,
            land_penalization=0,
        )
        return route


def main(
    name_benchmark: str = "DEHAM-USNYC",
    output: str = "output.csv",
    config_file: str = "config/benchmarks.json",
    data_folder: str = "/DATA1/SHARED/weather-routing-data/",
):
    """Generate the optimal route to circumnavigate between two points.

    Parameters
    ----------
    name_bechmark : str
        Name of the benchmark.
    output : str
        Name of the output file.
    config_file : str, optional
        Route to config file, by default "config/benchmarks.json"
    """
    dict_benchmark = load(
        name_benchmark, config_file=config_file, local_data_path=data_folder
    )

    optimizer = Circumnavigate()

    route = optimizer.optimize(**dict_benchmark)

    route.export_to_csv(output)


if __name__ == "__main__":
    typer.run(main)
