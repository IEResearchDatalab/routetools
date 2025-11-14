from collections.abc import Callable
from math import ceil
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from wrr_bench.benchmark import load
from wrr_bench.ocean import Ocean
from wrr_utils.optimization import Circumnavigate
from wrr_utils.route import Route

from routetools.cmaes import optimize
from routetools.fms import optimize_fms
from routetools.land import Land
from routetools.vectorfield import vectorfield_zero


def get_currents_to_vectorfield(
    ocean: Ocean,
) -> Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]:
    """Convert an Ocean instance to a vector field function.

    The returned function takes latitude, longitude, and time arrays as input,
    and returns the corresponding current vectors (v, u).

    Parameters
    ----------
    ocean : Ocean
        An instance of the Ocean class to retrieve current data from.

    Returns
    -------
    Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray
    , jnp.ndarray]]
        A function that takes latitude, longitude, and time arrays as input,
        and returns the corresponding current vectors (v, u).
    """

    # Get currents requires len(ts) == len(lat) == len(lon)
    # But our code handles len(ts) < len(lat)
    # So we create a wrapper function
    def vectorfield(
        lon: jnp.ndarray, lat: jnp.ndarray, ts: jnp.ndarray | int | float
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        # If `ts` is a single value, make it an array
        if isinstance(ts, int | float):
            ts = jnp.array([ts])
        # Repeat the last time value until it matches the length of lat and lon
        diff = len(lat) - len(ts)
        ts_full = jnp.concatenate([ts, jnp.full(diff, ts[-1])]) if diff > 0 else ts

        # Make ts_full match the shape of lat and lon if they are 2D,
        # by repeating columns
        if lat.ndim > 1:
            ts_full = jnp.repeat(ts_full[:, None], lat.shape[1], axis=1)
            # Next, flatten them
            shape = lat.shape  # Save the original shape
            ts_full = ts_full.flatten()
            lat = lat.flatten()
            lon = lon.flatten()
        else:
            shape = None  # No need to reshape later

        # Ensure JAX JIT is disabled, check if lat is a traced array
        if isinstance(lat, jax.core.Tracer):
            raise RuntimeError(
                "JAX JIT is enabled, which is incompatible with "
                "ocean.get_currents() from wrr_bench.ocean."
                "Please disable JIT by setting the environment variable "
                "JAX_DISABLE_JIT=1 before importing any routetools module."
            )
        v, u = ocean.get_currents(lat, lon, ts_full)

        # Reshape to the original shape if needed
        if shape is not None:
            u = u.reshape(shape)
            v = v.reshape(shape)

        return u, v

    return vectorfield


class LandBenchmark(Land):
    """Land penalization for benchmark instances."""

    def __init__(
        self,
        ocean: Ocean,
        resolution: int | tuple[int, int] | None = None,
        **kwargs: Any,
    ) -> None:
        """Land penalization for benchmark instances."""
        # Ensure resolution is 2D
        if resolution is None:
            resolution = (10, 10)
        elif isinstance(resolution, int):
            resolution = (resolution, resolution)
        elif len(resolution) != 2:
            raise ValueError(
                f"""
                Resolution must be a tuple of length 2, not {len(resolution)}
                """
            )
        self.ocean = ocean
        bottom, left, up, right = ocean.bounding_box
        lenx = ceil(right - left) * resolution[0]
        leny = ceil(up - bottom) * resolution[1]
        self.x = jnp.linspace(left, right, lenx)
        self.y = jnp.linspace(bottom, up, leny)
        xx, yy = jnp.meshgrid(self.x, self.y, indexing="ij")
        xx = xx.flatten()
        yy = yy.flatten()
        array = ocean.get_land(yy, xx)  # Takes lat, lon
        # Transpose to match x,y
        array = array.reshape((lenx, leny)).astype(jnp.float32)
        super().__init__(
            xlim=(left, right),
            ylim=(bottom, up),
            resolution=resolution,
            land_array=array,
            **kwargs,
        )


def load_benchmark_instance(
    instance_name: str,
    date_start: str = "2023-01-08",
    vel_ship: int = 6,
    data_path: str = "./data",
) -> dict[str, Any]:
    """
    Extract relevant information from a benchmark instance dictionary.

    Parameters
    ----------
    instance_name : str
        Name of the benchmark instance to load.
    date_start : str, optional
        Start date for the benchmark instance, by default "2023-01-08".
    vel_ship : int, optional
        Velocity of the ship in knots, by default 6.
    data_path : str, optional
        Path to the data directory, by default "./data".

    Returns
    -------
    dict[str, Any]
        A dictionary containing the extracted information:
        - src: jnp.ndarray of shape (2,) with the starting point (lon, lat)
        - dst: jnp.ndarray of shape (2,) with the ending point (lon, lat)
        - travel_stw: float or None with the speed through water
        - travel_time: float or None with the total travel time
        - vectorfield: Callable function to get the current vectors
        - land: Land instance for land penalization
    """
    dict_instance = load(
        instance_name,
        date_start=date_start,
        vel_ship=vel_ship,
        data_path=data_path,
        use_waves=False,
    )

    # Load ocean and land data
    ocean: Ocean = dict_instance["data"]
    vectorfield = get_currents_to_vectorfield(ocean)
    land = LandBenchmark(ocean, outbounds_is_land=True)

    # jnp arrays for src and dst
    src = jnp.array([dict_instance["lon_start"], dict_instance["lat_start"]])
    dst = jnp.array([dict_instance["lon_end"], dict_instance["lat_end"]])

    return {
        # These ones are used for circumnavigation
        "lat_start": dict_instance["lat_start"],
        "lon_start": dict_instance["lon_start"],
        "lat_end": dict_instance["lat_end"],
        "lon_end": dict_instance["lon_end"],
        "vel_ship": dict_instance.get("vel_ship", None),
        "travel_time": dict_instance.get("travel_time", None),
        "date_start": dict_instance["date_start"],
        "date_end": dict_instance.get("date_end", None),
        # These ones are used in CMA-ES optimization
        "src": src,
        "dst": dst,
        "data": ocean,
        "vectorfield": vectorfield,
        "land": land,
        "travel_stw": dict_instance.get("vel_ship", None),
    }


def circumnavigate(
    lat_start: float,
    lon_start: float,
    lat_end: float,
    lon_end: float,
    ocean: Ocean,
    land: LandBenchmark,
    date_start: np.datetime64,
    date_end: np.datetime64 | None = None,
    vel_ship: float = 10.0,
    verbose: bool = False,
) -> jnp.ndarray:
    """Run A* on the h3 cell graph and return a list of (lat, lon) points.

    If start or end are not inside any cell in `cells`, we snap them to the
    nearest available cell centroid.
    It then refines the resulting route using FMS optimization.

    Parameters
    ----------
    src : tuple[float, float]
        Source point as (lon, lat).
    dst : tuple[float, float]
        Destination point as (lon, lat).
    ocean : Ocean
        Ocean instance to retrieve current data from.
    land : Land | None, optional
        Land instance to derive navigable cells from. If None, assumes no land
        constraints, by default None.
    date_start : np.datetime64
        Start date for the route.
    verbose : bool, optional
        Whether to print verbose output, by default False.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        Tuple containing:
        - The refined route as an array of (lon, lat) points.
        - The initial A* route as an array of (lon, lat) points.
    """
    # Circumnavigate optimizer
    opt = Circumnavigate(num_iter=0)  # No FMS refinement here
    route: Route = opt.optimize(
        lat_start=lat_start,
        lon_start=lon_start,
        lat_end=lat_end,
        lon_end=lon_end,
        data=ocean,
        bounding_box=ocean.bounding_box,  # Required explicitly
        date_start=date_start,
        date_end=date_end,  # Required explicitly, but not used
        vel_ship=vel_ship,  # Required explicitly, but not used
    )

    # Retrieve the curve from the Route instance
    lats = jnp.asarray(route.lats)
    lons = jnp.asarray(route.lons)
    curve = jnp.stack([lons, lats], axis=1)
    assert (
        curve.ndim == 2 and curve.shape[1] == 2
    ), f"Curve must have shape (L, 2), but got {curve.shape}"

    # Refine the route using FMS optimization
    curves, _ = optimize_fms(
        vectorfield_zero, curve=curve, land=land, travel_stw=1.0, verbose=verbose
    )

    return curves[0]


def optimize_benchmark_instance(
    dict_instance: dict[str, Any],
    penalty: float = 10,
    K: int = 6,
    L: int = 64,
    num_pieces: int = 1,
    popsize: int = 200,
    sigma0: float = 1,
    tolfun: float = 1e-4,
    damping: float = 1,
    maxfevals: int = 25000,
    weight_l1: float = 1.0,
    weight_l2: float = 0.0,
    seed: float = jnp.nan,
    init_circumnavigate: bool = True,
    verbose: bool = True,
) -> tuple[jnp.ndarray, dict[str, Any]]:
    """
    Solve the vessel routing problem for a given vector field.

    Two modes are supported:
        - Fixed speed-through-water. Optimize the vessel heading
        - Fixed total travel time. Optimize heading and speed-through-water

    Algorithm: parameterize the space of solutions with a Bézier curve,
    and optimize the control points using the CMA-ES optimization method.

    Parameters
    ----------
    dict_instance : dict
        Dictionary containing the problem instance, as loaded with
        `wrr_bench.benchmark.load`
        The problem instance contains the following information:
        lat_start, lon_start, lat_end, lon_end, date_start, vel_ship, bounding_box, data
    penalty : float, optional
        Penalty for land points, by default 10
    K : int, optional
        Number of free Bézier control points. By default 6
    L : int, optional
        Number of points evaluated in each Bézier curve. By default 64
    popsize : int, optional
        Population size for the CMA-ES optimizer. By default 200
    sigma0 : float, optional
        Initial standard deviation to sample new solutions. By default 1
    tolfun : float, optional
        Tolerance for the optimizer. By default 1e-4
    damping : float, optional
        Damping factor for the optimizer. By default 1
    maxfevals : int, optional
        Maximum number of function evaluations. By default 25000
    weight_l1 : float, optional
        Weight for the L1 norm in the combined cost. Default is 1.0.
    weight_l2 : float, optional
        Weight for the L2 norm in the combined cost. Default is 0.0.
    seed : int, optional
        Random seed for reproducibility. By default jnp.nan
    verbose : bool, optional
        By default True

    Returns
    -------
    tuple[jnp.ndarray, float]
        The optimized curve (shape L x 2), and the fuel cost
    """
    if init_circumnavigate:
        if verbose:
            print("[INFO] Initializing with circumnavigation route...")
        # Initialize the circumnavigation route
        curve0 = circumnavigate(
            lat_start=dict_instance["lat_start"],
            lon_start=dict_instance["lon_start"],
            lat_end=dict_instance["lat_end"],
            lon_end=dict_instance["lon_end"],
            ocean=dict_instance["data"],
            land=dict_instance["land"],
            date_start=dict_instance["date_start"],
            date_end=dict_instance.get("date_end"),
            vel_ship=dict_instance.get("vel_ship"),
            verbose=verbose,
        )
        assert (
            curve0.ndim == 2 and curve0.shape[1] == 2
        ), f"Curve must have shape (L, 2), but got {curve0.shape}"
        if verbose:
            print("[INFO] Circumnavigation route initialized.")
    else:
        curve0 = None

    curve_best, dict_cmaes = optimize(
        vectorfield=dict_instance["vectorfield"],
        src=dict_instance["src"],
        dst=dict_instance["dst"],
        curve0=curve0,
        land=dict_instance["land"],
        travel_stw=dict_instance["travel_stw"],
        travel_time=dict_instance["travel_time"],
        penalty=penalty,
        K=K,
        L=L,
        num_pieces=num_pieces,
        popsize=popsize,
        sigma0=sigma0,
        tolfun=tolfun,
        damping=damping,
        maxfevals=maxfevals,
        weight_l1=weight_l1,
        weight_l2=weight_l2,
        seed=seed,
        verbose=verbose,
    )

    return curve_best, dict_cmaes


def optimize_fms_benchmark_instance(
    dict_instance: dict[str, Any],
    curve: jnp.ndarray,
    tolfun: float = 1e-5,
    damping: float = 1,
    maxfevals: int = 10000,
    weight_l1: float = 1.0,
    weight_l2: float = 0.0,
    verbose: bool = True,
) -> tuple[jnp.ndarray, dict[str, Any]]:
    """
    Refine a given path using the FMS variational optimization algorithm.

    Parameters
    ----------
    dict_instance : dict
        Dictionary containing the problem instance, as loaded with
        `wrr_bench.benchmark.load`
        The problem instance contains the following information:
        lat_start, lon_start, lat_end, lon_end, date_start, vel_ship, bounding_box, data
    curve : jnp.ndarray
        Initial path to refine (shape L x 2)
    tolfun : float, optional
        Tolerance for the optimizer. By default 1e-5
    damping : float, optional
        Damping factor for the optimizer. By default 1
    maxfevals : int, optional
        Maximum number of function evaluations. By default 10000
    weight_l1 : float, optional
        Weight for the L1 norm in the combined cost. Default is 1.0.
    weight_l2 : float, optional
        Weight for the L2 norm in the combined cost. Default is 0.0.
    verbose : bool, optional
        By default True

    Returns
    -------
    tuple[jnp.ndarray, dict[str, Any]]
        The optimized curve (shape L x 2), and a dictionary with information about
        the optimization process
    """
    assert (
        curve.ndim == 2 and curve.shape[1] == 2
    ), f"Curve must have shape (L, 2), but got {curve.shape}"

    curve_opt, dict_fms = optimize_fms(
        vectorfield=dict_instance["vectorfield"],
        curve=curve,
        land=dict_instance["land"],
        travel_stw=dict_instance["travel_stw"],
        travel_time=dict_instance["travel_time"],
        tolfun=tolfun,
        damping=damping,
        maxfevals=maxfevals,
        weight_l1=weight_l1,
        weight_l2=weight_l2,
        verbose=verbose,
    )

    return curve_opt, dict_fms
