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
    interpolator = ocean.currents_interpolator
    begin = jnp.array(interpolator.begin, dtype=jnp.float32)[None, :]
    spacing = jnp.array(interpolator.spacing, dtype=jnp.float32)[None, :]
    vmat = jnp.array(interpolator.data[0], dtype=jnp.float32)
    umat = jnp.array(interpolator.data[1], dtype=jnp.float32)
    order = interpolator.order

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

        # Create the coordinates for interpolation
        x = jnp.array([ts_full, lat, lon]).T

        # Normalize coordinates
        coords = (x - begin) / spacing

        # Interpolate u and v components
        u = jax.scipy.ndimage.map_coordinates(umat, coords.T, order=order, mode="wrap")
        v = jax.scipy.ndimage.map_coordinates(vmat, coords.T, order=order, mode="wrap")

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
    bounding_border: int = 10,
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
    bounding_border: int, optional
        Border size for bounding box, by default 10.
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
        bounding_border=bounding_border,
        use_waves=False,
    )

    # Load ocean and land data
    ocean: Ocean = dict_instance["data"]
    vectorfield = get_currents_to_vectorfield(ocean)
    land = LandBenchmark(ocean, outbounds_is_land=True, penalize_segments=False)

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
    grid_resolution: int = 4,
    neighbour_disk_size: int = 3,
    land_dilation: int = 0,
    fms_patience: int = 100,
    fms_damping: float = 0.9,
    fms_maxfevals: int = int(1e6),
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
    date_end : np.datetime64 | None, optional
        End date for the route, by default None.
    vel_ship : float, optional
        Speed through water of the ship in knots, by default 10.0.
    grid_resolution : int, optional
        Grid resolution in kilometers, by default 4.
    neighbour_disk_size : int, optional
        Neighbour disk size for A* search, by default 3.
    land_dilation : int, optional
        Land dilation in number of cells, by default 0.
    fms_patience : int, optional
        Patience for FMS optimization, by default 100.
    fms_damping : float, optional
        Damping factor for FMS optimization, by default 0.9.
    fms_maxfevals : int, optional
        Maximum number of function evaluations for FMS optimization, by default 1e6.
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
    opt = Circumnavigate(
        grid_resolution=grid_resolution,
        neighbour_disk_size=neighbour_disk_size,
        land_dilation=land_dilation,
        num_iter=0,  # No FMS refinement here
    )
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

    # Include x10 points in the middle of each segment for better FMS refinement
    curve_fine = []
    for i in range(len(curve) - 1):
        p0 = curve[i]
        p1 = curve[i + 1]
        curve_fine.append(p0)
        for j in range(1, 10):
            t = j / 10.0
            p = (1 - t) * p0 + t * p1
            curve_fine.append(p)
    curve_fine.append(curve[-1])
    curve = jnp.array(curve_fine)

    # Refine the route using FMS optimization
    curves, _ = optimize_fms(
        vectorfield_zero,
        curve=curve,
        land=land,
        travel_stw=1.0,
        patience=fms_patience,
        damping=fms_damping,
        maxfevals=fms_maxfevals,
        spherical_correction=True,
        verbose=verbose,
    )

    return curves[0]


def optimize_benchmark_instance(
    dict_instance: dict[str, Any],
    penalty: float = 1e8,
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
    curve0: jnp.ndarray | None = None,
    init_circumnavigate: bool = False,
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
        Penalty for land points, by default 1e8
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
    curve0 : jnp.ndarray | None, optional
        Initial curve for the optimization (shape L x 2). Coordinates (lon, lat).
        If None, will initialize with circumnavigation route if `init_circumnavigate`
        is True. By default None
    init_circumnavigate : bool, optional
        Whether to initialize the optimization with a circumnavigation route.
        By default False.
    seed : int, optional
        Random seed for reproducibility. By default jnp.nan
    verbose : bool, optional
        By default True

    Returns
    -------
    tuple[jnp.ndarray, float]
        The optimized curve (shape L x 2), and the fuel cost
    """
    if curve0 is not None:
        if verbose:
            print("[INFO] Using provided initial curve for optimization.")
    elif init_circumnavigate:
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
            grid_resolution=4,
            neighbour_disk_size=3,
            land_dilation=0,
            fms_maxfevals=10000,
            fms_damping=0.0,
            verbose=verbose,
        )
        assert (
            curve0.ndim == 2 and curve0.shape[1] == 2
        ), f"Curve must have shape (L, 2), but got {curve0.shape}"
        if verbose:
            print("[INFO] Circumnavigation route initialized.")
    else:
        curve0 = None

    with jax.disable_jit():
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
            spherical_correction=True,
            seed=seed,
            verbose=verbose,
        )

    return curve_best, dict_cmaes
