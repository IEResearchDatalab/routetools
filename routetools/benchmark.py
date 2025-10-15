import time
from collections.abc import Callable
from math import ceil
from typing import Any

import jax.numpy as jnp
from wrr_bench.ocean import Ocean

from routetools.cmaes import _cma_evolution_strategy, control_to_curve
from routetools.land import Land


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
        lat: jnp.ndarray, lon: jnp.ndarray, ts: jnp.ndarray
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

        # Get currents
        v, u = ocean.get_currents(lat, lon, ts_full)

        # Reshape to the original shape if needed
        if shape is not None:
            v = v.reshape(shape)
            u = u.reshape(shape)

        return v, u

    return vectorfield


class LandBenchmark(Land):
    """Land penalization for benchmark instances."""

    def __init__(
        self,
        ocean: Ocean,
        resolution: int | tuple[int, int] | None = None,
    ) -> None:
        """Land penalization for benchmark instances."""
        # Ensure resolution is 2D
        if resolution is None:
            resolution = (1, 1)
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
        lenx = ceil(up - bottom) * resolution[0]
        leny = ceil(right - left) * resolution[1]
        self.x = jnp.linspace(bottom, up, lenx)
        self.y = jnp.linspace(left, right, leny)
        xx, yy = jnp.meshgrid(self.x, self.y, indexing="ij")
        xx = xx.flatten()
        yy = yy.flatten()
        array = ocean.get_land(xx, yy)
        array = array.reshape((lenx, leny))  # Transpose to match x,y
        super().__init__(
            xlim=(bottom, up),
            ylim=(left, right),
            water_level=0.5,
            resolution=resolution,
            land_array=array,
            outbounds_is_land=True,
        )


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
    seed: float = jnp.nan,
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
    seed : int, optional
        Random seed for reproducibility. By default jnp.nan
    verbose : bool, optional
        By default True

    Returns
    -------
    tuple[jnp.ndarray, float]
        The optimized curve (shape L x 2), and the fuel cost
    """
    # Extract relevant information from the problem instance
    src = jnp.array([dict_instance["lat_start"], dict_instance["lon_start"]])
    dst = jnp.array([dict_instance["lat_end"], dict_instance["lon_end"]])

    # Load ocean and land data
    ocean: Ocean = dict_instance["data"]
    vectorfield = get_currents_to_vectorfield(ocean)
    land = LandBenchmark(ocean)

    # Extract other parameters
    travel_stw = dict_instance.get("vel_ship")
    travel_time = dict_instance.get("travel_time")

    # Initial solution as a straight line
    x0 = jnp.linspace(src, dst, K - 2).flatten()
    # Initial standard deviation to sample new solutions
    # One sigma is half the distance between src and dst
    sigma0 = float(jnp.linalg.norm(dst - src) * sigma0 / 2)

    start = time.time()
    es = _cma_evolution_strategy(
        vectorfield=vectorfield,
        src=src,
        dst=dst,
        x0=x0,
        land=land,
        penalty=penalty,
        travel_stw=travel_stw,
        travel_time=travel_time,
        L=L,
        num_pieces=num_pieces,
        popsize=popsize,
        sigma0=sigma0,
        tolfun=tolfun,
        damping=damping,
        maxfevals=maxfevals,
        seed=seed,
        verbose=verbose,
    )
    if verbose:
        print("Optimization time:", time.time() - start)
        print("Fuel cost:", es.best.f)

    Xbest = es.best.x[None, :]
    curve_best = control_to_curve(Xbest, src, dst, L=L, num_pieces=num_pieces)[0, ...]

    dict_cmaes = {
        "cost": es.best.f,
        "niter": es.countiter,
        "comp_time": int(round(time.time() - start)),
        "vectorfield": vectorfield,
        "land": land,
    }
    return curve_best, dict_cmaes
