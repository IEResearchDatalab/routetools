# Disable JAX before anything else
# This must be at the very top, before any JAX import
# Required to ensure compatibility with wrr_bench
import os

os.environ["JAX_DISABLE_JIT"] = "1"

# Now import the rest of the modules

from collections.abc import Callable
from math import ceil
from typing import Any

import jax.numpy as jnp
from wrr_bench.ocean import Ocean

from routetools.cmaes import optimize
from routetools.fms import optimize_fms
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
        lon: jnp.ndarray, lat: jnp.ndarray, ts: jnp.ndarray
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


def extract_benchmark_instance(
    dict_instance: dict[str, Any],
) -> dict[str, Any]:
    """
    Extract relevant information from a benchmark instance dictionary.

    Parameters
    ----------
    dict_instance : dict
        Dictionary containing the problem instance, as loaded with
        `wrr_bench.benchmark.load`
        The problem instance contains the following information:
        lat_start, lon_start, lat_end, lon_end, date_start, vel_ship, bounding_box, data

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
    # Extract relevant information from the problem instance
    src = jnp.array([dict_instance["lon_start"], dict_instance["lat_start"]])
    dst = jnp.array([dict_instance["lon_end"], dict_instance["lat_end"]])

    # Load ocean and land data
    ocean: Ocean = dict_instance["data"]
    vectorfield = get_currents_to_vectorfield(ocean)
    land = LandBenchmark(ocean, outbounds_is_land=True)

    # Extract other parameters
    travel_stw = dict_instance.get("vel_ship")
    travel_time = dict_instance.get("travel_time")

    return {
        "src": src,
        "dst": dst,
        "travel_stw": travel_stw,
        "travel_time": travel_time,
        "vectorfield": vectorfield,
        "land": land,
    }


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
    dict_extracted = extract_benchmark_instance(dict_instance)

    curve_best, dict_cmaes = optimize(
        src=dict_extracted["src"],
        dst=dict_extracted["dst"],
        vectorfield=dict_extracted["vectorfield"],
        land=dict_extracted["land"],
        travel_stw=dict_extracted["travel_stw"],
        travel_time=dict_extracted["travel_time"],
        penalty=penalty,
        K=K,
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

    return curve_best, dict_cmaes


def optimize_fms_benchmark_instance(
    dict_instance: dict[str, Any],
    curve: jnp.ndarray,
    tolfun: float = 1e-5,
    damping: float = 1,
    maxfevals: int = 10000,
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
    verbose : bool, optional
        By default True

    Returns
    -------
    tuple[jnp.ndarray, dict[str, Any]]
        The optimized curve (shape L x 2), and a dictionary with information about
        the optimization process
    """
    dict_extracted = extract_benchmark_instance(dict_instance)

    curve_opt, dict_fms = optimize_fms(
        vectorfield=dict_extracted["vectorfield"],
        curve=curve,
        land=dict_extracted["land"],
        travel_stw=dict_extracted["travel_stw"],
        travel_time=dict_extracted["travel_time"],
        tolfun=tolfun,
        damping=damping,
        maxfevals=maxfevals,
        verbose=verbose,
    )

    return curve_opt, dict_fms
