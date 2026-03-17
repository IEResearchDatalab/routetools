import time
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import typer
from jax import grad, jacfwd, jacrev, jit, vmap

from routetools.cost import cost_function
from routetools.land import Land
from routetools.vectorfield import vectorfield_fourvortices
from routetools.weather import (
    DEFAULT_HS_LIMIT,
    DEFAULT_TWS_LIMIT,
)
from routetools.weather import (
    weather_penalty as _weather_penalty,
)


def _weather_violation_mask(
    curve: jnp.ndarray,
    *,
    windfield: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]
    ]
    | None = None,
    wavefield: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]
    ]
    | None = None,
    enforce_weather_limits: bool = False,
    tws_limit: float = DEFAULT_TWS_LIMIT,
    hs_limit: float = DEFAULT_HS_LIMIT,
    travel_stw: float | None = None,
    travel_time: float | None = None,
    spherical_correction: bool = False,
    time_offset: float = 0.0,
) -> jnp.ndarray:
    """Return a per-route mask for weather-limit violations."""
    if not enforce_weather_limits or (windfield is None and wavefield is None):
        return jnp.zeros(curve.shape[0], dtype=bool)

    return (
        _weather_penalty(
            curve,
            windfield=windfield,
            wavefield=wavefield,
            tws_limit=tws_limit,
            hs_limit=hs_limit,
            penalty=1.0,
            travel_stw=travel_stw,
            travel_time=travel_time,
            spherical_correction=spherical_correction,
            time_offset=time_offset,
        )
        > 0
    )


def _apply_curve_constraints(
    curve: jnp.ndarray,
    curve_old: jnp.ndarray,
    *,
    land: Land | None = None,
    penalty: float = 0.0,
) -> jnp.ndarray:
    """Reject invalid FMS updates and keep the previous feasible curve.

    Land updates are reverted point-wise.  Weather is *not* enforced here;
    callers use ``_weather_violation_mask`` + ``effective_cost`` to gate
    ``curve_best`` updates instead.  Rolling back an entire route on weather
    violations would prevent FMS from escaping an initially-violating state.
    """
    if land is None or penalty <= 0:
        return curve

    is_land = land(curve) > 0
    return jnp.where(is_land[..., None], curve_old, curve)


def random_piecewise_curve(
    src: jnp.ndarray,
    dst: jnp.ndarray,
    num_curves: int = 1,
    num_points: int = 200,
    seed: int = 0,
) -> jnp.ndarray:
    """
    Generate random piecewise linear curves between src and dst.

    Parameters
    ----------
    src : jnp.ndarray
        Starting point of the curves.
    dst : jnp.ndarray
        Ending point of the curves.
    num_curves : int
        Number of curves to generate.
    key : jax.random.PRNGKey
        Random key for generating random numbers.

    Returns
    -------
    jnp.ndarray
        Generated curves with shape (num_curves, num_segments, 2).
    """
    key = jax.random.PRNGKey(seed)
    num_segments = jax.random.randint(key, (num_curves,), minval=2, maxval=5)
    ls_angs = jax.random.uniform(key, (num_curves * 5,), minval=-0.5, maxval=0.5)
    ls_dist = jax.random.uniform(key, (num_curves * 5,), minval=0.1, maxval=0.9)

    curves = []
    for idx_route in range(num_curves):
        x_start, y_start = src
        x_end, y_end = dst
        x_pts: list[jnp.ndarray] = [x_start]
        y_pts: list[jnp.ndarray] = [y_start]
        ls_d: list[jnp.ndarray] = []
        for idx_seg in range(num_segments[idx_route] - 1):
            dx = x_end - x_pts[-1]
            dy = y_end - y_pts[-1]
            ang = jnp.arctan2(dy, dx)
            ang_dev = 0.5 * ls_angs[idx_route * 5 + idx_seg]
            d = jnp.sqrt(dx**2 + dy**2) * ls_dist[idx_route * 5 + idx_seg]
            x_pts.append(x_pts[-1] + d * jnp.cos(ang + ang_dev))
            y_pts.append(y_pts[-1] + d * jnp.sin(ang + ang_dev))
            ls_d.append(d)
        x_pts.append(x_end)
        y_pts.append(y_end)
        ls_d.append(jnp.sqrt((x_end - x_pts[-2]) ** 2 + (y_end - y_pts[-2]) ** 2))
        dist = jnp.array(ls_d).flatten()
        # To ensure the points of the route are equi-distant,
        # the number of points per segment will depend on its distance
        # in relation to the total distance travelled
        num_points_seg = (num_points * dist / dist.sum()).astype(int)
        # Start generating the points
        x = jnp.array([x_start])
        y = jnp.array([y_start])
        for idx_seg in range(num_segments[idx_route]):
            nps: int = int(num_points_seg[idx_seg]) + 1
            x_new = jnp.linspace(x_pts[idx_seg], x_pts[idx_seg + 1], nps).flatten()
            x = jnp.concatenate([x, x_new[1:]])
            y_new = jnp.linspace(y_pts[idx_seg], y_pts[idx_seg + 1], nps).flatten()
            y = jnp.concatenate([y, y_new[1:]])
        # Ensure the total number of points matches num_points
        if len(x) < num_points:
            x = jnp.concatenate([x, jnp.full(num_points - len(x), x_end)])
            y = jnp.concatenate([y, jnp.full(num_points - len(y), y_end)])
        elif len(x) > num_points:
            x = jnp.concatenate([x[: num_points - 1], x_end])
            y = jnp.concatenate([y[: num_points - 1], y_end])
        curves.append(jnp.stack([x, y], axis=-1))

    return jnp.stack(curves)


def hessian(
    f: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray], argnums: int = 0
) -> Any:
    """
    Compute the Hessian of a function.

    Parameters
    ----------
    f : Callable
        Function to differentiate.
    argnums : int, optional
        Argument number to differentiate, by default 0

    Returns
    -------
    Callable
        Hessian of the function.
    """
    return jacfwd(jacrev(f, argnums=argnums), argnums=argnums)


def optimize_fms(
    vectorfield: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]
    ],
    src: jnp.ndarray | None = None,
    dst: jnp.ndarray | None = None,
    curve: jnp.ndarray | None = None,
    land: Land | None = None,
    windfield: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]
    ]
    | None = None,
    wavefield: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]
    ]
    | None = None,
    penalty: float = 1e10,
    num_curves: int = 10,
    num_points: int = 200,
    travel_stw: float | None = None,
    travel_time: float | None = None,
    patience: int = 50,
    damping: float = 0.9,
    maxfevals: int = 5000,
    weight_l1: float = 1.0,
    weight_l2: float = 0.0,
    spherical_correction: bool = False,
    costfun: Callable | None = None,
    seed: int = 0,
    verbose: bool = True,
    time_offset: float = 0.0,
    enforce_weather_limits: bool = False,
    tws_limit: float = DEFAULT_TWS_LIMIT,
    hs_limit: float = DEFAULT_HS_LIMIT,
) -> tuple[jnp.ndarray, dict[str, Any]]:
    """
    Optimize a curve using the FMS algorithm.

    Source:
    https://doi.org/10.1016/j.ifacol.2021.11.097

    Parameters
    ----------
    vectorfield : Callable[[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]
        Vector field function.
    src : jnp.ndarray | None, optional
        Origin point, by default None
    dst : jnp.ndarray | None, optional
        Destination point, by default None
    curve : jnp.ndarray | None, optional
        Curve to optimize, shape L x 2, by default None
    land_function : Callable[[jnp.ndarray], jnp.ndarray] | None, optional
        Land function, by default None
    windfield : Callable, optional
        Wind field closure used to enforce route weather limits.
    num_curves : int, optional
        Number of curves to optimize, only used when initial curves are not provided,
        by default 10
    num_points : int, optional
        Number of points per curve, only used when initial curves are not provided,
        by default 200
    travel_stw : float | None, optional
        Fixed speed through water, by default None
    travel_time : float | None, optional
        Fixed travel time, by default None
    patience : int, optional
        Number of iterations without improvement before stopping, by default 50
    damping : float, optional
        Damping factor, by default 0.9
    maxfevals : int, optional
        Maximum number of iterations, by default 5000
    weight_l1 : float, optional
        Weight for the L1 norm in the combined cost. Default is 1.0.
    weight_l2 : float, optional
        Weight for the L2 norm in the combined cost. Default is 0.0.
    spherical_correction : bool, optional
        Whether to apply spherical correction, by default False
    costfun : Callable | None, optional
        Custom cost function, by default None
    seed : int, optional
        Random seed for reproducibility, by default 0
    verbose : bool, optional
        Print optimization progress, by default True
    time_offset : float, optional
        Offset added to segment timestamps before querying time-variant
        fields, by default 0.0
    enforce_weather_limits : bool, optional
        Reject FMS updates that violate the configured weather limits,
        by default False.
    tws_limit : float, optional
        Maximum allowed true wind speed in m/s when weather limits are enforced.
    hs_limit : float, optional
        Maximum allowed significant wave height in m when weather limits are
        enforced.

    Returns
    -------
    jnp.ndarray
        Optimized curve with shape L x 2
    """
    start = time.time()

    # Initialize solution
    if (src is not None) and (dst is not None):
        curve = random_piecewise_curve(
            src, dst, num_curves=num_curves, num_points=num_points, seed=seed
        )
    elif curve is None:
        raise ValueError("Either src and dst or curve must be provided")
    if curve.ndim == 2:
        # Add an extra dimension
        curve = curve[None, ...]
    elif curve.ndim != 3:
        raise ValueError("Input curve must be 2D (L x 2) or 3D (B x L x 2)")
    assert curve.shape[-1] == 2, "Last dimension must be 2 (X, Y)"

    # If land is provided, ensure that no points are on land at initialization
    if land is not None and penalty > 0:
        is_land = land(curve) > 0
        if is_land.any():
            # List the indices in land
            indices_in_land = jnp.argwhere(is_land).tolist()
            raise ValueError(
                "[ERROR] Initial curve has points on land at indices: "
                f"{indices_in_land} "
                "Please provide a valid curve for FMS."
            )

    # Define cost function
    if costfun is None:
        costfun = cost_function

    def _evaluate_cost(
        curve_eval: jnp.ndarray,
        *,
        travel_stw_eval: float | None = None,
        travel_time_eval: float | None = None,
        time_offset_eval: float = 0.0,
    ) -> jnp.ndarray:
        return costfun(
            vectorfield=vectorfield,
            curve=curve_eval,
            wavefield=wavefield,
            travel_stw=travel_stw_eval,
            travel_time=travel_time_eval,
            weight_l1=weight_l1,
            weight_l2=weight_l2,
            spherical_correction=spherical_correction,
            time_offset=time_offset_eval,
        )

    # Initialize lagrangians
    if travel_stw is not None:
        # Average distance between points
        d = jnp.mean(jnp.linalg.norm(curve[:, 1:] - curve[:, :-1], axis=-1))
        h = float(d / travel_stw)
        segment_time_offsets = jnp.asarray(
            time_offset + jnp.arange(curve.shape[1] - 1) * h,
            dtype=jnp.float32,
        )

        def lagrangian(
            q0: jnp.ndarray,
            q1: jnp.ndarray,
            segment_time_offset: float,
        ) -> jnp.ndarray:
            # Stack q0 and q1 to form array of shape (1, 2, 2)
            q = jnp.vstack([q0, q1])[None, ...]
            lag = _evaluate_cost(
                q,
                travel_stw_eval=travel_stw,
                time_offset_eval=segment_time_offset,
            )
            ld = jnp.sum(h * lag**2)
            # Do note: The original formula used q0, q1 to compute l1, l2 and then
            # took the average of (l1**2 + l2**2) / 2
            # We simplified that without loss of generality
            return ld

    elif travel_time is not None:
        assert travel_time > 0, "Travel time must be positive"
        h = float(travel_time / (curve.shape[1] - 1))
        segment_time_offsets = jnp.asarray(
            time_offset + jnp.arange(curve.shape[1] - 1) * h,
            dtype=jnp.float32,
        )

        def lagrangian(
            q0: jnp.ndarray,
            q1: jnp.ndarray,
            segment_time_offset: float,
        ) -> jnp.ndarray:
            # Stack q0 and q1 to form array of shape (1, 2, 2)
            q = jnp.vstack([q0, q1])[None, ...]
            lag = _evaluate_cost(
                q,
                travel_time_eval=h,
                time_offset_eval=segment_time_offset,
            )
            ld = jnp.sum(h * lag)
            # Do note: The original formula used q0, q1 to compute l1, l2 and then
            # took the average of (l1 + l2) / 2
            # We simplified that without loss of generality
            return ld

    else:
        raise ValueError("Either travel_stw or travel_time must be provided")

    d1ld = grad(lagrangian, argnums=0)
    d2ld = grad(lagrangian, argnums=1)
    d11ld = hessian(lagrangian, argnums=0)
    d22ld = hessian(lagrangian, argnums=1)

    @jit  # type: ignore[misc]
    def jacobian(
        qkm1: jnp.ndarray,
        qk: jnp.ndarray,
        qkp1: jnp.ndarray,
        left_time_offset: float,
        right_time_offset: float,
    ) -> jnp.ndarray:
        b = -d2ld(qkm1, qk, left_time_offset) - d1ld(qk, qkp1, right_time_offset)
        a = d22ld(qkm1, qk, left_time_offset) + d11ld(
            qk,
            qkp1,
            right_time_offset,
        )
        q: jnp.ndarray = jnp.linalg.solve(a, b)
        return jnp.nan_to_num(q)

    jac_vectorized = vmap(jacobian, in_axes=(0, 0, 0, 0, 0), out_axes=(0))

    @jit  # type: ignore[misc]
    def solve_equation(curve: jnp.ndarray) -> jnp.ndarray:
        curve_new = jnp.copy(curve)
        q = jac_vectorized(
            curve[:-2],
            curve[1:-1],
            curve[2:],
            segment_time_offsets[:-1],
            segment_time_offsets[1:],
        )
        return curve_new.at[1:-1].set((1 - damping) * q + curve[1:-1])

    solve_vectorized: Callable[[jnp.ndarray], jnp.ndarray] = vmap(
        solve_equation, in_axes=(0), out_axes=(0)
    )

    cost_now = _evaluate_cost(
        curve,
        travel_stw_eval=travel_stw,
        travel_time_eval=travel_time,
        time_offset_eval=time_offset,
    )
    initial_weather_violations = _weather_violation_mask(
        curve,
        windfield=windfield,
        wavefield=wavefield,
        enforce_weather_limits=enforce_weather_limits,
        tws_limit=tws_limit,
        hs_limit=hs_limit,
        travel_stw=travel_stw,
        travel_time=travel_time,
        spherical_correction=spherical_correction,
        time_offset=time_offset,
    )
    effective_cost_now = jnp.where(initial_weather_violations, jnp.inf, cost_now)
    cost_best = effective_cost_now.copy()
    curve_best = curve.copy()
    early_stop = jnp.zeros(cost_now.shape)

    # Loop iterations
    idx = 0
    while (idx < maxfevals) & (early_stop < patience).any():
        curve_old = curve.copy()
        curve = solve_vectorized(curve)
        curve = _apply_curve_constraints(curve, curve_old, land=land, penalty=penalty)
        cost_now = _evaluate_cost(
            curve,
            travel_stw_eval=travel_stw,
            travel_time_eval=travel_time,
            time_offset_eval=time_offset,
        )
        weather_violations = _weather_violation_mask(
            curve,
            windfield=windfield,
            wavefield=wavefield,
            enforce_weather_limits=enforce_weather_limits,
            tws_limit=tws_limit,
            hs_limit=hs_limit,
            travel_stw=travel_stw,
            travel_time=travel_time,
            spherical_correction=spherical_correction,
            time_offset=time_offset,
        )
        effective_cost_now = jnp.where(weather_violations, jnp.inf, cost_now)

        # Update early stopping counter.
        # Only count stagnation once a feasible (finite-cost) solution exists;
        # while cost_best is still inf the optimizer is still searching.
        has_feasible_best = jnp.isfinite(cost_best)
        early_stop += jnp.where(
            has_feasible_best & (effective_cost_now >= cost_best), 1, 0
        )
        early_stop = jnp.where(cost_best > effective_cost_now, 0, early_stop)

        # Store best solution
        improved = effective_cost_now < cost_best
        cost_best = jnp.where(improved, effective_cost_now, cost_best)
        curve_best = jnp.where(improved[:, None, None], curve, curve_best)

        idx += 1
        if verbose and (idx % 500 == 0 or idx == 1):
            print(f"FMS - Iteration {idx}, cost: {effective_cost_now.min():.4f}")

    if verbose:
        print("FMS - Number of iterations:", idx)
        print("FMS - Optimization time:", time.time() - start)
        print("FMS - Fuel cost:", cost_best.min())

    dict_fms = {
        "cost": cost_best.tolist(),
        "niter": idx,
        "comp_time": int(round(time.time() - start)),
    }

    return curve_best, dict_fms


def main(gpu: bool = True, optimize_time: bool = False) -> None:
    """
    Demonstrate usage of the optimization algorithm.

    The vector field is a superposition of four vortices.
    """
    if not gpu:
        jax.config.update("jax_platforms", "cpu")

    # Check if JAX is using the GPU
    print("JAX devices:", jax.devices())

    src = jnp.array([0, 0])
    dst = jnp.array([6, 2])

    curve, _ = optimize_fms(
        vectorfield_fourvortices,
        src=src,
        dst=dst,
        num_curves=50,
        num_points=200,
        travel_stw=None if optimize_time else 1,
        travel_time=10 if optimize_time else None,
        patience=50,
    )

    xmin, xmax = curve[..., 0].min(), curve[..., 0].max()
    ymin, ymax = curve[..., 1].min(), curve[..., 1].max()

    x: jnp.ndarray = jnp.arange(xmin, xmax, 0.5)
    y: jnp.ndarray = jnp.arange(ymin, ymax, 0.5)
    X, Y = jnp.meshgrid(x, y)
    U, V = vectorfield_fourvortices(X, Y, None)

    plt.figure()
    plt.quiver(X, Y, U, V)
    plt.plot(src[0], src[1], "o", color="blue")
    plt.plot(dst[0], dst[1], "o", color="green")
    for idx in range(curve.shape[0]):
        plt.plot(curve[idx, :, 0], curve[idx, :, 1], color="red")
    label = "time" if optimize_time else "speed"
    plt.savefig(f"output/main_fms_{label}.png")
    plt.close()


if __name__ == "__main__":
    typer.run(main)
