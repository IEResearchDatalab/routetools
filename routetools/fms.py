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
    num_curves: int = 10,
    num_points: int = 200,
    travel_stw: float | None = None,
    travel_time: float | None = None,
    tolfun: float = 1e-4,
    damping: float = 0.9,
    maxfevals: int = 5000,
    weight_l1: float = 1.0,
    weight_l2: float = 0.0,
    seed: int = 0,
    verbose: bool = True,
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
    tolfun : float, optional
        Tolerance for the cost reduction between epochs, by default 1e-4
    damping : float, optional
        Damping factor, by default 0.9
    maxfevals : int, optional
        Maximum number of iterations, by default 5000
    weight_l1 : float, optional
        Weight for the L1 norm in the combined cost. Default is 1.0.
    weight_l2 : float, optional
        Weight for the L2 norm in the combined cost. Default is 0.0.
    seed : int, optional
        Random seed for reproducibility, by default 0
    verbose : bool, optional
        Print optimization progress, by default True

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

    # Initialize lagrangians
    if travel_stw is not None:
        # Average distance between points
        d = jnp.mean(jnp.linalg.norm(curve[:, 1:] - curve[:, :-1], axis=-1))
        h = float(d / travel_stw)

        def lagrangian(q0: jnp.ndarray, q1: jnp.ndarray) -> jnp.ndarray:
            # Stack q0 and q1 to form array of shape (1, 2, 2)
            q = jnp.vstack([q0, q1])[None, ...]
            lag = cost_function(vectorfield=vectorfield, curve=q, travel_stw=travel_stw, weight_l1=weight_l1, weight_l2=weight_l2)
            ld = jnp.sum(h * lag**2)
            # Do note: The original formula used q0, q1 to compute l1, l2 and then
            # took the average of (l1**2 + l2**2) / 2
            # We simplified that without loss of generality
            return ld

    elif travel_time is not None:
        assert travel_time > 0, "Travel time must be positive"
        h = float(travel_time / curve.shape[1])

        def lagrangian(q0: jnp.ndarray, q1: jnp.ndarray) -> jnp.ndarray:
            # Stack q0 and q1 to form array of shape (1, 2, 2)
            q = jnp.vstack([q0, q1])[None, ...]
            lag = cost_function(vectorfield=vectorfield, curve=q, travel_time=h, weight_l1=weight_l1, weight_l2=weight_l2)
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
    def jacobian(qkm1: jnp.ndarray, qk: jnp.ndarray, qkp1: jnp.ndarray) -> jnp.ndarray:
        b = -d2ld(qkm1, qk) - d1ld(qk, qkp1)
        a = d22ld(qkm1, qk) + d11ld(qk, qkp1)
        q: jnp.ndarray = jnp.linalg.solve(a, b)
        return jnp.nan_to_num(q)

    jac_vectorized = vmap(jacobian, in_axes=(0, 0, 0), out_axes=(0))

    @jit  # type: ignore[misc]
    def solve_equation(curve: jnp.ndarray) -> jnp.ndarray:
        curve_new = jnp.copy(curve)
        q = jac_vectorized(curve[:-2], curve[1:-1], curve[2:])
        return curve_new.at[1:-1].set((1 - damping) * q + curve[1:-1])

    solve_vectorized: Callable[[jnp.ndarray], jnp.ndarray] = vmap(
        solve_equation, in_axes=(0), out_axes=(0)
    )

    cost_now = cost_function(
        vectorfield=vectorfield,
        curve=curve,
        travel_stw=travel_stw,
        travel_time=travel_time,
        weight_l1=weight_l1,
        weight_l2=weight_l2,
    )
    delta = jnp.array([jnp.inf])

    # Loop iterations
    idx = 0
    while (idx < maxfevals) & (delta >= tolfun).any():
        cost_old = cost_now
        curve_old = curve.copy()
        curve = solve_vectorized(curve)
        # Replace points on land with previous iteration
        if land is not None:
            is_land = land(curve) > 0
            curve = jnp.where(is_land[..., None], curve_old, curve)
        cost_now = cost_function(
            vectorfield=vectorfield,
            curve=curve,
            travel_stw=travel_stw,
            travel_time=travel_time,
            weight_l1=weight_l1,
            weight_l2=weight_l2,
        )
        delta = 1 - cost_now / cost_old
        idx += 1
        if verbose and (idx % 500 == 0 or idx == 1):
            print(f"FMS - Iteration {idx}, cost: {cost_now.min():.4f}")

    if verbose:
        print("FMS - Number of iterations:", idx)
        print("FMS - Optimization time:", time.time() - start)
        print("FMS - Fuel cost:", cost_now.min())

    dict_fms = {
        "cost": cost_now.tolist(),
        "niter": idx,
        "comp_time": int(round(time.time() - start)),
    }

    return curve, dict_fms


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
        tolfun=1e-6,
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
