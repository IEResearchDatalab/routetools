import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import cma
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import scipy
import typer
from jax import jit

from routetools.cost import cost_function
from routetools.land import Land
from routetools.vectorfield import vectorfield_fourvortices


@jit  # type: ignore[misc]
def batch_bezier(t: jnp.ndarray, control: jnp.ndarray) -> jnp.ndarray:
    """
    Evaluate a batch of Bézier curves (using de Casteljau's algorithm).

    Parameters
    ----------
    t : jnp.ndarray
        Evaluation points (vector of shape K), all between 0 and 1
    control : jnp.ndarray
        Batched matrix of control points, with shape B x P x N

    Returns
    -------
    jnp.ndarray
        Batch of curves (matrix of shape B x K x N)
    """
    control = jnp.tile(control[:, :, None, :], [1, 1, len(t), 1])
    while control.shape[1] > 1:
        control = (1 - t[None, None, :, None]) * control[:, :-1, :, :] + t[
            None, None, :, None
        ] * control[:, 1:, :, :]
    return control[:, 0, ...]


@partial(jit, static_argnums=(3, 4))
def control_to_curve(
    control: jnp.ndarray,
    src: jnp.ndarray,
    dst: jnp.ndarray,
    L: int = 64,
    num_pieces: int = 1,
) -> jnp.ndarray:
    """
    Convert a batch of free parameters into a batch of Bézier curves.

    Parameters
    ----------
    control : jnp.ndarray
        A B x 2K matrix, or a 2K vector.
        The first K columns are the x positions of the Bézier
        control points, and the last K are the y positions
    L : int, optional
        Number of points evaluated in each Bézier curve, by default 64
    num_pieces : int, optional
        Number of Bézier curves, by default 1

    Returns
    -------
    jnp.ndarray
        A B x L x 2 matrix with the batch of Bézier curves,
        or a L x 2 matrix if control is 1D.
    """
    # If control is 1D, add a batch dimension
    if control.ndim == 1:
        control = control[jnp.newaxis, :]
        one_dim = True
    else:
        one_dim = False
    # Reshape the control points
    control = control.reshape(control.shape[0], -1, 2)

    # Add the fixed endpoints
    first_point = jnp.broadcast_to(src, (control.shape[0], 1, 2))
    last_point = jnp.broadcast_to(dst, (control.shape[0], 1, 2))
    control = jnp.hstack([first_point, control, last_point])

    # Initialize the result
    result: jnp.ndarray
    if num_pieces > 1:
        # Ensure that the number of control points is divisible by the number of pieces
        control_per_piece = (control.shape[1] - 1) / num_pieces
        if control_per_piece < 2:
            raise ValueError(
                "The number of control points - 1 must be at least 3 per piece. "
                f"Got {control.shape[1]} control points and {num_pieces} pieces."
            )
        elif int(control_per_piece) != control_per_piece:
            control_rec = int(control_per_piece) * num_pieces + 1
            raise ValueError(
                "The number of control points must be divisible by num_pieces. "
                f"Got {control.shape[1]} control points and {num_pieces} pieces."
                f"Consider using {control_rec} control points."
            )
        else:
            control_per_piece = int(control_per_piece)
        # Ensure the number of waypoints is divisible by the number of pieces
        waypoints_per_piece = (L - 1) / num_pieces
        if int(waypoints_per_piece) != waypoints_per_piece:
            L_rec = int(waypoints_per_piece) * num_pieces + 1
            raise ValueError(
                "The number of waypoints - 1 must be divisible by num_pieces. "
                f"Got {L} waypoints and {num_pieces} pieces. "
                f"Consider using {L_rec} waypoints."
            )
        else:
            waypoints_per_piece = int(waypoints_per_piece) + 1

        # Split the control points into pieces
        ls_pieces: list[jnp.ndarray] = []
        for i in range(num_pieces):
            start = i * control_per_piece
            end = (i + 1) * control_per_piece + 1
            piece: jnp.ndarray = batch_bezier(
                t=jnp.linspace(0, 1, waypoints_per_piece),
                control=control[:, start:end, :],
            )[:, :-1]
            # The last point of each piece is omitted to avoid duplicates
            ls_pieces.append(piece)
        # Concatenate the pieces into a single curve
        result = jnp.hstack(ls_pieces)
        # Add the destination (last) point (was omitted in the loop)
        result = jnp.hstack([result, last_point])
    else:
        result = batch_bezier(t=jnp.linspace(0, 1, L), control=control)

    if one_dim:
        return result[0]
    else:
        return result


def _single_piece_to_control(curve: jnp.ndarray, K: int = 6) -> jnp.ndarray:
    """Fit Bézier control points from a sampled piece.

    Returns an array of control points with shape (2K-4), compatible with CMA-ES.
    The start and end control points are not included in the output,
    but are taken into account for the total K.

    Parameters
    ----------
    curve : jnp.ndarray
        The sampled curve, with shape (L, 2)
    K : int, optional
        Number of free Bézier control points, accounting for source and destination
        that will not be included in the output. By default 6

    Returns
    -------
    jnp.ndarray
        The fitted control points, with shape (2K-4)
    """
    if curve.ndim != 2 or curve.shape[1] != 2:
        raise ValueError("curve must be shape (L,2)")

    x = curve[:, 0]
    y = curve[:, 1]
    # parameterization: uniform t in [0,1]
    L = curve.shape[0]
    t = jnp.linspace(0.0, 1.0, L)

    # Convert timestamps to [0, 1] if needed
    if (t[0] != 0) or (t[-1] != 1):
        t = (t - t[0]) / (t[-1] - t[0])

    rhs = jnp.column_stack([x, y])

    # Subtract the contribution of the endpoints
    degree = K - 1

    assert (
        degree >= 1
    ), "The number of control points K must be at least 2 to match endpoints."
    A = jnp.zeros([len(x), degree - 1])
    for d in range(1, degree):
        A = A.at[:, d - 1].set(
            scipy.special.comb(degree, d) * (1 - t) ** (degree - d) * t**d
        )
    rhs = rhs.at[:, 0].set(rhs[:, 0] - ((1 - t) ** degree * x[0] + t**degree * x[-1]))
    rhs = rhs.at[:, 1].set(rhs[:, 1] - ((1 - t) ** degree * y[0] + t**degree * y[-1]))
    control = jnp.linalg.lstsq(A, rhs, rcond=None)[0]
    control = jnp.vstack(
        [jnp.column_stack([x[0], y[0]]), control, jnp.column_stack([x[-1], y[-1]])]
    )

    # Take the interior control points only
    # Return as a flattened array for compatibility with CMA-ES (x0)
    return control[1:-1].flatten()


def curve_to_control(
    curve: jnp.ndarray, K: int = 6, num_pieces: int = 1
) -> jnp.ndarray:
    """Fit Bézier control points from a sampled curve.

    Returns an array of control points with shape (2K), compatible with CMA-ES.

    Parameters
    ----------
    curve : jnp.ndarray
        The sampled curve, with shape (L, 2)
    K : int, optional
        Number of free Bézier control points, accounting for source and destination
        that will not be included in the output. By default 6

    Returns
    -------
    jnp.ndarray
        The fitted control points, with shape (2K-4)
    """
    if num_pieces == 1:
        return _single_piece_to_control(curve, K=K)
    else:
        # Ensure that the number of control points is divisible by the number of pieces
        control_per_piece = (K - 1) / num_pieces
        if control_per_piece < 2:
            raise ValueError(
                "The number of control points - 1 must be at least 3 per piece. "
                f"Got {K} control points and {num_pieces} pieces."
            )
        elif int(control_per_piece) != control_per_piece:
            control_rec = int(control_per_piece) * num_pieces + 1
            raise ValueError(
                "The number of control points must be divisible by num_pieces. "
                f"Got {K} control points and {num_pieces} pieces."
                f"Consider using {control_rec} control points."
            )
        else:
            control_per_piece = int(control_per_piece) + 1
        # Distribute the number of waypoints per piece (may be uneven)
        L = curve.shape[0]
        waypoints_per_piece = L // num_pieces
        remainder = L % num_pieces
        # Fit each piece separately
        ls_control: list[jnp.ndarray] = []
        start_idx = 0
        for i in range(num_pieces):
            end_idx = start_idx + waypoints_per_piece
            if i < remainder:
                end_idx += 1
            piece = curve[start_idx:end_idx, :]
            control_piece = _single_piece_to_control(piece, K=control_per_piece + 2)
            # Reshape to (control_per_piece, 2)
            control_piece = control_piece.reshape(-1, 2)
            # Keep the first control point. But not the last (will be repeated)
            # unless it is the final piece
            if i < num_pieces - 1:
                control_piece = control_piece[:-1, :]
            ls_control.append(control_piece)
            start_idx = end_idx
        # Concatenate the control points from all pieces and flatten
        control = jnp.vstack(ls_control)
        return control[1:-1].flatten()


def _cma_evolution_strategy(
    vectorfield: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]
    ],
    src: jnp.ndarray,
    dst: jnp.ndarray,
    x0: jnp.ndarray,
    land: Land | None = None,
    wavefield: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]
    ]
    | None = None,
    penalty: float = 10,
    travel_stw: float | None = None,
    travel_time: float | None = None,
    L: int = 64,
    num_pieces: int = 1,
    popsize: int = 200,
    sigma0: float = 1,
    tolfun: float = 1e-4,
    damping: float = 1,
    maxfevals: int = 25000,
    seed: float = jnp.nan,
    weight_l1: float = 1.0,
    weight_l2: float = 0.0,
    keep_top: float = 0.0,
    spherical_correction: bool = False,
    verbose: bool = True,
    **kwargs: dict[str, Any],
) -> cma.CMAEvolutionStrategy:
    curve: jnp.ndarray
    # Initialize the optimizer
    es = cma.CMAEvolutionStrategy(
        x0,
        sigma0,
        inopts={
            "popsize": popsize,
            "tolfun": tolfun,
            "maxfevals": maxfevals,
            "seed": seed,
            "CSA_dampfac": damping,  # v positive multiplier for step-size damping
        }
        | kwargs,
    )
    # Check if the land penalization is consistent
    if land is not None:
        assert penalty is not None, "penalty must be a number"

    # Turn the percentage into a number
    num_top = int(keep_top * popsize)
    # Initialize storage for the top solutions
    top_curves: jnp.ndarray = jnp.zeros((num_top, L, 2))
    top_costs: jnp.ndarray = jnp.full((num_top,), jnp.inf)

    # Optimization loop
    while not es.stop():
        X = es.ask()  # sample len(X) candidate solutions

        # Transform controls into curves and compute costs
        curve = control_to_curve(jnp.array(X), src, dst, L=L, num_pieces=num_pieces)

        cost: jnp.ndarray = cost_function(
            vectorfield=vectorfield,
            curve=curve,
            wavefield=wavefield,
            travel_stw=travel_stw,
            travel_time=travel_time,
            weight_l1=weight_l1,
            weight_l2=weight_l2,
            spherical_correction=spherical_correction,
        )

        # Land penalization
        if land is not None and penalty > 0:
            cost += land.penalization(curve, penalty=penalty)

        # Replace the worst solutions with the best found so far
        if keep_top > 0 and es.countiter > 1:
            num_replace = min(num_top, len(X))
            idx_sorted = jnp.argsort(cost)
            # Indices of the worst solutions in the current batch
            idx_worst = idx_sorted[-num_replace:]
            # Replace them with the top solutions found so far
            # only if they are better
            for i in range(num_replace):
                if top_costs[i] < cost[idx_worst[i]]:
                    cost = cost.at[idx_worst[i]].set(top_costs[i])
                    curve = curve.at[idx_worst[i], ...].set(top_curves[i, ...])

        es.tell(X, cost.tolist())  # update the optimizer
        if verbose:
            es.disp()

        # Save the top solutions (use cost as fitness)
        if keep_top > 0:
            idx_sorted = jnp.argsort(cost)
            top_curves = curve[idx_sorted[:num_top], ...]
            top_costs = cost[idx_sorted[:num_top]]

    return es


def optimize(
    vectorfield: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]
    ],
    src: jnp.ndarray,
    dst: jnp.ndarray,
    curve0: jnp.ndarray | None = None,
    land: Land | None = None,
    wavefield: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]
    ]
    | None = None,
    penalty: float = 10,
    travel_stw: float | None = None,
    travel_time: float | None = None,
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
    keep_top: float = 0.0,
    spherical_correction: bool = False,
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
    vectorfield : callable
        A function that returns the horizontal and vertical components of the vector
    src : jnp.ndarray
        Source point (2D)
    dst : jnp.ndarray
        Destination point (2D)
    curve0 : jnp.ndarray | None, optional
        Initial curve to start the optimization from, with shape (L,2).
        If None, a straight line is used, by default None
    land : callable, optional
        A function that checks if points on a curve are on land, by default None
    wavefield : callable, optional
        A function that returns the height and direction of the wave field,
        by default None
    penalty : float, optional
        Penalty for land points, by default 10
    travel_stw : float, optional
        The boat will have this fixed speed through water (STW).
        If set, then `travel_time` must be None. By default None
    travel_time : float, optional
        The boat can regulate its STW but must complete the path in exactly this time.
        If set, then `travel_stw` must be None
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
    keep_top : float, optional
        Percentage of top solutions to keep across generations (between 0 and 1).
        By default 0.0
    seed : int, optional
        Random seed for reproducibility. By default jnp.nan
    verbose : bool, optional
        By default True

    Returns
    -------
    tuple[jnp.ndarray, float]
        The optimized curve (shape L x 2), and the fuel cost
    """
    if curve0 is None:
        # Initial solution as a straight line
        x0 = jnp.linspace(src, dst, K - 2).flatten()
    else:
        # Validate src and dst match endpoints of curve0
        if not jnp.allclose(curve0[0, :], src):
            raise ValueError(
                "The starting point of curve0 does not match src. "
                f"curve0[0,:]={curve0[0, :]}, src={src}"
            )
        if not jnp.allclose(curve0[-1, :], dst):
            raise ValueError(
                "The ending point of curve0 does not match dst. "
                f"curve0[-1,:]={curve0[-1, :]}, dst={dst}"
            )
        # Validate the given curve does not cross land if a land function is provided
        if land is not None and land(curve0).any():
            raise ValueError("[ERROR] The provided initial curve0 crosses land.")
        # Initial solution from provided curve
        x0 = curve_to_control(curve0, K=K, num_pieces=num_pieces)
        # Validate that, after conversion, it still does not cross land
        curve_check = control_to_curve(x0, src, dst, L=L, num_pieces=num_pieces)
        if land is not None and land(curve_check).any():
            raise ValueError(
                "[ERROR] The provided initial curve0 crosses land "
                "after conversion to control points."
            )

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
        wavefield=wavefield,
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
        weight_l1=weight_l1,
        weight_l2=weight_l2,
        seed=seed,
        keep_top=keep_top,
        spherical_correction=spherical_correction,
        verbose=verbose,
    )
    if verbose:
        print("Optimization time:", time.time() - start)
        print("Fuel cost:", es.best.f)

    curve_best = control_to_curve(
        jnp.asarray(es.best.x), src, dst, L=L, num_pieces=num_pieces
    )

    dict_cmaes = {
        "cost": es.best.f,
        "niter": es.countiter,
        "comp_time": int(round(time.time() - start)),
    }
    return curve_best, dict_cmaes


def optimize_with_increasing_penalization(
    vectorfield: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]
    ],
    src: jnp.ndarray,
    dst: jnp.ndarray,
    land: Land,
    wavefield: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]
    ]
    | None = None,
    penalty_init: float = 0,
    penalty_increment: float = 10,
    maxiter: int = 10,
    travel_stw: float | None = None,
    travel_time: float | None = None,
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
    spherical_correction: bool = False,
    seed: float = jnp.nan,
    verbose: bool = True,
) -> tuple[list[jnp.ndarray], list[float]]:
    """
    Solve the vessel routing problem for a given vector field.

    Two modes are supported:
        - Fixed speed-through-water. Optimize the vessel heading
        - Fixed total travel time. Optimize heading and speed-through-water

    Algorithm: parameterize the space of solutions with a Bézier curve,
    and optimize the control points using the CMA-ES optimization method.

    Parameters
    ----------
    vectorfield : callable
        A function that returns the horizontal and vertical components of the vector
    src : jnp.ndarray
        Source point (2D)
    dst : jnp.ndarray
        Destination point (2D)
    land : callable, optional
        A function that checks if points on a curve are on land
    wavefield : callable, optional
        A function that returns the height and direction of the wave field,
        by default None
    penalty_init : float, optional
        Initial penalty for land points, by default 0
    penalty_increment : float, optional
        Increment in the penalty for land points. By default 10
    maxiter : int, optional
        Maximum number of iterations. By default 10
    travel_stw : float, optional
        The boat will have this fixed speed through water (STW).
        If set, then `travel_time` must be None. By default None
    travel_time : float, optional
        The boat can regulate its STW but must complete the path in exactly this time.
        If set, then `travel_stw` must be None
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
    tuple[list[jnp.ndarray], list[float]]
        The list of optimized curves (each with shape L x 2), and the list of fuel costs
    """
    # Initial solution as a straight line
    x0 = jnp.linspace(src, dst, K - 2).flatten()
    # Initial standard deviation to sample new solutions
    # One sigma is half the distance between src and dst
    sigma0 = float(jnp.linalg.norm(dst - src) * sigma0 / 2)

    # Initializations
    penalty = penalty_init
    is_land = True
    niter = 1
    ls_curve = []
    ls_cost = []

    start = time.time()
    while is_land and (niter < maxiter):
        es = _cma_evolution_strategy(
            vectorfield=vectorfield,
            src=src,
            dst=dst,
            x0=x0,
            land=land,
            wavefield=wavefield,
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
            weight_l1=weight_l1,
            weight_l2=weight_l2,
            spherical_correction=spherical_correction,
            seed=seed,
            verbose=verbose,
        )
        if verbose:
            print("Optimization time:", time.time() - start)
            print("Fuel cost:", es.best.f)

        curve: jnp.ndarray = control_to_curve(
            es.best.x, src, dst, L=L, num_pieces=num_pieces
        )
        # sigma0 = es.sigma0
        if land(curve).any():
            penalty += penalty_increment
            x0 = es.best.x
        else:
            is_land = False

        niter += 1
        ls_curve.append(curve)
        ls_cost.append(es.best.f)

    return ls_curve, ls_cost


def main(gpu: bool = True, optimize_time: bool = False) -> None:
    """
    Demonstrate usage of the optimization algorithm.

    The vector field is a superposition of four vortices.
    """
    if not gpu:
        jax.config.update("jax_platforms", "cpu")

    # Check if JAX is using the GPU
    print("JAX devices:", jax.devices())

    # Create the output folder if needed
    output_folder = Path("output")
    output_folder.mkdir(exist_ok=True)

    src = jnp.array([0, 0])
    dst = jnp.array([6, 2])

    curve, _ = optimize(
        vectorfield_fourvortices,
        src=src,
        dst=dst,
        travel_stw=None if optimize_time else 1,
        travel_time=10 if optimize_time else None,
        K=13,
        L=64,
        num_pieces=3,
        popsize=500,
        sigma0=3,
        tolfun=1e-6,
    )

    xmin, xmax = curve[:, 0].min(), curve[:, 0].max()
    ymin, ymax = curve[:, 1].min(), curve[:, 1].max()

    x: jnp.ndarray = jnp.arange(xmin, xmax, 0.5)
    y: jnp.ndarray = jnp.arange(ymin, ymax, 0.5)
    X, Y = jnp.meshgrid(x, y)
    t = jnp.zeros_like(X)
    U, V = vectorfield_fourvortices(X, Y, t)

    plt.figure()
    plt.quiver(X, Y, U, V)
    plt.plot(curve[:, 0], curve[:, 1], color="red")
    plt.plot(src[0], src[1], "o", color="blue")
    plt.plot(dst[0], dst[1], "o", color="green")
    label = "time" if optimize_time else "speed"
    plt.savefig(output_folder / f"main_cmaes_{label}.png")
    plt.close()


if __name__ == "__main__":
    typer.run(main)
