from collections.abc import Callable
from functools import partial

import jax.numpy as jnp
from jax import jit, lax


@partial(
    jit,
    static_argnames=(
        "vectorfield",
        "travel_stw",
        "travel_time",
        "weight_l1",
        "weight_l2",
    ),
)
def cost_function(
    vectorfield: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]
    ],
    curve: jnp.ndarray,
    travel_stw: float | None = None,
    travel_time: float | None = None,
    weight_l1: float = 1.0,
    weight_l2: float = 0.0,
) -> jnp.ndarray:
    """
    Compute the cost of a batch of paths navigating over a vector field.

    This function selects the appropriate cost function based on the
    parameters provided. It can handle time-invariant and time-variant vector
    fields, as well as fixed speed through water (STW) or fixed travel time.

    Parameters
    ----------
    vectorfield : Callable
        A function that returns the horizontal and vertical components of the
        vector field.
    curve : jnp.ndarray
        A batch of trajectories (an array of shape B x L x 2).
    travel_stw : float, optional
        The boat will have this fixed speed through water (STW).
    travel_time : float, optional
        The boat can regulate its STW but must complete the path in exactly this time.
    weight_l1 : float, optional
        Weight for the L1 norm in the combined cost. Default is 1.0.
    weight_l2 : float, optional
        Weight for the L2 norm in the combined cost. Default is 0.0.

    Returns
    -------
    jnp.ndarray
        A batch of scalars (vector of shape B) representing the cost for each path.
    """
    is_time_variant: bool = getattr(vectorfield, "is_time_variant", False)
    # Choose which cost function to use
    if (travel_stw is not None) and is_time_variant:
        cost = cost_function_constant_speed_time_variant(vectorfield, curve, travel_stw)
    elif (travel_stw is not None) and (not is_time_variant):
        cost = cost_function_constant_speed_time_invariant(
            vectorfield, curve, travel_stw
        )
    elif (travel_time is not None) and is_time_variant:
        # Not supported
        raise NotImplementedError(
            "Time-variant cost function with fixed travel time is not implemented."
        )
    elif (travel_time is not None) and (not is_time_variant):
        cost = cost_function_constant_cost_time_invariant(
            vectorfield, curve, travel_time
        )
    else:
        # Arguments missing
        raise ValueError("Either travel_stw or travel_time must be provided.")

    # Turn any possible inf values into large finite numbers
    cost = jnp.where(jnp.isinf(cost), jnp.nanmax(cost, initial=1e10) * 10, cost)

    # Compute L1 and L2 norms
    l1 = jnp.sum(jnp.abs(cost), axis=1)
    l2 = jnp.sqrt(jnp.sum(cost**2, axis=1))
    return weight_l1 * l1 + weight_l2 * l2


@partial(jit, static_argnames=("vectorfield", "travel_stw"))
def cost_function_constant_speed_time_invariant(
    vectorfield: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]
    ],
    curve: jnp.ndarray,
    travel_stw: float,
) -> jnp.ndarray:
    """
    Compute the travel time of a batch of paths navigating over a vector field.

    Parameters
    ----------
    vectorfield : Callable
        A function that returns the horizontal and vertical components of the vector
    curve : jnp.ndarray
        A batch of trajectories (an array of shape B x L x 2)
    travel_stw : float
        The boat will have this fixed speed through water (STW)

    Returns
    -------
    jnp.ndarray
        A batch of scalars (B x L-1)
    """
    # Interpolate the vector field at the midpoints
    curvex = (curve[:, :-1, 0] + curve[:, 1:, 0]) / 2
    curvey = (curve[:, :-1, 1] + curve[:, 1:, 1]) / 2
    uinterp, vinterp = vectorfield(curvex, curvey, jnp.array([0.0]))

    # Distances between points in X and Y
    dx = jnp.diff(curve[:, :, 0], axis=1)
    dy = jnp.diff(curve[:, :, 1], axis=1)
    # Power of the distance (segment lengths)
    d2 = jnp.power(dx, 2) + jnp.power(dy, 2)
    # Power of the speed through water
    v2 = travel_stw**2
    # Power of the current speed
    w2 = uinterp**2 + vinterp**2
    dw = dx * uinterp + dy * vinterp
    # Cost is the time to travel the segment
    dt = jnp.sqrt(d2 / (v2 - w2) + dw**2 / (v2 - w2) ** 2) - dw / (v2 - w2)
    # Current > speed -> infeasible path
    # dt = lax.stop_gradient(jnp.where(v2 <= w2, 1e10, 0.0))
    # t_total = jnp.sum(dt, axis=1)
    return dt


@partial(jit, static_argnames=("vectorfield", "travel_stw"))
def cost_function_constant_speed_time_variant(
    vectorfield: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]
    ],
    curve: jnp.ndarray,
    travel_stw: float,
) -> jnp.ndarray:
    """
    Compute the travel time of a batch of paths navigating over a vector field.

    Parameters
    ----------
    vectorfield : Callable
        A function that returns the horizontal and vertical components of the vector
    curve : jnp.ndarray
        A batch of trajectories (an array of shape B x L x 2)
    travel_stw : float
        The boat will have this fixed speed through water (STW)

    Returns
    -------
    jnp.ndarray
        A batch of scalars (B x L-1)
    """
    # We will interpolate the vector field at the midpoints
    curvex = (curve[:, :-1, 0] + curve[:, 1:, 0]) / 2
    curvey = (curve[:, :-1, 1] + curve[:, 1:, 1]) / 2

    # Distances between points in X and Y
    dx = jnp.diff(curve[:, :, 0], axis=1)
    dy = jnp.diff(curve[:, :, 1], axis=1)
    # Power of the distance (segment lengths)
    d2 = jnp.power(dx, 2) + jnp.power(dy, 2)
    # Power of the speed through water
    v2 = travel_stw**2

    def step(
        t: jnp.ndarray,
        inputs: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        x, y, dx_step, dy_step, d2_step = inputs
        # When sailing from i-1 to i, we interpolate the vector field at the midpoint
        uinterp, vinterp = vectorfield(x, y, t)
        # Power of the current speed
        w2 = uinterp**2 + vinterp**2
        dw = dx_step * uinterp + dy_step * vinterp
        # Cost is the time to travel the segment
        dt = jnp.sqrt(d2_step / (v2 - w2) + dw**2 / (v2 - w2) ** 2) - dw / (v2 - w2)
        # Current > speed -> infeasible path
        dt = jnp.where(v2 <= w2, 1e10, dt)
        # Update the times
        return t + dt, dt

    # Initialize inputs for the JAX-native looping construct
    inputs = (curvex.T, curvey.T, dx.T, dy.T, d2.T)
    t_init = jnp.zeros(curve.shape[0])

    # Use lax to implement the for loop
    t_final, dt_array = lax.scan(step, t_init, inputs)
    return dt_array


@partial(jit, static_argnames=("vectorfield", "travel_time"))
def cost_function_constant_cost_time_invariant(
    vectorfield: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]
    ],
    curve: jnp.ndarray,
    travel_time: float,
) -> jnp.ndarray:
    """
    Compute the fuel consumption of a batch of paths navigating over a vector field.

    Parameters
    ----------
    vectorfield : Callable
        A function that returns the horizontal and vertical components of the vector
    curve : jnp.ndarray
        A batch of trajectories (an array of shape B x L x 2)
    travel_time : float
        The boat can regulate its STW but must complete the path in exactly this time.

    Returns
    -------
    jnp.ndarray
        A batch of scalars (B x L-1)
    """
    # Interpolate the vector field at the midpoints
    curvex = (curve[:, :-1, 0] + curve[:, 1:, 0]) / 2
    curvey = (curve[:, :-1, 1] + curve[:, 1:, 1]) / 2
    uinterp, vinterp = vectorfield(curvex, curvey, jnp.array([0.0]))

    # Distances between points
    dx = jnp.diff(curve[:, :, 0], axis=1)
    dy = jnp.diff(curve[:, :, 1], axis=1)
    # Times between points
    dt = travel_time / (curve.shape[1] - 1)
    # We compute the speed over ground from the displacement
    dxdt = dx / dt
    dydt = dy / dt

    # We must navigate the path in a fixed time
    cost = ((dxdt - uinterp) ** 2 + (dydt - vinterp) ** 2) / 2
    return cost * dt
