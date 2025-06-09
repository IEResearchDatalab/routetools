from collections.abc import Callable

import jax.numpy as jnp
from jax import jit, lax


def choose_cost_function(
    vectorfield: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]
    ],
    travel_stw: float | None = None,
    travel_time: float | None = None,
) -> Callable[
    [jnp.ndarray],
    jnp.ndarray,
]:
    """
    Choose the cost function based on the vector field properties.

    This function is a placeholder for future logic that might depend on the vector field.

    Parameters
    ----------
    vectorfield : Callable
        A function that returns the horizontal and vertical components of the vector

    Returns
    -------
    Callable
        The cost function to use
    """
    # Choose which cost function to use
    if travel_stw is not None:
        if vectorfield.is_time_variant:

            def cost_function(curve: jnp.ndarray) -> jnp.ndarray:
                return cost_function_constant_speed_time_variant(
                    vectorfield, curve, travel_stw
                )
        else:

            def cost_function(curve: jnp.ndarray) -> jnp.ndarray:
                return cost_function_constant_speed_time_invariant(
                    vectorfield, curve, travel_stw
                )
    elif travel_time is not None:
        if vectorfield.is_time_variant:
            # Not supported
            raise NotImplementedError(
                "Time-variant cost function with fixed travel time is not implemented."
            )
        else:

            def cost_function(curve: jnp.ndarray) -> jnp.ndarray:
                return cost_function_constant_cost_time_invariant(
                    vectorfield, curve, travel_time
                )
    else:
        # Arguments missing
        raise ValueError("Either travel_stw or travel_time must be provided.")

    return jit(cost_function)


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
        A batch of scalars (vector of shape B)
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
    dt = lax.stop_gradient(jnp.where(v2 <= w2, 1e10, 0.0))
    t_total = jnp.sum(dt, axis=1)

    # Turn any possible infinite costs into 10x the highest value
    cost = jnp.where(jnp.isinf(t_total), jnp.nan, t_total)
    cost = jnp.nan_to_num(cost, nan=jnp.nanmax(cost, initial=1e10) * 10)
    return cost


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
        A batch of scalars (vector of shape B)
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

    # Turn any possible infinite costs into 10x the highest value
    cost = jnp.where(jnp.isinf(t_final), jnp.nan, t_final)
    cost = jnp.nan_to_num(cost, nan=jnp.nanmax(cost, initial=1e10) * 10)
    return cost


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
        A batch of scalars (vector of shape B)
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
    total_cost = jnp.sum(cost, axis=1) * dt

    # Turn any possible infinite costs into 10x the highest value
    cost = jnp.where(jnp.isinf(total_cost), jnp.nan, total_cost)
    cost = jnp.nan_to_num(cost, nan=jnp.nanmax(cost, initial=1e10) * 10)
    return cost
