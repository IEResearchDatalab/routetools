from __future__ import annotations

from collections.abc import Callable
from functools import partial

import jax.numpy as jnp
from jax import jit, lax

from routetools._cost.haversine import (
    haversine_distance_from_curve as haversine_distance_from_curve,
)
from routetools._cost.haversine import haversine_meters_components
from routetools._cost.waves import wave_adjusted_speed
from routetools.land import Land, move_curve_away_from_land


def angle_wrt_true_north(dx: jnp.ndarray, dy: jnp.ndarray) -> jnp.ndarray:
    """Compute the angle with respect to true North in degrees.

    Parameters
    ----------
    dx : jnp.ndarray
        Displacement in the x direction.
    dy : jnp.ndarray
        Displacement in the y direction.

    Returns
    -------
    jnp.ndarray
        Angle with respect to true North in degrees.
    """
    # The arctan2 computes angles between -180 and 180 degrees
    angle_wrt_north = jnp.degrees(jnp.arctan2(dy, dx))
    return jnp.mod(angle_wrt_north, 360)


@partial(
    jit,
    static_argnames=(
        "vectorfield",
        "wavefield",
        "travel_stw",
        "travel_time",
        "weight_l1",
        "weight_l2",
        "spherical_correction",
    ),
)
def cost_function(
    vectorfield: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]
    ],
    curve: jnp.ndarray,
    wavefield: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]
    ]
    | None = None,
    travel_stw: float | None = None,
    travel_time: float | None = None,
    weight_l1: float = 1.0,
    weight_l2: float = 0.0,
    spherical_correction: bool = False,
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
        Coordinates are (lon, lat) or (x, y).
    travel_stw : float, optional
        The boat will have this fixed speed through water (STW). If applying the
        spherical correction, this speed is in meters per second.
    travel_time : float, optional
        The boat can regulate its STW but must complete the path in exactly this time.
        If applying the spherical correction, this time is in seconds.
    weight_l1 : float, optional
        Weight for the L1 norm in the combined cost. Default is 1.0.
    weight_l2 : float, optional
        Weight for the L2 norm in the combined cost. Default is 0.0.
    spherical_correction : bool, optional
        Whether to apply spherical correction to distances. Default is False.

    Returns
    -------
    jnp.ndarray
        A batch of scalars (vector of shape B) representing the cost for each path.
    """
    is_time_variant: bool = getattr(vectorfield, "is_time_variant", False)
    # Choose which cost function to use
    if (travel_stw is not None) and is_time_variant:
        cost = cost_function_constant_speed_time_variant(
            vectorfield,
            curve,
            travel_stw,
            wavefield=wavefield,
            spherical_correction=spherical_correction,
        )
    elif (travel_stw is not None) and (not is_time_variant):
        cost = cost_function_constant_speed_time_invariant(
            vectorfield,
            curve,
            travel_stw,
            wavefield=wavefield,
            spherical_correction=spherical_correction,
        )
    elif (travel_time is not None) and is_time_variant:
        # Not supported
        raise NotImplementedError(
            "Time-variant cost function with fixed travel time is not implemented."
        )
    elif (travel_time is not None) and (not is_time_variant):
        cost = cost_function_constant_cost_time_invariant(
            vectorfield,
            curve,
            travel_time,
            wavefield=wavefield,
            spherical_correction=spherical_correction,
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


@partial(
    jit,
    static_argnames=("vectorfield", "travel_stw", "wavefield", "spherical_correction"),
)
def cost_function_constant_speed_time_invariant(
    vectorfield: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]
    ],
    curve: jnp.ndarray,
    travel_stw: float,
    wavefield: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]
    ]
    | None = None,
    spherical_correction: bool = False,
) -> jnp.ndarray:
    """
    Compute the travel time of a batch of paths navigating over a vector field.

    Parameters
    ----------
    vectorfield : Callable
        A function that returns the horizontal and vertical components of the vector
    curve : jnp.ndarray
        A batch of trajectories (an array of shape B x L x 2).
        Coordinates are (lon, lat) or (x, y)
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
    curvet = jnp.zeros_like(curvex)

    uinterp, vinterp = vectorfield(curvex, curvey, curvet)

    # Distances between points in X and Y
    if spherical_correction:
        dx, dy = haversine_meters_components(
            curve[:, :-1, 1],
            curve[:, :-1, 0],
            curve[:, 1:, 1],
            curve[:, 1:, 0],
        )
    else:
        dx = jnp.diff(curve[:, :, 0], axis=1)
        dy = jnp.diff(curve[:, :, 1], axis=1)
    # Power of the distance (segment lengths)
    d2 = jnp.power(dx, 2) + jnp.power(dy, 2)
    if wavefield is not None:
        # Ship's angle with respect to true North in degrees
        angle = angle_wrt_true_north(dx, dy)
        wave_height, wave_direction = wavefield(curvex, curvey, curvet)
        # TODO: Problem with dimensions of time
        travel_stw_mod = wave_adjusted_speed(
            angle=angle,
            wave_height=wave_height,
            wave_angle=wave_direction,
            vel_ship=travel_stw,
        )
    else:
        travel_stw_mod = travel_stw
    # Power of the speed through water
    v2 = travel_stw_mod**2
    # Power of the current speed
    w2 = uinterp**2 + vinterp**2
    dw = dx * uinterp + dy * vinterp
    # Cost is the time to travel the segment
    dt = jnp.sqrt(d2 / (v2 - w2) + dw**2 / (v2 - w2) ** 2) - dw / (v2 - w2)
    # Current > speed -> infeasible path
    # dt = lax.stop_gradient(jnp.where(v2 <= w2, 1e10, 0.0))
    return dt


@partial(
    jit,
    static_argnames=("vectorfield", "travel_stw", "wavefield", "spherical_correction"),
)
def cost_function_constant_speed_time_variant(
    vectorfield: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]
    ],
    curve: jnp.ndarray,
    travel_stw: float,
    wavefield: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]
    ]
    | None = None,
    spherical_correction: bool = False,
) -> jnp.ndarray:
    """
    Compute the travel time of a batch of paths navigating over a vector field.

    Parameters
    ----------
    vectorfield : Callable
        A function that returns the horizontal and vertical components of the vector
    curve : jnp.ndarray
        A batch of trajectories (an array of shape B x L x 2).
        Coordinates are (lon, lat) or (x, y).
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
    if spherical_correction:
        dx, dy = haversine_meters_components(
            curve[:, :-1, 1],
            curve[:, :-1, 0],
            curve[:, 1:, 1],
            curve[:, 1:, 0],
        )
    else:
        dx = jnp.diff(curve[:, :, 0], axis=1)
        dy = jnp.diff(curve[:, :, 1], axis=1)
    # Power of the distance (segment lengths)
    d2 = jnp.power(dx, 2) + jnp.power(dy, 2)

    def step(
        t: jnp.ndarray,
        inputs: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        x, y, dx_step, dy_step, d2_step = inputs
        # When sailing from i-1 to i, we interpolate the vector field at the midpoint
        uinterp, vinterp = vectorfield(x, y, t)
        # Apply reduction of speed due to waves if wavefield is provided
        if wavefield is not None:
            # Ship's angle with respect to true North in degrees
            angle = angle_wrt_true_north(dx_step, dy_step)
            wave_height, wave_direction = wavefield(x, y, t)
            travel_stw_mod = wave_adjusted_speed(
                angle=angle,
                wave_height=wave_height,
                wave_angle=wave_direction,
                vel_ship=travel_stw,
            )
        else:
            travel_stw_mod = travel_stw
        # Power of the current speed
        w2 = uinterp**2 + vinterp**2
        dw = dx_step * uinterp + dy_step * vinterp
        # Power of the speed through water
        v2 = travel_stw_mod**2
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
    _, dt_array = lax.scan(step, t_init, inputs)
    # dt_array has shape (L-1, B), we transpose it to (B, L-1)
    return dt_array.T


@partial(
    jit,
    static_argnames=("vectorfield", "travel_time", "wavefield", "spherical_correction"),
)
def cost_function_constant_cost_time_invariant(
    vectorfield: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]
    ],
    curve: jnp.ndarray,
    travel_time: float,
    wavefield: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]
    ]
    | None = None,
    spherical_correction: bool = False,
) -> jnp.ndarray:
    """
    Compute the fuel consumption of a batch of paths navigating over a vector field.

    Parameters
    ----------
    vectorfield : Callable
        A function that returns the horizontal and vertical components of the vector
    curve : jnp.ndarray
        A batch of trajectories (an array of shape B x L x 2).
        Coordinates are (lon, lat) or (x, y).
    travel_time : float
        The boat can regulate its STW but must complete the path in exactly this time.

    Returns
    -------
    jnp.ndarray
        A batch of scalars (B x L-1)
    """
    # TODO: Implement wavefield effects in this cost function

    # Interpolate the vector field at the midpoints
    curvex = (curve[:, :-1, 0] + curve[:, 1:, 0]) / 2
    curvey = (curve[:, :-1, 1] + curve[:, 1:, 1]) / 2
    uinterp, vinterp = vectorfield(curvex, curvey, jnp.array([0.0]))

    # Distances between points
    if spherical_correction:
        dx, dy = haversine_meters_components(
            curve[:, :-1, 1],
            curve[:, :-1, 0],
            curve[:, 1:, 1],
            curve[:, 1:, 0],
        )
    else:
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


def interpolate_to_constant_cost(
    curve: jnp.ndarray,
    vectorfield: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]
    ],
    travel_stw: float,
    cost_per_segment: float,
    wavefield: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]
    ]
    | None = None,
    spherical_correction: bool = False,
    oversampling_factor: int = 1000,
    land: Land | None = None,
) -> jnp.ndarray:
    """
    Reinterpolate a curve so that each segment has approximately the same cost.

    Parameters
    ----------
    curve : jnp.ndarray
        A single trajectory (an array of shape L x 2).
        Coordinates are (lon, lat) or (x, y).
    vectorfield : Callable
        A function that returns the horizontal and vertical components of the vector
    travel_stw : float
        The boat will have this fixed speed through water (STW)
    cost_per_segment : float
        Desired cost per segment after reinterpolation.
    wavefield : Callable, optional
        A function that returns the wave height and direction.
    spherical_correction : bool, optional
        Whether to apply spherical correction to distances. Default is False.
    oversampling_factor : int, optional
        Factor by which to oversample the original curve for reinterpolation.
        Default is 1000.
    land : Land, optional
        An optional Land object to ensure the new curve does not go on land.

    Returns
    -------
    jnp.ndarray
        A reinterpolated trajectory (an array of shape L' x 2).
    """
    # First, compute the cost between the original segments
    original_cost = cost_function_constant_speed_time_variant(
        vectorfield,
        curve[jnp.newaxis, :, :],
        travel_stw=travel_stw,
        wavefield=wavefield,
        spherical_correction=spherical_correction,
    )[0]  # Shape (L_orig - 1)
    # We are going to reinterpolate the curve, weighting by the cost
    # The segments with higher cost should have more points
    weight_per_segment = original_cost / jnp.sum(original_cost)
    # Scale up for oversampling
    points_per_segment = jnp.ceil(weight_per_segment * oversampling_factor).astype(int)

    # Build a fine curve by oversampling each segment
    fine_curve: list[jnp.ndarray] = []
    for i in range(curve.shape[0] - 1):
        n_points = points_per_segment[i] + 1  # +1 to include endpoint
        segment = jnp.linspace(curve[i], curve[i + 1], n_points)
        fine_curve.append(segment)
    fine_curve = jnp.concatenate(fine_curve, axis=0)  # Shape (L_fine, 2)

    # Compute the cost of the interpolated fine curve
    cost = cost_function_constant_speed_time_variant(
        vectorfield,
        fine_curve[jnp.newaxis, :, :],
        travel_stw=travel_stw,
        wavefield=wavefield,
        spherical_correction=spherical_correction,
    )[0]  # Shape (L_fine - 1,)
    # Compute the cumulative cost along the path
    cumulative_cost = jnp.cumsum(cost)
    cumulative_cost = jnp.concatenate([jnp.array([0.0]), cumulative_cost])
    # Compute the desired cumulative cost at each segment
    cumulative_cost_desired = jnp.arange(
        0, cumulative_cost[-1] + cost_per_segment, cost_per_segment
    )
    # Find the indices in the fine curve that are closest to
    # the desired cumulative costs
    new_indices = jnp.searchsorted(cumulative_cost, cumulative_cost_desired)
    new_indices = jnp.clip(new_indices, 0, fine_curve.shape[0] - 1)
    # Ensure the last point is included
    new_indices = jnp.concatenate([new_indices, jnp.array([fine_curve.shape[0] - 1])])
    # Ensure the indices are unique and sorted
    new_indices = jnp.sort(jnp.unique(new_indices))

    # Build the new curve
    new_curve = fine_curve[new_indices, :]

    # If land is provided, ensure the new curve does not go on land
    if land is not None:
        new_curve = move_curve_away_from_land(new_curve, land)

    return new_curve
