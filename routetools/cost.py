from __future__ import annotations

from collections.abc import Callable
from functools import partial

import jax.numpy as jnp
import numpy as np
from jax import jit, lax

from routetools._cost.haversine import (
    curve_distance_nm,
    haversine_meters_components,
)
from routetools._cost.haversine import (
    haversine_distance_from_curve as haversine_distance_from_curve,
)
from routetools._cost.waves import wave_adjusted_speed
from routetools.land import Land, move_curve_away_from_land

try:
    from routetools.performance import predict_power_batch, predict_power_jax
except ModuleNotFoundError:
    predict_power_batch = None
    predict_power_jax = None


def segment_bearings_deg(curve: jnp.ndarray) -> np.ndarray:
    """Compute true-north bearing (degrees) for each route segment.

    Parameters
    ----------
    curve : jnp.ndarray
        Shape ``(L, 2)`` with ``(lon, lat)`` in degrees.

    Returns
    -------
    np.ndarray
        Shape ``(L-1,)`` bearing in degrees on ``[0, 360)``.
    """
    lon = np.asarray(curve[:, 0], dtype=np.float64)
    lat = np.asarray(curve[:, 1], dtype=np.float64)

    dlon = np.diff(lon)
    dlat = np.diff(lat)
    lat_mid = np.radians((lat[:-1] + lat[1:]) / 2)

    dx = dlon * np.cos(lat_mid)
    dy = dlat
    return np.degrees(np.arctan2(dx, dy)) % 360.0


def evaluate_route_energy(
    curve: jnp.ndarray,
    passage_hours: float,
    wps: bool,
    windfield: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]
    ]
    | None = None,
    wavefield: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]
    ]
    | None = None,
    departure_offset_h: float = 0.0,
) -> tuple[float, float, float]:
    """Evaluate total route energy in MWh with optional wind and wave fields.

    Parameters
    ----------
    curve : jnp.ndarray
        Shape ``(L, 2)`` with ``(lon, lat)`` in degrees.
    passage_hours : float
        Total passage time in hours.
    wps : bool
        Whether wingsails are deployed.
    windfield : Callable, optional
        ``(lon, lat, t) -> (u10, v10)`` in m/s.
    wavefield : Callable, optional
        ``(lon, lat, t) -> (hs, mwd)`` where ``hs`` is in metres and ``mwd``
        is degrees from North.
    departure_offset_h : float
        Hours from the field time origin to departure.

    Returns
    -------
    tuple[float, float, float]
        ``(energy_mwh, max_tws_mps, max_hs_m)``.
    """
    if predict_power_batch is None:
        raise ModuleNotFoundError(
            "routetools.performance is required for evaluate_route_energy."
        )

    n_points = curve.shape[0]
    if n_points < 2:
        raise ValueError(f"curve must have at least 2 points, got {n_points}")
    n_seg = n_points - 1

    mid_lon = np.asarray((curve[:-1, 0] + curve[1:, 0]) / 2, dtype=np.float64)
    mid_lat = np.asarray((curve[:-1, 1] + curve[1:, 1]) / 2, dtype=np.float64)

    seg_frac = (np.arange(n_seg) + 0.5) / n_seg
    t_hours = departure_offset_h + seg_frac * passage_hours

    bearing_deg = segment_bearings_deg(curve)

    distance_nm = curve_distance_nm(curve)
    v_mps = (distance_nm * 1852.0) / (passage_hours * 3600.0)

    if windfield is not None:
        u10, v10 = windfield(jnp.array(mid_lon), jnp.array(mid_lat), jnp.array(t_hours))
        u10 = np.asarray(u10, dtype=np.float64)
        v10 = np.asarray(v10, dtype=np.float64)
        tws = np.sqrt(u10**2 + v10**2)
        wind_from_deg = (180.0 + np.degrees(np.arctan2(u10, v10))) % 360.0
        twa = (wind_from_deg - bearing_deg) % 360.0
    else:
        tws = np.zeros(n_seg)
        twa = np.zeros(n_seg)

    if wavefield is not None:
        hs, mwd = wavefield(jnp.array(mid_lon), jnp.array(mid_lat), jnp.array(t_hours))
        hs = np.asarray(hs, dtype=np.float64)
        mwd = np.asarray(mwd, dtype=np.float64)
        mwa = (mwd - bearing_deg) % 360.0
    else:
        hs = np.zeros(n_seg)
        mwa = np.zeros(n_seg)

    v_arr = np.full(n_seg, v_mps)
    power_kw = predict_power_batch(tws, twa, hs, mwa, v_arr, wps=wps)

    dt_hours = passage_hours / n_seg
    energy_kwh = float(jnp.sum(jnp.asarray(power_kw)) * dt_hours)
    energy_mwh = energy_kwh / 1000.0

    max_tws_mps = float(np.max(tws)) if windfield is not None else 0.0
    max_hs_m = float(np.max(hs)) if wavefield is not None else 0.0
    return energy_mwh, max_tws_mps, max_hs_m


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
    time_offset: float = 0.0,
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
        Coordinates are ordered as ``(lon, lat)`` for geographic fields, or
        ``(x, y)`` for projected planar fields.
    travel_stw : float, optional
        The boat will have this fixed speed through water (STW). If applying the
        spherical correction, this speed is in meters per second.
    travel_time : float, optional
        The boat can regulate its STW but must complete the path in exactly this
        time. Units must match the vector field time axis (for ERA5, hours).
    weight_l1 : float, optional
        Weight for the L1 norm in the combined cost. Default is 1.0.
    weight_l2 : float, optional
        Weight for the L2 norm in the combined cost. Default is 0.0.
    spherical_correction : bool, optional
        Whether to apply spherical correction to distances. If False, coordinates
        are expected to already be in projected metric units.
    time_offset : float, optional
        Offset added to segment timestamps before querying time-variant fields.
        Units must match ``travel_time`` (for ERA5, hours).

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
        cost = cost_function_constant_cost_time_variant(
            vectorfield,
            curve,
            travel_time,
            wavefield=wavefield,
            spherical_correction=spherical_correction,
            time_offset=time_offset,
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


@partial(
    jit,
    static_argnames=("vectorfield", "wavefield", "spherical_correction"),
)
def cost_function_constant_cost_time_variant(
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
    time_offset: float = 0.0,
) -> jnp.ndarray:
    """Compute energy cost for fixed-time routes with time-variant vector fields.

    Each segment takes ``dt = travel_time / (L-1)``.  The vector field is
    sampled at the midpoint of each segment at its corresponding timestamp,
    shifted by *time_offset*.

    Parameters
    ----------
    vectorfield : Callable
        ``(lon, lat, t) -> (u, v)`` where ``lon, lat`` are in degrees,
        ``u, v`` are in m/s, and ``t`` is in the same units as
        *travel_time* (typically hours for ERA5).
    curve : jnp.ndarray
        Batch of trajectories, shape ``(B, L, 2)``. Coordinates are
        ``(lon, lat)`` when ``spherical_correction=True`` and projected
        ``(x, y)`` in metres when ``spherical_correction=False``.
    travel_time : float
        Fixed total passage time (hours for ERA5).
    wavefield : Callable, optional
        ``(lon, lat, t) -> (height, direction)``. Currently unused in this
        function and kept for API symmetry.
    spherical_correction : bool
        Whether to compute distances on the sphere. If False, ``curve``
        coordinates must already be in metres.
    time_offset : float
        Offset added to segment timestamps before querying the field (hours).
        For ERA5 this is the departure's offset in hours from the
        dataset epoch (2024-01-01T00:00).

    Returns
    -------
    jnp.ndarray
        Cost per segment, shape ``(B, L-1)``. With SI inputs this has
        units of m^2/s.
    """
    n_seg = curve.shape[1] - 1
    dt = travel_time / n_seg

    # Segment midpoints (position)
    curvex = (curve[:, :-1, 0] + curve[:, 1:, 0]) / 2
    curvey = (curve[:, :-1, 1] + curve[:, 1:, 1]) / 2

    # Segment midpoints (time), shifted by departure offset.
    seg_times = time_offset + (jnp.arange(n_seg) + 0.5) * dt  # shape (n_seg,)
    # Broadcast to batch: shape (B, n_seg)
    curvet = jnp.broadcast_to(seg_times[None, :], curvex.shape)

    uinterp, vinterp = vectorfield(curvex, curvey, curvet)

    # TODO: wavefield is accepted for API symmetry with cost_function_rise
    # but is not yet used in this function.  Wire it into an added-resistance
    # term once the STW cost model supports wave effects.

    # Distances between waypoints
    if spherical_correction:
        dx, dy = haversine_meters_components(
            curve[:, :-1, 1],
            curve[:, :-1, 0],
            curve[:, 1:, 1],
            curve[:, 1:, 0],
        )
    else:
        # NOTE: when spherical_correction is False the curve coordinates
        # must already be in metres (projected x/y).  Raw lon/lat degrees
        # will produce dimensionally incorrect costs.
        dx = jnp.diff(curve[:, :, 0], axis=1)
        dy = jnp.diff(curve[:, :, 1], axis=1)

    # SOG = displacement / dt  (convert dt from hours to seconds so
    # that SOG is in m/s, matching the wind field units).
    dt_s = dt * 3600.0
    dxdt = dx / dt_s
    dydt = dy / dt_s

    # STW cost = ‖SOG - current‖² / 2 · dt_s
    cost = ((dxdt - uinterp) ** 2 + (dydt - vinterp) ** 2) / 2
    return cost * dt_s


# ---------------------------------------------------------------------------
# RISE performance-model cost (for SWOPP3)
# ---------------------------------------------------------------------------
@partial(
    jit,
    static_argnames=(
        "windfield",
        "wavefield",
        "travel_time",
        "wps",
    ),
)
def cost_function_rise(
    windfield: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]
    ],
    curve: jnp.ndarray,
    travel_time: float,
    wavefield: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]
    ]
    | None = None,
    wps: bool = False,
    time_offset: float = 0.0,
) -> jnp.ndarray:
    """Compute SWOPP3 energy consumption for a batch of fixed-time routes.

    Uses the closed-form RISE performance model (hull drag, aerodynamic
    drag, wave added resistance, wingsail thrust) evaluated entirely in
    JAX so that the function is JIT-compilable and could support
    gradient-based optimisation in the future.

    Each segment takes ``dt = travel_time / (L-1)``.  Wind and waves are
    sampled at each segment midpoint at its corresponding timestamp.

    Parameters
    ----------
    windfield : Callable
        ``(lon, lat, t) -> (u10, v10)`` where ``u10, v10`` are wind
        components in m/s.  ``t`` is in the same unit as *travel_time*
        (hours for ERA5).
    curve : jnp.ndarray
        Batch of trajectories, shape ``(B, L, 2)`` with ``(lon, lat)``
        in degrees. Coordinate order must match the wind/wave fields.
    travel_time : float
        Fixed total passage time (hours).
    wavefield : Callable, optional
        ``(lon, lat, t) -> (hs, mwd)`` where ``hs`` is significant wave
        height in metres and ``mwd`` is mean wave direction (degrees
        from North).
    wps : bool
        Whether wingsails are deployed.
    time_offset : float
        Departure offset in hours from the field's time origin.

    Returns
    -------
    jnp.ndarray
        Total energy in MWh per route, shape ``(B,)``.
    """
    if predict_power_jax is None:
        raise ModuleNotFoundError(
            "routetools.performance is required for cost_function_rise."
        )

    n_seg = curve.shape[1] - 1
    dt_h = travel_time / n_seg  # hours per segment

    # ---- segment midpoints (position) ----
    mid_lon = (curve[:, :-1, 0] + curve[:, 1:, 0]) / 2  # (B, n_seg)
    mid_lat = (curve[:, :-1, 1] + curve[:, 1:, 1]) / 2

    # ---- segment midpoints (time) ----
    seg_times = time_offset + (jnp.arange(n_seg) + 0.5) * dt_h  # (n_seg,)
    curvet = jnp.broadcast_to(seg_times[None, :], mid_lon.shape)  # (B, n_seg)

    # ---- segment bearings (great-circle, all in JAX) ----
    lon1_rad = jnp.radians(curve[:, :-1, 0])
    lon2_rad = jnp.radians(curve[:, 1:, 0])
    lat1_rad = jnp.radians(curve[:, :-1, 1])
    lat2_rad = jnp.radians(curve[:, 1:, 1])
    dlon_rad = lon2_rad - lon1_rad
    x = jnp.sin(dlon_rad) * jnp.cos(lat2_rad)
    y = jnp.cos(lat1_rad) * jnp.sin(lat2_rad) - jnp.sin(lat1_rad) * jnp.cos(
        lat2_rad
    ) * jnp.cos(dlon_rad)
    bearing_rad = jnp.arctan2(x, y)
    bearing_deg = jnp.mod(jnp.degrees(bearing_rad), 360.0)

    # ---- segment distances (haversine, metres) & ship speed ----
    dx_m, dy_m = haversine_meters_components(
        curve[:, :-1, 1],
        curve[:, :-1, 0],
        curve[:, 1:, 1],
        curve[:, 1:, 0],
    )
    seg_dist_m = jnp.sqrt(dx_m**2 + dy_m**2)  # (B, n_seg)
    dt_s = dt_h * 3600.0
    v_mps = seg_dist_m / dt_s  # ship speed m/s per segment

    # ---- wind ----
    u10, v10 = windfield(mid_lon, mid_lat, curvet)
    tws = jnp.sqrt(u10**2 + v10**2)
    # Wind FROM direction (meteorological convention)
    wind_from_deg = jnp.mod(180.0 + jnp.degrees(jnp.arctan2(u10, v10)), 360.0)
    twa = jnp.mod(wind_from_deg - bearing_deg, 360.0)

    # ---- waves ----
    if wavefield is not None:
        hs, mwd = wavefield(mid_lon, mid_lat, curvet)
        mwa = jnp.mod(mwd - bearing_deg, 360.0)
    else:
        hs = jnp.zeros_like(mid_lon)
        mwa = jnp.zeros_like(mid_lon)

    # ---- RISE power model (kW) ----
    power_kw = predict_power_jax(tws, twa, hs, mwa, v_mps, wps=wps)

    # ---- integrate: energy = Σ P_kW · Δt_h → kWh, then /1000 → MWh ----
    energy_mwh = jnp.sum(power_kw, axis=1) * dt_h / 1000.0

    return energy_mwh


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
