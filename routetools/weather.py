"""Weather constraint enforcement for route optimization.

Provides penalty functions that discourage routes from traversing regions
where weather conditions exceed safe operating limits.  The two SWOPP3
constraints are:

- True Wind Speed (TWS) < 20 m/s
- Significant Wave Height (Hs) < 7 m

The penalty functions follow the same pattern as ``Land.penalization`` and
are designed to be added to the cost in the CMA-ES optimization loop::

    cost += weather_penalty(curve, windfield, wavefield, ...)

The module also provides a ``RouteWeatherStats`` dataclass to report
per-route max TWS and max Hs (required columns in SWOPP3 File A).

Example
-------
>>> from routetools.weather import weather_penalty
>>> # penalty = weather_penalty(curve, windfield, wavefield)
>>> # cost += penalty
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp

from routetools._cost.haversine import haversine_meters_components

# ---------------------------------------------------------------------------
# Default SWOPP3 constraint thresholds
# ---------------------------------------------------------------------------
DEFAULT_TWS_LIMIT: float = 20.0
"""Maximum allowed true wind speed in m/s (SWOPP3 spec)."""

DEFAULT_HS_LIMIT: float = 7.0
"""Maximum allowed significant wave height in m (SWOPP3 spec)."""

__all__ = [
    "DEFAULT_TWS_LIMIT",
    "DEFAULT_HS_LIMIT",
    "RouteWeatherStats",
    "evaluate_weather",
    "weather_penalty",
    "weather_penalty_smooth",
]


# ---------------------------------------------------------------------------
# Route weather statistics
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RouteWeatherStats:
    """Per-route weather statistics for reporting.

    Attributes
    ----------
    max_tws : jnp.ndarray
        Maximum true wind speed along each route, shape ``(B,)``.
    max_hs : jnp.ndarray
        Maximum significant wave height along each route, shape ``(B,)``.
    tws_exceeded : jnp.ndarray
        Boolean array: whether any segment midpoint exceeded the TWS limit,
        shape ``(B,)``.
    hs_exceeded : jnp.ndarray
        Boolean array: whether any segment midpoint exceeded the Hs limit,
        shape ``(B,)``.
    """

    max_tws: jnp.ndarray
    max_hs: jnp.ndarray
    tws_exceeded: jnp.ndarray
    hs_exceeded: jnp.ndarray


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------
def _segment_midpoints(
    curve: jnp.ndarray,
    travel_stw: float | None = None,
    travel_time: float | None = None,
    spherical_correction: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute segment midpoint coordinates and elapsed timestamps.

    Parameters
    ----------
    curve : jnp.ndarray
        Shape ``(B, L, 2)`` with ``(lon, lat)``.
    travel_stw : float, optional
        Constant speed through water (m/s).  Used to estimate elapsed time
        per segment as ``distance / travel_stw``; ``t_mid`` will be in
        seconds when ``spherical_correction`` is ``True``.
    travel_time : float, optional
        Total travel time.  Each segment is assumed to occupy an equal share
        of the total duration, ``travel_time / (L - 1)``. The units of
        ``travel_time`` carry through to ``t_mid`` (e.g. pass hours to get
        hours).
    spherical_correction : bool
        If ``True`` (default), use haversine distances (metres).

    Returns
    -------
    mid_lon, mid_lat, t_mid : jnp.ndarray
        Each of shape ``(B, L-1)``.  ``t_mid`` is the estimated elapsed
        time at the midpoint of each segment, in the same units as
        ``travel_time`` or seconds when ``travel_stw`` is used.
    """
    mid_lon = (curve[:, :-1, 0] + curve[:, 1:, 0]) / 2
    mid_lat = (curve[:, :-1, 1] + curve[:, 1:, 1]) / 2

    if travel_stw is None and travel_time is None:
        # Fallback: no time information → zeros (original behaviour)
        return mid_lon, mid_lat, jnp.zeros_like(mid_lon)

    if curve.shape[1] < 2:
        return mid_lon, mid_lat, jnp.zeros_like(mid_lon)

    # Compute segment distances
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
    segment_dist = jnp.sqrt(dx**2 + dy**2)

    if travel_time is not None:
        # Fixed-time SWOPP3 routes assume equal time spacing between waypoints.
        segment_dt = jnp.full_like(segment_dist, travel_time / (curve.shape[1] - 1))
    else:
        # Constant STW: t = distance / speed
        segment_dt = segment_dist / travel_stw

    # Cumulative time at segment midpoints
    cumulative_t = jnp.cumsum(segment_dt, axis=1)
    t_mid = cumulative_t - segment_dt / 2

    return mid_lon, mid_lat, t_mid


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------
def evaluate_weather(
    curve: jnp.ndarray,
    windfield: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray],
        tuple[jnp.ndarray, jnp.ndarray],
    ]
    | None = None,
    wavefield: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray],
        tuple[jnp.ndarray, jnp.ndarray],
    ]
    | None = None,
    tws_limit: float = DEFAULT_TWS_LIMIT,
    hs_limit: float = DEFAULT_HS_LIMIT,
    travel_stw: float | None = None,
    travel_time: float | None = None,
    spherical_correction: bool = True,
) -> RouteWeatherStats:
    """Evaluate weather conditions along routes and report statistics.

    Samples the windfield and wavefield at the **midpoints** of each
    route segment (same convention as the cost function) and computes
    per-route maxima.

    Parameters
    ----------
    curve : jnp.ndarray
        Batch of trajectories, shape ``(B, L, 2)`` with ``(lon, lat)``
        coordinates.
    windfield : Callable, optional
        ``(lon, lat, t) -> (u10, v10)`` closure.  If ``None``, TWS stats
        are returned as zeros.
    wavefield : Callable, optional
        ``(lon, lat, t) -> (hs, mwd)`` closure.  If ``None``, Hs stats
        are returned as zeros.
    tws_limit : float
        TWS threshold in m/s.
    hs_limit : float
        Hs threshold in m.
    travel_stw : float, optional
        Constant speed through water (m/s) for elapsed-time estimation.
    travel_time : float, optional
        Total travel time in the same units expected by the field closures.
        Each segment is assumed to occupy ``travel_time / (L - 1)``.
    spherical_correction : bool
        Use haversine distances (default ``True``).

    Returns
    -------
    RouteWeatherStats
        Per-route maxima and exceedance flags.
    """
    B = curve.shape[0]

    # Guard: curves with fewer than 2 points have no segments
    if curve.shape[1] < 2:
        return RouteWeatherStats(
            max_tws=jnp.zeros(B),
            max_hs=jnp.zeros(B),
            tws_exceeded=jnp.zeros(B, dtype=bool),
            hs_exceeded=jnp.zeros(B, dtype=bool),
        )

    # Midpoints of each segment
    mid_lon, mid_lat, t_mid = _segment_midpoints(
        curve,
        travel_stw=travel_stw,
        travel_time=travel_time,
        spherical_correction=spherical_correction,
    )

    # Wind
    if windfield is not None:
        u10, v10 = windfield(mid_lon, mid_lat, t_mid)
        tws = jnp.sqrt(u10**2 + v10**2)
        max_tws = jnp.max(tws, axis=1)
    else:
        max_tws = jnp.zeros(B)

    # Waves
    if wavefield is not None:
        hs, _ = wavefield(mid_lon, mid_lat, t_mid)
        max_hs = jnp.max(hs, axis=1)
    else:
        max_hs = jnp.zeros(B)

    return RouteWeatherStats(
        max_tws=max_tws,
        max_hs=max_hs,
        tws_exceeded=max_tws > tws_limit,
        hs_exceeded=max_hs > hs_limit,
    )


# ---------------------------------------------------------------------------
# Hard penalty (step function — same pattern as Land.penalization)
# ---------------------------------------------------------------------------
def weather_penalty(
    curve: jnp.ndarray,
    windfield: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray],
        tuple[jnp.ndarray, jnp.ndarray],
    ]
    | None = None,
    wavefield: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray],
        tuple[jnp.ndarray, jnp.ndarray],
    ]
    | None = None,
    tws_limit: float = DEFAULT_TWS_LIMIT,
    hs_limit: float = DEFAULT_HS_LIMIT,
    penalty: float = 10.0,
    travel_stw: float | None = None,
    travel_time: float | None = None,
    spherical_correction: bool = True,
    time_offset: float = 0.0,
) -> jnp.ndarray:
    """Compute a hard penalty for weather constraint violations.

    For each route in the batch, counts the number of **segments** where
    TWS or Hs exceeds the threshold, multiplied by ``penalty``.  This is
    analogous to ``Land.penalization``.

    Parameters
    ----------
    curve : jnp.ndarray
        Batch of trajectories, shape ``(B, L, 2)``.
    windfield : Callable, optional
        ``(lon, lat, t) -> (u10, v10)``.
    wavefield : Callable, optional
        ``(lon, lat, t) -> (hs, mwd)``.
    tws_limit : float
        TWS threshold in m/s (default 20).
    hs_limit : float
        Hs threshold in m (default 7).
    penalty : float
        Penalty per violating segment (default 10).
    travel_stw : float, optional
        Constant speed through water (m/s) for elapsed-time estimation.
    travel_time : float, optional
        Total travel time with arbitrary time units. Each segment is assumed
        to occupy ``travel_time / (L - 1)``. Units must match those used by
        ``time_offset`` and the time coordinate expected by the field closures.
    spherical_correction : bool
        Use haversine distances (default ``True``).
    time_offset : float
        Offset added to ``t_mid`` before querying field closures.  Must
        be in the same units as ``travel_time`` (or seconds when using
        ``travel_stw``).  Typically the departure offset in hours when
        ``travel_time`` is also in hours.

    Returns
    -------
    jnp.ndarray
        Penalty per route, shape ``(B,)``.
    """
    mid_lon, mid_lat, t_mid = _segment_midpoints(
        curve,
        travel_stw=travel_stw,
        travel_time=travel_time,
        spherical_correction=spherical_correction,
    )
    t_mid = t_mid + time_offset

    violations = jnp.zeros(curve.shape[0])

    if windfield is not None:
        u10, v10 = windfield(mid_lon, mid_lat, t_mid)
        tws = jnp.sqrt(u10**2 + v10**2)
        violations = violations + jnp.sum(tws > tws_limit, axis=1)

    if wavefield is not None:
        hs, _ = wavefield(mid_lon, mid_lat, t_mid)
        violations = violations + jnp.sum(hs > hs_limit, axis=1)

    return violations * penalty


# ---------------------------------------------------------------------------
# Smooth penalty (differentiable — useful for gradient-based methods)
# ---------------------------------------------------------------------------
def weather_penalty_smooth(
    curve: jnp.ndarray,
    windfield: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray],
        tuple[jnp.ndarray, jnp.ndarray],
    ]
    | None = None,
    wavefield: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray],
        tuple[jnp.ndarray, jnp.ndarray],
    ]
    | None = None,
    tws_limit: float = DEFAULT_TWS_LIMIT,
    hs_limit: float = DEFAULT_HS_LIMIT,
    penalty: float = 10.0,
    sharpness: float = 5.0,
    travel_stw: float | None = None,
    travel_time: float | None = None,
    spherical_correction: bool = True,
    time_offset: float = 0.0,
) -> jnp.ndarray:
    """Compute a smooth (differentiable) penalty for weather violations.

    Uses a squared ReLU ramp so that the penalty increases continuously
    as conditions worsen beyond the threshold.  For each segment:

        ``penalty_i = sharpness · max(0, value - limit)²``

    This is summed over all segments per route and scaled by ``penalty``.

    Parameters
    ----------
    curve : jnp.ndarray
        Batch of trajectories, shape ``(B, L, 2)``.
    windfield : Callable, optional
        ``(lon, lat, t) -> (u10, v10)``.
    wavefield : Callable, optional
        ``(lon, lat, t) -> (hs, mwd)``.
    tws_limit : float
        TWS threshold in m/s (default 20).
    hs_limit : float
        Hs threshold in m (default 7).
    penalty : float
        Scaling factor (default 10).
    sharpness : float
        Linear multiplier on the squared excess (default 5).
    travel_stw : float, optional
        Constant speed through water (m/s) used to infer elapsed time from
        segment distances.  When provided, elapsed times are computed in
        seconds.
    travel_time : float, optional
        Total travel duration in arbitrary time units; distributed
        proportionally by segment distance.  Units are not interpreted by
        this function, but they are carried through to ``t_mid`` and must
        be consistent with what ``windfield``/``wavefield`` expect.
    spherical_correction : bool
        Use haversine distances (default ``True``).
    time_offset : float
        Offset added to ``t_mid`` before querying field closures.  Must
        be in the same units as the resulting time axis: seconds when
        using ``travel_stw``, or the same arbitrary units as
        ``travel_time`` when it is provided directly (for example, hours
        when ``travel_time`` is in hours).

    Returns
    -------
    jnp.ndarray
        Smooth penalty per route, shape ``(B,)``.
    """
    mid_lon, mid_lat, t_mid = _segment_midpoints(
        curve,
        travel_stw=travel_stw,
        travel_time=travel_time,
        spherical_correction=spherical_correction,
    )
    t_mid = t_mid + time_offset

    total = jnp.zeros(curve.shape[0])

    if windfield is not None:
        u10, v10 = windfield(mid_lon, mid_lat, t_mid)
        tws = jnp.sqrt(u10**2 + v10**2)
        excess = jnp.maximum(tws - tws_limit, 0.0)
        total = total + jnp.sum(excess**2, axis=1) * sharpness

    if wavefield is not None:
        hs, _ = wavefield(mid_lon, mid_lat, t_mid)
        excess = jnp.maximum(hs - hs_limit, 0.0)
        total = total + jnp.sum(excess**2, axis=1) * sharpness

    return total * penalty
