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
    "wind_penalty_smooth",
    "wave_penalty_smooth",
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
def _safe_mean(x: jnp.ndarray, axis: int) -> jnp.ndarray:
    """Mean over *axis*, returning 0 when that axis is empty."""
    n = x.shape[axis]
    return jnp.where(n > 0, jnp.sum(x, axis=axis) / jnp.maximum(n, 1), 0.0)


def _segment_midpoints(
    curve: jnp.ndarray,
    travel_stw: float | None = None,
    travel_time: float | None = None,
    spherical_correction: bool = True,
    time_offset: float = 0.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute segment midpoint coordinates and timestamps.

    Parameters
    ----------
    curve : jnp.ndarray
        Shape ``(B, L, 2)`` with ``(lon, lat)``.
    travel_stw : float, optional
        Constant speed through water (m/s).  Used to estimate elapsed time
        per segment as ``distance / travel_stw``.
    travel_time : float, optional
        Total travel time.  Time is distributed proportionally
        by segment distance.  Units must match *time_offset*.
    spherical_correction : bool
        If ``True`` (default), use haversine distances (metres).
    time_offset : float
        Constant added to all midpoint timestamps (e.g. departure time
        offset in the same unit as *travel_time*).  Default ``0.0``.

    Returns
    -------
    mid_lon, mid_lat, t_mid : jnp.ndarray
        Each of shape ``(B, L-1)``.  ``t_mid`` is ``time_offset`` plus
        the estimated elapsed time at the midpoint of each segment.
    """
    mid_lon = (curve[:, :-1, 0] + curve[:, 1:, 0]) / 2
    mid_lat = (curve[:, :-1, 1] + curve[:, 1:, 1]) / 2

    if travel_stw is None and travel_time is None:
        # Fallback: no time information → time_offset only
        return mid_lon, mid_lat, jnp.full_like(mid_lon, time_offset)

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
        # Constant time: distribute total time proportionally by distance
        total_dist = jnp.sum(segment_dist, axis=1, keepdims=True)
        # Avoid division by zero for degenerate curves
        total_dist = jnp.maximum(total_dist, 1e-12)
        segment_dt = (segment_dist / total_dist) * travel_time
    else:
        # Constant STW: t = distance / speed
        segment_dt = segment_dist / travel_stw

    # Cumulative time at segment midpoints
    cumulative_t = jnp.cumsum(segment_dt, axis=1)
    t_mid = cumulative_t - segment_dt / 2 + time_offset

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
    time_offset: float = 0.0,
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
        Total travel time (seconds); distributed proportionally by distance.
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
        time_offset=time_offset,
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
        Total travel time (seconds); distributed proportionally by distance.
    spherical_correction : bool
        Use haversine distances (default ``True``).

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
        time_offset=time_offset,
    )

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

    This is averaged over all segments per route and scaled by ``penalty``,
    making the penalty independent of route resolution.

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
        Constant speed through water (m/s) for elapsed-time estimation.
    travel_time : float, optional
        Total travel time (seconds); distributed proportionally by distance.
    spherical_correction : bool
        Use haversine distances (default ``True``).

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
        time_offset=time_offset,
    )

    total = jnp.zeros(curve.shape[0])

    if windfield is not None:
        u10, v10 = windfield(mid_lon, mid_lat, t_mid)
        tws = jnp.sqrt(u10**2 + v10**2)
        excess = jnp.maximum(tws - tws_limit, 0.0)
        total = total + _safe_mean(excess**2, axis=1) * sharpness

    if wavefield is not None:
        hs, _ = wavefield(mid_lon, mid_lat, t_mid)
        excess = jnp.maximum(hs - hs_limit, 0.0)
        total = total + _safe_mean(excess**2, axis=1) * sharpness

    return total * penalty


# ---------------------------------------------------------------------------
# Split penalties — independent wind and wave smooth penalties
# ---------------------------------------------------------------------------
def wind_penalty_smooth(
    curve: jnp.ndarray,
    windfield: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray],
        tuple[jnp.ndarray, jnp.ndarray],
    ],
    tws_limit: float = DEFAULT_TWS_LIMIT,
    weight: float = 50.0,
    travel_stw: float | None = None,
    travel_time: float | None = None,
    spherical_correction: bool = True,
    time_offset: float = 0.0,
) -> jnp.ndarray:
    """Smooth penalty for wind-speed constraint violations only.

    For each segment where TWS exceeds ``tws_limit``:

        ``penalty_i = max(0, TWS_i - tws_limit)²``

    The per-segment penalties are averaged and scaled by ``weight``,
    making the penalty independent of route resolution.

    Parameters
    ----------
    curve : jnp.ndarray
        Batch of trajectories, shape ``(B, L, 2)``.
    windfield : Callable
        ``(lon, lat, t) -> (u10, v10)``.
    tws_limit : float
        TWS threshold in m/s (default 20).
    weight : float
        Scaling factor applied to the mean squared-excess penalty
        (default 50).
    travel_stw : float, optional
        Constant speed through water (m/s).
    travel_time : float, optional
        Total travel time (seconds).
    spherical_correction : bool
        Use haversine distances (default ``True``).

    Returns
    -------
    jnp.ndarray
        Smooth wind penalty per route, shape ``(B,)``.
    """
    mid_lon, mid_lat, t_mid = _segment_midpoints(
        curve,
        travel_stw=travel_stw,
        travel_time=travel_time,
        spherical_correction=spherical_correction,
        time_offset=time_offset,
    )
    u10, v10 = windfield(mid_lon, mid_lat, t_mid)
    tws = jnp.sqrt(u10**2 + v10**2)
    excess = jnp.maximum(tws - tws_limit, 0.0)
    return weight * _safe_mean(excess**2, axis=1)


def wave_penalty_smooth(
    curve: jnp.ndarray,
    wavefield: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray],
        tuple[jnp.ndarray, jnp.ndarray],
    ],
    hs_limit: float = DEFAULT_HS_LIMIT,
    weight: float = 50.0,
    travel_stw: float | None = None,
    travel_time: float | None = None,
    spherical_correction: bool = True,
    time_offset: float = 0.0,
) -> jnp.ndarray:
    """Smooth penalty for wave-height constraint violations only.

    For each segment where Hs exceeds ``hs_limit``:

        ``penalty_i = max(0, Hs_i - hs_limit)²``

    The per-segment penalties are averaged and scaled by ``weight``,
    making the penalty independent of route resolution.

    Parameters
    ----------
    curve : jnp.ndarray
        Batch of trajectories, shape ``(B, L, 2)``.
    wavefield : Callable
        ``(lon, lat, t) -> (hs, mwd)``.
    hs_limit : float
        Hs threshold in m (default 7).
    weight : float
        Scaling factor applied to the mean squared-excess penalty
        (default 50).
    travel_stw : float, optional
        Constant speed through water (m/s).
    travel_time : float, optional
        Total travel time (seconds).
    spherical_correction : bool
        Use haversine distances (default ``True``).

    Returns
    -------
    jnp.ndarray
        Smooth wave penalty per route, shape ``(B,)``.
    """
    mid_lon, mid_lat, t_mid = _segment_midpoints(
        curve,
        travel_stw=travel_stw,
        travel_time=travel_time,
        spherical_correction=spherical_correction,
        time_offset=time_offset,
    )
    hs, _ = wavefield(mid_lon, mid_lat, t_mid)
    excess = jnp.maximum(hs - hs_limit, 0.0)
    return weight * _safe_mean(excess**2, axis=1)
