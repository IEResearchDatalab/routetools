from __future__ import annotations

import math
from datetime import datetime, timedelta

import jax.numpy as jnp

EARTH_RADIUS = 6371000  # in meters, globe mean radius
NAUTICAL_MILE_METERS = 1852.0


def haversine_meters_module(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Great-circle distance in meters between two points given in degrees."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * EARTH_RADIUS * math.asin(math.sqrt(a))


def haversine_meters_components(
    lat1: jnp.ndarray,
    lon1: jnp.ndarray,
    lat2: jnp.ndarray,
    lon2: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Great-circle distance in meters between two points given in degrees."""
    # Convert from degrees to radians
    phi1, phi2 = jnp.radians(lat1), jnp.radians(lat2)
    lambda1, lambda2 = jnp.radians(lon1), jnp.radians(lon2)
    # Compute components in meters
    dy = EARTH_RADIUS * (phi2 - phi1)
    dx = EARTH_RADIUS * (lambda2 - lambda1) * jnp.cos((phi1 + phi2) / 2)
    return dx, dy


def haversine_distance_from_curve(curve: jnp.ndarray) -> jnp.ndarray:
    """Given a curve, return its Haversine distance in meters.

    Parameters
    ----------
    curve : jnp.ndarray
        An array of shape (L, 2) representing a trajectory (lon, lat).

    Returns
    -------
    jnp.ndarray
        An array of shape (L-1,) representing the distances between consecutive points.
    """
    lon, lat = curve[:, 0], curve[:, 1]
    dx, dy = haversine_meters_components(lat[:-1], lon[:-1], lat[1:], lon[1:])
    return jnp.sqrt(dx**2 + dy**2)


def curve_distance_nm(curve: jnp.ndarray) -> float:
    """Return total route length in nautical miles.

    Parameters
    ----------
    curve : jnp.ndarray
        Route waypoints with shape ``(L, 2)`` in ``(lon, lat)`` degrees.

    Returns
    -------
    float
        Total distance in nautical miles.
    """
    return float(jnp.sum(haversine_distance_from_curve(curve))) / NAUTICAL_MILE_METERS


def waypoint_times_uniform(
    curve: jnp.ndarray,
    departure: datetime,
    passage_hours: float,
) -> list[datetime]:
    """Compute UTC timestamps at each waypoint with uniform time spacing.

    Parameters
    ----------
    curve : jnp.ndarray
        Waypoints with shape ``(L, 2)``.
    departure : datetime
        Departure timestamp.
    passage_hours : float
        Total passage duration in hours.

    Returns
    -------
    list[datetime]
        One timestamp per waypoint.

    Raises
    ------
    ValueError
        If *curve* has no waypoints.
    """
    n_points = curve.shape[0]
    if n_points == 0:
        raise ValueError("curve must contain at least one waypoint")
    if n_points == 1:
        return [departure]

    total_seconds = passage_hours * 3600.0
    return [
        departure + timedelta(seconds=total_seconds * i / (n_points - 1))
        for i in range(n_points)
    ]


def great_circle_route(
    src: jnp.ndarray,
    dst: jnp.ndarray,
    n_points: int = 100,
) -> jnp.ndarray:
    """Compute a great-circle route between two points.

    Uses spherical interpolation (SLERP) in Cartesian 3-D, then projects
    back to ``(lon, lat)`` in degrees while unwrapping longitude across the
    antimeridian.

    Parameters
    ----------
    src : jnp.ndarray
        Source ``(lon, lat)`` in degrees, shape ``(2,)``.
    dst : jnp.ndarray
        Destination ``(lon, lat)`` in degrees, shape ``(2,)``.
    n_points : int
        Number of waypoints including endpoints.

    Returns
    -------
    jnp.ndarray
        Route shape ``(n_points, 2)`` in ``(lon, lat)``.
    """
    lon1, lat1 = jnp.deg2rad(src[0]), jnp.deg2rad(src[1])
    lon2, lat2 = jnp.deg2rad(dst[0]), jnp.deg2rad(dst[1])

    x1 = jnp.cos(lat1) * jnp.cos(lon1)
    y1 = jnp.cos(lat1) * jnp.sin(lon1)
    z1 = jnp.sin(lat1)

    x2 = jnp.cos(lat2) * jnp.cos(lon2)
    y2 = jnp.cos(lat2) * jnp.sin(lon2)
    z2 = jnp.sin(lat2)

    dot = x1 * x2 + y1 * y2 + z1 * z2
    omega = jnp.arccos(jnp.clip(dot, -1.0, 1.0))

    t = jnp.linspace(0.0, 1.0, n_points)
    sin_omega = jnp.sin(omega)
    is_coincident = sin_omega < 1e-10

    safe_sin = jnp.where(is_coincident, 1.0, sin_omega)
    a = jnp.where(is_coincident, 1.0 - t, jnp.sin((1.0 - t) * omega) / safe_sin)
    b = jnp.where(is_coincident, t, jnp.sin(t * omega) / safe_sin)

    x = a * x1 + b * x2
    y = a * y1 + b * y2
    z = a * z1 + b * z2

    lat = jnp.rad2deg(jnp.arcsin(jnp.clip(z, -1.0, 1.0)))
    lon = jnp.rad2deg(jnp.arctan2(y, x))

    dlon = jnp.diff(lon)
    dlon_wrapped = (dlon + 180.0) % 360.0 - 180.0
    lon = jnp.concatenate([lon[:1], lon[0] + jnp.cumsum(dlon_wrapped)])

    return jnp.stack([lon, lat], axis=1)
