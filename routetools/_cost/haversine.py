from __future__ import annotations

import math

import jax.numpy as jnp

EARTH_RADIUS = 6371000  # in meters, globe mean radius


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
