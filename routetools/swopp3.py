"""SWOPP3 competition configuration — routes, departures, and cases.

Defines the two SWOPP3 routes (Trans-Atlantic and Trans-Pacific), their
fixed passage times, and the 366 daily departures throughout 2024.

The 8 SWOPP3 cases (Table 2 in the SWOPP3 info package):

====  =========  ====================  =========================  ===
Case  Name       Route                 Strategy                   WPS
====  =========  ====================  =========================  ===
1     AO_WPS     Atlantic westbound    Optimised route + speed    yes
2     AO_noWPS   Atlantic westbound    Optimised route + speed    no
3     AGC_WPS    Atlantic westbound    Great circle, fixed speed  yes
4     AGC_noWPS  Atlantic westbound    Great circle, fixed speed  no
5     PO_WPS     Pacific eastbound     Optimised route + speed    yes
6     PO_noWPS   Pacific eastbound     Optimised route + speed    no
7     PGC_WPS    Pacific eastbound     Great circle, fixed speed  yes
8     PGC_noWPS  Pacific eastbound     Great circle, fixed speed  no
====  =========  ====================  =========================  ===

- **Optimised (O):** CMA-ES finds minimum-energy route and speed profile.
- **Great Circle (GC):** Fixed geodesic route, constant speed.
- **WPS:** RISE polar model with wingsails enabled.
- **noWPS:** RISE polar model with wingsails disabled (engine only).

Example
-------
>>> from routetools.swopp3 import SWOPP3_CASES, departures_2024
>>> case = SWOPP3_CASES["AO_WPS"]
>>> deps = departures_2024()
>>> print(f"{case['name']}: {len(deps)} departures, {case['passage_hours']}h passage")
AO_WPS: 366 departures, 354h passage
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Port coordinates  (lon, lat)  — matching routetools._ports.DICT_PORTS
# ---------------------------------------------------------------------------
PORTS: dict[str, dict] = {
    "ESSDR": {
        "name": "Santander",
        "lat": 43.6,
        "lon": -4.0,
    },
    "USNYC": {
        "name": "New York",
        "lat": 40.53,
        "lon": -73.80,
    },
    "JPTYO": {
        "name": "Tokyo / Yokohama",
        "lat": 34.8,
        "lon": 140.0,
    },
    "USLAX": {
        "name": "Los Angeles",
        "lat": 34.4,
        "lon": -121.0,
    },
}

# ---------------------------------------------------------------------------
# Route definitions
# ---------------------------------------------------------------------------
ROUTE_ATLANTIC = {
    "id": "atlantic",
    "label": "Trans-Atlantic",
    "src_port": "ESSDR",
    "dst_port": "USNYC",
    "passage_hours": 354,
    "gc_distance_nm": 2833,
}

ROUTE_PACIFIC = {
    "id": "pacific",
    "label": "Trans-Pacific",
    "src_port": "JPTYO",
    "dst_port": "USLAX",
    "passage_hours": 583,
    "gc_distance_nm": 4663,
}

# ---------------------------------------------------------------------------
# The 8 SWOPP3 cases
# ---------------------------------------------------------------------------
SWOPP3_CASES: dict[str, dict] = {
    "AO_WPS": {
        "name": "AO_WPS",
        "label": "Atlantic Optimised, with WPS",
        "route": "atlantic",
        "src_port": "ESSDR",
        "dst_port": "USNYC",
        "passage_hours": 354,
        "strategy": "optimised",
        "wps": True,
    },
    "AO_noWPS": {
        "name": "AO_noWPS",
        "label": "Atlantic Optimised, without WPS",
        "route": "atlantic",
        "src_port": "ESSDR",
        "dst_port": "USNYC",
        "passage_hours": 354,
        "strategy": "optimised",
        "wps": False,
    },
    "AGC_WPS": {
        "name": "AGC_WPS",
        "label": "Atlantic Great Circle, with WPS",
        "route": "atlantic",
        "src_port": "ESSDR",
        "dst_port": "USNYC",
        "passage_hours": 354,
        "strategy": "gc",
        "wps": True,
    },
    "AGC_noWPS": {
        "name": "AGC_noWPS",
        "label": "Atlantic Great Circle, without WPS",
        "route": "atlantic",
        "src_port": "ESSDR",
        "dst_port": "USNYC",
        "passage_hours": 354,
        "strategy": "gc",
        "wps": False,
    },
    "PO_WPS": {
        "name": "PO_WPS",
        "label": "Pacific Optimised, with WPS",
        "route": "pacific",
        "src_port": "JPTYO",
        "dst_port": "USLAX",
        "passage_hours": 583,
        "strategy": "optimised",
        "wps": True,
    },
    "PO_noWPS": {
        "name": "PO_noWPS",
        "label": "Pacific Optimised, without WPS",
        "route": "pacific",
        "src_port": "JPTYO",
        "dst_port": "USLAX",
        "passage_hours": 583,
        "strategy": "optimised",
        "wps": False,
    },
    "PGC_WPS": {
        "name": "PGC_WPS",
        "label": "Pacific Great Circle, with WPS",
        "route": "pacific",
        "src_port": "JPTYO",
        "dst_port": "USLAX",
        "passage_hours": 583,
        "strategy": "gc",
        "wps": True,
    },
    "PGC_noWPS": {
        "name": "PGC_noWPS",
        "label": "Pacific Great Circle, without WPS",
        "route": "pacific",
        "src_port": "JPTYO",
        "dst_port": "USLAX",
        "passage_hours": 583,
        "strategy": "gc",
        "wps": False,
    },
}

# ---------------------------------------------------------------------------
# Departure schedule
# ---------------------------------------------------------------------------
_YEAR = 2024  # Leap year → 366 days
_DEPARTURE_HOUR = 12  # Noon UTC


def departures_2024() -> list[datetime]:
    """Return the 366 daily noon-UTC departures for 2024.

    Returns
    -------
    list[datetime]
        366 timezone-aware ``datetime`` objects (UTC), one per day from
        2024-01-01 12:00 UTC through 2024-12-31 12:00 UTC.
    """
    start = datetime(_YEAR, 1, 1, _DEPARTURE_HOUR, tzinfo=UTC)
    return [start + timedelta(days=d) for d in range(366)]


def departure_strings(fmt: str = "%Y-%m-%dT%H:%M:%S") -> list[str]:
    """Return departure datetimes as formatted strings.

    Parameters
    ----------
    fmt : str
        ``strftime`` format, default ISO-8601 without timezone suffix.

    Returns
    -------
    list[str]
        366 formatted date strings.
    """
    return [d.strftime(fmt) for d in departures_2024()]


# ---------------------------------------------------------------------------
# Helper: build (src, dst) JAX arrays for a case
# ---------------------------------------------------------------------------
def case_endpoints(case_id: str) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return ``(src, dst)`` as JAX arrays of ``(lon, lat)`` for a case.

    Parameters
    ----------
    case_id : str
        Identifier of a SWOPP3 case, i.e. one of ``SWOPP3_CASES.keys()`` (for
        example ``"AO_WPS"``).

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        ``src`` and ``dst``, each shape ``(2,)`` with ``(lon, lat)``.

    Raises
    ------
    KeyError
        If *case_id* is not a valid SWOPP3 case.
    """
    case = SWOPP3_CASES[case_id]
    src_port = PORTS[case["src_port"]]
    dst_port = PORTS[case["dst_port"]]
    src = jnp.array([src_port["lon"], src_port["lat"]])
    dst = jnp.array([dst_port["lon"], dst_port["lat"]])
    return src, dst


def case_travel_time_seconds(case_id: str) -> float:
    """Return the fixed passage time in seconds for a case.

    Parameters
    ----------
    case_id : str
        Identifier of a SWOPP3 case, i.e. one of ``SWOPP3_CASES.keys()``.

    Returns
    -------
    float
        Passage time in seconds.
    """
    return float(SWOPP3_CASES[case_id]["passage_hours"] * 3600)


# ---------------------------------------------------------------------------
# Great-circle baseline
# ---------------------------------------------------------------------------
def great_circle_route(
    src: jnp.ndarray,
    dst: jnp.ndarray,
    n_points: int = 100,
) -> jnp.ndarray:
    """Compute a great-circle route between two points.

    Uses spherical interpolation (SLERP) in Cartesian 3-D, then projects
    back to ``(lon, lat)`` in degrees.  Handles the antimeridian correctly
    (no 360° wrapping artifacts).

    Parameters
    ----------
    src : jnp.ndarray
        ``(lon, lat)`` in degrees, shape ``(2,)``.
    dst : jnp.ndarray
        ``(lon, lat)`` in degrees, shape ``(2,)``.
    n_points : int
        Number of waypoints (including endpoints).

    Returns
    -------
    jnp.ndarray
        Shape ``(n_points, 2)`` with ``(lon, lat)`` columns in degrees.
    """
    # Convert to radians
    lon1, lat1 = jnp.deg2rad(src[0]), jnp.deg2rad(src[1])
    lon2, lat2 = jnp.deg2rad(dst[0]), jnp.deg2rad(dst[1])

    # To Cartesian
    x1 = jnp.cos(lat1) * jnp.cos(lon1)
    y1 = jnp.cos(lat1) * jnp.sin(lon1)
    z1 = jnp.sin(lat1)

    x2 = jnp.cos(lat2) * jnp.cos(lon2)
    y2 = jnp.cos(lat2) * jnp.sin(lon2)
    z2 = jnp.sin(lat2)

    # Central angle
    dot = x1 * x2 + y1 * y2 + z1 * z2
    omega = jnp.arccos(jnp.clip(dot, -1.0, 1.0))

    # SLERP
    t = jnp.linspace(0.0, 1.0, n_points)
    sin_omega = jnp.sin(omega)
    is_coincident = sin_omega < 1e-10

    # For coincident points, fall back to linear interpolation (a=1-t, b=t)
    safe_sin = jnp.where(is_coincident, 1.0, sin_omega)
    a = jnp.where(is_coincident, 1.0 - t, jnp.sin((1.0 - t) * omega) / safe_sin)
    b = jnp.where(is_coincident, t, jnp.sin(t * omega) / safe_sin)

    x = a * x1 + b * x2
    y = a * y1 + b * y2
    z = a * z1 + b * z2

    # Back to (lon, lat) degrees
    lat = jnp.rad2deg(jnp.arcsin(jnp.clip(z, -1.0, 1.0)))
    lon = jnp.rad2deg(jnp.arctan2(y, x))

    # Unwrap longitude to avoid antimeridian discontinuities
    # (differences > 180° are wrapped by ±360°)
    dlon = jnp.diff(lon)
    dlon_wrapped = (dlon + 180.0) % 360.0 - 180.0
    lon = jnp.concatenate([lon[:1], lon[0] + jnp.cumsum(dlon_wrapped)])

    return jnp.stack([lon, lat], axis=1)
