"""SWOPP3 competition configuration — routes, departures, and cases.

Defines the two SWOPP3 routes (Trans-Atlantic and Trans-Pacific), their
fixed passage times, and the 366 daily departures throughout 2024.

The 8 SWOPP3 cases are:

========  ==============================  =============  ==========
Case      Route                           Direction      Passage (h)
========  ==============================  =============  ==========
1         Santander → New York            Westbound      354
2         New York → Santander            Eastbound      354
3         Santander → New York (no wx)    Westbound      354
4         New York → Santander (no wx)    Eastbound      354
5         Tokyo → Los Angeles             Eastbound      583
6         Los Angeles → Tokyo             Westbound      583
7         Tokyo → Los Angeles (no wx)     Eastbound      583
8         Los Angeles → Tokyo (no wx)     Westbound      583
========  ==============================  =============  ==========

Cases 3, 4, 7, 8 use the same routes but **without weather constraints**
(TWS/Hs limits are not enforced).

Example
-------
>>> from routetools.swopp3 import SWOPP3_CASES, departures_2024
>>> case = SWOPP3_CASES["case1"]
>>> deps = departures_2024()
>>> print(f"{case['name']}: {len(deps)} departures, {case['passage_hours']}h passage")
case1: 366 departures, 354h passage
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

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
    "case1": {
        "name": "ESSDR-USNYC",
        "label": "Santander → New York (weather constrained)",
        "src_port": "ESSDR",
        "dst_port": "USNYC",
        "passage_hours": 354,
        "weather_constrained": True,
    },
    "case2": {
        "name": "USNYC-ESSDR",
        "label": "New York → Santander (weather constrained)",
        "src_port": "USNYC",
        "dst_port": "ESSDR",
        "passage_hours": 354,
        "weather_constrained": True,
    },
    "case3": {
        "name": "ESSDR-USNYC-nowx",
        "label": "Santander → New York (no weather constraints)",
        "src_port": "ESSDR",
        "dst_port": "USNYC",
        "passage_hours": 354,
        "weather_constrained": False,
    },
    "case4": {
        "name": "USNYC-ESSDR-nowx",
        "label": "New York → Santander (no weather constraints)",
        "src_port": "USNYC",
        "dst_port": "ESSDR",
        "passage_hours": 354,
        "weather_constrained": False,
    },
    "case5": {
        "name": "JPTYO-USLAX",
        "label": "Tokyo → Los Angeles (weather constrained)",
        "src_port": "JPTYO",
        "dst_port": "USLAX",
        "passage_hours": 583,
        "weather_constrained": True,
    },
    "case6": {
        "name": "USLAX-JPTYO",
        "label": "Los Angeles → Tokyo (weather constrained)",
        "src_port": "USLAX",
        "dst_port": "JPTYO",
        "passage_hours": 583,
        "weather_constrained": True,
    },
    "case7": {
        "name": "JPTYO-USLAX-nowx",
        "label": "Tokyo → Los Angeles (no weather constraints)",
        "src_port": "JPTYO",
        "dst_port": "USLAX",
        "passage_hours": 583,
        "weather_constrained": False,
    },
    "case8": {
        "name": "USLAX-JPTYO-nowx",
        "label": "Los Angeles → Tokyo (no weather constraints)",
        "src_port": "USLAX",
        "dst_port": "JPTYO",
        "passage_hours": 583,
        "weather_constrained": False,
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
    start = datetime(_YEAR, 1, 1, _DEPARTURE_HOUR, tzinfo=timezone.utc)
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
        One of ``"case1"`` through ``"case8"``.

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
        One of ``"case1"`` through ``"case8"``.

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
