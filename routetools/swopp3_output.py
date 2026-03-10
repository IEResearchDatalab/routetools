"""SWOPP3 output formatters — File A (energy summary) and File B (tracks).

Produces CSV files in the exact format required by the SWOPP3 competition.

**File A** — one CSV per case, 366 rows (one per departure):

    departure_time_utc, arrival_time_utc, energy_cons_mwh,
    max_wind_mps, max_hs_m, sailed_distance_nm, details_filename

**File B** — one CSV per departure (referenced by ``details_filename``):

    time_utc, lat_deg, lon_deg

Naming convention::

    IEUniversity-{submission}-{casename}.csv      (File A)
    IEUniversity-{submission}-{casename}-{dep}.csv (File B)
"""

from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path

import jax.numpy as jnp

from routetools._cost.haversine import haversine_distance_from_curve

# Nautical mile in metres
_NM = 1852.0

# SWOPP3 datetime format
_DTFMT = "%Y-%m-%d %H:%M:%S"

# Team identifier
TEAM = "IEUniversity"

__all__ = [
    "TEAM",
    "file_a_row",
    "write_file_a",
    "write_file_b",
    "sailed_distance_nm",
    "waypoint_times",
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def sailed_distance_nm(curve: jnp.ndarray) -> float:
    """Total sailed distance in nautical miles.

    Parameters
    ----------
    curve : jnp.ndarray
        Shape ``(L, 2)`` with ``(lon, lat)`` in degrees.

    Returns
    -------
    float
        Distance in nautical miles.
    """
    segment_m = haversine_distance_from_curve(curve)
    return float(jnp.sum(segment_m)) / _NM


def waypoint_times(
    curve: jnp.ndarray,
    departure: datetime,
    passage_hours: float,
) -> list[datetime]:
    """Compute UTC timestamps at each waypoint assuming constant speed.

    Distributes waypoints uniformly in time over the passage duration.

    Parameters
    ----------
    curve : jnp.ndarray
        Shape ``(L, 2)`` waypoints.
    departure : datetime
        Departure time (UTC).
    passage_hours : float
        Total passage time in hours.

    Returns
    -------
    list[datetime]
        ``L`` UTC datetimes, one per waypoint.
    """
    L = curve.shape[0]
    if L < 2:
        return [departure]
    total_seconds = passage_hours * 3600.0
    return [
        departure + timedelta(seconds=total_seconds * i / (L - 1)) for i in range(L)
    ]


# ---------------------------------------------------------------------------
# File A — Energy summary
# ---------------------------------------------------------------------------
def file_a_row(
    departure: datetime,
    passage_hours: float,
    energy_mwh: float,
    max_wind_mps: float,
    max_hs_m: float,
    distance_nm: float,
    details_filename: str,
) -> dict[str, str]:
    """Build one row of File A as a dict.

    Parameters
    ----------
    departure : datetime
        Departure time UTC.
    passage_hours : float
        Passage time in hours.
    energy_mwh : float
        Total energy consumption in MWh.
    max_wind_mps : float
        Maximum true wind speed encountered (m/s).
    max_hs_m : float
        Maximum significant wave height encountered (m).
    distance_nm : float
        Sailed distance in nautical miles.
    details_filename : str
        Name of the corresponding File B CSV.

    Returns
    -------
    dict[str, str]
        Column-name → string-value mapping.
    """
    arrival = departure + timedelta(hours=passage_hours)
    return {
        "departure_time_utc": departure.strftime(_DTFMT),
        "arrival_time_utc": arrival.strftime(_DTFMT),
        "energy_cons_mwh": f"{energy_mwh:.6f}",
        "max_wind_mps": f"{max_wind_mps:.4f}",
        "max_hs_m": f"{max_hs_m:.4f}",
        "sailed_distance_nm": f"{distance_nm:.4f}",
        "details_filename": details_filename,
    }


_FILE_A_COLUMNS = [
    "departure_time_utc",
    "arrival_time_utc",
    "energy_cons_mwh",
    "max_wind_mps",
    "max_hs_m",
    "sailed_distance_nm",
    "details_filename",
]


def file_a_name(submission: int, casename: str) -> str:
    """Generate the File A filename.

    Parameters
    ----------
    submission : int
        Submission number (e.g. 1, 2, …).
    casename : str
        Case name (e.g. ``"AO_WPS"``).

    Returns
    -------
    str
        Filename like ``IEUniversity-1-AO_WPS.csv``.
    """
    return f"{TEAM}-{submission}-{casename}.csv"


def write_file_a(
    rows: list[dict[str, str]],
    path: str | Path,
) -> Path:
    """Write a File A CSV.

    Parameters
    ----------
    rows : list[dict]
        List of row dicts (from :func:`file_a_row`).
    path : str or Path
        Output file path.

    Returns
    -------
    Path
        The written path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FILE_A_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


# ---------------------------------------------------------------------------
# File B — Track coordinates
# ---------------------------------------------------------------------------
_FILE_B_COLUMNS = ["time_utc", "lat_deg", "lon_deg"]


def file_b_name(submission: int, casename: str, departure: datetime) -> str:
    """Generate the File B filename.

    Parameters
    ----------
    submission : int
        Submission number.
    casename : str
        Case name.
    departure : datetime
        Departure date.

    Returns
    -------
    str
        Filename like ``IEUniversity-1-AOWPS-20240101.csv``.
    """
    date_str = departure.strftime("%Y%m%d")
    return f"{TEAM}-{submission}-{casename}-{date_str}.csv"


def write_file_b(
    curve: jnp.ndarray,
    times: list[datetime],
    path: str | Path,
) -> Path:
    """Write a File B (track) CSV.

    Parameters
    ----------
    curve : jnp.ndarray
        Shape ``(L, 2)`` with ``(lon, lat)`` in degrees.
    times : list[datetime]
        ``L`` UTC timestamps (one per waypoint).
    path : str or Path
        Output file path.

    Returns
    -------
    Path
        The written path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    L = curve.shape[0]
    if len(times) != L:
        raise ValueError(f"Expected {L} times, got {len(times)}")

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FILE_B_COLUMNS)
        writer.writeheader()
        for i in range(L):
            writer.writerow(
                {
                    "time_utc": times[i].strftime(_DTFMT),
                    "lat_deg": f"{float(curve[i, 1]):.6f}",
                    "lon_deg": f"{float(curve[i, 0]):.6f}",
                }
            )
    return path
