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
from typing import TYPE_CHECKING

import jax.numpy as jnp

from routetools._cost.haversine import curve_distance_nm, waypoint_times_uniform

if TYPE_CHECKING:
    import pandas as pd

# SWOPP3 datetime format
_DTFMT = "%Y-%m-%d %H:%M:%S"

# Team identifier
TEAM = "IEUniversity"

__all__ = [
    "TEAM",
    "file_a_row",
    "resolve_file_a_path",
    "resolve_file_b_path",
    "read_file_a_dataframe",
    "read_file_b_dataframe",
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
    return curve_distance_nm(curve)


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
        Shape ``(L, 2)`` waypoints in ``(lon, lat)`` order.
    departure : datetime
        Departure time (UTC).
    passage_hours : float
        Total passage time in hours.

    Returns
    -------
    list[datetime]
        ``L`` UTC datetimes, one per waypoint.

    Raises
    ------
    ValueError
        If *curve* has no waypoints.
    """
    return waypoint_times_uniform(curve, departure, passage_hours)


def resolve_file_a_path(
    input_dir: str | Path,
    casename: str,
    submission: int | None = None,
) -> Path:
    """Resolve File A path for a case.

    When *submission* is ``None``, returns the latest submission for the case.
    """
    input_dir = Path(input_dir)
    if submission is None:
        pattern = f"{TEAM}-*-{casename}.csv"
        candidates = sorted(input_dir.glob(pattern))
        if not candidates:
            raise FileNotFoundError(
                f"No summary CSV found for case '{casename}' in '{input_dir}'. "
                f"Tried pattern '{pattern}'."
            )

        def _submission_key(path: Path) -> int:
            parts = path.stem.split("-")
            if len(parts) >= 3:
                try:
                    return int(parts[1])
                except ValueError:
                    return -1
            return -1

        candidates.sort(key=lambda path: (_submission_key(path), path.name))
        return candidates[-1]

    path = input_dir / file_a_name(submission, casename)
    if not path.exists():
        raise FileNotFoundError(f"Summary CSV not found: {path}")
    return path


def resolve_file_b_path(
    input_dir: str | Path,
    filename: str,
) -> Path:
    """Resolve File B path from details filename."""
    path = Path(input_dir) / "tracks" / filename
    if not path.exists():
        raise FileNotFoundError(f"Track CSV not found: {path}")
    return path


def read_file_a_dataframe(
    input_dir: str | Path,
    casename: str,
    submission: int | None = None,
) -> pd.DataFrame:
    """Read a File A CSV into a pandas DataFrame."""
    import pandas as pd

    path = resolve_file_a_path(input_dir, casename, submission=submission)
    return pd.read_csv(path, parse_dates=["departure_time_utc", "arrival_time_utc"])


def read_file_b_dataframe(
    input_dir: str | Path,
    filename: str,
) -> pd.DataFrame:
    """Read a File B CSV into a pandas DataFrame."""
    import pandas as pd

    path = resolve_file_b_path(input_dir, filename)
    return pd.read_csv(path, parse_dates=["time_utc"])


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
        Case name (e.g. ``"AO_WPS"``).
    departure : datetime
        Departure date.

    Returns
    -------
    str
        Filename like ``IEUniversity-1-AO_WPS-20240101.csv``.
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
