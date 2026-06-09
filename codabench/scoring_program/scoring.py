#!/usr/bin/env python
"""CodaBench scoring program for the SWOPP3 Weather Routing Benchmark.

CodaBench invokes this script with two arguments:
    python scoring.py <input_dir> <output_dir>

Where:
    <input_dir>/ref/   — reference data (ERA5 NetCDF files + land mask)
    <input_dir>/res/   — participant submission (unzipped)
    <output_dir>/      — write scores.json here

The scoring program:
1. Validates submission structure (File A and File B CSVs).
2. Validates endpoint positions and timestamps against expected ports.
3. Checks that waypoints do not cross land (using Natural Earth shapefile).
4. Checks operational constraints (Hs < 7 m, TWS < 20 m/s).
5. Validates passage time consistency.
6. Re-evaluates energy using the RISE performance model with official ERA5 data
   (when ERA5 files are present in reference_data).
7. Writes scores.json with per-case and total energy metrics.
"""

from __future__ import annotations

import base64
import csv
import gc
import html as html_mod
import io
import json
import math
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

# ─── CodaBench entry point ───────────────────────────────────────────

if len(sys.argv) < 3:
    print(f"Usage: {sys.argv[0]} <input_dir> <output_dir>", file=sys.stderr)
    sys.exit(1)

input_dir = Path(sys.argv[1])
output_dir = Path(sys.argv[2])

submission_dir = input_dir / "res"
reference_dir = input_dir / "ref"

# Worker-local data: /codabench/data on the host is mounted as /app/data
# inside the scoring container. Check there first for ERA5 + shapefile.
_WORKER_DATA = Path("/app/data")
data_dir = _WORKER_DATA if _WORKER_DATA.is_dir() else reference_dir

output_dir.mkdir(parents=True, exist_ok=True)

# ─── Configuration ───────────────────────────────────────────────────

EXPECTED_DEPARTURES = 366
DTFMT = "%Y-%m-%d %H:%M:%S"

# Operational limits (SWOPP3)
MAX_WIND_MPS = 20.0  # True wind speed limit (m/s)
MAX_HS_M = 7.0  # Significant wave height limit (m)

# Endpoint tolerance (degrees) — allow small rounding differences
ENDPOINT_TOLERANCE_DEG = 0.5

# Passage time tolerance (hours)
PASSAGE_TIME_TOLERANCE_H = 1.0

# Resampling interval for energy integration (minutes)
# Decouples waypoint density (Δt₁) from integration accuracy (Δt₂).
RESAMPLE_DT_MINUTES = 30

# Interpolation order for ERA5 queries (1=trilinear, 3=tricubic)
INTERP_ORDER = 3

CASE_NAMES = [
    "AO_WPS",
    "AO_noWPS",
    "AGC_WPS",
    "AGC_noWPS",
    "PO_WPS",
    "PO_noWPS",
    "PGC_WPS",
    "PGC_noWPS",
]

GC_CASES = {"AGC_WPS", "AGC_noWPS", "PGC_WPS", "PGC_noWPS"}

# Case definitions: expected ports, passage times, and route corridor
CASE_DEFS = {
    "AO_WPS": {
        "src": (43.6, -4.0),
        "dst": (40.53, -73.80),
        "passage_h": 354,
        "route": "atlantic",
        "wps": True,
    },
    "AO_noWPS": {
        "src": (43.6, -4.0),
        "dst": (40.53, -73.80),
        "passage_h": 354,
        "route": "atlantic",
        "wps": False,
    },
    "AGC_WPS": {
        "src": (43.6, -4.0),
        "dst": (40.53, -73.80),
        "passage_h": 354,
        "route": "atlantic",
        "wps": True,
    },
    "AGC_noWPS": {
        "src": (43.6, -4.0),
        "dst": (40.53, -73.80),
        "passage_h": 354,
        "route": "atlantic",
        "wps": False,
    },
    "PO_WPS": {
        "src": (34.8, 140.0),
        "dst": (34.4, -121.0),
        "passage_h": 583,
        "route": "pacific",
        "wps": True,
    },
    "PO_noWPS": {
        "src": (34.8, 140.0),
        "dst": (34.4, -121.0),
        "passage_h": 583,
        "route": "pacific",
        "wps": False,
    },
    "PGC_WPS": {
        "src": (34.8, 140.0),
        "dst": (34.4, -121.0),
        "passage_h": 583,
        "route": "pacific",
        "wps": True,
    },
    "PGC_noWPS": {
        "src": (34.8, 140.0),
        "dst": (34.4, -121.0),
        "passage_h": 583,
        "route": "pacific",
        "wps": False,
    },
}

# Expected departure schedule: 366 days of 2024, noon UTC
EXPECTED_DEPARTURES_LIST = [
    datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC) + timedelta(days=d) for d in range(366)
]

FILE_A_COLUMNS = [
    "departure_time_utc",
    "arrival_time_utc",
    "energy_cons_mwh",
    "max_wind_mps",
    "max_hs_m",
    "sailed_distance_nm",
    "details_filename",
]
FILE_B_COLUMNS = ["time_utc", "lat_deg", "lon_deg"]


# ─── Helpers ─────────────────────────────────────────────────────────


def _coord_distance_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Euclidean distance in degrees with antimeridian-safe longitude delta."""
    dlon = (lon1 - lon2 + 180) % 360 - 180  # wrapped to [-180, 180]
    return math.sqrt((lat1 - lat2) ** 2 + dlon**2)


def _slerp(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    f: float,
) -> tuple[float, float]:
    """Spherical linear interpolation at fraction *f* in [0, 1].

    Parameters and return values are in degrees.
    """
    phi1, lam1 = math.radians(lat1), math.radians(lon1)
    phi2, lam2 = math.radians(lat2), math.radians(lon2)

    p1 = np.array(
        [
            math.cos(phi1) * math.cos(lam1),
            math.cos(phi1) * math.sin(lam1),
            math.sin(phi1),
        ]
    )
    p2 = np.array(
        [
            math.cos(phi2) * math.cos(lam2),
            math.cos(phi2) * math.sin(lam2),
            math.sin(phi2),
        ]
    )

    dot = float(np.clip(np.dot(p1, p2), -1.0, 1.0))
    sigma = math.acos(dot)

    if sigma < 1e-12:
        lat = lat1 + f * (lat2 - lat1)
        lon = lon1 + f * ((lon2 - lon1 + 180) % 360 - 180)
        return lat, lon

    a = math.sin((1 - f) * sigma) / math.sin(sigma)
    b = math.sin(f * sigma) / math.sin(sigma)
    p = a * p1 + b * p2

    lat = math.degrees(math.atan2(p[2], math.sqrt(p[0] ** 2 + p[1] ** 2)))
    lon = math.degrees(math.atan2(p[1], p[0]))
    return lat, lon


def resample_track(
    waypoints: list[tuple[datetime, float, float]],
    dt_minutes: float = RESAMPLE_DT_MINUTES,
) -> list[tuple[datetime, float, float]]:
    """Resample a track to uniform temporal spacing via geodesic interpolation.

    Parameters
    ----------
    waypoints
        List of ``(time, lat_deg, lon_deg)`` — the original track.
    dt_minutes
        Target sub-segment interval in minutes.

    Returns
    -------
    list[tuple[datetime, float, float]]
        Resampled waypoints at approximately uniform Δt₂.
        First and last points are preserved exactly.
    """
    if len(waypoints) < 2:
        return list(waypoints)

    result: list[tuple[datetime, float, float]] = []

    for i in range(len(waypoints) - 1):
        t0, lat0, lon0 = waypoints[i]
        t1, lat1, lon1 = waypoints[i + 1]

        seg_seconds = (t1 - t0).total_seconds()
        if seg_seconds <= 0:
            result.append((t0, lat0, lon0))
            continue

        dt_seconds = dt_minutes * 60.0
        n_sub = max(1, math.ceil(seg_seconds / dt_seconds))

        for j in range(n_sub):
            f = j / n_sub
            t = t0 + timedelta(seconds=f * seg_seconds)
            lat, lon = _slerp(lat0, lon0, lat1, lon1, f)
            result.append((t, lat, lon))

    result.append(waypoints[-1])
    return result


def find_team_prefix(sub_dir: Path) -> str | None:
    """Detect the team name prefix from submitted CSV files."""
    for f in sub_dir.glob("*.csv"):
        for case in CASE_NAMES:
            if f.name.endswith(f"-{case}.csv"):
                return f.name[: -(len(case) + 5)]  # strip "-{case}.csv"
    return None


# ─── Land checking (shapefile-based, no JAX needed) ──────────────────


def _load_land_checker(ref_dir: Path):
    """Load a land-checking function from Natural Earth shapefile in ref_dir.

    Returns a callable (lat, lon) -> bool, or None if shapefile not found.
    """
    shapefile = ref_dir / "ne_10m_land.shp"
    if not shapefile.exists():
        # Try alternative locations
        for candidate in ref_dir.glob("**/ne_*_land.shp"):
            shapefile = candidate
            break
        else:
            return None

    try:
        import shapefile as shp
        from shapely.geometry import Point, shape

        sf = shp.Reader(str(shapefile))
        land_shapes = [shape(s) for s in sf.shapes()]

        # Build a union for efficient queries
        from shapely.ops import unary_union

        land_union = unary_union(land_shapes)

        def is_on_land(lat: float, lon: float) -> bool:
            return land_union.contains(Point(lon, lat))

        return is_on_land
    except ImportError:
        return None


# ─── Validation: File A ─────────────────────────────────────────────


def validate_file_a(
    path: Path,
    case_id: str,
    expected_rows: int = 366,
) -> tuple[list[str], list[dict]]:
    """Validate a File A CSV. Returns (errors, parsed_rows)."""
    errors: list[str] = []
    fname = path.name
    case_def = CASE_DEFS[case_id]
    passage_h = case_def["passage_h"]

    if not path.exists():
        errors.append(f"{fname}: File not found")
        return errors, []

    with path.open() as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            errors.append(f"{fname}: No header row")
            return errors, []

        missing = set(FILE_A_COLUMNS) - set(reader.fieldnames)
        if missing:
            errors.append(f"{fname}: Missing columns: {missing}")

        rows = list(reader)

    if len(rows) != expected_rows:
        errors.append(f"{fname}: Expected {expected_rows} rows, got {len(rows)}")

    for i, row in enumerate(rows, 1):
        # ── Datetime parsing ──
        dep_dt = arr_dt = None
        for col in ("departure_time_utc", "arrival_time_utc"):
            try:
                dt = datetime.strptime(row.get(col, ""), DTFMT)
                if col == "departure_time_utc":
                    dep_dt = dt
                else:
                    arr_dt = dt
            except ValueError:
                errors.append(f"{fname} row {i}: Bad datetime in '{col}'")

        # ── Passage time check ──
        if dep_dt and arr_dt:
            actual_h = (arr_dt - dep_dt).total_seconds() / 3600
            if abs(actual_h - passage_h) > PASSAGE_TIME_TOLERANCE_H:
                errors.append(
                    f"{fname} row {i}: Passage time {actual_h:.1f}h != "
                    f"expected {passage_h}h"
                )

        # ── Departure time must match official schedule ──
        if dep_dt and i <= len(EXPECTED_DEPARTURES_LIST):
            expected_dep = EXPECTED_DEPARTURES_LIST[i - 1]
            expected_naive = expected_dep.replace(tzinfo=None)
            if dep_dt != expected_naive:
                errors.append(
                    f"{fname} row {i}: Departure {dep_dt} != "
                    f"expected {expected_naive}"
                )

        # ── Numeric fields ──
        for col in (
            "energy_cons_mwh",
            "max_wind_mps",
            "max_hs_m",
            "sailed_distance_nm",
        ):
            try:
                v = float(row.get(col, ""))
                if v != v:  # NaN
                    errors.append(f"{fname} row {i}: NaN in '{col}'")
                if v < 0:
                    errors.append(f"{fname} row {i}: Negative value in '{col}'")
            except ValueError:
                errors.append(f"{fname} row {i}: Non-numeric '{col}'")

        # ── Operational constraints (skipped for GC cases) ──
        if case_id not in GC_CASES:
            try:
                wind = float(row.get("max_wind_mps", "0"))
                if wind > MAX_WIND_MPS:
                    errors.append(
                        f"{fname} row {i}: max_wind_mps={wind:.1f} exceeds "
                        f"limit {MAX_WIND_MPS} m/s"
                    )
            except ValueError:
                pass

            try:
                hs = float(row.get("max_hs_m", "0"))
                if hs > MAX_HS_M:
                    errors.append(
                        f"{fname} row {i}: max_hs_m={hs:.1f} exceeds "
                        f"limit {MAX_HS_M} m"
                    )
            except ValueError:
                pass

    return errors, rows


# ─── Validation: File B ──────────────────────────────────────────────


def validate_file_b(
    path: Path,
    case_id: str,
    departure_dt: datetime | None = None,
    arrival_dt: datetime | None = None,
    land_checker=None,
) -> list[str]:
    """Validate a File B (track waypoints) CSV."""
    errors: list[str] = []
    fname = path.name
    case_def = CASE_DEFS[case_id]

    if not path.exists():
        errors.append(f"{fname}: File not found")
        return errors

    with path.open() as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            errors.append(f"{fname}: No header row")
            return errors

        missing = set(FILE_B_COLUMNS) - set(reader.fieldnames)
        if missing:
            errors.append(f"{fname}: Missing columns: {missing}")

        rows = list(reader)

    if len(rows) < 2:
        errors.append(f"{fname}: Less than 2 waypoints")
        return errors

    # Parse all waypoints
    waypoints: list[tuple[datetime | None, float | None, float | None]] = []
    prev_time = None
    for i, row in enumerate(rows, 1):
        t = lat = lon = None
        try:
            t = datetime.strptime(row.get("time_utc", ""), DTFMT)
            if prev_time is not None and t <= prev_time:
                errors.append(f"{fname} row {i}: Timestamps not strictly increasing")
            prev_time = t
        except ValueError:
            errors.append(f"{fname} row {i}: Bad time_utc")

        for col, lo, hi in [("lat_deg", -90, 90), ("lon_deg", -360, 360)]:
            try:
                v = float(row.get(col, ""))
                if v < lo or v > hi:
                    errors.append(f"{fname} row {i}: '{col}'={v} out of [{lo},{hi}]")
                if col == "lat_deg":
                    lat = v
                else:
                    lon = v
            except ValueError:
                errors.append(f"{fname} row {i}: Non-numeric '{col}'")

        waypoints.append((t, lat, lon))

    if not waypoints:
        return errors

    # ── Endpoint position checks ──
    src_lat, src_lon = case_def["src"]
    dst_lat, dst_lon = case_def["dst"]

    first_lat, first_lon = waypoints[0][1], waypoints[0][2]
    last_lat, last_lon = waypoints[-1][1], waypoints[-1][2]

    if first_lat is not None and first_lon is not None:
        dist = _coord_distance_deg(first_lat, first_lon, src_lat, src_lon)
        if dist > ENDPOINT_TOLERANCE_DEG:
            errors.append(
                f"{fname}: Start ({first_lat:.2f}, {first_lon:.2f}) too far "
                f"from expected port ({src_lat}, {src_lon}), dist={dist:.2f}°"
            )

    if last_lat is not None and last_lon is not None:
        dist = _coord_distance_deg(last_lat, last_lon, dst_lat, dst_lon)
        if dist > ENDPOINT_TOLERANCE_DEG:
            errors.append(
                f"{fname}: End ({last_lat:.2f}, {last_lon:.2f}) too far "
                f"from expected port ({dst_lat}, {dst_lon}), dist={dist:.2f}°"
            )

    # ── Endpoint time checks ──
    first_time = waypoints[0][0]
    last_time = waypoints[-1][0]

    if departure_dt and first_time and first_time != departure_dt:
        errors.append(
            f"{fname}: First waypoint time {first_time} != " f"departure {departure_dt}"
        )

    if arrival_dt and last_time and last_time != arrival_dt:
        errors.append(
            f"{fname}: Last waypoint time {last_time} != arrival {arrival_dt}"
        )

    # ── Land crossing checks (skipped for GC cases) ──
    if land_checker is not None and case_id not in GC_CASES:
        land_rows: list[int] = []
        for idx, (_, lat, lon) in enumerate(waypoints):
            if lat is not None and lon is not None and land_checker(lat, lon):
                land_rows.append(idx + 1)  # 1-based row number
        if land_rows:
            # Show up to 10 row numbers to keep the message readable
            shown = land_rows[:10]
            suffix = f", ... ({len(land_rows)} total)" if len(land_rows) > 10 else ""
            rows_str = ", ".join(str(r) for r in shown) + suffix
            errors.append(
                f"{fname}: {len(land_rows)} waypoint(s) on land "
                f"at row(s) {rows_str}"
            )

    return errors


# ─── ERA5 loading (self-contained, numpy + netCDF4 only) ─────────────


def _load_era5_grid(nc_paths: list[str]) -> dict | None:
    """Load and concatenate ERA5 NetCDF files into a numpy dict.

    Returns dict with keys ``data`` (dict of variable arrays),
    ``lat``, ``lon``, ``times_h`` (1-D coordinate arrays), and
    ``t0`` (numpy datetime64 of first timestamp).
    """
    try:
        import netCDF4
    except ImportError:
        return None

    datasets = []
    for p in nc_paths:
        ds = netCDF4.Dataset(p, "r")
        datasets.append(ds)

    if not datasets:
        return None

    ds0 = datasets[0]

    # Detect coordinate names
    lat_name = "latitude" if "latitude" in ds0.variables else "lat"
    lon_name = "longitude" if "longitude" in ds0.variables else "lon"
    time_name = "valid_time" if "valid_time" in ds0.variables else "time"

    lat = np.array(ds0.variables[lat_name][:], dtype=np.float64)
    lon = np.array(ds0.variables[lon_name][:], dtype=np.float64)

    # Detect data variable names (everything except coordinates)
    coord_names = {lat_name, lon_name, time_name}
    var_names = [v for v in ds0.variables if v not in coord_names]

    # Mapping from long ERA5 variable names to short names
    LONG_TO_SHORT = {
        "10m_u_component_of_wind": "u10",
        "10m_v_component_of_wind": "v10",
        "significant_height_of_combined_wind_waves_and_swell": "swh",
        "mean_wave_direction": "mwd",
    }

    # Concatenate along time across files
    all_times = []
    all_data = {LONG_TO_SHORT.get(v, v): [] for v in var_names}
    for ds in datasets:
        t_name = "valid_time" if "valid_time" in ds.variables else "time"
        t_var = ds.variables[t_name]
        cal = getattr(t_var, "calendar", "standard")
        times = netCDF4.num2date(t_var[:], t_var.units, cal)
        all_times.extend(times)
        for v in var_names:
            short = LONG_TO_SHORT.get(v, v)
            all_data[short].append(np.array(ds.variables[v][:], dtype=np.float32))

    for ds in datasets:
        ds.close()

    # Build time array in hours since first timestamp
    t0 = np.datetime64(all_times[0])
    times_np = np.array([np.datetime64(t) for t in all_times])
    times_h = (times_np - t0) / np.timedelta64(1, "h")
    times_h = times_h.astype(np.float64)

    # Concatenate data arrays along time (axis 0)
    data = {}
    for v in all_data:
        arr = np.concatenate(all_data[v], axis=0)
        # Replace NaN with 0 (ERA5 wave fields are NaN over land)
        np.nan_to_num(arr, copy=False, nan=0.0)
        data[v] = arr

    # Ensure ascending latitude
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        for v in data:
            data[v] = data[v][:, ::-1, :]

    # Ensure ascending longitude
    if lon[0] > lon[-1]:
        lon = lon[::-1]
        for v in data:
            data[v] = data[v][:, :, ::-1]

    return {
        "data": data,
        "lat": lat,
        "lon": lon,
        "times_h": times_h,
        "t0": t0,
    }


def _interp_era5(
    grid: dict,
    var_name: str,
    query_lat: np.ndarray,
    query_lon: np.ndarray,
    query_t_h: np.ndarray,
    order: int = INTERP_ORDER,
) -> np.ndarray:
    """Interpolate an ERA5 variable at query points.

    Parameters
    ----------
    grid : dict from ``_load_era5_grid``
    var_name : variable name in the grid
    query_lat, query_lon : arrays of shape (N,) in degrees
    query_t_h : array of shape (N,) — hours since grid t0
    order : interpolation order (1=trilinear, 3=tricubic)

    Returns array of shape (N,).
    """
    arr = grid["data"][var_name]  # (T, lat, lon)
    lat = grid["lat"]
    lon = grid["lon"]
    times_h = grid["times_h"]

    dt = times_h[1] - times_h[0] if len(times_h) > 1 else 1.0
    dlat = lat[1] - lat[0] if len(lat) > 1 else 1.0
    dlon = lon[1] - lon[0] if len(lon) > 1 else 1.0

    # Fractional indices
    fi_t = (query_t_h - times_h[0]) / dt
    fi_lat = (query_lat - lat[0]) / dlat
    fi_lon = (query_lon - lon[0]) / dlon

    # Clamp to valid range
    fi_t = np.clip(fi_t, 0, len(times_h) - 1)
    fi_lat = np.clip(fi_lat, 0, len(lat) - 1)
    fi_lon = np.clip(fi_lon, 0, len(lon) - 1)

    if order > 1:
        from scipy.ndimage import map_coordinates

        coords = np.array([fi_t, fi_lat, fi_lon])
        return map_coordinates(arr, coords, order=order, mode="nearest").astype(
            np.float64
        )

    # Fallback: manual trilinear (order=1)
    i0_t = np.clip(np.floor(fi_t).astype(int), 0, len(times_h) - 2)
    i0_lat = np.clip(np.floor(fi_lat).astype(int), 0, len(lat) - 2)
    i0_lon = np.clip(np.floor(fi_lon).astype(int), 0, len(lon) - 2)

    wt = (fi_t - i0_t).astype(np.float32)
    wlat = (fi_lat - i0_lat).astype(np.float32)
    wlon = (fi_lon - i0_lon).astype(np.float32)

    result = np.zeros(len(query_lat), dtype=np.float32)
    for dt_off in (0, 1):
        for dlat_off in (0, 1):
            for dlon_off in (0, 1):
                w = (
                    ((1 - wt) if dt_off == 0 else wt)
                    * ((1 - wlat) if dlat_off == 0 else wlat)
                    * ((1 - wlon) if dlon_off == 0 else wlon)
                )
                it = np.clip(i0_t + dt_off, 0, arr.shape[0] - 1)
                ila = np.clip(i0_lat + dlat_off, 0, arr.shape[1] - 1)
                ilo = np.clip(i0_lon + dlon_off, 0, arr.shape[2] - 1)
                result += w * arr[it, ila, ilo]

    return result.astype(np.float64)


# ─── RISE performance model (self-contained, numpy only) ─────────────

# Constants
_KH = 969.0 / 226.0  # Hull resistance
_KA = 49.0 / 320.0  # Aerodynamic drag
_AW = 11.1395  # Wave added resistance amplitude
_KW = 125.0 / 432.0  # Wave added resistance decay
_KS = 27489.0 / 32000.0  # Wingsail thrust
_DEAD_ZONE_DEG = 10.0  # Wingsail dead zone


def _rise_power(tws, twa_deg, swh, mwa_deg, v, wps):
    """Compute RISE power (kW) for arrays of segment values.

    All inputs are numpy arrays of shape (N,).
    Returns power array of shape (N,).
    """
    twa_rad = np.radians(twa_deg)

    # Hull resistance
    p_hull = _KH * v**3

    # Apparent wind
    ux = tws * np.cos(twa_rad) + v
    uy = tws * np.sin(twa_rad)
    vr = np.sqrt(ux**2 + uy**2)

    # Aerodynamic drag
    p_wind = _KA * v * (vr * ux - v**2)

    # Wave added resistance
    mwa_rad = np.radians(mwa_deg)
    p_wave = _AW * swh**2 * v**1.5 * np.exp(-_KW * np.abs(mwa_rad) ** 3)

    power = p_hull + p_wind + p_wave

    # Wingsail thrust (WPS only)
    if wps:
        awa_deg = np.degrees(np.arctan2(np.abs(uy), ux))
        sail_active = awa_deg >= _DEAD_ZONE_DEG
        alpha = np.where(sail_active, np.radians(awa_deg - _DEAD_ZONE_DEG), 0.0)
        sin_a = np.sin(alpha)
        p_sail = _KS * sin_a * (1.0 + 0.15 * sin_a**2) * vr**2 * v
        power = power - np.where(sail_active, p_sail, 0.0)

    return np.maximum(power, 0.0)


def _haversine_m(lat1, lon1, lat2, lon2):
    """Haversine distance in metres between arrays of points."""
    R = 6_371_000.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    lat1r = np.radians(lat1)
    lat2r = np.radians(lat2)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def _forward_bearing_deg(lat1, lon1, lat2, lon2):
    """Forward bearing in degrees [0, 360) between arrays of points."""
    lat1r, lat2r = np.radians(lat1), np.radians(lat2)
    dlon = np.radians(lon2 - lon1)
    x = np.sin(dlon) * np.cos(lat2r)
    y = np.cos(lat1r) * np.sin(lat2r) - np.sin(lat1r) * np.cos(lat2r) * np.cos(dlon)
    return np.mod(np.degrees(np.arctan2(x, y)), 360.0)


# ─── ERA5 re-evaluation (self-contained) ─────────────────────────────


def try_load_era5_scorer(ref_dir: Path):
    """Load ERA5 data from reference directory and return a scoring function.

    Returns a callable (case_id, waypoints, departure_str) -> energy_mwh,
    or None if ERA5 files are missing or netCDF4 is not installed.

    Data is loaded lazily per corridor to limit peak memory usage. Only
    one corridor's grids are kept in memory at a time.
    """
    # Discover file paths without loading data
    corridor_files: dict[str, dict[str, list[str]]] = {}

    for corridor in ("atlantic", "pacific"):
        wind_2024 = ref_dir / f"era5_wind_{corridor}_2024.nc"
        waves_2024 = ref_dir / f"era5_waves_{corridor}_2024.nc"
        if not (wind_2024.exists() and waves_2024.exists()):
            continue

        wind_files = [str(wind_2024)]
        wave_files = [str(waves_2024)]
        for suffix in (
            f"era5_wind_{corridor}_2025_01.nc",
            f"era5_wind_{corridor}_2025.nc",
        ):
            p = ref_dir / suffix
            if p.exists():
                wind_files.append(str(p))
                break
        for suffix in (
            f"era5_waves_{corridor}_2025_01.nc",
            f"era5_waves_{corridor}_2025.nc",
        ):
            p = ref_dir / suffix
            if p.exists():
                wave_files.append(str(p))
                break

        corridor_files[corridor] = {"wind": wind_files, "waves": wave_files}

    if not corridor_files:
        return None

    # Mutable cache: holds at most one corridor's grids at a time
    _cache: dict[str, dict | None] = {"corridor": None, "wind": None, "waves": None}

    def _get_grids(corridor: str) -> tuple[dict, dict] | None:
        """Return (wind_grid, wave_grid) for *corridor*, loading lazily."""
        if _cache["corridor"] != corridor:
            # Release previous corridor data
            _cache["wind"] = None
            _cache["waves"] = None
            gc.collect()
            _cache["corridor"] = corridor
            files = corridor_files.get(corridor)
            if files is None:
                return None
            _cache["wind"] = _load_era5_grid(files["wind"])
            _cache["waves"] = _load_era5_grid(files["waves"])
        wind = _cache["wind"]
        waves = _cache["waves"]
        if wind is None or waves is None:
            return None
        return wind, waves

    def evaluate_route(
        case_id: str,
        waypoints: list[tuple[datetime, float, float]],
        departure_str: str,
    ) -> tuple[float, int, int] | None:
        """Re-evaluate a single route using ERA5 + RISE trapezoidal rule.

        Parameters
        ----------
        case_id : one of the CASE_NAMES
        waypoints : list of (datetime, lat_deg, lon_deg) tuples
            Should already be resampled to uniform Δt₂.
        departure_str : departure datetime as string

        Returns
        -------
        tuple[float, int, int] | None
            ``(energy_mwh, wind_violations, wave_violations)`` or None if
            evaluation is not possible.  Violation counts are the number
            of evaluation points where TWS > 20 m/s or Hs > 7 m.
        """
        case_def = CASE_DEFS[case_id]
        corridor = case_def["route"]
        if corridor not in corridor_files:
            return None

        grids = _get_grids(corridor)
        if grids is None:
            return None
        wind_grid, wave_grid = grids

        n_wp = len(waypoints)
        if n_wp < 2:
            return None

        lats = np.array([wp[1] for wp in waypoints])
        lons = np.array([wp[2] for wp in waypoints])

        # Compute per-segment dt from actual waypoint timestamps
        wp_times = np.array(
            [np.datetime64(wp[0]) for wp in waypoints], dtype="datetime64[s]"
        )
        seg_dt_h = ((wp_times[1:] - wp_times[:-1]) / np.timedelta64(1, "h")).astype(
            np.float64
        )
        seg_dt_h = np.maximum(seg_dt_h, 1e-6)

        # Normalize lons to match ERA5 grid convention [0, 360)
        grid_lon = wind_grid["lon"]
        if grid_lon[0] >= 0 and grid_lon[-1] > 180:
            lons = np.where(lons < 0, lons + 360, lons)

        # ── Trapezoidal rule: evaluate weather at ALL waypoints ──
        dep_dt64 = wp_times[0]
        dep_offset_h = float((dep_dt64 - wind_grid["t0"]) / np.timedelta64(1, "h"))
        wp_times_h = np.zeros(n_wp, dtype=np.float64)
        wp_times_h[0] = dep_offset_h
        wp_times_h[1:] = dep_offset_h + np.cumsum(seg_dt_h)

        # Interpolate weather at every waypoint (not just midpoints)
        u10 = _interp_era5(wind_grid, "u10", lats, lons, wp_times_h)
        v10 = _interp_era5(wind_grid, "v10", lats, lons, wp_times_h)
        swh = _interp_era5(wave_grid, "swh", lats, lons, wp_times_h)
        mwd = _interp_era5(wave_grid, "mwd", lats, lons, wp_times_h)

        # Weather at all waypoints for constraint checking
        tws_all = np.sqrt(u10**2 + v10**2)
        wind_violations = int(np.sum(tws_all > MAX_WIND_MPS))
        wave_violations = int(np.sum(swh > MAX_HS_M))

        # Ship speed and bearing per segment
        seg_dist_m = _haversine_m(lats[:-1], lons[:-1], lats[1:], lons[1:])
        v_mps = seg_dist_m / (seg_dt_h * 3600.0)
        bearing_deg = _forward_bearing_deg(lats[:-1], lons[:-1], lats[1:], lons[1:])

        # ── RISE power at segment start points ──
        tws_start = tws_all[:-1]
        wind_from_start = np.mod(
            180.0 + np.degrees(np.arctan2(u10[:-1], v10[:-1])), 360.0
        )
        twa_start = np.mod(wind_from_start - bearing_deg, 360.0)
        mwa_start = np.mod(mwd[:-1] - bearing_deg, 360.0)
        power_start = _rise_power(
            tws_start, twa_start, swh[:-1], mwa_start, v_mps, case_def["wps"]
        )

        # ── RISE power at segment end points ──
        tws_end = tws_all[1:]
        wind_from_end = np.mod(180.0 + np.degrees(np.arctan2(u10[1:], v10[1:])), 360.0)
        twa_end = np.mod(wind_from_end - bearing_deg, 360.0)
        mwa_end = np.mod(mwd[1:] - bearing_deg, 360.0)
        power_end = _rise_power(
            tws_end, twa_end, swh[1:], mwa_end, v_mps, case_def["wps"]
        )

        # ── Trapezoidal average ──
        # E_i = (P_start + P_end) / 2 * dt
        power_avg = (power_start + power_end) / 2.0
        energy_mwh = float(np.sum(power_avg * seg_dt_h) / 1000.0)
        return energy_mwh, wind_violations, wave_violations

    return evaluate_route


# ─── Plot generation (matplotlib, optional) ──────────────────────────


def _try_import_matplotlib():
    """Import matplotlib with Agg backend. Returns (plt, True) or (None, False)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt, True
    except ImportError:
        return None, False


def _fig_to_base64(fig) -> str:
    """Render a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("ascii")
    buf.close()
    return encoded


def _generate_energy_timeseries(plt, case_energies: dict, corridor: str) -> str | None:
    """Generate an energy timeseries plot for one corridor.

    Parameters
    ----------
    plt : matplotlib.pyplot module
    case_energies : dict mapping case_id -> list of (departure_date, energy_mwh)
    corridor : "atlantic" or "pacific"

    Returns base64 PNG string, or None if no data.
    """
    prefix = "A" if corridor == "atlantic" else "P"
    cases = [
        f"{prefix}O_WPS",
        f"{prefix}O_noWPS",
        f"{prefix}GC_WPS",
        f"{prefix}GC_noWPS",
    ]
    labels = ["Opt WPS", "Opt noWPS", "GC WPS", "GC noWPS"]
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#F44336"]

    has_data = any(len(case_energies.get(c, [])) > 0 for c in cases)
    if not has_data:
        return None

    fig, ax = plt.subplots(figsize=(14, 5))
    for case_id, label, color in zip(cases, labels, colors, strict=False):
        data = case_energies.get(case_id, [])
        if not data:
            continue
        dates = [d for d, _ in data]
        energies = [e for _, e in data]
        ax.plot(dates, energies, linewidth=0.8, alpha=0.85, label=label, color=color)

    corridor_title = "Trans-Atlantic" if corridor == "atlantic" else "Trans-Pacific"
    ax.set_title(f"Energy per Departure — {corridor_title}", fontsize=13)
    ax.set_xlabel("Departure Date")
    ax.set_ylabel("Energy (MWh)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    result = _fig_to_base64(fig)
    plt.close(fig)
    return result


def _generate_route_spaghetti(
    plt, case_routes: dict, corridor: str, wps: bool, land_polygons=None
) -> str | None:
    """Generate a route spaghetti plot for one corridor/WPS combination.

    Parameters
    ----------
    plt : matplotlib.pyplot module
    case_routes : dict mapping case_id -> list of (lats_array, lons_array)
    corridor : "atlantic" or "pacific"
    wps : True for WPS cases, False for noWPS
    land_polygons : list of shapely geometries for land, or None

    Returns base64 PNG string, or None if no data.
    """
    prefix = "A" if corridor == "atlantic" else "P"
    opt_case = f"{prefix}O_{'WPS' if wps else 'noWPS'}"
    gc_case = f"{prefix}GC_{'WPS' if wps else 'noWPS'}"

    opt_routes = case_routes.get(opt_case, [])
    gc_routes = case_routes.get(gc_case, [])
    if not opt_routes and not gc_routes:
        return None

    # Fixed bounding boxes per corridor (lon_min, lon_max, lat_min, lat_max)
    # Pacific crosses antimeridian: plot in [0,360] convention
    view = (-80, 5, 25, 60) if corridor == "atlantic" else (130, 250, 15, 55)

    lon_min, lon_max, lat_min, lat_max = view
    is_pacific = corridor == "pacific"

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

    def _plot_lons(lons_list):
        """For Pacific, keep lons in [0,360]; for Atlantic, keep as-is."""
        if is_pacific:
            return [lon + 360 if lon < 0 else lon for lon in lons_list]
        return list(lons_list)

    # Draw land polygons clipped to view
    if land_polygons is not None:
        from matplotlib.collections import PatchCollection
        from matplotlib.patches import Polygon as MplPolygon
        from shapely import affinity
        from shapely.geometry import MultiPolygon, Polygon, box

        clip_box = box(lon_min - 1, lat_min - 1, lon_max + 1, lat_max + 1)
        patches = []
        for geom in land_polygons:
            geoms_to_try = [geom]
            # For Pacific, also try shifting the geometry by +360
            if is_pacific:
                geoms_to_try.append(affinity.translate(geom, xoff=360))
            for g in geoms_to_try:
                try:
                    clipped = g.intersection(clip_box)
                except Exception:
                    continue
                if clipped.is_empty:
                    continue
                polys = []
                if isinstance(clipped, MultiPolygon):
                    polys = list(clipped.geoms)
                elif isinstance(clipped, Polygon):
                    polys = [clipped]
                for poly in polys:
                    if poly.is_empty:
                        continue
                    xs, ys = poly.exterior.coords.xy
                    patches.append(
                        MplPolygon(list(zip(xs, ys, strict=False)), closed=True)
                    )
        if patches:
            pc = PatchCollection(
                patches,
                facecolor="#E8E0D8",
                edgecolor="#B0A090",
                linewidth=0.3,
                zorder=0,
            )
            ax.add_collection(pc)

    # Set ocean background
    ax.set_facecolor("#D6EAF8")

    # Plot GC routes first (background, grey)
    for lats, lons in gc_routes[:1]:
        ax.plot(
            _plot_lons(lons), lats, color="#BDBDBD", linewidth=0.5, alpha=0.6, zorder=1
        )
    if gc_routes:
        ax.plot([], [], color="#BDBDBD", linewidth=1, label="Great Circle")

    # Plot optimised routes
    for _i, (lats, lons) in enumerate(opt_routes):
        ax.plot(
            _plot_lons(lons), lats, color="#1565C0", linewidth=0.3, alpha=0.15, zorder=2
        )
    if opt_routes:
        ax.plot([], [], color="#1565C0", linewidth=1.5, alpha=0.7, label="Optimised")

    # Mark ports
    case_def = CASE_DEFS[opt_case]
    src_lat, src_lon = case_def["src"]
    dst_lat, dst_lon = case_def["dst"]
    p_src_lon = _plot_lons([src_lon])[0]
    p_dst_lon = _plot_lons([dst_lon])[0]
    ax.plot(p_src_lon, src_lat, "go", markersize=8, zorder=5, label="Departure")
    ax.plot(p_dst_lon, dst_lat, "rs", markersize=8, zorder=5, label="Arrival")

    wps_label = "with WPS" if wps else "without WPS"
    corridor_title = "Trans-Atlantic" if corridor == "atlantic" else "Trans-Pacific"
    ax.set_title(f"Routes \u2014 {corridor_title} ({wps_label})", fontsize=13)
    ax.set_xlabel("Longitude (\u00b0)")
    ax.set_ylabel("Latitude (\u00b0)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # For Pacific, relabel x-axis ticks to show real geographic longitudes
    if is_pacific:
        import numpy as np

        ticks = np.arange(int(lon_min / 10) * 10, lon_max + 1, 10)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t - 360}°" if t > 180 else f"{t}°" for t in ticks])

    ax.set_aspect("equal")
    fig.tight_layout()
    result = _fig_to_base64(fig)
    plt.close(fig)
    return result


def _write_detailed_results(
    output_dir: Path,
    scores: dict,
    all_errors: list,
    all_warnings: list,
    case_energies: dict,
    case_routes: dict,
    data_dir: Path | None = None,
):
    """Write detailed_results.html with scores, diagnostics, and plots."""
    plt, has_mpl = _try_import_matplotlib()

    # Load land polygons for route plots
    land_polygons = None
    if has_mpl and data_dir is not None:
        shapefile_path = data_dir / "ne_10m_land.shp"
        if shapefile_path.exists():
            try:
                import shapefile as shp
                from shapely.geometry import shape

                sf = shp.Reader(str(shapefile_path))
                land_polygons = [shape(s) for s in sf.shapes()]
            except Exception:
                pass

    # Generate plot images
    plots: list[tuple[str, str]] = []
    if has_mpl:
        for corridor in ("atlantic", "pacific"):
            img = _generate_energy_timeseries(plt, case_energies, corridor)
            if img:
                name = "Trans-Atlantic" if corridor == "atlantic" else "Trans-Pacific"
                plots.append((f"Energy Timeseries — {name}", img))

        for corridor in ("atlantic", "pacific"):
            for wps in (True, False):
                img = _generate_route_spaghetti(
                    plt, case_routes, corridor, wps, land_polygons=land_polygons
                )
                if img:
                    name = (
                        "Trans-Atlantic" if corridor == "atlantic" else "Trans-Pacific"
                    )
                    wps_label = "WPS" if wps else "noWPS"
                    plots.append((f"Routes — {name} ({wps_label})", img))

    # Build HTML
    html_parts = [
        '<!DOCTYPE html><html><head><meta charset="utf-8">',
        "<style>",
        "body { font-family: Arial, sans-serif; max-width: 1200px;"
        " margin: 0 auto; padding: 20px; }",
        "h1 { color: #2c3e50; }"
        " h2 { color: #34495e; border-bottom: 1px solid #eee;"
        " padding-bottom: 0.3em; }",
        "table { border-collapse: collapse; margin: 1em 0; }",
        "th, td { border: 1px solid #ddd; padding: 6px 12px; text-align: right; }",
        "th { background: #f5f5f5; text-align: left; }",
        ".warn { color: #e67e22; } .err { color: #e74c3c; } .ok { color: #27ae60; }",
        "img { max-width: 100%; margin: 10px 0; }",
        "</style></head><body>",
        "<h1>SWOPP3 — Detailed Results</h1>",
    ]

    # Scores table
    html_parts.append("<h2>Scores</h2><table><tr><th>Metric</th><th>Value</th></tr>")
    for k, v in scores.items():
        ek = html_mod.escape(str(k))
        if isinstance(v, float) and v < 1e11:
            html_parts.append(f"<tr><td>{ek}</td><td>{v:,.4f}</td></tr>")
        else:
            html_parts.append(
                f"<tr><td>{ek}</td><td>{html_mod.escape(str(v))}</td></tr>"
            )
    html_parts.append("</table>")

    # Warnings
    if all_warnings:
        html_parts.append(f'<h2 class="warn">Warnings ({len(all_warnings)})</h2><ul>')
        for w in all_warnings:
            html_parts.append(f"<li>{html_mod.escape(w)}</li>")
        html_parts.append("</ul>")

    # Errors
    if all_errors:
        html_parts.append(
            f'<h2 class="err">Validation Errors ({len(all_errors)})</h2><ul>'
        )
        for e in all_errors[:100]:  # cap at 100 to keep HTML manageable
            html_parts.append(f"<li>{html_mod.escape(e)}</li>")
        if len(all_errors) > 100:
            html_parts.append(f"<li>... and {len(all_errors) - 100} more</li>")
        html_parts.append("</ul>")
    else:
        html_parts.append(
            '<p class="ok"><strong>All validation checks passed ✓</strong></p>'
        )

    # Plots
    if plots:
        html_parts.append("<h2>Figures</h2>")
        for title, img_b64 in plots:
            safe_title = html_mod.escape(title)
            html_parts.append(f"<h3>{safe_title}</h3>")
            html_parts.append(
                f'<img src="data:image/png;base64,{img_b64}"' f' alt="{safe_title}">'
            )

    html_parts.append("</body></html>")

    html_path = output_dir / "detailed_results.html"
    html_path.write_text("\n".join(html_parts))


# ─── Main scoring logic ─────────────────────────────────────────────


def score_submission() -> dict:
    """Validate and score a SWOPP3 submission."""
    all_errors: list[str] = []
    all_warnings: list[str] = []
    scores: dict[str, float] = {}

    # Data collectors for detailed results
    case_energies: dict[str, list] = {c: [] for c in CASE_NAMES}
    case_routes: dict[str, list] = {c: [] for c in CASE_NAMES}

    # Detect team prefix
    team_prefix = find_team_prefix(submission_dir)
    if team_prefix is None:
        found_csvs = sorted(f.name for f in submission_dir.glob("*.csv"))
        found_all = (
            sorted(f.name for f in submission_dir.iterdir())
            if submission_dir.is_dir()
            else []
        )
        msg = (
            "Cannot detect team prefix from CSV filenames. "
            "Expected File-A CSVs named <TeamName>-<N>-<CaseName>.csv "
            f"(e.g. MyTeam-1-AO_WPS.csv) at the root of the zip. "
            f"Found {len(found_csvs)} CSV(s) at root: {found_csvs[:10]}. "
            f"All root entries ({len(found_all)}): {found_all[:20]}. "
            "Common fix: zip files from *inside* the directory, not the "
            "directory itself (cd into it, then zip)."
        )
        all_errors.append(msg)
        print(f"ERROR: {msg}", file=sys.stderr)
        scores["total_energy_mwh"] = 1e12
        scores["validation_errors"] = 1
        for case in CASE_NAMES:
            scores[f"{case}_energy_mwh"] = 1e12

        # Write log even on early return so it's available in CodaBench
        log_path = output_dir / "scoring_log.txt"
        with log_path.open("w") as f:
            f.write(f"VALIDATION: {len(all_errors)} issue(s) found\n")
            for err in all_errors:
                f.write(f"  ✗ {err}\n")
            f.write("\nSCORES:\n")
            for k, v in scores.items():
                f.write(f"  {k}: {v}\n")

        return scores

    # Try to load land checker
    land_checker = _load_land_checker(data_dir)
    if land_checker is None:
        all_warnings.append("Land shapefile not found — skipping land crossing checks")

    # Try to load ERA5 scorer
    era5_scorer = try_load_era5_scorer(data_dir)
    if era5_scorer is None:
        all_warnings.append(
            "ERA5 data not found in reference — using self-reported energy"
        )

    total_energy = 0.0

    for case in CASE_NAMES:
        file_a_path = submission_dir / f"{team_prefix}-{case}.csv"

        # Validate File A
        fa_errors, fa_rows = validate_file_a(file_a_path, case, EXPECTED_DEPARTURES)
        all_errors.extend(fa_errors)

        if not file_a_path.exists():
            scores[f"{case}_energy_mwh"] = 1e12
            total_energy += 1e12
            continue

        # Load reported energies
        try:
            energies = [float(r["energy_cons_mwh"]) for r in fa_rows]
            case_energy = sum(energies)
        except (KeyError, ValueError) as exc:
            all_errors.append(f"{case}: Cannot load energies: {exc}")
            scores[f"{case}_energy_mwh"] = 1e12
            total_energy += 1e12
            continue

        # Validate File B (track files) and collect waypoints for re-eval
        tracks_dir = submission_dir / "tracks"
        re_eval_energy = 0.0
        re_eval_ok = era5_scorer is not None
        case_wind_violations = 0
        case_wave_violations = 0
        case_land_crossings = 0
        for row in fa_rows:
            fb_name = row.get("details_filename", "")
            if not fb_name:
                all_errors.append(f"{case}: Missing details_filename in File A row")
                re_eval_ok = False
                continue

            fb_path = (
                tracks_dir / fb_name
                if tracks_dir.is_dir()
                else submission_dir / fb_name
            )

            # Parse departure/arrival for endpoint time checks
            dep_dt = arr_dt = None
            try:
                dep_dt = datetime.strptime(row["departure_time_utc"], DTFMT)
                arr_dt = datetime.strptime(row["arrival_time_utc"], DTFMT)
            except (ValueError, KeyError):
                pass

            fb_errors = validate_file_b(
                fb_path,
                case,
                departure_dt=dep_dt,
                arrival_dt=arr_dt,
                land_checker=None,  # defer land check to resampled track
            )
            all_errors.extend(fb_errors)

            # Load waypoints for resampling, land check, and re-evaluation
            waypoints = None
            if fb_path.exists():
                try:
                    with fb_path.open() as fbf:
                        wps_reader = csv.DictReader(fbf)
                        waypoints = []
                        for wp_row in wps_reader:
                            t = datetime.strptime(wp_row["time_utc"], DTFMT)
                            lat = float(wp_row["lat_deg"])
                            lon = float(wp_row["lon_deg"])
                            waypoints.append((t, lat, lon))
                except Exception:
                    waypoints = None

            if waypoints is None or len(waypoints) < 2:
                if re_eval_ok:
                    re_eval_ok = False
                continue

            # Resample to uniform Δt₂ for fair evaluation
            waypoints_resampled = resample_track(waypoints, RESAMPLE_DT_MINUTES)

            # ── Land crossing check on resampled waypoints (STRONG) ──
            if land_checker is not None and case not in GC_CASES:
                land_rows: list[int] = []
                for idx, (_, lat, lon) in enumerate(waypoints_resampled):
                    if land_checker(lat, lon):
                        land_rows.append(idx + 1)
                if land_rows:
                    case_land_crossings += len(land_rows)
                    n_calc = len(waypoints_resampled)
                    shown = land_rows[:10]
                    suffix = (
                        f", ... ({len(land_rows)} total)" if len(land_rows) > 10 else ""
                    )
                    rows_str = ", ".join(str(r) for r in shown) + suffix
                    all_errors.append(
                        f"{fb_path.name}: {len(land_rows)}/{n_calc} "
                        f"calculation points on land at index(es) {rows_str}"
                    )

            # ── ERA5 re-evaluation on resampled track ──
            if re_eval_ok and dep_dt is not None:
                try:
                    dep_str = dep_dt.strftime(DTFMT)
                    result = era5_scorer(case, waypoints_resampled, dep_str)
                    if result is not None:
                        e, wv, wav = result
                        re_eval_energy += e
                        case_wind_violations += wv
                        case_wave_violations += wav
                        case_energies[case].append((dep_dt, e))
                        # Collect route coords (subsample for memory)
                        lats = [wp[1] for wp in waypoints]
                        lons = [wp[2] for wp in waypoints]
                        step = max(1, len(lats) // 100)
                        case_routes[case].append(
                            (lats[::step] + [lats[-1]], lons[::step] + [lons[-1]])
                        )
                    else:
                        re_eval_ok = False
                except Exception:
                    re_eval_ok = False
            elif re_eval_ok:
                re_eval_ok = False

        # Use re-evaluated energy when successful, else fall back
        if re_eval_ok and era5_scorer is not None:
            case_energy_final = re_eval_energy
        elif era5_scorer is None:
            # ERA5 unavailable — use self-reported energy
            case_energy_final = case_energy
        else:
            # ERA5 available but re-evaluation failed — penalty
            case_energy_final = 1e12

        scores[f"{case}_energy_mwh"] = round(case_energy_final, 4)
        scores[f"{case}_reported_mwh"] = round(case_energy, 4)
        total_energy += case_energy_final

        # ── Strong violations (land) — route physically impossible ──
        if case_land_crossings > 0:
            all_errors.append(
                f"{case}: STRONG — {case_land_crossings} calculation points "
                f"cross land"
            )
        scores[f"{case}_land_violations"] = case_land_crossings

        # ── Weak violations (weather) — route dangerous but possible ──
        if case_wind_violations > 0 and case not in GC_CASES:
            all_warnings.append(
                f"{case}: {case_wind_violations} calculation points "
                f"exceed TWS limit ({MAX_WIND_MPS} m/s)"
            )
        if case_wave_violations > 0 and case not in GC_CASES:
            all_warnings.append(
                f"{case}: {case_wave_violations} calculation points "
                f"exceed Hs limit ({MAX_HS_M} m)"
            )
        scores[f"{case}_wind_violations"] = case_wind_violations
        scores[f"{case}_wave_violations"] = case_wave_violations

    # ── Cross-case consistency checks ──
    wps_pairs = [
        ("AO_WPS", "AO_noWPS"),
        ("AGC_WPS", "AGC_noWPS"),
        ("PO_WPS", "PO_noWPS"),
        ("PGC_WPS", "PGC_noWPS"),
    ]
    for wps_case, nowps_case in wps_pairs:
        wps_e = scores.get(f"{wps_case}_energy_mwh", 0)
        nowps_e = scores.get(f"{nowps_case}_energy_mwh", 0)
        if wps_e > nowps_e + 1e-3 and nowps_e < 1e12:
            all_warnings.append(
                f"WPS energy ({wps_case}={wps_e:.1f}) > noWPS energy "
                f"({nowps_case}={nowps_e:.1f})"
            )

    scores["total_energy_mwh"] = round(total_energy, 4)
    scores["validation_errors"] = len(all_errors)

    # Write detailed log
    log_path = output_dir / "scoring_log.txt"
    with log_path.open("w") as f:
        if all_warnings:
            f.write(f"WARNINGS: {len(all_warnings)}\n")
            for w in all_warnings:
                f.write(f"  ⚠ {w}\n")
            f.write("\n")

        if all_errors:
            f.write(f"VALIDATION: {len(all_errors)} issue(s) found\n")
            for err in all_errors:
                f.write(f"  ✗ {err}\n")
        else:
            f.write("VALIDATION: All checks passed ✓\n")

        f.write("\nSCORES:\n")
        for k, v in scores.items():
            f.write(f"  {k}: {v}\n")

    # Write detailed results with plots
    _write_detailed_results(
        output_dir,
        scores,
        all_errors,
        all_warnings,
        case_energies,
        case_routes,
        data_dir=data_dir,
    )

    return scores


# ─── Entry point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    scores = score_submission()

    scores_path = output_dir / "scores.json"
    with scores_path.open("w") as f:
        json.dump(scores, f, indent=2)

    # Print errors/warnings prominently so they appear in CodaBench logs
    n_errors = int(scores.get("validation_errors", 0))
    if n_errors > 0:
        log_path = output_dir / "scoring_log.txt"
        if log_path.exists():
            print("\n" + "=" * 60, file=sys.stderr)
            print(
                "SCORING LOG (see detailed_results.html for full report):",
                file=sys.stderr,
            )
            print("=" * 60, file=sys.stderr)
            print(log_path.read_text(), file=sys.stderr)
            print("=" * 60 + "\n", file=sys.stderr)

    print(f"Scores written to {scores_path}")
    for k, v in scores.items():
        print(f"  {k}: {v}")
