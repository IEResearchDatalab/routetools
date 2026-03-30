"""Tests for track resampling with geodesic interpolation.

Tests the resample_track() function that will be used by the CodaBench scorer
to decouple waypoint density (Δt₁) from energy integration accuracy (Δt₂).

Two test categories:
1. Unit tests for resample_track itself (deterministic, no ERA5 data).
2. Integration tests for convergence and invariance (require ERA5 data).
"""

from __future__ import annotations

import csv
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

# netCDF4 may warn about numpy binary incompatibility on some platforms
pytestmark = pytest.mark.filterwarnings(
    "ignore:numpy.ndarray size changed:RuntimeWarning",
)


# ── resample_track implementation (to be moved to scorer later) ──────


def slerp(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    f: float,
) -> tuple[float, float]:
    """Spherical linear interpolation at fraction *f* in [0, 1].

    Parameters and return values are in **degrees**.
    """
    phi1, lam1 = math.radians(lat1), math.radians(lon1)
    phi2, lam2 = math.radians(lat2), math.radians(lon2)

    # To Cartesian unit vectors
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
        # Nearly coincident – linear fallback
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
    dt_minutes: float = 15.0,
) -> list[tuple[datetime, float, float]]:
    """Resample a track to uniform temporal spacing via geodesic interpolation.

    Parameters
    ----------
    waypoints
        List of ``(time, lat_deg, lon_deg)`` tuples — the original track.
    dt_minutes
        Target sub-segment interval in minutes.

    Returns
    -------
    list[tuple[datetime, float, float]]
        Resampled waypoints at approximately uniform Δt₂.
        The first and last points are always preserved exactly.
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
            lat, lon = slerp(lat0, lon0, lat1, lon1, f)
            result.append((t, lat, lon))

    # Always append the final waypoint
    result.append(waypoints[-1])
    return result


# ── Helper: load a track CSV ────────────────────────────────────────


def _load_track_csv(
    path: Path,
) -> list[tuple[datetime, float, float]]:
    """Load track CSV (time_utc, lat_deg, lon_deg) -> list of tuples."""
    with path.open() as f:
        reader = csv.DictReader(f)
        return [
            (
                datetime.strptime(row["time_utc"], "%Y-%m-%d %H:%M:%S").replace(
                    tzinfo=UTC,
                ),
                float(row["lat_deg"]),
                float(row["lon_deg"]),
            )
            for row in reader
        ]


def _downsample_waypoints(
    waypoints: list[tuple[datetime, float, float]],
    factor: int,
) -> list[tuple[datetime, float, float]]:
    """Keep every *factor*-th waypoint plus the last one."""
    result = waypoints[::factor]
    if result[-1] != waypoints[-1]:
        result.append(waypoints[-1])
    return result


# ══════════════════════════════════════════════════════════════════════
# Unit tests (no data required)
# ══════════════════════════════════════════════════════════════════════


class TestSlerp:
    """Spherical linear interpolation."""

    def test_endpoints(self) -> None:
        lat, lon = slerp(34.8, 140.0, 34.4, -121.0, 0.0)
        assert lat == pytest.approx(34.8, abs=1e-10)
        assert lon == pytest.approx(140.0, abs=1e-10)

        lat, lon = slerp(34.8, 140.0, 34.4, -121.0, 1.0)
        assert lat == pytest.approx(34.4, abs=1e-10)
        assert lon == pytest.approx(-121.0, abs=1e-10)

    def test_midpoint_on_great_circle(self) -> None:
        """Midpoint should be north of linear midpoint for east-west routes."""
        lat_mid, _ = slerp(34.8, 140.0, 34.4, -121.0, 0.5)
        linear_lat = (34.8 + 34.4) / 2
        # Great circle from Japan to California goes north
        assert lat_mid > linear_lat + 1.0

    def test_coincident_points(self) -> None:
        lat, lon = slerp(45.0, 10.0, 45.0, 10.0, 0.5)
        assert lat == pytest.approx(45.0, abs=1e-6)
        assert lon == pytest.approx(10.0, abs=1e-6)

    def test_antimeridian_crossing(self) -> None:
        """Interpolation across the antimeridian should take the short way."""
        lat, lon = slerp(40.0, 170.0, 40.0, -170.0, 0.5)
        # Should cross at lon ≈ 180, not go the long way around
        assert abs(lon) > 170.0

    def test_returns_degrees(self) -> None:
        lat, lon = slerp(0.0, 0.0, 0.0, 90.0, 0.5)
        assert -90 <= lat <= 90
        assert -180 <= lon <= 360


class TestResampleTrack:
    """resample_track() function."""

    @pytest.fixture()
    def simple_track(self) -> list[tuple[datetime, float, float]]:
        """Two-point track: 2 hours apart."""
        t0 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        return [
            (t0, 34.8, 140.0),
            (t0 + timedelta(hours=2), 34.6, 139.0),
        ]

    def test_preserves_endpoints(self, simple_track) -> None:
        result = resample_track(simple_track, dt_minutes=15)
        assert result[0] == simple_track[0]
        assert result[-1] == simple_track[-1]

    def test_correct_point_count(self, simple_track) -> None:
        # 2 hours = 120 min; dt=15 → ceil(120/15) = 8 sub-segments + final
        result = resample_track(simple_track, dt_minutes=15)
        assert len(result) == 9  # 8 sub-points from first segment + 1 final

    def test_uniform_spacing(self, simple_track) -> None:
        result = resample_track(simple_track, dt_minutes=30)
        # 120 min / 30 min = 4 sub-segments
        times = [r[0] for r in result]
        dts = [(times[i + 1] - times[i]).total_seconds() for i in range(len(times) - 1)]
        # All should be 30 min = 1800 s
        for dt_val in dts:
            assert dt_val == pytest.approx(1800.0, abs=0.1)

    def test_single_waypoint_passthrough(self) -> None:
        t0 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        wps = [(t0, 34.8, 140.0)]
        assert resample_track(wps) == wps

    def test_dt_larger_than_segment(self) -> None:
        """If dt > segment duration, keep at least one sub-segment."""
        t0 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        wps = [
            (t0, 34.8, 140.0),
            (t0 + timedelta(minutes=5), 34.81, 140.01),
        ]
        result = resample_track(wps, dt_minutes=60)
        # n_sub = max(1, ceil(5/60)) = 1 → start + final
        assert len(result) == 2

    def test_multi_segment(self) -> None:
        """Three-point track: each segment resampled independently."""
        t0 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        wps = [
            (t0, 34.8, 140.0),
            (t0 + timedelta(hours=1), 35.0, 141.0),
            (t0 + timedelta(hours=2), 35.2, 142.0),
        ]
        result = resample_track(wps, dt_minutes=30)
        # Seg 1: 60 min / 30 = 2 sub -> 2 points
        # Seg 2: 60 min / 30 = 2 sub -> 2 points
        # + final = 5
        assert len(result) == 5

    def test_all_lats_within_range(self) -> None:
        """Pacific crossing should keep all latitudes in [-90, 90]."""
        t0 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        wps = [
            (t0, 34.8, 140.0),
            (t0 + timedelta(hours=583), 34.4, -121.0),
        ]
        result = resample_track(wps, dt_minutes=60)
        for _, lat, _lon in result:
            assert -90 <= lat <= 90


class TestCircularInterpolation:
    """Circular (sin/cos) interpolation for angular variables like MWD."""

    def _make_tiny_grid(self, mwd_values: np.ndarray) -> dict:
        """Build a minimal 1×2×1 wave grid with given MWD at two time steps."""
        mwd = mwd_values.reshape(2, 1, 1).astype(np.float32)
        mwd_rad = np.radians(mwd)
        return {
            "data": {
                "mwd": mwd,
                "mwd_sin": np.sin(mwd_rad).astype(np.float32),
                "mwd_cos": np.cos(mwd_rad).astype(np.float32),
            },
            "lat": np.array([40.0]),
            "lon": np.array([10.0]),
            "times_h": np.array([0.0, 1.0]),
            "t0": np.datetime64("2024-01-01T00:00"),
        }

    def test_no_wrap_matches_linear(self) -> None:
        """Away from 0/360, circular and linear interp should agree."""
        grid = self._make_tiny_grid(np.array([100.0, 120.0]))
        lat = np.array([40.0])
        lon = np.array([10.0])
        t_h = np.array([0.5])

        linear = _interp_era5_trilinear(grid, "mwd", lat, lon, t_h)
        circular = _interp_era5_angle_trilinear(grid, "mwd", lat, lon, t_h)
        assert circular[0] == pytest.approx(linear[0], abs=0.5)
        assert circular[0] == pytest.approx(110.0, abs=0.5)

    def test_wrap_around_360(self) -> None:
        """Midpoint of 350° and 10° should be ~0° (360°), not 180°."""
        grid = self._make_tiny_grid(np.array([350.0, 10.0]))
        lat = np.array([40.0])
        lon = np.array([10.0])
        t_h = np.array([0.5])

        circular = _interp_era5_angle_trilinear(grid, "mwd", lat, lon, t_h)
        linear = _interp_era5_trilinear(grid, "mwd", lat, lon, t_h)

        # Circular should give ~0° (i.e. 360°)
        circ_centered = (circular[0] + 180) % 360 - 180  # center around 0
        assert abs(circ_centered) < 5.0, f"Circular gave {circular[0]}°, expected ~0°"

        # Linear gives ~180° — the wrong answer
        assert abs(linear[0] - 180.0) < 1.0, f"Linear gave {linear[0]}°, expected ~180°"

    def test_wrap_around_near_north(self) -> None:
        """5° and 355° should average to ~0°, not 180°."""
        grid = self._make_tiny_grid(np.array([5.0, 355.0]))
        lat = np.array([40.0])
        lon = np.array([10.0])
        t_h = np.array([0.5])

        circular = _interp_era5_angle_trilinear(grid, "mwd", lat, lon, t_h)
        circ_centered = (circular[0] + 180) % 360 - 180
        assert abs(circ_centered) < 5.0


# ══════════════════════════════════════════════════════════════════════
# Integration tests (require ERA5 data + routetools)
# ══════════════════════════════════════════════════════════════════════

_TRACK_DIR = Path("output/swopp3_gpu/tracks")
_DATA_DIR = Path("data/era5")

# Use Atlantic corridor (smaller ~3.4GB vs ~5.1GB Pacific)
_REF_TRACK = _TRACK_DIR / "IEUniversity-1-AO_WPS-20240701.csv"
_WIND_NC = _DATA_DIR / "era5_wind_atlantic_2024.nc"
_WAVE_NC = _DATA_DIR / "era5_waves_atlantic_2024.nc"
_PASSAGE_HOURS = 354.0

_requires_era5 = pytest.mark.skipif(
    not (_REF_TRACK.exists() and _WIND_NC.exists() and _WAVE_NC.exists()),
    reason="ERA5 data or reference tracks not available",
)


def _load_era5_numpy(nc_path: Path) -> dict:
    """Load a single ERA5 NetCDF file into a numpy dict.

    Returns dict with keys: data, lat, lon, times_h, t0.
    Mirrors ``_load_era5_grid`` from the CodaBench scorer.
    """
    import netCDF4

    ds = netCDF4.Dataset(str(nc_path), "r")

    lat_name = "latitude" if "latitude" in ds.variables else "lat"
    lon_name = "longitude" if "longitude" in ds.variables else "lon"
    time_name = "valid_time" if "valid_time" in ds.variables else "time"

    lat = np.array(ds.variables[lat_name][:], dtype=np.float64)
    lon = np.array(ds.variables[lon_name][:], dtype=np.float64)

    coord_names = {lat_name, lon_name, time_name}
    var_names = [v for v in ds.variables if v not in coord_names]

    _LONG_TO_SHORT = {
        "10m_u_component_of_wind": "u10",
        "10m_v_component_of_wind": "v10",
        "significant_height_of_combined_wind_waves_and_swell": "swh",
        "mean_wave_direction": "mwd",
    }

    t_var = ds.variables[time_name]
    cal = getattr(t_var, "calendar", "standard")
    times = netCDF4.num2date(t_var[:], t_var.units, cal)

    t0 = np.datetime64(times[0])
    times_np = np.array([np.datetime64(t) for t in times])
    times_h = ((times_np - t0) / np.timedelta64(1, "h")).astype(np.float64)

    data = {}
    for v in var_names:
        short = _LONG_TO_SHORT.get(v, v)
        arr = np.array(ds.variables[v][:], dtype=np.float32)
        np.nan_to_num(arr, copy=False, nan=0.0)
        data[short] = arr

    ds.close()

    if lat[0] > lat[-1]:
        lat = lat[::-1]
        for v in data:
            data[v] = data[v][:, ::-1, :]

    if lon[0] > lon[-1]:
        lon = lon[::-1]
        for v in data:
            data[v] = data[v][:, :, ::-1]

    # Decompose angular variables into sin/cos for correct interpolation
    if "mwd" in data:
        mwd_rad = np.radians(data["mwd"])
        data["mwd_sin"] = np.sin(mwd_rad).astype(np.float32)
        data["mwd_cos"] = np.cos(mwd_rad).astype(np.float32)

    return {"data": data, "lat": lat, "lon": lon, "times_h": times_h, "t0": t0}


def _interp_era5_trilinear(
    grid: dict,
    var_name: str,
    query_lat: np.ndarray,
    query_lon: np.ndarray,
    query_t_h: np.ndarray,
) -> np.ndarray:
    """Trilinear interpolation of an ERA5 variable at query points."""
    arr = grid["data"][var_name]
    lat, lon, times_h = grid["lat"], grid["lon"], grid["times_h"]

    dt = times_h[1] - times_h[0] if len(times_h) > 1 else 1.0
    dlat = lat[1] - lat[0] if len(lat) > 1 else 1.0
    dlon = lon[1] - lon[0] if len(lon) > 1 else 1.0

    fi_t = np.clip((query_t_h - times_h[0]) / dt, 0, len(times_h) - 1)
    fi_lat = np.clip((query_lat - lat[0]) / dlat, 0, len(lat) - 1)
    fi_lon = np.clip((query_lon - lon[0]) / dlon, 0, len(lon) - 1)

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


def _interp_era5_angle_trilinear(
    grid: dict,
    var_name: str,
    query_lat: np.ndarray,
    query_lon: np.ndarray,
    query_t_h: np.ndarray,
) -> np.ndarray:
    """Interpolate an angular ERA5 variable using sin/cos decomposition."""
    sin_vals = _interp_era5_trilinear(
        grid, f"{var_name}_sin", query_lat, query_lon, query_t_h
    )
    cos_vals = _interp_era5_trilinear(
        grid, f"{var_name}_cos", query_lat, query_lon, query_t_h
    )
    return np.mod(np.degrees(np.arctan2(sin_vals, cos_vals)), 360.0)


# RISE performance model constants (must match codabench scorer)
_KH = 969.0 / 226.0
_KA = 49.0 / 320.0
_AW = 11.1395
_KW = 125.0 / 432.0
_KS = 27489.0 / 32000.0
_DEAD_ZONE_DEG = 10.0


def _rise_power_np(tws, twa_deg, swh, mwa_deg, v, wps):
    """RISE power (kW) — numpy arrays."""
    twa_rad = np.radians(twa_deg)
    p_hull = _KH * v**3
    ux = tws * np.cos(twa_rad) + v
    uy = tws * np.sin(twa_rad)
    vr = np.sqrt(ux**2 + uy**2)
    p_wind = _KA * v * (vr * ux - v**2)
    mwa_rad = np.radians(mwa_deg)
    p_wave = _AW * swh**2 * v**1.5 * np.exp(-_KW * np.abs(mwa_rad) ** 3)
    power = p_hull + p_wind + p_wave
    if wps:
        awa_deg = np.degrees(np.arctan2(np.abs(uy), ux))
        sail_active = awa_deg >= _DEAD_ZONE_DEG
        alpha = np.where(sail_active, np.radians(awa_deg - _DEAD_ZONE_DEG), 0.0)
        sin_a = np.sin(alpha)
        p_sail = _KS * sin_a * (1.0 + 0.15 * sin_a**2) * vr**2 * v
        power = power - np.where(sail_active, p_sail, 0.0)
    return np.maximum(power, 0.0)


def _np_haversine_m(lat1, lon1, lat2, lon2):
    """Haversine distance in metres."""
    R = 6_371_000.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    lat1r, lat2r = np.radians(lat1), np.radians(lat2)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def _np_forward_bearing_deg(lat1, lon1, lat2, lon2):
    """Forward bearing in degrees [0, 360)."""
    lat1r, lat2r = np.radians(lat1), np.radians(lat2)
    dlon = np.radians(lon2 - lon1)
    x = np.sin(dlon) * np.cos(lat2r)
    y = np.cos(lat1r) * np.sin(lat2r) - np.sin(lat1r) * np.cos(lat2r) * np.cos(dlon)
    return np.mod(np.degrees(np.arctan2(x, y)), 360.0)


def _evaluate_energy(
    waypoints: list[tuple[datetime, float, float]],
    passage_hours: float,
    wps: bool,
    wind_grid: dict,
    wave_grid: dict,
) -> float:
    """Evaluate route energy using numpy-only RISE + trapezoidal rule.

    Mirrors the CodaBench scorer pipeline without JAX dependency.
    *passage_hours* is accepted for call-site compatibility but unused —
    segment durations are derived from waypoint timestamps.
    """
    n_wp = len(waypoints)
    if n_wp < 2:
        return 0.0

    lats = np.array([wp[1] for wp in waypoints])
    lons = np.array([wp[2] for wp in waypoints])

    wp_times = np.array(
        [np.datetime64(wp[0].replace(tzinfo=None)) for wp in waypoints],
        dtype="datetime64[s]",
    )
    seg_dt_h = ((wp_times[1:] - wp_times[:-1]) / np.timedelta64(1, "h")).astype(
        np.float64
    )
    seg_dt_h = np.maximum(seg_dt_h, 1e-6)

    # Normalize lons to ERA5 grid convention [0, 360)
    grid_lon = wind_grid["lon"]
    if grid_lon[0] >= 0 and grid_lon[-1] > 180:
        lons = np.where(lons < 0, lons + 360, lons)

    dep_dt64 = wp_times[0]
    dep_offset_h = float((dep_dt64 - wind_grid["t0"]) / np.timedelta64(1, "h"))
    wp_times_h = np.zeros(n_wp, dtype=np.float64)
    wp_times_h[0] = dep_offset_h
    wp_times_h[1:] = dep_offset_h + np.cumsum(seg_dt_h)

    # Interpolate weather at all waypoints (trapezoidal endpoints)
    u10 = _interp_era5_trilinear(wind_grid, "u10", lats, lons, wp_times_h)
    v10 = _interp_era5_trilinear(wind_grid, "v10", lats, lons, wp_times_h)
    swh = _interp_era5_trilinear(wave_grid, "swh", lats, lons, wp_times_h)
    mwd = _interp_era5_angle_trilinear(wave_grid, "mwd", lats, lons, wp_times_h)

    tws_all = np.sqrt(u10**2 + v10**2)

    seg_dist_m = _np_haversine_m(lats[:-1], lons[:-1], lats[1:], lons[1:])
    v_mps = seg_dist_m / (seg_dt_h * 3600.0)
    bearing_deg = _np_forward_bearing_deg(lats[:-1], lons[:-1], lats[1:], lons[1:])

    # RISE power at segment start and end points
    wind_from_s = np.mod(180.0 + np.degrees(np.arctan2(u10[:-1], v10[:-1])), 360.0)
    twa_s = np.mod(wind_from_s - bearing_deg, 360.0)
    mwa_s = np.mod(mwd[:-1] - bearing_deg, 360.0)
    power_s = _rise_power_np(tws_all[:-1], twa_s, swh[:-1], mwa_s, v_mps, wps)

    wind_from_e = np.mod(180.0 + np.degrees(np.arctan2(u10[1:], v10[1:])), 360.0)
    twa_e = np.mod(wind_from_e - bearing_deg, 360.0)
    mwa_e = np.mod(mwd[1:] - bearing_deg, 360.0)
    power_e = _rise_power_np(tws_all[1:], twa_e, swh[1:], mwa_e, v_mps, wps)

    power_avg = (power_s + power_e) / 2.0
    return float(np.sum(power_avg * seg_dt_h) / 1000.0)


# Share a single ERA5 load across ALL integration tests (module scope)
# to avoid loading data twice.  Uses numpy only — no JAX.
@pytest.fixture(scope="module")
def era5_fields():
    """Load ERA5 wind + wave grids as numpy dicts (no JAX)."""
    wind_grid = _load_era5_numpy(_WIND_NC)
    wave_grid = _load_era5_numpy(_WAVE_NC)
    return wind_grid, wave_grid


@pytest.fixture(scope="module")
def reference_track():
    """Load the reference track once."""
    return _load_track_csv(_REF_TRACK)


@_requires_era5
class TestConvergence:
    """Energy should converge as Δt₂ decreases.

    Take a native L=584 track (Δt₁ ≈ 1h), resample to Δt₂ = 60, 30, 15,
    10, 5 min, and verify that energy values converge (decreasing successive
    differences).
    """

    def test_convergence_wps(self, era5_fields, reference_track) -> None:
        wind_grid, wave_grid = era5_fields

        dt_values = [60, 30, 15, 10, 5]
        energies = {}

        for dt_min in dt_values:
            resampled = resample_track(reference_track, dt_minutes=dt_min)
            e = _evaluate_energy(
                resampled,
                _PASSAGE_HOURS,
                wps=True,
                wind_grid=wind_grid,
                wave_grid=wave_grid,
            )
            energies[dt_min] = e

        diffs = [
            abs(energies[dt_values[i]] - energies[dt_values[i + 1]])
            for i in range(len(dt_values) - 1)
        ]

        print("\nConvergence test (WPS):")
        for dt_min in dt_values:
            print(f"  Δt₂={dt_min:>3}min → {energies[dt_min]:.4f} MWh")
        print("Successive diffs:", [f"{d:.4f}" for d in diffs])

        # Energy values should stabilize: total spread < 0.5%
        all_e = list(energies.values())
        pct_range = (max(all_e) - min(all_e)) / np.mean(all_e) * 100
        assert (
            pct_range < 0.5
        ), f"Energy spread {pct_range:.2f}% exceeds 0.5% across resolutions"

        # At dt=15min and dt=5min, energy should agree within 2%
        ref = energies[5]
        pct_15 = abs(energies[15] - ref) / ref * 100
        assert pct_15 < 2.0, f"dt=15min vs dt=5min: {pct_15:.2f}% difference"

    def test_convergence_nowps(self, era5_fields, reference_track) -> None:
        wind_grid, wave_grid = era5_fields

        dt_values = [60, 30, 15, 10, 5]
        energies = {}

        for dt_min in dt_values:
            resampled = resample_track(reference_track, dt_minutes=dt_min)
            e = _evaluate_energy(
                resampled,
                _PASSAGE_HOURS,
                wps=False,
                wind_grid=wind_grid,
                wave_grid=wave_grid,
            )
            energies[dt_min] = e

        diffs = [
            abs(energies[dt_values[i]] - energies[dt_values[i + 1]])
            for i in range(len(dt_values) - 1)
        ]

        print("\nConvergence test (noWPS):")
        for dt_min in dt_values:
            print(f"  Δt₂={dt_min:>3}min → {energies[dt_min]:.4f} MWh")
        print("Successive diffs:", [f"{d:.4f}" for d in diffs])

        assert diffs[-1] < diffs[0]

        ref = energies[5]
        pct_15 = abs(energies[15] - ref) / ref * 100
        assert pct_15 < 2.0, f"dt=15min vs dt=5min: {pct_15:.2f}% difference"


@_requires_era5
class TestInvariance:
    """Energy should be invariant to submission waypoint density after resampling.

    Take the native track, downsample to L≈50, L≈100, L≈200, resample all
    back to Δt₂=15min, and verify that energy values agree within tolerance.
    """

    def test_invariance_nowps(self, era5_fields, reference_track) -> None:
        wind_grid, wave_grid = era5_fields

        ref_resampled = resample_track(reference_track, dt_minutes=15)
        ref_energy = _evaluate_energy(
            ref_resampled,
            _PASSAGE_HOURS,
            wps=False,
            wind_grid=wind_grid,
            wave_grid=wave_grid,
        )

        factors = [12, 6, 3]  # → ~49, ~97, ~195 waypoints
        energies = {"native(584)": ref_energy}

        for factor in factors:
            sparse = _downsample_waypoints(reference_track, factor)
            resampled = resample_track(sparse, dt_minutes=15)
            e = _evaluate_energy(
                resampled,
                _PASSAGE_HOURS,
                wps=False,
                wind_grid=wind_grid,
                wave_grid=wave_grid,
            )
            energies[f"L≈{len(sparse)}"] = e

        print("\nInvariance test (noWPS, dt=15min):")
        for label, e in energies.items():
            pct = abs(e - ref_energy) / ref_energy * 100
            print(f"  {label:>15} → {e:.4f} MWh  ({pct:+.2f}%)")

        for label, e in energies.items():
            pct = abs(e - ref_energy) / ref_energy * 100
            assert pct < 3.0, (
                f"{label}: {e:.4f} MWh deviates {pct:.2f}% from "
                f"reference {ref_energy:.4f} MWh"
            )

    def test_invariance_wps(self, era5_fields, reference_track) -> None:
        wind_grid, wave_grid = era5_fields

        ref_resampled = resample_track(reference_track, dt_minutes=15)
        ref_energy = _evaluate_energy(
            ref_resampled,
            _PASSAGE_HOURS,
            wps=True,
            wind_grid=wind_grid,
            wave_grid=wave_grid,
        )

        factors = [12, 6, 3]
        energies = {"native(584)": ref_energy}

        for factor in factors:
            sparse = _downsample_waypoints(reference_track, factor)
            resampled = resample_track(sparse, dt_minutes=15)
            e = _evaluate_energy(
                resampled,
                _PASSAGE_HOURS,
                wps=True,
                wind_grid=wind_grid,
                wave_grid=wave_grid,
            )
            energies[f"L≈{len(sparse)}"] = e

        print("\nInvariance test (WPS, dt=15min):")
        for label, e in energies.items():
            pct = abs(e - ref_energy) / ref_energy * 100
            print(f"  {label:>15} → {e:.4f} MWh  ({pct:+.2f}%)")

        # WPS is more sensitive — allow 5% tolerance
        for label, e in energies.items():
            pct = abs(e - ref_energy) / ref_energy * 100
            assert pct < 5.0, (
                f"{label}: {e:.4f} MWh deviates {pct:.2f}% from "
                f"reference {ref_energy:.4f} MWh"
            )
