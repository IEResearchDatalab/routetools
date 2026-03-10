"""Tests for routetools.swopp3_output — File A / File B formatters."""

import csv
from datetime import UTC, datetime, timedelta
from pathlib import Path

import jax.numpy as jnp
import pytest

from routetools.swopp3_output import (
    _FILE_A_COLUMNS,
    _FILE_B_COLUMNS,
    TEAM,
    file_a_name,
    file_a_row,
    file_b_name,
    sailed_distance_nm,
    waypoint_times,
    write_file_a,
    write_file_b,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _straight_curve(n: int = 10) -> jnp.ndarray:
    """Straight line from (0, 0) to (1, 0) — ~111 km ≈ 60 nm."""
    src = jnp.array([0.0, 0.0])
    dst = jnp.array([1.0, 0.0])
    return jnp.linspace(src, dst, n)


_DEP = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# sailed_distance_nm
# ---------------------------------------------------------------------------
class TestSailedDistance:
    def test_positive(self):
        curve = _straight_curve()
        d = sailed_distance_nm(curve)
        assert d > 0

    def test_equator_one_degree(self):
        """1° of longitude at the equator ≈ 60 nm."""
        curve = _straight_curve(100)
        d = sailed_distance_nm(curve)
        assert 59.0 < d < 61.0, f"Expected ~60 nm, got {d}"


# ---------------------------------------------------------------------------
# waypoint_times
# ---------------------------------------------------------------------------
class TestWaypointTimes:
    def test_count(self):
        curve = _straight_curve(50)
        times = waypoint_times(curve, _DEP, passage_hours=354)
        assert len(times) == 50

    def test_first_is_departure(self):
        curve = _straight_curve()
        times = waypoint_times(curve, _DEP, passage_hours=10)
        assert times[0] == _DEP

    def test_last_is_arrival(self):
        curve = _straight_curve()
        times = waypoint_times(curve, _DEP, passage_hours=10)
        expected = _DEP + timedelta(hours=10)
        # Allow tiny floating-point rounding
        delta = abs((times[-1] - expected).total_seconds())
        assert delta < 1.0, f"Last time off by {delta}s"

    def test_uniform_spacing(self):
        curve = _straight_curve(11)
        times = waypoint_times(curve, _DEP, passage_hours=10)
        deltas = [(times[i + 1] - times[i]).total_seconds() for i in range(10)]
        assert all(abs(d - deltas[0]) < 0.01 for d in deltas)


# ---------------------------------------------------------------------------
# file_a_row
# ---------------------------------------------------------------------------
class TestFileARow:
    def test_keys(self):
        row = file_a_row(
            departure=_DEP,
            passage_hours=354,
            energy_mwh=12.5,
            max_wind_mps=18.0,
            max_hs_m=5.0,
            distance_nm=2800.0,
            details_filename="test.csv",
        )
        assert set(row.keys()) == set(_FILE_A_COLUMNS)

    def test_departure_format(self):
        row = file_a_row(_DEP, 354, 1.0, 1.0, 1.0, 1.0, "x.csv")
        assert row["departure_time_utc"] == "2024-01-01 12:00:00"

    def test_arrival_format(self):
        row = file_a_row(_DEP, 354, 1.0, 1.0, 1.0, 1.0, "x.csv")
        expected_arrival = _DEP + timedelta(hours=354)
        assert row["arrival_time_utc"] == expected_arrival.strftime("%Y-%m-%d %H:%M:%S")

    def test_details_filename_passthrough(self):
        row = file_a_row(_DEP, 354, 1.0, 1.0, 1.0, 1.0, "my_track.csv")
        assert row["details_filename"] == "my_track.csv"


# ---------------------------------------------------------------------------
# file_a_name / file_b_name
# ---------------------------------------------------------------------------
class TestNaming:
    def test_file_a_name(self):
        assert file_a_name(1, "AO_WPS") == "IEUniversity-1-AO_WPS.csv"

    def test_file_b_name(self):
        name = file_b_name(1, "AO_WPS", _DEP)
        assert name == "IEUniversity-1-AO_WPS-20240101.csv"

    def test_team(self):
        assert TEAM == "IEUniversity"


# ---------------------------------------------------------------------------
# write_file_a
# ---------------------------------------------------------------------------
class TestWriteFileA:
    def test_roundtrip(self, tmp_path: Path):
        rows = [
            file_a_row(_DEP, 354, 12.5, 18.0, 5.1, 2800.0, "track1.csv"),
            file_a_row(
                _DEP + timedelta(days=1), 354, 13.0, 19.0, 6.0, 2810.0, "track2.csv"
            ),
        ]
        out = write_file_a(rows, tmp_path / "output" / "file_a.csv")
        assert out.exists()

        # Parse back
        with out.open() as f:
            reader = csv.DictReader(f)
            parsed = list(reader)
        assert len(parsed) == 2
        assert parsed[0]["departure_time_utc"] == "2024-01-01 12:00:00"
        assert parsed[1]["energy_cons_mwh"] == "13.000000"
        assert set(reader.fieldnames) == set(_FILE_A_COLUMNS)

    def test_creates_parent_dirs(self, tmp_path: Path):
        rows = [file_a_row(_DEP, 10, 1.0, 1.0, 1.0, 100.0, "x.csv")]
        out = write_file_a(rows, tmp_path / "a" / "b" / "c" / "file.csv")
        assert out.exists()


# ---------------------------------------------------------------------------
# write_file_b
# ---------------------------------------------------------------------------
class TestWriteFileB:
    def test_roundtrip(self, tmp_path: Path):
        curve = _straight_curve(5)
        times = waypoint_times(curve, _DEP, passage_hours=10)
        out = write_file_b(curve, times, tmp_path / "track.csv")
        assert out.exists()

        with out.open() as f:
            reader = csv.DictReader(f)
            parsed = list(reader)
        assert len(parsed) == 5
        assert set(reader.fieldnames) == set(_FILE_B_COLUMNS)
        # First row: time is departure, lat=0, lon=0
        assert parsed[0]["time_utc"] == "2024-01-01 12:00:00"
        assert float(parsed[0]["lat_deg"]) == pytest.approx(0.0, abs=1e-4)
        assert float(parsed[0]["lon_deg"]) == pytest.approx(0.0, abs=1e-4)
        # Last row: lon=1, lat=0
        assert float(parsed[-1]["lon_deg"]) == pytest.approx(1.0, abs=1e-4)

    def test_mismatched_lengths_raises(self, tmp_path: Path):
        curve = _straight_curve(5)
        times = waypoint_times(curve, _DEP, passage_hours=10)
        with pytest.raises(ValueError):
            write_file_b(curve, times[:3], tmp_path / "bad.csv")

    def test_lon_lat_order(self, tmp_path: Path):
        """Curve is (lon, lat) but File B columns are lat_deg, lon_deg."""
        # Point with lon=10, lat=50
        curve = jnp.array([[10.0, 50.0], [11.0, 51.0]])
        times = [_DEP, _DEP + timedelta(hours=5)]
        out = write_file_b(curve, times, tmp_path / "coords.csv")
        with out.open() as f:
            reader = csv.DictReader(f)
            row = next(reader)
        assert float(row["lon_deg"]) == pytest.approx(10.0, abs=1e-4)
        assert float(row["lat_deg"]) == pytest.approx(50.0, abs=1e-4)
