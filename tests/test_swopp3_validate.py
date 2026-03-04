"""Tests for routetools.swopp3_validate — output validation."""

from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from routetools.swopp3_validate import (
    ValidationError,
    validate_case_pair_wps,
    validate_file_a,
    validate_file_b,
    validate_submission_dir,
)

_DTFMT = "%Y-%m-%d %H:%M:%S"
_DEP = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

_FILE_A_COLS = [
    "departure_time_utc", "arrival_time_utc", "energy_cons_mwh",
    "max_wind_mps", "max_hs_m", "sailed_distance_nm", "details_filename",
]

_FILE_B_COLS = ["time_utc", "lat_deg", "lon_deg"]


def _write_file_a(path: Path, n: int = 2) -> None:
    """Write a valid File A CSV for testing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FILE_A_COLS)
        writer.writeheader()
        for i in range(n):
            dep = _DEP + timedelta(days=i)
            arr = dep + timedelta(hours=354)
            writer.writerow({
                "departure_time_utc": dep.strftime(_DTFMT),
                "arrival_time_utc": arr.strftime(_DTFMT),
                "energy_cons_mwh": f"{10.0 + i:.6f}",
                "max_wind_mps": f"{15.0:.4f}",
                "max_hs_m": f"{3.0:.4f}",
                "sailed_distance_nm": f"{2800.0:.4f}",
                "details_filename": f"IEUniversity-1-TEST-{dep.strftime('%Y%m%d')}.csv",
            })


def _write_file_b(path: Path, n_waypoints: int = 5) -> None:
    """Write a valid File B CSV for testing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FILE_B_COLS)
        writer.writeheader()
        for i in range(n_waypoints):
            t = _DEP + timedelta(hours=i * 10)
            writer.writerow({
                "time_utc": t.strftime(_DTFMT),
                "lat_deg": f"{43.6 - i * 0.5:.6f}",
                "lon_deg": f"{-4.0 - i * 10:.6f}",
            })


# ---------------------------------------------------------------------------
# validate_file_a
# ---------------------------------------------------------------------------
class TestValidateFileA:
    def test_valid_file(self, tmp_path: Path):
        fa = tmp_path / "IEUniversity-1-AGC_WPS.csv"
        _write_file_a(fa)
        errs = validate_file_a(fa, expected_rows=2)
        assert not errs, errs

    def test_missing_file(self, tmp_path: Path):
        fa = tmp_path / "IEUniversity-1-AGC_WPS.csv"
        errs = validate_file_a(fa)
        assert any("not found" in e.message for e in errs)

    def test_wrong_row_count(self, tmp_path: Path):
        fa = tmp_path / "IEUniversity-1-AGC_WPS.csv"
        _write_file_a(fa, n=3)
        errs = validate_file_a(fa, expected_rows=5)
        assert any("Expected 5 rows" in e.message for e in errs)

    def test_missing_column(self, tmp_path: Path):
        fa = tmp_path / "IEUniversity-1-AGC_WPS.csv"
        fa.parent.mkdir(parents=True, exist_ok=True)
        with fa.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["departure_time_utc"])
            writer.writeheader()
            writer.writerow({"departure_time_utc": "2024-01-01 12:00:00"})
        errs = validate_file_a(fa, expected_rows=1)
        assert any("Missing columns" in e.message for e in errs)

    def test_bad_datetime(self, tmp_path: Path):
        fa = tmp_path / "IEUniversity-1-AGC_WPS.csv"
        fa.parent.mkdir(parents=True, exist_ok=True)
        with fa.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_FILE_A_COLS)
            writer.writeheader()
            writer.writerow({
                "departure_time_utc": "not-a-date",
                "arrival_time_utc": "2024-01-01 12:00:00",
                "energy_cons_mwh": "10.0",
                "max_wind_mps": "5.0",
                "max_hs_m": "2.0",
                "sailed_distance_nm": "2800",
                "details_filename": "track.csv",
            })
        errs = validate_file_a(fa, expected_rows=1)
        assert any("Bad datetime" in e.message for e in errs)

    def test_nan_energy(self, tmp_path: Path):
        fa = tmp_path / "IEUniversity-1-AGC_WPS.csv"
        fa.parent.mkdir(parents=True, exist_ok=True)
        with fa.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_FILE_A_COLS)
            writer.writeheader()
            writer.writerow({
                "departure_time_utc": "2024-01-01 12:00:00",
                "arrival_time_utc": "2024-01-16 18:00:00",
                "energy_cons_mwh": "nan",
                "max_wind_mps": "5.0",
                "max_hs_m": "2.0",
                "sailed_distance_nm": "2800",
                "details_filename": "track.csv",
            })
        errs = validate_file_a(fa, expected_rows=1)
        assert any("NaN" in e.message for e in errs)

    def test_naming_convention(self, tmp_path: Path):
        fa = tmp_path / "wrong_name.csv"
        _write_file_a(fa)
        errs = validate_file_a(fa, expected_rows=2)
        assert any("doesn't match pattern" in e.message for e in errs)


# ---------------------------------------------------------------------------
# validate_file_b
# ---------------------------------------------------------------------------
class TestValidateFileB:
    def test_valid_file(self, tmp_path: Path):
        fb = tmp_path / "track.csv"
        _write_file_b(fb)
        errs = validate_file_b(fb)
        assert not errs, errs

    def test_missing_file(self, tmp_path: Path):
        fb = tmp_path / "missing.csv"
        errs = validate_file_b(fb)
        assert any("not found" in e.message for e in errs)

    def test_too_few_waypoints(self, tmp_path: Path):
        fb = tmp_path / "short.csv"
        _write_file_b(fb, n_waypoints=1)
        errs = validate_file_b(fb, min_waypoints=5)
        assert any("waypoints" in e.message for e in errs)

    def test_non_increasing_time(self, tmp_path: Path):
        fb = tmp_path / "bad_time.csv"
        fb.parent.mkdir(parents=True, exist_ok=True)
        with fb.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_FILE_B_COLS)
            writer.writeheader()
            writer.writerow({"time_utc": "2024-01-02 12:00:00", "lat_deg": "43", "lon_deg": "-4"})
            writer.writerow({"time_utc": "2024-01-01 12:00:00", "lat_deg": "42", "lon_deg": "-5"})
        errs = validate_file_b(fb)
        assert any("not strictly increasing" in e.message for e in errs)

    def test_lat_out_of_range(self, tmp_path: Path):
        fb = tmp_path / "bad_lat.csv"
        fb.parent.mkdir(parents=True, exist_ok=True)
        with fb.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_FILE_B_COLS)
            writer.writeheader()
            writer.writerow({"time_utc": "2024-01-01 12:00:00", "lat_deg": "100", "lon_deg": "-4"})
        errs = validate_file_b(fb)
        assert any("out of range" in e.message for e in errs)


# ---------------------------------------------------------------------------
# validate_case_pair_wps
# ---------------------------------------------------------------------------
class TestValidateCasePairWPS:
    def test_wps_leq_nowps(self, tmp_path: Path):
        """WPS energy ≤ noWPS → no errors."""
        wps = tmp_path / "IEUniversity-1-AGC_WPS.csv"
        nowps = tmp_path / "IEUniversity-1-AGC_noWPS.csv"
        # WPS has lower energy
        _write_file_a_custom(wps, [8.0, 9.0])
        _write_file_a_custom(nowps, [10.0, 11.0])
        errs = validate_case_pair_wps(wps, nowps)
        assert not errs

    def test_wps_greater_nowps(self, tmp_path: Path):
        """WPS energy > noWPS → one error."""
        wps = tmp_path / "IEUniversity-1-AGC_WPS.csv"
        nowps = tmp_path / "IEUniversity-1-AGC_noWPS.csv"
        _write_file_a_custom(wps, [12.0, 13.0])
        _write_file_a_custom(nowps, [10.0, 11.0])
        errs = validate_case_pair_wps(wps, nowps)
        assert len(errs) == 1
        assert "WPS energy > noWPS" in errs[0].message


def _write_file_a_custom(path: Path, energies: list[float]) -> None:
    """Helper: write File A with specified energies."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FILE_A_COLS)
        writer.writeheader()
        for i, e in enumerate(energies):
            dep = _DEP + timedelta(days=i)
            arr = dep + timedelta(hours=354)
            writer.writerow({
                "departure_time_utc": dep.strftime(_DTFMT),
                "arrival_time_utc": arr.strftime(_DTFMT),
                "energy_cons_mwh": f"{e:.6f}",
                "max_wind_mps": "15.0000",
                "max_hs_m": "3.0000",
                "sailed_distance_nm": "2800.0000",
                "details_filename": f"track_{i}.csv",
            })


# ---------------------------------------------------------------------------
# validate_submission_dir (integration test)
# ---------------------------------------------------------------------------
class TestValidateSubmissionDir:
    def test_empty_dir_reports_missing(self, tmp_path: Path):
        errs = validate_submission_dir(tmp_path, expected_departures=2, verbose=False)
        # Should report missing File A for all 8 cases
        missing = [e for e in errs if "not found" in e.message]
        assert len(missing) == 8
