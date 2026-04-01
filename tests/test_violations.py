"""Tests for the routetools.violations module."""

import csv
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from routetools.violations import (
    ScenarioViolationCounts,
    count_folder_violations,
    count_land_violations,
    count_summary_weather_violations,
    find_team_prefix,
    format_grouped_violation_table,
    is_gc_case,
    write_grouped_violation_csv,
)

_DTFMT = "%Y-%m-%d %H:%M:%S"
_DEP = datetime(2024, 1, 1, 12, 0, 0)
_FILE_A_COLS = [
    "departure_time_utc",
    "arrival_time_utc",
    "energy_cons_mwh",
    "max_wind_mps",
    "max_hs_m",
    "sailed_distance_nm",
    "details_filename",
]
_FILE_B_COLS = ["time_utc", "lat_deg", "lon_deg"]


def _write_file_a(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_FILE_A_COLS)
        writer.writeheader()
        writer.writerows(rows)


def _write_file_b(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_FILE_B_COLS)
        writer.writeheader()
        writer.writerows(rows)


def _summary_row(
    *, dep: datetime, wind: float, wave: float, details: str
) -> dict[str, str]:
    return {
        "departure_time_utc": dep.strftime(_DTFMT),
        "arrival_time_utc": (dep + timedelta(hours=10)).strftime(_DTFMT),
        "energy_cons_mwh": "100.0",
        "max_wind_mps": f"{wind:.1f}",
        "max_hs_m": f"{wave:.1f}",
        "sailed_distance_nm": "2800.0",
        "details_filename": details,
    }


def _track_rows(n_waypoints: int = 6) -> list[dict[str, str]]:
    rows = []
    for index in range(n_waypoints):
        rows.append(
            {
                "time_utc": (_DEP + timedelta(hours=index)).strftime(_DTFMT),
                "lat_deg": f"{40.0 + index:.6f}",
                "lon_deg": f"{-10.0 - index:.6f}",
            }
        )
    return rows


def test_count_summary_weather_violations(tmp_path: Path) -> None:
    summary_path = tmp_path / "IEUniversity-1-AO_WPS.csv"
    _write_file_a(
        summary_path,
        [
            _summary_row(dep=_DEP, wind=19.0, wave=6.5, details="track_a.csv"),
            _summary_row(
                dep=_DEP + timedelta(days=1), wind=21.0, wave=6.0, details="track_b.csv"
            ),
            _summary_row(
                dep=_DEP + timedelta(days=2), wind=18.0, wave=7.5, details="track_c.csv"
            ),
        ],
    )

    wind_violations, wave_violations = count_summary_weather_violations(
        summary_path,
        "AO_WPS",
    )

    assert wind_violations == 1
    assert wave_violations == 1


def test_count_land_violations_uses_land_checker(tmp_path: Path) -> None:
    track_path = tmp_path / "track.csv"
    _write_file_b(track_path, _track_rows(n_waypoints=5))

    def land_checker(lat: float, lon: float) -> bool:
        return lat >= 43.0

    assert count_land_violations(track_path, land_checker) == 2


def test_count_folder_violations_counts_one_folder(tmp_path: Path) -> None:
    input_dir = tmp_path / "swopp3_penalty"
    tracks_dir = input_dir / "tracks"

    _write_file_a(
        input_dir / "IEUniversity-1-AO_WPS.csv",
        [
            _summary_row(dep=_DEP, wind=21.0, wave=8.0, details="ao_track.csv"),
        ],
    )
    _write_file_b(tracks_dir / "ao_track.csv", _track_rows(n_waypoints=4))

    _write_file_a(
        input_dir / "IEUniversity-1-AGC_WPS.csv",
        [
            _summary_row(dep=_DEP, wind=21.0, wave=8.0, details="gc_track.csv"),
        ],
    )
    _write_file_b(tracks_dir / "gc_track.csv", _track_rows(n_waypoints=4))

    rows = count_folder_violations(
        input_dir,
        land_checker=lambda lat, lon: lat >= 42.0,
        include_gc=False,
    )

    assert len(rows) == 1
    assert rows[0].folder == "swopp3_penalty"
    assert rows[0].case_id == "AO_WPS"
    assert rows[0].wind_violations == 1
    assert rows[0].wave_violations == 1
    assert rows[0].land_violations == 2


def test_find_team_prefix_detects_known_pattern(tmp_path: Path) -> None:
    _write_file_a(
        tmp_path / "IEUniversity-7-AO_WPS.csv",
        [_summary_row(dep=_DEP, wind=10.0, wave=2.0, details="track.csv")],
    )

    assert find_team_prefix(tmp_path) == "IEUniversity-7"


def test_find_team_prefix_raises_when_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No SWOPP3 File A CSV files found"):
        find_team_prefix(tmp_path)


@pytest.mark.parametrize(
    ("case_id", "expected"),
    [
        ("AGC_WPS", True),
        ("AGC_noWPS", True),
        ("AO_WPS", False),
        ("PO_noWPS", False),
    ],
)
def test_is_gc_case(case_id: str, expected: bool) -> None:
    assert is_gc_case(case_id) is expected


@pytest.mark.parametrize(
    ("wind", "wave", "land", "wind_pen", "wave_pen"),
    [
        (0, 0, 0, 0.0, 0.0),
        (1, 2, 3, 4.5, 5.5),
        (10, 20, 30, 1.25, 2.75),
    ],
)
def test_scenario_violation_counts_totals(
    wind: int,
    wave: int,
    land: int,
    wind_pen: float,
    wave_pen: float,
) -> None:
    row = ScenarioViolationCounts(
        folder="folder",
        case_id="AO_WPS",
        wind_violations=wind,
        wave_violations=wave,
        land_violations=land,
        wind_penalty=wind_pen,
        wave_penalty=wave_pen,
    )

    assert row.total_violations == wind + wave + land
    assert row.total_penalty == pytest.approx(wind_pen + wave_pen)


def test_grouped_violation_table_and_csv_roundtrip(tmp_path: Path) -> None:
    rows = [
        ScenarioViolationCounts(
            folder="swopp3_no_penalty",
            case_id="AO_WPS",
            wind_violations=1,
            wave_violations=2,
            land_violations=3,
            wind_penalty=4.0,
            wave_penalty=5.0,
        ),
        ScenarioViolationCounts(
            folder="swopp3_penalty",
            case_id="AO_WPS",
            wind_violations=0,
            wave_violations=1,
            land_violations=1,
            wind_penalty=1.5,
            wave_penalty=2.5,
        ),
    ]

    table = format_grouped_violation_table(rows)
    output_path = write_grouped_violation_csv(rows, tmp_path / "violations.csv")

    with output_path.open(newline="") as handle:
        exported = list(csv.DictReader(handle))

    assert "swopp3_no_penalty" in table
    assert "swopp3_penalty" in table
    assert any(row["case_id"] == "AO_WPS" for row in exported)
    total_rows = [row for row in exported if row["case_id"] == "TOTAL"]
    assert len(total_rows) == 2
    assert total_rows[0]["total_violations"] == "6"
    assert total_rows[1]["total_penalty"] == "4.0"
