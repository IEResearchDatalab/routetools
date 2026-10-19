import csv
from pathlib import Path

import numpy as np

from revision.task5_env_penalty_sensitivity import (
    STRESS_DEPARTURES,
    build_summary_table,
    prepare_stress_subset,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_prepare_stress_subset_selects_exact_dates(tmp_path):
    source = tmp_path / "source"
    tracks = source / "tracks"
    tracks.mkdir(parents=True)
    for case_id, dates in STRESS_DEPARTURES.items():
        rows = []
        for date in (*dates, "2024-06-01"):
            filename = f"IEUniversity-1-{case_id}-{date.replace('-', '')}.csv"
            rows.append(
                {
                    "departure_time_utc": f"{date} 12:00:00",
                    "details_filename": filename,
                }
            )
            (tracks / filename).write_text("lat_deg,lon_deg\n0,0\n")
        _write_csv(source / f"IEUniversity-1-{case_id}.csv", rows)

    subset = prepare_stress_subset(source, tmp_path / "subset")

    assert len(list(subset.glob("IEUniversity-1-*.csv"))) == 4
    assert len(list((subset / "tracks").glob("*.csv"))) == 8
    for case_id, dates in STRESS_DEPARTURES.items():
        with (subset / f"IEUniversity-1-{case_id}.csv").open(newline="") as handle:
            selected = list(csv.DictReader(handle))
        assert {row["departure_time_utc"][:10] for row in selected} == set(dates)


def test_summary_uses_paired_route_results():
    rows = []
    for weight, any_violations, energy_changes in (
        (25.0, (True, False), (2.0, -4.0)),
        (50.0, (False, False), (1.0, -3.0)),
    ):
        for index in range(2):
            rows.append(
                {
                    "weather_penalty_weight": weight,
                    "energy_change_vs_cmaes_pct": energy_changes[index],
                    "wind_exceedance_mps": 0.2 if any_violations[index] else 0.0,
                    "wave_exceedance_m": 0.0,
                    "any_violation": any_violations[index],
                }
            )

    summary = build_summary_table(rows)

    assert [row["n_with_any_violation"] for row in summary] == [1, 0]
    np.testing.assert_allclose(
        [row["mean_energy_change_vs_cmaes_pct"] for row in summary],
        [-1.0, -1.0],
    )
