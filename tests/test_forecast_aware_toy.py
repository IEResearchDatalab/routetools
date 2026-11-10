"""Unit tests for the forecast-aware routing experiment driver."""

import csv
import json
from datetime import datetime
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_script():
    path = (
        Path(__file__).parents[1] / "scripts" / "experiments" / "forecast_aware_toy.py"
    )
    spec = spec_from_file_location("forecast_aware_toy", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TOY = _load_script()


def test_departures_are_inclusive() -> None:
    assert TOY._departures(datetime(2025, 7, 15), datetime(2025, 7, 17)) == [
        datetime(2025, 7, 15),
        datetime(2025, 7, 16),
        datetime(2025, 7, 17),
    ]


def test_regret_percentage() -> None:
    assert TOY._pct(5.0, 100.0) == 5.0


def test_summary_is_rebuilt_in_departure_order(tmp_path) -> None:
    base = {name: 0 for name in TOY.SUMMARY_FIELDS}
    for stamp in ("20250716_00z", "20250715_00z"):
        record = dict(base)
        record["departure_time_utc"] = stamp
        (tmp_path / f"{stamp}.json").write_text(json.dumps(record))

    summary = TOY._write_summary(tmp_path)
    with summary.open(newline="") as stream:
        rows = list(csv.DictReader(stream))

    assert [row["departure_time_utc"] for row in rows] == [
        "20250715_00z",
        "20250716_00z",
    ]
