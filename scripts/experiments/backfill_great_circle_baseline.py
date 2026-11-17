#!/usr/bin/env python
"""Add an ERA5-scored Great Circle baseline to completed toy results."""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from routetools.era5.loader import load_era5_wavefield, load_era5_windfield
from routetools.swopp3 import SWOPP3_CASES, case_endpoints, great_circle_route
from routetools.swopp3_output import sailed_distance_nm
from routetools.swopp3_runner import evaluate_energy

DEFAULT_ROOT = Path("/home/SHARED/data/weather-routing")
GC_FIELDS = [
    "great_circle_truth_mwh",
    "great_circle_truth_max_tws_mps",
    "great_circle_truth_max_hs_m",
    "great_circle_distance_nm",
    "forecast_vs_great_circle_saving_mwh",
    "forecast_vs_great_circle_saving_pct",
    "truth_vs_great_circle_saving_mwh",
    "truth_vs_great_circle_saving_pct",
]


def pct(value: float, baseline: float) -> float:
    """Return a percentage relative to a nonzero baseline."""
    return 100.0 * value / baseline if baseline else float("nan")


def atomic_json(path: Path, payload: dict) -> None:
    """Replace a JSON result without exposing a partial file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Replace a route archive without exposing a partial file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(tmp, path)


def rebuild_summary(output_dir: Path) -> None:
    """Rebuild the cumulative CSV while preserving its existing field order."""
    records = [
        json.loads(path.read_text())
        for path in sorted(output_dir.glob("20??????_00z.json"))
    ]
    path = output_dir / "summary.csv"
    with path.open(newline="") as stream:
        existing_fields = next(csv.reader(stream))
    fields = [
        *existing_fields,
        *(field for field in GC_FIELDS if field not in existing_fields),
    ]
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in fields})
    os.replace(tmp, path)


def main() -> None:
    """Backfill every completed departure in an output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--era5-wind",
        type=Path,
        default=(
            DEFAULT_ROOT / "era5/2025/toy_20250715_20250724/atlantic/"
            "era5_wind_atlantic_20250715_20250808.nc"
        ),
    )
    parser.add_argument(
        "--era5-wave",
        type=Path,
        default=(
            DEFAULT_ROOT / "era5/2025/toy_20250715_20250724/atlantic/"
            "era5_waves_atlantic_20250715_20250808.nc"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_ROOT / "results/toy_20250715_20250724/AO_WPS",
    )
    args = parser.parse_args()

    for result_path in sorted(args.output_dir.glob("20??????_00z.json")):
        record = json.loads(result_path.read_text())
        departure = datetime.fromisoformat(record["departure_time_utc"])
        case_id = record["case"]
        case = SWOPP3_CASES[case_id]
        passage_hours = float(record["passage_hours"])
        end = departure + timedelta(hours=passage_hours)
        wind = load_era5_windfield(
            args.era5_wind,
            departure_time=departure,
            voyage_hours=passage_hours,
            time_start=departure,
            time_end=end,
        )
        wave = load_era5_wavefield(
            args.era5_wave,
            departure_time=departure,
            voyage_hours=passage_hours,
            time_start=departure,
            time_end=end,
        )
        src, dst = case_endpoints(case_id)
        curve = great_circle_route(src, dst, n_points=int(record["n_points"]))
        score = evaluate_energy(
            curve,
            departure,
            passage_hours,
            wps=bool(case["wps"]),
            windfield=wind,
            wavefield=wave,
        )
        great_circle_truth = float(score[0])
        forecast_saving = great_circle_truth - float(record["forecast_route_truth_mwh"])
        truth_saving = great_circle_truth - float(record["truth_route_truth_mwh"])
        record.update(
            great_circle_truth_mwh=great_circle_truth,
            great_circle_truth_max_tws_mps=float(score[1]),
            great_circle_truth_max_hs_m=float(score[2]),
            great_circle_distance_nm=float(sailed_distance_nm(curve)),
            forecast_vs_great_circle_saving_mwh=forecast_saving,
            forecast_vs_great_circle_saving_pct=pct(
                forecast_saving,
                great_circle_truth,
            ),
            truth_vs_great_circle_saving_mwh=truth_saving,
            truth_vs_great_circle_saving_pct=pct(truth_saving, great_circle_truth),
        )
        route_path = args.output_dir / f"{result_path.stem}_routes.npz"
        with np.load(route_path) as saved:
            arrays = {name: saved[name] for name in saved.files}
        arrays["great_circle_route_lon_lat"] = np.asarray(curve)
        atomic_npz(route_path, arrays)
        atomic_json(result_path, record)
        print(
            f"{result_path.stem}: GC={great_circle_truth:.3f} MWh, "
            f"forecast saving={forecast_saving:+.3f}, "
            f"perfect saving={truth_saving:+.3f}",
            flush=True,
        )

    rebuild_summary(args.output_dir)


if __name__ == "__main__":
    main()
