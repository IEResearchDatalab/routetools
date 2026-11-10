#!/usr/bin/env python
"""Run the ten-departure forecast-versus-perfect-information toy experiment.

For every departure this script performs two otherwise identical optimisations:

1. optimise with the deterministic IFS forecast available at departure;
2. optimise with ERA5 reanalysis (the perfect-information reference).

Both final routes are then scored on both weather products.  The primary
quantity is ``information_regret_mwh``: the ERA5 energy of the forecast route
minus the ERA5 energy of the perfect-information route.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import jax
import numpy as np

from routetools.era5.loader import (
    load_era5_wavefield,
    load_era5_windfield,
    load_natural_earth_land_mask,
)
from routetools.fms import clear_fms_caches
from routetools.swopp3 import SWOPP3_CASES
from routetools.swopp3_runner import evaluate_energy, run_optimised_departure

DEFAULT_DATA_ROOT = Path("/home/SHARED/data/weather-routing")
DEFAULT_FORECAST_ROOT = (
    DEFAULT_DATA_ROOT / "ecmwf-open-data/2025/toy_20250715_20250724/atlantic"
)
DEFAULT_ERA5_ROOT = DEFAULT_DATA_ROOT / "era5/2025/toy_20250715_20250724/atlantic"
DEFAULT_OUTPUT_ROOT = DEFAULT_DATA_ROOT / "results/toy_20250715_20250724/AO_WPS"

SUMMARY_FIELDS = [
    "departure_time_utc",
    "case",
    "passage_hours",
    "forecast_route_forecast_mwh",
    "forecast_route_truth_mwh",
    "truth_route_forecast_mwh",
    "truth_route_truth_mwh",
    "information_regret_mwh",
    "information_regret_pct",
    "forecast_error_mwh",
    "forecast_error_pct",
    "forecast_route_distance_nm",
    "truth_route_distance_nm",
    "forecast_route_truth_max_tws_mps",
    "forecast_route_truth_max_hs_m",
    "truth_route_truth_max_tws_mps",
    "truth_route_truth_max_hs_m",
    "forecast_optimisation_s",
    "truth_optimisation_s",
    "seed",
    "n_points",
    "cmaes_k",
    "maxfevals",
    "popsize",
    "fms_patience",
    "fms_maxfevals",
]


def _parse_date(value: str) -> datetime:
    """Parse an ISO date as a midnight UTC-naive datetime."""
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected YYYY-MM-DD") from exc


def _departures(start: datetime, end: datetime) -> list[datetime]:
    """Return all midnight departures in the inclusive date interval."""
    if end < start:
        raise ValueError("end date must be on or after start date")
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON without exposing a partially written result."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _atomic_routes(
    path: Path,
    forecast_route: np.ndarray,
    truth_route: np.ndarray,
) -> None:
    """Write the paired routes as a compressed, atomically published NPZ."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(
            stream,
            forecast_route_lon_lat=forecast_route,
            truth_route_lon_lat=truth_route,
        )
    os.replace(tmp, path)


def _write_summary(output_dir: Path) -> Path:
    """Rebuild the summary CSV from completed per-departure JSON files."""
    records = [
        json.loads(path.read_text())
        for path in sorted(output_dir.glob("20??????_00z.json"))
    ]
    summary = output_dir / "summary.csv"
    tmp = summary.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({name: record[name] for name in SUMMARY_FIELDS})
    os.replace(tmp, summary)
    return summary


def _pct(numerator: float, denominator: float) -> float:
    """Return a percentage, or NaN for a zero denominator."""
    return 100.0 * numerator / denominator if denominator else float("nan")


def _score(
    curve,
    departure: datetime,
    case: dict[str, Any],
    windfield,
    wavefield,
) -> tuple[float, float, float]:
    """Score one route with a weather-product pair."""
    return evaluate_energy(
        curve,
        departure,
        float(case["passage_hours"]),
        wps=bool(case["wps"]),
        windfield=windfield,
        wavefield=wavefield,
    )


def _clear_jax_state() -> None:
    """Release field closures and compiled objectives between departures."""
    clear_fms_caches()
    gc.collect()
    if hasattr(jax, "clear_caches"):
        jax.clear_caches()


def run_departure(
    departure: datetime,
    *,
    case_id: str,
    forecast_root: Path,
    era5_wind: Path,
    era5_wave: Path,
    output_dir: Path,
    land,
    seed: int,
    n_points: int,
    cmaes_k: int,
    popsize: int,
    maxfevals: int,
    fms_patience: int,
    fms_maxfevals: int,
    force: bool,
) -> dict[str, Any]:
    """Optimise and cross-score one forecast/truth departure pair."""
    stamp = departure.strftime("%Y%m%d_00z")
    result_path = output_dir / f"{stamp}.json"
    route_path = output_dir / f"{stamp}_routes.npz"
    if result_path.exists() and route_path.exists() and not force:
        print(f"[{stamp}] already complete; skipping", flush=True)
        return json.loads(result_path.read_text())

    case = SWOPP3_CASES[case_id]
    passage_hours = float(case["passage_hours"])
    forecast_wind_path = forecast_root / "oper" / f"{stamp}.nc"
    forecast_wave_path = forecast_root / "wave" / f"{stamp}.nc"
    for path in (forecast_wind_path, forecast_wave_path, era5_wind, era5_wave):
        if not path.exists():
            raise FileNotFoundError(path)

    print(f"[{stamp}] loading IFS and ERA5 fields", flush=True)
    forecast_wind = load_era5_windfield(
        forecast_wind_path,
        departure_time=departure,
        voyage_hours=passage_hours,
    )
    forecast_wave = load_era5_wavefield(
        forecast_wave_path,
        departure_time=departure,
        voyage_hours=passage_hours,
    )
    truth_end = departure + timedelta(hours=passage_hours)
    truth_wind = load_era5_windfield(
        era5_wind,
        departure_time=departure,
        voyage_hours=passage_hours,
        time_start=departure,
        time_end=truth_end,
    )
    truth_wave = load_era5_wavefield(
        era5_wave,
        departure_time=departure,
        voyage_hours=passage_hours,
        time_start=departure,
        time_end=truth_end,
    )

    optimizer = {
        "K": cmaes_k,
        "popsize": popsize,
        "maxfevals": maxfevals,
        "seed": seed,
        "fms_patience": fms_patience,
        "fms_maxfevals": fms_maxfevals,
    }
    print(f"[{stamp}] optimising under the departure forecast", flush=True)
    forecast_result = run_optimised_departure(
        case_id,
        departure,
        vectorfield=forecast_wind,
        windfield=forecast_wind,
        wavefield=forecast_wave,
        land=land,
        n_points=n_points,
        verbosity=1,
        **optimizer,
    )
    print(f"[{stamp}] optimising with perfect-information ERA5", flush=True)
    truth_result = run_optimised_departure(
        case_id,
        departure,
        vectorfield=truth_wind,
        windfield=truth_wind,
        wavefield=truth_wave,
        land=land,
        n_points=n_points,
        verbosity=1,
        **optimizer,
    )

    forecast_on_truth = _score(
        forecast_result.curve,
        departure,
        case,
        truth_wind,
        truth_wave,
    )
    truth_on_forecast = _score(
        truth_result.curve,
        departure,
        case,
        forecast_wind,
        forecast_wave,
    )
    forecast_expected = float(forecast_result.energy_mwh)
    forecast_realized = float(forecast_on_truth[0])
    truth_perfect = float(truth_result.energy_mwh)
    information_regret = forecast_realized - truth_perfect
    forecast_error = forecast_realized - forecast_expected

    record: dict[str, Any] = {
        "departure_time_utc": departure.strftime("%Y-%m-%dT%H:%M:%S"),
        "case": case_id,
        "passage_hours": passage_hours,
        "forecast_route_forecast_mwh": forecast_expected,
        "forecast_route_truth_mwh": forecast_realized,
        "truth_route_forecast_mwh": float(truth_on_forecast[0]),
        "truth_route_truth_mwh": truth_perfect,
        "information_regret_mwh": information_regret,
        "information_regret_pct": _pct(information_regret, truth_perfect),
        "forecast_error_mwh": forecast_error,
        "forecast_error_pct": _pct(forecast_error, forecast_expected),
        "forecast_route_distance_nm": float(forecast_result.distance_nm),
        "truth_route_distance_nm": float(truth_result.distance_nm),
        "forecast_route_truth_max_tws_mps": float(forecast_on_truth[1]),
        "forecast_route_truth_max_hs_m": float(forecast_on_truth[2]),
        "truth_route_truth_max_tws_mps": float(truth_result.max_tws_mps),
        "truth_route_truth_max_hs_m": float(truth_result.max_hs_m),
        "forecast_optimisation_s": float(forecast_result.comp_time_s),
        "truth_optimisation_s": float(truth_result.comp_time_s),
        "seed": seed,
        "n_points": n_points,
        "cmaes_k": cmaes_k,
        "maxfevals": maxfevals,
        "popsize": popsize,
        "fms_patience": fms_patience,
        "fms_maxfevals": fms_maxfevals,
    }
    _atomic_routes(
        route_path,
        np.asarray(forecast_result.curve),
        np.asarray(truth_result.curve),
    )
    _atomic_json(result_path, record)
    print(
        f"[{stamp}] truth regret={information_regret:+.3f} MWh "
        f"({_pct(information_regret, truth_perfect):+.3f}%)",
        flush=True,
    )
    return record


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=_parse_date, default=_parse_date("2025-07-15"))
    parser.add_argument("--end", type=_parse_date, default=_parse_date("2025-07-24"))
    parser.add_argument("--case", default="AO_WPS")
    parser.add_argument("--forecast-root", type=Path, default=DEFAULT_FORECAST_ROOT)
    parser.add_argument(
        "--era5-wind",
        type=Path,
        default=DEFAULT_ERA5_ROOT / "era5_wind_atlantic_20250715_20250808.nc",
    )
    parser.add_argument(
        "--era5-wave",
        type=Path,
        default=DEFAULT_ERA5_ROOT / "era5_waves_atlantic_20250715_20250808.nc",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-points", type=int, default=180)
    parser.add_argument("--cmaes-k", type=int, default=10)
    parser.add_argument("--popsize", type=int, default=100)
    parser.add_argument("--maxfevals", type=int, default=5000)
    parser.add_argument("--fms-patience", type=int, default=30)
    parser.add_argument("--fms-maxfevals", type=int, default=2000)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    """Run all requested departures and publish a cumulative summary."""
    args = build_parser().parse_args()
    if args.case not in SWOPP3_CASES:
        raise ValueError(f"unknown SWOPP3 case: {args.case}")
    case = SWOPP3_CASES[args.case]
    if case["route"] != "atlantic" or case["strategy"] != "optimised":
        raise ValueError("the toy runner requires an optimised Atlantic case")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"JAX devices: {jax.devices()}", flush=True)
    print("Building the Atlantic Natural Earth land mask", flush=True)
    land = load_natural_earth_land_mask((-80.0, 10.0), (25.0, 60.0))

    for departure in _departures(args.start, args.end):
        try:
            run_departure(
                departure,
                case_id=args.case,
                forecast_root=args.forecast_root,
                era5_wind=args.era5_wind,
                era5_wave=args.era5_wave,
                output_dir=args.output_dir,
                land=land,
                seed=args.seed,
                n_points=args.n_points,
                cmaes_k=args.cmaes_k,
                popsize=args.popsize,
                maxfevals=args.maxfevals,
                fms_patience=args.fms_patience,
                fms_maxfevals=args.fms_maxfevals,
                force=args.force,
            )
        finally:
            _clear_jax_state()
        summary = _write_summary(args.output_dir)
        print(f"Updated {summary}", flush=True)


if __name__ == "__main__":
    main()
