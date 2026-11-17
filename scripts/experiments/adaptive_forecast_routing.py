#!/usr/bin/env python
"""Run one daily-replanned forecast route and score sailed segments in ERA5."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timedelta
from pathlib import Path

import jax
import numpy as np

from routetools.era5.loader import (
    load_era5_wavefield,
    load_era5_windfield,
    load_natural_earth_land_mask,
)
from routetools.fms import clear_fms_caches
from routetools.swopp3 import SWOPP3_CASES, case_endpoints
from routetools.swopp3_output import sailed_distance_nm
from routetools.swopp3_runner import evaluate_energy, run_optimised_departure

DEFAULT_ROOT = Path("/home/SHARED/data/weather-routing")


def parse_date(value: str) -> datetime:
    """Parse one midnight departure date."""
    return datetime.strptime(value, "%Y-%m-%d")


def file_sha256(path: Path) -> str:
    """Return a reproducibility hash for one forecast file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_parameter_interval(
    curve: np.ndarray,
    start: float,
    end: float,
    n_points: int,
) -> np.ndarray:
    """Sample a curve at uniformly spaced parameter times."""
    source = np.linspace(0.0, 1.0, len(curve))
    target = np.linspace(start, end, n_points)
    return np.column_stack(
        [np.interp(target, source, curve[:, axis]) for axis in range(2)]
    )


def atomic_json(path: Path, payload: dict) -> None:
    """Publish a JSON result atomically."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Publish route arrays atomically."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(tmp, path)


def run(args: argparse.Namespace) -> None:
    """Run and save one adaptive voyage."""
    case = SWOPP3_CASES[args.case]
    total_hours = float(case["passage_hours"])
    departure = args.departure
    stamp = departure.strftime("%Y%m%d_00z")
    monthly_json = args.monthly_results / f"{stamp}.json"
    monthly_routes = args.monthly_results / f"{stamp}_routes.npz"
    if not monthly_json.exists() or not monthly_routes.exists():
        raise FileNotFoundError("monthly a priori/perfect result must exist first")
    baseline = json.loads(monthly_json.read_text())
    with np.load(monthly_routes) as saved:
        first_plan = np.asarray(saved["forecast_route_lon_lat"], dtype=float)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / f"{stamp}_adaptive.json"
    route_path = args.output_dir / f"{stamp}_adaptive_routes.npz"
    if result_path.exists() and route_path.exists() and not args.force:
        print(f"{stamp}: adaptive result already complete", flush=True)
        return

    truth_wind = load_era5_windfield(
        args.era5_wind,
        departure_time=departure,
        voyage_hours=total_hours,
        time_start=departure,
        time_end=departure + timedelta(hours=total_hours),
    )
    truth_wave = load_era5_wavefield(
        args.era5_wave,
        departure_time=departure,
        voyage_hours=total_hours,
        time_start=departure,
        time_end=departure + timedelta(hours=total_hours),
    )
    land = load_natural_earth_land_mask((-80.0, 10.0), (25.0, 60.0))
    _, destination = case_endpoints(args.case)
    plan = first_plan
    elapsed = 0.0
    realized_energy = 0.0
    max_tws = 0.0
    max_hs = 0.0
    plans: dict[str, np.ndarray] = {}
    segments: dict[str, np.ndarray] = {}
    daily: list[dict] = []
    assembled: list[np.ndarray] = []
    day = 0

    while elapsed < total_hours - 1e-9:
        decision = departure + timedelta(hours=elapsed)
        remaining = total_hours - elapsed
        duration = min(24.0, remaining)
        forecast_stamp = decision.strftime("%Y%m%d_00z")
        wind_path = args.forecast_root / "oper" / f"{forecast_stamp}.nc"
        wave_path = args.forecast_root / "wave" / f"{forecast_stamp}.nc"
        if not wind_path.exists() or not wave_path.exists():
            raise FileNotFoundError(f"missing daily forecast pair: {forecast_stamp}")

        if day > 0:
            forecast_wind = load_era5_windfield(
                wind_path, departure_time=decision, voyage_hours=remaining
            )
            forecast_wave = load_era5_wavefield(
                wave_path, departure_time=decision, voyage_hours=remaining
            )
            result = run_optimised_departure(
                args.case,
                decision,
                vectorfield=forecast_wind,
                windfield=forecast_wind,
                wavefield=forecast_wave,
                land=land,
                n_points=args.n_points,
                verbosity=1,
                route_src=plan[0],
                route_dst=destination,
                route_travel_time=remaining,
                initial_curve=plan,
                K=args.cmaes_k,
                popsize=args.popsize,
                maxfevals=args.maxfevals,
                seed=args.seed + day,
                fms_patience=args.fms_patience,
                fms_maxfevals=args.fms_maxfevals,
            )
            plan = np.asarray(result.curve, dtype=float)
            predicted_energy = float(result.energy_mwh)
        else:
            predicted_energy = float(baseline["forecast_route_forecast_mwh"])

        plans[f"plan_day_{day:02d}_lon_lat"] = plan
        fraction = duration / remaining
        segment_points = max(9, int(math.ceil(duration)) + 1)
        segment = sample_parameter_interval(plan, 0.0, fraction, segment_points)
        segments[f"sailed_day_{day:02d}_lon_lat"] = segment
        score = evaluate_energy(
            segment,
            decision,
            duration,
            wps=bool(case["wps"]),
            windfield=truth_wind,
            wavefield=truth_wave,
            departure_offset_h=elapsed,
        )
        realized_energy += float(score[0])
        max_tws = max(max_tws, float(score[1]))
        max_hs = max(max_hs, float(score[2]))
        if not assembled:
            assembled.append(segment)
        else:
            assembled.append(segment[1:])
        daily.append(
            {
                "day": day,
                "decision_time_utc": decision.isoformat(),
                "remaining_hours": remaining,
                "sailed_hours": duration,
                "forecast_wind_file": str(wind_path),
                "forecast_wave_file": str(wave_path),
                "forecast_wind_sha256": file_sha256(wind_path),
                "forecast_wave_sha256": file_sha256(wave_path),
                "forecast_predicted_remaining_mwh": predicted_energy,
                "era5_realized_segment_mwh": float(score[0]),
            }
        )

        elapsed += duration
        if elapsed < total_hours - 1e-9:
            plan = sample_parameter_interval(plan, fraction, 1.0, args.n_points)
            plan[0] = segment[-1]
            plan[-1] = np.asarray(destination)
        clear_fms_caches()
        if hasattr(jax, "clear_caches"):
            jax.clear_caches()
        day += 1

    adaptive_route = np.vstack(assembled)
    gc = float(baseline["great_circle_truth_mwh"])
    apriori = float(baseline["forecast_route_truth_mwh"])
    perfect = float(baseline["truth_route_truth_mwh"])
    payload = {
        "departure_time_utc": departure.isoformat(),
        "case": args.case,
        "passage_hours": total_hours,
        "continuity": "C0 position continuity; tangent is not constrained",
        "forecast_policy": "new deterministic IFS oper+wave 00Z run each day",
        "forecast_availability_assumption": "00Z run treated as available at 00 UTC",
        "great_circle_truth_mwh": gc,
        "apriori_truth_mwh": apriori,
        "adaptive_truth_mwh": realized_energy,
        "perfect_information_truth_mwh": perfect,
        "apriori_vs_gc_saving_mwh": gc - apriori,
        "adaptive_vs_gc_saving_mwh": gc - realized_energy,
        "perfect_vs_gc_saving_mwh": gc - perfect,
        "adaptive_vs_apriori_mwh": realized_energy - apriori,
        "adaptive_information_regret_mwh": realized_energy - perfect,
        "adaptive_distance_nm": float(sailed_distance_nm(adaptive_route)),
        "adaptive_truth_max_tws_mps": max_tws,
        "adaptive_truth_max_hs_m": max_hs,
        "daily": daily,
        "seed": args.seed,
        "n_points": args.n_points,
        "cmaes_k": args.cmaes_k,
        "popsize": args.popsize,
        "maxfevals": args.maxfevals,
        "fms_patience": args.fms_patience,
        "fms_maxfevals": args.fms_maxfevals,
    }
    arrays = {**plans, **segments, "adaptive_route_lon_lat": adaptive_route}
    atomic_npz(route_path, arrays)
    atomic_json(result_path, payload)
    print(
        json.dumps(
            {
                k: payload[k]
                for k in (
                    "apriori_truth_mwh",
                    "adaptive_truth_mwh",
                    "perfect_information_truth_mwh",
                    "adaptive_vs_apriori_mwh",
                )
            },
            indent=2,
        ),
        flush=True,
    )


def parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--departure", type=parse_date, required=True)
    p.add_argument("--case", default="AO_WPS")
    p.add_argument("--forecast-root", type=Path, required=True)
    p.add_argument("--era5-wind", type=Path, required=True)
    p.add_argument("--era5-wave", type=Path, required=True)
    p.add_argument("--monthly-results", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-points", type=int, default=120)
    p.add_argument("--cmaes-k", type=int, default=10)
    p.add_argument("--popsize", type=int, default=50)
    p.add_argument("--maxfevals", type=int, default=1000)
    p.add_argument("--fms-patience", type=int, default=15)
    p.add_argument("--fms-maxfevals", type=int, default=500)
    p.add_argument("--force", action="store_true")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
