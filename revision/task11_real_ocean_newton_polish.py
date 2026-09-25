r"""Task 11 (R1-2) — distance of the real-ocean BERS routes to a local minimum.

Task 8 showed that the exported real-ocean routes do not meet a stringent
full-route stationarity tolerance, although many satisfy the second-order
conditions.  Task 10 showed that, for the synthetic fields, the same FMS
behaviour leaves routes one or two coupled Newton steps away from a
numerically certified local minimum, with a negligible change in cost.

This script applies that same diagnostic to the real-ocean routes.  Starting
from every exported BERS route, it runs a safeguarded coupled Newton
iteration on the *complete* block-tridiagonal Hessian of exactly the action
audited by Task 8 (propulsive energy plus the smooth wind/wave penalties,
fixed endpoints, fixed segment durations):

* where the Hessian is positive definite, the pure Newton step is used;
* otherwise a Levenberg--Marquardt shift ``H + mu I`` makes the step a descent
  direction;
* every step is backtracked until the action does not increase *and* the
  sampled land-violation count does not increase (the FMS rollback rule).

The routes reported in the paper are **not** replaced.  The script measures
how far each exported route is from the nearest point that passes the Task 8
certificate (positive-definite complete Hessian and coupled Newton correction
below tolerance), and how much energy that distance is worth.

Run on the server after Task 8, CPU only::

    JAX_PLATFORMS=cpu uv run python revision/task11_real_ocean_newton_polish.py \\
      --real-ocean-dir output/sweep_combined_fms_strict \\
      --experiment-manifest revision/task8_sweep_combined_fms_strict_manifest.json \\
      --tws-limit 19.9 --hs-limit 6.9 --land-distance-weight 0 \\
      --output-dir output/task11_newton_polish

Outputs (under ``--output-dir``)
--------------------------------
- ``task11_polish_routes.csv``: one row per route.
- ``task11_polish_summary.csv`` / ``.tex``: aggregate per case.
- ``--checkpoint-dir/*.json``: resumable per-route checkpoints.
"""

from __future__ import annotations

import csv
import gc
import json
import math
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import typer

sys.path.insert(0, str(Path(__file__).resolve().parent))

from task8_local_optimality import (  # noqa: E402
    OPTIMIZED_CASES,
    AuditSettings,
    _build_land_mask,
    _build_segment_derivative_function,
    _load_weather_batch,
    _monthly_batches,
    _read_case_rows,
    _resolve_path,
    _validate_experiment_manifest,
    assemble_interior_blocks,
    factor_spd_block_tridiagonal,
    scale_blocks_to_meters,
    solve_block_ldlt,
)

from routetools.swopp3 import SWOPP3_CASES  # noqa: E402
from routetools.swopp3_runner import _penalized_rise_cost  # noqa: E402
from routetools.violations import (  # noqa: E402
    departure_offset_hours,
    normalise_route_longitudes,
    read_track_curve,
)
from routetools.weather import DEFAULT_HS_LIMIT, DEFAULT_TWS_LIMIT  # noqa: E402

jax.config.update("jax_enable_x64", True)

EARTH_RADIUS_M = 6_371_008.8
M_PER_DEG = EARTH_RADIUS_M * math.pi / 180.0


def _local_model(curve, derivative_function, offsets, settings):
    flat = jnp.concatenate([curve[:-1], curve[1:]], axis=1)
    values, grads, hessians = derivative_function(flat, offsets)
    gradient, diagonal, upper = assemble_interior_blocks(
        np.asarray(grads), np.asarray(hessians)
    )
    gradient_m, diagonal_m, upper_m = scale_blocks_to_meters(
        gradient, diagonal, upper, np.asarray(curve[1:-1, 1])
    )
    factorization = factor_spd_block_tridiagonal(
        diagonal_m, upper_m, relative_tolerance=settings.pd_relative_tolerance
    )
    return (
        float(np.sum(np.asarray(values))),
        gradient_m,
        diagonal_m,
        upper_m,
        factorization,
    )


def _metres_to_degrees(step_m: np.ndarray, latitudes: np.ndarray) -> np.ndarray:
    out = np.empty_like(step_m)
    out[:, 0] = step_m[:, 0] / (M_PER_DEG * np.cos(np.radians(latitudes)))
    out[:, 1] = step_m[:, 1] / M_PER_DEG
    return out


def _shifted_step(gradient_m, diagonal_m, upper_m, settings):
    """Levenberg--Marquardt step for an indefinite block Hessian."""
    scale = max(float(np.max(np.abs(diagonal_m))), np.finfo(float).tiny)
    mu = 1e-8 * scale
    eye = np.eye(2)[None, :, :]
    for _ in range(80):
        factorization = factor_spd_block_tridiagonal(
            diagonal_m + mu * eye,
            upper_m,
            relative_tolerance=settings.pd_relative_tolerance,
        )
        if factorization.is_positive_definite:
            return solve_block_ldlt(factorization, -gradient_m)
        mu *= 4.0
    return None


def _energy(curve, windfield, wavefield, travel_time, wps, offset, settings):
    return float(
        _penalized_rise_cost(
            curve=curve[None, ...],
            windfield=windfield,
            wavefield=wavefield,
            travel_time=travel_time,
            wps=wps,
            spherical_correction=settings.spherical_correction,
            time_offset=offset,
            weather_penalty_weight=0.0,
        )[0]
    )


def polish_route(
    *,
    curve,
    derivative_function,
    offsets,
    land,
    windfield,
    wavefield,
    travel_time_h,
    wps,
    departure_offset_h,
    settings,
    max_iterations,
):
    """Polish one exported route and report certificate diagnostics."""
    curve = jnp.asarray(curve, dtype=jnp.float64)
    base_land = int(np.sum(np.asarray(land(curve[None, ...]))))
    segment_m = np.hypot(
        np.diff(np.asarray(curve[:, 0]))
        * M_PER_DEG
        * np.cos(np.radians(np.asarray(curve[:-1, 1]))),
        np.diff(np.asarray(curve[:, 1])) * M_PER_DEG,
    )
    median_segment_m = float(np.median(segment_m))
    tolerance_m = min(
        settings.stationarity_abs_m, settings.stationarity_relative * median_segment_m
    )
    energy0 = _energy(
        curve, windfield, wavefield, travel_time_h, wps, departure_offset_h, settings
    )
    action0, g, d, u, fac = _local_model(curve, derivative_function, offsets, settings)
    action = action0
    current = curve
    iterations = 0
    pd = fac.is_positive_definite
    correction = (
        float(np.max(np.linalg.norm(solve_block_ldlt(fac, -g), axis=1)))
        if pd
        else math.nan
    )
    initial_pd, initial_correction = pd, correction
    while iterations < max_iterations and not (pd and correction <= tolerance_m):
        step_m = solve_block_ldlt(fac, -g) if pd else _shifted_step(g, d, u, settings)
        if step_m is None:
            break
        step_deg = _metres_to_degrees(step_m, np.asarray(current[1:-1, 1]))
        alpha = 1.0
        accepted = False
        for _ in range(30):
            trial = current.at[1:-1].add(jnp.asarray(alpha * step_deg))
            trial_land = int(np.sum(np.asarray(land(trial[None, ...]))))
            if trial_land <= base_land:
                trial_action, tg, td, tu, tfac = _local_model(
                    trial, derivative_function, offsets, settings
                )
                if np.isfinite(trial_action) and trial_action <= action:
                    accepted = True
                    break
            alpha *= 0.5
        if not accepted:
            break
        current, action, g, d, u, fac = trial, trial_action, tg, td, tu, tfac
        iterations += 1
        pd = fac.is_positive_definite
        correction = (
            float(np.max(np.linalg.norm(solve_block_ldlt(fac, -g), axis=1)))
            if pd
            else math.nan
        )

    shift_deg = np.asarray(current - curve)
    shift_m = np.hypot(
        shift_deg[:, 0] * M_PER_DEG * np.cos(np.radians(np.asarray(curve[:, 1]))),
        shift_deg[:, 1] * M_PER_DEG,
    )
    energy1 = _energy(
        current, windfield, wavefield, travel_time_h, wps, departure_offset_h, settings
    )
    certified = bool(pd and correction <= tolerance_m)
    return {
        "initial_hessian_pd": initial_pd,
        "initial_max_newton_correction_m": initial_correction,
        "stationarity_limit_m": tolerance_m,
        "median_segment_length_m": median_segment_m,
        "iterations": iterations,
        "initial_action": action0,
        "polished_action": action,
        "relative_action_change": (action - action0) / abs(action0),
        "initial_energy_mwh": energy0,
        "polished_energy_mwh": energy1,
        "relative_energy_change": (energy1 - energy0) / abs(energy0),
        "max_shift_m": float(np.max(shift_m)),
        "max_shift_over_segment": float(np.max(shift_m)) / median_segment_m,
        "polished_hessian_pd": bool(pd),
        "polished_max_newton_correction_m": correction,
        "certified_after_polish": certified,
        "sampled_land_violations": base_land,
    }


def main(
    real_ocean_dir: str = "output/sweep_combined_fms_strict",
    output_dir: str = "output/task11_newton_polish",
    checkpoint_dir: str = "output/task11_newton_polish/checkpoints",
    experiment_manifest: str = "revision/task8_sweep_combined_fms_strict_manifest.json",
    wind_path_atlantic: str = "data/era5/era5_wind_atlantic_2024.nc",
    wave_path_atlantic: str = "data/era5/era5_waves_atlantic_2024.nc",
    wind_path_pacific: str = "data/era5/era5_wind_pacific_2024.nc",
    wave_path_pacific: str = "data/era5/era5_waves_pacific_2024.nc",
    weather_penalty_weight: float = 0.0,
    wind_penalty_weight: float = 50.0,
    wave_penalty_weight: float = 50.0,
    tws_limit: float = DEFAULT_TWS_LIMIT,
    hs_limit: float = DEFAULT_HS_LIMIT,
    weather_penalty_sharpness: float = 5.0,
    land_distance_weight: float = 100.0,
    land_distance_epsilon: float = 1.0,
    distance_penalty_weight: float = 0.0,
    stationarity_abs_m: float = 100.0,
    stationarity_relative: float = 1.0e-3,
    pd_relative_tolerance: float = 1.0e-10,
    max_iterations: int = 50,
    cases: str = ",".join(OPTIMIZED_CASES),
    max_routes: int = 0,
) -> None:
    """Polish every stored BERS route with safeguarded coupled Newton steps."""
    input_dir = _resolve_path(real_ocean_dir)
    out_dir = _resolve_path(output_dir)
    checkpoints = _resolve_path(checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    settings = AuditSettings(
        weather_penalty_weight=weather_penalty_weight,
        wind_penalty_weight=wind_penalty_weight,
        wave_penalty_weight=wave_penalty_weight,
        tws_limit=tws_limit,
        hs_limit=hs_limit,
        weather_penalty_sharpness=weather_penalty_sharpness,
        land_distance_weight=land_distance_weight,
        land_distance_epsilon=land_distance_epsilon,
        distance_penalty_weight=distance_penalty_weight,
        stationarity_abs_m=stationarity_abs_m,
        stationarity_relative=stationarity_relative,
        pd_relative_tolerance=pd_relative_tolerance,
    )
    manifest = _resolve_path(experiment_manifest) if experiment_manifest else None
    _validate_experiment_manifest(input_dir, settings, manifest)
    weather_paths = {
        "atlantic": (
            _resolve_path(wind_path_atlantic),
            _resolve_path(wave_path_atlantic),
        ),
        "pacific": (_resolve_path(wind_path_pacific), _resolve_path(wave_path_pacific)),
    }
    wanted = [c for c in cases.split(",") if c]
    case_rows = {c: _read_case_rows(input_dir, c) for c in wanted}
    done = 0
    for corridor in ("atlantic", "pacific"):
        corridor_cases = [c for c in wanted if SWOPP3_CASES[c]["route"] == corridor]
        if not corridor_cases:
            continue
        departures = sorted(
            {r["departure"] for c in corridor_cases for r in case_rows[c]}
        )
        wind_path, wave_path = weather_paths[corridor]
        land = _build_land_mask(wave_path)
        for start, end, batch in _monthly_batches(departures):
            epoch, windfield, wavefield = _load_weather_batch(
                wind_path, wave_path, start, end
            )
            batch_set = set(batch)
            functions = {}
            for case_id in corridor_cases:
                case = SWOPP3_CASES[case_id]
                for row in case_rows[case_id]:
                    if row["departure"] not in batch_set:
                        continue
                    route_id = str(row["details_filename"])
                    ckpt = checkpoints / f"{Path(route_id).stem}.json"
                    if ckpt.exists():
                        done += 1
                        continue
                    curve = jnp.asarray(
                        normalise_route_longitudes(
                            read_track_curve(input_dir / "tracks" / route_id),
                            windfield.longitude_bounds,
                        ),
                        dtype=jnp.float64,
                    )
                    travel_time_h = float(case["passage_hours"])
                    n_seg = curve.shape[0] - 1
                    segment_h = travel_time_h / n_seg
                    key = (bool(case["wps"]), segment_h)
                    if key not in functions:
                        functions[key] = _build_segment_derivative_function(
                            windfield=windfield,
                            wavefield=wavefield,
                            land=land,
                            segment_hours=segment_h,
                            wps=bool(case["wps"]),
                            settings=settings,
                        )
                    offset_h = departure_offset_hours(row["departure"], epoch)
                    offsets = jnp.asarray(
                        offset_h + np.arange(n_seg) * segment_h, dtype=jnp.float64
                    )
                    result = polish_route(
                        curve=curve,
                        derivative_function=functions[key],
                        offsets=offsets,
                        land=land,
                        windfield=windfield,
                        wavefield=wavefield,
                        travel_time_h=travel_time_h,
                        wps=bool(case["wps"]),
                        departure_offset_h=offset_h,
                        settings=settings,
                        max_iterations=max_iterations,
                    )
                    result = {"case_id": case_id, "route_id": route_id, **result}
                    ckpt.write_text(json.dumps(result, indent=1))
                    done += 1
                    print(
                        f"[{done}] {route_id} it={result['iterations']} "
                        f"dE={result['relative_energy_change']:.2e} "
                        f"shift={result['max_shift_m'] / 1000:.1f} km "
                        f"certified={result['certified_after_polish']}"
                    )
                    if max_routes and done >= max_routes:
                        break
                if max_routes and done >= max_routes:
                    break
            del functions, windfield, wavefield
            gc.collect()
            jax.clear_caches()
            if max_routes and done >= max_routes:
                break
        if max_routes and done >= max_routes:
            break

    rows = [json.loads(p.read_text()) for p in sorted(checkpoints.glob("*.json"))]
    if not rows:
        print("No routes polished.")
        return
    with (out_dir / "task11_polish_routes.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for case_id in wanted:
        grp = [r for r in rows if r["case_id"] == case_id]
        if not grp:
            continue
        cert = [r for r in grp if r["certified_after_polish"]]
        de = (
            np.array([r["relative_energy_change"] for r in cert])
            if cert
            else np.array([np.nan])
        )
        sh = (
            np.array([r["max_shift_over_segment"] for r in cert])
            if cert
            else np.array([np.nan])
        )
        summary.append(
            {
                "case_id": case_id,
                "n_routes": len(grp),
                "certified_after_polish": len(cert),
                "median_iterations": float(np.median([r["iterations"] for r in cert]))
                if cert
                else math.nan,
                "median_rel_energy_change_certified": float(np.nanmedian(de)),
                "max_abs_rel_energy_change_certified": float(np.nanmax(np.abs(de))),
                "median_shift_over_segment_certified": float(np.nanmedian(sh)),
                "max_shift_over_segment_certified": float(np.nanmax(sh)),
            }
        )
    with (out_dir / "task11_polish_summary.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    tex = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Case & $N$ & Certified after polish & "
        r"Median $|\Delta E|/E$ & Median shift / segment \\",
        r"\midrule",
    ]
    for s in summary:
        tex.append(
            f"{s['case_id']} & {s['n_routes']} & {s['certified_after_polish']} & "
            f"{abs(s['median_rel_energy_change_certified']):.1e} & "
            f"{s['median_shift_over_segment_certified']:.2f} \\\\"
        )
    tex += [r"\bottomrule", r"\end{tabular}"]
    (out_dir / "task11_polish_summary.tex").write_text("\n".join(tex) + "\n")
    print(f"Wrote Task 11 outputs to {out_dir}")


if __name__ == "__main__":
    typer.run(main)
