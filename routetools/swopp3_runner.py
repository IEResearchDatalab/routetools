"""SWOPP3 runner — execute all 8 cases × 366 departures.

Orchestrates the end-to-end pipeline:

1. **Great-Circle (GC) cases** — fixed geodesic route, constant speed,
   energy evaluated via the RISE performance model.
2. **Optimised (O) cases** — CMA-ES route optimisation followed by energy
   evaluation.

Both modes support WPS (wingsails on) and noWPS (engine only).

Main entry points:

- :func:`evaluate_energy` — evaluate energy along a route.
- :func:`run_gc_departure` — single GC departure.
- :func:`run_optimised_departure` — single optimised departure.
- :func:`run_case` — all departures for one case.
"""

from __future__ import annotations

import logging
import time as _time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import jax.numpy as jnp

from routetools.cost import cost_function_rise, evaluate_route_energy
from routetools.cost import segment_bearings_deg as _segment_bearings_deg
from routetools.swopp3 import (
    SWOPP3_CASES,
    case_endpoints,
    great_circle_route,
)
from routetools.swopp3_output import (
    file_a_name,
    file_a_row,
    file_b_name,
    sailed_distance_nm,
    waypoint_times,
    write_file_a,
    write_file_b,
)
from routetools.weather import (
    DEFAULT_HS_LIMIT,
    DEFAULT_TWS_LIMIT,
    weather_penalty,
    weather_penalty_smooth,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DepartureResult",
    "segment_bearings_deg",
    "evaluate_energy",
    "run_gc_departure",
    "run_optimised_departure",
    "run_case",
    "log_run_parameters",
]

# Type alias for field closures: (lon, lat, t) -> (comp1, comp2)
FieldClosure = Callable[
    [jnp.ndarray, jnp.ndarray, jnp.ndarray],
    tuple[jnp.ndarray, jnp.ndarray],
]


def _penalized_rise_cost(
    curve: jnp.ndarray,
    *,
    windfield: FieldClosure,
    wavefield: FieldClosure | None,
    travel_time: float,
    wps: bool,
    spherical_correction: bool,
    time_offset: float = 0.0,
    tws_limit: float = DEFAULT_TWS_LIMIT,
    hs_limit: float = DEFAULT_HS_LIMIT,
    weather_penalty_weight: float = 0.0,
    wind_penalty_weight: float = 0.0,
    wave_penalty_weight: float = 0.0,
    distance_penalty_weight: float = 0.0,
    weather_penalty_type: str = "smooth",
    weather_penalty_sharpness: float = 5.0,
    land: Any | None = None,
    land_distance_epsilon: float = 1.0,
) -> jnp.ndarray:
    """Return the SWOPP3 objective optimized by both CMA-ES and FMS."""
    total_cost = cost_function_rise(
        windfield=windfield,
        curve=curve,
        travel_time=travel_time,
        wavefield=wavefield,
        wps=wps,
        time_offset=time_offset,
    )

    combined_weather_penalty = (
        weather_penalty_weight + wind_penalty_weight + wave_penalty_weight
    )

    if combined_weather_penalty > 0:
        penalty_fn = (
            weather_penalty_smooth
            if weather_penalty_type == "smooth"
            else weather_penalty
        )
        penalty_kwargs: dict[str, Any] = {
            "curve": curve,
            "windfield": windfield,
            "wavefield": wavefield,
            "tws_limit": tws_limit,
            "hs_limit": hs_limit,
            "penalty": combined_weather_penalty,
            "travel_time": travel_time,
            "spherical_correction": spherical_correction,
            "time_offset": time_offset,
        }
        if weather_penalty_type == "smooth":
            penalty_kwargs["sharpness"] = weather_penalty_sharpness
        elif weather_penalty_type != "hard":
            raise ValueError("weather_penalty_type must be 'hard' or 'smooth'")
        total_cost = total_cost + penalty_fn(**penalty_kwargs)

    if land is not None and distance_penalty_weight > 0:
        total_cost = total_cost + land.distance_penalty(
            curve,
            weight=distance_penalty_weight,
            epsilon=land_distance_epsilon,
        )

    return total_cost


def _resolve_runner_verbosity(
    verbose: bool | None,
    verbosity: int | None,
    *,
    default: int = 1,
) -> int:
    """Resolve backward-compatible runner verbosity settings."""
    if verbosity is not None:
        if verbosity not in (0, 1, 2):
            raise ValueError("verbosity must be one of 0, 1, or 2")
        return verbosity
    if verbose is None:
        return default
    return 1 if verbose else 0


# ---------------------------------------------------------------------------
# Parameter logging
# ---------------------------------------------------------------------------
def log_run_parameters(
    case_id: str,
    n_departures: int,
    n_points: int,
    **kwargs: object,
) -> None:
    """Log run configuration parameters at the start of a case.

    Parameters
    ----------
    case_id : str
        SWOPP3 case identifier.
    n_departures : int
        Number of departures to run.
    n_points : int
        Number of waypoints.
    **kwargs
        Additional parameters (penalty weights, CMA-ES settings, etc.).
    """
    case = SWOPP3_CASES[case_id]
    lines = [
        "=" * 60,
        f"SWOPP3 Run: {case['name']} ({case_id})",
        f"  strategy:    {case['strategy']}",
        f"  wps:         {case['wps']}",
        f"  passage_h:   {case['passage_hours']}",
        f"  departures:  {n_departures}",
        f"  n_points:    {n_points}",
    ]
    for key, val in sorted(kwargs.items()):
        lines.append(f"  {key}: {val}")
    lines.append("=" * 60)
    msg = "\n".join(lines)
    logger.info(msg)
    print(msg)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class DepartureResult:
    """Result for a single departure evaluation.

    Attributes
    ----------
    departure : datetime
        Departure datetime (UTC).
    curve : jnp.ndarray
        Optimised or GC route, shape ``(L, 2)`` with ``(lon, lat)``.
    energy_mwh : float
        Total energy consumption in MWh.
    max_tws_mps : float
        Maximum true wind speed encountered (m/s).
    max_hs_m : float
        Maximum significant wave height encountered (m).
    distance_nm : float
        Total sailed distance in nautical miles.
    comp_time_s : float
        Computation time in seconds.
    """

    departure: datetime
    curve: jnp.ndarray
    energy_mwh: float
    max_tws_mps: float
    max_hs_m: float
    distance_nm: float
    comp_time_s: float = 0.0


# ---------------------------------------------------------------------------
# Ship bearing computation
# ---------------------------------------------------------------------------
def segment_bearings_deg(curve: jnp.ndarray) -> jnp.ndarray:
    """Compute true-north bearing (degrees) for each route segment.

    This wrapper is kept for backwards compatibility and delegates to
    :func:`routetools.cost.segment_bearings_deg`.
    """
    return _segment_bearings_deg(curve)


# ---------------------------------------------------------------------------
# Energy evaluation
# ---------------------------------------------------------------------------
def evaluate_energy(
    curve: jnp.ndarray,
    departure: datetime,
    passage_hours: float,
    wps: bool,
    windfield: FieldClosure | None = None,
    wavefield: FieldClosure | None = None,
    departure_offset_h: float = 0.0,
) -> tuple[float, float, float]:
    """Evaluate energy consumption along a route.

    This wrapper preserves the SWOPP3 runner API and delegates to
    :func:`routetools.cost.evaluate_route_energy`.
    """
    _ = departure
    return evaluate_route_energy(
        curve,
        passage_hours,
        wps=wps,
        windfield=windfield,
        wavefield=wavefield,
        departure_offset_h=departure_offset_h,
    )


# ---------------------------------------------------------------------------
# GC departure
# ---------------------------------------------------------------------------
def run_gc_departure(
    case_id: str,
    departure: datetime,
    windfield: FieldClosure | None = None,
    wavefield: FieldClosure | None = None,
    departure_offset_h: float = 0.0,
    n_points: int = 100,
) -> DepartureResult:
    """Evaluate a single Great-Circle departure.

    Parameters
    ----------
    case_id : str
        SWOPP3 case identifier (e.g. ``"AGC_WPS"``).
    departure : datetime
        Departure time (UTC).
    windfield, wavefield : FieldClosure, optional
        Pre-loaded closures.
    departure_offset_h : float
        Time offset (hours) from field origin to departure.
    n_points : int
        Number of waypoints on the great-circle route.

    Returns
    -------
    DepartureResult
    """
    case = SWOPP3_CASES[case_id]
    src, dst = case_endpoints(case_id)

    t0 = _time.time()
    curve = great_circle_route(src, dst, n_points=n_points)
    distance_nm = sailed_distance_nm(curve)

    energy_mwh, max_tws, max_hs = evaluate_energy(
        curve,
        departure,
        case["passage_hours"],
        wps=case["wps"],
        windfield=windfield,
        wavefield=wavefield,
        departure_offset_h=departure_offset_h,
    )
    comp_time = _time.time() - t0

    return DepartureResult(
        departure=departure,
        curve=curve,
        energy_mwh=energy_mwh,
        max_tws_mps=max_tws,
        max_hs_m=max_hs,
        distance_nm=distance_nm,
        comp_time_s=comp_time,
    )


# ---------------------------------------------------------------------------
# Optimised departure
# ---------------------------------------------------------------------------
def run_optimised_departure(
    case_id: str,
    departure: datetime,
    vectorfield: FieldClosure | None = None,
    windfield: FieldClosure | None = None,
    wavefield: FieldClosure | None = None,
    land=None,
    departure_offset_h: float = 0.0,
    n_points: int = 100,
    verbosity: int | None = None,
    **cmaes_kwargs,
) -> DepartureResult:
    """Optimise and evaluate a single departure using CMA-ES and FMS.

    Both CMA-ES and FMS optimize the same SWOPP3 objective: RISE energy plus
    the configured weather and land-distance penalties. Energy is then
    evaluated post-hoc with the SWOPP3 performance model.
    Missing optimisation inputs are treated as an error; this function does
    not fall back to a great-circle route.

    Parameters
    ----------
    case_id : str
        SWOPP3 case identifier (e.g. ``"AO_WPS"``).
    departure : datetime
        Departure time (UTC).
    vectorfield : FieldClosure, optional
        Vector field for the CMA-ES cost function.  For SWOPP3 this is
        typically the ERA5 wind field.  If ``None``, this function raises
        ``ValueError`` instead of falling back to a great-circle route.
    windfield, wavefield : FieldClosure, optional
        Pre-loaded closures for energy evaluation.
    land : Land, optional
        Land mask for penalisation.
    departure_offset_h : float
        Time offset (hours) from field origin to departure.
    n_points : int
        Number of waypoints (CMA-ES ``L`` parameter).
    verbosity : int, optional
        Output level. ``0`` silences routine prints, ``1`` prints runner
        progress, and ``2`` also enables CMA-ES and FMS verbose printing.
    **cmaes_kwargs
        Additional keyword arguments passed to :func:`routetools.cmaes.optimize`.

    Returns
    -------
    DepartureResult

    Raises
    ------
    ValueError
        If ``vectorfield`` is ``None``. Optimised departures require an ERA5
        wind-derived vector field so CMA-ES cannot silently degrade to a
        great-circle route.
    """
    case = SWOPP3_CASES[case_id]
    src, dst = case_endpoints(case_id)
    travel_time = float(case["passage_hours"])
    resolved_verbosity = _resolve_runner_verbosity(None, verbosity)

    t0 = _time.time()

    if vectorfield is not None:
        import warnings

        if windfield is None:
            warnings.warn(
                "vectorfield provided without windfield; defaulting "
                "windfield to vectorfield for RISE energy cost.",
                stacklevel=2,
            )
            windfield = vectorfield
        # Lazy import to avoid circular dependency / heavy JAX load
        from routetools.cmaes import optimize as cmaes_optimize
        from routetools.fms import optimize_fms

        # Initialise from the great-circle route so CMA-ES starts near
        # the geodesic.
        gc_init = great_circle_route(src, dst, n_points=n_points)
        # great_circle_route may unwrap longitude through the
        # antimeridian (e.g. -121° becomes 239°).  Use the unwrapped
        # endpoints so the CMA-ES endpoint check passes and the Bézier
        # curve stays in a consistent longitude range.
        src_opt = jnp.array([gc_init[0, 0], gc_init[0, 1]])
        dst_opt = jnp.array([gc_init[-1, 0], gc_init[-1, 1]])

        _wps = case["wps"]

        defaults_cmaes: dict[str, Any] = dict(
            K=10,
            L=n_points,
            curve0=gc_init,
            sigma0=0.1,
            penalty=1000,
            land_margin=2,
            verbose=resolved_verbosity >= 2,
            time_offset=departure_offset_h,
            windfield=windfield,
            wavefield=wavefield,
            travel_time=travel_time,
            spherical_correction=True,
            weather_penalty_weight=10.0,
            wind_penalty_weight=0.0,
            wave_penalty_weight=0.0,
            distance_penalty_weight=0.0,
            tws_limit=DEFAULT_TWS_LIMIT,
            hs_limit=DEFAULT_HS_LIMIT,
        )
        defaults_fms = dict(
            patience=cmaes_kwargs.pop("fms_patience", 200),
            damping=cmaes_kwargs.pop("fms_damping", 0.95),
            maxfevals=cmaes_kwargs.pop("fms_maxfevals", 10000),
        )
        weather_penalty_type = str(cmaes_kwargs.pop("weather_penalty_type", "smooth"))
        weather_penalty_sharpness = float(
            cmaes_kwargs.pop("weather_penalty_sharpness", 5.0)
        )
        land_distance_epsilon = float(cmaes_kwargs.pop("land_distance_epsilon", 1.0))
        if cmaes_kwargs.pop("cmaes_verbose", False):
            cmaes_kwargs["verbose"] = True
        defaults_cmaes.update(cmaes_kwargs)
        defaults_fms["verbose"] = bool(defaults_cmaes["verbose"])

        objective_travel_time = float(defaults_cmaes.get("travel_time", travel_time))
        objective_time_offset = float(
            defaults_cmaes.get("time_offset", departure_offset_h)
        )
        objective_spherical_correction = bool(
            defaults_cmaes.get("spherical_correction", True)
        )
        tws_limit = float(defaults_cmaes.get("tws_limit", DEFAULT_TWS_LIMIT))
        hs_limit = float(defaults_cmaes.get("hs_limit", DEFAULT_HS_LIMIT))
        weather_penalty_weight = float(
            defaults_cmaes.get("weather_penalty_weight", 0.0)
        )
        wind_penalty_weight = float(defaults_cmaes.get("wind_penalty_weight", 0.0))
        wave_penalty_weight = float(defaults_cmaes.get("wave_penalty_weight", 0.0))
        distance_penalty_weight = float(
            defaults_cmaes.get("distance_penalty_weight", 0.0)
        )

        def _shared_cost(
            curve: jnp.ndarray,
            *,
            travel_time: float | None = None,
            time_offset: float | None = None,
            spherical_correction: bool | None = None,
            **_: Any,
        ) -> jnp.ndarray:
            return _penalized_rise_cost(
                curve=curve,
                windfield=windfield,
                wavefield=wavefield,
                travel_time=(
                    objective_travel_time if travel_time is None else travel_time
                ),
                wps=_wps,
                spherical_correction=(
                    objective_spherical_correction
                    if spherical_correction is None
                    else spherical_correction
                ),
                time_offset=(
                    objective_time_offset if time_offset is None else time_offset
                ),
                tws_limit=tws_limit,
                hs_limit=hs_limit,
                weather_penalty_weight=weather_penalty_weight,
                wind_penalty_weight=wind_penalty_weight,
                wave_penalty_weight=wave_penalty_weight,
                distance_penalty_weight=distance_penalty_weight,
                weather_penalty_type=weather_penalty_type,
                weather_penalty_sharpness=weather_penalty_sharpness,
                land=land,
                land_distance_epsilon=land_distance_epsilon,
            )

        defaults_cmaes["cost_fn"] = _shared_cost
        fms_costfun_kwargs: dict[str, Any] = {
            "windfield": windfield,
            "wavefield": wavefield,
            "wps": _wps,
            "spherical_correction": objective_spherical_correction,
            "tws_limit": tws_limit,
            "hs_limit": hs_limit,
            "weather_penalty_weight": weather_penalty_weight,
            "wind_penalty_weight": wind_penalty_weight,
            "wave_penalty_weight": wave_penalty_weight,
            "distance_penalty_weight": distance_penalty_weight,
            "weather_penalty_type": weather_penalty_type,
            "weather_penalty_sharpness": weather_penalty_sharpness,
            "land": land,
            "land_distance_epsilon": land_distance_epsilon,
        }

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Sampling standard deviation i=.*",
                category=UserWarning,
                module=r"cma\.evolution_strategy",
            )
            warnings.filterwarnings(
                "ignore",
                message=(
                    r"\[WARNING\] The optimized curve has a higher cost "
                    r"than the initial curve0 provided .*"
                ),
                category=UserWarning,
            )
            cmaes_t0 = _time.time()
            curve_cmaes, _ = cmaes_optimize(
                vectorfield=vectorfield,
                src=src_opt,
                dst=dst_opt,
                land=land,
                **defaults_cmaes,
            )
        cmaes_distance_nm = sailed_distance_nm(curve_cmaes)
        energy_cmaes, max_tws_cmaes, max_hs_cmaes = evaluate_energy(
            curve_cmaes,
            departure,
            case["passage_hours"],
            wps=case["wps"],
            windfield=windfield,
            wavefield=wavefield,
            departure_offset_h=departure_offset_h,
        )
        cmaes_comp_time_s = _time.time() - cmaes_t0

        fms_t0 = _time.time()
        curve_fms, _ = optimize_fms(
            vectorfield=vectorfield,
            curve=curve_cmaes,
            land=land,
            windfield=windfield,
            wavefield=wavefield,
            travel_time=objective_travel_time,
            spherical_correction=objective_spherical_correction,
            time_offset=objective_time_offset,
            enforce_weather_limits=True,
            tws_limit=tws_limit,
            hs_limit=hs_limit,
            costfun=_penalized_rise_cost,
            costfun_kwargs=fms_costfun_kwargs,
            **defaults_fms,
        )
        curve_fms = curve_fms[0]
        fms_distance_nm = sailed_distance_nm(curve_fms)
        energy_fms, max_tws_fms, max_hs_fms = evaluate_energy(
            curve_fms,
            departure,
            case["passage_hours"],
            wps=case["wps"],
            windfield=windfield,
            wavefield=wavefield,
            departure_offset_h=departure_offset_h,
        )
        fms_comp_time_s = _time.time() - fms_t0

        if resolved_verbosity >= 1:
            print()
            print(
                f"    CMA-ES  E={energy_cmaes:.2f} MWh  "
                f"d={cmaes_distance_nm:.0f} nm  "
                f"t={cmaes_comp_time_s:.1f}s"
            )
            print(
                f"    FMS     E={energy_fms:.2f} MWh  "
                f"d={fms_distance_nm:.0f} nm  "
                f"t={fms_comp_time_s:.1f}s"
            )

        cmaes_is_valid = max_tws_cmaes <= tws_limit and max_hs_cmaes <= hs_limit
        fms_is_valid = max_tws_fms <= tws_limit and max_hs_fms <= hs_limit

        if fms_is_valid and (not cmaes_is_valid or energy_fms < energy_cmaes):
            curve = curve_fms
            energy_mwh = energy_fms
            max_tws = max_tws_fms
            max_hs = max_hs_fms
        else:
            curve = curve_cmaes
            energy_mwh = energy_cmaes
            max_tws = max_tws_cmaes
            max_hs = max_hs_cmaes
    else:
        # No vectorfield → raise error
        raise ValueError(
            "Optimised departure requires a vectorfield for CMA-ES.  "
            "Provide a vectorfield or use run_gc_departure for a great-circle route.",
        )

    distance_nm = sailed_distance_nm(curve)
    comp_time = _time.time() - t0

    return DepartureResult(
        departure=departure,
        curve=curve,
        energy_mwh=energy_mwh,
        max_tws_mps=max_tws,
        max_hs_m=max_hs,
        distance_nm=distance_nm,
        comp_time_s=comp_time,
    )


# ---------------------------------------------------------------------------
# Case runner
# ---------------------------------------------------------------------------
def run_case(
    case_id: str,
    departures: list[datetime],
    vectorfield: FieldClosure | None = None,
    windfield: FieldClosure | None = None,
    wavefield: FieldClosure | None = None,
    land=None,
    output_dir: str | Path | None = None,
    submission: int = 1,
    n_points: int = 100,
    verbose: bool = True,
    dataset_epoch: datetime | None = None,
    **cmaes_kwargs,
) -> list[DepartureResult]:
    """Run all departures for a single SWOPP3 case.

    Dispatches to :func:`run_gc_departure` or :func:`run_optimised_departure`
    depending on the case strategy.  When *output_dir* is provided, writes
    File A and File B CSVs.

    Parameters
    ----------
    case_id : str
        SWOPP3 case identifier (e.g. ``"AGC_WPS"``).
    departures : list[datetime]
        List of departure times.
    vectorfield : FieldClosure, optional
        Vector field for CMA-ES optimisation (optimised cases only).
    windfield, wavefield : FieldClosure, optional
        Pre-loaded closures for energy evaluation.
    land : Land, optional
        Land mask for penalisation.
    output_dir : str or Path, optional
        If provided, writes output CSVs to this directory.
    submission : int
        Submission number for file naming.
    n_points : int
        Number of route waypoints.
    verbose : bool
        Print progress.
    dataset_epoch : datetime, optional
        First timestamp of the loaded ERA5 dataset (UTC).  When provided,
        the departure-to-field time offset is computed automatically for
        each departure.  If ``None``, offset = 0 (suitable only when each
        departure loads its own field with ``departure_time``).
    **cmaes_kwargs
        Extra arguments for CMA-ES (optimised cases only). Optimised cases
        require ``vectorfield`` to be provided.

    Returns
    -------
    list[DepartureResult]
        One result per departure.
    """
    case = SWOPP3_CASES[case_id]
    casename = case["name"]
    is_gc = case["strategy"] == "gc"
    results: list[DepartureResult] = []

    if verbose:
        log_run_parameters(
            case_id,
            n_departures=len(departures),
            n_points=n_points,
            **{k: v for k, v in cmaes_kwargs.items() if v is not None and v != 0},
        )

    for i, dep in enumerate(departures):
        if verbose:
            print(
                f"[{casename}] Departure {i + 1}/{len(departures)}  "
                f"{dep.strftime('%Y-%m-%d')}",
                end="  ",
                flush=True,
            )

        # Compute time offset for this departure relative to field origin.
        # When fields are loaded once for the whole year (without a per-
        # departure reload), dataset_epoch tells us where t=0 lives.
        if dataset_epoch is not None:
            dep_naive = dep.replace(tzinfo=None) if dep.tzinfo else dep
            epoch_naive = (
                dataset_epoch.replace(tzinfo=None)
                if hasattr(dataset_epoch, "tzinfo") and dataset_epoch.tzinfo
                else dataset_epoch
            )
            departure_offset_h = (dep_naive - epoch_naive).total_seconds() / 3600.0
        else:
            departure_offset_h = 0.0

        if is_gc:
            result = run_gc_departure(
                case_id,
                dep,
                windfield=windfield,
                wavefield=wavefield,
                departure_offset_h=departure_offset_h,
                n_points=n_points,
            )
        else:
            result = run_optimised_departure(
                case_id,
                dep,
                vectorfield=vectorfield,
                windfield=windfield,
                wavefield=wavefield,
                land=land,
                departure_offset_h=departure_offset_h,
                n_points=n_points,
                **cmaes_kwargs,
            )

        results.append(result)
        if verbose:
            # Flag constraint violations
            tws_flag = " [TWS!]" if result.max_tws_mps > 20.0 else ""
            hs_flag = " [Hs!]" if result.max_hs_m > 7.0 else ""
            print(
                f"E={result.energy_mwh:.2f} MWh  "
                f"d={result.distance_nm:.0f} nm  "
                f"TWS={result.max_tws_mps:.1f}{tws_flag}  "
                f"Hs={result.max_hs_m:.1f}{hs_flag}  "
                f"t={result.comp_time_s:.1f}s"
            )

    # ---- Write outputs ----
    if output_dir is not None:
        output_dir = Path(output_dir)
        _write_case_outputs(
            case_id,
            results,
            output_dir,
            submission=submission,
        )

    return results


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------
def _write_case_outputs(
    case_id: str,
    results: list[DepartureResult],
    output_dir: Path,
    submission: int = 1,
) -> None:
    """Write File A and File B CSVs for a completed case."""
    case = SWOPP3_CASES[case_id]
    casename = case["name"]
    passage_hours = case["passage_hours"]

    file_b_dir = output_dir / "tracks"
    output_dir.mkdir(parents=True, exist_ok=True)
    file_b_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for res in results:
        # File B
        fb_name = file_b_name(submission, casename, res.departure)
        times = waypoint_times(res.curve, res.departure, passage_hours)
        write_file_b(res.curve, times, file_b_dir / fb_name)

        # File A row
        rows.append(
            file_a_row(
                departure=res.departure,
                passage_hours=passage_hours,
                energy_mwh=res.energy_mwh,
                max_wind_mps=res.max_tws_mps,
                max_hs_m=res.max_hs_m,
                distance_nm=res.distance_nm,
                details_filename=fb_name,
            )
        )

    # File A
    fa_path = output_dir / file_a_name(submission, casename)
    write_file_a(rows, fa_path)
