"""SWOPP3 runner — execute all 8 cases × 366 departures.

Orchestrates the end-to-end pipeline:

1. **Great-Circle (GC) cases** — fixed geodesic route, constant speed,
   energy evaluated via the RISE performance model.
2. **Optimised (O) cases** — CMA-ES route optimisation, FMS refinement,
    and energy evaluation.

Both modes support WPS (wingsails on) and noWPS (engine only).

Main entry points:

- :func:`evaluate_energy` — evaluate energy along a route.
- :func:`run_gc_departure` — single GC departure.
- :func:`run_optimised_departure` — single optimised departure.
- :func:`run_case` — all departures for one case.
"""

from __future__ import annotations

import csv
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
    weather_penalty_smooth,
)

__all__ = [
    "DepartureResult",
    "segment_bearings_deg",
    "evaluate_energy",
    "run_gc_departure",
    "run_optimised_departure",
    "run_case",
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
) -> jnp.ndarray:
    """Return the SWOPP3 optimisation objective used by CMA-ES and FMS."""
    return cost_function_rise(
        windfield=windfield,
        curve=curve,
        travel_time=travel_time,
        wavefield=wavefield,
        wps=wps,
        time_offset=time_offset,
    ) + weather_penalty_smooth(
        curve,
        windfield=windfield,
        wavefield=wavefield,
        travel_time=travel_time,
        spherical_correction=spherical_correction,
        time_offset=time_offset,
    )


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

    The CMA-ES optimizer minimises SWOPP3 energy through the RISE cost.
    The resulting route is then refined with FMS before post-hoc energy
    evaluation with the SWOPP3 performance model.
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
        Output level used to choose the default CMA-ES verbosity when
        ``verbose`` is not provided in ``cmaes_kwargs``. ``0`` disables
        routine output, ``1`` shows runner prints only, and ``2`` enables
        CMA-ES verbose printing.
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

        # Build a RISE-based cost closure for CMA-ES.
        # This directly minimises SWOPP3 energy (MWh) instead of the
        # ocean-current proxy ‖SOG − wind‖².
        _wps = case["wps"]

        def _rise_cost(curve_batch: jnp.ndarray) -> jnp.ndarray:
            return _penalized_rise_cost(
                curve=curve_batch,
                windfield=windfield,
                wavefield=wavefield,
                travel_time=travel_time,
                wps=_wps,
                spherical_correction=True,
                time_offset=departure_offset_h,
            )

        defaults_cmaes: dict[str, Any] = dict(
            K=10,
            L=n_points,
            curve0=gc_init,
            sigma0=0.1,
            cost_fn=_rise_cost,
            penalty=1e6,
            land_margin=2,
            verbose=resolved_verbosity >= 2,
            windfield=windfield,
            wavefield=wavefield,
            spherical_correction=True,
            travel_time=travel_time,
            time_offset=departure_offset_h,
            # Smooth distance-to-land repulsion via EDT
            land_distance_weight=50.0,
        )
        defaults_fms = dict(
            patience=cmaes_kwargs.pop("fms_patience", 200),
            damping=cmaes_kwargs.pop("fms_damping", 0.95),
            maxfevals=cmaes_kwargs.pop("fms_maxfevals", 10000),
        )

        defaults_cmaes.update(cmaes_kwargs)
        defaults_fms["verbose"] = bool(defaults_cmaes["verbose"])
        tws_limit = float(defaults_cmaes.get("tws_limit", DEFAULT_TWS_LIMIT))
        hs_limit = float(defaults_cmaes.get("hs_limit", DEFAULT_HS_LIMIT))

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
            travel_time=travel_time,
            spherical_correction=True,
            time_offset=departure_offset_h,
            enforce_weather_limits=True,  # revert steps that newly violate limits
            tws_limit=tws_limit,
            hs_limit=hs_limit,
            # FMS uses pure RISE energy (no weather penalty) so its gradient
            # aligns with the evaluate_energy metric used for comparison.
            costfun=cost_function_rise,
            costfun_kwargs={
                "windfield": windfield,
                "wavefield": wavefield,
                "wps": _wps,
            },
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
            if resolved_verbosity >= 1 and not cmaes_is_valid:
                print(
                    "Selected FMS refinement because the CMA-ES route "
                    "exceeded weather limits."
                )
            curve = curve_fms
            energy_mwh = energy_fms
            max_tws = max_tws_fms
            max_hs = max_hs_fms
        else:
            # Warn if FMS refinement fails to improve energy
            # or violates weather constraints
            if (
                resolved_verbosity >= 1
                and cmaes_is_valid
                and energy_fms >= energy_cmaes
            ):
                print(
                    f"FMS refinement did not reduce energy: "
                    f"{energy_fms:.2f} MWh (FMS) vs {energy_cmaes:.2f} MWh (CMA-ES)."
                )
            if resolved_verbosity >= 1 and max_tws_fms > tws_limit:
                print(
                    f"FMS refinement exceeded TWS limit: "
                    f"{max_tws_fms:.1f} m/s > {tws_limit:.1f} m/s."
                )
            if resolved_verbosity >= 1 and max_hs_fms > hs_limit:
                print(
                    f"FMS refinement exceeded Hs limit: "
                    f"{max_hs_fms:.1f} m > {hs_limit:.1f} m."
                )
            if resolved_verbosity >= 1 and max_tws_cmaes > tws_limit:
                print(
                    f"CMA-ES route exceeded TWS limit: "
                    f"{max_tws_cmaes:.1f} m/s > {tws_limit:.1f} m/s."
                )
            if resolved_verbosity >= 1 and max_hs_cmaes > hs_limit:
                print(
                    f"CMA-ES route exceeded Hs limit: "
                    f"{max_hs_cmaes:.1f} m > {hs_limit:.1f} m."
                )
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
    verbose: bool | None = True,
    dataset_epoch: datetime | None = None,
    verbosity: int | None = None,
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
    verbose : bool | None
        Backward-compatible runner progress flag. ``True`` maps to
        ``verbosity=1`` and ``False`` maps to ``verbosity=0``. Ignored when
        ``verbosity`` is provided.
    dataset_epoch : datetime, optional
        First timestamp of the loaded ERA5 dataset (UTC).  When provided,
        the departure-to-field time offset is computed automatically for
        each departure.  If ``None``, offset = 0 (suitable only when each
        departure loads its own field with ``departure_time``).
    verbosity : int, optional
        Output level. ``0`` silences routine prints, ``1`` prints runner
        progress, and ``2`` also enables CMA-ES verbose printing.
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
    resolved_verbosity = _resolve_runner_verbosity(verbose, verbosity)

    output_path: Path | None = None
    if output_dir is not None:
        output_path = Path(output_dir)
        _prepare_case_output(case_id, output_path, submission=submission)

    for i, dep in enumerate(departures):
        if resolved_verbosity >= 1:
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
                verbosity=resolved_verbosity,
                **cmaes_kwargs,
            )

        results.append(result)
        if output_path is not None:
            _append_case_output(
                case_id,
                result,
                output_path,
                submission=submission,
            )
        if resolved_verbosity >= 1:
            print(
                f"E={result.energy_mwh:.2f} MWh  "
                f"d={result.distance_nm:.0f} nm  "
                f"t={result.comp_time_s:.1f}s"
            )

    return results


# ---------------------------------------------------------------------------
# Incremental output writing
# ---------------------------------------------------------------------------
def _prepare_case_output(
    case_id: str,
    output_dir: Path,
    submission: int = 1,
) -> None:
    """Prepare case output files for incremental writes."""
    case = SWOPP3_CASES[case_id]
    fa_path = output_dir / file_a_name(submission, case["name"])
    if fa_path.exists():
        fa_path.unlink()


def _append_case_output(
    case_id: str,
    result: DepartureResult,
    output_dir: Path,
    submission: int = 1,
) -> None:
    """Persist a single departure result to File A and File B."""
    case = SWOPP3_CASES[case_id]
    casename = case["name"]
    passage_hours = case["passage_hours"]

    file_b_dir = output_dir / "tracks"
    output_dir.mkdir(parents=True, exist_ok=True)
    file_b_dir.mkdir(parents=True, exist_ok=True)

    fb_name = file_b_name(submission, casename, result.departure)
    times = waypoint_times(result.curve, result.departure, passage_hours)
    write_file_b(result.curve, times, file_b_dir / fb_name)

    row = file_a_row(
        departure=result.departure,
        passage_hours=passage_hours,
        energy_mwh=result.energy_mwh,
        max_wind_mps=result.max_tws_mps,
        max_hs_m=result.max_hs_m,
        distance_nm=result.distance_nm,
        details_filename=fb_name,
    )
    fa_path = output_dir / file_a_name(submission, casename)
    write_header = not fa_path.exists() or fa_path.stat().st_size == 0
    with fa_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


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
