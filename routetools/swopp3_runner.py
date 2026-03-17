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

import time as _time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import jax.numpy as jnp

from routetools.cost import evaluate_route_energy
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
        from routetools.cost import cost_function_rise
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

        # Compute geographic bounds for CMA-ES control points from the
        # GC route envelope.  Control points are flattened as
        # [lon1, lat1, lon2, lat2, …], so bounds alternate lon/lat.
        gc_lons = gc_init[:, 0]
        gc_lats = gc_init[:, 1]
        lon_margin = 15.0  # degrees margin around GC corridor
        lat_margin = 10.0
        lon_lo = float(jnp.min(gc_lons)) - lon_margin
        lon_hi = float(jnp.max(gc_lons)) + lon_margin
        lat_lo = float(jnp.max(jnp.array([jnp.min(gc_lats) - lat_margin, -85.0])))
        lat_hi = float(jnp.min(jnp.array([jnp.max(gc_lats) + lat_margin, 85.0])))
        n_free = (10 - 2) * 2  # K=10 → 8 free points × 2 coords
        ctrl_lower = [lon_lo if i % 2 == 0 else lat_lo for i in range(n_free)]
        ctrl_upper = [lon_hi if i % 2 == 0 else lat_hi for i in range(n_free)]
        ctrl_bounds = [ctrl_lower, ctrl_upper]

        # Build a RISE-based cost closure for CMA-ES.
        # This directly minimises SWOPP3 energy (MWh) instead of the
        # ocean-current proxy ‖SOG − wind‖².
        _wps = case["wps"]
        fms_patience = cmaes_kwargs.pop("fms_patience", 50)
        fms_damping = cmaes_kwargs.pop("fms_damping", 0.9)
        fms_maxfevals = cmaes_kwargs.pop("fms_maxfevals", 5000)

        def _rise_cost(curve_batch: jnp.ndarray) -> jnp.ndarray:
            return cost_function_rise(
                windfield=windfield,
                curve=curve_batch,
                travel_time=travel_time,
                wavefield=wavefield,
                wps=_wps,
                time_offset=departure_offset_h,
            )

        def _rise_fms_cost(*, curve: jnp.ndarray, travel_time: float, **kwargs):
            return cost_function_rise(
                windfield=windfield,
                curve=curve,
                travel_time=travel_time,
                wavefield=kwargs.get("wavefield"),
                wps=_wps,
                time_offset=departure_offset_h,
            )

        defaults = dict(
            K=10,
            L=n_points,
            curve0=gc_init,
            cost_fn=_rise_cost,
            maxfevals=50000,
            tolfun=1.0,
            penalty=1e6,
            land_margin=2,
            verbose=resolved_verbosity >= 2,
            # Operational weather constraints (SWOPP3: TWS < 20 m/s, Hs < 7 m)
            windfield=windfield,
            wavefield=wavefield,
            spherical_correction=True,
            weather_penalty_weight=100.0,
            travel_time=travel_time,
            time_offset=departure_offset_h,
            # Smooth distance-to-land repulsion via EDT
            land_distance_weight=50.0,
            # Geographic bounds on control points (prevents wild turns)
            bounds=ctrl_bounds,
        )
        defaults.update(cmaes_kwargs)
        cmaes_verbose = bool(defaults["verbose"])

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
            curve, info = cmaes_optimize(
                vectorfield=vectorfield,
                src=src_opt,
                dst=dst_opt,
                land=land,
                **defaults,
            )
        curve_fms, _ = optimize_fms(
            vectorfield=vectorfield,
            curve=curve,
            land=land,
            wavefield=wavefield,
            travel_time=travel_time,
            patience=fms_patience,
            damping=fms_damping,
            maxfevals=fms_maxfevals,
            costfun=_rise_fms_cost,
            verbose=cmaes_verbose,
        )
        curve = curve_fms[0]
    else:
        # No vectorfield → raise error
        raise ValueError(
            "Optimised departure requires a vectorfield for CMA-ES.  "
            "Provide a vectorfield or use run_gc_departure for a great-circle route.",
        )

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
        if resolved_verbosity >= 1:
            print(
                f"E={result.energy_mwh:.2f} MWh  "
                f"d={result.distance_nm:.0f} nm  "
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
