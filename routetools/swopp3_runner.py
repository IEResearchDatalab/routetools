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

import time as _time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import jax.numpy as jnp
from jax import jit
import numpy as np


@jit
def _zero_vectorfield(x, y, t):
    """Constant-zero vectorfield for CMA-ES distance optimisation."""
    return jnp.zeros_like(x), jnp.zeros_like(y)


_zero_vectorfield.is_time_variant = False

from routetools.performance import predict_power_batch
from routetools.swopp3 import (
    PORTS,
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
def _segment_bearings_deg(curve: jnp.ndarray) -> np.ndarray:
    """Compute true-north bearing (degrees) at each segment midpoint.

    Parameters
    ----------
    curve : jnp.ndarray
        Shape ``(L, 2)`` with ``(lon, lat)`` in degrees.

    Returns
    -------
    np.ndarray
        Shape ``(L-1,)`` bearing in degrees [0, 360).
    """
    lon = np.asarray(curve[:, 0], dtype=np.float64)
    lat = np.asarray(curve[:, 1], dtype=np.float64)

    dlon = np.diff(lon)
    dlat = np.diff(lat)
    lat_mid = np.radians((lat[:-1] + lat[1:]) / 2)

    # Approximate bearing using Mercator projection
    dx = dlon * np.cos(lat_mid)  # eastward component
    dy = dlat  # northward component

    bearing = np.degrees(np.arctan2(dx, dy)) % 360.0
    return bearing


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

    Samples wind and wave conditions at each segment midpoint,
    computes the corresponding power via the SWOPP3 performance model,
    and integrates over time.

    Parameters
    ----------
    curve : jnp.ndarray
        Shape ``(L, 2)`` with ``(lon, lat)`` in degrees.
    departure : datetime
        Departure time (UTC).  Used for time indexing into the fields.
    passage_hours : float
        Total passage time in hours.
    wps : bool
        Whether wingsails are deployed.
    windfield : FieldClosure, optional
        ``(lon, lat, t) -> (u10, v10)`` in m/s.  ``t`` is hours since
        the field's reference time.  If ``None``, assume zero wind.
    wavefield : FieldClosure, optional
        ``(lon, lat, t) -> (hs, mwd)`` where ``hs`` is in metres and
        ``mwd`` is degrees from North.  If ``None``, assume calm seas.
    departure_offset_h : float
        Hours from the field's time origin to the departure.  When the
        field was loaded with ``departure_time``, this is typically 0.

    Returns
    -------
    tuple[float, float, float]
        ``(energy_mwh, max_tws_mps, max_hs_m)``.
    """
    L = curve.shape[0]
    n_seg = L - 1

    # ---- segment midpoints (position) ----
    mid_lon = np.asarray((curve[:-1, 0] + curve[1:, 0]) / 2, dtype=np.float64)
    mid_lat = np.asarray((curve[:-1, 1] + curve[1:, 1]) / 2, dtype=np.float64)

    # ---- segment midpoints (time) — hours since field origin ----
    seg_frac = (np.arange(n_seg) + 0.5) / n_seg  # midpoint fractions
    t_hours = departure_offset_h + seg_frac * passage_hours

    # ---- ship bearing ----
    bearing_deg = _segment_bearings_deg(curve)  # (n_seg,)

    # ---- constant ship speed (m/s) ----
    distance_nm = sailed_distance_nm(curve)
    v_mps = (distance_nm * 1852.0) / (passage_hours * 3600.0)  # m/s

    # ---- wind ----
    if windfield is not None:
        u10, v10 = windfield(
            jnp.array(mid_lon), jnp.array(mid_lat), jnp.array(t_hours)
        )
        u10 = np.asarray(u10, dtype=np.float64)
        v10 = np.asarray(v10, dtype=np.float64)
        tws = np.sqrt(u10**2 + v10**2)
        # Wind FROM direction (meteorological): 180° + atan2(u10, v10)
        wind_from_deg = (180.0 + np.degrees(np.arctan2(u10, v10))) % 360.0
        twa = (wind_from_deg - bearing_deg) % 360.0
    else:
        tws = np.zeros(n_seg)
        twa = np.zeros(n_seg)

    # ---- waves ----
    if wavefield is not None:
        hs, mwd = wavefield(
            jnp.array(mid_lon), jnp.array(mid_lat), jnp.array(t_hours)
        )
        hs = np.asarray(hs, dtype=np.float64)
        mwd = np.asarray(mwd, dtype=np.float64)
        mwa = (mwd - bearing_deg) % 360.0
    else:
        hs = np.zeros(n_seg)
        mwa = np.zeros(n_seg)

    # ---- power (kW) at each segment ----
    v_arr = np.full(n_seg, v_mps)
    power_kw = predict_power_batch(tws, twa, hs, mwa, v_arr, wps=wps)

    # ---- integrate: energy = Σ P_kW · Δt_h (kWh), then → MWh ----
    dt_hours = passage_hours / n_seg
    energy_kwh = float(np.sum(power_kw) * dt_hours)
    energy_mwh = energy_kwh / 1000.0

    max_tws_mps = float(np.max(tws)) if windfield is not None else 0.0
    max_hs_m = float(np.max(hs)) if wavefield is not None else 0.0

    return energy_mwh, max_tws_mps, max_hs_m


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
    **cmaes_kwargs,
) -> DepartureResult:
    """Optimise and evaluate a single departure using CMA-ES.

    The CMA-ES optimizer minimises travel cost through the wind field.
    Energy is then evaluated post-hoc with the SWOPP3 performance model.

    Parameters
    ----------
    case_id : str
        SWOPP3 case identifier (e.g. ``"AO_WPS"``).
    departure : datetime
        Departure time (UTC).
    vectorfield : FieldClosure, optional
        Vector field for the CMA-ES cost function.  For SWOPP3 this is
        typically the ERA5 wind field.  If ``None``, assumes zero-field
        and falls back to a great-circle route.
    windfield, wavefield : FieldClosure, optional
        Pre-loaded closures for energy evaluation.
    land : Land, optional
        Land mask for penalisation.
    departure_offset_h : float
        Time offset (hours) from field origin to departure.
    n_points : int
        Number of waypoints (CMA-ES ``L`` parameter).
    **cmaes_kwargs
        Additional keyword arguments passed to :func:`routetools.cmaes.optimize`.

    Returns
    -------
    DepartureResult
    """
    case = SWOPP3_CASES[case_id]
    src, dst = case_endpoints(case_id)
    travel_time = float(case["passage_hours"])

    t0 = _time.time()

    if vectorfield is not None:
        # Lazy import to avoid circular dependency / heavy JAX load
        from routetools.cmaes import optimize as cmaes_optimize

        # Use a constant-zero vectorfield for CMA-ES optimisation.
        # The wind vectorfield would be treated as an ocean current
        # (‖SOG − wind‖²), but wind does not propel a ship like a
        # current.  With a zero field the optimizer minimises route
        # distance (the best shape proxy for fixed-time energy); the
        # actual energy is evaluated post-hoc with the RISE model.
        defaults = dict(
            L=n_points,
            travel_time=travel_time,
            spherical_correction=True,
            verbose=False,
        )
        defaults.update(cmaes_kwargs)

        curve, info = cmaes_optimize(
            vectorfield=_zero_vectorfield,
            src=src,
            dst=dst,
            land=land,
            **defaults,
        )
    else:
        # No vectorfield → fall back to great circle
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
        Extra arguments for CMA-ES (optimised cases only).

    Returns
    -------
    list[DepartureResult]
        One result per departure.
    """
    case = SWOPP3_CASES[case_id]
    casename = case["name"]
    is_gc = case["strategy"] == "gc"
    results: list[DepartureResult] = []

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
            departure_offset_h = (
                dep_naive - epoch_naive
            ).total_seconds() / 3600.0
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
            print(
                f"E={result.energy_mwh:.2f} MWh  "
                f"d={result.distance_nm:.0f} nm  "
                f"t={result.comp_time_s:.1f}s"
            )

    # ---- Write outputs ----
    if output_dir is not None:
        output_dir = Path(output_dir)
        _write_case_outputs(
            case_id, results, output_dir, submission=submission,
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
