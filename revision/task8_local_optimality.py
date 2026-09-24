"""Task 8 (R1-2) — route-by-route local-optimality audit.

This script post-processes the final BERS (post-FMS) real-ocean tracks.  It
does **not** rerun CMA-ES or FMS.  For each optimized departure it evaluates:

1. first-order stationarity of the fixed-endpoint discrete action through the
   per-waypoint Newton--Jacobi correction used by FMS;
2. positive-definiteness of the complete block-tridiagonal discrete Hessian;
3. the 2x2 RISE raw-power velocity Hessian on every route segment; and
4. margins to the explicitly known non-smooth RISE/weather surfaces and the
   low-speed/high-wind regime identified during the revision audit.

The discrete action exactly mirrors the FMS segment objective: RISE energy,
the configured smooth wind/wave penalties, and the inverse-distance land
penalty, multiplied by the fixed segment duration.  Endpoints are held fixed.

The result is a *numerical certificate to declared tolerances*, not an
analytic proof.  A strict local-minimum theorem requires an exact stationary
point and a C2 objective.  The ERA5 and EDT interpolants are piecewise smooth,
and the RISE surrogate contains explicit kinks.  The output therefore keeps
stationarity, curvature, smoothness, envelope, and geometric feasibility as
separate columns instead of collapsing them into one unsupported claim.

Run on the server after Tasks 4 and 7, forcing CPU execution so monthly ERA5
arrays and autodiff executables use host memory::

    JAX_PLATFORMS=cpu uv run python revision/task8_local_optimality.py \
      --real-ocean-dir output/bers_revision_2024_final_YYYYMMDD/bers \
      --land-verification-csv revision/task7_route_results.csv

Legacy outputs without an embedded manifest can be audited against a separate,
explicit objective-provenance manifest with ``--experiment-manifest``.  The
penalty weights supplied on the command line must match that manifest exactly.

Outputs (under ``--output-dir``)
--------------------------------
- ``task8_local_optimality_routes.csv``: one row per final route.
- ``task8_local_optimality_segments.csv``: one row per segment.
- ``task8_local_optimality_summary.csv``: aggregate by case.
- ``task8_local_optimality_summary.tex``: traceable manuscript table.
- ``task8_local_optimality_failures.csv``: routes failing at least one
  numerical local-minimum condition.
- ``task8_local_optimality_report.md``: methods, tolerances, results, and
  limitations suitable for the revision traceability record.
- ``--checkpoint-dir/*.json``: resumable per-route checkpoints.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import typer
import xarray as xr

from routetools._cost.haversine import haversine_meters_components
from routetools.era5 import (
    load_era5_wavefield,
    load_era5_windfield,
    load_natural_earth_land_mask,
)
from routetools.era5.loader import load_dataset_epoch, loadable_era5_paths
from routetools.performance import (
    predict_power_raw_velocity_jax,
    predict_power_velocity_hessian_jax,
)
from routetools.swopp3 import SWOPP3_CASES
from routetools.swopp3_runner import _penalized_rise_cost
from routetools.violations import (
    departure_offset_hours,
    find_team_prefix,
    read_track_curve,
)
from routetools.weather import DEFAULT_HS_LIMIT, DEFAULT_TWS_LIMIT

jax.config.update("jax_enable_x64", True)

REPO_ROOT = Path(__file__).resolve().parent.parent
_DTFMT = "%Y-%m-%d %H:%M:%S"
OPTIMIZED_CASES = ("AO_WPS", "AO_noWPS", "PO_WPS", "PO_noWPS")


@dataclass(frozen=True)
class AuditSettings:
    """Frozen objective and numerical-classification settings."""

    weather_penalty_weight: float = 0.0
    wind_penalty_weight: float = 50.0
    wave_penalty_weight: float = 50.0
    tws_limit: float = DEFAULT_TWS_LIMIT
    hs_limit: float = DEFAULT_HS_LIMIT
    weather_penalty_sharpness: float = 5.0
    land_distance_weight: float = 100.0
    land_distance_epsilon: float = 1.0
    distance_penalty_weight: float = 0.0
    spherical_correction: bool = True
    stationarity_abs_m: float = 100.0
    stationarity_relative: float = 1.0e-3
    pd_relative_tolerance: float = 1.0e-10
    hessian_symmetry_relative_tolerance: float = 1.0e-8
    raw_power_margin_kw: float = 1.0e-6
    angle_margin_deg: float = 1.0e-3
    crosswind_margin_mps: float = 1.0e-6
    threshold_margin: float = 1.0e-6
    low_speed_mps: float = 2.0
    high_wind_mps: float = 20.0


@dataclass(frozen=True)
class BlockFactorization:
    """Block-LDL factors and numerical definiteness diagnostics."""

    schur_blocks: tuple[np.ndarray, ...]
    lower_factors: tuple[np.ndarray, ...]
    is_positive_definite: bool
    minimum_pivot_eigenvalue: float
    relative_pivot_margin: float
    threshold: float


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _unwrap_route_longitudes(curve: np.ndarray) -> np.ndarray:
    """Return a route with continuous longitudes across the antimeridian.

    The strict Pacific output files wrap stored longitudes into [-180, 180],
    whereas the ERA5 grid and the original FMS optimization use a continuous
    0--360-like sequence.  Differentiating a polyline containing a 358-degree
    coordinate jump does not reproduce the optimized discrete action.
    """
    result = np.asarray(curve, dtype=np.float64).copy()
    if result.ndim != 2 or result.shape[1] != 2:
        raise ValueError("curve must have shape (L, 2)")
    if len(result) > 1:
        result[:, 0] = np.rad2deg(np.unwrap(np.deg2rad(result[:, 0])))
    return result


def _month_start(value: datetime) -> datetime:
    return value.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def _add_months(value: datetime, months: int) -> datetime:
    month_index = value.month - 1 + months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    return value.replace(year=year, month=month, day=1)


def _monthly_batches(
    departures: list[datetime],
) -> list[tuple[datetime, datetime, list[datetime]]]:
    groups: dict[datetime, list[datetime]] = {}
    for departure in sorted(departures):
        groups.setdefault(_month_start(departure), []).append(departure)
    return [
        (month, _add_months(month, 2), values)
        for month, values in sorted(groups.items())
    ]


def assemble_interior_blocks(
    segment_gradients: np.ndarray,
    segment_hessians: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble gradient and block-tridiagonal Hessian for fixed endpoints.

    Parameters
    ----------
    segment_gradients
        Shape ``(n_segments, 4)`` ordered as ``(q0_x, q0_y, q1_x, q1_y)``.
    segment_hessians
        Shape ``(n_segments, 4, 4)`` in the same coordinate order.

    Returns
    -------
    gradient, diagonal, upper
        Interior gradient ``(n_interior, 2)``, diagonal Hessian blocks
        ``(n_interior, 2, 2)``, and upper off-diagonal blocks
        ``(n_interior - 1, 2, 2)``.
    """
    gradients = np.asarray(segment_gradients, dtype=np.float64)
    hessians = np.asarray(segment_hessians, dtype=np.float64)
    if gradients.ndim != 2 or gradients.shape[1] != 4:
        raise ValueError("segment_gradients must have shape (n_segments, 4)")
    if hessians.shape != (gradients.shape[0], 4, 4):
        raise ValueError("segment_hessians must have shape (n_segments, 4, 4)")
    if gradients.shape[0] < 2:
        raise ValueError("at least two segments are required")

    n_interior = gradients.shape[0] - 1
    gradient = gradients[:-1, 2:4] + gradients[1:, 0:2]
    diagonal = hessians[:-1, 2:4, 2:4] + hessians[1:, 0:2, 0:2]
    diagonal = 0.5 * (diagonal + np.swapaxes(diagonal, 1, 2))
    upper_forward = hessians[1:n_interior, 0:2, 2:4]
    upper_reverse = np.swapaxes(
        hessians[1:n_interior, 2:4, 0:2],
        1,
        2,
    )
    upper = 0.5 * (upper_forward + upper_reverse)
    return gradient, diagonal, upper


def scale_blocks_to_meters(
    gradient: np.ndarray,
    diagonal: np.ndarray,
    upper: np.ndarray,
    interior_latitudes_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform degree-coordinate derivatives to local east/north metres.

    Positive-definiteness is invariant under this nonsingular congruence
    transformation.  The transformed Newton correction is physically
    interpretable in metres.
    """
    latitudes = np.asarray(interior_latitudes_deg, dtype=np.float64)
    if latitudes.shape != (gradient.shape[0],):
        raise ValueError("interior_latitudes_deg has the wrong shape")

    earth_radius_m = 6_371_008.8
    radians_per_degree = math.pi / 180.0
    metres_per_degree_lat = earth_radius_m * radians_per_degree
    metres_per_degree_lon = metres_per_degree_lat * np.cos(np.radians(latitudes))
    if np.any(np.abs(metres_per_degree_lon) < 1.0):
        raise ValueError("longitude scaling is singular too close to a pole")

    transforms = np.zeros((len(latitudes), 2, 2), dtype=np.float64)
    transforms[:, 0, 0] = 1.0 / metres_per_degree_lon
    transforms[:, 1, 1] = 1.0 / metres_per_degree_lat

    transforms_t = np.swapaxes(transforms, 1, 2)
    gradient_m = np.einsum("nij,nj->ni", transforms_t, gradient)
    diagonal_m = np.einsum(
        "nij,njk,nkl->nil", transforms_t, diagonal, transforms
    )
    if len(upper):
        upper_m = np.einsum(
            "nij,njk,nkl->nil",
            transforms_t[:-1],
            upper,
            transforms[1:],
        )
    else:
        upper_m = np.empty((0, 2, 2), dtype=np.float64)
    return gradient_m, diagonal_m, upper_m


def factor_spd_block_tridiagonal(
    diagonal: np.ndarray,
    upper: np.ndarray,
    *,
    relative_tolerance: float,
) -> BlockFactorization:
    """Block-LDL factorization and a tolerance-aware SPD test."""
    diagonal = np.asarray(diagonal, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if diagonal.ndim != 3 or diagonal.shape[1:] != (2, 2):
        raise ValueError("diagonal must have shape (n, 2, 2)")
    if upper.shape != (max(len(diagonal) - 1, 0), 2, 2):
        raise ValueError("upper must have shape (n-1, 2, 2)")
    if relative_tolerance < 0:
        raise ValueError("relative_tolerance must be non-negative")

    magnitude_arrays = [np.abs(diagonal).reshape(-1)]
    if len(upper):
        magnitude_arrays.append(np.abs(upper).reshape(-1))
    magnitudes = np.concatenate(magnitude_arrays)
    finite_magnitudes = magnitudes[np.isfinite(magnitudes)]
    scale = max(
        float(np.max(finite_magnitudes)) if len(finite_magnitudes) else 0.0,
        np.finfo(np.float64).tiny,
    )
    threshold = relative_tolerance * scale
    schur_blocks: list[np.ndarray] = []
    lower_factors: list[np.ndarray] = []
    minimum = math.inf
    is_pd = True

    for index, block in enumerate(diagonal):
        schur = np.array(block, dtype=np.float64, copy=True)
        if index:
            previous = schur_blocks[-1]
            coupling = upper[index - 1]
            try:
                lower = np.linalg.solve(previous.T, coupling).T
            except np.linalg.LinAlgError:
                is_pd = False
                lower = np.zeros((2, 2), dtype=np.float64)
            lower_factors.append(lower)
            schur = schur - lower @ coupling
        schur = 0.5 * (schur + schur.T)
        if np.all(np.isfinite(schur)):
            try:
                eigenvalues = np.linalg.eigvalsh(schur)
                local_minimum = float(eigenvalues[0])
            except np.linalg.LinAlgError:
                local_minimum = -math.inf
        else:
            local_minimum = -math.inf
        minimum = min(minimum, local_minimum)
        if not np.isfinite(local_minimum) or local_minimum <= threshold:
            is_pd = False
        schur_blocks.append(schur)

    relative_margin = minimum / scale
    return BlockFactorization(
        schur_blocks=tuple(schur_blocks),
        lower_factors=tuple(lower_factors),
        is_positive_definite=is_pd,
        minimum_pivot_eigenvalue=minimum,
        relative_pivot_margin=relative_margin,
        threshold=threshold,
    )


def solve_block_ldlt(
    factorization: BlockFactorization,
    right_hand_side: np.ndarray,
) -> np.ndarray:
    """Solve a block-tridiagonal system from ``factor_spd_block_tridiagonal``."""
    if not factorization.is_positive_definite:
        raise np.linalg.LinAlgError("block Hessian is not positive-definite")
    rhs = np.asarray(right_hand_side, dtype=np.float64)
    if rhs.shape != (len(factorization.schur_blocks), 2):
        raise ValueError("right_hand_side must have shape (n, 2)")

    forward = np.empty_like(rhs)
    forward[0] = rhs[0]
    for index in range(1, len(rhs)):
        forward[index] = (
            rhs[index] - factorization.lower_factors[index - 1] @ forward[index - 1]
        )

    diagonal_solution = np.empty_like(rhs)
    for index, (block, value) in enumerate(
        zip(factorization.schur_blocks, forward, strict=True)
    ):
        diagonal_solution[index] = np.linalg.solve(block, value)

    solution = np.empty_like(rhs)
    solution[-1] = diagonal_solution[-1]
    for index in range(len(rhs) - 2, -1, -1):
        solution[index] = diagonal_solution[index] - (
            factorization.lower_factors[index].T @ solution[index + 1]
        )
    return solution


def solve_fms_local_corrections(
    diagonal: np.ndarray,
    gradient: np.ndarray,
) -> np.ndarray:
    """Return the simultaneous per-waypoint Newton corrections used by FMS.

    FMS solves each discrete Euler--Lagrange equation with its local 2x2
    diagonal Hessian block and applies the resulting corrections as a Jacobi
    sweep.  This fixed-point residual is defined independently of whether the
    *complete* route Hessian is positive definite, which keeps the first- and
    second-order tests logically separate.
    """
    diagonal = np.asarray(diagonal, dtype=np.float64)
    gradient = np.asarray(gradient, dtype=np.float64)
    if diagonal.ndim != 3 or diagonal.shape[1:] != (2, 2):
        raise ValueError("diagonal must have shape (n, 2, 2)")
    if gradient.shape != (len(diagonal), 2):
        raise ValueError("gradient must have shape (n, 2)")

    corrections = np.full_like(gradient, np.nan)
    for index, (block, value) in enumerate(
        zip(diagonal, gradient, strict=True)
    ):
        try:
            corrections[index] = np.linalg.solve(block, -value)
        except np.linalg.LinAlgError:
            continue
    return corrections


def _build_segment_derivative_function(
    *,
    windfield: Callable[..., Any],
    wavefield: Callable[..., Any],
    land: Any,
    segment_hours: float,
    wps: bool,
    settings: AuditSettings,
) -> Callable[[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, ...]]:
    """Build a vectorized exact segment value/gradient/Hessian evaluator."""

    def segment_action(
        flat_segment: jnp.ndarray,
        time_offset: jnp.ndarray,
    ) -> jnp.ndarray:
        curve = flat_segment.reshape((1, 2, 2))
        value = _penalized_rise_cost(
            curve=curve,
            windfield=windfield,
            wavefield=wavefield,
            travel_time=segment_hours,
            wps=wps,
            spherical_correction=settings.spherical_correction,
            time_offset=time_offset,
            tws_limit=settings.tws_limit,
            hs_limit=settings.hs_limit,
            weather_penalty_weight=settings.weather_penalty_weight,
            weather_penalty_type="smooth",
            weather_penalty_sharpness=settings.weather_penalty_sharpness,
            wind_penalty_weight=settings.wind_penalty_weight,
            wave_penalty_weight=settings.wave_penalty_weight,
            land=land,
            land_distance_weight=settings.land_distance_weight,
            land_distance_epsilon=settings.land_distance_epsilon,
            distance_penalty_weight=settings.distance_penalty_weight,
        )[0]
        return segment_hours * value

    value_and_gradient = jax.value_and_grad(segment_action, argnums=0)
    hessian_function = jax.hessian(segment_action, argnums=0)

    def derivatives(
        flat_segment: jnp.ndarray, time_offset: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        value, gradient = value_and_gradient(flat_segment, time_offset)
        hessian = hessian_function(flat_segment, time_offset)
        return value, gradient, hessian

    return jax.jit(jax.vmap(derivatives, in_axes=(0, 0)))


@lru_cache(maxsize=2)
def _build_velocity_diagnostics_function(
    wps: bool,
) -> Callable[..., tuple[jnp.ndarray, jnp.ndarray]]:
    """Build the reusable segment-level raw-power/Hessian evaluator."""

    def raw_and_hessian(
        u_value: jnp.ndarray,
        v_value: jnp.ndarray,
        hs_value: jnp.ndarray,
        mwd_value: jnp.ndarray,
        ve_value: jnp.ndarray,
        vn_value: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        raw = predict_power_raw_velocity_jax(
            u_value,
            v_value,
            hs_value,
            mwd_value,
            ve_value,
            vn_value,
            wps=wps,
        )
        hessian = predict_power_velocity_hessian_jax(
            u_value,
            v_value,
            hs_value,
            mwd_value,
            ve_value,
            vn_value,
            wps=wps,
        )
        return raw, hessian

    return jax.jit(jax.vmap(raw_and_hessian, in_axes=(0, 0, 0, 0, 0, 0)))


def _segment_environment_diagnostics(
    curve: jnp.ndarray,
    *,
    windfield: Callable[..., Any],
    wavefield: Callable[..., Any],
    travel_time_h: float,
    departure_offset_h: float,
    wps: bool,
    settings: AuditSettings,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Evaluate per-segment operating envelope and continuous Hessians."""
    n_segments = int(curve.shape[0] - 1)
    dt_h = travel_time_h / n_segments
    dt_s = dt_h * 3600.0
    mid_lon = (curve[:-1, 0] + curve[1:, 0]) / 2.0
    mid_lat = (curve[:-1, 1] + curve[1:, 1]) / 2.0
    times = departure_offset_h + (jnp.arange(n_segments) + 0.5) * dt_h

    u10, v10 = windfield(mid_lon, mid_lat, times)
    hs, mwd = wavefield(mid_lon, mid_lat, times)
    east_m, north_m = haversine_meters_components(
        curve[:-1, 1],
        curve[:-1, 0],
        curve[1:, 1],
        curve[1:, 0],
    )
    ve = east_m / dt_s
    vn = north_m / dt_s

    raw_power, velocity_hessians = _build_velocity_diagnostics_function(wps)(
        u10,
        v10,
        hs,
        mwd,
        ve,
        vn,
    )

    u10_np = np.asarray(u10, dtype=np.float64)
    v10_np = np.asarray(v10, dtype=np.float64)
    hs_np = np.asarray(hs, dtype=np.float64)
    mwd_np = np.asarray(mwd, dtype=np.float64)
    ve_np = np.asarray(ve, dtype=np.float64)
    vn_np = np.asarray(vn, dtype=np.float64)
    raw_np = np.asarray(raw_power, dtype=np.float64)
    hessian_np = np.asarray(velocity_hessians, dtype=np.float64)

    speed = np.hypot(ve_np, vn_np)
    tws = np.hypot(u10_np, v10_np)
    bearing_deg = np.mod(np.degrees(np.arctan2(ve_np, vn_np)), 360.0)
    wind_from_deg = np.mod(
        180.0 + np.degrees(np.arctan2(u10_np, v10_np)), 360.0
    )
    twa_deg = np.mod(wind_from_deg - bearing_deg, 360.0)
    wave_relative_deg = np.mod(mwd_np - bearing_deg + 180.0, 360.0) - 180.0
    twa_rad = np.radians(twa_deg)
    apparent_x = tws * np.cos(twa_rad) + speed
    apparent_y = tws * np.sin(twa_rad)
    awa_deg = np.degrees(np.arctan2(np.abs(apparent_y), apparent_x))
    velocity_eigenvalues = np.linalg.eigvalsh(
        0.5 * (hessian_np + np.swapaxes(hessian_np, 1, 2))
    )
    velocity_minimum = velocity_eigenvalues[:, 0]

    raw_clip_active = raw_np <= 0.0
    raw_clip_near = np.abs(raw_np) <= settings.raw_power_margin_kw
    dead_zone_near = (
        np.abs(awa_deg - 10.0) <= settings.angle_margin_deg
        if wps
        else np.zeros(n_segments, dtype=bool)
    )
    fore_aft_near = (
        np.abs(apparent_y) <= settings.crosswind_margin_mps
        if wps
        else np.zeros(n_segments, dtype=bool)
    )
    wave_wrap_near = (
        np.abs(np.abs(wave_relative_deg) - 180.0) <= settings.angle_margin_deg
    )
    wind_threshold_near = (
        np.abs(tws - settings.tws_limit) <= settings.threshold_margin
    )
    wave_threshold_near = (
        np.abs(hs_np - settings.hs_limit) <= settings.threshold_margin
    )
    extreme_regime = (speed < settings.low_speed_mps) & (
        tws >= settings.high_wind_mps
    )
    velocity_nonpositive = (~np.isfinite(velocity_minimum)) | (
        velocity_minimum <= 0.0
    )

    segment_rows: list[dict[str, object]] = []
    for index in range(n_segments):
        segment_rows.append(
            {
                "segment_index": index,
                "speed_mps": float(speed[index]),
                "tws_mps": float(tws[index]),
                "hs_m": float(hs_np[index]),
                "mwd_deg": float(mwd_np[index]),
                "raw_power_kw": float(raw_np[index]),
                "velocity_hessian_min_eigenvalue": float(velocity_minimum[index]),
                "awa_deg": float(awa_deg[index]),
                "apparent_crosswind_mps": float(apparent_y[index]),
                "raw_clip_active": bool(raw_clip_active[index]),
                "raw_clip_near_kink": bool(raw_clip_near[index]),
                "wps_dead_zone_near_kink": bool(dead_zone_near[index]),
                "fore_aft_near_kink": bool(fore_aft_near[index]),
                "wave_direction_wrap_near_kink": bool(wave_wrap_near[index]),
                "wind_threshold_near_kink": bool(wind_threshold_near[index]),
                "wave_threshold_near_kink": bool(wave_threshold_near[index]),
                "low_speed_high_wind": bool(extreme_regime[index]),
                "velocity_hessian_nonpositive": bool(velocity_nonpositive[index]),
            }
        )

    smooth_pass = not bool(
        np.any(
            raw_clip_near
            | dead_zone_near
            | fore_aft_near
            | wave_wrap_near
            | wind_threshold_near
            | wave_threshold_near
        )
    )
    summary: dict[str, object] = {
        "n_segments": n_segments,
        "minimum_speed_mps": float(np.min(speed)),
        "maximum_speed_mps": float(np.max(speed)),
        "maximum_tws_mps": float(np.max(tws)),
        "maximum_hs_m": float(np.max(hs_np)),
        "minimum_raw_power_kw": float(np.min(raw_np)),
        "minimum_velocity_hessian_eigenvalue": float(np.min(velocity_minimum)),
        "raw_clip_active_count": int(np.sum(raw_clip_active)),
        "raw_clip_near_kink_count": int(np.sum(raw_clip_near)),
        "wps_dead_zone_near_kink_count": int(np.sum(dead_zone_near)),
        "fore_aft_near_kink_count": int(np.sum(fore_aft_near)),
        "wave_direction_wrap_near_kink_count": int(np.sum(wave_wrap_near)),
        "wind_threshold_near_kink_count": int(np.sum(wind_threshold_near)),
        "wave_threshold_near_kink_count": int(np.sum(wave_threshold_near)),
        "low_speed_high_wind_count": int(np.sum(extreme_regime)),
        "velocity_hessian_nonpositive_count": int(np.sum(velocity_nonpositive)),
        "rise_weather_smoothness_pass": smooth_pass,
        "raw_power_unclipped_region": not bool(np.any(raw_clip_active)),
        "velocity_hessian_pass": not bool(np.any(velocity_nonpositive)),
        "extreme_regime_absent": not bool(np.any(extreme_regime)),
    }
    return segment_rows, summary


def _route_distance_statistics(curve: np.ndarray) -> tuple[float, float]:
    east_m, north_m = haversine_meters_components(
        jnp.asarray(curve[:-1, 1]),
        jnp.asarray(curve[:-1, 0]),
        jnp.asarray(curve[1:, 1]),
        jnp.asarray(curve[1:, 0]),
    )
    distances = np.hypot(np.asarray(east_m), np.asarray(north_m))
    return float(np.median(distances)), float(np.min(distances))


def _audit_route(
    *,
    case_id: str,
    route_id: str,
    departure: datetime,
    curve: jnp.ndarray,
    dataset_epoch: datetime,
    windfield: Callable[..., Any],
    wavefield: Callable[..., Any],
    land: Any,
    derivative_function: Callable[..., tuple[jnp.ndarray, ...]],
    settings: AuditSettings,
    geometric_land_crossing: bool | None,
) -> dict[str, object]:
    case = SWOPP3_CASES[case_id]
    travel_time_h = float(case["passage_hours"])
    wps = bool(case["wps"])
    n_segments = int(curve.shape[0] - 1)
    segment_h = travel_time_h / n_segments
    departure_offset_h = departure_offset_hours(departure, dataset_epoch)
    segment_offsets = departure_offset_h + np.arange(n_segments) * segment_h

    flat_segments = jnp.concatenate([curve[:-1], curve[1:]], axis=1)
    segment_values, segment_gradients, segment_hessians = derivative_function(
        flat_segments,
        jnp.asarray(segment_offsets, dtype=curve.dtype),
    )
    jax.block_until_ready(segment_hessians)

    segment_hessians_np = np.asarray(segment_hessians)
    hessian_scale = max(
        float(np.max(np.abs(segment_hessians_np))),
        np.finfo(np.float64).tiny,
    )
    hessian_symmetry_error = float(
        np.max(
            np.abs(
                segment_hessians_np
                - np.swapaxes(segment_hessians_np, 1, 2)
            )
        )
    )
    hessian_symmetry_relative_error = hessian_symmetry_error / hessian_scale
    hessian_symmetry_pass = bool(
        np.isfinite(hessian_symmetry_relative_error)
        and hessian_symmetry_relative_error
        <= settings.hessian_symmetry_relative_tolerance
    )
    gradient, diagonal, upper = assemble_interior_blocks(
        np.asarray(segment_gradients),
        segment_hessians_np,
    )
    gradient_m, diagonal_m, upper_m = scale_blocks_to_meters(
        gradient,
        diagonal,
        upper,
        np.asarray(curve[1:-1, 1]),
    )
    factorization = factor_spd_block_tridiagonal(
        diagonal_m,
        upper_m,
        relative_tolerance=settings.pd_relative_tolerance,
    )
    fms_correction = solve_fms_local_corrections(diagonal_m, gradient_m)
    fms_correction_norms = np.linalg.norm(fms_correction, axis=1)
    if np.all(np.isfinite(fms_correction_norms)):
        maximum_fms_correction_m = float(np.max(fms_correction_norms))
        rms_fms_correction_m = float(
            np.sqrt(np.mean(fms_correction_norms**2))
        )
    else:
        maximum_fms_correction_m = math.nan
        rms_fms_correction_m = math.nan

    if factorization.is_positive_definite:
        full_newton_correction = solve_block_ldlt(factorization, -gradient_m)
        full_correction_norms = np.linalg.norm(full_newton_correction, axis=1)
        maximum_full_newton_correction_m = float(
            np.max(full_correction_norms)
        )
        rms_full_newton_correction_m = float(
            np.sqrt(np.mean(full_correction_norms**2))
        )
    else:
        maximum_full_newton_correction_m = math.nan
        rms_full_newton_correction_m = math.nan

    curve_np = np.asarray(curve, dtype=np.float64)
    median_segment_m, minimum_segment_m = _route_distance_statistics(curve_np)
    stationarity_limit_m = min(
        settings.stationarity_abs_m,
        settings.stationarity_relative * median_segment_m,
    )
    stationarity_pass = bool(
        np.isfinite(maximum_fms_correction_m)
        and maximum_fms_correction_m <= stationarity_limit_m
    )
    full_newton_stationarity_pass = bool(
        factorization.is_positive_definite
        and np.isfinite(maximum_full_newton_correction_m)
        and maximum_full_newton_correction_m <= stationarity_limit_m
    )

    segment_rows, environment = _segment_environment_diagnostics(
        curve,
        windfield=windfield,
        wavefield=wavefield,
        travel_time_h=travel_time_h,
        departure_offset_h=departure_offset_h,
        wps=wps,
        settings=settings,
    )
    sampled_land_violation_count = int(np.sum(np.asarray(land(curve[None, ...]))))

    numerical_local_minimum = bool(
        stationarity_pass
        and full_newton_stationarity_pass
        and factorization.is_positive_definite
        and hessian_symmetry_pass
        and environment["rise_weather_smoothness_pass"]
    )
    envelope_pass = bool(
        environment["raw_power_unclipped_region"]
        and environment["velocity_hessian_pass"]
        and environment["extreme_regime_absent"]
    )
    sampled_feasible = sampled_land_violation_count == 0
    geometric_feasible = geometric_land_crossing is False
    complete_route_certificate = bool(
        numerical_local_minimum
        and envelope_pass
        and sampled_feasible
        and geometric_feasible
    )

    route_row: dict[str, object] = {
        "case_id": case_id,
        "corridor": str(case["route"]),
        "wps": wps,
        "route_id": route_id,
        "departure_time_utc": departure.strftime(_DTFMT),
        "n_waypoints": int(curve.shape[0]),
        "discrete_action": float(np.sum(np.asarray(segment_values))),
        "gradient_l2_per_m": float(np.linalg.norm(gradient_m.reshape(-1))),
        "gradient_max_per_m": float(np.max(np.abs(gradient_m))),
        "discrete_hessian_pd": factorization.is_positive_definite,
        "hessian_symmetry_max_abs_error": hessian_symmetry_error,
        "hessian_symmetry_relative_error": hessian_symmetry_relative_error,
        "hessian_symmetry_pass": hessian_symmetry_pass,
        "minimum_ldlt_pivot_eigenvalue": (
            factorization.minimum_pivot_eigenvalue
        ),
        "relative_ldlt_pivot_margin": factorization.relative_pivot_margin,
        "pd_numerical_threshold": factorization.threshold,
        "maximum_fms_correction_m": maximum_fms_correction_m,
        "rms_fms_correction_m": rms_fms_correction_m,
        "maximum_full_newton_correction_m": (
            maximum_full_newton_correction_m
        ),
        "rms_full_newton_correction_m": rms_full_newton_correction_m,
        "stationarity_limit_m": stationarity_limit_m,
        "stationarity_pass": stationarity_pass,
        "full_newton_stationarity_pass": full_newton_stationarity_pass,
        "median_segment_length_m": median_segment_m,
        "minimum_segment_length_m": minimum_segment_m,
        "sampled_land_violation_count": sampled_land_violation_count,
        "geometric_land_crossing": geometric_land_crossing,
        "numerical_local_minimum_to_tolerance": numerical_local_minimum,
        "operating_envelope_pass": envelope_pass,
        "complete_route_certificate": complete_route_certificate,
        **environment,
    }
    for segment_row in segment_rows:
        segment_row.update(
            {
                "case_id": case_id,
                "route_id": route_id,
                "departure_time_utc": departure.strftime(_DTFMT),
            }
        )
    return {"route": route_row, "segments": segment_rows}


def _read_case_rows(input_dir: Path, case_id: str) -> list[dict[str, object]]:
    team_prefix = find_team_prefix(input_dir)
    summary_path = input_dir / f"{team_prefix}-{case_id}.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing final case summary: {summary_path}")
    rows: list[dict[str, object]] = []
    with summary_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    **row,
                    "departure": datetime.strptime(row["departure_time_utc"], _DTFMT),
                }
            )
    return rows


def _load_land_crossings(path: Path | None) -> dict[str, bool]:
    if path is None or not path.exists():
        return {}
    results: dict[str, bool] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("source", "real_ocean") != "real_ocean":
                continue
            value = str(row["crosses_land"]).strip().lower()
            results[row["route_id"]] = value in {"1", "true", "yes"}
    return results


def _load_weather_batch(
    wind_path: Path,
    wave_path: Path,
    start: datetime,
    end: datetime,
) -> tuple[datetime, Any, Any]:
    wind_paths = loadable_era5_paths(wind_path)
    wave_paths = loadable_era5_paths(wave_path)
    wind_target: Any = wind_paths if len(wind_paths) > 1 else wind_paths[0]
    wave_target: Any = wave_paths if len(wave_paths) > 1 else wave_paths[0]
    epoch = load_dataset_epoch(wind_target, time_start=start, time_end=end)
    windfield = load_era5_windfield(
        wind_target,
        time_start=start,
        time_end=end,
    )
    wavefield = load_era5_wavefield(
        wave_target,
        time_start=start,
        time_end=end,
    )
    return epoch, windfield, wavefield


def _build_land_mask(weather_path: Path) -> Any:
    base_path = loadable_era5_paths(weather_path)[0]
    with xr.open_dataset(base_path) as dataset:
        lon_name = "longitude" if "longitude" in dataset.coords else "lon"
        lat_name = "latitude" if "latitude" in dataset.coords else "lat"
        lon_values = np.asarray(dataset[lon_name].values)
        lat_values = np.asarray(dataset[lat_name].values)
    return load_natural_earth_land_mask(
        (float(np.min(lon_values)), float(np.max(lon_values))),
        (float(np.min(lat_values)), float(np.max(lat_values))),
    )


def _settings_signature(
    settings: AuditSettings,
    input_dir: Path,
    weather_paths: dict[str, tuple[Path, Path]],
    manifest_sha256: str,
    land_verification_sha256: str,
) -> str:
    payload = {
        "settings": asdict(settings),
        "input_dir": str(input_dir.resolve()),
        "weather_paths": {
            key: [str(value[0].resolve()), str(value[1].resolve())]
            for key, value in weather_paths.items()
        },
        "manifest_sha256": manifest_sha256,
        "land_verification_sha256": land_verification_sha256,
        "schema": 8,
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_experiment_manifest(
    input_dir: Path,
    settings: AuditSettings,
    manifest_path: Path | None = None,
) -> tuple[Path, str]:
    """Verify that stored routes used the objective audited by this script."""
    manifest_path = (
        input_dir / "experiment_manifest.json"
        if manifest_path is None
        else manifest_path
    )
    if not manifest_path.exists():
        raise FileNotFoundError(
            "The final experiment manifest is required for objective provenance: "
            f"{manifest_path}"
        )
    raw_bytes = manifest_path.read_bytes()
    manifest = json.loads(raw_bytes)
    runs = manifest.get("runs")
    if not isinstance(runs, list):
        raise ValueError(f"Malformed experiment manifest: {manifest_path}")

    expected = {
        "weather_penalty_weight": settings.weather_penalty_weight,
        "wind_penalty_weight": settings.wind_penalty_weight,
        "wave_penalty_weight": settings.wave_penalty_weight,
        "tws_limit": settings.tws_limit,
        "hs_limit": settings.hs_limit,
        "land_distance_weight": settings.land_distance_weight,
        "land_distance_epsilon": settings.land_distance_epsilon,
        "distance_penalty_weight": settings.distance_penalty_weight,
        "spherical_correction": settings.spherical_correction,
    }
    cases_seen: set[str] = set()
    problems: list[str] = []
    for run in runs:
        if not isinstance(run, dict):
            problems.append("non-object run entry")
            continue
        cases = {str(case_id) for case_id in run.get("cases", [])}
        relevant = cases.intersection(OPTIMIZED_CASES)
        if not relevant:
            continue
        cases_seen.update(relevant)
        for key, expected_value in expected.items():
            if key not in run:
                problems.append(f"run {run.get('name', '?')}: missing {key}")
                continue
            actual_value = float(run[key])
            if not math.isclose(
                actual_value,
                expected_value,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                problems.append(
                    f"run {run.get('name', '?')}: {key}={actual_value:g}, "
                    f"expected {expected_value:g}"
                )

    missing_cases = set(OPTIMIZED_CASES).difference(cases_seen)
    if missing_cases:
        problems.append(f"optimized cases absent: {sorted(missing_cases)}")
    if problems:
        details = "\n  - ".join(problems)
        raise ValueError(
            "The experiment manifest does not match the audited objective:\n"
            f"  - {details}"
        )
    return manifest_path, hashlib.sha256(raw_bytes).hexdigest()


def _write_checkpoint(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=True))
    temporary.replace(path)


def _refresh_route_classification(route: dict[str, object]) -> bool:
    """Reclassify cached derivatives using the conservative route-wide test."""
    before = (
        route.get("full_newton_stationarity_pass"),
        route.get("numerical_local_minimum_to_tolerance"),
        route.get("complete_route_certificate"),
    )
    maximum_full_correction = float(
        route.get("maximum_full_newton_correction_m", math.nan)
    )
    full_newton_stationarity_pass = bool(
        route["discrete_hessian_pd"]
        and np.isfinite(maximum_full_correction)
        and maximum_full_correction <= float(route["stationarity_limit_m"])
    )
    numerical_local_minimum = bool(
        route["stationarity_pass"]
        and full_newton_stationarity_pass
        and route["discrete_hessian_pd"]
        and route["hessian_symmetry_pass"]
        and route["rise_weather_smoothness_pass"]
    )
    complete_route_certificate = bool(
        numerical_local_minimum
        and route["operating_envelope_pass"]
        and int(route["sampled_land_violation_count"]) == 0
        and route["geometric_land_crossing"] is False
    )
    route["full_newton_stationarity_pass"] = full_newton_stationarity_pass
    route["numerical_local_minimum_to_tolerance"] = numerical_local_minimum
    route["complete_route_certificate"] = complete_route_certificate
    after = (
        full_newton_stationarity_pass,
        numerical_local_minimum,
        complete_route_certificate,
    )
    return before != after


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows available for {path.name}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _summarize_routes(route_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for case_id in OPTIMIZED_CASES:
        rows = [row for row in route_rows if row["case_id"] == case_id]
        if not rows:
            continue
        n_routes = len(rows)

        def count(key: str, case_rows: list[dict[str, object]] = rows) -> int:
            return sum(bool(row[key]) for row in case_rows)

        def finite_min(
            key: str,
            case_rows: list[dict[str, object]] = rows,
        ) -> float:
            values = [float(row[key]) for row in case_rows]
            finite = [value for value in values if np.isfinite(value)]
            return min(finite, default=math.nan)

        def finite_max(
            key: str,
            case_rows: list[dict[str, object]] = rows,
        ) -> float:
            values = [float(row[key]) for row in case_rows]
            finite = [value for value in values if np.isfinite(value)]
            return max(finite, default=math.nan)

        summary.append(
            {
                "case_id": case_id,
                "n_routes": n_routes,
                "stationarity_pass_n": count("stationarity_pass"),
                "full_newton_stationarity_pass_n": count(
                    "full_newton_stationarity_pass"
                ),
                "discrete_hessian_pd_n": count("discrete_hessian_pd"),
                "hessian_symmetry_pass_n": count("hessian_symmetry_pass"),
                "smoothness_pass_n": count("rise_weather_smoothness_pass"),
                "velocity_hessian_pass_n": count("velocity_hessian_pass"),
                "raw_power_unclipped_n": count("raw_power_unclipped_region"),
                "extreme_regime_absent_n": count("extreme_regime_absent"),
                "numerical_local_minimum_n": count(
                    "numerical_local_minimum_to_tolerance"
                ),
                "complete_route_certificate_n": count("complete_route_certificate"),
                "numerical_local_minimum_pct": round(
                    100.0
                    * count("numerical_local_minimum_to_tolerance")
                    / n_routes,
                    3,
                ),
                "complete_route_certificate_pct": round(
                    100.0 * count("complete_route_certificate") / n_routes,
                    3,
                ),
                "minimum_speed_mps": finite_min("minimum_speed_mps"),
                "minimum_velocity_hessian_eigenvalue": finite_min(
                    "minimum_velocity_hessian_eigenvalue"
                ),
                "maximum_fms_correction_m": finite_max(
                    "maximum_fms_correction_m"
                ),
                "minimum_full_newton_correction_m": finite_min(
                    "maximum_full_newton_correction_m"
                ),
            }
        )
    return summary


def _write_latex_summary(
    path: Path,
    summary_rows: list[dict[str, object]],
) -> None:
    """Write the route-level certificate counts as a manuscript-ready table."""
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        "Case & $N$ & FMS fixed point & Hessian PD & Full Newton & "
        r"Velocity Hessian & Certified \\",
        r"\midrule",
    ]
    for row in summary_rows:
        lines.append(
            f"{row['case_id'].replace('_', r'\_')} & {row['n_routes']} & "
            f"{row['stationarity_pass_n']} & {row['discrete_hessian_pd_n']} & "
            f"{row['full_newton_stationarity_pass_n']} & "
            f"{row['velocity_hessian_pass_n']} & "
            f"{row['numerical_local_minimum_n']} " + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report(
    path: Path,
    *,
    input_dir: Path,
    manifest_path: Path,
    manifest_sha256: str,
    land_verification_path: Path,
    land_verification_sha256: str,
    settings: AuditSettings,
    route_rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
    geometry_available: bool,
) -> None:
    lines = [
        "# Task 8 — route-by-route local-optimality audit",
        "",
        f"Input route directory: `{input_dir}`",
        f"Experiment manifest: `{manifest_path}`",
        f"Manifest SHA-256: `{manifest_sha256}`",
        f"Land-verification CSV: `{land_verification_path}`",
        f"Land-verification SHA-256: `{land_verification_sha256}`",
        f"Routes audited: {len(route_rows)}",
        f"Exact geometric land results joined: {geometry_available}",
        "",
        "## Classification tolerances",
        "",
        "- Absolute per-waypoint FMS correction limit: "
        f"{settings.stationarity_abs_m:g} m.",
        "- Relative Newton-correction limit: "
        f"{settings.stationarity_relative:g} times the median segment length.",
        "- The effective stationarity limit is the smaller of those two values.",
        "- Relative block-LDL pivot tolerance: "
        f"{settings.pd_relative_tolerance:g}.",
        "- Relative automatic-Hessian symmetry tolerance: "
        f"{settings.hessian_symmetry_relative_tolerance:g}.",
        f"- Low-speed threshold: {settings.low_speed_mps:g} m/s.",
        f"- High-wind threshold: {settings.high_wind_mps:g} m/s.",
        "",
        "## Method",
        "",
        "- The audited action is the sum of the same fixed-duration segment "
        "objectives differentiated by FMS, with both endpoints held fixed.",
        "- Exact automatic derivatives assemble the complete reduced "
        "block-tridiagonal Hessian, including adjacent-segment mixed blocks.",
        "- Positive definiteness is tested by a symmetric block-LDL "
        "factorization after a local east/north metre-coordinate congruence.",
        "- The simultaneous per-waypoint Newton--Jacobi correction used by "
        "FMS is reported as a local fixed-point diagnostic; it is not treated "
        "as complete-route stationarity.",
        "- When the complete block Hessian is positive definite, its coupled "
        "Newton correction provides the route-wide first-order test. Both "
        "the FMS-local and full-route corrections must meet the stated limit "
        "for a numerical local-minimum classification.",
        "- The continuous 2x2 RISE velocity Hessian is reported separately as "
        "an operating-envelope diagnostic; it is not substituted for the "
        "complete discrete Hessian.",
        "",
        "## Results",
        "",
        "| Case | Routes | FMS fixed point | Discrete Hessian PD | "
        "Full-route stationary | Velocity Hessian PD | Certified local minima |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['case_id']} | {row['n_routes']} | "
            f"{row['stationarity_pass_n']} | {row['discrete_hessian_pd_n']} | "
            f"{row['full_newton_stationarity_pass_n']} | "
            f"{row['velocity_hessian_pass_n']} | "
            f"{row['numerical_local_minimum_n']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- `numerical_local_minimum_to_tolerance` requires both a small "
            "FMS-local correction and a small coupled full-route Newton "
            "correction, a positive-definite complete discrete Hessian, and no "
            "detected contact with the explicit RISE/weather kink surfaces.",
            "- `complete_route_certificate` additionally requires the sampled "
            "land check, exact geometric land verification, and the segment-level "
            "operating-envelope checks to pass.",
            "- The certificate is for the implemented fixed-endpoint discrete "
            "penalized action. It is not a claim of global optimality.",
            "- ERA5 and EDT interpolation are piecewise smooth; this script does "
            "not provide an interval proof between grid cells.",
            "- Near-zero eigenvalues and routes close to a non-smooth surface are "
            "reported as failures rather than rounded into passing cases.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main(
    real_ocean_dir: str = "output/bers_revision_2024_final/bers",
    output_dir: str = "revision",
    checkpoint_dir: str = "revision/task8_checkpoints",
    experiment_manifest: str = "",
    wind_path_atlantic: str = "data/era5/era5_wind_atlantic_2024.nc",
    wave_path_atlantic: str = "data/era5/era5_waves_atlantic_2024.nc",
    wind_path_pacific: str = "data/era5/era5_wind_pacific_2024.nc",
    wave_path_pacific: str = "data/era5/era5_waves_pacific_2024.nc",
    land_verification_csv: str = "revision/task7_route_results.csv",
    weather_penalty_weight: float = 0.0,
    wind_penalty_weight: float = 50.0,
    wave_penalty_weight: float = 50.0,
    tws_limit: float = DEFAULT_TWS_LIMIT,
    hs_limit: float = DEFAULT_HS_LIMIT,
    weather_penalty_sharpness: float = 5.0,
    land_distance_weight: float = 100.0,
    land_distance_epsilon: float = 1.0,
    distance_penalty_weight: float = 0.0,
    spherical_correction: bool = True,
    stationarity_abs_m: float = 100.0,
    stationarity_relative: float = 1.0e-3,
    pd_relative_tolerance: float = 1.0e-10,
    max_routes: int = 0,
    allow_non_cpu: bool = False,
) -> None:
    """Audit all stored final BERS routes without rerunning optimization."""
    platforms = {device.platform for device in jax.devices()}
    if platforms != {"cpu"} and not allow_non_cpu:
        raise RuntimeError(
            "Task 8 is intentionally CPU-only to avoid capturing monthly ERA5 "
            "arrays in GPU executables. Rerun with JAX_PLATFORMS=cpu."
        )

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
        spherical_correction=spherical_correction,
        stationarity_abs_m=stationarity_abs_m,
        stationarity_relative=stationarity_relative,
        pd_relative_tolerance=pd_relative_tolerance,
    )
    weather_paths = {
        "atlantic": (
            _resolve_path(wind_path_atlantic),
            _resolve_path(wave_path_atlantic),
        ),
        "pacific": (
            _resolve_path(wind_path_pacific),
            _resolve_path(wave_path_pacific),
        ),
    }
    manifest_override = (
        _resolve_path(experiment_manifest) if experiment_manifest else None
    )
    manifest_path, manifest_sha256 = _validate_experiment_manifest(
        input_dir,
        settings,
        manifest_override,
    )
    land_csv_path = _resolve_path(land_verification_csv)
    land_verification_sha256 = hashlib.sha256(land_csv_path.read_bytes()).hexdigest()
    land_crossings = _load_land_crossings(land_csv_path)
    signature = _settings_signature(
        settings,
        input_dir,
        weather_paths,
        manifest_sha256,
        land_verification_sha256,
    )

    case_rows = {
        case_id: _read_case_rows(input_dir, case_id)
        for case_id in OPTIMIZED_CASES
    }
    total_requested = sum(len(rows) for rows in case_rows.values())
    print(f"Auditing {total_requested} final optimized routes on {jax.devices()}")

    completed = 0
    for corridor in ("atlantic", "pacific"):
        corridor_cases = [
            case_id
            for case_id in OPTIMIZED_CASES
            if SWOPP3_CASES[case_id]["route"] == corridor
        ]
        departures = sorted(
            {
                row["departure"]
                for case_id in corridor_cases
                for row in case_rows[case_id]
            }
        )
        wind_path, wave_path = weather_paths[corridor]
        print(f"Building Natural Earth mask for {corridor} ...")
        land = _build_land_mask(wave_path)

        for batch_index, (start, end, batch_departures) in enumerate(
            _monthly_batches(departures), start=1
        ):
            print(
                f"[{corridor}] batch {batch_index}: {start.date()} -> {end.date()} "
                f"({len(batch_departures)} departures)"
            )
            epoch, windfield, wavefield = _load_weather_batch(
                wind_path, wave_path, start, end
            )
            batch_set = set(batch_departures)
            derivative_functions: dict[
                tuple[bool, float],
                Callable[..., tuple[jnp.ndarray, ...]],
            ] = {}

            for case_id in corridor_cases:
                for row in case_rows[case_id]:
                    departure = row["departure"]
                    if departure not in batch_set:
                        continue
                    route_id = str(row["details_filename"])
                    checkpoint_path = checkpoints / f"{Path(route_id).stem}.json"
                    if checkpoint_path.exists():
                        payload = json.loads(checkpoint_path.read_text())
                        if payload.get("settings_signature") == signature:
                            completed += 1
                            continue

                    track_path = input_dir / "tracks" / route_id
                    curve = jnp.asarray(
                        _unwrap_route_longitudes(read_track_curve(track_path)),
                        dtype=jnp.float64,
                    )
                    case = SWOPP3_CASES[case_id]
                    segment_h = float(case["passage_hours"]) / (curve.shape[0] - 1)
                    derivative_key = (bool(case["wps"]), segment_h)
                    if derivative_key not in derivative_functions:
                        derivative_functions[derivative_key] = (
                            _build_segment_derivative_function(
                                windfield=windfield,
                                wavefield=wavefield,
                                land=land,
                                segment_hours=segment_h,
                                wps=bool(case["wps"]),
                                settings=settings,
                            )
                        )
                    print(f"  [{completed + 1}/{total_requested}] {route_id}")
                    result = _audit_route(
                        case_id=case_id,
                        route_id=route_id,
                        departure=departure,
                        curve=curve,
                        dataset_epoch=epoch,
                        windfield=windfield,
                        wavefield=wavefield,
                        land=land,
                        derivative_function=derivative_functions[derivative_key],
                        settings=settings,
                        geometric_land_crossing=land_crossings.get(route_id),
                    )
                    result["route"]["objective_manifest_sha256"] = manifest_sha256
                    result["settings_signature"] = signature
                    _write_checkpoint(checkpoint_path, result)
                    completed += 1
                    if max_routes > 0 and completed >= max_routes:
                        break
                if max_routes > 0 and completed >= max_routes:
                    break
            del derivative_functions, windfield, wavefield
            gc.collect()
            jax.clear_caches()
            if max_routes > 0 and completed >= max_routes:
                break
        if max_routes > 0 and completed >= max_routes:
            break

    checkpoint_payloads = []
    for path in sorted(checkpoints.glob("*.json")):
        payload = json.loads(path.read_text())
        if payload.get("settings_signature") == signature:
            if _refresh_route_classification(payload["route"]):
                _write_checkpoint(path, payload)
            checkpoint_payloads.append(payload)
    route_rows = [payload["route"] for payload in checkpoint_payloads]
    segment_rows = [
        segment
        for payload in checkpoint_payloads
        for segment in payload["segments"]
    ]
    route_rows.sort(key=lambda row: (str(row["case_id"]), str(row["route_id"])))
    segment_rows.sort(
        key=lambda row: (
            str(row["case_id"]),
            str(row["route_id"]),
            int(row["segment_index"]),
        )
    )

    _write_csv(out_dir / "task8_local_optimality_routes.csv", route_rows)
    _write_csv(out_dir / "task8_local_optimality_segments.csv", segment_rows)
    summary_rows = _summarize_routes(route_rows)
    _write_csv(out_dir / "task8_local_optimality_summary.csv", summary_rows)
    _write_latex_summary(
        out_dir / "task8_local_optimality_summary.tex",
        summary_rows,
    )
    failure_rows = [
        row
        for row in route_rows
        if not bool(row["numerical_local_minimum_to_tolerance"])
    ]
    if failure_rows:
        _write_csv(out_dir / "task8_local_optimality_failures.csv", failure_rows)
    else:
        (out_dir / "task8_local_optimality_failures.csv").write_text(
            ",".join(route_rows[0].keys()) + "\n"
        )
    _write_report(
        out_dir / "task8_local_optimality_report.md",
        input_dir=input_dir,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        land_verification_path=land_csv_path,
        land_verification_sha256=land_verification_sha256,
        settings=settings,
        route_rows=route_rows,
        summary_rows=summary_rows,
        geometry_available=bool(land_crossings),
    )
    print(f"Wrote Task 8 outputs to {out_dir}")


if __name__ == "__main__":
    typer.run(main)
