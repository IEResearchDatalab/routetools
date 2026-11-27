"""Temporal interpolation helpers for ECMWF IFS ensemble forecasts."""

from __future__ import annotations

from typing import Literal

import jax.numpy as jnp
import numpy as np
from jax.scipy.ndimage import map_coordinates

from routetools.ifs.loader import (
    lead_time_hours,
    load_ensemble_waves,
    load_ensemble_wind,
)
from routetools.interpolate import map_coordinates_cubic, prefilter_cubic

__all__ = [
    "NATIVE_FORECAST_RESOLUTION_H",
    "SUPPORTED_INTERPOLATION_RESOLUTIONS_H",
    "interpolate_field_cubic",
    "interpolate_field_linear",
    "lead_time_hours",
    "load_ensemble_waves",
    "load_ensemble_wind",
    "relative_rmse_percent",
    "resample_field_at_resolutions",
    "resample_wind_at_resolutions",
    "temporal_stride_for_resolution",
]

NATIVE_FORECAST_RESOLUTION_H = 3.0
SUPPORTED_INTERPOLATION_RESOLUTIONS_H = (24.0, 12.0, 6.0, 3.0)


def temporal_stride_for_resolution(
    source_resolution_h: float,
    target_resolution_h: float,
) -> int:
    """Convert forecast lead-time resolution into a timestep stride.

    Parameters
    ----------
    source_resolution_h : float
        Native interval between forecast lead times, in hours.
    target_resolution_h : float
        Coarser interval to retain, in hours.
    """
    if source_resolution_h <= 0 or target_resolution_h <= 0:
        raise ValueError("Forecast resolutions must be positive")

    ratio = target_resolution_h / source_resolution_h
    stride = round(ratio)
    if not np.isclose(ratio, stride) or stride < 1:
        raise ValueError(
            "Target resolution must be an exact multiple of the source resolution"
        )
    return int(stride)


def interpolate_field_cubic(
    values: np.ndarray,
    source_hours: np.ndarray,
    target_hours: np.ndarray,
) -> np.ndarray:
    """Interpolate fields with the routing's cubic B-spline implementation."""
    return _interpolate_field(values, source_hours, target_hours, method="cubic")


def interpolate_field_linear(
    values: np.ndarray,
    source_hours: np.ndarray,
    target_hours: np.ndarray,
) -> np.ndarray:
    """Interpolate fields with the routing's default linear map_coordinates."""
    return _interpolate_field(values, source_hours, target_hours, method="linear")


def _interpolate_field(
    values: np.ndarray,
    source_hours: np.ndarray,
    target_hours: np.ndarray,
    *,
    method: Literal["linear", "cubic"],
) -> np.ndarray:
    if values.shape[0] != len(source_hours):
        raise ValueError("First values axis must match the source lead times")
    if len(source_hours) < 2:
        raise ValueError("At least two forecast lead times are required")
    source_intervals = np.diff(source_hours)
    if np.any(source_intervals <= 0) or not np.allclose(
        source_intervals,
        source_intervals[0],
    ):
        raise ValueError("Source lead times must be strictly increasing and uniform")
    if np.any(target_hours < source_hours[0]) or np.any(
        target_hours > source_hours[-1]
    ):
        raise ValueError("Target lead times must lie within the source lead-time range")

    values = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0)

    target_indices = ((target_hours - source_hours[0]) / source_intervals[0]).astype(
        np.float32
    )
    if values.ndim == 1:
        coordinates = jnp.asarray(target_indices[None, :])
    else:
        spatial_indices = np.indices(values.shape[1:], dtype=np.float32).reshape(
            values.ndim - 1,
            -1,
        )
        spatial_size = spatial_indices.shape[1]
        coordinates_np = np.empty(
            (values.ndim, len(target_hours) * spatial_size),
            dtype=np.float32,
        )
        coordinates_np[0] = np.repeat(target_indices, spatial_size)
        coordinates_np[1:] = np.tile(spatial_indices, (1, len(target_hours)))
        coordinates = jnp.asarray(coordinates_np)

    if method == "cubic":
        coefficients, npad = prefilter_cubic(values)
        interpolated = map_coordinates_cubic(
            coefficients,
            coordinates,
            npad=npad,
        )
    else:
        interpolated = map_coordinates(
            jnp.asarray(values),
            tuple(coordinates[axis] for axis in range(values.ndim)),
            order=1,
            mode="nearest",
        )

    return np.asarray(interpolated).reshape((len(target_hours), *values.shape[1:]))


def relative_rmse_percent(
    rmse: np.ndarray,
    reference_mean: np.ndarray,
) -> np.ndarray:
    """Normalize spatial RMSE by the reference's spatial mean, in percent.

    Parameters
    ----------
    rmse : np.ndarray
        Spatial RMSE at each forecast lead time.
    reference_mean : np.ndarray
        Spatial mean of the reference field at the same lead times.
    """
    rmse = np.asarray(rmse, dtype=np.float64)
    reference_mean = np.asarray(reference_mean, dtype=np.float64)
    if rmse.shape != reference_mean.shape:
        raise ValueError("RMSE and reference mean must have matching shapes")
    if np.any(~np.isfinite(reference_mean)) or np.any(reference_mean <= 0):
        raise ValueError("Reference mean must be finite and greater than zero")
    return np.asarray(rmse / reference_mean * 100.0, dtype=np.float64)


def resample_field_at_resolutions(
    hours: np.ndarray,
    values: np.ndarray,
    resolutions_h: tuple[float, ...] = SUPPORTED_INTERPOLATION_RESOLUTIONS_H,
    *,
    method: Literal["linear", "cubic"] = "cubic",
) -> dict[float, np.ndarray]:
    """Resample a scalar forecast field with routing interpolation."""
    interpolation = {
        "linear": interpolate_field_linear,
        "cubic": interpolate_field_cubic,
    }[method]
    fields: dict[float, np.ndarray] = {}
    for resolution_h in resolutions_h:
        stride = temporal_stride_for_resolution(
            NATIVE_FORECAST_RESOLUTION_H,
            resolution_h,
        )
        if resolution_h == NATIVE_FORECAST_RESOLUTION_H:
            fields[resolution_h] = values.copy()
            continue

        source_hours = hours[::stride]
        fields[resolution_h] = interpolation(
            values[::stride],
            source_hours,
            hours,
        )
    return fields


def resample_wind_at_resolutions(
    hours: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    resolutions_h: tuple[float, ...] = SUPPORTED_INTERPOLATION_RESOLUTIONS_H,
    *,
    method: Literal["linear", "cubic"] = "cubic",
) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """Build U/V fields with routing linear or cubic interpolation."""
    u_fields = resample_field_at_resolutions(
        hours,
        u,
        resolutions_h,
        method=method,
    )
    v_fields = resample_field_at_resolutions(
        hours,
        v,
        resolutions_h,
        method=method,
    )
    return {
        resolution_h: (u_fields[resolution_h], v_fields[resolution_h])
        for resolution_h in resolutions_h
    }
