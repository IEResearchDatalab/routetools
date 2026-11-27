"""Temporal interpolation helpers for ECMWF IFS ensemble forecasts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
from scipy.interpolate import CubicSpline

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


def lead_time_hours(ds: xr.Dataset) -> np.ndarray:
    """Return forecast lead times in hours from an IFS NetCDF dataset."""
    if "step" not in ds.coords:
        raise KeyError("IFS forecast dataset has no 'step' coordinate")
    values = ds["step"].values / np.timedelta64(1, "h")
    return np.asarray(values, dtype=np.float64)


def load_ensemble_wind(
    path: str | Path,
    *,
    ensemble_member: int | None = None,
    lon_bounds: tuple[float, float] | None = None,
    lat_bounds: tuple[float, float] | None = None,
    max_hours: float = 144.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load U/V wind components and metadata from an IFS ensemble NetCDF.

    By default, ensemble members are averaged. A single member can instead be
    selected with ``ensemble_member``.
    """
    (
        retained_hours,
        lats,
        lons,
        u,
        v,
    ) = _load_ensemble_variables(
        path,
        ("10m_u_component_of_wind", "10m_v_component_of_wind"),
        ensemble_member=ensemble_member,
        lon_bounds=lon_bounds,
        lat_bounds=lat_bounds,
        max_hours=max_hours,
    )
    return retained_hours, lats, lons, u, v


def load_ensemble_waves(
    path: str | Path,
    *,
    ensemble_member: int | None = None,
    lon_bounds: tuple[float, float] | None = None,
    lat_bounds: tuple[float, float] | None = None,
    max_hours: float = 144.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load ensemble significant wave height and grid metadata from IFS data.

    The ensemble members are averaged by default. A single member can instead
    be selected with ``ensemble_member``.
    """
    (
        retained_hours,
        lats,
        lons,
        significant_wave_height,
    ) = _load_ensemble_variables(
        path,
        ("significant_height_of_combined_wind_waves_and_swell",),
        ensemble_member=ensemble_member,
        lon_bounds=lon_bounds,
        lat_bounds=lat_bounds,
        max_hours=max_hours,
    )
    return retained_hours, lats, lons, significant_wave_height


def _load_ensemble_variables(
    path: str | Path,
    variable_names: tuple[str, ...],
    *,
    ensemble_member: int | None,
    lon_bounds: tuple[float, float] | None,
    lat_bounds: tuple[float, float] | None,
    max_hours: float,
) -> tuple[np.ndarray, ...]:
    try:
        ds = xr.open_dataset(path, engine="scipy")
    except (OSError, TypeError, ValueError):
        ds = xr.open_dataset(path, engine="netcdf4")
    try:
        missing = set(variable_names) - set(ds.data_vars)
        if missing:
            raise KeyError(
                f"IFS forecast file is missing variables: {sorted(missing)}"
            )

        hours = lead_time_hours(ds)
        time_mask = hours <= max_hours
        ds = ds.isel(step=np.flatnonzero(time_mask))

        if lon_bounds is not None:
            ds = ds.sel(longitude=slice(lon_bounds[0], lon_bounds[1]))
        if lat_bounds is not None:
            latitudes = ds["latitude"].values
            if latitudes[0] >= latitudes[-1]:
                ds = ds.sel(latitude=slice(lat_bounds[1], lat_bounds[0]))
            else:
                ds = ds.sel(latitude=slice(lat_bounds[0], lat_bounds[1]))

        if ensemble_member is not None:
            if "number" not in ds.coords:
                raise KeyError("IFS forecast dataset has no 'number' coordinate")
            ds = ds.sel(number=ensemble_member)
        else:
            ds = ds.mean("number")

        lats = np.asarray(ds["latitude"].values, dtype=np.float64)
        lons = np.asarray(ds["longitude"].values, dtype=np.float64)
        retained_hours = lead_time_hours(ds)
        values = tuple(
            np.asarray(ds[name].load().values, dtype=np.float64)
            for name in variable_names
        )
    finally:
        ds.close()

    return retained_hours, lats, lons, *values


def interpolate_field_cubic(
    values: np.ndarray,
    source_hours: np.ndarray,
    target_hours: np.ndarray,
) -> np.ndarray:
    """Interpolate forecast fields along lead time, preserving missing cells.

    Spatial cells with incomplete temporal data are interpolated from their
    available samples only. Cells with fewer than two samples remain masked
    except at target times that exactly match an available sample.
    """
    if values.shape[0] != len(source_hours):
        raise ValueError("First values axis must match the source lead times")
    if len(source_hours) < 2:
        raise ValueError("At least two forecast lead times are required")
    if (
        np.any(target_hours < source_hours[0])
        or np.any(target_hours > source_hours[-1])
    ):
        raise ValueError(
            "Target lead times must lie within the source lead-time range"
        )

    values = np.asarray(values, dtype=np.float64)
    flat_values = values.reshape((len(source_hours), -1))
    flat_result = np.full(
        (len(target_hours), flat_values.shape[1]),
        np.nan,
        dtype=np.float64,
    )
    finite = np.isfinite(flat_values)
    complete_columns = np.all(finite, axis=0)

    if np.any(complete_columns):
        spline = CubicSpline(
            source_hours,
            flat_values[:, complete_columns],
            axis=0,
        )
        flat_result[:, complete_columns] = spline(target_hours)

    for column in np.flatnonzero(~complete_columns & np.any(finite, axis=0)):
        valid = finite[:, column]
        column_hours = source_hours[valid]
        column_values = flat_values[valid, column]
        if len(column_hours) == 1:
            exact = target_hours == column_hours[0]
            flat_result[exact, column] = column_values[0]
            continue

        inside = (target_hours >= column_hours[0]) & (
            target_hours <= column_hours[-1]
        )
        spline = CubicSpline(column_hours, column_values)
        flat_result[inside, column] = spline(target_hours[inside])

    return flat_result.reshape((len(target_hours), *values.shape[1:]))


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
) -> dict[float, np.ndarray]:
    """Resample a scalar forecast field at temporal resolutions."""
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
        fields[resolution_h] = interpolate_field_cubic(
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
) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """Build interpolated U/V fields for the requested temporal resolutions."""
    u_fields = resample_field_at_resolutions(hours, u, resolutions_h)
    v_fields = resample_field_at_resolutions(hours, v, resolutions_h)
    return {
        resolution_h: (u_fields[resolution_h], v_fields[resolution_h])
        for resolution_h in resolutions_h
    }
