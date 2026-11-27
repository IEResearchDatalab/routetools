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
    try:
        ds = xr.open_dataset(path, engine="scipy")
    except (OSError, TypeError, ValueError):
        ds = xr.open_dataset(path, engine="netcdf4")
    try:
        if "10m_u_component_of_wind" not in ds.data_vars:
            raise KeyError(
                "IFS wind file must contain '10m_u_component_of_wind'"
            )
        if "10m_v_component_of_wind" not in ds.data_vars:
            raise KeyError(
                "IFS wind file must contain '10m_v_component_of_wind'"
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

        u = np.asarray(ds["10m_u_component_of_wind"].load().values, dtype=np.float64)
        v = np.asarray(ds["10m_v_component_of_wind"].load().values, dtype=np.float64)
        lats = np.asarray(ds["latitude"].values, dtype=np.float64)
        lons = np.asarray(ds["longitude"].values, dtype=np.float64)
        retained_hours = lead_time_hours(ds)
    finally:
        ds.close()

    return retained_hours, lats, lons, u, v


def interpolate_field_cubic(
    values: np.ndarray,
    source_hours: np.ndarray,
    target_hours: np.ndarray,
) -> np.ndarray:
    """Interpolate forecast fields along the lead-time axis with cubic splines."""
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

    spline = CubicSpline(source_hours, values, axis=0)
    return np.asarray(spline(target_hours), dtype=np.float64)


def resample_wind_at_resolutions(
    hours: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    resolutions_h: tuple[float, ...] = SUPPORTED_INTERPOLATION_RESOLUTIONS_H,
) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """Build interpolated U/V fields for the requested temporal resolutions."""
    fields: dict[float, tuple[np.ndarray, np.ndarray]] = {}

    for resolution_h in resolutions_h:
        stride = temporal_stride_for_resolution(
            NATIVE_FORECAST_RESOLUTION_H,
            resolution_h,
        )
        if resolution_h == NATIVE_FORECAST_RESOLUTION_H:
            fields[resolution_h] = (u.copy(), v.copy())
            continue

        source_hours = hours[::stride]
        fields[resolution_h] = (
            interpolate_field_cubic(u[::stride], source_hours, hours),
            interpolate_field_cubic(v[::stride], source_hours, hours),
        )

    return fields
