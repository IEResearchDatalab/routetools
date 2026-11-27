"""Load ECMWF IFS ensemble forecasts from NetCDF or GRIB files."""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import xarray as xr

GRIB_VARIABLE_NAMES = {"10u": "u10", "10v": "v10", "swh": "swh"}


def _is_grib_file(path: str | Path) -> bool:
    """Return whether *path* has a GRIB filename suffix."""
    return Path(path).suffix.lower() in {".grib", ".grib2", ".grb", ".grb2"}


def lead_time_hours(ds: xr.Dataset) -> np.ndarray:
    """Return forecast lead times in hours from an IFS dataset."""
    if "step" not in ds.coords:
        raise KeyError("IFS forecast dataset has no 'step' coordinate")
    values = ds["step"].values / np.timedelta64(1, "h")
    return np.asarray(values, dtype=np.float64)


def load_ensemble_wind(
    path: str | Path,
    *,
    v_path: str | Path | None = None,
    ensemble_member: int | None = None,
    lon_bounds: tuple[float, float] | None = None,
    lat_bounds: tuple[float, float] | None = None,
    max_hours: float = 144.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load U/V wind components and grid metadata from an IFS forecast.

    NetCDF files containing both components can be given as *path*. For GRIB,
    provide the U-component path as *path* and V-component path as *v_path*.
    Ensemble members are averaged by default or selected by *ensemble_member*.
    """
    if _is_grib_file(path):
        if v_path is None:
            raise ValueError("A separate GRIB V-wind file must be provided as v_path")
        u_result = _load_ensemble_variables(
            path,
            ("u10",),
            grib_short_names=("10u",),
            ensemble_member=ensemble_member,
            lon_bounds=lon_bounds,
            lat_bounds=lat_bounds,
            max_hours=max_hours,
        )
        v_result = _load_ensemble_variables(
            v_path,
            ("v10",),
            grib_short_names=("10v",),
            ensemble_member=ensemble_member,
            lon_bounds=lon_bounds,
            lat_bounds=lat_bounds,
            max_hours=max_hours,
        )
        for axis_name, u_axis, v_axis in zip(
            ("lead times", "latitudes", "longitudes"),
            u_result[:3],
            v_result[:3],
            strict=True,
        ):
            if not np.array_equal(u_axis, v_axis):
                raise ValueError(f"IFS U/V wind files have different {axis_name}")
        return (
            u_result[0],
            u_result[1],
            u_result[2],
            u_result[3],
            v_result[3],
        )

    if v_path is not None:
        raise ValueError("v_path is only used when path is a GRIB file")

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
    if _is_grib_file(path):
        (
            retained_hours,
            lats,
            lons,
            significant_wave_height,
        ) = _load_ensemble_variables(
            path,
            ("swh",),
            grib_short_names=("swh",),
            ensemble_member=ensemble_member,
            lon_bounds=lon_bounds,
            lat_bounds=lat_bounds,
            max_hours=max_hours,
        )
        return retained_hours, lats, lons, significant_wave_height

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
    grib_short_names: tuple[str, ...] | None = None,
    ensemble_member: int | None,
    lon_bounds: tuple[float, float] | None,
    lat_bounds: tuple[float, float] | None,
    max_hours: float,
) -> tuple[np.ndarray, ...]:
    if _is_grib_file(path):
        if grib_short_names is None or len(grib_short_names) != len(variable_names):
            raise ValueError("A GRIB short name is required for each requested field")
        try:
            importlib.import_module("cfgrib")
        except ImportError as exc:
            raise ImportError(
                "Reading IFS GRIB files requires cfgrib. Install it with "
                "'uv sync --extra ifs'."
            ) from exc
        ds = xr.open_dataset(
            path,
            engine="cfgrib",
            backend_kwargs={
                "indexpath": "",
                "filter_by_keys": {"shortName": grib_short_names[0]},
            },
        )
        dataset_variable_names = tuple(
            GRIB_VARIABLE_NAMES[short_name] for short_name in grib_short_names
        )
    else:
        try:
            ds = xr.open_dataset(path, engine="scipy")
        except (OSError, TypeError, ValueError):
            ds = xr.open_dataset(path, engine="netcdf4")
        dataset_variable_names = variable_names
    try:
        missing = set(dataset_variable_names) - set(ds.data_vars)
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
        if ds.sizes.get("latitude", 0) == 0 or ds.sizes.get("longitude", 0) == 0:
            raise ValueError(
                "Requested crop does not overlap the forecast grid; check "
                "longitude and latitude bounds"
            )

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
            for name in dataset_variable_names
        )
    finally:
        ds.close()

    return retained_hours, lats, lons, *values
