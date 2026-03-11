"""Download ERA5 data from Google Cloud Storage (Zarr format).

Accesses ERA5 reanalysis data from the WeatherBench2 / Pangeo archive
hosted on Google Cloud.  No API key is required — the data is publicly
accessible.

References
----------
- WeatherBench2: https://weatherbench2.readthedocs.io/
- ERA5 on GCS: gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import xarray as xr

logger = logging.getLogger(__name__)

# Google Cloud Storage paths for ERA5 (ARCO-ERA5, 0.25° hourly)
GCS_ERA5_SINGLE_LEVEL = (
    "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
)

# Route corridor bounds: {name: (lat_min, lat_max, lon_min, lon_max)}
# Longitudes in 0..360 convention for the Pacific to handle antimeridian.
CORRIDORS: dict[str, tuple[float, float, float, float]] = {
    "atlantic": (25.0, 60.0, -80.0, 10.0),
    "pacific": (15.0, 55.0, 120.0, 240.0),
}

# ERA5 variable names in the ARCO-ERA5 Zarr archive
WIND_U_VAR = "10m_u_component_of_wind"
WIND_V_VAR = "10m_v_component_of_wind"
WAVE_HS_VAR = "significant_height_of_combined_wind_waves_and_swell"
WAVE_DIR_VAR = "mean_wave_direction"


def _ensure_deps() -> None:
    """Check that xarray, gcsfs, and zarr are available."""
    missing = []
    for pkg in ("xarray", "gcsfs", "zarr"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        raise ImportError(
            f"The following packages are required for GCS downloads: "
            f"{', '.join(missing)}.  Install them with:\n"
            f"  pip install {' '.join(missing)}"
        )


def _open_era5_zarr(variables: list[str]) -> xr.Dataset:
    """Open the ERA5 Zarr store on GCS and select the given variables."""
    import gcsfs
    import xarray as xr

    fs = gcsfs.GCSFileSystem(token="anon")
    store = fs.get_mapper(GCS_ERA5_SINGLE_LEVEL)
    ds = xr.open_zarr(store, consolidated=True)
    return ds[variables]


def _detect_time_dim(ds: xr.Dataset) -> str:
    """Return the name of the time dimension ('time' or 'valid_time')."""
    for name in ("time", "valid_time"):
        if name in ds.dims or name in ds.coords:
            return name
    raise KeyError(
        f"Cannot find time dimension in dataset. "
        f"Available dims: {list(ds.dims)}, coords: {list(ds.coords)}"
    )


def _normalize_time_dim(ds: xr.Dataset) -> xr.Dataset:
    """Rename the time dimension to 'valid_time' for CDS compatibility."""
    time_dim = _detect_time_dim(ds)
    if time_dim != "valid_time":
        ds = ds.rename({time_dim: "valid_time"})
    return ds


def _select_corridor(
    ds: xr.Dataset,
    corridor: str,
    year: str = "2024",
    months: list[int] | None = None,
    time_step: int = 6,
) -> xr.Dataset:
    """Subset dataset to a corridor, year, and temporal step.

    Parameters
    ----------
    ds : xarray.Dataset
        Full ERA5 dataset.
    corridor : str
        Name of the corridor (``"atlantic"`` or ``"pacific"``).
    year : str
        Year to select.
    months : list[int], optional
        Months to include (1-12). Default: all 12.
    time_step : int
        Hours between time steps (default 6 for 6-hourly).

    Returns
    -------
    xarray.Dataset
        Subset dataset.
    """
    lat_min, lat_max, lon_min, lon_max = CORRIDORS[corridor]
    time_dim = _detect_time_dim(ds)

    # Time selection
    if months is not None:
        first_month = min(months)
        last_month = max(months)
        t_start = f"{year}-{first_month:02d}-01"
        # Use end of last month by selecting up to the start of next month
        if last_month == 12:
            t_end = f"{year}-12-31T23:59:59"
        else:
            t_end = f"{year}-{last_month + 1:02d}-01"
        ds = ds.sel({time_dim: slice(t_start, t_end)})
        # Filter to exact months in case non-contiguous months are requested
        ds = ds.sel({time_dim: ds[time_dim].dt.month.isin(months)})
    else:
        ds = ds.sel({time_dim: slice(f"{year}-01-01", f"{year}-12-31")})
    if time_step > 1:
        ds = ds.isel({time_dim: slice(None, None, time_step)})

    # Spatial selection
    # ERA5 latitude is typically 90 to -90 (descending)
    lat_dim = "latitude" if "latitude" in ds.dims else "lat"
    lon_dim = "longitude" if "longitude" in ds.dims else "lon"

    lats = ds[lat_dim].values
    lons = ds[lon_dim].values

    # Latitude selection (ERA5 lat is often descending: 90 → -90)
    if lats[0] > lats[-1]:
        lat_slice = slice(lat_max, lat_min)
    else:
        lat_slice = slice(lat_min, lat_max)

    # Determine whether the dataset uses [0, 360) longitudes
    ds_uses_0_360 = float(np.min(lons)) >= 0 and float(np.max(lons)) > 180

    if lon_min < 0 and ds_uses_0_360:
        # Corridor spans negative longitudes but dataset is in [0, 360).
        # E.g. Atlantic (-80, 10) → need [280, 360) ∪ [0, 10] then
        # relabel to [-80, 10].
        import xarray as xr

        lon_min_360 = lon_min % 360  # -80 → 280
        part_west = ds.sel({lat_dim: lat_slice, lon_dim: slice(lon_min_360, 360.0)})
        part_east = ds.sel({lat_dim: lat_slice, lon_dim: slice(0.0, lon_max)})
        ds = xr.concat([part_west, part_east], dim=lon_dim)
        # Relabel longitudes to [-180, 180) convention
        new_lons = ds[lon_dim].values.copy()
        new_lons[new_lons >= 180] -= 360
        ds = ds.assign_coords({lon_dim: new_lons})
    elif lon_max > 180:
        # Pacific-style corridor already in [0, 360) range (e.g. 120–240)
        # Convert dataset lons from [-180, 180] to [0, 360] if needed
        if np.any(lons < 0):
            ds = ds.assign_coords({lon_dim: np.mod(lons, 360)})
            ds = ds.sortby(lon_dim)

        ds = ds.sel({lat_dim: lat_slice, lon_dim: slice(lon_min, lon_max)})
    else:
        # Both corridor and dataset are in compatible ranges
        ds = ds.sel({lat_dim: lat_slice, lon_dim: slice(lon_min, lon_max)})

    return ds


def _output_filename(
    output_dir: Path,
    field: str,
    corridor: str,
    year: str,
    months: list[int] | None = None,
) -> Path:
    """Build the output filename, including month range for partial years."""
    if months is not None and sorted(months) != list(range(1, 13)):
        m_min, m_max = min(months), max(months)
        suffix = f"{m_min:02d}" if m_min == m_max else f"{m_min:02d}-{m_max:02d}"
        return output_dir / f"era5_{field}_{corridor}_{year}_{suffix}.nc"
    return output_dir / f"era5_{field}_{corridor}_{year}.nc"


def download_era5_wind_gcs(
    output_dir: str | Path = "data/era5",
    corridor: str = "atlantic",
    year: str = "2024",
    months: list[int] | None = None,
    time_step: int = 6,
) -> Path:
    """Download ERA5 10-m wind data from GCS and save as NetCDF.

    Parameters
    ----------
    output_dir : str or Path
        Output directory.
    corridor : str
        Route corridor name.
    year : str
        Year to download.
    months : list[int], optional
        Months to include (1-12). Default: all 12.
    time_step : int
        Hours between time steps (default 6).

    Returns
    -------
    Path
        Path to saved NetCDF file.
    """
    _ensure_deps()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = _output_filename(output_dir, "wind", corridor, year, months)

    if filename.exists():
        logger.info("File already exists, skipping: %s", filename)
        return filename

    logger.info("Opening ERA5 wind data on GCS for %s/%s ...", corridor, year)
    ds = _open_era5_zarr([WIND_U_VAR, WIND_V_VAR])
    ds = _select_corridor(ds, corridor, year, months, time_step)

    # Normalize time dimension to 'valid_time' for CDS compatibility
    ds = _normalize_time_dim(ds)

    logger.info("Downloading and saving to %s ...", filename)
    ds.to_netcdf(filename)
    logger.info("Saved: %s (%.1f MB)", filename, filename.stat().st_size / 1e6)
    return filename


def download_era5_waves_gcs(
    output_dir: str | Path = "data/era5",
    corridor: str = "atlantic",
    year: str = "2024",
    months: list[int] | None = None,
    time_step: int = 6,
) -> Path:
    """Download ERA5 wave data (Hs + direction) from GCS and save as NetCDF.

    Parameters
    ----------
    output_dir : str or Path
        Output directory.
    corridor : str
        Route corridor name.
    year : str
        Year to download.
    months : list[int], optional
        Months to include (1-12). Default: all 12.
    time_step : int
        Hours between time steps (default 6).

    Returns
    -------
    Path
        Path to saved NetCDF file.
    """
    _ensure_deps()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = _output_filename(output_dir, "waves", corridor, year, months)

    if filename.exists():
        logger.info("File already exists, skipping: %s", filename)
        return filename

    logger.info("Opening ERA5 wave data on GCS for %s/%s ...", corridor, year)
    ds = _open_era5_zarr([WAVE_HS_VAR, WAVE_DIR_VAR])
    ds = _select_corridor(ds, corridor, year, months, time_step)

    # Normalize time dimension to 'valid_time' for CDS compatibility
    ds = _normalize_time_dim(ds)

    logger.info("Downloading and saving to %s ...", filename)
    ds.to_netcdf(filename)
    logger.info("Saved: %s (%.1f MB)", filename, filename.stat().st_size / 1e6)
    return filename


def download_all_gcs(
    output_dir: str | Path = "data/era5",
    year: str = "2024",
    months: list[int] | None = None,
    corridors: list[str] | None = None,
    time_step: int = 6,
) -> list[Path]:
    """Download all ERA5 data needed for SWOPP3 from GCS.

    Parameters
    ----------
    output_dir : str or Path
        Output directory.
    year : str
        Year to download.
    months : list[int], optional
        Months to include (1-12). Default: all 12.
    corridors : list[str], optional
        Corridors to download (default: both).
    time_step : int
        Hours between time steps.

    Returns
    -------
    list[Path]
        Paths to all downloaded NetCDF files.
    """
    corridors = corridors or ["atlantic", "pacific"]
    files: list[Path] = []
    for corridor in corridors:
        files.append(
            download_era5_wind_gcs(
                output_dir=output_dir,
                corridor=corridor,
                year=year,
                months=months,
                time_step=time_step,
            )
        )
        files.append(
            download_era5_waves_gcs(
                output_dir=output_dir,
                corridor=corridor,
                year=year,
                months=months,
                time_step=time_step,
            )
        )
    return files
