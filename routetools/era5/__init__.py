"""ERA5 weather data ingestion for routetools.

This module provides two backends for accessing ERA5 reanalysis data:

1. **CDS API** (``download_cds``): Downloads ERA5 data from the Copernicus
   Climate Data Store using the ``cdsapi`` package.  Requires a CDS account
   and API key.

2. **Google Cloud Storage** (``download_gcs``): Accesses the ERA5 dataset
   stored as Zarr on Google Cloud (via the WeatherBench2 / Pangeo archive).
   No API key required.

Both backends produce NetCDF files on disk that can be loaded with
:func:`load_era5_vectorfield` and :func:`load_era5_wavefield` to obtain
JAX-compatible field closures matching the interface expected by
:func:`routetools.cost.cost_function`.
"""

from routetools.era5.loader import (
    load_era5_vectorfield as load_era5_vectorfield,
    load_era5_wavefield as load_era5_wavefield,
    load_era5_windfield as load_era5_windfield,
)

__all__ = [
    "load_era5_vectorfield",
    "load_era5_wavefield",
    "load_era5_windfield",
]
