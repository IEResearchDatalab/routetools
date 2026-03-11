"""Download ERA5 data from the Copernicus Climate Data Store (CDS).

Requires the ``cdsapi`` package and valid CDS API credentials.
See https://cds.climate.copernicus.eu/how-to-api for setup instructions.

The functions in this module download ERA5 reanalysis single-level and wave
data for the route corridors needed by SWOPP3 and store them as NetCDF files.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Route corridor bounding boxes (North, West, South, East)
# With generous padding around the actual route endpoints.
# ---------------------------------------------------------------------------
CORRIDORS: dict[str, list[float]] = {
    # Atlantic: Santander (43.6, -4.0) → New York (40.6, -69.0)
    # Pad: +10° N/S, +5° E/W
    "atlantic": [60, -80, 25, 10],
    # Pacific: Tokyo (34.8, 140.0) → Los Angeles (34.4, -121.0)
    # Great-circle goes up to ~50°N; pad generously
    # Split into two boxes? No — ERA5 handles wrap-around.
    # We use lon 120E to 240E (== -120W) via 0-360 convention.
    "pacific": [55, 120, 15, 240],
}

# ERA5 variables we need
WIND_VARIABLES = ["10m_u_component_of_wind", "10m_v_component_of_wind"]
WAVE_VARIABLES = [
    "significant_height_of_combined_wind_waves_and_swell",
    "mean_wave_direction",
]

# Default years / months for SWOPP3 (all of 2024)
DEFAULT_YEAR = "2024"
DEFAULT_MONTHS = [f"{m:02d}" for m in range(1, 13)]
DEFAULT_DAYS = [f"{d:02d}" for d in range(1, 32)]
DEFAULT_TIMES = [f"{h:02d}:00" for h in range(0, 24, 6)]  # 6-hourly


def _output_filename(
    output_dir: Path,
    field: str,
    corridor: str,
    year: str,
    months: list[str],
) -> Path:
    """Build the output filename, including month range for partial years."""
    all_months = [f"{m:02d}" for m in range(1, 13)]
    if sorted(months) != all_months:
        m_min, m_max = months[0], months[-1]
        suffix = m_min if m_min == m_max else f"{m_min}-{m_max}"
        return output_dir / f"era5_{field}_{corridor}_{year}_{suffix}.nc"
    return output_dir / f"era5_{field}_{corridor}_{year}.nc"


def _ensure_cdsapi() -> Any:
    """Import and return a CDS API client, raising a clear error if missing."""
    try:
        import cdsapi
    except ImportError as exc:
        raise ImportError(
            "The 'cdsapi' package is required for CDS downloads.  "
            "Install it with:  pip install cdsapi\n"
            "Then configure your API key: https://cds.climate.copernicus.eu/how-to-api"
        ) from exc
    return cdsapi.Client()


def download_era5_wind(
    output_dir: str | Path = "data/era5",
    corridor: str = "atlantic",
    year: str = DEFAULT_YEAR,
    months: list[str] | None = None,
    days: list[str] | None = None,
    times: list[str] | None = None,
    grid: list[float] | None = None,
) -> Path:
    """Download ERA5 10-m wind components for a route corridor.

    Parameters
    ----------
    output_dir : str or Path
        Directory to store downloaded files.
    corridor : str
        One of ``"atlantic"`` or ``"pacific"``.
    year : str
        Year to download (default ``"2024"``).
    months : list[str], optional
        Months (default: all 12).
    days : list[str], optional
        Days (default: all 31).
    times : list[str], optional
        Hours in ``"HH:00"`` format (default: 6-hourly).
    grid : list[float], optional
        ``[lat_res, lon_res]`` in degrees (default ``[0.25, 0.25]``).

    Returns
    -------
    Path
        Path to the downloaded NetCDF file.
    """
    client = _ensure_cdsapi()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    months = months or DEFAULT_MONTHS
    days = days or DEFAULT_DAYS
    times = times or DEFAULT_TIMES
    grid = grid or [0.25, 0.25]

    area = CORRIDORS[corridor]
    filename = _output_filename(output_dir, "wind", corridor, year, months)

    if filename.exists():
        logger.info("File already exists, skipping download: %s", filename)
        return filename

    logger.info(
        "Downloading ERA5 wind data for %s corridor, year %s ...", corridor, year
    )

    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": WIND_VARIABLES,
            "year": year,
            "month": months,
            "day": days,
            "time": times,
            "area": area,
            "grid": grid,
            "data_format": "netcdf",
        },
        str(filename),
    )

    logger.info("Downloaded: %s", filename)
    return filename


def download_era5_waves(
    output_dir: str | Path = "data/era5",
    corridor: str = "atlantic",
    year: str = DEFAULT_YEAR,
    months: list[str] | None = None,
    days: list[str] | None = None,
    times: list[str] | None = None,
    grid: list[float] | None = None,
) -> Path:
    """Download ERA5 wave data (Hs and mean direction) for a route corridor.

    Parameters
    ----------
    output_dir : str or Path
        Directory to store downloaded files.
    corridor : str
        One of ``"atlantic"`` or ``"pacific"``.
    year : str
        Year to download (default ``"2024"``).
    months : list[str], optional
        Months (default: all 12).
    days : list[str], optional
        Days (default: all 31).
    times : list[str], optional
        Hours in ``"HH:00"`` format (default: 6-hourly).
    grid : list[float], optional
        ``[lat_res, lon_res]`` in degrees (default ``[0.25, 0.25]``).

    Returns
    -------
    Path
        Path to the downloaded NetCDF file.
    """
    client = _ensure_cdsapi()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    months = months or DEFAULT_MONTHS
    days = days or DEFAULT_DAYS
    times = times or DEFAULT_TIMES
    grid = grid or [0.25, 0.25]

    area = CORRIDORS[corridor]
    filename = _output_filename(output_dir, "waves", corridor, year, months)

    if filename.exists():
        logger.info("File already exists, skipping download: %s", filename)
        return filename

    logger.info(
        "Downloading ERA5 wave data for %s corridor, year %s ...", corridor, year
    )

    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": WAVE_VARIABLES,
            "year": year,
            "month": months,
            "day": days,
            "time": times,
            "area": area,
            "grid": grid,
            "data_format": "netcdf",
        },
        str(filename),
    )

    logger.info("Downloaded: %s", filename)
    return filename


def download_all(
    output_dir: str | Path = "data/era5",
    year: str = DEFAULT_YEAR,
    corridors: list[str] | None = None,
    **kwargs: object,
) -> list[Path]:
    """Download all ERA5 data needed for SWOPP3.

    Downloads wind and wave data for both Atlantic and Pacific corridors.

    Parameters
    ----------
    output_dir : str or Path
        Directory to store downloaded files.
    year : str
        Year to download.
    corridors : list[str], optional
        Corridors to download (default: both).
    **kwargs
        Forwarded to :func:`download_era5_wind` and
        :func:`download_era5_waves`.

    Returns
    -------
    list[Path]
        Paths to all downloaded NetCDF files.
    """
    corridors = corridors or ["atlantic", "pacific"]
    files: list[Path] = []
    for corridor in corridors:
        files.append(
            download_era5_wind(
                output_dir=output_dir, corridor=corridor, year=year, **kwargs
            )
        )
        files.append(
            download_era5_waves(
                output_dir=output_dir, corridor=corridor, year=year, **kwargs
            )
        )
    return files
