#!/usr/bin/env python
"""Download ERA5 weather data for SWOPP3 route corridors.

Provides a single entry point for downloading ERA5 wind and wave data
needed by the routetools weather-routing pipeline.  Two backends are
supported:

- **GCS** (default): Downloads from the public Google Cloud archive
  (WeatherBench2 / Pangeo).  No API key required.
- **CDS**: Downloads from the Copernicus Climate Data Store.
  Requires ``cdsapi`` and a valid CDS API key.

Usage
-----
Download all data for 2024 (both Atlantic and Pacific corridors)::

    uv run scripts/download_era5.py

Download only the Atlantic corridor for 2023 via GCS::

    uv run scripts/download_era5.py --corridor atlantic --year 2023

Download via the CDS API instead::

    uv run scripts/download_era5.py --backend cds

Output
------
Downloaded files are stored in ``data/era5/`` by default::

    data/era5/
    ├── era5_wind_atlantic_2024.nc
    ├── era5_waves_atlantic_2024.nc
    ├── era5_wind_pacific_2024.nc
    └── era5_waves_pacific_2024.nc

These filenames are the defaults consumed by ``scripts/swopp3_run.py``. A new
user can therefore run the full SWOPP3 pipeline without overriding any paths::

    uv run scripts/download_era5.py
    uv run scripts/swopp3_run.py

If you download a different year or only one corridor, ``scripts/swopp3_run.py``
must be given matching ``--wind-path*`` and ``--wave-path*`` options.

The downloaded files can also be loaded programmatically via::

    from routetools.era5 import load_era5_windfield, load_era5_wavefield

    windfield = load_era5_windfield("data/era5/era5_wind_atlantic_2024.nc")
    wavefield = load_era5_wavefield("data/era5/era5_waves_atlantic_2024.nc")
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

BACKENDS = ("gcs", "cds")
CORRIDORS = ("atlantic", "pacific")


def main() -> None:
    """Parse CLI arguments and download ERA5 files for selected corridors."""
    parser = argparse.ArgumentParser(
        description="Download ERA5 wind and wave data for SWOPP3 corridors.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--backend",
        choices=BACKENDS,
        default="gcs",
        help="Download backend: 'gcs' (default, no API key) or 'cds'.",
    )
    parser.add_argument(
        "--corridor",
        choices=CORRIDORS,
        default=None,
        help="Download a single corridor. Default: both.",
    )
    parser.add_argument(
        "--year",
        default="2024",
        help="Year to download (default: 2024).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/era5",
        help="Output directory (default: data/era5).",
    )
    parser.add_argument(
        "--time-step",
        type=int,
        default=6,
        help="Hours between time steps (default: 6). GCS backend only.",
    )
    args = parser.parse_args()

    corridors = [args.corridor] if args.corridor else list(CORRIDORS)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Backend=%s  Year=%s  Corridors=%s  Output=%s",
        args.backend,
        args.year,
        corridors,
        output_dir,
    )

    if args.backend == "gcs":
        from routetools.era5.download_gcs import download_all_gcs

        files = download_all_gcs(
            output_dir=output_dir,
            year=args.year,
            corridors=corridors,
            time_step=args.time_step,
        )
    elif args.backend == "cds":
        from routetools.era5.download_cds import download_all

        files = download_all(
            output_dir=output_dir,
            year=args.year,
            corridors=corridors,
        )
    else:
        print(f"Unknown backend: {args.backend}", file=sys.stderr)
        sys.exit(1)

    logger.info("Download complete. %d files:", len(files))
    for f in files:
        logger.info("  %s", f)


if __name__ == "__main__":
    main()
