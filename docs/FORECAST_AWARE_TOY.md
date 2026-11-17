# Forecast-aware routing toy experiment

This experiment compares routes optimized using the archived deterministic IFS
forecast available at departure with perfect-information routes optimized on
ERA5 reanalysis.  The initial window contains ten Atlantic departures at 00 UTC
from 2025-07-15 through 2025-07-24.  ERA5 extends through 2025-08-08 so every
departure has the same 15-day verification horizon.

## Shared data

The canonical server root is `/home/SHARED/data/weather-routing`.  The 2024 BERS
files and January 2025 continuation are catalogued there as symlinks to the
existing `/codabench/data` files; they are not duplicated.

```text
/home/SHARED/data/weather-routing/
├── era5/
│   ├── 2024/                         # links to existing BERS data
│   └── 2025/
│       ├── 01/                       # links to existing continuation data
│       └── toy_20250715_20250724/
│           └── atlantic/
└── ecmwf-open-data/
    └── 2025/
        └── toy_20250715_20250724/
            └── atlantic/{oper,wave}/
```

ERA5 is sourced anonymously from the public ARCO-ERA5 Zarr archive and is
covered by the Copernicus licence.  Credit Hersbach et al. (2020).  Forecasts
are sourced from ECMWF Open Data on Google Cloud under CC BY 4.0; credit ECMWF.

The crop is the exact BERS Atlantic grid: 25–60°N and 80°W–10°E, inclusive, at
0.25°, with descending latitude and longitudes stored as -80…10.  ERA5 remains
hourly.  IFS retains its native 85 lead times: every 3 hours through 144 hours,
then every 6 hours through 360 hours.

Variables use the ERA5 names in both products:

- `10m_u_component_of_wind`
- `10m_v_component_of_wind`
- `significant_height_of_combined_wind_waves_and_swell`
- `mean_wave_direction`
- `mean_wave_period`

## Completed toy dataset

The initial dataset was completed and validated on 2026-11-10:

- ERA5: two files covering 2025-07-15 00:00 through 2025-08-08 23:00
  (600 hourly timestamps; about 583 MiB on disk).
- IFS: ten `oper` files and ten `wave` files, one 00 UTC run per day from
  2025-07-15 through 2025-07-24 (85 lead times per file; about 827 MiB).
- Total retained size: 1,476,993,245 bytes (about 1.48 GB).

All files use the exact 141 x 361 BERS Atlantic grid.  The IFS files were
also loaded through `routetools.era5.loader` and sampled at lead hours 0, 3,
144, and 360.  Download logs are in
`/home/SHARED/data/weather-routing/logs/toy_20250715_20250724`.

Install the forecast readers with `uv sync --extra forecast`.  On Linux this
also installs the bundled ecCodes runtime, avoiding dependence on the server's
system ecCodes version.

## Download commands on vangelis

```bash
ERA5_ROOT=/home/SHARED/data/weather-routing/era5/2025/toy_20250715_20250724/atlantic
IFS_ROOT=/home/SHARED/data/weather-routing/ecmwf-open-data/2025/toy_20250715_20250724
LOG_ROOT=/home/SHARED/data/weather-routing/logs/toy_20250715_20250724

uv run scripts/data/download_era5_window.py \
  --start 2025-07-15 --end 2025-08-08 --corridor atlantic \
  --output-dir "$ERA5_ROOT" --log-file "$LOG_ROOT/era5.log"

uv run --extra forecast scripts/data/download_ecmwf_forecasts.py \
  --start 2025-07-15 --end 2025-07-24 --hour 0 --corridor atlantic \
  --output-dir "$IFS_ROOT" --log-file "$LOG_ROOT/ifs.log"
```

Both commands skip completed files, retry transient failures, and publish via an
atomic rename.  Temporary global GRIB files are removed immediately after each
run is cropped.

## Run the routing comparison

The initial routing comparison uses the Atlantic optimised WPS case (`AO_WPS`)
for the same ten 00 UTC departures.  Each departure produces two routes using
the same seed and optimisation budget: one using the IFS forecast and one using
perfect-information ERA5.  Both routes are then scored on both weather
products.

```bash
uv sync --extra forecast --extra cuda --no-dev --locked

uv run --no-sync scripts/experiments/forecast_aware_toy.py
```

Results are resumable and are written by default under
`/home/SHARED/data/weather-routing/results/toy_20250715_20250724/AO_WPS`.
There is one JSON and one compressed route file per departure plus a cumulative
`summary.csv`.  The main comparison column is `information_regret_mwh`, defined
as the ERA5-scored energy of the forecast-optimised route minus the ERA5-scored
energy of the ERA5-optimised route.  Positive values are the cost of acting on
the forecast relative to perfect information.

The completed ten-departure results and interpretation are in
[`FORECAST_AWARE_TOY_RESULTS.md`](FORECAST_AWARE_TOY_RESULTS.md).
