# ECMWF IFS forecast downloads

This workflow downloads perturbed ECMWF IFS forecasts from ECMWF Open Data.
It is separate from the ERA5 reanalysis workflow and writes raw GRIB files; it
does not crop, merge, or convert the data for the ERA5 loader.

Install the optional download client:

```bash
uv sync --extra ifs
```

Download a date range:

```bash
uv run scripts/data/download_ifs_forecasts.py \
  --start 2025-01-19 \
  --end 2025-01-20 \
  --output-dir data/ifs_forecasts
```

The downloader requests forecast type `pf` for the `enfo` wind stream
(`10u`, `10v`) and the `waef` wave stream (`swh`, `mwd`, `mwp`). The default
cycle is 00Z. Use `--hour 12` for 12Z or `--streams enfo` to select a single
stream. Forecast lead times default to 0–144 hours every 3 hours and 150–360
hours every 6 hours.

Files are organized by stream and parameter, for example:

```text
data/ifs_forecasts/
├── enfo/
│   ├── 10u/20250119_00z.grib2
│   └── 10v/20250119_00z.grib2
└── waef/
    ├── mwd/20250119_00z.grib2
    ├── mwp/20250119_00z.grib2
    └── swh/20250119_00z.grib2
```

Completed non-empty files are skipped on subsequent runs. These files are not
NetCDF inputs for `routetools.era5`; connecting IFS forecasts to routing
requires a separate forecast-aware data loader.
