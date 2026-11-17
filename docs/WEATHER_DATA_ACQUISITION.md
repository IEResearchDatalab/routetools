# January wind and wave data on Evangelis

12 November 2026

**The January acquisition pilot is complete and validated. Its files already use the permanent shared dataset layout.** Adding dates, forecast issues or missing leads uses the same script and extends these folders. There is no new demo dataset to migrate later. Work is on branch `codex/weather-data-january` in `/home/SHARED/routetools-weather-data`; data belong to the team under `/home/SHARED/data/weather-routing`, outside Git. Existing July and other legacy experiment paths remain untouched for compatibility.

## What is available now

| Product                              | Actual coverage                                                                                                     | Completed NetCDF files |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------- | ---------------------: |
| Deterministic/control wind and waves | 15 and 16 January 2025, 00 and 12 UTC; all 85 leads per issue, through +360 h                                       |                    680 |
| Ensemble wind and waves              | 15 January, 00 and 12 UTC; leads 0, 24, 72, 144, 240 and 360 h; control plus all 50 perturbed members at every lead |                     24 |
| Hourly ERA5 wind and waves           | 15–31 January, plus a fresh-download check for 1 January                                                            |                     36 |
| **Total**                            | **740 audited files; 655,684,639 bytes, approximately 656 MB, excluding receipts/logs/figures**                     |                **740** |

The ensemble pilot deliberately samples six leads, rather than complete 85-lead runs. Every existing file is a complete production shard; missing leads are absent files, not padded values or reduced-member ensembles. These sparse ensembles are suitable for the present data diagnostics, but are not yet a complete input for continuous voyage routing.

## Sources and download method

**Forecasts:** ECMWF operational IFS, cycle 49r1, issued in 2025. The public Google Cloud mirror distributes GRIB2 files and JSON-lines indexes. The downloader reads each index, checks the exact parameter/member/run/lead inventory, then retrieves the selected GRIB messages using HTTPS byte ranges. At most eight range requests run concurrently. It checks HTTP 206, byte lengths, object-version headers and decoded GRIB timestamps. ecCodes/cfgrib decodes the messages; the script crops them without interpolation and writes NetCDF4. Temporary global GRIBs are removed after processing. Google hosts the mirror; ECMWF produces the forecasts. [ECMWF Open Data](https://www.ecmwf.int/en/forecasts/datasets/open-data)

Example source index:

```text
https://storage.googleapis.com/ecmwf-open-data/20250115/00z/ifs/0p25/waef/20250115000000-72h-waef-ef.index
```

**Reanalysis:** ECMWF ERA5 via Google's public ARCO-ERA5 Zarr copy:

```text
gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3
```

xarray/gcsfs reads this anonymously. The code checks the archive's `valid_time_stop` to require final ERA5 rather than preliminary ERA5T. Most January reference data already existed in the winter files, so the script validated and reused them locally. A separate 1 January download tested the actual public Zarr access successfully. ARCO is a transformed distribution of ERA5; its 0.25° wave grid must not be confused with finer independent wave information. A CDS cross-check remains future validation. [ARCO source and processing documentation](https://github.com/google-research/arco-era5)

IFS attribution is ECMWF, CC BY 4.0. ERA5 attribution follows the Copernicus terms and credits ECMWF/Copernicus and Hersbach et al. The source URLs and applicable source metadata are retained with the data.

## Format and compatibility with the old NC files

**Yes: these are `.nc` NetCDF4 files, opened using xarray like our existing weather files.** They use float32 fields, lossless deflate level 4 compression and the exact BERS Atlantic grid: 141 latitudes from 60 to 25°N, descending, and 361 longitudes from −80 to 10°, ascending, at 0.25°. No additional regridding was performed.

| Variable name, unchanged from existing ERA5 conventions | Units                    |
| ------------------------------------------------------- | ------------------------ |
| `10m_u_component_of_wind`                               | m/s, eastward component  |
| `10m_v_component_of_wind`                               | m/s, northward component |
| `significant_height_of_combined_wind_waves_and_swell`   | m                        |
| `mean_wave_direction`                                   | degrees, direction from  |
| `mean_wave_period`                                      | s                        |

ERA5 has dimensions `(valid_time: 24, latitude: 141, longitude: 361)` in one file per UTC day and field group. Deterministic forecasts have `(step: 1, latitude: 141, longitude: 361)`. Ensemble forecasts add `number: 51`, with **0 = control, 1–50 = perturbed members**. Forecast `time` is the scalar initialization; `valid_time = time + step`. Initialization is not historical availability time: release latency remains a separate assumption for later routing experiments.

Forecast leads retain the disseminated schedule: three-hourly through 144 h and six-hourly thereafter through 360 h for 00/12 UTC. Multiple shards can be opened with `xr.open_mfdataset(paths, combine="by_coords")`. Ensemble member selection and combining lead files need an explicit reader step; these files do not make the old deterministic routing loader ensemble-aware automatically. Keep wind and wave members paired within their initialization.

## Permanent paths and safe extension

```text
/home/SHARED/data/weather-routing/
  ifs/v1/dataset.json
  ifs/v1/atlantic/2025/01/20250115T0000Z/
    oper/lead_072.nc                 # deterministic wind
    wave/lead_072.nc                 # deterministic waves
    enfo/lead_072.nc                 # all 51 wind members
    waef/lead_072.nc                 # all 51 wave members
  era5/v1/dataset.json
  era5/v1/atlantic/2025/01/
    20250118_wind.nc
    20250118_waves.nc
  diagnostics/january_2025/
  logs/acquisition/
```

Each NC has a neighbouring JSON receipt containing the request, source index/URLs or reused local source, source headers, selected byte ranges, SHA256, units/coverage audit, size and elapsed time. Source files are never changed. Same-output writes are locked, new outputs use temporary files and atomic rename, and resume requires a matching receipt, checksum and decoded coverage. File existence alone never means complete. Files and directories are group writable, inheriting the shared `nav` group.

During development, acquisition processes ran while the script was updated. Affected receipts explicitly mark the executing code version as uncertain; the later independent audit records the frozen validation code hash. New processes capture the script hash at startup. The original source provenance and final NC hashes remain preserved.

One extra availability check found no deterministic `scda` GRIB inventory for 15 January 06 UTC, although an `enfo` inventory exists. A July result must not be generalized to every January stream. The production pilot and commands below use **00/12 UTC**; four-cycle annual acquisition needs further inventory and a declared control-source strategy.

## Script and commands

Use the existing repository environment; no packages were installed. Its `.venv` currently points to the existing forecast environment. Run from the new worktree so imports use this branch:

```bash
cd /home/SHARED/routetools-weather-data
export PYTHONPATH="$PWD"

# Reproduce the sampled ensemble acquisition; verified files are skipped.
.venv/bin/python scripts/data/acquire_weather.py ifs \
  --start 2025-01-15 --end 2025-01-15 --hours 0 12 \
  --streams enfo waef --leads 0 24 72 144 240 360 --workers 8

# Extend to all 85 ensemble leads for those same issues; existing leads are reused.
.venv/bin/python scripts/data/acquire_weather.py ifs \
  --start 2025-01-15 --end 2025-01-15 --hours 0 12 \
  --streams enfo waef --workers 8

# The same API works for additional deterministic issues and reference days.
.venv/bin/python scripts/data/acquire_weather.py ifs \
  --start 2025-01-15 --end 2025-01-16 --hours 0 12 --streams oper wave
.venv/bin/python scripts/data/acquire_weather.py era5 \
  --start 2025-01-15 --end 2025-01-31

# Revalidate the pilot and regenerate diagnostics.
.venv/bin/python scripts/data/inspect_weather.py
```

The extension command is documented, not launched. The downloader accepts other 2025 date ranges without changing naming or format; full-year retrieval still needs a date-by-date source inventory. Add January 2026 reference/update coverage when evaluating December 2025 departures. Code and schema are reviewed; no repository changes were pushed.

## Measured sizes and times

These are successful January measurements on Evangelis, not guaranteed service rates. MB and GB are decimal. Local reuse is conversion time, not network download time.

| NC file unit                             | Retained size |               Measured acquisition time |
| ---------------------------------------- | ------------: | --------------------------------------: |
| Deterministic wind, one lead             |  0.19–0.23 MB |                median 1.27 s from cloud |
| Deterministic waves, one lead            |  0.24–0.27 MB |                median 1.26 s from cloud |
| Ensemble wind, one lead, all 51 members  |  8.74–9.12 MB |                          median 13.21 s |
| Ensemble waves, one lead, all 51 members | 9.81–10.33 MB |                           median 9.89 s |
| ERA5 wind, 24 hourly timestamps          | about 7.59 MB | 10.77 s fresh; about 0.37 s local reuse |
| ERA5 waves, 24 hourly timestamps         | about 7.75 MB |  6.38 s fresh; about 0.53 s local reuse |

Successful forecast byte ranges totalled **3.27 GB transferred**; this excludes indexes, retries and the fresh ERA5 Zarr traffic, which was not instrumented. Retained size is much smaller because only the Atlantic crop is saved. Wave leads transferred about 127 MB globally but retained about 10 MB.

Extrapolating the sampled ensemble leads gives roughly **1.6 GB retained, 18 GB transferred and 33 minutes per complete 85-lead issue**, at the measured concurrency. Two daily issues over 365 days suggest around **1.2 TB retained and 13 TB transferred**, and about **17 days of sequential acquisition**. Compression and throughput vary with weather, lead and server load; these are planning estimates. Free disk was approximately **2.3 TiB** before the pilot. Do not reserve it all: existing research, working space and backup needs also count. Annual jobs have not been launched.

## Checks and first results

All **740 files** passed the final checksum/content audit. All **60 matched control-versus-deterministic field comparisons were exactly equal**, including missing-data patterns. Four complete deterministic issues have the expected 85 leads; every ensemble shard has all 51 members. The targeted code suite passed **14 tests**, including missing members, duplicate records, corrupted resume files, wrong raw GRIB dates, units, ERA5T cutoff and dateline coordinates. A separate scientific/code review approved the milestone.

Source wave directions sometimes reach about 360.004° because of packing. Validation allows only a 0.01° angular tolerance and preserves raw values; circular derived calculations must wrap them. Physical checks and direction overshoot counts are recorded in `physical_checks.csv`.

For the 15 January 00 UTC issue at +72 h, area-weighted ocean-cell RMSE against ERA5 was **2.16 m/s for control wind speed versus 1.71 m/s for mean member speed**, and **0.402 m versus 0.357 m for significant wave height**. Here “mean member speed” averages scalar speeds, not wind components. At the offshore example point, successive forecasts for 18 January change visibly and do not improve monotonically with each issue. These are descriptive examples from two issues, not evidence of annual skill or calibration. ERA5 is a reference, not observed truth; independent buoy verification is still pending.

The figures and CSVs are also copied into [the OneDrive output folder](<D:/OneDrive - IE University/WeatherRouting/paper_02_uncertainty/output/january_data_2025>): [ensemble fan charts](<D:/OneDrive - IE University/WeatherRouting/paper_02_uncertainty/output/january_data_2025/ensemble_fans.png>), [forecast revisions](<D:/OneDrive - IE University/WeatherRouting/paper_02_uncertainty/output/january_data_2025/forecast_revisions.png>), wind/wave maps, paired scores and file inventory. No routes were optimized.
