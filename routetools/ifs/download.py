#!/usr/bin/env python
"""Download and immediately crop archived deterministic ECMWF IFS runs.

Global GRIB files exist only as temporary files.  Each completed output is an
atomic NetCDF rename, so interrupted runs are safe to resume.  The output grid
and variable names match the BERS ERA5 corridor files.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import xarray as xr

from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

LOG = logging.getLogger("download_ecmwf_forecasts")         
FORECAST_STEPS = [*range(0, 145, 3), *range(150, 361, 6)]   
STREAMS = {
    "oper": {
        "params": ["10u", "10v"],
        "rename": {
            "u10": "10m_u_component_of_wind",
            "v10": "10m_v_component_of_wind",
        },
    },
    "wave": {
        "params": ["swh", "mwd", "mwp"],
        "rename": {
            "swh": "significant_height_of_combined_wind_waves_and_swell",
            "mwd": "mean_wave_direction",
            "mwp": "mean_wave_period",
        },
    },
}


def _parse_date(value: str) -> date:
    """Parse an ISO date for argparse."""
    return date.fromisoformat(value)


def _configure_logging(log_file: Path | None) -> None:
    """Configure console logging and an optional persistent log file."""
    handlers: list[logging.Handler] = [logging.StreamHandler()] 
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)      
        handlers.append(logging.FileHandler(log_file))          
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )                                                           


def _dates(start: date, end: date) -> list[date]:
    """Return every date in an inclusive interval."""
    return [start + timedelta(days=offset) for offset in range((end - start).days + 1)]


def _crop_corridor(ds: xr.Dataset, corridor: str) -> xr.Dataset:
    """Crop a cfgrib dataset to the exact BERS corridor grid."""
    lat_min, lat_max, lon_min, lon_max = CORRIDORS[corridor]
    lat_name = "latitude"
    lon_name = "longitude"
    lat_values = ds[lat_name].values
    lat_slice = (
        slice(lat_max, lat_min)
        if lat_values[0] > lat_values[-1]
        else slice(lat_min, lat_max)
    )
    ds = ds.sel({lat_name: lat_slice})

    lons = ds[lon_name].values
    uses_360 = float(np.nanmin(lons)) >= 0 and float(np.nanmax(lons)) > 180
    if lon_min < 0 and uses_360:
        west = ds.sel({lon_name: slice(lon_min % 360, 360.0)})
        east = ds.sel({lon_name: slice(0.0, lon_max)})
        ds = xr.concat([west, east], dim=lon_name)
        converted = ds[lon_name].values.copy()
        converted[converted >= 180] -= 360
        ds = ds.assign_coords({lon_name: converted})
    else:
        if lon_max > 180 and np.any(lons < 0):
            ds = ds.assign_coords({lon_name: np.mod(lons, 360)}).sortby(lon_name)
        ds = ds.sel({lon_name: slice(lon_min, lon_max)})
    return ds


def _open_grib(path: Path) -> xr.Dataset:
    """Open a deterministic IFS GRIB without creating a persistent index."""
    return xr.open_dataset(
        path,
        engine="cfgrib",
        backend_kwargs={"indexpath": ""},
    )


def _write_cropped(
    raw_paths: list[Path],
    netcdf_part: Path,
    *,
    stream: str,
    corridor: str,
    expected_steps: int,
) -> None:
    """Merge temporary single-parameter GRIBs into a corridor NetCDF."""
    datasets: list[xr.Dataset] = []
    try:
        for raw_path in raw_paths:
            parameter_ds = _open_grib(raw_path)
            if parameter_ds.sizes.get("step") != expected_steps:
                parameter_ds.close()
                raise ValueError(
                    "expected "
                    f"{expected_steps} forecast steps, "
                    f"got {parameter_ds.sizes.get('step')} in {raw_path.name}"
                )
            datasets.append(_crop_corridor(parameter_ds, corridor))
        ds = xr.merge(datasets, compat="no_conflicts")
        rename = {
            old: new
            for old, new in STREAMS[stream]["rename"].items()
            if old in ds.data_vars
        }
        missing = set(STREAMS[stream]["rename"]) - set(rename)
        if missing:
            raise ValueError(f"missing expected {stream} variables: {sorted(missing)}")
        ds = ds.rename(rename)
        ds.attrs.update(
            {
                "source": "ECMWF Open Data on Google Cloud",
                "model": "IFS",
                "resolution": "0.25 degree",
                "stream": stream,
                "data": corridor,
                "license": "CC BY 4.0; credit ECMWF",
            }
        )
        ds.to_netcdf(netcdf_part, format="NETCDF4")
    finally:
        for dataset in datasets:
            dataset.close()

def _fetch_param(
    client: object,
    run_date: date,
    hour: int,
    stream: str,
    steps: list[int],
    param: str,
    target: Path,
    token: str
) -> None:
    """Download a single GRIB parameter file."""
    if target.exists():                                          # Cheking if the file was already downloaded succesfully
        LOG.info("File already exists; skipping: %s", target)
        return

    raw_part = target.with_name(f".{target.name}.{token}.{param}.grib2.part")

    try:
        client.retrieve(  # type: ignore[attr-defined]
            date=run_date.strftime("%Y%m%d"),
            time=hour,
            stream=stream,
            type="fc",
            step=steps,
            param=param,
            target=str(raw_part),
        )
        raw_part.rename(str(target))
    except Exception:
        LOG.exception(f"Download {param} attempt failed")
        raise
    finally:
        raw_part.unlink(missing_ok=True)

    LOG.info("Saved %s (%.1f MB)", target, target.stat().st_size / 1e6)

def _download_run(
    *,
    client: object,
    run_date: date,
    hour: int,
    stream: str,
    output_dir: Path,
    steps: list[int],
    max_retries: int,
    retry_delay: float,
) -> list[Path]:
    """Download, crop, and atomically publish one IFS run and stream."""
    stream_dir  = output_dir / stream
    target_dirs = [
        stream_dir / param
        for param in STREAMS[stream]["params"]
        ]
    
    targets = []
    
    for t_dir in target_dirs:
        t_dir.mkdir(parents = True , exist_ok = True)
        targets.append(t_dir / f"{run_date:%Y%m%d}_{hour:02d}z.grib2")


    for attempt in range(1, max_retries + 1):
        token = f"{os.getpid()}.{attempt}.{threading.get_ident()}"

        try:
            LOG.info(
                "Downloading %s %s %02dZ (%d steps), attempt %d/%d",
                stream,
                run_date,
                hour,
                len(steps),
                attempt,
                max_retries,
            )
            # Google Cloud Storage does not accept the multi-range HTTP request
            # generated when multiple parameters from one GRIB are selected at
            # once.  One parameter per request produces a single byte range and
            # works consistently; the cropped fields are merged below.
            # Download all parameters in parallel using a ThreadPoolExecutor
            max_param_workers = len(STREAMS[stream]["params"])
            with ThreadPoolExecutor(max_workers=max_param_workers) as executor:
                future_to_param = {
                    executor.submit(
                        _fetch_param,
                        client=client,
                        run_date=run_date,
                        hour=hour,
                        stream=stream,
                        steps=steps,
                        param=param,
                        target=target,
                        token=token,
                    ): param
                    for param, target in zip(STREAMS[stream]["params"], targets, strict=True)
                }
                for future in as_completed(future_to_param):
                    future.result()  # Raises exception if any parameter download failed

            LOG.info("All parameters downloaded successfully for %s %02dZ", run_date, hour)

            return targets
        except Exception:
            LOG.exception("Forecast download attempt failed")
            if attempt == max_retries:
                raise
            time.sleep(retry_delay * attempt)

    raise RuntimeError("unreachable")

def _iter_dates(
    start_date: date,
    end_date: date,
    *,
    client: object,
    hour: int,
    stream: str,
    output_dir: Path,
    steps: list[int],
    max_retries: int,
    retry_delay: float,
) -> None:
    for run_date in _dates(start_date , end_date):
        LOG.info("Processing date: %s for stream: %s", run_date, stream)
        try:
            _download_run(
                client=client,
                run_date=run_date,
                hour=hour,
                stream=stream,
                output_dir=output_dir,
                steps=steps,
                max_retries=max_retries,
                retry_delay=retry_delay,
            )
            LOG.info('%s: date %s compleated successfully',stream, run_date)
        except Exception as exc:
            raise Exception('At run date %s: %s', run_date, exc)
        


def main() -> None:
    """Download deterministic IFS forecasts for an inclusive run-date range."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=_parse_date, required=True)
    parser.add_argument("--end", type=_parse_date, required=True)
    parser.add_argument("--hour", type=int, choices=(0, 12), default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--streams",
        nargs="+",
        choices=tuple(STREAMS),
        default=list(STREAMS),
    )
    parser.add_argument("--steps", nargs="+", type=int, default=FORECAST_STEPS)
    parser.add_argument("--log-file", type=Path)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-delay", type=float, default=20.0)
    args = parser.parse_args()

    if args.end < args.start:
        parser.error("--end must not precede --start")
    if sorted(set(args.steps)) != args.steps:
        parser.error("--steps must be unique and ascending")
    _configure_logging(args.log_file)

    from ecmwf.opendata import Client

    client = Client(source="google", model="ifs", resol="0p25")
    # Process dates sequentially, but download multiple streams in parallel
    LOG.info("Processing streams: %s", args.streams)
    with ThreadPoolExecutor(max_workers=len(args.streams)) as executor:
        future_to_stream = {
            executor.submit(
                _iter_dates,
                start_date=args.start,
                end_date=args.end,
                client=client,
                hour=args.hour,
                stream=stream,
                output_dir=args.output_dir,
                steps=args.steps,
                max_retries=args.max_retries,
                retry_delay=args.retry_delay,
            ): stream
            for stream in args.streams
        }
        
        for future in as_completed(future_to_stream):
            stream = future_to_stream[future]
            try:
                future.result()
                LOG.info("Stream '%s' completed successfully", stream)
            except Exception as exc:
                LOG.error("Stream '%s' failed: %s", stream, exc)
                raise


if __name__ == "__main__":
    main()
