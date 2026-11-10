#!/usr/bin/env python
"""Download an exact ERA5 time window for a route corridor.

The regular ERA5 downloader works in whole months or years.  This entry point
is for small forecast-versus-reanalysis experiments where the truth data must
cover an exact inclusive date window.  It preserves the BERS grid, coordinate
order, variable names, dtype, and uncompressed NetCDF4 layout.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path

from routetools.era5.download_gcs import (
    WAVE_DIR_VAR,
    WAVE_HS_VAR,
    WAVE_PERIOD_VAR,
    WIND_U_VAR,
    WIND_V_VAR,
    _normalize_time_dim,
    _open_era5_zarr,
    _save_with_low_memory,
    _select_corridor,
)

LOG = logging.getLogger("download_era5_window")
FIELDS = {
    "wind": [WIND_U_VAR, WIND_V_VAR],
    "waves": [WAVE_HS_VAR, WAVE_DIR_VAR, WAVE_PERIOD_VAR],
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


def _months_between(start: date, end: date) -> list[int]:
    """Return the ordered months spanned by a same-year interval."""
    if start.year != end.year:
        raise ValueError("ERA5 window currently must stay within one calendar year")
    return list(range(start.month, end.month + 1))


def _output_path(
    output_dir: Path,
    field: str,
    corridor: str,
    start: date,
    end: date,
) -> Path:
    """Return the deterministic output filename for a time window."""
    return output_dir / (
        f"era5_{field}_{corridor}_{start:%Y%m%d}_{end:%Y%m%d}.nc"
    )


def _download_field(
    *,
    output_dir: Path,
    field: str,
    corridor: str,
    start: date,
    end: date,
    max_retries: int,
    retry_delay: float,
) -> Path:
    """Download one field group, retrying without exposing partial outputs."""
    target = _output_path(output_dir, field, corridor, start, end)
    if target.exists():
        LOG.info("File already exists; skipping: %s", target)
        return target

    variables = FIELDS[field]
    months = _months_between(start, end)
    end_timestamp = datetime.combine(end + timedelta(days=1), datetime.min.time())
    end_timestamp -= timedelta(hours=1)

    for attempt in range(1, max_retries + 1):
        part = target.with_name(f".{target.name}.{os.getpid()}.{attempt}.part")
        try:
            LOG.info(
                "Opening ERA5 %s for %s, %s through %s (attempt %d/%d)",
                field,
                corridor,
                start,
                end,
                attempt,
                max_retries,
            )
            ds = _open_era5_zarr(variables)
            try:
                ds = _select_corridor(
                    ds,
                    corridor,
                    year=str(start.year),
                    months=months,
                    time_step=1,
                )
                ds = _normalize_time_dim(ds)
                ds = ds.sel(valid_time=slice(str(start), end_timestamp.isoformat()))
                ds.attrs.update(
                    {
                        "experiment": "forecast-aware routing toy experiment",
                        "window_start": str(start),
                        "window_end": str(end),
                        "corridor": corridor,
                    }
                )
                LOG.info("Writing temporary file: %s", part)
                _save_with_low_memory(ds, part)
            finally:
                ds.close()
            part.replace(target)
            LOG.info("Saved %s (%.1f MB)", target, target.stat().st_size / 1e6)
            return target
        except Exception:
            LOG.exception("ERA5 %s download attempt failed", field)
            part.unlink(missing_ok=True)
            if attempt == max_retries:
                raise
            time.sleep(retry_delay * attempt)

    raise RuntimeError("unreachable")


def main() -> None:
    """Download wind and wave truth data for an exact date window."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=_parse_date, required=True)
    parser.add_argument("--end", type=_parse_date, required=True)
    parser.add_argument("--corridor", choices=("atlantic", "pacific"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--fields",
        nargs="+",
        choices=tuple(FIELDS),
        default=list(FIELDS),
    )
    parser.add_argument("--log-file", type=Path)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-delay", type=float, default=15.0)
    args = parser.parse_args()

    if args.end < args.start:
        parser.error("--end must not precede --start")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _configure_logging(args.log_file)

    for field in args.fields:
        _download_field(
            output_dir=args.output_dir,
            field=field,
            corridor=args.corridor,
            start=args.start,
            end=args.end,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
        )


if __name__ == "__main__":
    main()
