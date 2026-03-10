#!/usr/bin/env python
r"""Run SWOPP3 cases — CLI entry-point.

Usage
-----
Run all 8 cases (GC only, no ERA5 data required):

    python scripts/swopp3_run.py --strategy gc --output-dir output/swopp3

Run Atlantic cases with ERA5 data:

    python scripts/swopp3_run.py \\
        --cases AGC_WPS AGC_noWPS \\
        --wind-path data/era5/era5_wind_atlantic_2024.nc \\
        --wave-path data/era5/era5_waves_atlantic_2024.nc \\
        --output-dir output/swopp3

Run all 8 cases with per-corridor ERA5 data:

    python scripts/swopp3_run.py \\
        --wind-path-atlantic data/era5/era5_wind_atlantic_2024.nc \\
        --wave-path-atlantic data/era5/era5_waves_atlantic_2024.nc \\
        --wind-path-pacific  data/era5/era5_wind_pacific_2024.nc  \\
        --wave-path-pacific  data/era5/era5_waves_pacific_2024.nc  \\
        --output-dir output/swopp3

Run only the first 3 departures (quick test):

    python scripts/swopp3_run.py --max-departures 3 --output-dir output/swopp3
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import typer

app = typer.Typer(help="SWOPP3 competition runner.")


@app.command()
def main(
    cases: list[str] | None = typer.Option(  # noqa: B008
        None,
        "--cases",
        "-c",
        help="Case IDs to run (e.g. AGC_WPS PO_noWPS).  Default: all 8.",
    ),
    strategy: str | None = typer.Option(  # noqa: B008
        None,
        "--strategy",
        "-s",
        help="Filter by strategy: 'gc' or 'optimised'.  Default: both.",
    ),
    wind_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--wind-path",
        help="Path to ERA5 wind NetCDF (single corridor, used for all cases).",
    ),
    wave_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--wave-path",
        help="Path to ERA5 wave NetCDF (single corridor, used for all cases).",
    ),
    wind_path_atlantic: Path | None = typer.Option(  # noqa: B008
        None,
        "--wind-path-atlantic",
        help="Path to ERA5 wind NetCDF for Atlantic corridor.",
    ),
    wave_path_atlantic: Path | None = typer.Option(  # noqa: B008
        None,
        "--wave-path-atlantic",
        help="Path to ERA5 wave NetCDF for Atlantic corridor.",
    ),
    wind_path_pacific: Path | None = typer.Option(  # noqa: B008
        None,
        "--wind-path-pacific",
        help="Path to ERA5 wind NetCDF for Pacific corridor.",
    ),
    wave_path_pacific: Path | None = typer.Option(  # noqa: B008
        None,
        "--wave-path-pacific",
        help="Path to ERA5 wave NetCDF for Pacific corridor.",
    ),
    output_dir: Path = typer.Option(  # noqa: B008
        "output/swopp3",
        "--output-dir",
        "-o",
        help="Output directory for CSV files.",
    ),
    submission: int = typer.Option(  # noqa: B008
        1,
        "--submission",
        help="Submission number for file naming.",
    ),
    n_points: int = typer.Option(  # noqa: B008
        100,
        "--n-points",
        help="Number of route waypoints.",
    ),
    max_departures: int | None = typer.Option(  # noqa: B008
        None,
        "--max-departures",
        "-n",
        help="Limit number of departures (for quick testing).",
    ),
    quiet: bool = typer.Option(  # noqa: B008
        False,
        "--quiet",
        "-q",
        help="Suppress progress output.",
    ),
) -> None:
    """Run SWOPP3 competition cases."""
    import numpy as np

    from routetools.swopp3 import SWOPP3_CASES, departures_2024
    from routetools.swopp3_runner import run_case

    # ---- Select cases ----
    if cases is not None:
        case_ids = cases
        for cid in case_ids:
            if cid not in SWOPP3_CASES:
                typer.echo(f"Unknown case: {cid}", err=True)
                raise typer.Exit(1)
    else:
        case_ids = list(SWOPP3_CASES.keys())

    if strategy is not None:
        case_ids = [
            cid for cid in case_ids if SWOPP3_CASES[cid]["strategy"] == strategy
        ]
        if not case_ids:
            typer.echo(f"No cases match strategy '{strategy}'", err=True)
            raise typer.Exit(1)

    # ---- Departures ----
    departures = departures_2024()
    if max_departures is not None:
        departures = departures[:max_departures]

    typer.echo(f"Running {len(case_ids)} case(s) × {len(departures)} departure(s)")

    # ---- Build per-corridor field map ----
    # Resolve which wind/wave paths to use for each corridor.
    corridor_wind = {}
    corridor_wave = {}

    if wind_path_atlantic is not None:
        corridor_wind["atlantic"] = wind_path_atlantic
    if wave_path_atlantic is not None:
        corridor_wave["atlantic"] = wave_path_atlantic
    if wind_path_pacific is not None:
        corridor_wind["pacific"] = wind_path_pacific
    if wave_path_pacific is not None:
        corridor_wave["pacific"] = wave_path_pacific

    # --wind-path / --wave-path act as fallback for all corridors
    if wind_path is not None:
        corridor_wind.setdefault("atlantic", wind_path)
        corridor_wind.setdefault("pacific", wind_path)
    if wave_path is not None:
        corridor_wave.setdefault("atlantic", wave_path)
        corridor_wave.setdefault("pacific", wave_path)

    # ---- Load fields per corridor (cache so we load each file once) ----
    from routetools.era5.loader import (
        load_era5_vectorfield,
        load_era5_wavefield,
        load_era5_windfield,
        load_natural_earth_land_mask,
    )

    _loaded_wind: dict[str, tuple] = {}  # corridor -> (windfield, epoch)
    _loaded_wave: dict[str, tuple] = {}  # corridor -> (wavefield, epoch)
    _loaded_vf: dict[str, object] = {}  # corridor -> vectorfield
    _loaded_land: dict[str, object] = {}  # corridor -> Land

    def _dataset_epoch(path: Path) -> datetime:
        """Extract the first timestamp from a NetCDF dataset."""
        import xarray as xr

        with xr.open_dataset(path) as ds:
            # Handle both 'time' and 'valid_time' coordinate names
            for tname in ("time", "valid_time"):
                if tname in ds.coords:
                    epoch_np = ds[tname].values[0]
                    break
            else:
                raise KeyError(f"No time coordinate found in {path}")
        # Convert numpy datetime64 -> Python datetime (UTC)
        ts = (epoch_np - np.datetime64("1970-01-01T00:00:00")) / np.timedelta64(1, "s")
        return datetime.fromtimestamp(float(ts), tz=UTC).replace(tzinfo=None)

    def _get_wind(corridor: str):
        """Return (windfield_closure, dataset_epoch) for corridor, or (None, None)."""
        if corridor in _loaded_wind:
            return _loaded_wind[corridor]
        wp = corridor_wind.get(corridor)
        if wp is None:
            _loaded_wind[corridor] = (None, None)
            return None, None
        typer.echo(f"Loading wind field for {corridor} from {wp} …")
        epoch = _dataset_epoch(wp)
        wf = load_era5_windfield(wp)
        _loaded_wind[corridor] = (wf, epoch)
        return wf, epoch

    def _get_vectorfield(corridor: str):
        """Return vectorfield closure for corridor, or None."""
        if corridor in _loaded_vf:
            return _loaded_vf[corridor]
        wp = corridor_wind.get(corridor)
        if wp is None:
            _loaded_vf[corridor] = None
            return None
        # Reuse the wind-load epoch but build a vectorfield
        typer.echo(f"Loading vectorfield for {corridor} from {wp} …")
        vf = load_era5_vectorfield(wp)
        _loaded_vf[corridor] = vf
        return vf

    def _get_wave(corridor: str):
        """Return (wavefield_closure, dataset_epoch) for corridor, or (None, None)."""
        if corridor in _loaded_wave:
            return _loaded_wave[corridor]
        wp = corridor_wave.get(corridor)
        if wp is None:
            _loaded_wave[corridor] = (None, None)
            return None, None
        typer.echo(f"Loading wave field for {corridor} from {wp} …")
        epoch = _dataset_epoch(wp)
        wvf = load_era5_wavefield(wp)
        _loaded_wave[corridor] = (wvf, epoch)
        return wvf, epoch

    def _get_land(corridor: str):
        """Return Land mask for corridor (Natural Earth shapefiles)."""
        if corridor in _loaded_land:
            return _loaded_land[corridor]
        # Determine corridor extent from the ERA5 wave or wind file
        wp = corridor_wave.get(corridor) or corridor_wind.get(corridor)
        if wp is None:
            _loaded_land[corridor] = None
            return None
        import xarray as xr

        with xr.open_dataset(wp) as ds:
            for cname in ("longitude", "lon"):
                if cname in ds.coords:
                    lons = ds[cname].values
                    break
            else:
                raise KeyError(f"No longitude coordinate found in {wp}")
            for cname in ("latitude", "lat"):
                if cname in ds.coords:
                    lats = ds[cname].values
                    break
            else:
                raise KeyError(f"No latitude coordinate found in {wp}")
        lon_range = (float(lons.min()), float(lons.max()))
        lat_range = (float(lats.min()), float(lats.max()))
        typer.echo(
            f"Building Natural Earth land mask for {corridor} "
            f"lon={lon_range}, lat={lat_range} …"
        )
        land = load_natural_earth_land_mask(lon_range, lat_range)
        _loaded_land[corridor] = land
        return land

    # ---- Run ----
    for cid in case_ids:
        case = SWOPP3_CASES[cid]
        corridor = case["route"]  # "atlantic" or "pacific"
        typer.echo(f"\n{'=' * 60}")
        typer.echo(f"Case {cid}: {case['label']}")
        typer.echo(
            f"  strategy={case['strategy']}  wps={case['wps']}  route={corridor}"
        )
        typer.echo(f"{'=' * 60}")

        windfield, wind_epoch = _get_wind(corridor)
        wavefield, wave_epoch = _get_wave(corridor)
        vectorfield = _get_vectorfield(corridor)
        land = _get_land(corridor)

        # Use the wind field epoch as canonical dataset epoch (both fields
        # share the same 2024-01-01 epoch from the ERA5 download).
        dataset_epoch = wind_epoch or wave_epoch

        results = run_case(
            cid,
            departures,
            vectorfield=vectorfield,
            windfield=windfield,
            wavefield=wavefield,
            land=land,
            output_dir=output_dir,
            submission=submission,
            n_points=n_points,
            verbose=not quiet,
            dataset_epoch=dataset_epoch,
        )

        # Summary
        energies = [r.energy_mwh for r in results]
        total_time = sum(r.comp_time_s for r in results)
        typer.echo(
            f"  {len(results)} departures  "
            f"mean E={sum(energies) / len(energies):.2f} MWh  "
            f"total comp time={total_time:.1f}s"
        )

    typer.echo(f"\nOutputs written to {output_dir}")


if __name__ == "__main__":
    app()
