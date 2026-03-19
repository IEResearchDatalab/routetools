#!/usr/bin/env python
r"""Run SWOPP3 cases — CLI entry-point.

Usage
-----
Default pipeline for the full 2024 SWOPP3 dataset:

    uv run scripts/download_era5.py
    uv run scripts/swopp3_run.py

The defaults in this script expect the four NetCDF files written by
``scripts/download_era5.py`` for year 2024. If any required file is missing,
the command fails before execution and explains exactly which datasets are
missing and why they are required.

Run all 8 cases with explicit per-corridor paths:

    uv run scripts/swopp3_run.py \
        --wind-path-atlantic "data/era5/era5_wind_atlantic_2024.nc" \
        --wave-path-atlantic "data/era5/era5_waves_atlantic_2024.nc" \
        --wind-path-pacific  "data/era5/era5_wind_pacific_2024.nc" \
        --wave-path-pacific  "data/era5/era5_waves_pacific_2024.nc" \
        --output-dir "output" \
        --n-points 500  --verbosity 2

Run only Atlantic cases after downloading Atlantic data:

    uv run python scripts/swopp3_run.py \
    --cases AO_WPS --cases AO_noWPS --cases AGC_WPS --cases AGC_noWPS \
    --wind-path-atlantic "data/era5/era5_wind_atlantic_2024.nc" \
    --wave-path-atlantic "data/era5/era5_waves_atlantic_2024.nc" \
    --output-dir "output" \
    --n-points 355  --verbosity 2

Run only Pacific cases after downloading Pacific data:

    uv run python scripts/swopp3_run.py \
    --cases PO_WPS --cases PO_noWPS --cases PGC_WPS --cases PGC_noWPS \
    --wind-path-pacific "data/era5/era5_wind_pacific_2024.nc" \
    --wave-path-pacific "data/era5/era5_waves_pacific_2024.nc" \
    --output-dir "output" \
    --n-points 584 --verbosity 2

Run only the first 3 departures (quick test):

    uv run scripts/swopp3_run.py --max-departures 3 --output-dir output/swopp3
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import typer

if TYPE_CHECKING:
    from routetools.swopp3_runner import FieldClosure

app = typer.Typer(
    help=(
        "SWOPP3 competition runner. Expects ERA5 files produced by "
        "scripts/download_era5.py unless explicit paths are provided."
    )
)

_ERA5_FILE_RE = re.compile(
    r"^(?P<prefix>era5_[^_]+_[^_]+_)(?P<year>\d{4})(?:_(?P<suffix>\d{2}(?:-\d{2})?))?\.nc$"
)


def _resolve_cli_verbosity(verbosity: int) -> int:
    """Validate the CLI verbosity value."""
    if verbosity not in (0, 1, 2):
        raise ValueError("verbosity must be one of 0, 1, or 2")
    return verbosity


def _selected_corridors(case_ids: list[str]) -> list[str]:
    """Return the sorted set of route corridors required by ``case_ids``."""
    from routetools.swopp3 import SWOPP3_CASES

    return sorted({str(SWOPP3_CASES[cid]["route"]) for cid in case_ids})


def _loadable_era5_paths(path: Path) -> list[Path]:
    """Return the base ERA5 file plus any next-year continuation files."""
    match = _ERA5_FILE_RE.match(path.name)
    if match is None:
        return [path]

    prefix = match.group("prefix")
    next_year = int(match.group("year")) + 1
    exact_next_year = path.with_name(f"{prefix}{next_year}.nc")
    if exact_next_year.exists():
        return [path, exact_next_year]

    continuation_paths = sorted(path.parent.glob(f"{prefix}{next_year}_*.nc"))
    return [path, *continuation_paths]


def _validate_required_data_paths(
    case_ids: list[str],
    corridor_wind: dict[str, Path],
    corridor_wave: dict[str, Path],
) -> None:
    """Fail fast when the ERA5 inputs required by the selected cases are missing."""
    required_corridors = _selected_corridors(case_ids)
    missing: list[str] = []

    for corridor in required_corridors:
        wind = corridor_wind.get(corridor)
        wave = corridor_wave.get(corridor)

        if wind is None:
            missing.append(f"{corridor} wind dataset path is not configured")
        elif not Path(wind).exists():
            missing.append(f"{corridor} wind dataset not found: {wind}")

        if wave is None:
            missing.append(f"{corridor} wave dataset path is not configured")
        elif not Path(wave).exists():
            missing.append(f"{corridor} wave dataset not found: {wave}")

    if not missing:
        return

    corridor_list = ", ".join(required_corridors)
    missing_lines = "\n".join(f"- {item}" for item in missing)
    raise FileNotFoundError(
        "SWOPP3 input validation failed.\n\n"
        f"Selected cases require ERA5 datasets for corridor(s): {corridor_list}.\n"
        "This CLI requires weather data for every selected case:\n"
        "- GC cases use wind and wave data during SWOPP3 energy evaluation.\n"
        "- Optimised cases use wind data to build the CMA-ES vectorfield "
        "and use wind/wave data during energy evaluation.\n"
        "- Missing ERA5 files are a hard error; there is no fallback "
        "to GC or no-weather mode.\n\n"
        f"Missing inputs:\n{missing_lines}\n\n"
        "Fix:\n"
        "- Run `uv run scripts/download_era5.py` to download the default "
        "2024 Atlantic and Pacific datasets, then rerun "
        "`uv run scripts/swopp3_run.py`.\n"
        "- If you downloaded a different year or only one corridor, "
        "pass matching `--wind-path*` and `--wave-path*` options."
    )


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
        help=(
            "Path to ERA5 wind NetCDF used for all selected corridors. "
            "Overrides the built-in corridor defaults when provided."
        ),
    ),
    wave_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--wave-path",
        help=(
            "Path to ERA5 wave NetCDF used for all selected corridors. "
            "Overrides the built-in corridor defaults when provided."
        ),
    ),
    wind_path_atlantic: Path | None = typer.Option(  # noqa: B008
        "data/era5/era5_wind_atlantic_2024.nc",
        "--wind-path-atlantic",
        help=(
            "Path to ERA5 wind NetCDF for Atlantic corridor. Defaults to the "
            "file written by scripts/download_era5.py for year 2024."
        ),
    ),
    wave_path_atlantic: Path | None = typer.Option(  # noqa: B008
        "data/era5/era5_waves_atlantic_2024.nc",
        "--wave-path-atlantic",
        help=(
            "Path to ERA5 wave NetCDF for Atlantic corridor. Defaults to the "
            "file written by scripts/download_era5.py for year 2024."
        ),
    ),
    wind_path_pacific: Path | None = typer.Option(  # noqa: B008
        "data/era5/era5_wind_pacific_2024.nc",
        "--wind-path-pacific",
        help=(
            "Path to ERA5 wind NetCDF for Pacific corridor. Defaults to the "
            "file written by scripts/download_era5.py for year 2024."
        ),
    ),
    wave_path_pacific: Path | None = typer.Option(  # noqa: B008
        "data/era5/era5_waves_pacific_2024.nc",
        "--wave-path-pacific",
        help=(
            "Path to ERA5 wave NetCDF for Pacific corridor. Defaults to the "
            "file written by scripts/download_era5.py for year 2024."
        ),
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
    verbosity: int = typer.Option(  # noqa: B008
        1,
        "--verbosity",
        "-v",
        help="Print level: 0 = none, 1 = runner only, 2 = runner + CMA-ES.",
    ),
    control_points: int | None = typer.Option(  # noqa: B008
        None,
        "--control-points",
        "-K",
        help="Number of Bézier control points (K). Overrides runner default (10).",
    ),
    weather_penalty_type: str | None = typer.Option(  # noqa: B008
        None,
        "--weather-penalty-type",
        help="Weather penalty: 'smooth' (squared-ReLU, default) or 'hard' (step).",
    ),
) -> None:
    """Run SWOPP3 competition cases.

    The default invocation expects the 2024 ERA5 files produced by
    ``scripts/download_era5.py``. The command validates those inputs before
    loading any corridor and exits with a precise message if a required file
    is missing.
    """
    import xarray as xr

    from routetools.era5.loader import (
        load_dataset_epoch,
        load_era5_wavefield,
        load_era5_windfield,
        load_natural_earth_land_mask,
    )
    from routetools.swopp3 import SWOPP3_CASES, departures_2024
    from routetools.swopp3_runner import run_case

    try:
        verbosity = _resolve_cli_verbosity(verbosity)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    show_progress = verbosity >= 1

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

    if show_progress:
        typer.echo(f"Running {len(case_ids)} case(s) × {len(departures)} departure(s)")

    # ---- Build per-corridor field map ----
    corridor_wind: dict[str, Path] = {}
    corridor_wave: dict[str, Path] = {}

    if wind_path_atlantic is not None:
        corridor_wind["atlantic"] = wind_path_atlantic
    if wave_path_atlantic is not None:
        corridor_wave["atlantic"] = wave_path_atlantic
    if wind_path_pacific is not None:
        corridor_wind["pacific"] = wind_path_pacific
    if wave_path_pacific is not None:
        corridor_wave["pacific"] = wave_path_pacific

    # Shared paths intentionally override the corridor defaults so the
    # simplest one-flag workflow still works for single-corridor runs.
    if wind_path is not None:
        corridor_wind["atlantic"] = wind_path
        corridor_wind["pacific"] = wind_path
    if wave_path is not None:
        corridor_wave["atlantic"] = wave_path
        corridor_wave["pacific"] = wave_path

    try:
        _validate_required_data_paths(case_ids, corridor_wind, corridor_wave)
    except FileNotFoundError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc

    # ---- Load fields per corridor (cache so we load each file once) ----

    _loaded_wind: dict[str, tuple[FieldClosure, datetime]] = {}
    _loaded_wave: dict[str, tuple[FieldClosure, datetime]] = {}
    _loaded_vf: dict[str, FieldClosure] = {}  # corridor -> vectorfield
    _loaded_land: dict[str, object] = {}  # corridor -> Land

    def _get_wind(corridor: str) -> tuple[FieldClosure, datetime]:
        """Return (windfield_closure, dataset_epoch) for corridor."""
        if corridor in _loaded_wind:
            return _loaded_wind[corridor]
        wp = corridor_wind.get(corridor)
        if wp is None:
            raise ValueError(f"No wind path available for corridor '{corridor}'")
        load_paths = _loadable_era5_paths(wp)
        load_target = load_paths if len(load_paths) > 1 else load_paths[0]
        if show_progress:
            typer.echo(
                f"Loading wind field for {corridor} from "
                f"{', '.join(str(path) for path in load_paths)} …"
            )
        epoch = load_dataset_epoch(load_target)
        wf = load_era5_windfield(load_target)
        _loaded_wind[corridor] = (wf, epoch)
        return wf, epoch

    def _get_vectorfield(corridor: str) -> FieldClosure:
        """Return vectorfield closure for corridor.

        Reuses the windfield closure since both load identical ERA5 10-m
        wind data — avoids duplicating ~4 GB of GPU memory per corridor.
        """
        if corridor in _loaded_vf:
            return _loaded_vf[corridor]
        wf, _ = _get_wind(corridor)
        _loaded_vf[corridor] = wf
        return wf

    def _get_wave(corridor: str) -> tuple[FieldClosure, datetime]:
        """Return (wavefield_closure, dataset_epoch) for corridor."""
        if corridor in _loaded_wave:
            return _loaded_wave[corridor]
        wp = corridor_wave.get(corridor)
        if wp is None:
            raise ValueError(f"No wave path available for corridor '{corridor}'")
        load_paths = _loadable_era5_paths(wp)
        load_target = load_paths if len(load_paths) > 1 else load_paths[0]
        if show_progress:
            typer.echo(
                f"Loading wave field for {corridor} from "
                f"{', '.join(str(path) for path in load_paths)} …"
            )
        epoch = load_dataset_epoch(load_target)
        wvf = load_era5_wavefield(load_target)
        _loaded_wave[corridor] = (wvf, epoch)
        return wvf, epoch

    def _get_land(corridor: str):
        """Return Land mask for corridor (Natural Earth shapefiles)."""
        if corridor in _loaded_land:
            return _loaded_land[corridor]
        # Determine corridor extent from the ERA5 wave or wind file
        wp = corridor_wave.get(corridor) or corridor_wind.get(corridor)
        if wp is None:
            raise ValueError(f"No wind/wave path available for corridor '{corridor}'")
        wp = _loadable_era5_paths(wp)[0]

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
        if show_progress:
            typer.echo(
                f"Building Natural Earth land mask for {corridor} "
                f"lon={lon_range}, lat={lat_range} …"
            )
        land = load_natural_earth_land_mask(lon_range, lat_range)
        _loaded_land[corridor] = land
        return land

    # ---- Run ----
    prev_corridor: str | None = None
    for cid in case_ids:
        case = SWOPP3_CASES[cid]
        corridor = case["route"]  # "atlantic" or "pacific"

        # Free previous corridor data when switching to a new one so that
        # both corridors' arrays don't coexist on the GPU.
        if prev_corridor is not None and corridor != prev_corridor:
            _loaded_wind.pop(prev_corridor, None)
            _loaded_wave.pop(prev_corridor, None)
            _loaded_vf.pop(prev_corridor, None)
            _loaded_land.pop(prev_corridor, None)
            import gc

            gc.collect()
            import jax

            jax.clear_caches()
            if show_progress:
                typer.echo(
                    "[info] Freed "
                    f"{prev_corridor} corridor data before loading {corridor}"
                )
        prev_corridor = corridor

        if show_progress:
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
        dataset_epoch = wind_epoch

        # Build extra CMA-ES keyword arguments from CLI flags.
        cmaes_extra: dict[str, object] = {}
        if control_points is not None:
            cmaes_extra["K"] = control_points
        if weather_penalty_type is not None:
            cmaes_extra["weather_penalty_type"] = weather_penalty_type

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
            verbosity=verbosity,
            dataset_epoch=dataset_epoch,
            **cmaes_extra,
        )

        if show_progress:
            energies = [r.energy_mwh for r in results]
            total_time = sum(r.comp_time_s for r in results)
            typer.echo(
                f"  {len(results)} departures  "
                f"mean E={sum(energies) / len(energies):.2f} MWh  "
                f"total comp time={total_time:.1f}s"
            )

    if show_progress:
        typer.echo(f"\nOutputs written to {output_dir}")


if __name__ == "__main__":
    app()
