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
        --wind-path-atlantic data/era5/era5_wind_atlantic_2024.nc \
        --wave-path-atlantic data/era5/era5_waves_atlantic_2024.nc \
        --wind-path-pacific  data/era5/era5_wind_pacific_2024.nc  \
        --wave-path-pacific  data/era5/era5_waves_pacific_2024.nc  \
        --output-dir output/swopp3

Run only Atlantic cases after downloading Atlantic data:

    uv run scripts/swopp3_run.py \
        --cases AGC_WPS AGC_noWPS \
        --wind-path data/era5/era5_wind_atlantic_2024.nc \
        --wave-path data/era5/era5_waves_atlantic_2024.nc \
        --output-dir output/swopp3

Run only the first 3 departures (quick test):

    uv run scripts/swopp3_run.py --max-departures 3 --output-dir output/swopp3
"""

from __future__ import annotations

import contextlib
import json
import re
import sys
import tomllib
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
_CONFIG_PATH_KEYS = {
    "output_dir",
    "wind_path",
    "wave_path",
    "wind_path_atlantic",
    "wave_path_atlantic",
    "wind_path_pacific",
    "wave_path_pacific",
}


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


def _resolve_case_ids(
    cases: list[str] | None,
    strategy: str | None,
) -> list[str]:
    """Return the case IDs selected by the given filters."""
    from routetools.swopp3 import SWOPP3_CASES

    if cases is not None:
        case_ids = cases
        for cid in case_ids:
            if cid not in SWOPP3_CASES:
                raise ValueError(f"Unknown case: {cid}")
    else:
        case_ids = list(SWOPP3_CASES.keys())

    if strategy is not None:
        case_ids = [
            cid for cid in case_ids if SWOPP3_CASES[cid]["strategy"] == strategy
        ]
        if not case_ids:
            raise ValueError(f"No cases match strategy '{strategy}'")

    return case_ids


def _resolve_config_value_path(config_path: Path, value: str | Path) -> Path:
    """Resolve a path-like config value relative to the config file."""
    path = Path(value)
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


def _resolve_profile_paths(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Resolve path-like fields in a profile or run dictionary."""
    resolved = dict(config)
    for key in _CONFIG_PATH_KEYS:
        if key in resolved and resolved[key] is not None:
            resolved[key] = _resolve_config_value_path(config_path, resolved[key])
    return resolved


def _load_experiment_profile(config_path: Path, experiment: str) -> dict[str, Any]:
    """Load and resolve one SWOPP3 experiment profile from TOML."""
    if not config_path.exists():
        raise FileNotFoundError(f"Experiment config not found: {config_path}")

    with config_path.open("rb") as handle:
        config = tomllib.load(handle)

    experiments = config.get("swopp3", {}).get("experiments", {})
    if experiment not in experiments:
        available = ", ".join(sorted(experiments)) or "(none)"
        raise KeyError(
            f"Unknown SWOPP3 experiment '{experiment}'. Available profiles: {available}"
        )

    raw_profile = experiments[experiment]
    defaults = _resolve_profile_paths(
        config_path,
        dict(raw_profile.get("defaults", {})),
    )
    raw_runs = raw_profile.get("runs", [])
    if not raw_runs:
        raise ValueError(f"Experiment '{experiment}' does not define any runs")

    resolved_runs: list[dict[str, Any]] = []
    for index, raw_run in enumerate(raw_runs, start=1):
        run = _resolve_profile_paths(config_path, {**defaults, **raw_run})
        cases = run.get("cases")
        if isinstance(cases, str):
            run["cases"] = [cases]
        elif cases is not None:
            run["cases"] = list(cases)

        run.setdefault("name", f"run_{index}")
        resolved_runs.append(run)

    resolved_output_dir = raw_profile.get("output_dir", f"output/{experiment}")
    return {
        "name": experiment,
        "description": raw_profile.get("description", ""),
        "source_script": raw_profile.get("source_script", ""),
        "output_dir": _resolve_config_value_path(config_path, resolved_output_dir),
        "runs": resolved_runs,
        "raw_profile": raw_profile,
    }


def _write_experiment_manifest(
    *,
    config_path: Path,
    profile: dict[str, Any],
) -> Path:
    """Write a resolved experiment manifest alongside the output files."""
    output_dir = Path(profile["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "experiment_manifest.json"
    manifest = {
        "experiment": profile["name"],
        "description": profile.get("description", ""),
        "source_script": profile.get("source_script", ""),
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "runs": [
            {
                key: (str(value) if isinstance(value, Path) else value)
                for key, value in run.items()
            }
            for run in profile["runs"]
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


class _TeeTextIO:
    """Mirror writes to multiple text streams."""

    def __init__(self, *streams: Any):
        self._streams = streams

    def write(self, text: str) -> int:
        for stream in self._streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


@contextlib.contextmanager
def _tee_output(log_path: Path | None):
    """Mirror stdout and stderr to a log file when requested."""
    if log_path is None:
        yield
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_handle:
        stdout_tee = _TeeTextIO(sys.stdout, log_handle)
        stderr_tee = _TeeTextIO(sys.stderr, log_handle)
        with (
            contextlib.redirect_stdout(stdout_tee),
            contextlib.redirect_stderr(stderr_tee),
        ):
            yield


def _run_swopp3_configuration(
    *,
    cases: list[str] | None,
    strategy: str | None,
    wind_path: Path | None,
    wave_path: Path | None,
    wind_path_atlantic: Path | None,
    wave_path_atlantic: Path | None,
    wind_path_pacific: Path | None,
    wave_path_pacific: Path | None,
    output_dir: Path,
    submission: int,
    n_points: int,
    max_departures: int | None,
    weather_penalty_weight: float,
    wind_penalty_weight: float,
    wave_penalty_weight: float,
    distance_penalty_weight: float,
    dt_eval_minutes: float,
    cmaes_k: int,
    sigma0: float,
    popsize: int,
    maxfevals: int,
    cmaes_verbose: bool,
    quiet: bool,
) -> None:
    """Execute one resolved SWOPP3 run configuration."""
    import xarray as xr

    from routetools.era5.loader import (
        load_dataset_epoch,
        load_era5_vectorfield,
        load_era5_wavefield,
        load_era5_windfield,
        load_natural_earth_land_mask,
    )
    from routetools.swopp3 import SWOPP3_CASES, departures_2024
    from routetools.swopp3_runner import run_case

    case_ids = _resolve_case_ids(cases, strategy)

    departures = departures_2024()
    if max_departures is not None:
        departures = departures[:max_departures]

    typer.echo(f"Running {len(case_ids)} case(s) × {len(departures)} departure(s)")

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

    if wind_path is not None:
        corridor_wind["atlantic"] = wind_path
        corridor_wind["pacific"] = wind_path
    if wave_path is not None:
        corridor_wave["atlantic"] = wave_path
        corridor_wave["pacific"] = wave_path

    _validate_required_data_paths(case_ids, corridor_wind, corridor_wave)

    _loaded_wind: dict[str, tuple[FieldClosure, datetime]] = {}
    _loaded_wave: dict[str, tuple[FieldClosure, datetime]] = {}
    _loaded_vf: dict[str, FieldClosure] = {}
    _loaded_land: dict[str, object] = {}

    def _get_wind(corridor: str) -> tuple[FieldClosure, datetime]:
        if corridor in _loaded_wind:
            return _loaded_wind[corridor]
        wp = corridor_wind.get(corridor)
        if wp is None:
            raise ValueError(f"No wind path available for corridor '{corridor}'")
        load_paths = _loadable_era5_paths(wp)
        load_target = load_paths if len(load_paths) > 1 else load_paths[0]
        typer.echo(
            f"Loading wind field for {corridor} from "
            f"{', '.join(str(path) for path in load_paths)} …"
        )
        epoch = load_dataset_epoch(load_target)
        wf = load_era5_windfield(load_target)
        _loaded_wind[corridor] = (wf, epoch)
        return wf, epoch

    def _get_vectorfield(corridor: str) -> FieldClosure:
        if corridor in _loaded_vf:
            return _loaded_vf[corridor]
        wp = corridor_wind.get(corridor)
        if wp is None:
            raise ValueError(f"No wind path available for corridor '{corridor}'")
        load_paths = _loadable_era5_paths(wp)
        load_target = load_paths if len(load_paths) > 1 else load_paths[0]
        typer.echo(
            f"Loading vectorfield for {corridor} from "
            f"{', '.join(str(path) for path in load_paths)} …"
        )
        vf = load_era5_vectorfield(load_target)
        _loaded_vf[corridor] = vf
        return vf

    def _get_wave(corridor: str) -> tuple[FieldClosure, datetime]:
        if corridor in _loaded_wave:
            return _loaded_wave[corridor]
        wp = corridor_wave.get(corridor)
        if wp is None:
            raise ValueError(f"No wave path available for corridor '{corridor}'")
        load_paths = _loadable_era5_paths(wp)
        load_target = load_paths if len(load_paths) > 1 else load_paths[0]
        typer.echo(
            f"Loading wave field for {corridor} from "
            f"{', '.join(str(path) for path in load_paths)} …"
        )
        epoch = load_dataset_epoch(load_target)
        wvf = load_era5_wavefield(load_target)
        _loaded_wave[corridor] = (wvf, epoch)
        return wvf, epoch

    def _get_land(corridor: str):
        if corridor in _loaded_land:
            return _loaded_land[corridor]
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
        typer.echo(
            f"Building Natural Earth land mask for {corridor} "
            f"lon={lon_range}, lat={lat_range} …"
        )
        land = load_natural_earth_land_mask(lon_range, lat_range)
        _loaded_land[corridor] = land
        return land

    for cid in case_ids:
        case = SWOPP3_CASES[cid]
        corridor = case["route"]
        typer.echo(f"\n{'=' * 60}")
        typer.echo(f"Case {cid}: {case['label']}")
        typer.echo(
            f"  strategy={case['strategy']}  wps={case['wps']}  route={corridor}"
        )
        typer.echo(f"{'=' * 60}")

        windfield, wind_epoch = _get_wind(corridor)
        wavefield, _ = _get_wave(corridor)
        vectorfield = _get_vectorfield(corridor)
        land = _get_land(corridor)
        dataset_epoch = wind_epoch

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
            weather_penalty_weight=weather_penalty_weight,
            wind_penalty_weight=wind_penalty_weight,
            wave_penalty_weight=wave_penalty_weight,
            distance_penalty_weight=distance_penalty_weight,
            dt_eval_minutes=dt_eval_minutes,
            K=cmaes_k,
            sigma0=sigma0,
            popsize=popsize,
            maxfevals=maxfevals,
            cmaes_verbose=cmaes_verbose,
        )

        energies = [r.energy_mwh for r in results]
        total_time = sum(r.comp_time_s for r in results)
        typer.echo(
            f"  {len(results)} departures  "
            f"mean E={sum(energies) / len(energies):.2f} MWh  "
            f"total comp time={total_time:.1f}s"
        )

    typer.echo(f"\nOutputs written to {output_dir}")


@app.command()
def main(
    experiment: str | None = typer.Argument(  # noqa: B008
        None,
        help=(
            "Experiment profile name from config.toml. When provided, the "
            "profile drives all run parameters."
        ),
    ),
    config_path: Path = typer.Option(  # noqa: B008
        "config.toml",
        "--config-path",
        help="Path to the TOML file that stores named experiment profiles.",
    ),
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
    log_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--log-path",
        help="Path to a log file that mirrors stdout and stderr for the run.",
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
    weather_penalty_weight: float = typer.Option(  # noqa: B008
        0.0,
        "--weather-penalty-weight",
        help="Hard weather penalty weight (step function). 0 to disable.",
    ),
    wind_penalty_weight: float = typer.Option(  # noqa: B008
        0.0,
        "--wind-penalty-weight",
        help="Smooth wind (TWS) penalty weight. 0 to disable.",
    ),
    wave_penalty_weight: float = typer.Option(  # noqa: B008
        0.0,
        "--wave-penalty-weight",
        help="Smooth wave (Hs) penalty weight. 0 to disable.",
    ),
    distance_penalty_weight: float = typer.Option(  # noqa: B008
        0.0,
        "--distance-penalty-weight",
        help="EDT distance-to-land penalty weight. 0 to disable.",
    ),
    dt_eval_minutes: float = typer.Option(  # noqa: B008
        0.0,
        "--dt-eval-minutes",
        help=(
            "Evaluation grid spacing in minutes (\u0394t\u2082). "
            "When positive, the optimizer evaluates B\u00e9zier curves at a "
            "finer resolution than --n-points for more accurate energy "
            "quadrature. 0 = use --n-points for both."
        ),
    ),
    cmaes_k: int = typer.Option(  # noqa: B008
        10,
        "--cmaes-k",
        help="Number of B\u00e9zier control points for CMA-ES.",
    ),
    sigma0: float = typer.Option(  # noqa: B008
        0.1,
        "--sigma0",
        help="Initial CMA-ES step size (sigma0).",
    ),
    popsize: int = typer.Option(  # noqa: B008
        200,
        "--popsize",
        help="CMA-ES population size.",
    ),
    maxfevals: int = typer.Option(  # noqa: B008
        25000,
        "--maxfevals",
        help="Maximum number of CMA-ES function evaluations.",
    ),
    cmaes_verbose: bool = typer.Option(  # noqa: B008
        False,
        "--cmaes-verbose",
        help="Print per-generation CMA-ES diagnostics.",
    ),
    quiet: bool = typer.Option(  # noqa: B008
        False,
        "--quiet",
        "-q",
        help="Suppress progress output.",
    ),
) -> None:
    """Run SWOPP3 competition cases.

    The default invocation expects the 2024 ERA5 files produced by
    ``scripts/download_era5.py``. The command validates those inputs before
    loading any corridor and exits with a precise message if a required file
    is missing.
    """

    def _unwrap_typer_default(value: Any) -> Any:
        if isinstance(value, typer.models.OptionInfo | typer.models.ArgumentInfo):
            return value.default
        return value

    experiment = _unwrap_typer_default(experiment)
    config_path = _unwrap_typer_default(config_path)
    cases = _unwrap_typer_default(cases)
    strategy = _unwrap_typer_default(strategy)
    wind_path = _unwrap_typer_default(wind_path)
    wave_path = _unwrap_typer_default(wave_path)
    wind_path_atlantic = _unwrap_typer_default(wind_path_atlantic)
    wave_path_atlantic = _unwrap_typer_default(wave_path_atlantic)
    wind_path_pacific = _unwrap_typer_default(wind_path_pacific)
    wave_path_pacific = _unwrap_typer_default(wave_path_pacific)
    output_dir = _unwrap_typer_default(output_dir)
    log_path = _unwrap_typer_default(log_path)
    submission = _unwrap_typer_default(submission)
    n_points = _unwrap_typer_default(n_points)
    max_departures = _unwrap_typer_default(max_departures)
    weather_penalty_weight = _unwrap_typer_default(weather_penalty_weight)
    wind_penalty_weight = _unwrap_typer_default(wind_penalty_weight)
    wave_penalty_weight = _unwrap_typer_default(wave_penalty_weight)
    distance_penalty_weight = _unwrap_typer_default(distance_penalty_weight)
    dt_eval_minutes = _unwrap_typer_default(dt_eval_minutes)
    cmaes_k = _unwrap_typer_default(cmaes_k)
    sigma0 = _unwrap_typer_default(sigma0)
    popsize = _unwrap_typer_default(popsize)
    maxfevals = _unwrap_typer_default(maxfevals)
    cmaes_verbose = _unwrap_typer_default(cmaes_verbose)

    with _tee_output(log_path):
        if log_path is not None and not quiet:
            typer.echo(f"Writing run log to {log_path}")

        try:
            if experiment is not None:
                profile = _load_experiment_profile(config_path, experiment)
                manifest_path = _write_experiment_manifest(
                    config_path=config_path,
                    profile=profile,
                )
                if not quiet:
                    typer.echo(
                        f"Loaded experiment '{experiment}' from {config_path} "
                        f"-> {profile['output_dir']}"
                    )
                    typer.echo(f"Wrote experiment manifest to {manifest_path}")

                for run_index, run in enumerate(profile["runs"], start=1):
                    if not quiet:
                        typer.echo(
                            f"\n--- Experiment run {run_index}/{len(profile['runs'])}: "
                            f"{run['name']} ---"
                        )
                    _run_swopp3_configuration(
                        cases=run.get("cases"),
                        strategy=run.get("strategy"),
                        wind_path=run.get("wind_path"),
                        wave_path=run.get("wave_path"),
                        wind_path_atlantic=run.get("wind_path_atlantic"),
                        wave_path_atlantic=run.get("wave_path_atlantic"),
                        wind_path_pacific=run.get("wind_path_pacific"),
                        wave_path_pacific=run.get("wave_path_pacific"),
                        output_dir=Path(profile["output_dir"]),
                        submission=int(run.get("submission", submission)),
                        n_points=int(run.get("n_points", n_points)),
                        max_departures=run.get("max_departures", max_departures),
                        weather_penalty_weight=float(
                            run.get("weather_penalty_weight", weather_penalty_weight)
                        ),
                        wind_penalty_weight=float(
                            run.get("wind_penalty_weight", wind_penalty_weight)
                        ),
                        wave_penalty_weight=float(
                            run.get("wave_penalty_weight", wave_penalty_weight)
                        ),
                        distance_penalty_weight=float(
                            run.get(
                                "distance_penalty_weight",
                                distance_penalty_weight,
                            )
                        ),
                        dt_eval_minutes=float(
                            run.get("dt_eval_minutes", dt_eval_minutes)
                        ),
                        cmaes_k=int(run.get("cmaes_k", cmaes_k)),
                        sigma0=float(run.get("sigma0", sigma0)),
                        popsize=int(run.get("popsize", popsize)),
                        maxfevals=int(run.get("maxfevals", maxfevals)),
                        cmaes_verbose=bool(run.get("cmaes_verbose", cmaes_verbose)),
                        quiet=quiet,
                    )
                return

            _run_swopp3_configuration(
                cases=cases,
                strategy=strategy,
                wind_path=wind_path,
                wave_path=wave_path,
                wind_path_atlantic=wind_path_atlantic,
                wave_path_atlantic=wave_path_atlantic,
                wind_path_pacific=wind_path_pacific,
                wave_path_pacific=wave_path_pacific,
                output_dir=output_dir,
                submission=submission,
                n_points=n_points,
                max_departures=max_departures,
                weather_penalty_weight=weather_penalty_weight,
                wind_penalty_weight=wind_penalty_weight,
                wave_penalty_weight=wave_penalty_weight,
                distance_penalty_weight=distance_penalty_weight,
                dt_eval_minutes=dt_eval_minutes,
                cmaes_k=cmaes_k,
                sigma0=sigma0,
                popsize=popsize,
                maxfevals=maxfevals,
                cmaes_verbose=cmaes_verbose,
                quiet=quiet,
            )
        except (FileNotFoundError, KeyError, ValueError) as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(1) from exc


if __name__ == "__main__":
    app()
