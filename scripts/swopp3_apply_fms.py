#!/usr/bin/env python
"""Apply FMS refinement to existing SWOPP3 route outputs.

This script reads a folder produced by ``scripts/swopp3_run.py`` and writes a
new folder with suffix ``_fms`` by default. Great-circle cases are copied as-is.
Optimised cases are refined route-by-route with FMS.

Usage
-----
Refine the default SWOPP3 output folder and write ``output/swopp3_fms``::

    uv run scripts/swopp3_apply_fms.py output/swopp3

Use explicit ERA5 paths for both corridors::

    uv run scripts/swopp3_apply_fms.py output/swopp3 \
        --wind-path-atlantic data/era5/era5_wind_atlantic_2024.nc \
        --wave-path-atlantic data/era5/era5_waves_atlantic_2024.nc \
        --wind-path-pacific  data/era5/era5_wind_pacific_2024.nc \
        --wave-path-pacific  data/era5/era5_waves_pacific_2024.nc
"""

from __future__ import annotations

import csv
import gc
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import typer

from routetools.cost import cost_function_rise_penalized
from routetools.fms import optimize_fms
from routetools.swopp3 import SWOPP3_CASES
from routetools.swopp3_output import (
    file_a_row,
    resolve_file_b_path,
    sailed_distance_nm,
    waypoint_times,
    write_file_a,
    write_file_b,
)
from routetools.swopp3_runner import evaluate_energy
from routetools.weather import DEFAULT_HS_LIMIT, DEFAULT_TWS_LIMIT

if TYPE_CHECKING:
    from routetools.swopp3_runner import FieldClosure


app = typer.Typer(help="Apply FMS refinement to an existing SWOPP3 output folder.")

_TEAM_FILE_RE = re.compile(r"^IEUniversity-(?P<submission>\d+)-(?P<case_id>.+)\.csv$")
_ERA5_FILE_RE = re.compile(
    r"^(?P<prefix>era5_[^_]+_[^_]+_)(?P<year>\d{4})(?:_(?P<suffix>\d{2}(?:-\d{2})?))?\.nc$"
)
_DTFMT = "%Y-%m-%d %H:%M:%S"
_DEFAULT_ERA5_BATCH_DAYS = 183.0
_DEFAULT_ERA5_RELOAD_MARGIN_DAYS = 20.0

# TODO: set to 0 for final runs, or make configurable via CLI options
WIND_PW = 1000
WAVE_PW = 1000
ENFORCE_WEATHER_LIMITS = False


@dataclass(frozen=True)
class CaseFile:
    """Summary CSV metadata discovered in an input folder."""

    case_id: str
    submission: int
    summary_path: Path


@dataclass(frozen=True)
class CorridorResources:
    """Loaded weather, vectorfield, and land resources for one corridor."""

    vectorfield: FieldClosure
    windfield: FieldClosure
    wavefield: FieldClosure
    land: Any
    dataset_epoch: datetime


def _default_output_dir(input_dir: Path) -> Path:
    """Return the default FMS output directory for an input folder."""
    return input_dir.with_name(f"{input_dir.name}_fms")


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


def _discover_case_files(input_dir: Path) -> list[CaseFile]:
    """Return all valid SWOPP3 summary CSVs in an input directory."""
    case_files: list[CaseFile] = []
    for path in sorted(input_dir.glob("IEUniversity-*.csv")):
        if not path.is_file():
            continue
        match = _TEAM_FILE_RE.match(path.name)
        if match is None:
            continue
        case_id = match.group("case_id")
        if case_id not in SWOPP3_CASES:
            continue
        case_files.append(
            CaseFile(
                case_id=case_id,
                submission=int(match.group("submission")),
                summary_path=path,
            )
        )
    return case_files


def _build_corridor_path_maps(
    *,
    wind_path: Path | None,
    wave_path: Path | None,
    wind_path_atlantic: Path | None,
    wave_path_atlantic: Path | None,
    wind_path_pacific: Path | None,
    wave_path_pacific: Path | None,
) -> tuple[dict[str, Path], dict[str, Path]]:
    """Build per-corridor field path maps."""
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

    return corridor_wind, corridor_wave


def _validate_required_data_paths(
    case_ids: list[str],
    corridor_wind: dict[str, Path],
    corridor_wave: dict[str, Path],
) -> None:
    """Fail fast when the required ERA5 inputs are missing."""
    required_corridors = sorted(
        {str(SWOPP3_CASES[case_id]["route"]) for case_id in case_ids}
    )
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
        "SWOPP3 FMS input validation failed.\n\n"
        f"Optimised cases require ERA5 datasets for corridor(s): {corridor_list}.\n"
        "The FMS post-processing step uses wind data for the vectorfield and\n"
        "uses wind and wave data to evaluate each original and refined route.\n\n"
        f"Missing inputs:\n{missing_lines}\n\n"
        "Fix:\n"
        "- Run `uv run scripts/download_era5.py` to download the default "
        "2024 datasets.\n"
        "- Or pass matching `--wind-path*` and `--wave-path*` options."
    )


def _load_corridor_resources_for_cases(
    case_ids: list[str],
    corridor_wind: dict[str, Path],
    corridor_wave: dict[str, Path],
    *,
    time_start: datetime | None = None,
    time_end: datetime | None = None,
    quiet: bool,
) -> dict[str, CorridorResources]:
    """Load weather, vectorfield, and land resources needed by optimised cases."""
    if not case_ids:
        return {}

    import xarray as xr

    from routetools.era5.loader import (
        load_dataset_epoch,
        load_era5_wavefield,
        load_era5_windfield,
        load_natural_earth_land_mask,
    )

    resources: dict[str, CorridorResources] = {}
    corridors = sorted({str(SWOPP3_CASES[case_id]["route"]) for case_id in case_ids})

    for corridor in corridors:
        wind_path = corridor_wind[corridor]
        wave_path = corridor_wave[corridor]

        wind_paths = _loadable_era5_paths(wind_path)
        wave_paths = _loadable_era5_paths(wave_path)
        wind_target = wind_paths if len(wind_paths) > 1 else wind_paths[0]
        wave_target = wave_paths if len(wave_paths) > 1 else wave_paths[0]

        if not quiet:
            typer.echo(
                f"Loading corridor {corridor}: wind from "
                f"{', '.join(str(path) for path in wind_paths)}"
            )
            typer.echo(
                f"Loading corridor {corridor}: waves from "
                f"{', '.join(str(path) for path in wave_paths)}"
            )

        dataset_epoch = load_dataset_epoch(
            wind_target,
            time_start=time_start,
            time_end=time_end,
        )
        windfield = load_era5_windfield(
            wind_target,
            time_start=time_start,
            time_end=time_end,
        )
        vectorfield = windfield
        wavefield = load_era5_wavefield(
            wave_target,
            time_start=time_start,
            time_end=time_end,
        )

        with xr.open_dataset(wave_paths[0]) as ds:
            for lon_name in ("longitude", "lon"):
                if lon_name in ds.coords:
                    lons = ds[lon_name].values
                    break
            else:
                raise KeyError(f"No longitude coordinate found in {wave_paths[0]}")

            for lat_name in ("latitude", "lat"):
                if lat_name in ds.coords:
                    lats = ds[lat_name].values
                    break
            else:
                raise KeyError(f"No latitude coordinate found in {wave_paths[0]}")

        land = load_natural_earth_land_mask(
            (float(lons.min()), float(lons.max())),
            (float(lats.min()), float(lats.max())),
        )

        resources[corridor] = CorridorResources(
            vectorfield=vectorfield,
            windfield=windfield,
            wavefield=wavefield,
            land=land,
            dataset_epoch=dataset_epoch,
        )

    return resources


def _read_file_a_rows(summary_path: Path) -> list[dict[str, str]]:
    """Read one SWOPP3 summary CSV as raw rows."""
    with summary_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _read_track_curve(track_path: Path) -> jnp.ndarray:
    """Read a SWOPP3 track CSV as a ``(L, 2)`` ``(lon, lat)`` array."""
    lons: list[float] = []
    lats: list[float] = []
    with track_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            lats.append(float(row["lat_deg"]))
            lons.append(float(row["lon_deg"]))
    return jnp.stack(
        [jnp.asarray(lons, dtype=jnp.float32), jnp.asarray(lats, dtype=jnp.float32)],
        axis=1,
    )


def _departure_offset_hours(departure: datetime, dataset_epoch: datetime) -> float:
    """Return departure offset in hours relative to the dataset epoch."""
    departure_naive = departure.replace(tzinfo=None) if departure.tzinfo else departure
    epoch_naive = (
        dataset_epoch.replace(tzinfo=None)
        if hasattr(dataset_epoch, "tzinfo") and dataset_epoch.tzinfo
        else dataset_epoch
    )
    return (departure_naive - epoch_naive).total_seconds() / 3600.0


def _copy_case_outputs(case_file: CaseFile, input_dir: Path, output_dir: Path) -> None:
    """Copy one case's summary CSV and all referenced tracks unchanged."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tracks").mkdir(parents=True, exist_ok=True)
    shutil.copy2(case_file.summary_path, output_dir / case_file.summary_path.name)

    for row in _read_file_a_rows(case_file.summary_path):
        filename = row["details_filename"]
        src = resolve_file_b_path(input_dir, filename)
        dst = output_dir / "tracks" / filename
        shutil.copy2(src, dst)


def _case_output_complete(case_file: CaseFile, output_dir: Path) -> bool:
    """Return whether a case summary and all referenced tracks already exist."""
    summary_path = output_dir / case_file.summary_path.name
    if not summary_path.exists():
        return False

    try:
        rows = _read_file_a_rows(summary_path)
    except (OSError, csv.Error, KeyError):
        return False

    if not rows:
        return False

    return all(
        (output_dir / "tracks" / row["details_filename"]).exists() for row in rows
    )


def _release_fms_state() -> None:
    """Release cached JAX/FMS state between optimised cases."""
    if hasattr(jax, "clear_caches"):
        jax.clear_caches()
    gc.collect()


def _batch_window_parameters(
    passage_hours: float,
    era5_batch_days: float,
    era5_reload_margin_days: float,
) -> tuple[timedelta, timedelta]:
    """Return batch duration and reload margin for rolling ERA5 windows."""
    if era5_batch_days <= 0:
        raise ValueError("era5_batch_days must be positive")
    if era5_reload_margin_days <= 0:
        raise ValueError("era5_reload_margin_days must be positive")

    margin_hours = max(passage_hours, era5_reload_margin_days * 24.0)
    batch_hours = max(margin_hours, era5_batch_days * 24.0)
    return timedelta(hours=batch_hours), timedelta(hours=margin_hours)


def apply_fms_to_outputs(
    input_dir: Path,
    *,
    output_dir: Path | None = None,
    wind_path: Path | None = None,
    wave_path: Path | None = None,
    wind_path_atlantic: Path | None = Path("data/era5/era5_wind_atlantic_2024.nc"),
    wave_path_atlantic: Path | None = Path("data/era5/era5_waves_atlantic_2024.nc"),
    wind_path_pacific: Path | None = Path("data/era5/era5_wind_pacific_2024.nc"),
    wave_path_pacific: Path | None = Path("data/era5/era5_waves_pacific_2024.nc"),
    fms_patience: int = 200,
    fms_damping: float = 0.95,
    fms_maxfevals: int = 10000,
    era5_batch_days: float = _DEFAULT_ERA5_BATCH_DAYS,
    era5_reload_margin_days: float = _DEFAULT_ERA5_RELOAD_MARGIN_DAYS,
    tws_limit: float = DEFAULT_TWS_LIMIT,
    hs_limit: float = DEFAULT_HS_LIMIT,
    quiet: bool = False,
) -> Path:
    """Apply FMS to an existing SWOPP3 output folder and write a sibling folder."""
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    resolved_output_dir = (
        _default_output_dir(input_dir) if output_dir is None else Path(output_dir)
    )
    if resolved_output_dir == input_dir:
        raise ValueError("output_dir must be different from input_dir")

    case_files = _discover_case_files(input_dir)
    if not case_files:
        raise FileNotFoundError(
            f"No SWOPP3 summary CSV files found in input directory: {input_dir}"
        )

    optimised_case_ids = [
        case_file.case_id
        for case_file in case_files
        if SWOPP3_CASES[case_file.case_id]["strategy"] == "optimised"
    ]

    corridor_wind, corridor_wave = _build_corridor_path_maps(
        wind_path=wind_path,
        wave_path=wave_path,
        wind_path_atlantic=wind_path_atlantic,
        wave_path_atlantic=wave_path_atlantic,
        wind_path_pacific=wind_path_pacific,
        wave_path_pacific=wave_path_pacific,
    )

    if optimised_case_ids:
        _validate_required_data_paths(optimised_case_ids, corridor_wind, corridor_wave)

    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    (resolved_output_dir / "tracks").mkdir(parents=True, exist_ok=True)

    for case_file in case_files:
        if _case_output_complete(case_file, resolved_output_dir):
            if not quiet:
                typer.echo(f"Skipping completed case {case_file.case_id}")
            continue

        case = SWOPP3_CASES[case_file.case_id]
        if case["strategy"] == "gc":
            if not quiet:
                typer.echo(f"Copying GC case {case_file.case_id} unchanged")
            _copy_case_outputs(case_file, input_dir, resolved_output_dir)
            continue

        corridor = str(case["route"])
        passage_hours = float(case["passage_hours"])
        batch_duration, reload_margin = _batch_window_parameters(
            passage_hours,
            era5_batch_days,
            era5_reload_margin_days,
        )
        output_rows: list[dict[str, str]] = []
        resources: CorridorResources | None = None
        reload_after: datetime | None = None

        try:
            rows = _read_file_a_rows(case_file.summary_path)
            for idx, row in enumerate(rows, start=1):
                departure = datetime.strptime(row["departure_time_utc"], _DTFMT)
                if (
                    resources is None
                    or reload_after is None
                    or departure >= reload_after
                ):
                    if resources is not None:
                        del resources
                        _release_fms_state()

                    batch_start = departure
                    batch_end = batch_start + batch_duration
                    reload_after = batch_end - reload_margin
                    resources = _load_corridor_resources_for_cases(
                        [case_file.case_id],
                        corridor_wind,
                        corridor_wave,
                        time_start=batch_start,
                        time_end=batch_end,
                        quiet=quiet,
                    )[corridor]
                    if not quiet:
                        typer.echo(
                            f"Loaded {corridor} ERA5 batch for {case_file.case_id}: "
                            f"{batch_start.strftime('%Y-%m-%d')} to "
                            f"{batch_end.strftime('%Y-%m-%d')}"
                        )

                if resources is None:
                    raise RuntimeError(
                        f"Failed to load corridor resources for {case_file.case_id}"
                    )

                details_filename = row["details_filename"]
                track_path = resolve_file_b_path(input_dir, details_filename)
                curve_original = _read_track_curve(track_path)
                departure_offset_h = _departure_offset_hours(
                    departure,
                    resources.dataset_epoch,
                )

                curve_fms_batch, _ = optimize_fms(
                    vectorfield=resources.vectorfield,
                    curve=curve_original,
                    land=resources.land,
                    windfield=resources.windfield,
                    wavefield=resources.wavefield,
                    penalty=1.0,
                    travel_time=passage_hours,
                    patience=fms_patience,
                    damping=fms_damping,
                    maxfevals=fms_maxfevals,
                    spherical_correction=True,
                    costfun=cost_function_rise_penalized,
                    costfun_kwargs={
                        "windfield": resources.windfield,
                        "wavefield": resources.wavefield,
                        "wps": bool(case["wps"]),
                        "wave_penalty_weight": WAVE_PW,
                        "wind_penalty_weight": WIND_PW,
                        "tws_limit": tws_limit,
                        "hs_limit": hs_limit,
                    },
                    verbose=not quiet,
                    time_offset=departure_offset_h,
                    enforce_weather_limits=ENFORCE_WEATHER_LIMITS,
                    tws_limit=tws_limit,
                    hs_limit=hs_limit,
                )
                curve_fms = curve_fms_batch[0]

                original_energy, original_max_tws, original_max_hs = evaluate_energy(
                    curve_original,
                    departure,
                    passage_hours,
                    wps=bool(case["wps"]),
                    windfield=resources.windfield,
                    wavefield=resources.wavefield,
                    departure_offset_h=departure_offset_h,
                )
                fms_energy, fms_max_tws, fms_max_hs = evaluate_energy(
                    curve_fms,
                    departure,
                    passage_hours,
                    wps=bool(case["wps"]),
                    windfield=resources.windfield,
                    wavefield=resources.wavefield,
                    departure_offset_h=departure_offset_h,
                )

                distance_nm = sailed_distance_nm(curve_fms)
                write_file_b(
                    curve_fms,
                    waypoint_times(curve_fms, departure, passage_hours),
                    resolved_output_dir / "tracks" / details_filename,
                )
                output_rows.append(
                    file_a_row(
                        departure=departure,
                        passage_hours=passage_hours,
                        energy_mwh=fms_energy,
                        max_wind_mps=fms_max_tws,
                        max_hs_m=fms_max_hs,
                        distance_nm=distance_nm,
                        details_filename=details_filename,
                    )
                )

                if not quiet:
                    typer.echo(
                        f"[{case_file.case_id}] {idx}/{len(rows)} "
                        f"{departure.strftime('%Y-%m-%d')} "
                        f"original={original_energy:.3f} MWh  "
                        f"fms={fms_energy:.3f} MWh"
                    )

            write_file_a(output_rows, resolved_output_dir / case_file.summary_path.name)
        finally:
            if resources is not None:
                del resources
                _release_fms_state()

    if not quiet:
        typer.echo(f"Wrote FMS-refined output folder to {resolved_output_dir}")
    return resolved_output_dir


@app.command()
def main(
    input_dir: Path = typer.Argument(  # noqa: B008
        ..., help="Folder produced by swopp3_run.py."
    ),
    output_dir: Path | None = typer.Option(  # noqa: B008
        None,
        "--output-dir",
        "-o",
        help="Output directory. Default: <input-dir>_fms.",
    ),
    wind_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--wind-path",
        help="Path to ERA5 wind NetCDF used for all selected corridors.",
    ),
    wave_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--wave-path",
        help="Path to ERA5 wave NetCDF used for all selected corridors.",
    ),
    wind_path_atlantic: Path | None = typer.Option(  # noqa: B008
        Path("data/era5/era5_wind_atlantic_2024.nc"),
        "--wind-path-atlantic",
        help="Path to ERA5 wind NetCDF for Atlantic routes.",
    ),
    wave_path_atlantic: Path | None = typer.Option(  # noqa: B008
        Path("data/era5/era5_waves_atlantic_2024.nc"),
        "--wave-path-atlantic",
        help="Path to ERA5 wave NetCDF for Atlantic routes.",
    ),
    wind_path_pacific: Path | None = typer.Option(  # noqa: B008
        Path("data/era5/era5_wind_pacific_2024.nc"),
        "--wind-path-pacific",
        help="Path to ERA5 wind NetCDF for Pacific routes.",
    ),
    wave_path_pacific: Path | None = typer.Option(  # noqa: B008
        Path("data/era5/era5_waves_pacific_2024.nc"),
        "--wave-path-pacific",
        help="Path to ERA5 wave NetCDF for Pacific routes.",
    ),
    fms_patience: int = typer.Option(  # noqa: B008
        200,
        "--fms-patience",
        help="Early-stopping patience for FMS.",
    ),
    fms_damping: float = typer.Option(  # noqa: B008
        0.95,
        "--fms-damping",
        help="FMS damping factor.",
    ),
    fms_maxfevals: int = typer.Option(  # noqa: B008
        10000,
        "--fms-maxfevals",
        help="Maximum FMS iterations per route.",
    ),
    era5_batch_days: float = typer.Option(  # noqa: B008
        _DEFAULT_ERA5_BATCH_DAYS,
        "--era5-batch-days",
        help="Maximum number of days of ERA5 data to keep loaded at once.",
    ),
    era5_reload_margin_days: float = typer.Option(  # noqa: B008
        _DEFAULT_ERA5_RELOAD_MARGIN_DAYS,
        "--era5-reload-margin-days",
        help=(
            "Reload ERA5 data when a departure is this close to the current batch end."
        ),
    ),
    tws_limit: float = typer.Option(  # noqa: B008
        DEFAULT_TWS_LIMIT,
        "--tws-limit",
        help="Maximum true wind speed allowed during FMS refinement.",
    ),
    hs_limit: float = typer.Option(  # noqa: B008
        DEFAULT_HS_LIMIT,
        "--hs-limit",
        help="Maximum significant wave height allowed during FMS refinement.",
    ),
    quiet: bool = typer.Option(  # noqa: B008
        False,
        "--quiet",
        "-q",
        help="Suppress progress output.",
    ),
) -> None:
    """Apply FMS to every non-GC route in an existing SWOPP3 output folder."""
    apply_fms_to_outputs(
        input_dir,
        output_dir=output_dir,
        wind_path=wind_path,
        wave_path=wave_path,
        wind_path_atlantic=wind_path_atlantic,
        wave_path_atlantic=wave_path_atlantic,
        wind_path_pacific=wind_path_pacific,
        wave_path_pacific=wave_path_pacific,
        fms_patience=fms_patience,
        fms_damping=fms_damping,
        fms_maxfevals=fms_maxfevals,
        era5_batch_days=era5_batch_days,
        era5_reload_margin_days=era5_reload_margin_days,
        tws_limit=tws_limit,
        hs_limit=hs_limit,
        quiet=quiet,
    )


if __name__ == "__main__":
    app()
