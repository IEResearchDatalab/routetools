#!/usr/bin/env python
"""Apply FMS refinement to scored SWOPP3 submission archives.

This script reads scored submission archives (produced by the competition
scoring system) and applies FMS post-processing to all non-GC routes,
writing a new folder with FMS-refined routes and comparison metrics.

Each scored archive contains an embedded resampled_tracks.zip with the
submitted routes. This script:

1. Extracts the submission from the scored archive.
2. Reconstructs summary CSVs (File A) from track metadata.
3. Applies FMS to non-GC routes.
4. Generates comparison tables (before/after FMS consumption and violations).

The FMS-refined submissions are stored in the same directory as the original
archives with a "_fms" suffix (e.g., "ohy123_fms"), allowing them to be
automatically discovered as additional competitors by swopp3_submission_compare.py.

Usage
-----
Apply FMS to submissions and store results in swopp3_submissions_score::

    uv run scripts/swopp3_apply_fms_to_scored_submissions.py \
        output/swopp3_submissions_score/*boatface*.zip \
        output/swopp3_submissions_score/*ohy*.zip

    # Creates: output/swopp3_submissions_score/mc boatface_fms/
    #          output/swopp3_submissions_score/ohy123_fms/

Apply to a single submission with custom output directory::

    uv run scripts/swopp3_apply_fms_to_scored_submissions.py \
        output/swopp3_submissions_score/704564_mc_boatface_*.zip \
        --output-dir output/custom_fms_results

Apply to multiple submissions with custom base directory::

    uv run scripts/swopp3_apply_fms_to_scored_submissions.py \
        output/swopp3_submissions_score/*boatface*.zip \
        output/swopp3_submissions_score/*ohy*.zip \
        --output-base output/swopp3_fms_archive

    # Creates: output/swopp3_fms_archive/mc boatface_fms/
    #          output/swopp3_fms_archive/ohy123_fms/
"""

from __future__ import annotations

import csv
import gc
import json
import re
import shutil
import struct
import zipfile
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
    sailed_distance_nm,
    waypoint_times,
    write_file_a,
    write_file_b,
)
from routetools.swopp3_runner import evaluate_energy
from routetools.weather import DEFAULT_HS_LIMIT, DEFAULT_TWS_LIMIT

if TYPE_CHECKING:
    from routetools.swopp3_runner import FieldClosure


app = typer.Typer(help="Apply FMS refinement to scored SWOPP3 submission archives.")

_TRACK_FILENAME_RE = re.compile(
    r"^details_(?P<case>.+?)_8kn_(?P<date>\d{4}-\d{2}-\d{2})T\d{2}\.csv$"
)
_ERA5_FILE_RE = re.compile(
    r"^(?P<prefix>era5_[^_]+_[^_]+_)(?P<year>\d{4})(?:_(?P<suffix>\d{2}(?:-\d{2})?))?\.nc$"
)
_DTFMT = "%Y-%m-%d %H:%M:%S"
_DEFAULT_ERA5_BATCH_DAYS = 183.0
_DEFAULT_ERA5_RELOAD_MARGIN_DAYS = 20.0

# Default penalty weights forwarded to cost_function_rise_penalized.
_DEFAULT_WIND_PENALTY_WEIGHT = 1000
_DEFAULT_WAVE_PENALTY_WEIGHT = 1000


@dataclass(frozen=True)
class TrackMetadata:
    """Extracted metadata from a track filename."""

    case_id: str
    departure_utc: datetime


@dataclass(frozen=True)
class CorridorResources:
    """Loaded weather, vectorfield, and land resources for one corridor."""

    vectorfield: FieldClosure
    windfield: FieldClosure
    wavefield: FieldClosure
    land: Any
    dataset_epoch: datetime


def _parse_track_filename(filename: str) -> TrackMetadata | None:
    """Extract case and departure datetime from a resampled track filename."""
    match = _TRACK_FILENAME_RE.match(filename)
    if match is None:
        return None

    case = match.group("case")
    date_str = match.group("date")
    # Departures are always at 12:00 UTC
    departure_utc = datetime.strptime(f"{date_str} 12:00:00", _DTFMT)
    return TrackMetadata(case_id=case, departure_utc=departure_utc)


def _extract_zip_member_bytes(archive_path: Path, member_name: str) -> bytes:
    """Return decompressed bytes for a ZIP member from a raw ZIP stream."""
    data = archive_path.read_bytes()
    member_bytes = member_name.encode("utf-8")
    member_idx = data.find(member_bytes)
    if member_idx < 0:
        raise KeyError(f"Member not found in ZIP: {member_name}")

    header_idx = data.rfind(b"PK\x03\x04", 0, member_idx)
    if header_idx < 0:
        raise KeyError(f"ZIP header not found for member: {member_name}")

    (
        _signature,
        _version,
        _flags,
        compression,
        _mtime,
        _mdate,
        _crc32,
        compressed_size,
        _uncompressed_size,
        name_length,
        extra_length,
    ) = struct.unpack("<IHHHHHIIIHH", data[header_idx : header_idx + 30])

    payload_start = header_idx + 30 + name_length + extra_length
    payload_end = payload_start + compressed_size
    payload = data[payload_start:payload_end]
    if compression == 0:
        return payload
    if compression == 8:
        import zlib

        return zlib.decompress(payload, -zlib.MAX_WBITS)
    raise ValueError(
        f"Unsupported ZIP compression method {compression} in {member_name}"
    )


def _extract_submission_from_scored_archive(
    archive_path: Path,
) -> tuple[dict[str, list[Path]], dict[str, dict[str, Any]], Path]:
    """Extract submission structure from a scored archive.

    Returns
    -------
    tuple[dict[str, list[Path]], dict[str, dict[str, Any]], Path]
        Mapping of case_id to list of extracted track paths,
        metadata dict for each case (for reconstructing summary CSVs),
        and temp directory path.
    """
    import tempfile

    temp_dir = Path(tempfile.mkdtemp(prefix="swopp3_fms_"))

    try:
        # Extract resampled_tracks.zip from the scored archive
        resampled_zip_bytes = _extract_zip_member_bytes(
            archive_path, "resampled_tracks.zip"
        )
        resampled_zip_path = temp_dir / "resampled_tracks.zip"
        resampled_zip_path.write_bytes(resampled_zip_bytes)

        # Extract all track files from resampled_tracks.zip
        tracks_by_case: dict[str, list[Path]] = {case: [] for case in SWOPP3_CASES}
        metadata_by_case: dict[str, dict[str, Any]] = {
            case: {
                "tracks": [],
                "departures": [],
                "distances": [],
                "energy_mwh": [],
                "max_wind_mps": [],
                "max_hs_m": [],
            }
            for case in SWOPP3_CASES
        }

        with zipfile.ZipFile(resampled_zip_path, "r") as resampled_zip:
            for member_name in resampled_zip.namelist():
                if not member_name.endswith(".csv") or "resampled/" not in member_name:
                    continue

                # Extract case_id from the folder path (e.g., resampled/AO_WPS/...)
                parts = Path(member_name).parts
                if len(parts) < 3 or parts[0] != "resampled":
                    continue
                case_id = parts[1]

                # Validate case_id
                if case_id not in SWOPP3_CASES:
                    continue

                filename = Path(member_name).name
                # Parse departure from filename (always 12:00 UTC)
                match = re.search(r"(\d{4}-\d{2}-\d{2})", filename)
                if match is None:
                    continue
                date_str = match.group(1)
                departure_utc = datetime.strptime(f"{date_str} 12:00:00", _DTFMT)

                # Extract and save the track file
                case_dir = temp_dir / "tracks" / case_id
                case_dir.mkdir(parents=True, exist_ok=True)
                track_path = case_dir / filename
                track_path.write_bytes(resampled_zip.read(member_name))

                tracks_by_case[case_id].append(track_path)
                metadata_by_case[case_id]["tracks"].append(str(track_path))
                metadata_by_case[case_id]["departures"].append(departure_utc)

        return tracks_by_case, metadata_by_case, temp_dir

    except Exception as e:
        # Clean up on error
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise e


def _read_track_curve(track_path: Path) -> jnp.ndarray:
    """Read a SWOPP3 track CSV as a ``(L, 2)`` ``(lon, lat)`` array.

    Handles both original SWOPP3 format (lat_deg, lon_deg) and
    resampled format (lat, lon).
    """
    lons: list[float] = []
    lats: list[float] = []
    with track_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            # Try both column name formats
            lat_val = row.get("lat_deg") or row.get("lat")
            lon_val = row.get("lon_deg") or row.get("lon")
            if lat_val is not None and lon_val is not None:
                lats.append(float(lat_val))
                lons.append(float(lon_val))
    return jnp.stack(
        [jnp.asarray(lons, dtype=jnp.float32), jnp.asarray(lats, dtype=jnp.float32)],
        axis=1,
    )


def _count_curve_land_violations(curve: jnp.ndarray, land: Any) -> int:
    """Return the number of land-invalid positions for one route."""
    curve_batch = curve if curve.ndim == 3 else curve[None, ...]
    return int(jnp.sum(land(curve_batch) > 0))


def _departure_offset_hours(departure: datetime, dataset_epoch: datetime) -> float:
    """Return departure offset in hours relative to the dataset epoch."""
    departure_naive = departure.replace(tzinfo=None) if departure.tzinfo else departure
    epoch_naive = (
        dataset_epoch.replace(tzinfo=None)
        if hasattr(dataset_epoch, "tzinfo") and dataset_epoch.tzinfo
        else dataset_epoch
    )
    return (departure_naive - epoch_naive).total_seconds() / 3600.0


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
        f"Non-GC cases require ERA5 datasets for corridor(s): {corridor_list}.\n"
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
    """Load weather, vectorfield, and land resources needed by non-GC cases."""
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


def _participant_name_from_scored_zip(zip_path: Path) -> str:
    """Extract participant name from scored submission zip filename."""
    stem = zip_path.stem
    # Expected pattern: XXXXXX_participant_name_PhaseId...
    match = re.match(r"^\d+_(.+?)_PhaseId.*$", stem, flags=re.IGNORECASE)
    participant = match.group(1) if match else stem
    return participant.replace("_", " ").strip() or stem


def _is_non_gc_case(case_id: str) -> bool:
    """Return whether a case is non-GC (optimised strategy)."""
    return SWOPP3_CASES[case_id]["strategy"] != "gc"


def apply_fms_to_scored_submission(
    archive_path: Path,
    *,
    output_dir: Path,
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
    wind_penalty_weight: float = _DEFAULT_WIND_PENALTY_WEIGHT,
    wave_penalty_weight: float = _DEFAULT_WAVE_PENALTY_WEIGHT,
    enforce_weather_limits: bool = False,
    quiet: bool = False,
) -> tuple[Path, dict[str, list[dict[str, Any]]]]:
    """Apply FMS to a scored submission archive and write results with comparisons.

    Returns
    -------
    tuple[Path, dict[str, list[dict[str, Any]]]]
        Output directory and comparison tables by case_id.
    """
    archive_path = Path(archive_path)
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract the submission from the scored archive
    if not quiet:
        typer.echo(f"Extracting submission from {archive_path.name}...")

    tracks_by_case, metadata_by_case, temp_dir = (
        _extract_submission_from_scored_archive(archive_path)
    )

    try:
        # Identify non-GC cases that need FMS
        non_gc_cases = [case for case in SWOPP3_CASES if _is_non_gc_case(case)]
        non_gc_case_ids = [case for case in non_gc_cases if tracks_by_case[case]]

        if not non_gc_case_ids:
            if not quiet:
                typer.echo("No non-GC cases found in submission.")
            return output_dir, {}

        corridor_wind, corridor_wave = _build_corridor_path_maps(
            wind_path=wind_path,
            wave_path=wave_path,
            wind_path_atlantic=wind_path_atlantic,
            wave_path_atlantic=wave_path_atlantic,
            wind_path_pacific=wind_path_pacific,
            wave_path_pacific=wave_path_pacific,
        )

        _validate_required_data_paths(non_gc_case_ids, corridor_wind, corridor_wave)

        (output_dir / "tracks").mkdir(parents=True, exist_ok=True)
        comparison_tables: dict[str, list[dict[str, Any]]] = {}

        for case_id in non_gc_case_ids:
            if not quiet:
                typer.echo(f"Processing case {case_id}...")

            case = SWOPP3_CASES[case_id]
            corridor = str(case["route"])
            passage_hours = float(case["passage_hours"])
            batch_duration, reload_margin = _batch_window_parameters(
                passage_hours,
                era5_batch_days,
                era5_reload_margin_days,
            )

            output_rows: list[dict[str, str]] = []
            comparison_rows: list[dict[str, Any]] = []
            resources: CorridorResources | None = None
            reload_after: datetime | None = None

            track_paths = sorted(tracks_by_case[case_id])

            try:
                for idx, track_path in enumerate(track_paths, start=1):
                    # Extract departure date from track filename (always 12:00 UTC)
                    match = re.search(r"(\d{4}-\d{2}-\d{2})", track_path.name)
                    if match is None:
                        continue
                    date_str = match.group(1)
                    departure = datetime.strptime(f"{date_str} 12:00:00", _DTFMT)

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
                            [case_id],
                            corridor_wind,
                            corridor_wave,
                            time_start=batch_start,
                            time_end=batch_end,
                            quiet=quiet,
                        )[corridor]
                        if not quiet:
                            typer.echo(
                                f"Loaded {corridor} ERA5 batch for {case_id}: "
                                f"{batch_start.strftime('%Y-%m-%d')} to "
                                f"{batch_end.strftime('%Y-%m-%d')}"
                            )

                    if resources is None:
                        raise RuntimeError(
                            f"Failed to load corridor resources for {case_id}"
                        )

                    curve_original = _read_track_curve(track_path)
                    departure_offset_h = _departure_offset_hours(
                        departure,
                        resources.dataset_epoch,
                    )

                    # Apply FMS
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
                            "wave_penalty_weight": wave_penalty_weight,
                            "wind_penalty_weight": wind_penalty_weight,
                            "tws_limit": tws_limit,
                            "hs_limit": hs_limit,
                        },
                        verbose=not quiet,
                        time_offset=departure_offset_h,
                        enforce_weather_limits=enforce_weather_limits,
                        tws_limit=tws_limit,
                        hs_limit=hs_limit,
                    )
                    curve_fms = curve_fms_batch[0]

                    # Check land violations
                    original_land_violations = _count_curve_land_violations(
                        curve_original,
                        resources.land,
                    )
                    fms_land_violations = _count_curve_land_violations(
                        curve_fms,
                        resources.land,
                    )
                    if fms_land_violations > original_land_violations:
                        curve_fms = curve_original
                        fms_land_violations = original_land_violations

                    # Evaluate energy and weather metrics
                    original_energy, original_max_tws, original_max_hs = (
                        evaluate_energy(
                            curve_original,
                            departure,
                            passage_hours,
                            wps=bool(case["wps"]),
                            windfield=resources.windfield,
                            wavefield=resources.wavefield,
                            departure_offset_h=departure_offset_h,
                        )
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
                    details_filename = track_path.name
                    write_file_b(
                        curve_fms,
                        waypoint_times(curve_fms, departure, passage_hours),
                        output_dir / "tracks" / details_filename,
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

                    # Record comparison
                    comparison_rows.append(
                        {
                            "departure_utc": departure.isoformat(),
                            "original_energy_mwh": float(original_energy),
                            "fms_energy_mwh": float(fms_energy),
                            "energy_delta_mwh": float(fms_energy - original_energy),
                            "energy_pct_change": (
                                100.0 * (fms_energy - original_energy) / original_energy
                                if original_energy > 0
                                else 0.0
                            ),
                            "original_max_tws_mps": float(original_max_tws),
                            "fms_max_tws_mps": float(fms_max_tws),
                            "original_max_hs_m": float(original_max_hs),
                            "fms_max_hs_m": float(fms_max_hs),
                            "original_land_violations": int(original_land_violations),
                            "fms_land_violations": int(fms_land_violations),
                        }
                    )

                    if not quiet:
                        typer.echo(
                            f"[{case_id}] {idx}/{len(track_paths)} "
                            f"{departure.strftime('%Y-%m-%d')} "
                            f"original={original_energy:.1f} MWh  "
                            f"fms={fms_energy:.1f} MWh  "
                            f"delta={fms_energy - original_energy:+.1f} MWh"
                        )

                # Write summary CSV for this case
                write_file_a(output_rows, output_dir / f"IEUniversity-{case_id}.csv")
                comparison_tables[case_id] = comparison_rows

            finally:
                if resources is not None:
                    del resources
                    _release_fms_state()

        return output_dir, comparison_tables

    finally:
        # Clean up temporary files
        shutil.rmtree(temp_dir, ignore_errors=True)


def _save_comparison_summary(
    comparison_tables: dict[str, list[dict[str, Any]]],
    output_dir: Path,
) -> None:
    """Save overall comparison summary as JSON."""
    summary = {}

    for case_id, rows in comparison_tables.items():
        if not rows:
            continue

        energy_delta_list = [
            row["energy_delta_mwh"] for row in rows if "energy_delta_mwh" in row
        ]
        energy_pct_list = [
            row["energy_pct_change"] for row in rows if "energy_pct_change" in row
        ]

        original_energy = sum(
            row["original_energy_mwh"] for row in rows if "original_energy_mwh" in row
        )
        fms_energy = sum(
            row["fms_energy_mwh"] for row in rows if "fms_energy_mwh" in row
        )

        summary[case_id] = {
            "num_departures": len(rows),
            "total_original_energy_mwh": round(original_energy, 2),
            "total_fms_energy_mwh": round(fms_energy, 2),
            "total_energy_delta_mwh": round(fms_energy - original_energy, 2),
            "avg_energy_delta_mwh": (
                round(sum(energy_delta_list) / len(energy_delta_list), 2)
                if energy_delta_list
                else 0.0
            ),
            "avg_energy_pct_change": (
                round(sum(energy_pct_list) / len(energy_pct_list), 2)
                if energy_pct_list
                else 0.0
            ),
            "min_energy_delta_mwh": (
                round(min(energy_delta_list), 2) if energy_delta_list else 0.0
            ),
            "max_energy_delta_mwh": (
                round(max(energy_delta_list), 2) if energy_delta_list else 0.0
            ),
        }

    summary_path = output_dir / "fms_comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))


@app.command()
def main(
    archive_paths: list[Path] = typer.Argument(  # noqa: B008
        ..., help="Scored submission archive zip file(s)."
    ),
    output_base: Path | None = typer.Option(  # noqa: B008
        None,
        "--output-base",
        "-o",
        help=(
            "Base output directory. For each submission, a subdirectory is created "
            "with the participant name followed by _fms. If omitted, outputs are "
            "stored in the same directory as the submission archive."
        ),
    ),
    output_dir: Path | None = typer.Option(  # noqa: B008
        None,
        "--output-dir",
        help=(
            "Explicit output directory (used only for single archive). "
            "If both are provided, output-dir takes precedence for single archive."
        ),
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
    wind_penalty_weight: float = typer.Option(  # noqa: B008
        _DEFAULT_WIND_PENALTY_WEIGHT,
        "--wind-penalty-weight",
        help="Penalty weight for wind violations in the RISE cost function.",
    ),
    wave_penalty_weight: float = typer.Option(  # noqa: B008
        _DEFAULT_WAVE_PENALTY_WEIGHT,
        "--wave-penalty-weight",
        help="Penalty weight for wave violations in the RISE cost function.",
    ),
    enforce_weather_limits: bool = typer.Option(  # noqa: B008
        False,
        "--enforce-weather-limits/--no-enforce-weather-limits",
        help=(
            "Reject FMS updates that newly violate the configured weather limits. "
            "Already-violating routes may keep moving so FMS can escape an "
            "initially infeasible route."
        ),
    ),
    quiet: bool = typer.Option(  # noqa: B008
        False,
        "--quiet",
        "-q",
        help="Suppress progress output.",
    ),
) -> None:
    """Apply FMS refinement to scored SWOPP3 submission archive(s)."""
    for archive_path in archive_paths:
        archive_path = Path(archive_path)
        if not archive_path.exists():
            typer.echo(f"Archive not found: {archive_path}")
            continue

        # Determine output directory for this archive
        participant_name = _participant_name_from_scored_zip(archive_path)
        fms_folder_name = f"{participant_name}_fms"

        if output_dir is not None and len(archive_paths) == 1:
            resolved_output_dir = Path(output_dir)
        elif output_base is not None:
            resolved_output_dir = Path(output_base) / fms_folder_name
        else:
            # Default: store FMS results in same directory as archive with _fms suffix
            # This allows them to be auto-discovered as competitor submissions by
            # swopp3_submission_compare.py
            resolved_output_dir = archive_path.parent / fms_folder_name

        if not quiet:
            typer.echo(f"\nProcessing {archive_path.name}...")

        result_dir, comparison_tables = apply_fms_to_scored_submission(
            archive_path,
            output_dir=resolved_output_dir,
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
            wind_penalty_weight=wind_penalty_weight,
            wave_penalty_weight=wave_penalty_weight,
            enforce_weather_limits=enforce_weather_limits,
            quiet=quiet,
        )

        if comparison_tables:
            _save_comparison_summary(comparison_tables, result_dir)

            # Save detailed comparison tables for each case
            for case_id, rows in comparison_tables.items():
                csv_path = result_dir / f"fms_comparison_{case_id}.csv"
                if rows:
                    keys = rows[0].keys()
                    with csv_path.open("w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=keys)
                        writer.writeheader()
                        writer.writerows(rows)

        if not quiet:
            typer.echo(f"Wrote results to {result_dir}")


if __name__ == "__main__":
    app()
