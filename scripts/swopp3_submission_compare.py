#!/usr/bin/env python
# ruff: noqa: E402
"""Compare SWOPP3 submissions and generate analysis outputs.

This script scans a root folder containing one subfolder per submission,
validates each candidate against SWOPP3 output structure requirements, and
generates the comparison outputs requested in the SWOPP3 review comments:

1. Consumption-over-departures line charts (per corridor/config).
2. Spread comparison between participants (per corridor/config).
3. Spread comparison between months (per corridor/config).
4. Hourly animation for one selected departure and case.

Expected submission structure
-----------------------------
Each valid submission folder must contain:

- Summary CSV files for the four optimised cases:
  ``AO_WPS``, ``AO_noWPS``, ``PO_WPS``, ``PO_noWPS``.
- A ``tracks/`` subdirectory.
- Every summary row must reference an existing track CSV inside ``tracks/``.
- Each summary CSV must have 366 departures for year 2024.

Usage
-----
    uv run scripts/swopp3_submission_compare.py \
        --input-root output \
        --output-dir output/analysis/submission_compare
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import struct
import tempfile
import zipfile
import zlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import jax

jax.config.update("jax_platform_name", "cpu")

import matplotlib.animation as animation
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import colors as mcolors
from PIL import Image, ImageSequence

from routetools.swopp3 import SWOPP3_CASES
from routetools.violations import find_team_prefix

REQUIRED_CASES = ["AO_WPS", "AO_noWPS", "PO_WPS", "PO_noWPS"]

CASE_LABELS = {
    "AO_WPS": "Atlantic / WPS",
    "AO_noWPS": "Atlantic / no WPS",
    "PO_WPS": "Pacific / WPS",
    "PO_noWPS": "Pacific / no WPS",
}

SUMMARY_COLUMNS = {
    "departure_time_utc",
    "arrival_time_utc",
    "energy_cons_mwh",
    "max_wind_mps",
    "max_hs_m",
    "sailed_distance_nm",
    "details_filename",
}

MONTH_NAMES = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]

SPREAD_SAMPLE_COUNT = 10
ANIMATION_BASE_FPS = 8
ANIMATION_SPEEDUP = 4
ANIMATION_FPS = ANIMATION_BASE_FPS * ANIMATION_SPEEDUP
PARTICIPANT_TABLE_DIRNAME = "participant_tables"
IDENTITY_MAP_FILENAME = "participant_identity.json"
SCORE_ARCHIVE_ROOT = Path("output/swopp3_submissions_score")
TOP5_REQUESTED_TOKENS = ["drprecious", "ohy", "boatface", "freol", "jung_tpn"]


@dataclass(frozen=True)
class SubmissionData:
    """Parsed and validated SWOPP3 submission."""

    name: str
    path: Path
    source_label: str
    team_prefix: str
    tracks_dir: Path
    summaries: dict[str, pd.DataFrame]


@dataclass(frozen=True)
class CandidateIssue:
    """Validation issue found while scanning submission folders."""

    folder: str
    reason: str


def _alias_label(index: int) -> str:
    """Return spreadsheet-style alias labels: A..Z, AA..AZ, ..."""
    if index < 0:
        raise ValueError("alias index must be non-negative")

    label: list[str] = []
    n = index
    while True:
        n, rem = divmod(n, 26)
        label.append(chr(ord("A") + rem))
        if n == 0:
            break
        n -= 1
    return "".join(reversed(label))


def _assign_participant_identity(
    submissions: list[SubmissionData],
    *,
    out_dir: Path,
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Load/create alias+color assignment and persist it as JSON.

    Returns
    -------
    tuple[dict[str, str], dict[str, str], dict[str, str]]
        ``name_to_alias``, ``alias_to_name``, ``alias_to_color``.
    """
    mapping_path = out_dir / IDENTITY_MAP_FILENAME
    names = sorted(sub.name for sub in submissions)

    alias_to_name: dict[str, str] = {}
    alias_to_color: dict[str, str] = {}
    if mapping_path.exists():
        try:
            payload = json.loads(mapping_path.read_text())
            alias_to_name = {
                str(alias): str(name)
                for alias, name in payload.get("letters_to_names", {}).items()
            }
            alias_to_color = {
                str(alias): str(color)
                for alias, color in payload.get("letters_to_colors", {}).items()
            }
        except Exception:  # noqa: BLE001
            alias_to_name = {}
            alias_to_color = {}

    used_aliases = set(alias_to_name)
    used_names = set(alias_to_name.values())
    missing_names = [name for name in names if name not in used_names]

    # Fixed palette order keeps participant colors stable and deterministic.
    palette = [mcolors.to_hex(c) for c in plt.get_cmap("tab20").colors]
    next_index = 0
    while missing_names:
        alias = _alias_label(next_index)
        next_index += 1
        if alias in used_aliases:
            continue

        name = missing_names.pop(0)
        alias_to_name[alias] = name
        alias_to_color[alias] = palette[(next_index - 1) % len(palette)]
        used_aliases.add(alias)

    # Keep only participants in this run.
    alias_to_name = {
        alias: name for alias, name in alias_to_name.items() if name in set(names)
    }
    alias_to_color = {
        alias: alias_to_color.get(alias, palette[idx % len(palette)])
        for idx, alias in enumerate(sorted(alias_to_name))
    }

    name_to_alias = {name: alias for alias, name in alias_to_name.items()}
    payload = {
        "letters_to_names": {
            alias: alias_to_name[alias] for alias in sorted(alias_to_name)
        },
        "letters_to_colors": {
            alias: alias_to_color[alias] for alias in sorted(alias_to_name)
        },
        "names_to_letters": {
            name: name_to_alias[name] for name in sorted(name_to_alias)
        },
    }
    mapping_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    return name_to_alias, alias_to_name, alias_to_color


def _update_participant_identity_top5(
    *,
    out_dir: Path,
    top5_requested: list[str],
    top5_participants: list[str],
    alias_by_name: dict[str, str],
) -> None:
    """Persist the top-5 participant lists alongside the alias/color mapping."""
    mapping_path = out_dir / IDENTITY_MAP_FILENAME
    payload = json.loads(mapping_path.read_text())
    payload["top_5_requested"] = top5_requested
    payload["top_5_participants"] = top5_participants
    payload["top_5_aliases"] = [alias_by_name[name] for name in top5_participants]
    mapping_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _participant_table_columns() -> list[str]:
    """Return expected columns for per-participant cached departure tables."""
    cols = ["departure_time_utc"]
    for case in REQUIRED_CASES:
        cols.extend([f"{case}_energy_mwh", f"{case}_violations"])
    cols.extend(["total_consumption_mwh", "total_violations"])
    return cols


def _extract_submission_id(source_label: str) -> int | None:
    """Extract numeric submission id from a source file/folder label."""
    match = re.match(r"^(\d+)_", source_label)
    if match is None:
        return None
    return int(match.group(1))


def _extract_submission_datetime(source_label: str, source_path: Path) -> datetime:
    """Extract submission datetime from source label, with mtime fallback."""
    fallback = datetime.fromtimestamp(source_path.stat().st_mtime, tz=UTC)

    date_match = re.search(r"_(\d{4}-\d{2}-\d{2})_", source_label)
    if date_match is None:
        return fallback

    date_text = date_match.group(1)
    base_date = datetime.strptime(date_text, "%Y-%m-%d").replace(tzinfo=UTC)

    time_match = re.search(
        rf"_{re.escape(date_text)}_(\d{{2}})-(\d{{2}})", source_label
    )
    if time_match is not None:
        hour = int(time_match.group(1))
        minute = int(time_match.group(2))
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return base_date.replace(hour=hour, minute=minute)

    return base_date.replace(hour=fallback.hour, minute=fallback.minute)


def _participant_table_path(out_dir: Path, alias: str) -> Path:
    """Return CSV cache file for one participant alias."""
    return out_dir / PARTICIPANT_TABLE_DIRNAME / f"participant_{alias}.csv"


def _normalize_participant_token(value: str) -> str:
    """Return a lowercase token used to match top-5 participant labels."""
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _resolve_requested_top5_participants(
    submissions: list[SubmissionData],
) -> list[str]:
    """Resolve the requested top-5 shorthand tokens to participant names."""
    names = [submission.name for submission in submissions]
    normalized_names = {name: _normalize_participant_token(name) for name in names}

    resolved: list[str] = []
    for token in TOP5_REQUESTED_TOKENS:
        normalized_token = _normalize_participant_token(token)
        matches = [
            name
            for name, normalized_name in normalized_names.items()
            if normalized_token in normalized_name
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Could not resolve top-5 token {token!r} to a unique participant"
            )
        resolved.append(matches[0])

    return resolved


def _ordered_submission_names(
    submissions: list[SubmissionData],
    *,
    alias_by_name: dict[str, str],
    selected_names: set[str] | None = None,
) -> list[str]:
    """Return participant names ordered by alias for stable alphabetical legends."""
    names = [
        submission.name
        for submission in submissions
        if selected_names is None or submission.name in selected_names
    ]
    return sorted(names, key=lambda name: (alias_by_name[name], name))


def _normalize_departure_key(value: object) -> pd.Timestamp:
    """Return timezone-aware UTC timestamp used for dictionary keys."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _extract_zip_member_bytes(archive_path: Path, member_name: str) -> bytes:
    """Return decompressed bytes for a ZIP member from a raw ZIP stream."""
    data = archive_path.read_bytes()
    member_bytes = member_name.encode("utf-8")
    member_idx = data.find(member_bytes)
    if member_idx < 0:
        raise KeyError(f"{member_name} not found in {archive_path.name}")

    header_idx = data.rfind(b"PK\x03\x04", 0, member_idx)
    if header_idx < 0:
        raise ValueError(f"Could not locate ZIP header for {member_name}")

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
        return zlib.decompress(payload, -15)
    raise ValueError(
        f"Unsupported ZIP compression method {compression} in {member_name}"
    )


def _load_resampled_tracks_zip(score_archive: Path) -> zipfile.ZipFile:
    """Open the embedded resampled tracks ZIP from a scored submission archive."""
    resampled_zip_bytes = _extract_zip_member_bytes(
        score_archive, "resampled_tracks.zip"
    )
    return zipfile.ZipFile(io.BytesIO(resampled_zip_bytes))


def _index_scored_route_members(
    score_zip: zipfile.ZipFile,
) -> dict[str, dict[str, str]]:
    """Map case/date pairs to resampled route CSV members."""
    route_members: dict[str, dict[str, str]] = {case: {} for case in REQUIRED_CASES}
    member_names = score_zip.namelist()
    for case in REQUIRED_CASES:
        for member_name in member_names:
            if not member_name.startswith(f"resampled/{case}/"):
                continue

            basename = Path(member_name).name
            compact_match = re.search(r"(\d{8})(?=\.csv$)", basename)
            if compact_match is not None:
                route_members[case][compact_match.group(1)] = member_name
                continue

            iso_match = re.search(r"(\d{4})-(\d{2})-(\d{2})T\d{2}(?=\.csv$)", basename)
            if iso_match is not None:
                route_members[case]["".join(iso_match.groups())] = member_name
                continue

            date_match = re.search(r"(\d{4}-\d{2}-\d{2})(?=\.csv$)", basename)
            if date_match is not None:
                route_members[case][date_match.group(1).replace("-", "")] = member_name
    return route_members


def _discover_score_archives(root: Path) -> dict[str, Path]:
    """Map participant display names to scored submission archives."""
    score_archives: dict[str, Path] = {}
    if not root.exists():
        raise FileNotFoundError(f"Score archive root does not exist: {root}")

    for entry in sorted(root.iterdir()):
        if not entry.is_file():
            continue
        display_name = _participant_name_from_submission_id(entry.stem)
        score_archives[display_name] = entry
    return score_archives


def _save_figure(fig: plt.Figure, path: Path, dpi: int) -> None:
    """Save a figure as PDF and PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")


def _gif_to_mp4(gif_path: Path, mp4_path: Path, fps: int) -> None:
    """Convert GIF to MP4 using OpenCV.

    This fallback is used when ffmpeg is unavailable in the environment.
    """
    try:
        import cv2
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "MP4 export requires ffmpeg or opencv-python for GIF->MP4 fallback"
        ) from exc

    with Image.open(gif_path) as im:
        frames = [frame.convert("RGB") for frame in ImageSequence.Iterator(im)]

    if not frames:
        raise RuntimeError(f"No frames found in GIF: {gif_path}")

    first = np.array(frames[0])
    height, width, _ = first.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(mp4_path), fourcc, float(fps), (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Could not open MP4 writer for {mp4_path}")

    try:
        for frame in frames:
            rgb = np.array(frame)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            writer.write(bgr)
    finally:
        writer.release()


def _read_summary_csv(path: Path) -> pd.DataFrame:
    """Read and normalize one SWOPP3 summary CSV."""
    df = pd.read_csv(path)
    missing = SUMMARY_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)} in {path.name}")

    df = df.copy()
    df["departure_time_utc"] = pd.to_datetime(df["departure_time_utc"], utc=True)
    df["arrival_time_utc"] = pd.to_datetime(df["arrival_time_utc"], utc=True)
    df["energy_cons_mwh"] = pd.to_numeric(df["energy_cons_mwh"], errors="coerce")
    if df["energy_cons_mwh"].isna().any():
        raise ValueError(f"Invalid numeric energy values in {path.name}")

    return df.sort_values("departure_time_utc").reset_index(drop=True)


def _is_2024_departure_series(series: pd.Series) -> bool:
    """Return whether all datetimes in the series belong to year 2024."""
    return bool((series.dt.year == 2024).all())


def _participant_name_from_submission_id(raw_name: str) -> str:
    """Convert official submission id to a participant display name.

    Expected pattern resembles ``XXXXXX_participant_name_PhaseId...``.
    Falls back to the original stem when no match is found.
    """
    stem = Path(raw_name).stem
    match = re.match(r"^\d+_(.+?)_PhaseId.*$", stem, flags=re.IGNORECASE)
    participant = match.group(1) if match else stem
    return participant.replace("_", " ").strip() or stem


def _find_submission_root(candidate_root: Path) -> Path:
    """Return the folder that directly contains submission CSVs and tracks/."""
    if (candidate_root / "tracks").is_dir():
        return candidate_root

    track_dirs = [p for p in candidate_root.rglob("tracks") if p.is_dir()]
    roots = sorted({p.parent for p in track_dirs})
    if not roots:
        raise ValueError("missing tracks/ directory")
    if len(roots) > 1:
        raise ValueError("multiple candidate submission folders found after extraction")
    return roots[0]


def _discover_submission_candidates(
    root: Path,
    extraction_root: Path,
) -> list[tuple[str, str, Path]]:
    """Discover folder/zip submission candidates.

    Returns tuples ``(display_name, source_label, submission_path)``.
    """
    candidates: list[tuple[str, str, Path]] = []

    for entry in sorted(root.iterdir()):
        if entry.is_dir():
            display_name = _participant_name_from_submission_id(entry.name)
            candidates.append((display_name, entry.name, entry))
            continue

        if entry.is_file() and entry.suffix.lower() == ".zip":
            extracted_dir = extraction_root / entry.stem
            extracted_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(entry) as zf:
                zf.extractall(extracted_dir)
            submission_path = _find_submission_root(extracted_dir)
            display_name = _participant_name_from_submission_id(entry.stem)
            candidates.append((display_name, entry.name, submission_path))

    return candidates


def scan_submissions(
    root: Path,
    *,
    required_cases: list[str],
    expected_departures: int,
    extraction_root: Path,
) -> tuple[list[SubmissionData], list[CandidateIssue]]:
    """Scan root folder and return valid submissions and rejected candidates."""
    submissions: list[SubmissionData] = []
    issues: list[CandidateIssue] = []

    if not root.exists():
        raise FileNotFoundError(f"Input root does not exist: {root}")

    used_names: dict[str, int] = {}
    discovered = _discover_submission_candidates(root, extraction_root)

    for display_name, source_label, candidate in discovered:
        try:
            tracks_dir = candidate / "tracks"
            if not tracks_dir.exists() or not tracks_dir.is_dir():
                raise ValueError("missing tracks/ directory")

            # Ensure legend/cache keys remain unique when two archives share names.
            occurrence = used_names.get(display_name, 0) + 1
            used_names[display_name] = occurrence
            if occurrence > 1:
                display_name = f"{display_name} ({occurrence})"

            team_prefix = find_team_prefix(candidate)
            summaries: dict[str, pd.DataFrame] = {}

            for case in required_cases:
                summary_path = candidate / f"{team_prefix}-{case}.csv"
                if not summary_path.exists():
                    raise ValueError(f"missing summary file for case {case}")

                summary = _read_summary_csv(summary_path)
                if len(summary) != expected_departures:
                    raise ValueError(
                        f"case {case} has {len(summary)} rows, expected "
                        f"{expected_departures}"
                    )
                if summary["departure_time_utc"].nunique() != expected_departures:
                    raise ValueError(f"case {case} has duplicated departures")
                if not _is_2024_departure_series(summary["departure_time_utc"]):
                    raise ValueError(f"case {case} includes non-2024 departures")

                missing_tracks = [
                    fname
                    for fname in summary["details_filename"].astype(str)
                    if not (tracks_dir / fname).exists()
                ]
                if missing_tracks:
                    raise ValueError(
                        f"case {case} references missing track files "
                        f"(example: {missing_tracks[0]})"
                    )

                summaries[case] = summary

            submissions.append(
                SubmissionData(
                    name=display_name,
                    path=candidate,
                    source_label=source_label,
                    team_prefix=team_prefix,
                    tracks_dir=tracks_dir,
                    summaries=summaries,
                )
            )

        except Exception as exc:  # noqa: BLE001
            issues.append(CandidateIssue(folder=source_label, reason=str(exc)))

    return submissions, issues


def _read_track(path: Path) -> pd.DataFrame:
    """Read a SWOPP3 track CSV and return time-sorted data."""
    track = pd.read_csv(path)
    required = {"time_utc", "lat_deg", "lon_deg"}
    if not required.issubset(track.columns):
        raise ValueError(f"Track file {path.name} missing required columns")

    track = track.copy()
    track["time_utc"] = pd.to_datetime(track["time_utc"], utc=True)
    track["lat_deg"] = pd.to_numeric(track["lat_deg"], errors="coerce")
    track["lon_deg"] = pd.to_numeric(track["lon_deg"], errors="coerce")
    track = track.dropna(subset=["time_utc", "lat_deg", "lon_deg"])

    return track.sort_values("time_utc").reset_index(drop=True)


def _curve_lonlat_from_track(track: pd.DataFrame) -> np.ndarray:
    """Return route waypoints as ``(lon, lat)`` float array."""
    return np.column_stack(
        (
            track["lon_deg"].to_numpy(dtype=np.float32),
            track["lat_deg"].to_numpy(dtype=np.float32),
        )
    )


def _build_or_load_participant_departure_table(
    submission: SubmissionData,
    *,
    alias: str,
    out_dir: Path,
    score_archive: Path,
) -> pd.DataFrame:
    """Load cached table for one participant or build it from scored CSVs."""
    table_path = _participant_table_path(out_dir, alias)
    expected_columns = _participant_table_columns()
    departures_union = pd.DatetimeIndex(
        pd.concat(
            [
                submission.summaries[case]["departure_time_utc"]
                for case in REQUIRED_CASES
            ],
            ignore_index=True,
        )
        .drop_duplicates()
        .sort_values()
    )

    departures = departures_union
    rows_by_case: dict[str, dict[pd.Timestamp, pd.Series]] = {
        case: {
            _normalize_departure_key(row["departure_time_utc"]): row
            for _, row in submission.summaries[case]
            .sort_values("departure_time_utc")
            .iterrows()
        }
        for case in REQUIRED_CASES
    }

    score_zip = _load_resampled_tracks_zip(score_archive)
    route_members = _index_scored_route_members(score_zip)
    table = pd.DataFrame({"departure_time_utc": departures})
    for case in REQUIRED_CASES:
        table[f"{case}_energy_mwh"] = np.nan
        table[f"{case}_violations"] = 0

    try:
        for idx, departure in enumerate(departures):
            departure_key = _normalize_departure_key(departure)
            departure_date = departure_key.strftime("%Y%m%d")
            for case in REQUIRED_CASES:
                row = rows_by_case[case].get(departure_key)
                if row is None:
                    continue

                score_name = route_members[case].get(departure_date)
                if score_name is None:
                    raise FileNotFoundError(
                        f"Missing scored route for {case} on {departure_date} in "
                        f"{score_archive.name}"
                    )
                if score_name not in score_zip.namelist():
                    raise FileNotFoundError(
                        f"Missing scored route {score_name} in {score_archive.name}"
                    )

                with score_zip.open(score_name) as score_file:
                    scored_route = pd.read_csv(
                        score_file, usecols=["E", "land_violation"]
                    )

                table.at[idx, f"{case}_energy_mwh"] = float(scored_route["E"].sum())
                table.at[idx, f"{case}_violations"] = int(
                    scored_route["land_violation"].sum()
                )
    finally:
        score_zip.close()

    energy_columns = [f"{case}_energy_mwh" for case in REQUIRED_CASES]
    violation_columns = [f"{case}_violations" for case in REQUIRED_CASES]
    table["total_consumption_mwh"] = table[energy_columns].sum(axis=1, skipna=True)
    table["total_violations"] = table[violation_columns].sum(axis=1).astype(int)

    table_path.parent.mkdir(parents=True, exist_ok=True)
    table[expected_columns].to_csv(table_path, index=False, float_format="%.6f")
    return table[expected_columns]


def _load_or_build_participant_tables(
    submissions: list[SubmissionData],
    *,
    out_dir: Path,
    alias_by_name: dict[str, str],
    score_archives_by_name: dict[str, Path],
) -> dict[str, pd.DataFrame]:
    """Load or compute per-participant departure cache tables."""
    tables: dict[str, pd.DataFrame] = {}
    for submission in submissions:
        alias = alias_by_name[submission.name]
        score_archive = score_archives_by_name.get(submission.name)
        if score_archive is None:
            raise FileNotFoundError(
                f"Missing scored archive for participant {submission.name}"
            )
        table_path = _participant_table_path(out_dir, alias)
        if table_path.exists():
            print(f"Refreshing participant table for {alias} from scored routes")
        else:
            print(f"Computing participant table for {alias} from scored routes")
        tables[submission.name] = _build_or_load_participant_departure_table(
            submission,
            alias=alias,
            out_dir=out_dir,
            score_archive=score_archive,
        )
    return tables


def _build_lookup_from_tables(
    participant_tables: dict[str, pd.DataFrame],
) -> tuple[
    dict[str, dict[str, dict[pd.Timestamp, float]]],
    dict[str, dict[str, dict[pd.Timestamp, int]]],
]:
    """Convert participant cache tables into energy/violation lookup dicts."""
    participant_names = sorted(participant_tables)
    energy_lookup: dict[str, dict[str, dict[pd.Timestamp, float]]] = {
        case: {name: {} for name in participant_names} for case in REQUIRED_CASES
    }
    violation_lookup: dict[str, dict[str, dict[pd.Timestamp, int]]] = {
        case: {name: {} for name in participant_names} for case in REQUIRED_CASES
    }

    for name, table in participant_tables.items():
        for _, row in table.iterrows():
            departure = _normalize_departure_key(row["departure_time_utc"])
            for case in REQUIRED_CASES:
                energy_lookup[case][name][departure] = float(row[f"{case}_energy_mwh"])
                violation_lookup[case][name][departure] = int(row[f"{case}_violations"])

    return energy_lookup, violation_lookup


def build_overall_evaluation_table(
    submissions: list[SubmissionData],
    *,
    participant_tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Build SWOPP3 overall leaderboard table across all optimized cases."""
    rows: list[dict[str, object]] = []

    for submission in submissions:
        table = participant_tables[submission.name]
        total_energy = float(table["total_consumption_mwh"].sum())

        wind_viol_count = 0
        wave_viol_count = 0
        total_route_count = 0
        for case in REQUIRED_CASES:
            summary = submission.summaries[case]
            wind_viol_count += int((summary["max_wind_mps"] > 20.0).sum())
            wave_viol_count += int((summary["max_hs_m"] > 7.0).sum())
            total_route_count += int(len(summary))

        wind_ratio = (
            float(wind_viol_count) / float(total_route_count)
            if total_route_count > 0
            else np.nan
        )
        wave_ratio = (
            float(wave_viol_count) / float(total_route_count)
            if total_route_count > 0
            else np.nan
        )
        land_departures = int(
            sum(int((table[f"{case}_violations"] > 0).sum()) for case in REQUIRED_CASES)
        )

        submitted_at = _extract_submission_datetime(
            submission.source_label,
            submission.path,
        )
        submission_id = _extract_submission_id(submission.source_label)

        rows.append(
            {
                "Participant": submission.name,
                "Date": submitted_at.strftime("%Y-%m-%d %H:%M"),
                "ID": submission_id,
                "Total Energy (MWh)": round(total_energy, 4),
                "Wind Violation %": round(wind_ratio, 4),
                "Wave Violation %": round(wave_ratio, 4),
                "Land Departures": land_departures,
            }
        )

    table = pd.DataFrame(rows)
    table = table.sort_values(
        by=["Total Energy (MWh)", "Participant"],
        ascending=[True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    table.insert(0, "Rank", np.arange(1, len(table) + 1, dtype=int))

    ordered_columns = [
        "Rank",
        "Participant",
        "Date",
        "ID",
        "Total Energy (MWh)",
        "Wind Violation %",
        "Wave Violation %",
        "Land Departures",
    ]
    return table[ordered_columns]


def save_overall_evaluation_table(table: pd.DataFrame, *, out_dir: Path) -> Path:
    """Save overall SWOPP3 evaluation table as CSV."""
    out_path = out_dir / "swopp3_evaluation.csv"
    table.to_csv(out_path, index=False)
    return out_path


def _interp_track_lonlat(
    track: pd.DataFrame,
    elapsed_hours: float,
) -> tuple[float, float]:
    """Interpolate track position at a given elapsed hour."""
    if track.empty:
        raise ValueError("Cannot interpolate empty track")

    t0 = track["time_utc"].iloc[0]
    elapsed = (track["time_utc"] - t0).dt.total_seconds().to_numpy() / 3600.0
    lon = track["lon_deg"].to_numpy(dtype=float)
    lat = track["lat_deg"].to_numpy(dtype=float)

    elapsed_target = float(np.clip(elapsed_hours, elapsed.min(), elapsed.max()))
    lon_i = float(np.interp(elapsed_target, elapsed, lon))
    lat_i = float(np.interp(elapsed_target, elapsed, lat))
    return lon_i, lat_i


def _sample_waypoints_for_case(
    submission: SubmissionData,
    case: str,
    sample_hours: np.ndarray,
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Sample all departures for one case at selected elapsed hours.

    Returns
    -------
    tuple[pd.DatetimeIndex, np.ndarray]
        Sorted departures and an array with shape ``(D, S, 2)`` in
        ``(lon, lat)`` order.
    """
    summary = (
        submission.summaries[case]
        .sort_values("departure_time_utc")
        .reset_index(drop=True)
    )
    departures = pd.DatetimeIndex(summary["departure_time_utc"])
    sampled = np.empty((len(summary), len(sample_hours), 2), dtype=float)

    for dep_idx, row in summary.iterrows():
        track = _read_track(submission.tracks_dir / str(row["details_filename"]))
        for sample_idx, elapsed_h in enumerate(sample_hours):
            lon_i, lat_i = _interp_track_lonlat(track, elapsed_h)
            sampled[dep_idx, sample_idx, 0] = lon_i
            sampled[dep_idx, sample_idx, 1] = lat_i

    return departures, sampled


def _mean_pairwise_haversine_km(points_lonlat: np.ndarray) -> float:
    """Mean pairwise haversine distance in km for ``(N, 2)`` lon/lat points."""
    n_points = int(points_lonlat.shape[0])
    if n_points < 2:
        return 0.0

    lon = np.radians(points_lonlat[:, 0][:, None])
    lat = np.radians(points_lonlat[:, 1][:, None])
    dlon = lon - lon.T
    dlat = lat - lat.T

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat) * np.cos(lat.T) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(a)))
    dist_km = 6371.0 * c

    tri = np.triu_indices(n_points, k=1)
    return float(np.mean(dist_km[tri]))


def plot_consumption(
    submissions: list[SubmissionData],
    *,
    energy_lookup: dict[str, dict[str, dict[pd.Timestamp, float]]],
    alias_by_name: dict[str, str],
    color_by_alias: dict[str, str],
    out_dir: Path,
    dpi: int,
    selected_names: set[str] | None = None,
    output_suffix: str = "",
) -> None:
    """Plot non-penalized consumption time series for each corridor/config."""
    for case in REQUIRED_CASES:
        fig, ax = plt.subplots(figsize=(12, 5))
        for name in _ordered_submission_names(
            submissions,
            alias_by_name=alias_by_name,
            selected_names=selected_names,
        ):
            submission = next(sub for sub in submissions if sub.name == name)
            summary = submission.summaries[case].sort_values("departure_time_utc")
            departures = pd.DatetimeIndex(summary["departure_time_utc"])
            energies = [
                energy_lookup[case][submission.name][
                    _normalize_departure_key(departure)
                ]
                for departure in departures
            ]
            ax.plot(
                departures,
                energies,
                linewidth=1.4,
                alpha=0.9,
                label=alias_by_name[submission.name],
                color=color_by_alias[alias_by_name[submission.name]],
            )

        ax.set_title(f"Consumption Across 2024 Departures - {CASE_LABELS[case]}")
        ax.set_xlabel("Departure date")
        ax.set_ylabel("Energy consumption (MWh)")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", ncols=2, fontsize=9)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

        out = out_dir / f"consumption_{case}{output_suffix}.pdf"
        _save_figure(fig, out, dpi=dpi)
        plt.close(fig)


def plot_participant_spread(
    submissions: list[SubmissionData],
    *,
    case: str,
    sampled_cache: dict[tuple[str, str], tuple[pd.DatetimeIndex, np.ndarray]],
    sample_hours: np.ndarray,
    alias_by_name: dict[str, str],
    color_by_alias: dict[str, str],
    out_dir: Path,
    dpi: int,
    selected_names: set[str] | None = None,
    output_suffix: str = "",
) -> None:
    """Plot spread vs time for each participant and case."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for name in _ordered_submission_names(
        submissions,
        alias_by_name=alias_by_name,
        selected_names=selected_names,
    ):
        submission = next(sub for sub in submissions if sub.name == name)
        _, points = sampled_cache[(submission.name, case)]
        spreads = [
            _mean_pairwise_haversine_km(points[:, sample_idx, :])
            for sample_idx in range(points.shape[1])
        ]
        alias = alias_by_name[submission.name]
        ax.plot(
            sample_hours,
            spreads,
            marker="o",
            label=alias,
            color=color_by_alias[alias],
        )

    ax.set_title(f"Participant Spread Comparison - {CASE_LABELS[case]}")
    ax.set_xlabel("Elapsed time (hours)")
    ax.set_ylabel("Mean pairwise waypoint distance (km)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", ncols=2, fontsize=9)

    out = out_dir / f"spread_participants_{case}{output_suffix}.pdf"
    _save_figure(fig, out, dpi=dpi)
    plt.close(fig)


def plot_month_spread(
    submissions: list[SubmissionData],
    *,
    case: str,
    sampled_cache: dict[tuple[str, str], tuple[pd.DatetimeIndex, np.ndarray]],
    sample_hours: np.ndarray,
    out_dir: Path,
    dpi: int,
    selected_names: set[str] | None = None,
    output_suffix: str = "",
) -> None:
    """Plot cross-participant spread grouped by month for each case."""
    per_submission: dict[str, tuple[pd.DatetimeIndex, np.ndarray]] = {
        sub.name: sampled_cache[(sub.name, case)]
        for sub in submissions
        if selected_names is None or sub.name in selected_names
    }

    selected_order = list(per_submission)
    common_dates = set(per_submission[selected_order[0]][0])
    for name in selected_order[1:]:
        common_dates &= set(per_submission[name][0])
    aligned_dates = sorted(common_dates)

    if not aligned_dates:
        return

    date_to_row = {
        name: {dt: idx for idx, dt in enumerate(per_submission[name][0])}
        for name in selected_order
    }

    by_month: dict[int, list[np.ndarray]] = {month: [] for month in range(1, 13)}
    for date in aligned_dates:
        month = date.month
        curve = np.zeros(len(sample_hours), dtype=float)

        for sample_idx in range(len(sample_hours)):
            points = np.array(
                [
                    per_submission[name][1][
                        date_to_row[name][date],
                        sample_idx,
                        :,
                    ]
                    for name in selected_order
                ],
                dtype=float,
            )
            curve[sample_idx] = _mean_pairwise_haversine_km(points)

        by_month[month].append(curve)

    fig, ax = plt.subplots(figsize=(10, 5))
    for month in range(1, 13):
        if not by_month[month]:
            continue
        mean_curve = np.mean(np.vstack(by_month[month]), axis=0)
        ax.plot(sample_hours, mean_curve, marker="o", label=MONTH_NAMES[month - 1])

    ax.set_title(f"Month Spread Comparison - {CASE_LABELS[case]}")
    ax.set_xlabel("Elapsed time (hours)")
    ax.set_ylabel("Mean cross-participant distance (km)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", ncols=4, fontsize=8)

    out = out_dir / f"spread_months_{case}{output_suffix}.pdf"
    _save_figure(fig, out, dpi=dpi)
    plt.close(fig)


def _mean_haversine_km_between_curves(
    a_lonlat: np.ndarray,
    b_lonlat: np.ndarray,
) -> float:
    """Return mean haversine distance (km) between paired lon/lat points."""
    if a_lonlat.shape != b_lonlat.shape:
        raise ValueError("Mismatched curve shapes for distance comparison")
    if a_lonlat.size == 0:
        return 0.0

    lon1 = np.radians(a_lonlat[:, 0])
    lat1 = np.radians(a_lonlat[:, 1])
    lon2 = np.radians(b_lonlat[:, 0])
    lat2 = np.radians(b_lonlat[:, 1])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    h = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(h)))
    return float(np.mean(6371.0 * c))


def plot_wps_vs_nowps_spread(
    submissions: list[SubmissionData],
    *,
    sampled_cache: dict[tuple[str, str], tuple[pd.DatetimeIndex, np.ndarray]],
    sample_hours_by_case: dict[str, np.ndarray],
    alias_by_name: dict[str, str],
    color_by_alias: dict[str, str],
    out_dir: Path,
    dpi: int,
    selected_names: set[str] | None = None,
    output_suffix: str = "",
) -> None:
    """Plot WPS vs no-WPS route spread per ocean and participant."""
    case_pairs = [
        ("AO_WPS", "AO_noWPS", "atlantic"),
        ("PO_WPS", "PO_noWPS", "pacific"),
    ]

    for wps_case, nowps_case, corridor in case_pairs:
        sample_hours = sample_hours_by_case[wps_case]
        fig, ax = plt.subplots(figsize=(10, 5))

        for name in _ordered_submission_names(
            submissions,
            alias_by_name=alias_by_name,
            selected_names=selected_names,
        ):
            submission = next(sub for sub in submissions if sub.name == name)
            dep_wps, points_wps = sampled_cache[(submission.name, wps_case)]
            dep_nowps, points_nowps = sampled_cache[(submission.name, nowps_case)]

            common_dates = sorted(set(dep_wps).intersection(set(dep_nowps)))
            if not common_dates:
                continue

            wps_row = {dt: idx for idx, dt in enumerate(dep_wps)}
            nowps_row = {dt: idx for idx, dt in enumerate(dep_nowps)}

            mean_distances: list[float] = []
            for sample_idx in range(len(sample_hours)):
                curve_wps = np.array(
                    [points_wps[wps_row[date], sample_idx, :] for date in common_dates],
                    dtype=float,
                )
                curve_nowps = np.array(
                    [
                        points_nowps[nowps_row[date], sample_idx, :]
                        for date in common_dates
                    ],
                    dtype=float,
                )
                mean_distances.append(
                    _mean_haversine_km_between_curves(curve_wps, curve_nowps)
                )

            alias = alias_by_name[submission.name]
            ax.plot(
                sample_hours,
                mean_distances,
                marker="o",
                label=alias,
                color=color_by_alias[alias],
            )

        ax.set_title(f"WPS vs no-WPS Spread - {corridor.title()}")
        ax.set_xlabel("Elapsed time (hours)")
        ax.set_ylabel("Mean waypoint distance between WPS and no-WPS (km)")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", ncols=2, fontsize=9)

        out = out_dir / f"spread_wps_vs_nowps_{corridor}{output_suffix}.pdf"
        _save_figure(fig, out, dpi=dpi)
        plt.close(fig)


def _scenario_columns_index() -> pd.MultiIndex:
    """Return common (ocean, wps) multi-index columns for scenario tables."""
    return pd.MultiIndex.from_tuples(
        [
            ("atlantic", "WPS")
            if case == "AO_WPS"
            else ("atlantic", "no WPS")
            if case == "AO_noWPS"
            else ("pacific", "WPS")
            if case == "PO_WPS"
            else ("pacific", "no WPS")
            for case in REQUIRED_CASES
        ],
        names=["ocean", "wps"],
    )


def _summary_case_from_case(case: str) -> str:
    """Map optimized case ids to summary case ids used in scored archives."""
    if case == "AO_WPS":
        return "AO_WPS"
    if case == "AO_noWPS":
        return "AO_noWPS"
    if case == "PO_WPS":
        return "PO_WPS"
    if case == "PO_noWPS":
        return "PO_noWPS"
    raise ValueError(f"Unknown case: {case}")


def _total_column_index() -> pd.MultiIndex:
    """Return a shared multi-index label for total columns."""
    return pd.MultiIndex.from_tuples(
        [("all", "total")],
        names=["ocean", "wps"],
    )


def build_departure_wins_table(
    submissions: list[SubmissionData],
    *,
    energy_lookup: dict[str, dict[str, dict[pd.Timestamp, float]]],
    violation_lookup: dict[str, dict[str, dict[pd.Timestamp, int]]],
) -> pd.DataFrame:
    """Build per-scenario departure-win counts, excluding land-violating routes."""
    participants = sorted(sub.name for sub in submissions)
    wins = {
        participant: {case: 0 for case in REQUIRED_CASES}
        for participant in participants
    }

    def _departure_day(value: pd.Timestamp) -> pd.Timestamp:
        """Normalize a timestamp to its UTC day for daily winner accounting."""
        return _normalize_departure_key(value).normalize()

    energy_by_day: dict[str, dict[str, dict[pd.Timestamp, float]]] = {
        case: {name: {} for name in participants} for case in REQUIRED_CASES
    }
    violation_by_day: dict[str, dict[str, dict[pd.Timestamp, int]]] = {
        case: {name: {} for name in participants} for case in REQUIRED_CASES
    }

    for case in REQUIRED_CASES:
        for name in participants:
            for departure, energy in energy_lookup[case][name].items():
                day = _departure_day(departure)
                energy_by_day[case][name][day] = energy
            for departure, violations in violation_lookup[case][name].items():
                day = _departure_day(departure)
                violation_by_day[case][name][day] = violations

    for case in REQUIRED_CASES:
        all_departures: set[pd.Timestamp] = set()
        for participant in participants:
            all_departures |= set(energy_by_day[case][participant].keys())

        for departure in sorted(all_departures):
            valid_candidates: list[tuple[str, float]] = []
            for sub in submissions:
                violations = violation_by_day[case][sub.name].get(departure)
                if violations is None or violations > 0:
                    continue

                energy = energy_by_day[case][sub.name].get(departure)
                if energy is None or not np.isfinite(energy):
                    continue

                valid_candidates.append((sub.name, energy))

            if not valid_candidates:
                continue

            winner_name, _ = min(valid_candidates, key=lambda item: (item[1], item[0]))
            wins[winner_name][case] += 1

    table = pd.DataFrame.from_dict(wins, orient="index")[REQUIRED_CASES]
    table.index.name = "participant"
    table.columns = _scenario_columns_index()
    table["all", "total"] = table.sum(axis=1).astype(int)
    table = table.reindex(columns=table.columns[:-1].append(_total_column_index()))
    return table.sort_index()


def save_departure_wins_table(table: pd.DataFrame, *, out_dir: Path, dpi: int) -> None:
    """Save departure-win table as CSV and table figure."""
    out_csv = out_dir / "departure_wins_by_scenario.csv"
    table.to_csv(out_csv)

    fig_height = max(3.2, 0.45 * len(table) + 1.6)
    fig, ax = plt.subplots(figsize=(8.5, fig_height))
    ax.axis("off")
    ax.set_title(
        "Departures Won by Participant (land-violation-free routes only)",
        fontsize=11,
        pad=12,
    )

    col_labels = [f"{ocean}\n{wps}" for ocean, wps in table.columns.to_flat_index()]
    cell_text = table.astype(int).values.tolist()
    row_labels = list(table.index)

    tab = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    tab.auto_set_font_size(False)
    tab.set_fontsize(9)
    tab.scale(1.0, 1.3)

    out = out_dir / "departure_wins_by_scenario.pdf"
    _save_figure(fig, out, dpi=dpi)
    plt.close(fig)


def build_average_consumption_table(
    submissions: list[SubmissionData],
    *,
    energy_lookup: dict[str, dict[str, dict[pd.Timestamp, float]]],
    violation_lookup: dict[str, dict[str, dict[pd.Timestamp, int]]],
) -> pd.DataFrame:
    """Build per-scenario average consumption, excluding land-violating routes."""
    participants = sorted(sub.name for sub in submissions)
    sums = {
        participant: {case: 0.0 for case in REQUIRED_CASES}
        for participant in participants
    }
    counts = {
        participant: {case: 0 for case in REQUIRED_CASES}
        for participant in participants
    }

    rows_by_case_by_sub: dict[str, dict[str, dict[pd.Timestamp, pd.Series]]] = {
        case: {
            sub.name: {
                _normalize_departure_key(row["departure_time_utc"]): row
                for _, row in sub.summaries[case]
                .sort_values("departure_time_utc")
                .reset_index(drop=True)
                .iterrows()
            }
            for sub in submissions
        }
        for case in REQUIRED_CASES
    }

    for case in REQUIRED_CASES:
        common_departures = set(rows_by_case_by_sub[case][participants[0]].keys())
        for participant in participants[1:]:
            common_departures &= set(rows_by_case_by_sub[case][participant].keys())

        for departure in sorted(common_departures):
            departure_key = _normalize_departure_key(departure)
            for sub in submissions:
                if violation_lookup[case][sub.name][departure_key] > 0:
                    continue

                energy = energy_lookup[case][sub.name][departure_key]
                sums[sub.name][case] += energy
                counts[sub.name][case] += 1

    averages = {
        participant: {
            case: (
                sums[participant][case] / counts[participant][case]
                if counts[participant][case] > 0
                else np.nan
            )
            for case in REQUIRED_CASES
        }
        for participant in participants
    }

    table = pd.DataFrame.from_dict(averages, orient="index")[REQUIRED_CASES]
    table.index.name = "participant"
    table.columns = _scenario_columns_index()
    table["all", "total"] = table.mean(axis=1, skipna=True)
    table = table.reindex(columns=table.columns[:-1].append(_total_column_index()))
    return table.sort_index()


def save_average_consumption_table(
    table: pd.DataFrame,
    *,
    out_dir: Path,
    dpi: int,
) -> None:
    """Save average-consumption table as CSV and figure."""
    out_csv = out_dir / "average_consumption_by_scenario.csv"
    table.to_csv(out_csv, float_format="%.3f")

    fig_height = max(3.2, 0.45 * len(table) + 1.6)
    fig, ax = plt.subplots(figsize=(8.5, fig_height))
    ax.axis("off")
    ax.set_title(
        "Average Consumption by Scenario (land-violation-free routes only)",
        fontsize=11,
        pad=12,
    )

    col_labels = [f"{ocean}\n{wps}" for ocean, wps in table.columns.to_flat_index()]
    cell_text = [
        ["-" if pd.isna(value) else f"{value:.2f}" for value in row]
        for row in table.to_numpy(dtype=float)
    ]
    row_labels = list(table.index)

    tab = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    tab.auto_set_font_size(False)
    tab.set_fontsize(9)
    tab.scale(1.0, 1.3)

    out = out_dir / "average_consumption_by_scenario.pdf"
    _save_figure(fig, out, dpi=dpi)
    plt.close(fig)


def _pick_data_var(dataset: xr.Dataset, candidates: list[str]) -> str:
    """Return the first available variable in dataset from a candidate list."""
    for candidate in candidates:
        if candidate in dataset.data_vars:
            return candidate
    raise KeyError(
        f"None of variables {candidates} found in dataset vars "
        f"{list(dataset.data_vars)}"
    )


def _pick_coord(dataset: xr.Dataset, candidates: list[str]) -> str:
    """Return first available coordinate/dimension name."""
    for candidate in candidates:
        if candidate in dataset.coords or candidate in dataset.dims:
            return candidate
    raise KeyError(
        f"None of coordinates {candidates} found in dataset coords "
        f"{list(dataset.coords)}"
    )


def _normalize_longitude(ds: xr.Dataset, lon_name: str) -> xr.Dataset:
    """Normalize longitude axis to [-180, 180) when needed."""
    lon = ds[lon_name]
    lon_vals = lon.values
    if np.nanmax(lon_vals) > 180.0:
        lon_shift = ((lon + 180.0) % 360.0) - 180.0
        ds = ds.assign_coords({lon_name: lon_shift}).sortby(lon_name)
    return ds


def _normalize_longitude_360(ds: xr.Dataset, lon_name: str) -> xr.Dataset:
    """Normalize longitude axis to [0, 360) and sort coordinates."""
    lon = ds[lon_name]
    lon_shift = lon % 360.0
    return ds.assign_coords({lon_name: lon_shift}).sortby(lon_name)


def _track_to_360(track: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of track data with longitudes wrapped to [-180, 180]."""
    normalized = track.copy()
    normalized["lon_deg"] = ((normalized["lon_deg"] + 180.0) % 360.0) - 180.0
    return normalized


def _slice_2d_region(
    ds: xr.Dataset,
    *,
    lat_name: str,
    lon_name: str,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> xr.Dataset:
    """Slice a dataset to a latitude/longitude box, handling axis ordering."""
    lat = ds[lat_name]
    lon = ds[lon_name]

    lat_slice = (
        slice(lat_min, lat_max)
        if float(lat.values[0]) <= float(lat.values[-1])
        else slice(lat_max, lat_min)
    )
    lon_slice = (
        slice(lon_min, lon_max)
        if float(lon.values[0]) <= float(lon.values[-1])
        else slice(lon_max, lon_min)
    )
    return ds.sel({lat_name: lat_slice, lon_name: lon_slice})


def _track_state_at_hour(
    track: pd.DataFrame,
    hour: float,
) -> tuple[float, float, float, float]:
    """Interpolate vessel state at an hour: lon, lat, dlon_dt, dlat_dt."""
    if track.empty:
        raise ValueError("Cannot sample empty track")

    t0 = track["time_utc"].iloc[0]
    elapsed = (track["time_utc"] - t0).dt.total_seconds().to_numpy() / 3600.0
    lon = track["lon_deg"].to_numpy(dtype=float)
    lat = track["lat_deg"].to_numpy(dtype=float)

    hour = float(np.clip(hour, elapsed.min(), elapsed.max()))
    lon_i = float(np.interp(hour, elapsed, lon))
    lat_i = float(np.interp(hour, elapsed, lat))

    delta = 0.5
    h0 = float(np.clip(hour - delta, elapsed.min(), elapsed.max()))
    h1 = float(np.clip(hour + delta, elapsed.min(), elapsed.max()))
    lon0 = float(np.interp(h0, elapsed, lon))
    lat0 = float(np.interp(h0, elapsed, lat))
    lon1 = float(np.interp(h1, elapsed, lon))
    lat1 = float(np.interp(h1, elapsed, lat))

    return lon_i, lat_i, lon1 - lon0, lat1 - lat0


def animate_departure(
    submissions: list[SubmissionData],
    *,
    case: str,
    departure: datetime,
    energy_lookup: dict[str, dict[str, dict[pd.Timestamp, float]]],
    alias_by_name: dict[str, str],
    color_by_alias: dict[str, str],
    output_path: Path,
    wave_path: Path,
    wind_path: Path,
    dpi: int,
) -> None:
    """Generate the requested hourly animation for one case/departure."""
    departure_utc = pd.Timestamp(departure, tz="UTC")
    case_hours = int(SWOPP3_CASES[case]["passage_hours"])
    frame_hours = np.arange(0, case_hours + 1, 1, dtype=int)

    tracks_by_submission: dict[str, pd.DataFrame] = {}
    total_energy: dict[str, float] = {}
    for sub in submissions:
        summary = sub.summaries[case]
        row = summary.loc[summary["departure_time_utc"] == departure_utc]
        if row.empty:
            continue
        details = str(row.iloc[0]["details_filename"])
        track = _read_track(sub.tracks_dir / details)
        tracks_by_submission[sub.name] = track
        total_energy[sub.name] = energy_lookup[case][sub.name][departure_utc]

    if not tracks_by_submission:
        raise ValueError(
            "No submissions contain departure "
            f"{departure_utc.isoformat()} for case {case}"
        )

    all_lons = np.concatenate(
        [
            track["lon_deg"].to_numpy(dtype=float)
            for track in tracks_by_submission.values()
        ]
    )
    all_lats = np.concatenate(
        [
            track["lat_deg"].to_numpy(dtype=float)
            for track in tracks_by_submission.values()
        ]
    )
    lon_min = float(np.min(all_lons) - 5.0)
    lon_max = float(np.max(all_lons) + 5.0)
    lat_min = float(np.min(all_lats) - 5.0)
    lat_max = float(np.max(all_lats) + 5.0)

    wind_ds = xr.open_dataset(wind_path)
    wave_ds = xr.open_dataset(wave_path)
    try:
        wind_u = _pick_data_var(wind_ds, ["u10", "10m_u_component_of_wind", "U10"])
        wind_v = _pick_data_var(wind_ds, ["v10", "10m_v_component_of_wind", "V10"])
        wave_h = _pick_data_var(
            wave_ds,
            [
                "swh",
                "significant_height_of_combined_wind_waves_and_swell",
                "hs",
                "Hs",
            ],
        )

        wind_time = _pick_coord(wind_ds, ["valid_time", "time"])
        wind_lat = _pick_coord(wind_ds, ["latitude", "lat"])
        wind_lon = _pick_coord(wind_ds, ["longitude", "lon"])

        wave_time = _pick_coord(wave_ds, ["valid_time", "time"])
        wave_lat = _pick_coord(wave_ds, ["latitude", "lat"])
        wave_lon = _pick_coord(wave_ds, ["longitude", "lon"])

        wind_ds = _normalize_longitude(wind_ds, wind_lon)
        wave_ds = _normalize_longitude(wave_ds, wave_lon)

        wind_region = _slice_2d_region(
            wind_ds,
            lat_name=wind_lat,
            lon_name=wind_lon,
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max,
        )
        wave_region = _slice_2d_region(
            wave_ds,
            lat_name=wave_lat,
            lon_name=wave_lon,
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max,
        )

        fig = plt.figure(figsize=(14, 8))
        grid = fig.add_gridspec(2, 1, height_ratios=[6.2, 1.15], hspace=0.08)
        ax_map = fig.add_subplot(grid[0, 0])
        ax_bar = fig.add_subplot(grid[1, 0])
        ax_bar.set_facecolor("#f7f9fc")

        ax_map.set_xlim(lon_min, lon_max)
        ax_map.set_ylim(lat_min, lat_max)
        ax_map.set_facecolor("#0d2538")
        ax_map.set_xlabel("Longitude (deg)")
        ax_map.set_ylabel("Latitude (deg)")
        ax_map.grid(alpha=0.25, color="#9cb4c3", linestyle="--", linewidth=0.5)

        sub_names = _ordered_submission_names(
            submissions,
            alias_by_name=alias_by_name,
            selected_names=set(tracks_by_submission),
        )
        color_by_name = {
            name: color_by_alias[alias_by_name[name]] for name in sub_names
        }
        line_handles: dict[str, plt.Line2D] = {}
        vessel_handles: dict[str, plt.Line2D] = {}
        vessel_shadow_handles: dict[str, plt.Line2D] = {}
        for name in sub_names:
            (line,) = ax_map.plot(
                [],
                [],
                color=color_by_name[name],
                linewidth=2.0,
                alpha=0.92,
                solid_capstyle="round",
                label=alias_by_name[name],
            )
            line_handles[name] = line
            (shadow,) = ax_map.plot(
                [],
                [],
                linestyle="None",
                marker=(3, 0, 0),
                markersize=13,
                markerfacecolor="black",
                markeredgecolor="none",
                alpha=0.35,
                zorder=8,
            )
            vessel_shadow_handles[name] = shadow
            (vessel,) = ax_map.plot(
                [],
                [],
                linestyle="None",
                marker=(3, 0, 0),
                markersize=10,
                markerfacecolor=color_by_name[name],
                markeredgecolor="white",
                markeredgewidth=1.2,
                zorder=9,
            )
            vessel_handles[name] = vessel

        wave_mesh = None
        wind_quiver = None

        best_name = min(total_energy, key=total_energy.get)
        max_gap = max(total_energy.values()) - min(total_energy.values())
        x_max = max(1.0, max_gap * 1.05)

        def _frame(abs_hour: int):
            nonlocal wave_mesh, wind_quiver
            current_time = departure_utc + pd.Timedelta(hours=int(abs_hour))
            current_time_naive = current_time.tz_localize(None)

            wave_slice = wave_region.sel(
                {wave_time: current_time_naive},
                method="nearest",
            )
            wind_slice = wind_region.sel(
                {wind_time: current_time_naive},
                method="nearest",
            )

            lon_w = wave_slice[wave_lon].values
            lat_w = wave_slice[wave_lat].values
            hs = wave_slice[wave_h].values
            hs = np.nan_to_num(hs, nan=0.0)

            if wave_mesh is not None:
                wave_mesh.remove()
            wave_mesh = ax_map.pcolormesh(
                lon_w,
                lat_w,
                hs,
                shading="auto",
                cmap="viridis",
                alpha=0.55,
                vmin=0.0,
                vmax=max(1.0, float(np.nanpercentile(hs, 95))),
            )

            lon_c = wind_slice[wind_lon].values
            lat_c = wind_slice[wind_lat].values
            u = wind_slice[wind_u].values
            v = wind_slice[wind_v].values

            step_lat = max(1, int(len(lat_c) / 20))
            step_lon = max(1, int(len(lon_c) / 25))
            lon_q = lon_c[::step_lon]
            lat_q = lat_c[::step_lat]
            u_q = u[::step_lat, ::step_lon]
            v_q = v[::step_lat, ::step_lon]
            lon_qq, lat_qq = np.meshgrid(lon_q, lat_q)

            if wind_quiver is not None:
                wind_quiver.remove()
            wind_quiver = ax_map.quiver(
                lon_qq,
                lat_qq,
                u_q,
                v_q,
                color="white",
                alpha=0.8,
                width=0.0018,
                scale=350,
            )

            ranking_rows: list[tuple[str, float, float]] = []
            for name in sub_names:
                track = tracks_by_submission[name]
                lon_i, lat_i, dlon, dlat = _track_state_at_hour(track, abs_hour)

                t0 = track["time_utc"].iloc[0]
                elapsed = (
                    track["time_utc"] - t0
                ).dt.total_seconds().to_numpy() / 3600.0
                cutoff = np.searchsorted(elapsed, abs_hour, side="right")
                trail = track.iloc[: max(cutoff, 2)]
                line_handles[name].set_data(
                    trail["lon_deg"].to_numpy(dtype=float),
                    trail["lat_deg"].to_numpy(dtype=float),
                )

                heading_deg = float(np.degrees(np.arctan2(dlat, dlon)))
                marker_style = (3, 0, heading_deg - 90.0)
                vessel_shadow_handles[name].set_marker(marker_style)
                vessel_handles[name].set_marker(marker_style)
                vessel_shadow_handles[name].set_data([lon_i], [lat_i])
                vessel_handles[name].set_data([lon_i], [lat_i])

                cumulative = total_energy[name] * min(abs_hour / case_hours, 1.0)
                ranking_rows.append((name, cumulative, total_energy[name]))

            ranking_rows.sort(key=lambda row: row[1])
            ranking_rows.sort(key=lambda row: row[1])

            if abs_hour >= case_hours:
                ax_map.set_title(
                    f"{CASE_LABELS[case]} - {departure_utc.date()} - "
                    f"Final (winner: {alias_by_name[best_name]})"
                )
            else:
                ax_map.set_title(
                    f"{CASE_LABELS[case]} - {departure_utc.date()} - +{abs_hour:03d}h"
                )

            ax_bar.clear()
            ax_bar.set_facecolor("#f7f9fc")
            ax_bar.set_title("Fuel Gap vs Current Leader", fontsize=10, pad=8)
            ax_bar.set_xlabel("Delta cumulative fuel (MWh)")

            names = [alias_by_name[name] for name, _, _ in ranking_rows]
            cumulatives = [cumulative for _, cumulative, _ in ranking_rows]
            leader = cumulatives[0] if cumulatives else 0.0
            deltas = [value - leader for value in cumulatives]
            positions = np.arange(len(names))

            bar_colors = [color_by_alias[alias] for alias in names]
            bars = ax_bar.barh(
                positions,
                deltas,
                color=bar_colors,
                alpha=0.88,
                edgecolor="#1f2933",
                linewidth=0.4,
            )
            ax_bar.set_yticks(positions, names)
            ax_bar.invert_yaxis()
            ax_bar.grid(axis="x", alpha=0.25, linestyle="--")
            ax_bar.set_xlim(0.0, x_max)

            for idx, (bar, cumulative) in enumerate(
                zip(bars, cumulatives, strict=False)
            ):
                ax_bar.text(
                    bar.get_width() + x_max * 0.015,
                    idx,
                    f"{cumulative:.1f}",
                    va="center",
                    ha="left",
                    fontsize=9,
                    color="#1f2933",
                )

            return [
                *line_handles.values(),
                *vessel_shadow_handles.values(),
                *vessel_handles.values(),
            ]

        anim = animation.FuncAnimation(
            fig,
            _frame,
            frames=frame_hours,
            interval=15,
            blit=False,
            repeat=False,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fps = ANIMATION_FPS
        mp4_path = output_path.with_suffix(".mp4")
        gif_path = output_path.with_suffix(".gif")

        # Always save GIF.
        gif_writer = animation.PillowWriter(fps=fps)
        anim.save(gif_path, writer=gif_writer, dpi=dpi)

        # Always save MP4 (ffmpeg preferred, OpenCV fallback).
        if animation.writers.is_available("ffmpeg"):
            mp4_writer = animation.FFMpegWriter(fps=fps)
            anim.save(mp4_path, writer=mp4_writer, dpi=dpi)
        else:
            _gif_to_mp4(gif_path, mp4_path, fps=fps)

        plt.close(fig)
    finally:
        wind_ds.close()
        wave_ds.close()


def _corridor_from_case(case: str) -> str:
    """Return corridor identifier from case id."""
    if case.startswith("AO") or case.startswith("AGC"):
        return "atlantic"
    if case.startswith("PO") or case.startswith("PGC"):
        return "pacific"
    raise ValueError(f"Unknown SWOPP3 case: {case}")


def _parse_departure_argument(value: str) -> datetime:
    """Parse departure input and default date-only values to 12:00 UTC."""
    text = value.strip()
    formats = [
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
    ]
    for fmt in formats:
        try:
            parsed = datetime.strptime(text, fmt)
            if fmt == "%Y-%m-%d":
                return parsed.replace(hour=12, minute=0, second=0)
            return parsed
        except ValueError:
            continue
    raise ValueError(
        "Invalid --animation-departure format. Use YYYY-MM-DD or YYYY-MM-DDTHH:MM[:SS]"
    )


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(
        description="SWOPP3 submission comparison plots and animation"
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("output/swopp3_submissions"),
        help="Folder containing participant submission zip files or directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/analysis/submission_compare"),
        help="Directory where figures and animation are written.",
    )
    parser.add_argument(
        "--score-root",
        type=Path,
        default=SCORE_ARCHIVE_ROOT,
        help=("Folder containing scored submission archives with resampled routes."),
    )
    parser.add_argument(
        "--expected-departures",
        type=int,
        default=366,
        help="Expected number of departures per case.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=10,
        help=(
            "Deprecated. Spread metrics always use 10 evenly-spaced times "
            "across the full passage duration."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Figure and animation DPI.",
    )
    parser.add_argument(
        "--animation-cases",
        nargs="+",
        choices=REQUIRED_CASES,
        default=REQUIRED_CASES,
        help=(
            "Case list used for animation generation. "
            "Defaults to all four optimized cases."
        ),
    )
    parser.add_argument(
        "--animation-departure",
        type=str,
        default="2024-01-01",
        help="Departure date used for animation (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--skip-animation",
        action="store_true",
        help="Skip animation generation.",
    )
    parser.add_argument(
        "--wind-path-atlantic",
        type=Path,
        default=Path("data/era5/era5_wind_atlantic_2024.nc"),
        help="Atlantic ERA5 wind dataset used for animation.",
    )
    parser.add_argument(
        "--wave-path-atlantic",
        type=Path,
        default=Path("data/era5/era5_waves_atlantic_2024.nc"),
        help="Atlantic ERA5 wave dataset used for animation.",
    )
    parser.add_argument(
        "--wind-path-pacific",
        type=Path,
        default=Path("data/era5/era5_wind_pacific_2024.nc"),
        help="Pacific ERA5 wind dataset used for animation.",
    )
    parser.add_argument(
        "--wave-path-pacific",
        type=Path,
        default=Path("data/era5/era5_waves_pacific_2024.nc"),
        help=(
            "Pacific ERA5 wave dataset used for route energy re-evaluation "
            "and animation."
        ),
    )
    return parser


def main() -> None:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="swopp3_submissions_") as tmpdir:
        extraction_root = Path(tmpdir)
        submissions, issues = scan_submissions(
            args.input_root,
            required_cases=REQUIRED_CASES,
            expected_departures=args.expected_departures,
            extraction_root=extraction_root,
        )

        print(f"Scanned {args.input_root}")
        print(f"Valid submissions: {len(submissions)}")
        if submissions:
            for sub in submissions:
                print(f"  - {sub.name}")
        print(f"Rejected folders: {len(issues)}")
        if issues:
            for issue in issues:
                print(f"  - {issue.folder}: {issue.reason}")

        if len(submissions) < 2:
            raise RuntimeError(
                "At least two valid submissions are required for comparison plots."
            )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        alias_by_name, alias_to_name, color_by_alias = _assign_participant_identity(
            submissions,
            out_dir=args.output_dir,
        )

        score_archives_by_name = _discover_score_archives(args.score_root)

        sample_hours_by_case: dict[str, np.ndarray] = {}
        sampled_cache: dict[tuple[str, str], tuple[pd.DatetimeIndex, np.ndarray]] = {}
        if args.sample_count != SPREAD_SAMPLE_COUNT:
            print(
                "Ignoring --sample-count="
                f"{args.sample_count}; using fixed "
                f"{SPREAD_SAMPLE_COUNT} samples for spread plots."
            )
        for case in REQUIRED_CASES:
            case_hours = float(SWOPP3_CASES[case]["passage_hours"])
            sample_hours = np.linspace(0.0, case_hours, SPREAD_SAMPLE_COUNT)
            sample_hours_by_case[case] = sample_hours
            for sub in submissions:
                sampled_cache[(sub.name, case)] = _sample_waypoints_for_case(
                    sub, case, sample_hours
                )

        print("Building per-participant departure tables (cache-aware)...")
        participant_tables = _load_or_build_participant_tables(
            submissions,
            out_dir=args.output_dir,
            alias_by_name=alias_by_name,
            score_archives_by_name=score_archives_by_name,
        )
        energy_lookup, violation_lookup = _build_lookup_from_tables(participant_tables)

        overall_table = build_overall_evaluation_table(
            submissions,
            participant_tables=participant_tables,
        )
        overall_path = save_overall_evaluation_table(
            overall_table, out_dir=args.output_dir
        )
        print(f"Saved overall leaderboard CSV: {overall_path}")

        top5_participants = overall_table.head(5)["Participant"].tolist()
        requested_top5_participants = _resolve_requested_top5_participants(submissions)
        if set(top5_participants) != set(requested_top5_participants):
            raise RuntimeError(
                "Requested top-5 participants do not match the lowest-consumption "
                "leaderboard entries"
            )
        _update_participant_identity_top5(
            out_dir=args.output_dir,
            top5_requested=TOP5_REQUESTED_TOKENS,
            top5_participants=top5_participants,
            alias_by_name=alias_by_name,
        )

        plot_consumption(
            submissions,
            energy_lookup=energy_lookup,
            alias_by_name=alias_by_name,
            color_by_alias=color_by_alias,
            out_dir=args.output_dir,
            dpi=args.dpi,
        )
        plot_consumption(
            submissions,
            energy_lookup=energy_lookup,
            alias_by_name=alias_by_name,
            color_by_alias=color_by_alias,
            out_dir=args.output_dir,
            dpi=args.dpi,
            selected_names=set(top5_participants),
            output_suffix="_top5",
        )

        for case in REQUIRED_CASES:
            plot_participant_spread(
                submissions,
                case=case,
                sampled_cache=sampled_cache,
                sample_hours=sample_hours_by_case[case],
                alias_by_name=alias_by_name,
                color_by_alias=color_by_alias,
                out_dir=args.output_dir,
                dpi=args.dpi,
            )
            plot_participant_spread(
                submissions,
                case=case,
                sampled_cache=sampled_cache,
                sample_hours=sample_hours_by_case[case],
                alias_by_name=alias_by_name,
                color_by_alias=color_by_alias,
                out_dir=args.output_dir,
                dpi=args.dpi,
                selected_names=set(top5_participants),
                output_suffix="_top5",
            )

        for case in REQUIRED_CASES:
            plot_month_spread(
                submissions,
                case=case,
                sampled_cache=sampled_cache,
                sample_hours=sample_hours_by_case[case],
                out_dir=args.output_dir,
                dpi=args.dpi,
            )
            plot_month_spread(
                submissions,
                case=case,
                sampled_cache=sampled_cache,
                sample_hours=sample_hours_by_case[case],
                out_dir=args.output_dir,
                dpi=args.dpi,
                selected_names=set(top5_participants),
                output_suffix="_top5",
            )

        wins_table = build_departure_wins_table(
            submissions,
            energy_lookup=energy_lookup,
            violation_lookup=violation_lookup,
        )
        wins_table = wins_table.rename(index=alias_by_name)
        save_departure_wins_table(wins_table, out_dir=args.output_dir, dpi=args.dpi)

        avg_table = build_average_consumption_table(
            submissions,
            energy_lookup=energy_lookup,
            violation_lookup=violation_lookup,
        )
        avg_table = avg_table.rename(index=alias_by_name)
        save_average_consumption_table(avg_table, out_dir=args.output_dir, dpi=args.dpi)

        plot_wps_vs_nowps_spread(
            submissions,
            sampled_cache=sampled_cache,
            sample_hours_by_case=sample_hours_by_case,
            alias_by_name=alias_by_name,
            color_by_alias=color_by_alias,
            out_dir=args.output_dir,
            dpi=args.dpi,
        )
        plot_wps_vs_nowps_spread(
            submissions,
            sampled_cache=sampled_cache,
            sample_hours_by_case=sample_hours_by_case,
            alias_by_name=alias_by_name,
            color_by_alias=color_by_alias,
            out_dir=args.output_dir,
            dpi=args.dpi,
            selected_names=set(top5_participants),
            output_suffix="_top5",
        )

        if not args.skip_animation:
            departure = _parse_departure_argument(args.animation_departure)
            for animation_case in args.animation_cases:
                corridor = _corridor_from_case(animation_case)
                if corridor == "atlantic":
                    wind_path = args.wind_path_atlantic
                    wave_path = args.wave_path_atlantic
                else:
                    wind_path = args.wind_path_pacific
                    wave_path = args.wave_path_pacific

                print(f"Rendering animation for {animation_case}...")
                animate_departure(
                    [sub for sub in submissions if sub.name in set(top5_participants)],
                    case=animation_case,
                    departure=departure,
                    energy_lookup=energy_lookup,
                    alias_by_name=alias_by_name,
                    color_by_alias=color_by_alias,
                    output_path=args.output_dir
                    / f"animation_{animation_case}_{args.animation_departure}_top5",
                    wave_path=wave_path,
                    wind_path=wind_path,
                    dpi=args.dpi,
                )


if __name__ == "__main__":
    main()
