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
        --input-root output/swopp3_submissions_score \
        --output-dir output/swopp3_submissions_compare
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

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.animation as animation
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
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


def _save_plotly_html(fig: go.Figure, path: Path) -> None:
    """Save a Plotly figure as a standalone interactive HTML file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path.with_suffix(".html")), include_plotlyjs="cdn")


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
    track["lon_deg"] = np.mod(track["lon_deg"], 360.0)

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
    lon_unwrapped = np.degrees(np.unwrap(np.radians(lon)))
    lat = track["lat_deg"].to_numpy(dtype=float)

    elapsed_target = float(np.clip(elapsed_hours, elapsed.min(), elapsed.max()))
    lon_i = float(np.mod(np.interp(elapsed_target, elapsed, lon_unwrapped), 360.0))
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
    dlon = (dlon + np.pi) % (2.0 * np.pi) - np.pi
    dlat = lat - lat.T

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat) * np.cos(lat.T) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(a)))
    dist_km = 6371.0 * c

    tri = np.triu_indices(n_points, k=1)
    return float(np.mean(dist_km[tri]))


def _great_circle_path(
    lon1: float, lat1: float, lon2: float, lat2: float, num_points: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """Generate great circle path between two points.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Arrays of longitudes and latitudes along the great circle.
    """
    # Simple linear interpolation in lat/lon (approximation for visualization).
    # For better accuracy, would use geodesic calculations.
    lons = np.linspace(lon1, lon2, num_points)
    lats = np.linspace(lat1, lat2, num_points)

    return lons, lats


def _find_representative_cases(
    submissions: list[SubmissionData],
    energy_lookup: dict[str, dict[str, dict[pd.Timestamp, float]]],
    top5_participants: list[str],
) -> tuple[str, pd.Timestamp, str, pd.Timestamp]:
    """Find representative summer (low variance) and winter (high variance) cases.

    Returns
    -------
    tuple[str, pd.Timestamp, str, pd.Timestamp]
        (summer_case, summer_departure, winter_case, winter_departure)
    """
    case_variances: dict[str, float] = {}
    case_cv_profiles: dict[str, np.ndarray] = {}
    case_departures: dict[str, pd.DatetimeIndex] = {}

    for case in REQUIRED_CASES:
        departures = None
        top5_energies: list[list[float]] = []

        for name in top5_participants:
            submission = next((s for s in submissions if s.name == name), None)
            if submission is None:
                continue

            summary = submission.summaries[case].sort_values("departure_time_utc")
            if departures is None:
                departures = pd.DatetimeIndex(summary["departure_time_utc"])

            energies = [
                energy_lookup[case][name][_normalize_departure_key(dep)]
                for dep in departures
            ]
            top5_energies.append(energies)

        if len(top5_energies) >= 2:
            energies_array = np.array(top5_energies)
            cv = np.std(energies_array, axis=0) / np.mean(energies_array, axis=0)
            case_variances[case] = float(np.mean(cv))
            case_cv_profiles[case] = cv
        else:
            case_variances[case] = 0.0
            case_cv_profiles[case] = np.zeros(1)

        if departures is not None:
            case_departures[case] = departures

    summer_case = min(
        REQUIRED_CASES,
        key=lambda c: case_variances.get(c, float("inf")),
    )
    winter_candidates = [c for c in REQUIRED_CASES if c != summer_case]
    winter_case = max(winter_candidates, key=lambda c: case_variances.get(c, 0.0))

    def _pick_dep(case_name: str, *, maximize: bool) -> pd.Timestamp:
        deps = case_departures.get(case_name)
        cv = case_cv_profiles.get(case_name, np.zeros(1))
        if deps is None or len(deps) == 0 or len(cv) == 0:
            month = 1 if maximize else 6
            return pd.Timestamp(f"2024-{month:02d}-15T12:00:00", tz="UTC")
        idx = int(np.argmax(cv)) if maximize else int(np.argmin(cv))
        ts = deps[idx]
        return ts if ts.tzinfo is not None else ts.tz_localize("UTC")

    summer_departure = _pick_dep(summer_case, maximize=False)
    winter_departure = _pick_dep(winter_case, maximize=True)

    return summer_case, summer_departure, winter_case, winter_departure


def _find_departure_with_wps_impact(
    submissions: list[SubmissionData],
    energy_lookup: dict[str, dict[str, dict[pd.Timestamp, float]]],
    reference_participant: str,
    ocean: str = "atlantic",
) -> datetime:
    """Find a departure where WPS has significant impact for the reference participant.

    Parameters
    ----------
    ocean : str
        Either "atlantic" or "pacific"

    Returns
    -------
    datetime
        A departure date with significant WPS impact difference.
    """
    wps_case = f"{ocean[0].upper()}O_WPS" if ocean == "atlantic" else "PO_WPS"
    nowps_case = f"{ocean[0].upper()}O_noWPS" if ocean == "atlantic" else "PO_noWPS"

    submission = next((s for s in submissions if s.name == reference_participant), None)
    if submission is None:
        raise ValueError(f"Reference participant {reference_participant} not found")

    wps_summary = submission.summaries[wps_case].sort_values("departure_time_utc")

    # Find departure with maximum WPS benefit
    max_benefit = 0.0
    best_departure = None

    for _, wps_row in wps_summary.iterrows():
        dep_time = _normalize_departure_key(wps_row["departure_time_utc"])
        wps_energy = energy_lookup[wps_case][reference_participant].get(
            dep_time, np.nan
        )
        nowps_energy = energy_lookup[nowps_case][reference_participant].get(
            dep_time, np.nan
        )

        if np.isfinite(wps_energy) and np.isfinite(nowps_energy):
            benefit = nowps_energy - wps_energy
            if benefit > max_benefit:
                max_benefit = benefit
                best_departure = wps_row["departure_time_utc"]

    if best_departure is None:
        # Fallback to first departure
        ts = pd.Timestamp(wps_summary.iloc[0]["departure_time_utc"])
    else:
        ts = pd.Timestamp(best_departure)

    # Strip tz so animate_departure can safely attach "UTC".
    return ts.tz_localize(None) if ts.tzinfo is not None else ts


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
        ax.set_ylim(bottom=0.0)
        ax.grid(alpha=0.3)
        ax.legend(loc="best", ncols=2, fontsize=9)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

        out = out_dir / f"consumption_{case}{output_suffix}.pdf"
        _save_figure(fig, out, dpi=dpi)
        plt.close(fig)

        # Interactive HTML (hover shows all participant values for a given date)
        traces: list[go.BaseTraceType] = []
        for name in _ordered_submission_names(
            submissions,
            alias_by_name=alias_by_name,
            selected_names=selected_names,
        ):
            submission = next(sub for sub in submissions if sub.name == name)
            summary = submission.summaries[case].sort_values("departure_time_utc")
            departures = pd.DatetimeIndex(summary["departure_time_utc"])
            energies = [
                energy_lookup[case][submission.name][_normalize_departure_key(dep)]
                for dep in departures
            ]
            alias = alias_by_name[name]
            traces.append(
                go.Scatter(
                    x=departures.tz_localize(None),
                    y=energies,
                    mode="lines",
                    name=alias,
                    line={"color": color_by_alias[alias], "width": 1.8},
                    hovertemplate="%{y:.1f} MWh<extra>%{fullData.name}</extra>",
                )
            )
        pfig = go.Figure(traces)
        pfig.update_layout(
            title=f"Consumption Across 2024 Departures — {CASE_LABELS[case]}",
            xaxis_title="Departure date",
            yaxis_title="Energy consumption (MWh)",
            yaxis={"rangemode": "tozero"},
            hovermode="x unified",
            legend={"orientation": "v"},
        )
        _save_plotly_html(pfig, out)


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
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", ncols=2, fontsize=9)

    out = out_dir / f"spread_participants_{case}{output_suffix}.pdf"
    _save_figure(fig, out, dpi=dpi)
    plt.close(fig)

    # Interactive HTML
    traces_p: list[go.BaseTraceType] = []
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
        alias = alias_by_name[name]
        traces_p.append(
            go.Scatter(
                x=sample_hours,
                y=spreads,
                mode="lines+markers",
                name=alias,
                line={"color": color_by_alias[alias]},
                hovertemplate="%{y:.1f} km<extra>%{fullData.name}</extra>",
            )
        )
    pfig_p = go.Figure(traces_p)
    pfig_p.update_layout(
        title=f"Participant Spread Comparison — {CASE_LABELS[case]}",
        xaxis_title="Elapsed time (hours)",
        yaxis_title="Mean pairwise waypoint distance (km)",
        yaxis={"rangemode": "tozero"},
        hovermode="x unified",
    )
    _save_plotly_html(pfig_p, out)


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
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", ncols=4, fontsize=8)

    out = out_dir / f"spread_months_{case}{output_suffix}.pdf"
    _save_figure(fig, out, dpi=dpi)
    plt.close(fig)

    # Interactive HTML
    traces_m: list[go.BaseTraceType] = []
    for month in range(1, 13):
        if not by_month[month]:
            continue
        mean_curve = np.mean(np.vstack(by_month[month]), axis=0)
        traces_m.append(
            go.Scatter(
                x=sample_hours,
                y=mean_curve.tolist(),
                mode="lines+markers",
                name=MONTH_NAMES[month - 1],
                hovertemplate="%{y:.1f} km<extra>%{fullData.name}</extra>",
            )
        )
    pfig_m = go.Figure(traces_m)
    pfig_m.update_layout(
        title=f"Month Spread Comparison — {CASE_LABELS[case]}",
        xaxis_title="Elapsed time (hours)",
        yaxis_title="Mean cross-participant distance (km)",
        yaxis={"rangemode": "tozero"},
        hovermode="x unified",
    )
    _save_plotly_html(pfig_m, out)


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
    dlon = (dlon + np.pi) % (2.0 * np.pi) - np.pi
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

        # Interactive HTML — rebuild the same data for Plotly
        traces_w: list[go.BaseTraceType] = []
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
            mean_distances_w: list[float] = []
            for sample_idx in range(len(sample_hours)):
                c_wps = np.array(
                    [points_wps[wps_row[d], sample_idx, :] for d in common_dates],
                    dtype=float,
                )
                c_nowps = np.array(
                    [points_nowps[nowps_row[d], sample_idx, :] for d in common_dates],
                    dtype=float,
                )
                mean_distances_w.append(
                    _mean_haversine_km_between_curves(c_wps, c_nowps)
                )
            alias = alias_by_name[name]
            traces_w.append(
                go.Scatter(
                    x=sample_hours,
                    y=mean_distances_w,
                    mode="lines+markers",
                    name=alias,
                    line={"color": color_by_alias[alias]},
                    hovertemplate=("%{y:.1f} km<extra>%{fullData.name}</extra>"),
                )
            )
        pfig_w = go.Figure(traces_w)
        pfig_w.update_layout(
            title=f"WPS vs no-WPS Spread — {corridor.title()}",
            xaxis_title="Elapsed time (hours)",
            yaxis_title="Mean waypoint distance between WPS and no-WPS (km)",
            hovermode="x unified",
        )
        _save_plotly_html(pfig_w, out)


def _average_exploration_by_participant(
    submissions: list[SubmissionData],
    *,
    sampled_cache: dict[tuple[str, str], tuple[pd.DatetimeIndex, np.ndarray]],
    selected_names: set[str] | None = None,
) -> dict[str, float]:
    """Return average exploration spread per participant across all cases/times."""
    averages: dict[str, float] = {}
    for submission in submissions:
        if selected_names is not None and submission.name not in selected_names:
            continue

        spread_samples: list[float] = []
        for case in REQUIRED_CASES:
            _, points = sampled_cache[(submission.name, case)]
            spread_samples.extend(
                _mean_pairwise_haversine_km(points[:, sample_idx, :])
                for sample_idx in range(points.shape[1])
            )

        averages[submission.name] = (
            float(np.mean(spread_samples)) if spread_samples else np.nan
        )
    return averages


def plot_consumption_vs_exploration_scatter(
    submissions: list[SubmissionData],
    *,
    sampled_cache: dict[tuple[str, str], tuple[pd.DatetimeIndex, np.ndarray]],
    average_consumption_by_name: dict[str, float],
    alias_by_name: dict[str, str],
    color_by_alias: dict[str, str],
    out_dir: Path,
    dpi: int,
    selected_names: set[str] | None = None,
    output_suffix: str = "",
) -> None:
    """Plot participant average consumption against average exploration spread."""
    avg_exploration = _average_exploration_by_participant(
        submissions,
        sampled_cache=sampled_cache,
        selected_names=selected_names,
    )

    ordered_names = _ordered_submission_names(
        submissions,
        alias_by_name=alias_by_name,
        selected_names=selected_names,
    )
    points = [
        (
            name,
            avg_exploration.get(name, np.nan),
            average_consumption_by_name.get(name, np.nan),
        )
        for name in ordered_names
    ]
    points = [p for p in points if np.isfinite(p[1]) and np.isfinite(p[2])]
    if not points:
        return

    fig, ax = plt.subplots(figsize=(8.5, 6))
    for name, exploration, consumption in points:
        alias = alias_by_name[name]
        ax.scatter(
            exploration,
            consumption,
            s=70,
            color=color_by_alias[alias],
            edgecolor="#1f2933",
            linewidth=0.5,
            alpha=0.9,
        )
        ax.text(
            exploration,
            consumption,
            f" {alias}",
            va="center",
            ha="left",
            fontsize=9,
        )

    ax.set_title("Average Consumption vs Average Exploration")
    ax.set_xlabel("Average exploration spread (km)")
    ax.set_ylabel("Average consumption (MWh)")
    ax.set_ylim(bottom=0.0)
    ax.set_xlim(left=0.0)
    ax.grid(alpha=0.3)

    out = out_dir / f"consumption_vs_exploration{output_suffix}.pdf"
    _save_figure(fig, out, dpi=dpi)
    plt.close(fig)

    # Interactive HTML scatter — hover shows alias, exploration, and consumption
    scatter_points: list[go.BaseTraceType] = []
    for name, exploration, consumption in points:
        alias = alias_by_name[name]
        scatter_points.append(
            go.Scatter(
                x=[exploration],
                y=[consumption],
                mode="markers+text",
                name=alias,
                text=[alias],
                textposition="middle right",
                marker={
                    "color": color_by_alias[alias],
                    "size": 10,
                    "line": {"color": "#1f2933", "width": 1},
                },
                hovertemplate=(
                    f"<b>{alias}</b><br>"
                    "Exploration: %{x:.1f} km<br>"
                    "Consumption: %{y:.1f} MWh"
                    "<extra></extra>"
                ),
            )
        )
    pfig_s = go.Figure(scatter_points)
    pfig_s.update_layout(
        title="Average Consumption vs Average Exploration",
        xaxis_title="Average exploration spread (km)",
        yaxis_title="Average consumption (MWh)",
        xaxis={"rangemode": "tozero"},
        yaxis={"rangemode": "tozero"},
        hovermode="closest",
        showlegend=False,
    )
    _save_plotly_html(pfig_s, out)


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


def _add_coastline_to_axes(
    ax: plt.Axes,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
) -> None:
    """Draw Natural Earth coastline as black lines on a plain matplotlib Axes.

    Longitudes are converted to [0, 360) to match the animation coordinate
    convention used throughout the script.  Lines are placed above the wave
    mesh (zorder=3) but below wind arrows (zorder=4).
    """
    from matplotlib.collections import LineCollection

    coastline = cfeature.NaturalEarthFeature("physical", "coastline", "110m")
    segments = []
    pad_lon = 20.0
    pad_lat = 10.0

    for geom in coastline.geometries():
        lines = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
        for line in lines:
            coords = np.array(line.coords)
            # Convert to [0, 360) to match track/weather convention.
            coords[:, 0] = np.mod(coords[:, 0], 360.0)
            # Split segments that jump across the antimeridian (|Δlon| > 180°).
            split_indices = np.where(np.abs(np.diff(coords[:, 0])) > 180.0)[0] + 1
            sub_segs = np.split(coords, split_indices)
            for sub in sub_segs:
                if len(sub) < 2:
                    continue
                if (
                    sub[:, 0].max() < lon_min - pad_lon
                    or sub[:, 0].min() > lon_max + pad_lon
                    or sub[:, 1].max() < lat_min - pad_lat
                    or sub[:, 1].min() > lat_max + pad_lat
                ):
                    continue
                segments.append(sub[:, :2])

    if segments:
        ax.add_collection(
            LineCollection(
                segments,
                colors="black",
                linewidths=0.8,
                zorder=3,
            )
        )


def _haversine_deg(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle angular distance in degrees between two (lon, lat) points."""
    lat1r, lat2r = np.radians(lat1), np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    )
    return float(np.degrees(2.0 * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))))


def _build_map_axes(
    fig: plt.Figure,
    subplot_spec,
    lon_min_360: float,
    lon_max_360: float,
    lat_min: float,
    lat_max: float,
):
    """Create a GeoAxes with NearsidePerspective projection for route visualization.

    The satellite height is computed so the full route bounding box fits inside
    the visible hemisphere with ~15° margin, giving a genuine globe-curvature
    effect.  Longitudes follow the [0, 360) convention used throughout this
    script.  Pass ERA5/track data to the returned axes with
    ``transform=ccrs.PlateCarree(central_longitude=180)`` and shift longitudes
    as ``lon_360 - 180.0``.
    """
    # Midpoint in 0-360, then convert to -180/180 for cartopy
    center_lon_360 = (lon_min_360 + lon_max_360) / 2.0
    center_lon_180 = float(((center_lon_360 + 180.0) % 360.0) - 180.0)
    center_lat = (lat_min + lat_max) / 2.0

    # Angular radius from centre to each corner
    corners = [
        (lon_min_360, lat_min),
        (lon_min_360, lat_max),
        (lon_max_360, lat_min),
        (lon_max_360, lat_max),
    ]
    max_angle = 0.0
    for clon_360, clat in corners:
        clon_180 = float(((clon_360 + 180.0) % 360.0) - 180.0)
        d = _haversine_deg(center_lon_180, center_lat, clon_180, clat)
        max_angle = max(max_angle, d)

    # Satellite height so the visible radius covers max_angle + 15° margin
    _R_EARTH = 6_371_000.0
    target_angle = min(max_angle + 15.0, 88.0)
    h = _R_EARTH * (1.0 / np.cos(np.radians(target_angle)) - 1.0)
    satellite_height = max(float(h), 3_000_000.0)

    projection = ccrs.NearsidePerspective(
        central_longitude=center_lon_180,
        central_latitude=center_lat,
        satellite_height=satellite_height,
    )
    ax = fig.add_subplot(subplot_spec, projection=projection)
    ax.set_facecolor("#0a1929")  # colour outside the visible hemisphere

    # Filled ocean and land
    ax.add_feature(
        cfeature.NaturalEarthFeature("physical", "ocean", "50m"),
        facecolor="#0d2538",
        zorder=1,
    )
    ax.add_feature(
        cfeature.NaturalEarthFeature("physical", "land", "50m"),
        facecolor="#2d5a27",
        edgecolor="#1a3b18",
        linewidth=0.4,
        zorder=2,
    )
    ax.add_feature(
        cfeature.NaturalEarthFeature("physical", "coastline", "50m"),
        edgecolor="#5a8f54",
        facecolor="none",
        linewidth=0.7,
        zorder=3,
    )
    ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=False,
        linewidth=0.4,
        color="gray",
        alpha=0.35,
        linestyle="--",
        zorder=0,
    )
    return ax


def _normalize_longitude(ds: xr.Dataset, lon_name: str) -> xr.Dataset:
    """Normalize longitude axis to [0, 360) and sort coordinates."""
    lon = ds[lon_name]
    lon_shift = lon % 360.0
    return ds.assign_coords({lon_name: lon_shift}).sortby(lon_name)


def _normalize_longitude_360(ds: xr.Dataset, lon_name: str) -> xr.Dataset:
    """Normalize longitude axis to [0, 360) and sort coordinates."""
    return _normalize_longitude(ds, lon_name)


def _track_to_360(track: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of track data with longitudes wrapped to [0, 360)."""
    normalized = track.copy()
    normalized["lon_deg"] = np.mod(normalized["lon_deg"], 360.0)
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
    lon_unwrapped = np.degrees(np.unwrap(np.radians(lon)))
    lat = track["lat_deg"].to_numpy(dtype=float)

    hour = float(np.clip(hour, elapsed.min(), elapsed.max()))
    lon_i = float(np.mod(np.interp(hour, elapsed, lon_unwrapped), 360.0))
    lat_i = float(np.interp(hour, elapsed, lat))

    delta = 0.5
    h0 = float(np.clip(hour - delta, elapsed.min(), elapsed.max()))
    h1 = float(np.clip(hour + delta, elapsed.min(), elapsed.max()))
    lon0 = float(np.interp(h0, elapsed, lon_unwrapped))
    lat0 = float(np.interp(h0, elapsed, lat))
    lon1 = float(np.interp(h1, elapsed, lon_unwrapped))
    lat1 = float(np.interp(h1, elapsed, lat))

    return lon_i, lat_i, lon1 - lon0, lat1 - lat0


def _load_scored_route_timeseries(
    score_archive: Path,
    *,
    case: str,
    departure_utc: pd.Timestamp,
) -> pd.DataFrame:
    """Load scored per-step energy for one case/departure from a score archive."""
    departure_date = _normalize_departure_key(departure_utc).strftime("%Y%m%d")
    score_zip = _load_resampled_tracks_zip(score_archive)
    try:
        route_members = _index_scored_route_members(score_zip)
        score_name = route_members[case].get(departure_date)
        if score_name is None:
            raise FileNotFoundError(
                f"Missing scored route for {case} on {departure_date} in "
                f"{score_archive.name}"
            )
        with score_zip.open(score_name) as score_file:
            return pd.read_csv(score_file, usecols=["timestamp", "E"])
    finally:
        score_zip.close()


def _build_cumulative_energy_profile(
    scored_route: pd.DataFrame,
    *,
    departure_utc: pd.Timestamp,
) -> tuple[np.ndarray, np.ndarray]:
    """Return elapsed hours and cumulative energy from scored route samples."""
    route = scored_route.copy()
    route["timestamp"] = pd.to_datetime(route["timestamp"], utc=True, errors="coerce")
    route["E"] = pd.to_numeric(route["E"], errors="coerce")
    route = route.dropna(subset=["timestamp", "E"]).sort_values("timestamp")
    if route.empty:
        return np.array([0.0]), np.array([0.0])

    elapsed_hours = (
        route["timestamp"] - _normalize_departure_key(departure_utc)
    ).dt.total_seconds().to_numpy(dtype=float) / 3600.0
    valid = elapsed_hours >= 0.0
    if not np.any(valid):
        return np.array([0.0]), np.array([0.0])

    elapsed_hours = elapsed_hours[valid]
    incremental_energy = route.loc[valid, "E"].to_numpy(dtype=float)
    cumulative_energy = np.cumsum(incremental_energy)

    if elapsed_hours[0] > 0.0:
        elapsed_hours = np.insert(elapsed_hours, 0, 0.0)
        cumulative_energy = np.insert(cumulative_energy, 0, 0.0)
    else:
        elapsed_hours[0] = 0.0

    return elapsed_hours, cumulative_energy


def _cumulative_energy_at_hour(
    profile: tuple[np.ndarray, np.ndarray],
    hour: float,
) -> float:
    """Interpolate cumulative energy profile at the requested absolute hour."""
    elapsed_hours, cumulative_energy = profile
    return float(np.interp(hour, elapsed_hours, cumulative_energy))


def animate_wps_comparison(
    submission: SubmissionData,
    *,
    wps_case: str,
    nowps_case: str,
    departure: datetime,
    energy_lookup: dict[str, dict[str, dict[pd.Timestamp, float]]],
    score_archives_by_name: dict[str, Path],
    alias_by_name: dict[str, str],
    output_path: Path,
    wave_path: Path,
    wind_path: Path,
    dpi: int,
) -> None:
    """Single animation showing WPS vs no-WPS route for one participant/departure.

    Both routes are overlaid on the same map: the WPS-routed track in teal
    (solid line) and the no-WPS track in orange (dashed line), each with its
    own vessel marker.  A horizontal bar chart below tracks cumulative fuel
    consumption for both cases simultaneously.
    """
    _ts = pd.Timestamp(departure)
    departure_utc = (
        _ts.tz_localize("UTC") if _ts.tzinfo is None else _ts.tz_convert("UTC")
    )

    def _load_case(
        case: str,
    ) -> tuple[pd.DataFrame, tuple[np.ndarray, np.ndarray]]:
        summary = submission.summaries[case]
        row = summary.loc[summary["departure_time_utc"] == departure_utc]
        if row.empty:
            raise ValueError(
                f"No data for {submission.name} / {case} / {departure_utc}"
            )
        details = str(row.iloc[0]["details_filename"])
        track = _read_track(submission.tracks_dir / details)
        score_archive = score_archives_by_name[submission.name]
        scored_route = _load_scored_route_timeseries(
            score_archive, case=case, departure_utc=departure_utc
        )
        profile = _build_cumulative_energy_profile(
            scored_route, departure_utc=departure_utc
        )
        return track, profile

    wps_track, wps_profile = _load_case(wps_case)
    nowps_track, nowps_profile = _load_case(nowps_case)

    case_hours = max(
        int(SWOPP3_CASES[wps_case]["passage_hours"]),
        int(SWOPP3_CASES[nowps_case]["passage_hours"]),
    )
    frame_hours = np.arange(0, case_hours + 1, 1, dtype=int)

    all_lons = np.concatenate(
        [
            wps_track["lon_deg"].to_numpy(dtype=float),
            nowps_track["lon_deg"].to_numpy(dtype=float),
        ]
    )
    all_lats = np.concatenate(
        [
            wps_track["lat_deg"].to_numpy(dtype=float),
            nowps_track["lat_deg"].to_numpy(dtype=float),
        ]
    )
    lon_min = float(np.min(all_lons) - 5.0)
    lon_max = float(np.max(all_lons) + 5.0)
    lat_min = float(np.min(all_lats) - 5.0)
    lat_max = float(np.max(all_lats) + 5.0)

    _WPS_COLOR = "#4dd0e1"  # teal  – weather-routing on
    _NOWPS_COLOR = "#ff9800"  # orange – weather-routing off

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
        _TC = ccrs.PlateCarree(central_longitude=180)
        ax_map = _build_map_axes(fig, grid[0, 0], lon_min, lon_max, lat_min, lat_max)
        ax_bar = fig.add_subplot(grid[1, 0])
        ax_bar.set_facecolor("#f7f9fc")

        # Route trail lines
        (wps_line,) = ax_map.plot(
            [],
            [],
            color=_WPS_COLOR,
            linewidth=2.0,
            alpha=0.92,
            solid_capstyle="round",
            label="With WPS",
            zorder=5,
            transform=_TC,
        )
        (nowps_line,) = ax_map.plot(
            [],
            [],
            color=_NOWPS_COLOR,
            linewidth=2.0,
            alpha=0.92,
            linestyle="--",
            solid_capstyle="round",
            label="No WPS",
            zorder=5,
            transform=_TC,
        )
        # Vessel shadows
        (wps_shadow,) = ax_map.plot(
            [],
            [],
            linestyle="None",
            marker=(3, 0, 0),
            markersize=13,
            markerfacecolor="black",
            markeredgecolor="none",
            alpha=0.35,
            zorder=8,
            transform=_TC,
        )
        (nowps_shadow,) = ax_map.plot(
            [],
            [],
            linestyle="None",
            marker=(3, 0, 0),
            markersize=13,
            markerfacecolor="black",
            markeredgecolor="none",
            alpha=0.35,
            zorder=8,
            transform=_TC,
        )
        # Vessel markers
        (wps_vessel,) = ax_map.plot(
            [],
            [],
            linestyle="None",
            marker=(3, 0, 0),
            markersize=10,
            markerfacecolor=_WPS_COLOR,
            markeredgecolor="white",
            markeredgewidth=1.2,
            zorder=9,
            transform=_TC,
        )
        (nowps_vessel,) = ax_map.plot(
            [],
            [],
            linestyle="None",
            marker=(3, 0, 0),
            markersize=10,
            markerfacecolor=_NOWPS_COLOR,
            markeredgecolor="white",
            markeredgewidth=1.2,
            zorder=9,
            transform=_TC,
        )
        ax_map.legend(loc="upper right", fontsize=9, framealpha=0.75)

        wave_mesh = None
        wind_quiver = None

        wps_total = float(wps_profile[1][-1])
        nowps_total = float(nowps_profile[1][-1])
        x_max = max(1.0, max(wps_total, nowps_total) * 1.05)

        def _frame(abs_hour: int):  # noqa: ANN202
            nonlocal wave_mesh, wind_quiver
            current_time = departure_utc + pd.Timedelta(hours=int(abs_hour))
            current_time_naive = current_time.tz_localize(None)

            wave_slice = wave_region.sel(
                {wave_time: current_time_naive}, method="nearest"
            )
            wind_slice = wind_region.sel(
                {wind_time: current_time_naive}, method="nearest"
            )

            lon_w = wave_slice[wave_lon].values
            lat_w = wave_slice[wave_lat].values
            hs = wave_slice[wave_h].values
            hs = np.nan_to_num(hs, nan=0.0)

            if wave_mesh is not None:
                wave_mesh.remove()
            wave_mesh = ax_map.pcolormesh(
                lon_w - 180.0,
                lat_w,
                hs,
                shading="auto",
                cmap="viridis",
                alpha=0.55,
                vmin=0.0,
                vmax=max(1.0, float(np.nanpercentile(hs, 95))),
                transform=_TC,
                zorder=1.5,
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
                lon_qq - 180.0,
                lat_qq,
                u_q,
                v_q,
                color="white",
                alpha=0.8,
                width=0.0012,
                scale=500,
                zorder=4,
                transform=_TC,
            )

            # WPS track
            lon_i, lat_i, dlon, dlat = _track_state_at_hour(wps_track, abs_hour)
            t0 = wps_track["time_utc"].iloc[0]
            elapsed = (
                wps_track["time_utc"] - t0
            ).dt.total_seconds().to_numpy() / 3600.0
            cutoff = np.searchsorted(elapsed, abs_hour, side="right")
            trail = wps_track.iloc[: max(cutoff, 2)]
            wps_line.set_data(
                trail["lon_deg"].to_numpy(float) - 180.0,
                trail["lat_deg"].to_numpy(float),
            )
            heading = float(np.degrees(np.arctan2(dlat, dlon)))
            wps_shadow.set_marker((3, 0, heading - 90.0))
            wps_vessel.set_marker((3, 0, heading - 90.0))
            wps_shadow.set_data([lon_i - 180.0], [lat_i])
            wps_vessel.set_data([lon_i - 180.0], [lat_i])

            # no-WPS track
            lon_i2, lat_i2, dlon2, dlat2 = _track_state_at_hour(nowps_track, abs_hour)
            t0_2 = nowps_track["time_utc"].iloc[0]
            elapsed2 = (
                nowps_track["time_utc"] - t0_2
            ).dt.total_seconds().to_numpy() / 3600.0
            cutoff2 = np.searchsorted(elapsed2, abs_hour, side="right")
            trail2 = nowps_track.iloc[: max(cutoff2, 2)]
            nowps_line.set_data(
                trail2["lon_deg"].to_numpy(float) - 180.0,
                trail2["lat_deg"].to_numpy(float),
            )
            heading2 = float(np.degrees(np.arctan2(dlat2, dlon2)))
            nowps_shadow.set_marker((3, 0, heading2 - 90.0))
            nowps_vessel.set_marker((3, 0, heading2 - 90.0))
            nowps_shadow.set_data([lon_i2 - 180.0], [lat_i2])
            nowps_vessel.set_data([lon_i2 - 180.0], [lat_i2])

            if abs_hour >= case_hours:
                ax_map.set_title(
                    f"WPS vs no-WPS \u2013 {departure_utc.date()} \u2013 Final"
                )
            else:
                ax_map.set_title(
                    f"WPS vs no-WPS \u2013 {departure_utc.date()} "
                    f"\u2013 +{abs_hour:03d}h"
                )

            ax_bar.clear()
            ax_bar.set_facecolor("#f7f9fc")
            ax_bar.set_title("Cumulative Fuel Consumption", fontsize=10, pad=8)
            ax_bar.set_xlabel("Cumulative fuel (MWh)")
            wps_cum = _cumulative_energy_at_hour(wps_profile, float(abs_hour))
            nowps_cum = _cumulative_energy_at_hour(nowps_profile, float(abs_hour))
            bars = ax_bar.barh(
                [0, 1],
                [wps_cum, nowps_cum],
                color=[_WPS_COLOR, _NOWPS_COLOR],
                alpha=0.88,
                edgecolor="#1f2933",
                linewidth=0.4,
            )
            ax_bar.set_yticks([0, 1], ["With WPS", "No WPS"])
            ax_bar.invert_yaxis()
            ax_bar.grid(axis="x", alpha=0.25, linestyle="--")
            ax_bar.set_xlim(0.0, x_max)
            for idx, (bar, cum) in enumerate(
                zip(bars, [wps_cum, nowps_cum], strict=False)
            ):
                if cum <= 0.0:
                    continue
                label_x = bar.get_width() + x_max * 0.015
                label_ha = "left"
                if label_x > x_max * 0.98:
                    label_x = max(bar.get_width() - x_max * 0.015, 0.0)
                    label_ha = "right"
                ax_bar.text(
                    label_x,
                    idx,
                    f"{cum:.1f}",
                    va="center",
                    ha=label_ha,
                    fontsize=9,
                    color="#1f2933",
                )

            return [
                wps_line,
                wps_shadow,
                wps_vessel,
                nowps_line,
                nowps_shadow,
                nowps_vessel,
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

        gif_writer = animation.PillowWriter(fps=fps)
        anim.save(gif_path, writer=gif_writer, dpi=dpi)

        if animation.writers.is_available("ffmpeg"):
            mp4_writer = animation.FFMpegWriter(fps=fps)
            anim.save(mp4_path, writer=mp4_writer, dpi=dpi)
        else:
            _gif_to_mp4(gif_path, mp4_path, fps=fps)

        plt.close(fig)
    finally:
        wind_ds.close()
        wave_ds.close()


def animate_departure(
    submissions: list[SubmissionData],
    *,
    case: str,
    departure: datetime,
    energy_lookup: dict[str, dict[str, dict[pd.Timestamp, float]]],
    score_archives_by_name: dict[str, Path],
    alias_by_name: dict[str, str],
    color_by_alias: dict[str, str],
    output_path: Path,
    wave_path: Path,
    wind_path: Path,
    dpi: int,
) -> None:
    """Generate the requested hourly animation for one case/departure."""
    _ts = pd.Timestamp(departure)
    departure_utc = (
        _ts.tz_localize("UTC") if _ts.tzinfo is None else _ts.tz_convert("UTC")
    )
    case_hours = int(SWOPP3_CASES[case]["passage_hours"])
    frame_hours = np.arange(0, case_hours + 1, 1, dtype=int)

    tracks_by_submission: dict[str, pd.DataFrame] = {}
    total_energy: dict[str, float] = {}
    cumulative_profiles: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for sub in submissions:
        summary = sub.summaries[case]
        row = summary.loc[summary["departure_time_utc"] == departure_utc]
        if row.empty:
            continue
        details = str(row.iloc[0]["details_filename"])
        track = _read_track(sub.tracks_dir / details)
        tracks_by_submission[sub.name] = track

        score_archive = score_archives_by_name.get(sub.name)
        if score_archive is None:
            raise FileNotFoundError(
                f"Missing scored archive for participant {sub.name}"
            )
        scored_route = _load_scored_route_timeseries(
            score_archive,
            case=case,
            departure_utc=departure_utc,
        )
        profile = _build_cumulative_energy_profile(
            scored_route,
            departure_utc=departure_utc,
        )
        cumulative_profiles[sub.name] = profile
        total_energy[sub.name] = float(profile[1][-1])

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

        _TC = ccrs.PlateCarree(central_longitude=180)
        fig = plt.figure(figsize=(14, 8))
        grid = fig.add_gridspec(2, 1, height_ratios=[6.2, 1.15], hspace=0.08)
        ax_map = _build_map_axes(fig, grid[0, 0], lon_min, lon_max, lat_min, lat_max)
        ax_bar = fig.add_subplot(grid[1, 0])
        ax_bar.set_facecolor("#f7f9fc")

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
                transform=_TC,
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
                transform=_TC,
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
                transform=_TC,
            )
            vessel_handles[name] = vessel

        wave_mesh = None
        wind_quiver = None

        best_name = min(total_energy, key=total_energy.get)
        x_max = max(1.0, max(total_energy.values()) * 1.05)
        bar_names = list(sub_names)

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
                lon_w - 180.0,
                lat_w,
                hs,
                shading="auto",
                cmap="viridis",
                alpha=0.55,
                vmin=0.0,
                vmax=max(1.0, float(np.nanpercentile(hs, 95))),
                transform=_TC,
                zorder=1.5,
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
                lon_qq - 180.0,
                lat_qq,
                u_q,
                v_q,
                color="white",
                alpha=0.8,
                width=0.0012,
                scale=500,
                zorder=4,
                transform=_TC,
            )

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
                    trail["lon_deg"].to_numpy(dtype=float) - 180.0,
                    trail["lat_deg"].to_numpy(dtype=float),
                )

                heading_deg = float(np.degrees(np.arctan2(dlat, dlon)))
                marker_style = (3, 0, heading_deg - 90.0)
                vessel_shadow_handles[name].set_marker(marker_style)
                vessel_handles[name].set_marker(marker_style)
                vessel_shadow_handles[name].set_data([lon_i - 180.0], [lat_i])
                vessel_handles[name].set_data([lon_i - 180.0], [lat_i])

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
            ax_bar.set_title("Cumulative Fuel Consumption", fontsize=10, pad=8)
            ax_bar.set_xlabel("Cumulative fuel (MWh)")

            names = [alias_by_name[name] for name in bar_names]
            cumulatives = [
                _cumulative_energy_at_hour(cumulative_profiles[name], float(abs_hour))
                for name in bar_names
            ]
            positions = np.arange(len(names))

            bar_colors = [color_by_alias[alias] for alias in names]
            bars = ax_bar.barh(
                positions,
                cumulatives,
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
                if cumulative <= 0.0:
                    continue

                label_x = bar.get_width() + x_max * 0.015
                label_ha = "left"
                if label_x > x_max * 0.98:
                    label_x = max(bar.get_width() - x_max * 0.015, 0.0)
                    label_ha = "right"

                ax_bar.text(
                    label_x,
                    idx,
                    f"{cumulative:.1f}",
                    va="center",
                    ha=label_ha,
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
        default=Path("output/swopp3_submissions_compare"),
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


def _forward_bearing_deg_vec(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """Vectorised forward bearing in [0, 360) between consecutive points."""
    lat1r, lat2r = np.radians(lat1), np.radians(lat2)
    dlon = np.radians(lon2 - lon1)
    x = np.sin(dlon) * np.cos(lat2r)
    y = np.cos(lat1r) * np.sin(lat2r) - np.sin(lat1r) * np.cos(lat2r) * np.cos(dlon)
    return np.mod(np.degrees(np.arctan2(x, y)), 360.0)


def _rescore_route_counterfactual(df: pd.DataFrame, *, wps: bool) -> float:
    """Re-score a resampled route CSV with the opposite WPS performance model.

    Replicates the official scorer logic:
      1. Compute per-segment ship bearing from consecutive lat/lon.
      2. Derive TWA and MWA relative to that bearing.
      3. Run :func:`~routetools.performance.predict_power` with *wps* flag.
      4. Integrate power over each segment's Δt.

    Parameters
    ----------
    df : pd.DataFrame
        Resampled route CSV loaded into a DataFrame (columns: timestamp,
        lat, lon, wind, wind_u, wind_v, wave_h, wave_a, velocity, …).
    wps : bool
        WPS flag to apply (True = wingsails deployed, False = retracted).

    Returns
    -------
    float
        Total energy in MWh.
    """
    from routetools.performance import predict_power

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    n = len(df)
    if n < 2:
        return 0.0

    lats = df["lat"].to_numpy(dtype=float)
    lons = df["lon"].to_numpy(dtype=float)
    # Segments: 0..N-2 → 1..N-1
    bearing = _forward_bearing_deg_vec(lats[:-1], lons[:-1], lats[1:], lons[1:])

    # Per-segment dt in hours (from timestamps)
    times = df["timestamp"].values
    dt_h = times[1:].astype("datetime64[s]") - times[:-1].astype("datetime64[s]")
    dt_h = dt_h.astype(float) / 3600.0
    dt_h = np.maximum(dt_h, 1e-6)

    # Weather at each segment (stored at the start-of-segment row)
    wu = df["wind_u"].to_numpy(dtype=float)[:-1]
    wv = df["wind_v"].to_numpy(dtype=float)[:-1]
    tws = np.sqrt(wu**2 + wv**2)
    wind_from_deg = np.mod(180.0 + np.degrees(np.arctan2(wu, wv)), 360.0)
    twa_deg = np.mod(wind_from_deg - bearing, 360.0)

    wave_h = df["wave_h"].to_numpy(dtype=float)[:-1]
    mwd = df["wave_a"].to_numpy(dtype=float)[:-1]
    mwa_deg = np.mod(mwd - bearing, 360.0)

    v_mps = df["velocity"].to_numpy(dtype=float)[:-1]

    power_kw = np.array(
        [
            predict_power(
                float(tws[i]),
                float(twa_deg[i]),
                float(wave_h[i]),
                float(mwa_deg[i]),
                float(v_mps[i]),
                wps=wps,
            )
            for i in range(len(tws))
        ]
    )
    return float(np.sum(power_kw * dt_h) / 1000.0)


def plot_drprecious_counterfactual_table(
    submission: SubmissionData,
    *,
    score_archive: Path,
    out_dir: Path,
    dpi: int,
) -> None:
    """Generate the 4×4 counterfactual re-scoring table for drprecious.

    For each of the 4 cases (AO_WPS, AO_noWPS, PO_WPS, PO_noWPS) the
    participant's routes are re-scored with the *opposite* WPS model.
    The table has 4 rows (one per case) and 4 columns:

    * **Original (MWh)** — mean total energy under the submitted WPS setting
    * **Counterfactual (MWh)** — mean total energy under the opposite setting
    * **Δ (MWh)** — counterfactual − original
    * **Δ (%)** — relative change
    """
    score_zip = _load_resampled_tracks_zip(score_archive)
    route_members = _index_scored_route_members(score_zip)

    rows = []
    for case in REQUIRED_CASES:
        wps_flag: bool = bool(SWOPP3_CASES[case]["wps"])
        original_energies: list[float] = []
        counterfactual_energies: list[float] = []

        summary = submission.summaries[case].sort_values("departure_time_utc")
        for _, row in summary.iterrows():
            departure_key = _normalize_departure_key(row["departure_time_utc"])
            departure_date = departure_key.strftime("%Y%m%d")
            score_name = route_members[case].get(departure_date)
            if score_name is None:
                continue
            with score_zip.open(score_name) as fh:
                df_route = pd.read_csv(fh)

            original_energies.append(float(df_route["E"].sum()))
            counterfactual_energies.append(
                _rescore_route_counterfactual(df_route, wps=not wps_flag)
            )

        if not original_energies:
            continue

        orig_mean = float(np.mean(original_energies))
        cf_mean = float(np.mean(counterfactual_energies))
        delta = cf_mean - orig_mean
        delta_pct = delta / orig_mean * 100.0 if orig_mean > 0 else float("nan")
        rows.append(
            {
                "Case": CASE_LABELS.get(case, case),
                "Original (MWh)": round(orig_mean, 2),
                "Counterfactual (MWh)": round(cf_mean, 2),
                "Δ (MWh)": round(delta, 2),
                "Δ (%)": round(delta_pct, 1),
            }
        )

    score_zip.close()

    if not rows:
        print("Warning: No counterfactual data available for drprecious.")
        return

    table_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, 2.4 + 0.5 * len(rows)))
    ax.axis("off")
    tbl = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.2, 1.8)

    # Colour header
    for j in range(len(table_df.columns)):
        tbl[0, j].set_facecolor("#2c5f8a")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # Colour Δ(%) cells: green if negative (WPS helps), red if positive
    col_idx = list(table_df.columns).index("Δ (%)")
    for i, row_dict in enumerate(rows):
        dpct = row_dict["Δ (%)"]
        if not np.isnan(dpct):
            color = "#d4edda" if dpct < 0 else "#f8d7da"
            tbl[i + 1, col_idx].set_facecolor(color)

    ax.set_title(
        "drprecious – Counterfactual re-scoring (opposite WPS model)",
        fontsize=13,
        pad=18,
    )

    out_path = out_dir / "drprecious_counterfactual_table"
    _save_figure(fig, out_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved counterfactual table: {out_path}.pdf / .png")


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

        avg_consumption_by_name = avg_table[("all", "total")].to_dict()
        plot_consumption_vs_exploration_scatter(
            submissions,
            sampled_cache=sampled_cache,
            average_consumption_by_name=avg_consumption_by_name,
            alias_by_name=alias_by_name,
            color_by_alias=color_by_alias,
            out_dir=args.output_dir,
            dpi=args.dpi,
        )
        plot_consumption_vs_exploration_scatter(
            submissions,
            sampled_cache=sampled_cache,
            average_consumption_by_name=avg_consumption_by_name,
            alias_by_name=alias_by_name,
            color_by_alias=color_by_alias,
            out_dir=args.output_dir,
            dpi=args.dpi,
            selected_names=set(top5_participants),
            output_suffix="_top5",
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

        # Counterfactual re-scoring table for drprecious
        try:
            drprecious_subs = [
                s
                for s in submissions
                if _normalize_participant_token(s.name) == "drprecious"
            ]
            if drprecious_subs:
                dp_archive = score_archives_by_name.get(drprecious_subs[0].name)
                if dp_archive is not None:
                    plot_drprecious_counterfactual_table(
                        drprecious_subs[0],
                        score_archive=dp_archive,
                        out_dir=args.output_dir,
                        dpi=args.dpi,
                    )
        except Exception as e:
            print(f"Warning: Could not generate counterfactual table: {e}")

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
                    score_archives_by_name=score_archives_by_name,
                    alias_by_name=alias_by_name,
                    color_by_alias=color_by_alias,
                    output_path=args.output_dir
                    / f"animation_{animation_case}_{args.animation_departure}_top5",
                    wave_path=wave_path,
                    wind_path=wind_path,
                    dpi=args.dpi,
                )

            # Find and render representative summer/winter cases
            try:
                (
                    summer_case,
                    summer_departure,
                    winter_case,
                    winter_departure,
                ) = _find_representative_cases(
                    submissions, energy_lookup, top5_participants
                )
                print(
                    f"Summer case (low variance): {summer_case} "
                    f"({summer_departure.date()}), "
                    f"Winter case (high variance): {winter_case} "
                    f"({winter_departure.date()})"
                )

                for season, anim_case, season_dep in [
                    ("summer", summer_case, summer_departure),
                    ("winter", winter_case, winter_departure),
                ]:
                    corridor = _corridor_from_case(anim_case)
                    if corridor == "atlantic":
                        wind_path = args.wind_path_atlantic
                        wave_path = args.wave_path_atlantic
                    else:
                        wind_path = args.wind_path_pacific
                        wave_path = args.wave_path_pacific

                    print(
                        f"Rendering {season} animation for {anim_case} "
                        f"({season_dep.date()})..."
                    )
                    animate_departure(
                        [
                            sub
                            for sub in submissions
                            if sub.name in set(top5_participants)
                        ],
                        case=anim_case,
                        departure=season_dep,
                        energy_lookup=energy_lookup,
                        score_archives_by_name=score_archives_by_name,
                        alias_by_name=alias_by_name,
                        color_by_alias=color_by_alias,
                        output_path=args.output_dir
                        / f"animation_{season}_{anim_case}_top5",
                        wave_path=wave_path,
                        wind_path=wind_path,
                        dpi=args.dpi,
                    )
            except Exception as e:
                print(f"Warning: Could not generate seasonal animations: {e}")

            # Render WPS vs no-WPS comparison for drprecious
            try:
                drprecious_submissions = [
                    s
                    for s in submissions
                    if _normalize_participant_token(s.name) == "drprecious"
                ]
                if drprecious_submissions:
                    ref_participant = drprecious_submissions[0].name
                    for ocean, corridor in [("atlantic", "AO"), ("pacific", "PO")]:
                        wps_case = f"{corridor}_WPS"
                        nowps_case = f"{corridor}_noWPS"

                        if corridor == "AO":
                            wind_path = args.wind_path_atlantic
                            wave_path = args.wave_path_atlantic
                        else:
                            wind_path = args.wind_path_pacific
                            wave_path = args.wave_path_pacific

                        # Find departure with significant WPS impact
                        departure_wps = _find_departure_with_wps_impact(
                            submissions, energy_lookup, ref_participant, ocean
                        )

                        print(
                            f"Rendering WPS comparison for {ocean} "
                            f"({wps_case}) vs ({nowps_case})..."
                        )

                        # Single video: both WPS and no-WPS routes overlaid.
                        animate_wps_comparison(
                            drprecious_submissions[0],
                            wps_case=wps_case,
                            nowps_case=nowps_case,
                            departure=departure_wps,
                            energy_lookup=energy_lookup,
                            score_archives_by_name=score_archives_by_name,
                            alias_by_name=alias_by_name,
                            output_path=(
                                args.output_dir / f"animation_wps_comparison_{ocean}"
                                f"_{departure_wps.date()}"
                            ),
                            wave_path=wave_path,
                            wind_path=wind_path,
                            dpi=args.dpi,
                        )
            except Exception as e:
                print(f"Warning: Could not generate WPS comparison animations: {e}")


if __name__ == "__main__":
    main()
