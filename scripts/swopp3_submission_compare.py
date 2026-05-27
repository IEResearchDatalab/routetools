#!/usr/bin/env python
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
import re
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
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


@dataclass(frozen=True)
class SubmissionData:
    """Parsed and validated SWOPP3 submission."""

    name: str
    path: Path
    team_prefix: str
    tracks_dir: Path
    summaries: dict[str, pd.DataFrame]


@dataclass(frozen=True)
class CandidateIssue:
    """Validation issue found while scanning submission folders."""

    folder: str
    reason: str


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
    out_dir: Path,
    dpi: int,
) -> None:
    """Plot non-penalized consumption time series for each corridor/config."""
    for case in REQUIRED_CASES:
        fig, ax = plt.subplots(figsize=(12, 5))
        for submission in submissions:
            summary = submission.summaries[case].sort_values("departure_time_utc")
            ax.plot(
                summary["departure_time_utc"],
                summary["energy_cons_mwh"],
                linewidth=1.4,
                alpha=0.9,
                label=submission.name,
            )

        ax.set_title(f"Consumption Across 2024 Departures - {CASE_LABELS[case]}")
        ax.set_xlabel("Departure date")
        ax.set_ylabel("Energy consumption (MWh)")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", ncols=2, fontsize=9)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

        out = out_dir / f"consumption_{case}.pdf"
        _save_figure(fig, out, dpi=dpi)
        plt.close(fig)


def plot_participant_spread(
    submissions: list[SubmissionData],
    *,
    case: str,
    sampled_cache: dict[tuple[str, str], tuple[pd.DatetimeIndex, np.ndarray]],
    sample_hours: np.ndarray,
    out_dir: Path,
    dpi: int,
) -> None:
    """Plot spread vs time for each participant and case."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for submission in submissions:
        _, points = sampled_cache[(submission.name, case)]
        spreads = [
            _mean_pairwise_haversine_km(points[:, sample_idx, :])
            for sample_idx in range(points.shape[1])
        ]
        ax.plot(sample_hours, spreads, marker="o", label=submission.name)

    ax.set_title(f"Participant Spread Comparison - {CASE_LABELS[case]}")
    ax.set_xlabel("Elapsed time (hours)")
    ax.set_ylabel("Mean pairwise waypoint distance (km)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", ncols=2, fontsize=9)

    out = out_dir / f"spread_participants_{case}.pdf"
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
) -> None:
    """Plot cross-participant spread grouped by month for each case."""
    per_submission: dict[str, tuple[pd.DatetimeIndex, np.ndarray]] = {
        sub.name: sampled_cache[(sub.name, case)] for sub in submissions
    }

    common_dates = set(per_submission[submissions[0].name][0])
    for sub in submissions[1:]:
        common_dates &= set(per_submission[sub.name][0])
    aligned_dates = sorted(common_dates)

    if not aligned_dates:
        return

    date_to_row = {
        sub.name: {dt: idx for idx, dt in enumerate(per_submission[sub.name][0])}
        for sub in submissions
    }

    by_month: dict[int, list[np.ndarray]] = {month: [] for month in range(1, 13)}
    for date in aligned_dates:
        month = date.month
        curve = np.zeros(len(sample_hours), dtype=float)

        for sample_idx in range(len(sample_hours)):
            points = np.array(
                [
                    per_submission[sub.name][1][
                        date_to_row[sub.name][date],
                        sample_idx,
                        :,
                    ]
                    for sub in submissions
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

    out = out_dir / f"spread_months_{case}.pdf"
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
        tracks_by_submission[sub.name] = _read_track(sub.tracks_dir / details)
        total_energy[sub.name] = float(row.iloc[0]["energy_cons_mwh"])

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
        grid = fig.add_gridspec(2, 1, height_ratios=[5.8, 1.2], hspace=0.06)
        ax_map = fig.add_subplot(grid[0, 0])
        ax_legend = fig.add_subplot(grid[1, 0])
        ax_legend.axis("off")

        ax_map.set_xlim(lon_min, lon_max)
        ax_map.set_ylim(lat_min, lat_max)
        ax_map.set_facecolor("#0d2538")
        ax_map.set_xlabel("Longitude (deg)")
        ax_map.set_ylabel("Latitude (deg)")
        ax_map.grid(alpha=0.25, color="#9cb4c3", linestyle="--", linewidth=0.5)

        sub_names = list(tracks_by_submission)
        colors = plt.cm.tab10(np.linspace(0, 1, len(sub_names)))
        line_handles: dict[str, plt.Line2D] = {}
        vessel_handles: dict[str, plt.Line2D] = {}
        vessel_shadow_handles: dict[str, plt.Line2D] = {}
        for idx, name in enumerate(sub_names):
            (line,) = ax_map.plot(
                [],
                [],
                color=colors[idx],
                linewidth=2.0,
                alpha=0.92,
                solid_capstyle="round",
                label=name,
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
                markerfacecolor=colors[idx],
                markeredgecolor="white",
                markeredgewidth=1.2,
                zorder=9,
            )
            vessel_handles[name] = vessel

        wave_mesh = None
        wind_quiver = None
        info_text = ax_legend.text(
            0.01,
            0.95,
            "",
            va="top",
            ha="left",
            fontsize=10.5,
            family="monospace",
            bbox={
                "boxstyle": "round,pad=0.4",
                "facecolor": "#f4f7fb",
                "edgecolor": "#d0d7de",
                "alpha": 0.9,
            },
        )

        best_name = min(total_energy, key=total_energy.get)

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
            top_five = ranking_rows[:5]
            lines = [
                "Live Fuel Leaderboard (Top 5)",
                "Rank  Participant                 Cum(MWh)   Total(MWh)",
                "----  --------------------------  --------   ----------",
            ]
            for rank, (name, cumulative, total) in enumerate(top_five, start=1):
                lines.append(
                    f"{rank:>4}  {name:<26}  {cumulative:8.2f}   {total:10.2f}"
                )

            if abs_hour >= case_hours:
                lines.append("")
                lines.append(
                    f"WINNER: {best_name} ({total_energy[best_name]:.2f} MWh total)"
                )
                ax_map.set_title(
                    f"{CASE_LABELS[case]} - {departure_utc.date()} - "
                    f"Final (winner: {best_name})"
                )
            else:
                ax_map.set_title(
                    f"{CASE_LABELS[case]} - {departure_utc.date()} - +{abs_hour:03d}h"
                )

            info_text.set_text("\n".join(lines))

            return [
                *line_handles.values(),
                *vessel_shadow_handles.values(),
                *vessel_handles.values(),
                info_text,
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
        help="Pacific ERA5 wave dataset used for animation.",
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

        plot_consumption(submissions, out_dir=args.output_dir, dpi=args.dpi)

        for case in REQUIRED_CASES:
            plot_participant_spread(
                submissions,
                case=case,
                sampled_cache=sampled_cache,
                sample_hours=sample_hours_by_case[case],
                out_dir=args.output_dir,
                dpi=args.dpi,
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
                    submissions,
                    case=animation_case,
                    departure=departure,
                    output_path=args.output_dir
                    / f"animation_{animation_case}_{args.animation_departure}",
                    wave_path=wave_path,
                    wind_path=wind_path,
                    dpi=args.dpi,
                )


if __name__ == "__main__":
    main()
