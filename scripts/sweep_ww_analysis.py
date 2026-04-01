#!/usr/bin/env python
"""Analyse wind×wave sweep results with per-evaluation-point violations.

Reads the track CSVs from each completed sweep directory, resamples them
to Δt₂ = 30 min, queries wind and wave fields at each evaluation point,
and reports violations as a percentage of total evaluation points.

Usage
-----
    python scripts/sweep_ww_analysis.py \
        --data-dir /data/fjsuarez/era5 \
        --output-root output

Produces a summary CSV at ``output/sweep_ww_summary.csv``.
"""

from __future__ import annotations

import csv
import os
import re
import statistics
from datetime import datetime
from pathlib import Path

import numpy as np
import typer

# Thresholds
TWS_LIMIT = 20.0  # m/s
HS_LIMIT = 7.0  # m

# Evaluation resolution
DT2_MINUTES = 30.0

CORRIDOR_CASES = {
    "atlantic": ["AO_WPS", "AO_noWPS", "AGC_WPS", "AGC_noWPS"],
    "pacific": ["PO_WPS", "PO_noWPS", "PGC_WPS", "PGC_noWPS"],
}

PASSAGE_HOURS = {
    "atlantic": 354,
    "pacific": 583,
}


def _load_fields(data_dir: Path, corridor: str):
    """Load wind and wave field closures for a corridor.

    Returns (windfield, wavefield, epoch) where epoch is the dataset
    time origin as a datetime.
    """
    from routetools.era5.loader import (
        load_dataset_epoch,
        load_era5_wavefield,
        load_era5_windfield,
    )

    wind_base = data_dir / f"era5_wind_{corridor}_2024.nc"
    wave_base = data_dir / f"era5_waves_{corridor}_2024.nc"
    wind_cont = data_dir / f"era5_wind_{corridor}_2025_01.nc"
    wave_cont = data_dir / f"era5_waves_{corridor}_2025_01.nc"

    wind_paths = [wind_base]
    if wind_cont.exists():
        wind_paths.append(wind_cont)
    wave_paths = [wave_base]
    if wave_cont.exists():
        wave_paths.append(wave_cont)

    wind_target = wind_paths if len(wind_paths) > 1 else wind_paths[0]
    wave_target = wave_paths if len(wave_paths) > 1 else wave_paths[0]

    epoch = load_dataset_epoch(wind_target)
    windfield = load_era5_windfield(wind_target)
    wavefield = load_era5_wavefield(wave_target)
    return windfield, wavefield, epoch


def _parse_departure_time(filename: str) -> datetime:
    """Extract departure date from track filename like ...-20240315.csv."""
    m = re.search(r"-(\d{8})\.csv$", filename)
    if m is None:
        raise ValueError(f"Cannot parse departure date from {filename}")
    d = m.group(1)
    # Return naive datetime matching the epoch format from load_dataset_epoch
    return datetime(int(d[:4]), int(d[4:6]), int(d[6:8]), 12)


def _corridor_for_case(case_id: str) -> str:
    for corridor, cases in CORRIDOR_CASES.items():
        if case_id in cases:
            return corridor
    raise ValueError(f"Unknown case: {case_id}")


def _analyse_tracks(
    track_dir: Path,
    case_id: str,
    windfield,
    wavefield,
    field_t0: datetime,
) -> dict:
    """Analyse all departure tracks for one case.

    Returns
    -------
    dict with keys: case, n_departures, n_total_points,
    wind_violations, wave_violations, wind_pct, wave_pct,
    mean_energy_mwh, mean_dist_nm
    """
    import jax.numpy as jnp

    from routetools.resample import resample_track

    pattern = re.compile(rf"IEUniversity-1-{re.escape(case_id)}-\d{{8}}\.csv$")
    track_files = sorted(f for f in os.listdir(track_dir) if pattern.match(f))

    total_points = 0
    wind_viol_points = 0
    wave_viol_points = 0

    for tf in track_files:
        # Read track waypoints
        with open(track_dir / tf) as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        if len(rows) < 2:
            continue

        # Parse track columns
        times_str = [r["time_utc"] for r in rows]
        lats = np.array([float(r["lat_deg"]) for r in rows])
        lons = np.array([float(r["lon_deg"]) for r in rows])
        times = np.array(
            [np.datetime64(t.replace(" ", "T")) for t in times_str],
            dtype="datetime64[ns]",
        )

        # Resample to Δt₂
        t_eval, lat_eval, lon_eval = resample_track(
            times, lats, lons, dt_minutes=DT2_MINUTES
        )

        n_pts = len(lat_eval)

        # Compute time offsets from field origin
        t0_ns = np.datetime64(field_t0.strftime("%Y-%m-%dT%H:%M:%S"), "ns")
        t_offset_h = (t_eval - t0_ns).astype("timedelta64[ns]").astype(np.float64) / (
            3600.0 * 1e9
        )

        # Query fields at evaluation points
        lon_j = jnp.array(np.asarray(lon_eval))
        lat_j = jnp.array(np.asarray(lat_eval))
        t_j = jnp.array(np.asarray(t_offset_h))

        if windfield is not None:
            u10, v10 = windfield(lon_j, lat_j, t_j)
            tws = np.sqrt(np.asarray(u10) ** 2 + np.asarray(v10) ** 2)
            wind_viol_points += int(np.sum(tws > TWS_LIMIT))
        if wavefield is not None:
            hs, _ = wavefield(lon_j, lat_j, t_j)
            hs = np.asarray(hs)
            wave_viol_points += int(np.sum(hs > HS_LIMIT))

        total_points += n_pts

    return {
        "n_departures": len(track_files),
        "n_total_points": total_points,
        "wind_violations": wind_viol_points,
        "wave_violations": wave_viol_points,
        "wind_pct": 100.0 * wind_viol_points / total_points if total_points else 0.0,
        "wave_pct": 100.0 * wave_viol_points / total_points if total_points else 0.0,
    }


def _read_summary_csv(fpath: Path) -> dict:
    """Read energy and distance from the per-case summary CSV."""
    with open(fpath) as fh:
        rows = list(csv.DictReader(fh))
    energies = [float(r["energy_cons_mwh"]) for r in rows]
    dists = [float(r["sailed_distance_nm"]) for r in rows]
    return {
        "mean_energy_mwh": statistics.mean(energies),
        "mean_dist_nm": statistics.mean(dists),
    }


app = typer.Typer()


@app.command()
def main(
    data_dir: Path = typer.Option(  # noqa: B008
        "data/era5", "--data-dir", help="Directory with ERA5 NetCDF files."
    ),
    output_root: Path = typer.Option(  # noqa: B008
        "output", "--output-root", help="Root output directory."
    ),
    csv_out: Path = typer.Option(  # noqa: B008
        "output/sweep_ww_summary.csv",
        "--csv-out",
        help="Path for summary CSV.",
    ),
):
    """Analyse wind×wave sweep results."""
    sweep_dirs = sorted(
        d
        for d in os.listdir(output_root)
        if d.startswith("swopp3_ww_") and (output_root / d / "tracks").is_dir()
    )

    if not sweep_dirs:
        print("No completed sweep directories found.")
        raise typer.Exit(1)

    print(f"Found {len(sweep_dirs)} sweep directories.")

    # Parse wind/wave identifiers
    dir_re = re.compile(r"swopp3_ww_w(\d+)_v(\d+)")

    # Load fields per corridor (reuse across configs)
    fields: dict[str, tuple] = {}
    field_t0s: dict[str, datetime] = {}
    for corridor in ("atlantic", "pacific"):
        print(f"Loading {corridor} fields from {data_dir} ...")
        wf, vf, epoch = _load_fields(data_dir, corridor)
        fields[corridor] = (wf, vf)
        field_t0s[corridor] = epoch

    all_cases = [
        "AO_WPS",
        "AO_noWPS",
        "AGC_WPS",
        "AGC_noWPS",
        "PO_WPS",
        "PO_noWPS",
        "PGC_WPS",
        "PGC_noWPS",
    ]

    header = [
        "wind_pw",
        "wave_pw",
        "case",
        "mean_energy_mwh",
        "mean_dist_nm",
        "n_departures",
        "n_eval_points",
        "wind_violations",
        "wave_violations",
        "wind_pct",
        "wave_pct",
    ]

    results = []

    for d in sweep_dirs:
        m = dir_re.match(d)
        if m is None:
            continue
        wind_pw = int(m.group(1))
        wave_pw = int(m.group(2))
        print(f"\n=== {d}  (wind={wind_pw}, wave={wave_pw}) ===")

        for case_id in all_cases:
            summary_path = output_root / d / f"IEUniversity-1-{case_id}.csv"
            track_dir = output_root / d / "tracks"

            if not summary_path.exists():
                continue

            corridor = _corridor_for_case(case_id)
            windfield, wavefield = fields[corridor]
            field_t0 = field_t0s[corridor]

            # Energy/distance from summary
            summary = _read_summary_csv(summary_path)

            # Per-point violations from tracks
            print(f"  Analysing {case_id} tracks ...")
            stats = _analyse_tracks(track_dir, case_id, windfield, wavefield, field_t0)

            row = {
                "wind_pw": wind_pw,
                "wave_pw": wave_pw,
                "case": case_id,
                "mean_energy_mwh": f"{summary['mean_energy_mwh']:.2f}",
                "mean_dist_nm": f"{summary['mean_dist_nm']:.2f}",
                "n_departures": stats["n_departures"],
                "n_eval_points": stats["n_total_points"],
                "wind_violations": stats["wind_violations"],
                "wave_violations": stats["wave_violations"],
                "wind_pct": f"{stats['wind_pct']:.3f}",
                "wave_pct": f"{stats['wave_pct']:.3f}",
            }
            results.append(row)
            print(
                f"    E={summary['mean_energy_mwh']:.1f} MWh  "
                f"wind={stats['wind_pct']:.2f}%  "
                f"wave={stats['wave_pct']:.2f}%"
            )

    # Write summary CSV
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSummary written to {csv_out}")


if __name__ == "__main__":
    app()
