"""Task 4 (R1-7) — weather threshold violation statistics.

For every SWOPP3 real-ocean route (366 departures x 4 corridor/WPS
configurations), compares the BERS-optimised route (``AO``/``PO``) against
the great-circle baseline (``AGC``/``PGC``) in terms of:

- Frequency: % of departures with at least one wind (TWS > 20 m/s) or wave
  (Hs > 7 m) violation.
- Magnitude: mean/max exceedance above the threshold among violating
  segments.
- Duration: mean/max time spent in violation per voyage (hours).

Per-segment TWS/Hs are sampled at segment midpoints using the real
departure-relative timestamps recorded in each track file (not an assumed
uniform schedule), consistent with ``routetools.weather`` conventions.

Outputs
-------
- ``revision/task4_route_metrics.csv``: one row per route with per-route
  violation stats.
- ``revision/task4_violations_table.csv`` / ``.tex``: aggregated table.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import typer

from routetools.swopp3 import SWOPP3_CASES
from routetools.violations import (
    CorridorWeatherResources,
    departure_offset_hours,
    find_team_prefix,
    is_gc_case,
    load_default_weather_resources,
    read_track_curve,
)
from routetools.weather import DEFAULT_HS_LIMIT, DEFAULT_TWS_LIMIT

REPO_ROOT = Path(__file__).resolve().parent.parent
_DTFMT = "%Y-%m-%d %H:%M:%S"

CASES = [
    "AO_WPS",
    "AO_noWPS",
    "AGC_WPS",
    "AGC_noWPS",
    "PO_WPS",
    "PO_noWPS",
    "PGC_WPS",
    "PGC_noWPS",
]


def _segment_midpoint_stats(
    curve: jnp.ndarray,
    times: list[datetime],
    resources: CorridorWeatherResources,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (tws_mid, hs_mid, dt_hours) at each segment midpoint.

    Uses the real per-waypoint timestamps recorded in the track file (rather
    than assuming a uniform schedule) to compute midpoint times and segment
    durations.
    """
    lon = curve[:, 0]
    lat = curve[:, 1]
    mid_lon = (lon[:-1] + lon[1:]) / 2
    mid_lat = (lat[:-1] + lat[1:]) / 2

    offsets_hours = np.array(
        [departure_offset_hours(t, resources.dataset_epoch) for t in times]
    )
    mid_t = (offsets_hours[:-1] + offsets_hours[1:]) / 2
    dt_hours = np.diff(offsets_hours)

    u10, v10 = resources.windfield(mid_lon, mid_lat, jnp.asarray(mid_t))
    tws = np.asarray(jnp.sqrt(u10**2 + v10**2))
    hs, _ = resources.wavefield(mid_lon, mid_lat, jnp.asarray(mid_t))
    hs = np.asarray(hs)

    return tws, hs, dt_hours


def _read_track_times(track_path: Path) -> list[datetime]:
    """Read the ``time_utc`` column of a track CSV as datetimes."""
    with track_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return [datetime.strptime(row["time_utc"], _DTFMT) for row in reader]


def compute_route_metrics(
    input_dir: Path,
    weather_resources: dict[str, CorridorWeatherResources],
) -> list[dict[str, object]]:
    """Compute per-route weather violation metrics for all SWOPP3 cases."""
    team_prefix = find_team_prefix(input_dir)
    tracks_dir = input_dir / "tracks"
    rows: list[dict[str, object]] = []

    for case_id in CASES:
        summary_path = input_dir / f"{team_prefix}-{case_id}.csv"
        if not summary_path.exists():
            print(f"  [skip] missing summary CSV: {summary_path}")
            continue

        case = SWOPP3_CASES[case_id]
        corridor = str(case["route"])
        resources = weather_resources[corridor]
        strategy = "GC" if is_gc_case(case_id) else "BERS"
        wps = "WPS" if case["wps"] else "noWPS"

        with summary_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                track_path = tracks_dir / row["details_filename"]
                curve = read_track_curve(track_path)
                times = _read_track_times(track_path)
                if curve.shape[0] < 2:
                    continue

                tws, hs, dt_hours = _segment_midpoint_stats(curve, times, resources)

                wind_viol = tws > DEFAULT_TWS_LIMIT
                wave_viol = hs > DEFAULT_HS_LIMIT
                wind_exceed = np.maximum(tws - DEFAULT_TWS_LIMIT, 0.0)
                wave_exceed = np.maximum(hs - DEFAULT_HS_LIMIT, 0.0)

                rows.append(
                    {
                        "case_id": case_id,
                        "corridor": corridor,
                        "strategy": strategy,
                        "wps": wps,
                        "route_id": row["details_filename"],
                        "wind_has_violation": bool(np.any(wind_viol)),
                        "wind_max_exceedance": float(np.max(wind_exceed)),
                        "wind_duration_h": float(np.sum(dt_hours[wind_viol])),
                        "wave_has_violation": bool(np.any(wave_viol)),
                        "wave_max_exceedance": float(np.max(wave_exceed)),
                        "wave_duration_h": float(np.sum(dt_hours[wave_viol])),
                    }
                )
    return rows


def _agg_stats(values: list[float]) -> tuple[float, float]:
    """Return (mean, max) of a list, or (0.0, 0.0) if empty."""
    if not values:
        return 0.0, 0.0
    arr = np.asarray(values)
    return float(arr.mean()), float(arr.max())


def build_summary_table(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Aggregate per-route metrics into frequency/magnitude/duration stats."""
    summary: list[dict[str, object]] = []
    group_keys = sorted({(r["corridor"], r["wps"], r["strategy"]) for r in rows})

    for corridor, wps, strategy in group_keys:
        group = [
            r
            for r in rows
            if r["corridor"] == corridor
            and r["wps"] == wps
            and r["strategy"] == strategy
        ]
        n = len(group)

        for hazard in ["wind", "wave"]:
            has_viol_key = f"{hazard}_has_violation"
            exceed_key = f"{hazard}_max_exceedance"
            dur_key = f"{hazard}_duration_h"

            n_viol = sum(1 for r in group if r[has_viol_key])
            freq_pct = 100.0 * n_viol / n if n else 0.0

            viol_group = [r for r in group if r[has_viol_key]]
            mean_exceed, max_exceed = _agg_stats(
                [float(r[exceed_key]) for r in viol_group]
            )
            mean_dur, max_dur = _agg_stats([float(r[dur_key]) for r in viol_group])

            summary.append(
                {
                    "corridor": corridor,
                    "wps": wps,
                    "strategy": strategy,
                    "hazard": hazard,
                    "n_departures": n,
                    "n_with_violation": n_viol,
                    "freq_pct": round(freq_pct, 2),
                    "mean_max_exceedance": round(mean_exceed, 4),
                    "max_exceedance": round(max_exceed, 4),
                    "mean_duration_h": round(mean_dur, 3),
                    "max_duration_h": round(max_dur, 3),
                }
            )
    return summary


def write_latex_table(summary: list[dict[str, object]], out_path: Path) -> None:
    """Write a LaTeX table comparing GC vs BERS wind/wave violation stats."""
    corridors = sorted({s["corridor"] for s in summary})

    def _get(corridor: str, wps: str, strategy: str, hazard: str) -> dict:
        for s in summary:
            if (
                s["corridor"] == corridor
                and s["wps"] == wps
                and s["strategy"] == strategy
                and s["hazard"] == hazard
            ):
                return s
        return {
            "freq_pct": 0.0,
            "mean_max_exceedance": 0.0,
            "mean_duration_h": 0.0,
            "max_duration_h": 0.0,
        }

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Weather threshold violation statistics: great-circle (GC) "
        r"baseline vs BERS-optimised routes.}",
        r"\label{tab:weather_violations}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
    ]
    for corridor in corridors:
        lines.append(
            rf"\multicolumn{{5}}{{l}}{{\textbf{{{corridor.capitalize()}}}}} \\"
        )
        lines.append(r" & GC (noWPS) & BERS (noWPS) & GC (WPS) & BERS (WPS) \\")
        for hazard, label in [("wind", "Wind > 20 m/s"), ("wave", "Wave Hs > 7 m")]:
            lines.append(rf"\multicolumn{{5}}{{l}}{{\textit{{{label}}}}} \\")
            row_freq = ["Departures with violation (\\%)"]
            row_mag = ["Mean max exceedance"]
            row_dur = ["Mean duration per voyage (h)"]
            for wps in ["noWPS", "WPS"]:
                for strategy in ["GC", "BERS"]:
                    d = _get(corridor, wps, strategy, hazard)
                    row_freq.append(f"{d['freq_pct']:.1f}")
                    row_mag.append(f"{d['mean_max_exceedance']:.2f}")
                    row_dur.append(f"{d['mean_duration_h']:.2f}")
            lines.append(" & ".join(row_freq) + r" \\")
            lines.append(" & ".join(row_mag) + r" \\")
            lines.append(" & ".join(row_dur) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    out_path.write_text("\n".join(lines) + "\n")


def main(
    real_ocean_dir: str = "output/sweep_combined_fms",
    output_dir: str = "revision",
) -> None:
    """Run Task 4 weather violation analysis and write CSV/LaTeX outputs."""
    out_dir = REPO_ROOT / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading ERA5 weather resources for both corridors...")
    weather_resources = load_default_weather_resources(
        wind_path_atlantic=REPO_ROOT / "data/era5/era5_wind_atlantic_2024.nc",
        wave_path_atlantic=REPO_ROOT / "data/era5/era5_waves_atlantic_2024.nc",
        wind_path_pacific=REPO_ROOT / "data/era5/era5_wind_pacific_2024.nc",
        wave_path_pacific=REPO_ROOT / "data/era5/era5_waves_pacific_2024.nc",
    )

    print(f"Computing per-route weather metrics from {real_ocean_dir} ...")
    rows = compute_route_metrics(REPO_ROOT / real_ocean_dir, weather_resources)

    route_csv = out_dir / "task4_route_metrics.csv"
    fieldnames = list(rows[0].keys())
    with route_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} route rows to {route_csv}")

    summary = build_summary_table(rows)
    summary_csv = out_dir / "task4_violations_table.csv"
    with summary_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    print(f"Wrote summary table to {summary_csv}")

    tex_path = out_dir / "task4_violations_table.tex"
    write_latex_table(summary, tex_path)
    print(f"Wrote LaTeX table to {tex_path}")

    print("\nSummary:")
    for s in summary:
        print(
            f"  {s['corridor']:8s} {s['wps']:6s} {s['strategy']:4s} {s['hazard']:4s}: "
            f"freq={s['freq_pct']:5.1f}%  mean_exceed={s['mean_max_exceedance']:.2f}  "
            f"mean_dur_h={s['mean_duration_h']:.2f}"
        )


if __name__ == "__main__":
    typer.run(main)
