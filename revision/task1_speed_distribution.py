"""Task 1 (R1-4) — segment speed distribution for BERS real-ocean routes.

For all 366 real-ocean BERS-optimised departures (4 configurations: Atlantic
and Pacific, with and without WPS), computes the implied segment speed
between every pair of consecutive waypoints:

    distance_nm = haversine(x_n, x_{n+1})   [nautical miles]
    dt_hours    = t_{n+1} - t_n              [hours, from the track schedule]
    speed_kn    = distance_nm / dt_hours

Aggregates all segment speeds per configuration and reports
min/P5/mean/median/P95/max, to check that BERS routes stay within a
physically reasonable operational envelope (~5-12 kn for the 88 m vessel).

Outputs
-------
- ``revision/task1_segment_speeds.csv``: one row per segment (all routes).
- ``revision/task1_speed_distribution.csv`` / ``.tex``: aggregated table.
- ``revision/task1_speed_distribution.pdf``: box plot per configuration.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import typer

from routetools._cost.haversine import NAUTICAL_MILE_METERS, haversine_meters_module
from routetools.violations import find_team_prefix

REPO_ROOT = Path(__file__).resolve().parent.parent
_DTFMT = "%Y-%m-%d %H:%M:%S"

CONFIGURATIONS = {
    "Atlantic, no WPS": "AO_noWPS",
    "Atlantic, WPS": "AO_WPS",
    "Pacific, no WPS": "PO_noWPS",
    "Pacific, WPS": "PO_WPS",
}

# Reviewer's expected operational envelope for the 88 m cargo vessel.
ENVELOPE_MIN_KN = 5.0
ENVELOPE_MAX_KN = 12.0
OUTLIER_LOW_KN = 2.0
OUTLIER_HIGH_KN = 15.0


def _read_track_rows(track_path: Path) -> list[tuple[datetime, float, float]]:
    """Read a track CSV into a list of (time, lat, lon) tuples."""
    with track_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            (
                datetime.strptime(row["time_utc"], _DTFMT),
                float(row["lat_deg"]),
                float(row["lon_deg"]),
            )
            for row in reader
        ]


def compute_segment_speeds(
    input_dir: Path, config_label: str, case_id: str
) -> list[dict[str, object]]:
    """Compute per-segment implied speeds for every departure of one case."""
    team_prefix = find_team_prefix(input_dir)
    tracks_dir = input_dir / "tracks"
    summary_path = input_dir / f"{team_prefix}-{case_id}.csv"

    rows: list[dict[str, object]] = []
    with summary_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            track_path = tracks_dir / row["details_filename"]
            waypoints = _read_track_rows(track_path)
            for (t0, lat0, lon0), (t1, lat1, lon1) in zip(
                waypoints[:-1], waypoints[1:], strict=False
            ):
                dt_hours = (t1 - t0).total_seconds() / 3600.0
                if dt_hours <= 0:
                    continue
                distance_nm = (
                    haversine_meters_module(lat0, lon0, lat1, lon1)
                    / NAUTICAL_MILE_METERS
                )
                speed_kn = distance_nm / dt_hours
                rows.append(
                    {
                        "configuration": config_label,
                        "case_id": case_id,
                        "route_id": row["details_filename"],
                        "dt_hours": dt_hours,
                        "distance_nm": distance_nm,
                        "speed_kn": speed_kn,
                    }
                )
    return rows


def _percentile_stats(values: np.ndarray) -> dict[str, float]:
    return {
        "min_kn": float(values.min()),
        "p5_kn": float(np.percentile(values, 5)),
        "mean_kn": float(values.mean()),
        "median_kn": float(np.median(values)),
        "p95_kn": float(np.percentile(values, 95)),
        "max_kn": float(values.max()),
    }


def build_distribution_table(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Aggregate segment speeds into a summary table per configuration."""
    summary: list[dict[str, object]] = []
    for config_label in CONFIGURATIONS:
        speeds = np.array(
            [r["speed_kn"] for r in rows if r["configuration"] == config_label]
        )
        stats = _percentile_stats(speeds)
        n_outlier_low = int(np.sum(speeds < OUTLIER_LOW_KN))
        n_outlier_high = int(np.sum(speeds > OUTLIER_HIGH_KN))
        summary.append(
            {
                "configuration": config_label,
                "n_segments": len(speeds),
                **{k: round(v, 3) for k, v in stats.items()},
                "n_outlier_low": n_outlier_low,
                "n_outlier_high": n_outlier_high,
            }
        )
    return summary


def write_latex_table(summary: list[dict[str, object]], out_path: Path) -> None:
    """Write the speed-distribution table in LaTeX."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Implied segment speed distribution for BERS-optimised "
        r"real-ocean routes.}",
        r"\label{tab:speed_distribution}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"\textbf{Configuration} & \textbf{Min} & \textbf{P5} & \textbf{Mean} "
        r"& \textbf{Median} & \textbf{P95} & \textbf{Max} (kn) \\",
        r"\midrule",
    ]
    for s in summary:
        lines.append(
            f"{s['configuration']} & {s['min_kn']:.2f} & {s['p5_kn']:.2f} & "
            f"{s['mean_kn']:.2f} & {s['median_kn']:.2f} & {s['p95_kn']:.2f} & "
            f"{s['max_kn']:.2f} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    out_path.write_text("\n".join(lines) + "\n")


def plot_speed_distribution(rows: list[dict[str, object]], out_path: Path) -> None:
    """Save a box plot of segment speeds per configuration."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    data = [
        [r["speed_kn"] for r in rows if r["configuration"] == label]
        for label in CONFIGURATIONS
    ]
    ax.boxplot(data, tick_labels=list(CONFIGURATIONS), showfliers=True)
    ax.axhspan(ENVELOPE_MIN_KN, ENVELOPE_MAX_KN, color="green", alpha=0.1)
    ax.set_ylabel("Implied segment speed (kn)")
    ax.set_title("BERS real-ocean segment speed distribution")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main(
    real_ocean_dir: str = "output/sweep_combined_fms_strict",
    output_dir: str = "revision",
) -> None:
    """Run Task 1 speed-distribution analysis and write outputs."""
    out_dir = REPO_ROOT / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    input_dir = REPO_ROOT / real_ocean_dir

    all_rows: list[dict[str, object]] = []
    for config_label, case_id in CONFIGURATIONS.items():
        print(f"Computing segment speeds for {config_label} ({case_id}) ...")
        all_rows.extend(compute_segment_speeds(input_dir, config_label, case_id))

    segments_csv = out_dir / "task1_segment_speeds.csv"
    with segments_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Wrote {len(all_rows)} segments to {segments_csv}")

    summary = build_distribution_table(all_rows)
    summary_csv = out_dir / "task1_speed_distribution.csv"
    with summary_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    print(f"Wrote summary table to {summary_csv}")

    tex_path = out_dir / "task1_speed_distribution.tex"
    write_latex_table(summary, tex_path)
    print(f"Wrote LaTeX table to {tex_path}")

    pdf_path = out_dir / "task1_speed_distribution.pdf"
    plot_speed_distribution(all_rows, pdf_path)
    print(f"Wrote figure to {pdf_path}")

    print("\nSummary:")
    for s in summary:
        flag = ""
        if s["n_outlier_low"] or s["n_outlier_high"]:
            flag = (
                f"  <-- FLAG: {s['n_outlier_low']} segments < {OUTLIER_LOW_KN} kn, "
                f"{s['n_outlier_high']} segments > {OUTLIER_HIGH_KN} kn"
            )
        print(
            f"  {s['configuration']:18s} n={s['n_segments']:6d}  "
            f"min={s['min_kn']:.2f} p5={s['p5_kn']:.2f} mean={s['mean_kn']:.2f} "
            f"median={s['median_kn']:.2f} p95={s['p95_kn']:.2f} max={s['max_kn']:.2f}"
            f"{flag}"
        )


if __name__ == "__main__":
    typer.run(main)
