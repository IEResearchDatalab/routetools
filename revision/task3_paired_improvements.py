"""Task 3 (R1-6) — paired per-departure improvements.

For the 366 real-ocean departures across all 4 configurations, computes,
per departure ``d``:

    improvement_vs_GC[d]    = (E_GC[d] - E_BERS[d]) / E_GC[d] * 100      (%)
    improvement_vs_CMAES[d] = (E_CMAES[d] - E_BERS[d]) / E_CMAES[d] * 100 (%)

where:
- E_GC     comes from the great-circle baseline (AGC/PGC), output/sweep_combined_fms
- E_CMAES  comes from the CMA-ES-only run (AO/PO), output/sweep_combined
- E_BERS   comes from the CMA-ES+FMS run (AO/PO), output/sweep_combined_fms

Reports summary statistics (mean, std, median, P5, P95, min, max) of these
paired improvements per configuration, to show BERS improves *consistently*
across departures (not just on average).

Outputs
-------
- ``revision/task3_paired_metrics.csv``: one row per departure with the raw
  energies and paired improvements.
- ``revision/task3_paired_improvements.csv`` / ``.tex``: aggregated table.
- ``revision/task3_paired_improvements.pdf``: per-departure scatter/histogram.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import typer

from routetools.violations import find_team_prefix

REPO_ROOT = Path(__file__).resolve().parent.parent

# (label, optimised_case_id, gc_case_id)
CONFIGURATIONS = {
    "Atlantic, no WPS": ("AO_noWPS", "AGC_noWPS"),
    "Atlantic, WPS": ("AO_WPS", "AGC_WPS"),
    "Pacific, no WPS": ("PO_noWPS", "PGC_noWPS"),
    "Pacific, WPS": ("PO_WPS", "PGC_WPS"),
}


def _read_energy_by_departure(summary_path: Path) -> dict[str, float]:
    """Return {departure_time_utc: energy_cons_mwh} for one summary CSV."""
    with summary_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return {
            row["departure_time_utc"]: float(row["energy_cons_mwh"]) for row in reader
        }


def compute_paired_metrics(bers_dir: Path, cmaes_dir: Path) -> list[dict[str, object]]:
    """Join GC/CMA-ES/BERS energies per departure and compute % improvements."""
    bers_team = find_team_prefix(bers_dir)
    cmaes_team = find_team_prefix(cmaes_dir)

    rows: list[dict[str, object]] = []
    for config_label, (opt_case, gc_case) in CONFIGURATIONS.items():
        e_bers = _read_energy_by_departure(bers_dir / f"{bers_team}-{opt_case}.csv")
        e_gc = _read_energy_by_departure(bers_dir / f"{bers_team}-{gc_case}.csv")
        e_cmaes = _read_energy_by_departure(cmaes_dir / f"{cmaes_team}-{opt_case}.csv")

        departures = sorted(set(e_bers) & set(e_gc) & set(e_cmaes))
        missing = (set(e_bers) | set(e_gc) | set(e_cmaes)) - set(departures)
        if missing:
            print(
                f"  [warn] {config_label}: {len(missing)} departures missing "
                "from one of GC/CMA-ES/BERS, excluded from pairing"
            )

        for dep in departures:
            gc = e_gc[dep]
            cmaes = e_cmaes[dep]
            bers = e_bers[dep]
            rows.append(
                {
                    "configuration": config_label,
                    "departure_time_utc": dep,
                    "energy_gc_mwh": gc,
                    "energy_cmaes_mwh": cmaes,
                    "energy_bers_mwh": bers,
                    "improvement_vs_gc_pct": (gc - bers) / gc * 100.0 if gc else 0.0,
                    "improvement_vs_cmaes_pct": (
                        (cmaes - bers) / cmaes * 100.0 if cmaes else 0.0
                    ),
                }
            )
    return rows


def _stats(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "median": float(np.median(values)),
        "p5": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def build_summary_table(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Aggregate paired improvements into a summary table per configuration."""
    summary: list[dict[str, object]] = []
    for config_label in CONFIGURATIONS:
        group = [r for r in rows if r["configuration"] == config_label]
        vs_gc = np.array([r["improvement_vs_gc_pct"] for r in group])
        vs_cmaes = np.array([r["improvement_vs_cmaes_pct"] for r in group])
        gc_stats = _stats(vs_gc)
        cmaes_stats = _stats(vs_cmaes)
        summary.append(
            {
                "configuration": config_label,
                "n_departures": len(group),
                "vs_gc_mean": round(gc_stats["mean"], 2),
                "vs_gc_std": round(gc_stats["std"], 2),
                "vs_gc_median": round(gc_stats["median"], 2),
                "vs_gc_p5": round(gc_stats["p5"], 2),
                "vs_gc_p95": round(gc_stats["p95"], 2),
                "vs_gc_min": round(gc_stats["min"], 2),
                "vs_gc_max": round(gc_stats["max"], 2),
                "vs_cmaes_mean": round(cmaes_stats["mean"], 2),
                "vs_cmaes_std": round(cmaes_stats["std"], 2),
                "vs_cmaes_median": round(cmaes_stats["median"], 2),
                "vs_cmaes_p5": round(cmaes_stats["p5"], 2),
                "vs_cmaes_p95": round(cmaes_stats["p95"], 2),
            }
        )
    return summary


def write_latex_table(summary: list[dict[str, object]], out_path: Path) -> None:
    """Write the paired-improvements table in LaTeX."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Paired per-departure improvements: BERS vs great-circle "
        r"(GC) baseline and vs CMA-ES-only.}",
        r"\label{tab:paired_improvements}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"\textbf{Configuration} & \textbf{BERS vs GC: Mean $\pm$ Std (\%)} & "
        r"\textbf{Median (\%)} & \textbf{P5 (\%)} & \textbf{P95 (\%)} & "
        r"\textbf{BERS vs CMA-ES: Mean $\pm$ Std (\%)} \\",
        r"\midrule",
    ]
    for s in summary:
        lines.append(
            f"{s['configuration']} & {s['vs_gc_mean']:.1f} $\\pm$ "
            f"{s['vs_gc_std']:.1f} & {s['vs_gc_median']:.1f} & {s['vs_gc_p5']:.1f} "
            f"& {s['vs_gc_p95']:.1f} & {s['vs_cmaes_mean']:.1f} $\\pm$ "
            f"{s['vs_cmaes_std']:.1f} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    out_path.write_text("\n".join(lines) + "\n")


def plot_paired_improvements(rows: list[dict[str, object]], out_path: Path) -> None:
    """Save histograms of per-departure % improvement vs GC, one per configuration."""
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True)
    for ax, config_label in zip(axes.flat, CONFIGURATIONS, strict=False):
        values = [
            r["improvement_vs_gc_pct"]
            for r in rows
            if r["configuration"] == config_label
        ]
        ax.hist(values, bins=30, color="steelblue", edgecolor="black")
        ax.axvline(0, color="red", linestyle="--", linewidth=1)
        ax.set_title(config_label)
        ax.set_xlabel("BERS improvement vs GC (%)")
        ax.set_ylabel("Departures")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main(
    bers_dir: str = "output/sweep_combined_fms",
    cmaes_dir: str = "output/sweep_combined",
    output_dir: str = "revision",
) -> None:
    """Run Task 3 paired-improvement analysis and write outputs."""
    out_dir = REPO_ROOT / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Reading BERS energies from {bers_dir}, CMA-ES energies from {cmaes_dir} ..."
    )
    rows = compute_paired_metrics(REPO_ROOT / bers_dir, REPO_ROOT / cmaes_dir)

    metrics_csv = out_dir / "task3_paired_metrics.csv"
    with metrics_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} paired departure rows to {metrics_csv}")

    summary = build_summary_table(rows)
    summary_csv = out_dir / "task3_paired_improvements.csv"
    with summary_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    print(f"Wrote summary table to {summary_csv}")

    tex_path = out_dir / "task3_paired_improvements.tex"
    write_latex_table(summary, tex_path)
    print(f"Wrote LaTeX table to {tex_path}")

    pdf_path = out_dir / "task3_paired_improvements.pdf"
    plot_paired_improvements(rows, pdf_path)
    print(f"Wrote figure to {pdf_path}")

    print("\nSummary:")
    for s in summary:
        print(
            f"  {s['configuration']:18s} n={s['n_departures']:4d}  "
            f"vsGC={s['vs_gc_mean']:.1f}+-{s['vs_gc_std']:.1f}% "
            f"(median={s['vs_gc_median']:.1f}, P5={s['vs_gc_p5']:.1f}, "
            f"P95={s['vs_gc_p95']:.1f})  "
            f"vsCMAES={s['vs_cmaes_mean']:.1f}+-{s['vs_cmaes_std']:.1f}%"
        )


if __name__ == "__main__":
    typer.run(main)
