"""Task 2 (R1-6) — statistical dispersion for synthetic benchmarks.

Runs BERS (CMA-ES followed by FMS refinement) 5 times with different random
seeds for each of the 5 literature synthetic vector fields (Circular, Four
Vortices, Double Gyre, Techy, Swirlys), using the best configuration from the
paper (sigma0 = 2.0, K = 9, popsize = 500, L = 200).

For each run, records the final (BERS) cost, the pre-FMS CMA-ES cost, and
the CMA-ES/FMS/total computation times, then reports mean, std, min, max and
IQR per vector field.

Outputs
-------
- ``revision/task2_runs.csv``: one row per (field, seed) run.
- ``revision/task2_dispersion_table.csv`` / ``.tex``: aggregated dispersion
  table extending the paper's Table 4.
"""

from __future__ import annotations

import csv
import tomllib
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import typer

from routetools.cmaes import optimize
from routetools.fms import optimize_fms

REPO_ROOT = Path(__file__).resolve().parent.parent

FIELDS = ["circular", "fourvortices", "doublegyre", "techy", "swirlys"]
FIELD_LABELS = {
    "circular": "Circular",
    "fourvortices": "Four Vortices",
    "doublegyre": "Double Gyre",
    "techy": "Techy",
    "swirlys": "Swirlys",
}
# Reference costs reported in the paper / scripts/synthetic/figures.py
REFERENCE_COST = {
    "circular": 1.98,
    "fourvortices": 8.95,
    "doublegyre": 1.01,
    "techy": 1.03,
    "swirlys": 5.73,
}

SEEDS = [0, 1, 2, 3, 4]

# Best configuration from the paper (sigma0=2.0, K=9).
CMAES_KWARGS = dict(
    K=9,
    L=200,
    num_pieces=1,
    popsize=500,
    sigma0=2.0,
    tolfun=1e-3,
    damping=1.0,
    maxfevals=500_000,
)
FMS_KWARGS = dict(patience=50, damping=0.5, maxfevals=500_000)


def run_single(vf_name: str, vf_config: dict, seed: int) -> dict[str, object]:
    """Run CMA-ES + FMS (BERS) once for a field/seed and return metrics."""
    vectorfield_module = __import__(
        "routetools.vectorfield", fromlist=[f"vectorfield_{vf_name}"]
    )
    vectorfield_fun = getattr(vectorfield_module, f"vectorfield_{vf_name}")
    src = jnp.array(vf_config["src"])
    dst = jnp.array(vf_config["dst"])
    travel_stw = vf_config.get("travel_stw")
    travel_time = vf_config.get("travel_time")

    curve_cmaes, dict_cmaes = optimize(
        vectorfield_fun,
        src,
        dst,
        travel_stw=travel_stw,
        travel_time=travel_time,
        seed=seed,
        verbose=False,
        **CMAES_KWARGS,
    )
    curve_bers, dict_bers = optimize_fms(
        vectorfield_fun,
        curve=curve_cmaes,
        travel_stw=travel_stw,
        travel_time=travel_time,
        seed=seed,
        verbose=False,
        **FMS_KWARGS,
    )

    cost_cmaes = float(dict_cmaes["cost"])
    cost_bers = float(dict_bers["cost"][0])
    comp_time_cmaes = float(dict_cmaes["comp_time"])
    comp_time_fms = float(dict_bers["comp_time"])

    return {
        "vectorfield": vf_name,
        "seed": seed,
        "cost_cmaes": cost_cmaes,
        "cost_bers": cost_bers,
        "comp_time_cmaes": comp_time_cmaes,
        "comp_time_fms": comp_time_fms,
        "comp_time_total": comp_time_cmaes + comp_time_fms,
    }


def run_all(config: dict) -> list[dict[str, object]]:
    """Run all (field, seed) combinations and return raw per-run metrics."""
    rows: list[dict[str, object]] = []
    for vf_name in FIELDS:
        vf_config = config["vectorfield"][vf_name]
        for seed in SEEDS:
            print(f"  running {vf_name} seed={seed} ...")
            rows.append(run_single(vf_name, vf_config, seed))
    return rows


def _iqr(arr: np.ndarray) -> float:
    """Return the interquartile range (Q3 - Q1) of an array."""
    return float(np.percentile(arr, 75) - np.percentile(arr, 25))


def build_dispersion_table(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Aggregate per-run metrics into mean/std/min/max/IQR per field."""
    summary: list[dict[str, object]] = []
    for vf_name in FIELDS:
        group = [r for r in rows if r["vectorfield"] == vf_name]
        cost_bers = np.array([r["cost_bers"] for r in group])
        cost_cmaes = np.array([r["cost_cmaes"] for r in group])
        t_total = np.array([r["comp_time_total"] for r in group])
        t_cmaes = np.array([r["comp_time_cmaes"] for r in group])
        t_fms = np.array([r["comp_time_fms"] for r in group])

        summary.append(
            {
                "vectorfield": vf_name,
                "label": FIELD_LABELS[vf_name],
                "reference_cost": REFERENCE_COST[vf_name],
                "n_runs": len(group),
                "bers_cost_mean": round(float(cost_bers.mean()), 4),
                "bers_cost_std": round(float(cost_bers.std()), 4),
                "bers_cost_min": round(float(cost_bers.min()), 4),
                "bers_cost_max": round(float(cost_bers.max()), 4),
                "bers_cost_iqr": round(_iqr(cost_bers), 4),
                "cmaes_cost_mean": round(float(cost_cmaes.mean()), 4),
                "cmaes_cost_std": round(float(cost_cmaes.std()), 4),
                "runtime_total_mean_s": round(float(t_total.mean()), 2),
                "runtime_total_std_s": round(float(t_total.std()), 2),
                "runtime_cmaes_mean_s": round(float(t_cmaes.mean()), 2),
                "runtime_cmaes_std_s": round(float(t_cmaes.std()), 2),
                "runtime_fms_mean_s": round(float(t_fms.mean()), 2),
                "runtime_fms_std_s": round(float(t_fms.std()), 2),
            }
        )
    return summary


def write_latex_table(summary: list[dict[str, object]], out_path: Path) -> None:
    """Write an extended Table 4 with dispersion measures, in LaTeX."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{BERS cost and runtime dispersion over 5 random seeds "
        r"($\sigma_0=2.0$, $K=9$) for the 5 literature vector fields.}",
        r"\label{tab:synthetic_dispersion}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"\textbf{Vector Field} & \textbf{Ref.\ Cost} & \textbf{BERS Mean "
        r"$\pm$ Std} & \textbf{BERS Min} & \textbf{BERS Max} & "
        r"\textbf{Runtime Mean $\pm$ Std (s)} \\",
        r"\midrule",
    ]
    for s in summary:
        lines.append(
            f"{s['label']} & {s['reference_cost']:.2f} & "
            f"{s['bers_cost_mean']:.2f} $\\pm$ {s['bers_cost_std']:.2f} & "
            f"{s['bers_cost_min']:.2f} & {s['bers_cost_max']:.2f} & "
            f"{s['runtime_total_mean_s']:.1f} $\\pm$ {s['runtime_total_std_s']:.1f} "
            r"\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    out_path.write_text("\n".join(lines) + "\n")


def main(
    config_path: str = "config.toml",
    output_dir: str = "revision",
) -> None:
    """Run the Task 2 synthetic dispersion experiment and write outputs."""
    out_dir = REPO_ROOT / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    with (REPO_ROOT / config_path).open("rb") as handle:
        config = tomllib.load(handle)

    print("Running BERS with 5 seeds for each of the 5 synthetic vector fields...")
    rows = run_all(config)

    runs_csv = out_dir / "task2_runs.csv"
    with runs_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} runs to {runs_csv}")

    summary = build_dispersion_table(rows)
    summary_csv = out_dir / "task2_dispersion_table.csv"
    with summary_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    print(f"Wrote dispersion table to {summary_csv}")

    tex_path = out_dir / "task2_dispersion_table.tex"
    write_latex_table(summary, tex_path)
    print(f"Wrote LaTeX table to {tex_path}")

    print("\nSummary:")
    for s in summary:
        print(
            f"  {s['label']:15s} ref={s['reference_cost']:.2f}  "
            f"BERS={s['bers_cost_mean']:.3f}+-{s['bers_cost_std']:.3f} "
            f"[{s['bers_cost_min']:.3f}, {s['bers_cost_max']:.3f}]  "
            f"runtime={s['runtime_total_mean_s']:.1f}+-{s['runtime_total_std_s']:.1f}s"
        )


if __name__ == "__main__":
    typer.run(main)
