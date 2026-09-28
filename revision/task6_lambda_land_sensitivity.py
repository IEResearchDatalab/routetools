"""Task 6 (R2-2) — lambda_land sensitivity study.

For 3 synthetic Four-Vortices + Perlin-land scenarios (Easy/Medium/Hard,
reusing the water-level convention from Task 7: 0.9/0.8/0.7), runs CMA-ES
ONLY (no FMS) with:

    lambda_land in {10, 50, 100, 500, 1000}
    5 random seeds each

using the same K=9, sigma0=2.0, popsize=500, L=200 configuration as the main
synthetic benchmarks (Task 2). For each run records whether the final CMA-ES
solution is land-free (per the optimiser's own ``Land.__call__`` check),
final cost, and the number of CMA-ES generations to converge.

NOTE: can take a long time (75 runs); intended to run in the background /
overnight.

Outputs
-------
- ``revision/task6_lambda_land_runs.csv``: one row per (scenario, lambda,
  seed) run.
- ``revision/task6_lambda_sensitivity.csv`` / ``.tex``: aggregated table.
"""

from __future__ import annotations

import csv
import tomllib
from pathlib import Path

import jax.numpy as jnp
import typer

from routetools.cmaes import optimize
from routetools.land import Land
from routetools.vectorfield import vectorfield_fourvortices

REPO_ROOT = Path(__file__).resolve().parent.parent

# Reuses the Task 7 water-level convention for difficulty.
SCENARIOS = {"Easy": 0.9, "Medium": 0.8, "Hard": 0.7}
LAMBDA_LAND_VALUES = [10, 50, 100, 500, 1000]
SEEDS = [0, 1, 2, 3, 4]
LAND_SEED = 0  # fixed land layout per scenario; only the optimiser seed varies

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


def _build_land(water_level: float) -> Land | None:
    """Build the fourvortices Land scenario, or None if src/dst is blocked."""
    with (REPO_ROOT / "config.toml").open("rb") as handle:
        config = tomllib.load(handle)
    vf = config["vectorfield"]["fourvortices"]
    xlim = tuple(vf["xlim"])
    ylim = tuple(vf["ylim"])
    src = jnp.array(vf["src"])
    dst = jnp.array(vf["dst"])

    land = Land(
        xlim,
        ylim,
        water_level=water_level,
        resolution=5,
        random_seed=LAND_SEED,
        outbounds_is_land=False,
    )
    if bool(land(src)) or bool(land(dst)):
        return None
    return land


def run_single(land: Land, lam: float, seed: int) -> dict[str, object]:
    """Run CMA-ES-only once for a given lambda_land / seed and return metrics."""
    with (REPO_ROOT / "config.toml").open("rb") as handle:
        config = tomllib.load(handle)
    vf = config["vectorfield"]["fourvortices"]
    src = jnp.array(vf["src"])
    dst = jnp.array(vf["dst"])
    travel_stw = vf.get("travel_stw")
    travel_time = vf.get("travel_time")

    curve, dict_cmaes = optimize(
        vectorfield_fourvortices,
        src,
        dst,
        land=land,
        penalty=float(lam),
        travel_stw=travel_stw,
        travel_time=travel_time,
        seed=seed,
        verbose=False,
        **CMAES_KWARGS,
    )
    land_violation_count = int(jnp.sum(land(curve[None])))
    is_feasible = land_violation_count == 0

    return {
        "lambda_land": lam,
        "seed": seed,
        "cost": float(dict_cmaes["cost"]),
        "n_generations": int(dict_cmaes["niter"]),
        "comp_time_s": float(dict_cmaes["comp_time"]),
        "land_violation_count": land_violation_count,
        "is_feasible": is_feasible,
    }


def run_all(output_dir: Path) -> list[dict[str, object]]:
    """Run the full lambda_land x seed sweep for each scenario."""
    rows: list[dict[str, object]] = []
    for scenario_name, water_level in SCENARIOS.items():
        print(f"Building land scenario {scenario_name} (water_level={water_level}) ...")
        land = _build_land(water_level)
        if land is None:
            print(f"  [skip] {scenario_name}: src/dst blocked by land, skipping")
            continue

        for lam in LAMBDA_LAND_VALUES:
            for seed in SEEDS:
                print(f"  running {scenario_name} lambda={lam} seed={seed} ...")
                result = run_single(land, lam, seed)
                rows.append({"scenario": scenario_name, **result})
                # Incremental checkpoint so partial progress is never lost.
                _write_runs_csv(rows, output_dir / "task6_lambda_land_runs.csv")
    return rows


def _write_runs_csv(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_summary_table(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Aggregate per-run metrics into per-(scenario, lambda) statistics."""
    summary: list[dict[str, object]] = []
    keys = sorted({(r["scenario"], r["lambda_land"]) for r in rows})
    for scenario, lam in keys:
        group = [
            r for r in rows if r["scenario"] == scenario and r["lambda_land"] == lam
        ]
        n = len(group)
        feasible = [r for r in group if r["is_feasible"]]
        feasibility_rate = 100.0 * len(feasible) / n if n else 0.0
        mean_cost_feasible = (
            sum(r["cost"] for r in feasible) / len(feasible)
            if feasible
            else float("nan")
        )
        mean_cost_all = sum(r["cost"] for r in group) / n if n else float("nan")
        mean_generations = sum(r["n_generations"] for r in group) / n if n else 0.0

        summary.append(
            {
                "scenario": scenario,
                "lambda_land": lam,
                "n_runs": n,
                "feasibility_rate_pct": round(feasibility_rate, 1),
                "mean_cost_feasible": round(mean_cost_feasible, 4)
                if feasible
                else None,
                "mean_cost_all": round(mean_cost_all, 4),
                "mean_generations": round(mean_generations, 1),
            }
        )
    return summary


def write_latex_table(summary: list[dict[str, object]], out_path: Path) -> None:
    """Write the lambda_land sensitivity table in LaTeX."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Sensitivity of CMA-ES-only land avoidance to the land "
        r"penalty weight $\lambda_{land}$ (Four Vortices field, 5 seeds).}",
        r"\label{tab:lambda_land}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"\textbf{Scenario} & $\lambda_{land}$ & \textbf{Feasibility (\%)} & "
        r"\textbf{Mean Cost (feasible)} & \textbf{Mean Cost (all)} \\",
        r"\midrule",
    ]
    for s in summary:
        cost_feasible = (
            f"{s['mean_cost_feasible']:.2f}"
            if s["mean_cost_feasible"] is not None
            else "--"
        )
        lines.append(
            f"{s['scenario']} & {s['lambda_land']} & "
            f"{s['feasibility_rate_pct']:.0f} & {cost_feasible} & "
            f"{s['mean_cost_all']:.2f} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    out_path.write_text("\n".join(lines) + "\n")


def main(output_dir: str = "revision") -> None:
    """Run the Task 6 lambda_land sensitivity study and write outputs."""
    out_dir = REPO_ROOT / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Running lambda_land sweep over {LAMBDA_LAND_VALUES} x {len(SEEDS)} seeds "
        f"for scenarios {list(SCENARIOS)} ..."
    )
    rows = run_all(out_dir)

    runs_csv = out_dir / "task6_lambda_land_runs.csv"
    _write_runs_csv(rows, runs_csv)
    print(f"Wrote {len(rows)} runs to {runs_csv}")

    summary = build_summary_table(rows)
    summary_csv = out_dir / "task6_lambda_sensitivity.csv"
    with summary_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    print(f"Wrote summary table to {summary_csv}")

    tex_path = out_dir / "task6_lambda_sensitivity.tex"
    write_latex_table(summary, tex_path)
    print(f"Wrote LaTeX table to {tex_path}")

    print("\nSummary:")
    for s in summary:
        print(
            f"  {s['scenario']:8s} lambda={s['lambda_land']:5d}  "
            f"feasibility={s['feasibility_rate_pct']:.0f}%  "
            f"cost_all={s['mean_cost_all']:.3f}  gens={s['mean_generations']:.1f}"
        )


if __name__ == "__main__":
    typer.run(main)
