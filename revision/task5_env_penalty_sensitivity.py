"""Task 5 (R1-7) — environmental penalty sensitivity analysis.

For a small set of representative winter departures (Atlantic + Pacific,
``noWPS`` cases), runs the full real-ocean BERS pipeline
(``routetools.swopp3_runner.run_optimised_departure``) while varying the
environmental (weather) penalty weight ``weather_penalty_weight`` (lambda_env).

For each setting records: final route energy (MWh), whether the wind/wave
thresholds are exceeded, the max exceedance, sailed distance (as a proxy for
route-geometry change), and computation time.

Each (corridor, departure, lambda_env) run executes in its OWN subprocess
("worker" mode below). This is deliberate: a single run loads GB-scale ERA5
fields into a JIT-compiled JAX program, and empirically this does not fully
release host/GPU memory between runs within one long-lived process (observed
an OOM-kill after 1-2 runs when looping in-process). One subprocess per
combination guarantees a clean memory slate every time, at the cost of a few
seconds of ERA5-reload overhead per run.

NOTE: deliberately restricted to a handful of departures per corridor
(not the full 366) per the task instructions.

Outputs
-------
- ``revision/task5_env_sensitivity_runs.csv``: one row per (departure,
  lambda_env) run.
- ``revision/task5_env_sensitivity.csv`` / ``.tex``: aggregated table.
- ``revision/task5_env_sensitivity.pdf``: cost vs. violation trade-off figure.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import typer

app = typer.Typer(add_completion=False)

REPO_ROOT = Path(__file__).resolve().parent.parent

# 3 representative winter (January) departures per corridor.
WINTER_DEPARTURES = ["2024-01-05", "2024-01-15", "2024-01-25"]

CORRIDOR_CASES = {"atlantic": "AO_noWPS", "pacific": "PO_noWPS"}
LAMBDA_ENV_VALUES = [0, 1, 5, 10, 50, 100]

ERA5_PATHS = {
    "atlantic": {
        "wind": "data/era5/era5_wind_atlantic_2024.nc",
        "wave": "data/era5/era5_waves_atlantic_2024.nc",
    },
    "pacific": {
        "wind": "data/era5/era5_wind_pacific_2024.nc",
        "wave": "data/era5/era5_waves_pacific_2024.nc",
    },
}


@app.command()
def worker(corridor: str, departure_str: str, lambda_env: float) -> None:
    """Run exactly one (corridor, departure, lambda_env) combination.

    Prints a single ``RESULT_JSON:``-prefixed line to stdout. Intended to be
    invoked as a short-lived subprocess so JAX/host memory is fully released
    by the OS when it exits.
    """
    import xarray as xr

    from routetools.era5.loader import (
        load_dataset_epoch,
        load_era5_wavefield,
        load_era5_windfield,
        load_natural_earth_land_mask,
        loadable_era5_paths,
    )
    from routetools.swopp3_runner import run_optimised_departure
    from routetools.weather import DEFAULT_HS_LIMIT, DEFAULT_TWS_LIMIT

    case_id = CORRIDOR_CASES[corridor]
    departure = datetime.strptime(departure_str, "%Y-%m-%d").replace(hour=12)

    wind_paths = loadable_era5_paths(REPO_ROOT / ERA5_PATHS[corridor]["wind"])
    wave_paths = loadable_era5_paths(REPO_ROOT / ERA5_PATHS[corridor]["wave"])
    wind_target = wind_paths if len(wind_paths) > 1 else wind_paths[0]
    wave_target = wave_paths if len(wave_paths) > 1 else wave_paths[0]

    dataset_epoch = load_dataset_epoch(wind_target)
    windfield = load_era5_windfield(wind_target)
    wavefield = load_era5_wavefield(wave_target)

    with xr.open_dataset(wave_paths[0]) as ds:
        for lon_name in ("longitude", "lon"):
            if lon_name in ds.coords:
                lons = ds[lon_name].values
                break
        for lat_name in ("latitude", "lat"):
            if lat_name in ds.coords:
                lats = ds[lat_name].values
                break
    land = load_natural_earth_land_mask(
        (float(lons.min()), float(lons.max())),
        (float(lats.min()), float(lats.max())),
    )

    departure_offset_h = (departure - dataset_epoch).total_seconds() / 3600.0

    t0 = time.time()
    result = run_optimised_departure(
        case_id=case_id,
        departure=departure,
        vectorfield=windfield,
        windfield=windfield,
        wavefield=wavefield,
        land=land,
        departure_offset_h=departure_offset_h,
        weather_penalty_weight=float(lambda_env),
        verbosity=0,
    )
    wall_time = time.time() - t0

    payload = {
        "corridor": corridor,
        "case_id": case_id,
        "departure": departure_str,
        "lambda_env": lambda_env,
        "energy_mwh": round(result.energy_mwh, 4),
        "max_tws_mps": round(result.max_tws_mps, 4),
        "max_hs_m": round(result.max_hs_m, 4),
        "wind_violation": bool(result.max_tws_mps > DEFAULT_TWS_LIMIT),
        "wave_violation": bool(result.max_hs_m > DEFAULT_HS_LIMIT),
        "wind_exceedance": round(max(result.max_tws_mps - DEFAULT_TWS_LIMIT, 0.0), 4),
        "wave_exceedance": round(max(result.max_hs_m - DEFAULT_HS_LIMIT, 0.0), 4),
        "distance_nm": round(result.distance_nm, 2),
        "comp_time_s": round(wall_time, 1),
    }
    print("RESULT_JSON:" + json.dumps(payload))


def _run_worker_subprocess(
    corridor: str, departure_str: str, lam: float
) -> dict[str, object]:
    """Run one worker subprocess and parse its ``RESULT_JSON:`` output line."""
    proc = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "worker",
            corridor,
            departure_str,
            str(lam),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            return json.loads(line[len("RESULT_JSON:") :])
    raise RuntimeError(
        f"Worker failed for corridor={corridor} departure={departure_str} "
        f"lambda={lam} (exit={proc.returncode}).\n"
        f"--- stdout (tail) ---\n{proc.stdout[-2000:]}\n"
        f"--- stderr (tail) ---\n{proc.stderr[-2000:]}"
    )


def run_sensitivity(output_dir: Path) -> list[dict[str, object]]:
    """Run the lambda_env sweep for all corridors/departures via subprocesses."""
    rows: list[dict[str, object]] = []

    for corridor in CORRIDOR_CASES:
        for departure_str in WINTER_DEPARTURES:
            baseline_distance_nm: float | None = None
            for lam in LAMBDA_ENV_VALUES:
                print(
                    f"  running corridor={corridor} departure={departure_str} "
                    f"lambda_env={lam} (subprocess) ..."
                )
                payload = _run_worker_subprocess(corridor, departure_str, lam)

                if baseline_distance_nm is None:
                    baseline_distance_nm = payload["distance_nm"]
                distance_change_pct = (
                    (payload["distance_nm"] - baseline_distance_nm)
                    / baseline_distance_nm
                    * 100.0
                    if baseline_distance_nm
                    else 0.0
                )
                payload["distance_change_pct_vs_lambda0"] = round(
                    distance_change_pct, 2
                )
                rows.append(payload)
                # Incremental checkpoint so partial progress is never lost.
                _write_runs_csv(rows, output_dir / "task5_env_sensitivity_runs.csv")
    return rows


def _write_runs_csv(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_summary_table(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Aggregate per-departure runs into mean stats per (corridor, lambda_env)."""
    summary: list[dict[str, object]] = []
    keys = sorted({(r["corridor"], r["lambda_env"]) for r in rows})
    for corridor, lam in keys:
        group = [
            r for r in rows if r["corridor"] == corridor and r["lambda_env"] == lam
        ]
        n = len(group)
        summary.append(
            {
                "corridor": corridor,
                "lambda_env": lam,
                "n_departures": n,
                "mean_energy_mwh": round(sum(r["energy_mwh"] for r in group) / n, 3),
                "n_wind_violations": sum(r["wind_violation"] for r in group),
                "n_wave_violations": sum(r["wave_violation"] for r in group),
                "mean_wind_exceedance": round(
                    sum(r["wind_exceedance"] for r in group) / n, 3
                ),
                "mean_wave_exceedance": round(
                    sum(r["wave_exceedance"] for r in group) / n, 3
                ),
                "mean_distance_change_pct": round(
                    sum(r["distance_change_pct_vs_lambda0"] for r in group) / n, 2
                ),
            }
        )
    return summary


def write_latex_table(summary: list[dict[str, object]], out_path: Path) -> None:
    """Write the lambda_env sensitivity summary in LaTeX."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Sensitivity of BERS routes to the environmental penalty "
        r"weight $\lambda_{env}$ (mean over 3 representative winter "
        r"departures per corridor).}",
        r"\label{tab:env_sensitivity}",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"\textbf{Corridor} & $\lambda_{env}$ & \textbf{Mean Energy (MWh)} & "
        r"\textbf{Wind Viol.} & \textbf{Wave Viol.} & "
        r"\textbf{Mean Wind Exc.} & \textbf{Mean Wave Exc.} \\",
        r"\midrule",
    ]
    for s in summary:
        lines.append(
            f"{s['corridor'].capitalize()} & {s['lambda_env']} & "
            f"{s['mean_energy_mwh']:.2f} & {s['n_wind_violations']}/"
            f"{s['n_departures']} & {s['n_wave_violations']}/{s['n_departures']} "
            f"& {s['mean_wind_exceedance']:.2f} & {s['mean_wave_exceedance']:.2f} "
            r"\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    out_path.write_text("\n".join(lines) + "\n")


def plot_tradeoff(summary: list[dict[str, object]], out_path: Path) -> None:
    """Plot energy and violation count vs lambda_env, one panel per corridor."""
    corridors = sorted({s["corridor"] for s in summary})
    fig, axes = plt.subplots(1, len(corridors), figsize=(6 * len(corridors), 4.5))
    if len(corridors) == 1:
        axes = [axes]
    for ax, corridor in zip(axes, corridors, strict=False):
        group = sorted(
            (s for s in summary if s["corridor"] == corridor),
            key=lambda s: s["lambda_env"],
        )
        lambdas = [s["lambda_env"] for s in group]
        energy = [s["mean_energy_mwh"] for s in group]
        viol = [s["n_wind_violations"] + s["n_wave_violations"] for s in group]

        ax.plot(lambdas, energy, "o-", color="steelblue", label="Mean energy (MWh)")
        ax.set_xlabel(r"$\lambda_{env}$")
        ax.set_ylabel("Mean energy (MWh)", color="steelblue")
        ax.set_xscale("symlog", linthresh=1)
        ax2 = ax.twinx()
        ax2.plot(lambdas, viol, "s--", color="firebrick", label="# violations")
        ax2.set_ylabel("# wind+wave violations", color="firebrick")
        ax.set_title(corridor.capitalize())
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


@app.command()
def main(output_dir: str = "revision") -> None:
    """Run Task 5 environmental penalty sensitivity analysis."""
    out_dir = REPO_ROOT / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Running lambda_env sweep over {LAMBDA_ENV_VALUES} for "
        f"{len(WINTER_DEPARTURES)} departures x {len(CORRIDOR_CASES)} corridors "
        "(one subprocess per run) ..."
    )
    rows = run_sensitivity(out_dir)

    runs_csv = out_dir / "task5_env_sensitivity_runs.csv"
    _write_runs_csv(rows, runs_csv)
    print(f"Wrote {len(rows)} runs to {runs_csv}")

    summary = build_summary_table(rows)
    summary_csv = out_dir / "task5_env_sensitivity.csv"
    with summary_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    print(f"Wrote summary table to {summary_csv}")

    tex_path = out_dir / "task5_env_sensitivity.tex"
    write_latex_table(summary, tex_path)
    print(f"Wrote LaTeX table to {tex_path}")

    pdf_path = out_dir / "task5_env_sensitivity.pdf"
    plot_tradeoff(summary, pdf_path)
    print(f"Wrote figure to {pdf_path}")

    print("\nSummary:")
    for s in summary:
        print(
            f"  {s['corridor']:10s} lambda={s['lambda_env']:5d}  "
            f"E={s['mean_energy_mwh']:.2f} MWh  "
            f"wind_viol={s['n_wind_violations']}/{s['n_departures']}  "
            f"wave_viol={s['n_wave_violations']}/{s['n_departures']}"
        )


if __name__ == "__main__":
    app()
