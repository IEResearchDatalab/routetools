"""Task 5 (R1-7) — fast, targeted weather-penalty sensitivity test.

This test starts from the preserved CMA-ES routes in ``output/sweep_combined``
and reruns only FMS.  It therefore isolates the actual split squared-excess
weather penalty used in the paper without repeating the expensive global
search.  Eight deliberately difficult departures are used: two from each
corridor/WPS configuration, selected from the largest weather exposures found
by the annual Task 4 audit.  Both wind and wave weights are varied together at
25, 50 (the reported setting), and 100.

The environmental penalty is quadratic in threshold excess; there is no
separate exponential-sharpness parameter in this implementation.

Outputs
-------
- ``task5_env_sensitivity_runs.csv``: one row per route and weight.
- ``task5_env_sensitivity.csv`` / ``.tex``: aggregate stress-test results.
- ``task5_env_sensitivity.pdf``: penalty-weight trade-off figure.
- ``weight_<value>/``: auditable FMS route outputs for every tested weight.
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import typer

from routetools.violations import find_team_prefix
from routetools.weather import DEFAULT_HS_LIMIT, DEFAULT_TWS_LIMIT

REPO_ROOT = Path(__file__).resolve().parent.parent

WEATHER_PENALTY_WEIGHTS = (25.0, 50.0, 100.0)
STRICT_TWS_LIMIT = 19.9
STRICT_HS_LIMIT = 6.9

# Two stress departures per configuration, selected from the annual Task 4
# audit of output/sweep_combined_fms_strict.  These dates intentionally target
# the largest residual wind/wave exposures rather than forming a random sample.
STRESS_DEPARTURES = {
    "AO_noWPS": ("2024-02-22", "2024-03-22"),
    "AO_WPS": ("2024-01-13", "2024-11-25"),
    "PO_noWPS": ("2024-02-26", "2024-12-29"),
    "PO_WPS": ("2024-02-26", "2024-12-30"),
}


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare_stress_subset(input_dir: Path, subset_dir: Path) -> Path:
    """Create a four-case, eight-route subset of preserved CMA-ES outputs."""
    team_prefix = find_team_prefix(input_dir)
    tracks_dir = subset_dir / "tracks"
    tracks_dir.mkdir(parents=True, exist_ok=True)

    for case_id, dates in STRESS_DEPARTURES.items():
        source_summary = input_dir / f"{team_prefix}-{case_id}.csv"
        rows = _read_rows(source_summary)
        selected = [
            row
            for row in rows
            if row["departure_time_utc"][:10] in set(dates)
        ]
        selected_dates = {row["departure_time_utc"][:10] for row in selected}
        missing_dates = set(dates) - selected_dates
        if missing_dates:
            raise ValueError(
                f"{case_id} is missing stress departures: {sorted(missing_dates)}"
            )

        _write_rows(subset_dir / source_summary.name, selected)
        for row in selected:
            filename = row["details_filename"]
            shutil.copy2(input_dir / "tracks" / filename, tracks_dir / filename)

    return subset_dir


def run_weight_sweep(
    input_dir: Path,
    output_dir: Path,
    weights: tuple[float, ...] = WEATHER_PENALTY_WEIGHTS,
) -> list[Path]:
    """Run strict FMS on the same stress routes at each penalty weight."""
    from scripts.swopp3_apply_fms import apply_fms_to_outputs

    subset_dir = prepare_stress_subset(input_dir, output_dir / "_input_subset")
    result_dirs: list[Path] = []
    for weight in weights:
        result_dir = output_dir / f"weight_{weight:g}"
        print(
            f"Running strict FMS stress subset with "
            f"wind_penalty_weight=wave_penalty_weight={weight:g} ..."
        )
        apply_fms_to_outputs(
            subset_dir,
            output_dir=result_dir,
            wind_path_atlantic=REPO_ROOT
            / "data/era5/era5_wind_atlantic_2024.nc",
            wave_path_atlantic=REPO_ROOT
            / "data/era5/era5_waves_atlantic_2024.nc",
            wind_path_pacific=REPO_ROOT
            / "data/era5/era5_wind_pacific_2024.nc",
            wave_path_pacific=REPO_ROOT
            / "data/era5/era5_waves_pacific_2024.nc",
            fms_patience=200,
            fms_damping=0.95,
            fms_maxfevals=10000,
            tws_limit=STRICT_TWS_LIMIT,
            hs_limit=STRICT_HS_LIMIT,
            wind_penalty_weight=weight,
            wave_penalty_weight=weight,
            enforce_weather_limits=False,
        )
        result_dirs.append(result_dir)
    return result_dirs


def collect_run_rows(
    input_dir: Path,
    result_dirs: list[Path],
    weights: tuple[float, ...] = WEATHER_PENALTY_WEIGHTS,
) -> list[dict[str, object]]:
    """Collect route-level energy and threshold results from FMS output CSVs."""
    input_team = find_team_prefix(input_dir)
    cmaes_energy: dict[tuple[str, str], float] = {}
    for case_id, dates in STRESS_DEPARTURES.items():
        rows = _read_rows(input_dir / f"{input_team}-{case_id}.csv")
        for row in rows:
            date = row["departure_time_utc"][:10]
            if date in dates:
                cmaes_energy[(case_id, date)] = float(row["energy_cons_mwh"])

    run_rows: list[dict[str, object]] = []
    for weight, result_dir in zip(weights, result_dirs, strict=True):
        team_prefix = find_team_prefix(result_dir)
        for case_id in STRESS_DEPARTURES:
            for row in _read_rows(result_dir / f"{team_prefix}-{case_id}.csv"):
                date = row["departure_time_utc"][:10]
                energy = float(row["energy_cons_mwh"])
                baseline = cmaes_energy[(case_id, date)]
                max_tws = float(row["max_wind_mps"])
                max_hs = float(row["max_hs_m"])
                run_rows.append(
                    {
                        "case_id": case_id,
                        "departure": date,
                        "weather_penalty_weight": weight,
                        "energy_mwh": energy,
                        "energy_change_vs_cmaes_pct": 100.0
                        * (energy - baseline)
                        / baseline,
                        "max_tws_mps": max_tws,
                        "max_hs_m": max_hs,
                        "wind_exceedance_mps": max(
                            max_tws - DEFAULT_TWS_LIMIT, 0.0
                        ),
                        "wave_exceedance_m": max(max_hs - DEFAULT_HS_LIMIT, 0.0),
                        "any_violation": bool(
                            max_tws > DEFAULT_TWS_LIMIT
                            or max_hs > DEFAULT_HS_LIMIT
                        ),
                        "details_filename": row["details_filename"],
                    }
                )
    return run_rows


def build_summary_table(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Aggregate the paired eight-route stress test by penalty weight."""
    summary: list[dict[str, object]] = []
    for weight in sorted({float(row["weather_penalty_weight"]) for row in rows}):
        group = [
            row for row in rows if float(row["weather_penalty_weight"]) == weight
        ]
        energy_changes = np.asarray(
            [float(row["energy_change_vs_cmaes_pct"]) for row in group]
        )
        wind_excess = np.asarray([float(row["wind_exceedance_mps"]) for row in group])
        wave_excess = np.asarray([float(row["wave_exceedance_m"]) for row in group])
        summary.append(
            {
                "weather_penalty_weight": weight,
                "n_routes": len(group),
                "n_with_any_violation": sum(
                    bool(row["any_violation"]) for row in group
                ),
                "n_with_wind_violation": int(np.sum(wind_excess > 0.0)),
                "n_with_wave_violation": int(np.sum(wave_excess > 0.0)),
                "mean_energy_change_vs_cmaes_pct": float(np.mean(energy_changes)),
                "mean_wind_exceedance_mps": float(np.mean(wind_excess)),
                "max_wind_exceedance_mps": float(np.max(wind_excess)),
                "mean_wave_exceedance_m": float(np.mean(wave_excess)),
                "max_wave_exceedance_m": float(np.max(wave_excess)),
            }
        )
    return summary


def write_latex_table(summary: list[dict[str, object]], out_path: Path) -> None:
    """Write the compact stress-test table used in the response letter."""
    lines = [
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        r"$\lambda_U=\lambda_H$ & Routes & Any violation & Wind violation & "
        r"Wave violation & $\Delta E$ vs CMA-ES (\%) \\",
        r"\midrule",
    ]
    for row in summary:
        lines.append(
            f"{float(row['weather_penalty_weight']):g} & {row['n_routes']} & "
            f"{row['n_with_any_violation']} & {row['n_with_wind_violation']} & "
            f"{row['n_with_wave_violation']} & "
            f"{float(row['mean_energy_change_vs_cmaes_pct']):.2f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    out_path.write_text("\n".join(lines) + "\n")


def plot_tradeoff(summary: list[dict[str, object]], out_path: Path) -> None:
    """Plot violation count and paired energy change against penalty weight."""
    weights = [float(row["weather_penalty_weight"]) for row in summary]
    violations = [int(row["n_with_any_violation"]) for row in summary]
    energy_change = [
        float(row["mean_energy_change_vs_cmaes_pct"]) for row in summary
    ]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(weights, violations, "o-", color="firebrick", label="Routes violating")
    ax.set_xlabel(r"Wind and wave penalty weights, $\lambda_U=\lambda_H$")
    ax.set_ylabel("Routes with any benchmark-threshold violation", color="firebrick")
    ax.set_xticks(weights)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.22, linewidth=0.6)
    ax2 = ax.twinx()
    ax2.plot(
        weights,
        energy_change,
        "s--",
        color="steelblue",
        label="Mean energy change",
    )
    ax2.set_ylabel("Mean energy change vs CMA-ES (%)", color="steelblue")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main(
    input_dir: str = "output/sweep_combined",
    output_dir: str = "output/task5_env_penalty_sensitivity_20260924",
) -> None:
    """Run the targeted strict-FMS weather-penalty stress test."""
    source_dir = REPO_ROOT / input_dir
    result_root = REPO_ROOT / output_dir
    result_root.mkdir(parents=True, exist_ok=True)

    result_dirs = run_weight_sweep(source_dir, result_root)
    rows = collect_run_rows(source_dir, result_dirs)
    _write_rows(result_root / "task5_env_sensitivity_runs.csv", rows)

    summary = build_summary_table(rows)
    _write_rows(result_root / "task5_env_sensitivity.csv", summary)
    write_latex_table(summary, result_root / "task5_env_sensitivity.tex")
    plot_tradeoff(summary, result_root / "task5_env_sensitivity.pdf")

    print("\nTargeted stress-test summary:")
    for row in summary:
        print(
            f"  weight={float(row['weather_penalty_weight']):g}: "
            f"violations={row['n_with_any_violation']}/{row['n_routes']}, "
            f"mean dE vs CMA-ES="
            f"{float(row['mean_energy_change_vs_cmaes_pct']):+.2f}%"
        )


if __name__ == "__main__":
    typer.run(main)
