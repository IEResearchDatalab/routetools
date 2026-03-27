#!/usr/bin/env python
"""Report Codabench-style violation counts for SWOPP3 output folders.

The reported totals use the convention established for local SWOPP3 analysis:

- wind violations: number of File A rows above the Codabench wind threshold
- wave violations: number of File A rows above the Codabench wave threshold
- land violations: total sampled File B waypoints on land using Codabench's
  waypoint subsampling rule

For the repository's ``output/cmaes_weather`` folder this reproduces the
expected total of 2087.
"""

from __future__ import annotations

from pathlib import Path

import typer

from routetools.violation_counts import (
    count_folder_violations,
    format_grouped_violation_table,
    load_default_land_checker,
    load_default_weather_resources,
    write_grouped_violation_csv,
)

app = typer.Typer(help="Count Codabench-style violations in SWOPP3 output folders.")


@app.command()
def main(
    input_dirs: list[Path] = typer.Argument(
        ..., help="One or more SWOPP3 output folders, e.g. output/cmaes_weather."
    ),
    land_shapefile: Path | None = typer.Option(
        None,
        "--land-shapefile",
        help="Optional Natural Earth land shapefile. Defaults to Cartopy's 10m land.",
    ),
    wind_penalty_weight: float = typer.Option(
        1000.0,
        "--wind-penalty-weight",
        help="Weight used for smooth wind penalties.",
    ),
    wave_penalty_weight: float = typer.Option(
        1000.0,
        "--wave-penalty-weight",
        help="Weight used for smooth wave penalties.",
    ),
    output_csv: Path = typer.Option(
        Path("output/weather_penalty_reports/violation_summary.csv"),
        "--output-csv",
        help="CSV destination written under output by default.",
    ),
) -> None:
    """Print per-scenario violation counts and save them as CSV."""
    land_checker = load_default_land_checker(land_shapefile)
    weather_resources = load_default_weather_resources()

    all_rows = []
    for input_dir in input_dirs:
        all_rows.extend(
            count_folder_violations(
                input_dir,
                land_checker=land_checker,
                weather_resources=weather_resources,
                wind_penalty_weight=wind_penalty_weight,
                wave_penalty_weight=wave_penalty_weight,
            )
        )
    typer.echo(format_grouped_violation_table(all_rows))
    csv_path = write_grouped_violation_csv(all_rows, output_csv)
    typer.echo()
    typer.echo(f"CSV written to {csv_path}")


if __name__ == "__main__":
    app()
