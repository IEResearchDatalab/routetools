#!/usr/bin/env python
"""Run SWOPP3 cases — CLI entry-point.

Usage
-----
Run all 8 cases (GC only, no ERA5 data required):

    python scripts/swopp3_run.py --strategy gc --output-dir output/swopp3

Run a single case with ERA5 data:

    python scripts/swopp3_run.py \\
        --cases AGC_WPS \\
        --wind-path data/era5_wind_2024.nc \\
        --wave-path data/era5_wave_2024.nc \\
        --output-dir output/swopp3

Run only the first 3 departures (quick test):

    python scripts/swopp3_run.py --max-departures 3 --output-dir output/swopp3
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(help="SWOPP3 competition runner.")


@app.command()
def main(
    cases: Optional[list[str]] = typer.Option(
        None,
        "--cases",
        "-c",
        help="Case IDs to run (e.g. AGC_WPS PO_noWPS).  Default: all 8.",
    ),
    strategy: Optional[str] = typer.Option(
        None,
        "--strategy",
        "-s",
        help="Filter by strategy: 'gc' or 'optimised'.  Default: both.",
    ),
    wind_path: Optional[Path] = typer.Option(
        None,
        "--wind-path",
        help="Path to ERA5 wind NetCDF file.",
    ),
    wave_path: Optional[Path] = typer.Option(
        None,
        "--wave-path",
        help="Path to ERA5 wave NetCDF file.",
    ),
    output_dir: Path = typer.Option(
        "output/swopp3",
        "--output-dir",
        "-o",
        help="Output directory for CSV files.",
    ),
    submission: int = typer.Option(
        1,
        "--submission",
        help="Submission number for file naming.",
    ),
    n_points: int = typer.Option(
        100,
        "--n-points",
        help="Number of route waypoints.",
    ),
    max_departures: Optional[int] = typer.Option(
        None,
        "--max-departures",
        "-n",
        help="Limit number of departures (for quick testing).",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Suppress progress output.",
    ),
) -> None:
    """Run SWOPP3 competition cases."""
    from routetools.swopp3 import SWOPP3_CASES, departures_2024
    from routetools.swopp3_runner import run_case

    # ---- Select cases ----
    if cases is not None:
        case_ids = cases
        for cid in case_ids:
            if cid not in SWOPP3_CASES:
                typer.echo(f"Unknown case: {cid}", err=True)
                raise typer.Exit(1)
    else:
        case_ids = list(SWOPP3_CASES.keys())

    if strategy is not None:
        case_ids = [
            cid for cid in case_ids
            if SWOPP3_CASES[cid]["strategy"] == strategy
        ]
        if not case_ids:
            typer.echo(f"No cases match strategy '{strategy}'", err=True)
            raise typer.Exit(1)

    # ---- Departures ----
    departures = departures_2024()
    if max_departures is not None:
        departures = departures[:max_departures]

    typer.echo(
        f"Running {len(case_ids)} case(s) × {len(departures)} departure(s)"
    )

    # ---- Load fields (if paths provided) ----
    windfield = None
    wavefield = None
    vectorfield = None

    if wind_path is not None:
        from routetools.era5.loader import (
            load_era5_vectorfield,
            load_era5_windfield,
        )
        typer.echo(f"Loading wind field from {wind_path} …")
        windfield = load_era5_windfield(wind_path)
        vectorfield = load_era5_vectorfield(wind_path)

    if wave_path is not None:
        from routetools.era5.loader import load_era5_wavefield
        typer.echo(f"Loading wave field from {wave_path} …")
        wavefield = load_era5_wavefield(wave_path)

    # ---- Run ----
    for cid in case_ids:
        case = SWOPP3_CASES[cid]
        typer.echo(f"\n{'='*60}")
        typer.echo(f"Case {cid}: {case['label']}")
        typer.echo(f"  strategy={case['strategy']}  wps={case['wps']}")
        typer.echo(f"{'='*60}")

        results = run_case(
            cid,
            departures,
            vectorfield=vectorfield,
            windfield=windfield,
            wavefield=wavefield,
            output_dir=output_dir,
            submission=submission,
            n_points=n_points,
            verbose=not quiet,
        )

        # Summary
        energies = [r.energy_mwh for r in results]
        total_time = sum(r.comp_time_s for r in results)
        typer.echo(
            f"  {len(results)} departures  "
            f"mean E={sum(energies)/len(energies):.2f} MWh  "
            f"total comp time={total_time:.1f}s"
        )

    typer.echo(f"\nOutputs written to {output_dir}")


if __name__ == "__main__":
    app()
