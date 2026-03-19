#!/usr/bin/env python3
"""Compare Pacific no-constraint vs constrained results.

Reads File A CSVs from two output directories and produces:
  1. Energy time-series overlay (constrained vs no-constraint)
  2. Energy delta (%) per departure
  3. Weather exposure comparison (max TWS, max Hs)
  4. Distance comparison

Usage::

    python scripts/plot_noconstraint_comparison.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Config ──
CONSTRAINED_DIR = Path("output/swopp3_gpu")
NOCONSTRAINT_DIR = Path("output/pacific_noconstraint")
FIG_DIR = NOCONSTRAINT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CASES = ["PO_noWPS", "PGC_noWPS"]
LABELS = {"PO_noWPS": "Optimised (no WPS)", "PGC_noWPS": "Great Circle (no WPS)"}

MAX_TWS = 20.0  # m/s operational limit
MAX_HS = 7.0  # m operational limit


def _load(directory: Path, case: str) -> pd.DataFrame:
    """Load File A CSV for a case, trying common submission numbers."""
    for sub in [1, 2, 3]:
        path = directory / f"IEUniversity-{sub}-{case}.csv"
        if path.exists():
            df = pd.read_csv(path, parse_dates=["departure_time_utc"])
            df["departure"] = df["departure_time_utc"]
            return df
    raise FileNotFoundError(f"No File A CSV found for {case} in {directory}")


def plot_energy_timeseries(fig_dir: Path) -> None:
    """Energy time-series: constrained vs no-constraint, both cases."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    for ax, case in zip(axes, CASES, strict=False):
        con = _load(CONSTRAINED_DIR, case)
        nocon = _load(NOCONSTRAINT_DIR, case)

        ax.plot(
            con["departure"],
            con["energy_cons_mwh"],
            color="#2166ac",
            linewidth=0.8,
            alpha=0.8,
            label="Constrained (wpw=100)",
        )
        ax.plot(
            nocon["departure"],
            nocon["energy_cons_mwh"],
            color="#b2182b",
            linewidth=0.8,
            alpha=0.8,
            label="No constraints (wpw=0)",
        )

        mean_con = con["energy_cons_mwh"].mean()
        mean_nocon = nocon["energy_cons_mwh"].mean()
        ax.axhline(mean_con, color="#2166ac", linestyle="--", alpha=0.4)
        ax.axhline(mean_nocon, color="#b2182b", linestyle="--", alpha=0.4)

        ax.set_ylabel("Energy (MWh)")
        ax.set_title(
            f"{LABELS[case]}  —  "
            f"Constrained: {mean_con:.1f} MWh  |  "
            f"No-constraint: {mean_nocon:.1f} MWh  |  "
            f"Δ = {(mean_nocon - mean_con) / mean_con * 100:+.1f}%"
        )
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Departure date")
    fig.suptitle(
        "Pacific Corridor — Constrained vs No-Constraint Energy",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout()
    out = fig_dir / "energy_timeseries_comparison.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {out}")


def plot_energy_delta(fig_dir: Path) -> None:
    """Per-departure energy delta (%) between constrained and no-constraint."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    for ax, case in zip(axes, CASES, strict=False):
        con = _load(CONSTRAINED_DIR, case)
        nocon = _load(NOCONSTRAINT_DIR, case)

        delta_pct = (
            (nocon["energy_cons_mwh"].values - con["energy_cons_mwh"].values)
            / con["energy_cons_mwh"].values
            * 100
        )

        colors = np.where(delta_pct < 0, "#1a9850", "#d73027")
        ax.bar(con["departure"], delta_pct, color=colors, width=1.0, alpha=0.7)
        ax.axhline(0, color="black", linewidth=0.5)

        n_better = np.sum(delta_pct < -0.5)
        n_worse = np.sum(delta_pct > 0.5)
        ax.set_ylabel("Energy Δ (%)")
        ax.set_title(
            f"{LABELS[case]}  —  "
            f"No-constraint better: {n_better}/{len(delta_pct)}  |  "
            f"Worse: {n_worse}/{len(delta_pct)}  |  "
            f"Mean Δ = {np.mean(delta_pct):+.1f}%"
        )
        ax.grid(alpha=0.3, axis="y")

    axes[-1].set_xlabel("Departure date")
    fig.suptitle(
        "Energy Change: No-Constraint minus Constrained",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout()
    out = fig_dir / "energy_delta_per_departure.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {out}")


def plot_weather_exposure(fig_dir: Path) -> None:
    """Max TWS and Hs: constrained vs no-constraint."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    for col, case in enumerate(CASES):
        con = _load(CONSTRAINED_DIR, case)
        nocon = _load(NOCONSTRAINT_DIR, case)

        # TWS
        ax = axes[0, col]
        ax.scatter(
            con["departure"],
            con["max_wind_mps"],
            s=4,
            alpha=0.5,
            color="#2166ac",
            label="Constrained",
        )
        ax.scatter(
            nocon["departure"],
            nocon["max_wind_mps"],
            s=4,
            alpha=0.5,
            color="#b2182b",
            label="No constraints",
        )
        ax.axhline(MAX_TWS, color="orange", linestyle="--", linewidth=1, label="Limit")
        ax.set_ylabel("Max TWS (m/s)")
        ax.set_title(f"{LABELS[case]} — TWS")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3)

        # Hs
        ax = axes[1, col]
        ax.scatter(
            con["departure"],
            con["max_hs_m"],
            s=4,
            alpha=0.5,
            color="#2166ac",
            label="Constrained",
        )
        ax.scatter(
            nocon["departure"],
            nocon["max_hs_m"],
            s=4,
            alpha=0.5,
            color="#b2182b",
            label="No constraints",
        )
        ax.axhline(MAX_HS, color="orange", linestyle="--", linewidth=1, label="Limit")
        ax.set_ylabel("Max Hs (m)")
        ax.set_title(f"{LABELS[case]} — Hs")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3)

    fig.suptitle(
        "Weather Exposure — Constrained vs No-Constraint",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout()
    out = fig_dir / "weather_exposure_comparison.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {out}")


def plot_distance_comparison(fig_dir: Path) -> None:
    """Sailed distance: constrained vs no-constraint."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, case in zip(axes, CASES, strict=False):
        con = _load(CONSTRAINED_DIR, case)
        nocon = _load(NOCONSTRAINT_DIR, case)

        ax.plot(
            con["departure"],
            con["sailed_distance_nm"],
            color="#2166ac",
            linewidth=0.8,
            alpha=0.8,
            label="Constrained",
        )
        ax.plot(
            nocon["departure"],
            nocon["sailed_distance_nm"],
            color="#b2182b",
            linewidth=0.8,
            alpha=0.8,
            label="No constraints",
        )

        gc_dist = 4653  # approximate GC distance
        ax.axhline(gc_dist, color="gray", linestyle=":", alpha=0.5, label="GC dist")
        ax.set_ylabel("Sailed distance (nm)")
        ax.set_xlabel("Departure date")
        ax.set_title(f"{LABELS[case]}")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle(
        "Sailed Distance — Constrained vs No-Constraint",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    out = fig_dir / "distance_comparison.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {out}")


def main() -> None:
    """Plot comparison figures for no-constraint experiment."""
    print("Plotting no-constraint comparison figures...")
    print(f"  Constrained:    {CONSTRAINED_DIR}")
    print(f"  No-constraint:  {NOCONSTRAINT_DIR}")
    print(f"  Output:         {FIG_DIR}\n")

    plot_energy_timeseries(FIG_DIR)
    plot_energy_delta(FIG_DIR)
    plot_weather_exposure(FIG_DIR)
    plot_distance_comparison(FIG_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
