#!/usr/bin/env python
"""Visualize SWOPP3 routes from output CSV files.

Generates publication-quality maps comparing Great-Circle vs Optimised
routes for each corridor (Atlantic, Pacific) and WPS configuration.

Usage
-----
    python scripts/swopp3_plot_routes.py --input-dir output/swopp3_rise

Outputs PNG figures into ``<input-dir>/figures/``.
"""

from __future__ import annotations

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import pandas as pd
import typer

app = typer.Typer(help="Visualize SWOPP3 route outputs.")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_summary(input_dir: Path, case_id: str, submission: int = 1) -> pd.DataFrame:
    """Load the summary CSV for a case."""
    path = input_dir / f"IEUniversity-{submission}-{case_id}.csv"
    return pd.read_csv(path, parse_dates=["departure_time_utc", "arrival_time_utc"])


def _load_track(input_dir: Path, filename: str) -> pd.DataFrame:
    """Load a single track (waypoints) CSV."""
    path = input_dir / "tracks" / filename
    return pd.read_csv(path, parse_dates=["time_utc"])


# ---------------------------------------------------------------------------
# Figure 1: All routes for one corridor (GC + Optimised, spaghetti)
# ---------------------------------------------------------------------------


def plot_corridor_spaghetti(
    input_dir: Path,
    corridor: str,
    wps: bool,
    fig_dir: Path,
    n_departures: int = 366,
    sample_step: int = 1,
) -> None:
    """Plot all departure routes for one corridor overlaid on a map.

    GC routes in blue, optimised routes in red, with low alpha for the
    spaghetti and a thicker line for the mean/median representative route.
    """
    wps_label = "WPS" if wps else "noWPS"
    gc_case = f"{'A' if corridor == 'atlantic' else 'P'}GC_{wps_label}"
    opt_case = f"{'A' if corridor == 'atlantic' else 'P'}O_{wps_label}"

    gc_summary = _load_summary(input_dir, gc_case)
    opt_summary = _load_summary(input_dir, opt_case)

    # Use central_longitude to avoid antimeridian wrapping issues
    if corridor == "pacific":
        proj = ccrs.PlateCarree(central_longitude=180)
        transform = ccrs.PlateCarree()
        extent = [120, 260, 20, 60]  # [lon_min, lon_max, lat_min, lat_max]
    else:
        proj = ccrs.PlateCarree()
        transform = ccrs.PlateCarree()
        extent = [-80, 5, 30, 60]

    fig, ax = plt.subplots(figsize=(14, 7), subplot_kw={"projection": proj})
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="#e8e8e8", edgecolor="none")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color="#888888")
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, color="#cccccc")
    ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)

    # --- Plot GC routes (all identical, just plot one thick) ---
    gc_track = _load_track(input_dir, gc_summary.iloc[0]["details_filename"])
    lons_gc = gc_track["lon_deg"].values
    lats_gc = gc_track["lat_deg"].values
    ax.plot(
        lons_gc,
        lats_gc,
        color="#2166ac",
        linewidth=2.5,
        alpha=0.9,
        transform=transform,
        label="Great Circle",
        zorder=5,
    )

    # --- Plot optimised routes ---
    indices = range(0, min(n_departures, len(opt_summary)), sample_step)
    for i in indices:
        row = opt_summary.iloc[i]
        track = _load_track(input_dir, row["details_filename"])
        lons = track["lon_deg"].values
        lats = track["lat_deg"].values
        ax.plot(
            lons,
            lats,
            color="#b2182b",
            linewidth=0.4,
            alpha=0.15,
            transform=transform,
            zorder=3,
        )

    # Plot the median-energy optimised route thicker
    median_idx = (
        opt_summary["energy_cons_mwh"]
        .sub(opt_summary["energy_cons_mwh"].median())
        .abs()
        .idxmin()
    )
    med_row = opt_summary.loc[median_idx]
    med_track = _load_track(input_dir, med_row["details_filename"])
    ax.plot(
        med_track["lon_deg"].values,
        med_track["lat_deg"].values,
        color="#b2182b",
        linewidth=2.5,
        alpha=0.9,
        transform=transform,
        label="Optimised (median)",
        zorder=6,
    )

    # Endpoints
    ax.plot(
        lons_gc[0],
        lats_gc[0],
        "o",
        color="#1a9850",
        markersize=8,
        transform=transform,
        zorder=10,
    )
    ax.plot(
        lons_gc[-1],
        lats_gc[-1],
        "s",
        color="#d73027",
        markersize=8,
        transform=transform,
        zorder=10,
    )

    # Energy stats
    gc_energy = gc_summary["energy_cons_mwh"].mean()
    opt_energy = opt_summary["energy_cons_mwh"].mean()
    savings_pct = (1 - opt_energy / gc_energy) * 100

    ax.set_title(
        f"{corridor.title()} Corridor — {wps_label}\n"
        f"GC: {gc_energy:.1f} MWh  |  Optimised: {opt_energy:.1f} MWh  "
        f"| Savings: {savings_pct:.1f}%",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(loc="lower left", fontsize=10)

    out = fig_dir / f"routes_{corridor}_{wps_label}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Figure 2: Energy distribution comparison (violin / box)
# ---------------------------------------------------------------------------


def plot_energy_comparison(input_dir: Path, fig_dir: Path) -> None:
    """Box plots comparing energy across all 8 cases."""
    cases_order = [
        "AGC_noWPS",
        "AO_noWPS",
        "AGC_WPS",
        "AO_WPS",
        "PGC_noWPS",
        "PO_noWPS",
        "PGC_WPS",
        "PO_WPS",
    ]
    labels = [
        "A-GC\nno WPS",
        "A-Opt\nno WPS",
        "A-GC\nWPS",
        "A-Opt\nWPS",
        "P-GC\nno WPS",
        "P-Opt\nno WPS",
        "P-GC\nWPS",
        "P-Opt\nWPS",
    ]
    colors = [
        "#2166ac",
        "#b2182b",
        "#2166ac",
        "#b2182b",
        "#2166ac",
        "#b2182b",
        "#2166ac",
        "#b2182b",
    ]

    data = []
    for cid in cases_order:
        df = _load_summary(input_dir, cid)
        data.append(df["energy_cons_mwh"].values)

    fig, ax = plt.subplots(figsize=(12, 6))
    bp = ax.boxplot(
        data,
        tick_labels=labels,
        patch_artist=True,
        widths=0.6,
        showfliers=True,
        flierprops=dict(marker=".", markersize=3, alpha=0.4),
    )
    for patch, color in zip(bp["boxes"], colors, strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Add mean markers
    means = [d.mean() for d in data]
    ax.scatter(
        range(1, len(means) + 1),
        means,
        color="black",
        marker="D",
        s=40,
        zorder=5,
        label="Mean",
    )

    ax.set_ylabel("Energy Consumption (MWh)", fontsize=12)
    ax.set_title(
        "SWOPP3 Energy Consumption — All Cases (366 departures each)",
        fontsize=13,
        fontweight="bold",
    )

    # Add vertical separator between Atlantic and Pacific
    ax.axvline(4.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.text(
        2.5,
        ax.get_ylim()[1] * 0.97,
        "Atlantic",
        ha="center",
        fontsize=11,
        fontstyle="italic",
        alpha=0.7,
    )
    ax.text(
        6.5,
        ax.get_ylim()[1] * 0.97,
        "Pacific",
        ha="center",
        fontsize=11,
        fontstyle="italic",
        alpha=0.7,
    )

    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    out = fig_dir / "energy_comparison_boxplot.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Figure 3: Energy time series (daily departures)
# ---------------------------------------------------------------------------


def plot_energy_timeseries(input_dir: Path, fig_dir: Path) -> None:
    """Plot energy consumption over time for each corridor."""
    for corridor in ["atlantic", "pacific"]:
        prefix = "A" if corridor == "atlantic" else "P"

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        for i, wps_label in enumerate(["noWPS", "WPS"]):
            ax = axes[i]
            gc_case = f"{prefix}GC_{wps_label}"
            opt_case = f"{prefix}O_{wps_label}"

            gc_df = _load_summary(input_dir, gc_case)
            opt_df = _load_summary(input_dir, opt_case)

            dates = gc_df["departure_time_utc"]
            ax.fill_between(
                dates,
                gc_df["energy_cons_mwh"],
                opt_df["energy_cons_mwh"],
                alpha=0.2,
                color="#2ca02c",
                label="Savings",
            )
            ax.plot(
                dates,
                gc_df["energy_cons_mwh"],
                color="#2166ac",
                linewidth=1,
                alpha=0.8,
                label=f"GC ({gc_df['energy_cons_mwh'].mean():.0f} MWh avg)",
            )
            ax.plot(
                dates,
                opt_df["energy_cons_mwh"],
                color="#b2182b",
                linewidth=1,
                alpha=0.8,
                label=f"Opt ({opt_df['energy_cons_mwh'].mean():.0f} MWh avg)",
            )

            savings = (
                1 - opt_df["energy_cons_mwh"].mean() / gc_df["energy_cons_mwh"].mean()
            ) * 100
            ax.set_ylabel("Energy (MWh)", fontsize=11)
            ax.set_title(
                f"{wps_label} — Savings: {savings:.1f}%", fontsize=11, fontweight="bold"
            )
            ax.legend(loc="upper right", fontsize=9)
            ax.grid(alpha=0.3)

        fig.suptitle(
            f"{corridor.title()} Corridor — Energy Consumption over 2024",
            fontsize=13,
            fontweight="bold",
        )
        axes[-1].set_xlabel("Departure Date", fontsize=11)
        fig.autofmt_xdate(rotation=30)

        out = fig_dir / f"energy_timeseries_{corridor}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Figure 4: Seasonal sample routes (best, worst, median)
# ---------------------------------------------------------------------------


def plot_seasonal_routes(input_dir: Path, fig_dir: Path) -> None:
    """Plot representative routes for different seasons on a single map."""
    for corridor in ["atlantic", "pacific"]:
        prefix = "A" if corridor == "atlantic" else "P"

        if corridor == "pacific":
            proj = ccrs.PlateCarree(central_longitude=180)
            transform = ccrs.PlateCarree()
            extent = [120, 260, 20, 60]
        else:
            proj = ccrs.PlateCarree()
            transform = ccrs.PlateCarree()
            extent = [-80, 5, 30, 60]

        for wps_label in ["WPS", "noWPS"]:
            opt_case = f"{prefix}O_{wps_label}"
            gc_case = f"{prefix}GC_{wps_label}"

            opt_df = _load_summary(input_dir, opt_case)
            gc_df = _load_summary(input_dir, gc_case)

            # Pick best, worst, and median departures
            best_idx = opt_df["energy_cons_mwh"].idxmin()
            worst_idx = opt_df["energy_cons_mwh"].idxmax()
            median_idx = (
                opt_df["energy_cons_mwh"]
                .sub(opt_df["energy_cons_mwh"].median())
                .abs()
                .idxmin()
            )

            picks = {
                "Best": (best_idx, "#1a9850"),
                "Median": (median_idx, "#f46d43"),
                "Worst": (worst_idx, "#d73027"),
            }

            fig, ax = plt.subplots(figsize=(14, 7), subplot_kw={"projection": proj})
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.LAND, facecolor="#e8e8e8", edgecolor="none")
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color="#888888")
            ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)

            # GC baseline
            gc_track = _load_track(input_dir, gc_df.iloc[0]["details_filename"])
            ax.plot(
                gc_track["lon_deg"].values,
                gc_track["lat_deg"].values,
                color="#2166ac",
                linewidth=2,
                alpha=0.7,
                transform=transform,
                label="Great Circle",
                linestyle="--",
            )

            for label_name, (idx, color) in picks.items():
                row = opt_df.loc[idx]
                track = _load_track(input_dir, row["details_filename"])
                dep_date = pd.to_datetime(row["departure_time_utc"]).strftime("%b %d")
                energy = row["energy_cons_mwh"]
                ax.plot(
                    track["lon_deg"].values,
                    track["lat_deg"].values,
                    color=color,
                    linewidth=2.5,
                    alpha=0.9,
                    transform=transform,
                    label=f"{label_name}: {dep_date} ({energy:.0f} MWh)",
                )

            ax.set_title(
                f"{corridor.title()} Optimised — {wps_label}\n"
                f"Best / Median / Worst departures out of 366",
                fontsize=13,
                fontweight="bold",
            )
            ax.legend(loc="lower left", fontsize=10)

            out = fig_dir / f"seasonal_routes_{corridor}_{wps_label}.png"
            fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Figure 5: Distance vs Energy scatter
# ---------------------------------------------------------------------------


def plot_distance_vs_energy(input_dir: Path, fig_dir: Path) -> None:
    """Scatter plot of sailed distance vs energy, colored by departure month."""
    for corridor in ["atlantic", "pacific"]:
        prefix = "A" if corridor == "atlantic" else "P"

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for i, wps_label in enumerate(["noWPS", "WPS"]):
            ax = axes[i]
            opt_case = f"{prefix}O_{wps_label}"
            gc_case = f"{prefix}GC_{wps_label}"

            opt_df = _load_summary(input_dir, opt_case)
            gc_df = _load_summary(input_dir, gc_case)

            months = pd.to_datetime(opt_df["departure_time_utc"]).dt.month
            sc = ax.scatter(
                opt_df["sailed_distance_nm"],
                opt_df["energy_cons_mwh"],
                c=months,
                cmap="hsv",
                s=12,
                alpha=0.7,
                vmin=1,
                vmax=12,
                label="Optimised",
            )
            # GC: same route every departure but energy varies with weather
            gc_months = pd.to_datetime(gc_df["departure_time_utc"]).dt.month
            ax.scatter(
                gc_df["sailed_distance_nm"],
                gc_df["energy_cons_mwh"],
                c=gc_months,
                cmap="hsv",
                s=12,
                alpha=0.7,
                vmin=1,
                vmax=12,
                marker="^",
                label="GC",
            )
            # GC mean reference star
            gc_d = gc_df["sailed_distance_nm"].mean()
            gc_e = gc_df["energy_cons_mwh"].mean()
            ax.scatter(
                gc_d,
                gc_e,
                color="navy",
                marker="*",
                s=200,
                zorder=10,
                edgecolors="white",
                linewidths=0.5,
                label=f"GC mean ({gc_d:.0f} nm, {gc_e:.0f} MWh)",
            )

            ax.set_xlabel("Sailed Distance (nm)", fontsize=11)
            ax.set_ylabel("Energy (MWh)", fontsize=11)
            ax.set_title(f"{wps_label}", fontsize=12, fontweight="bold")
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)

        fig.suptitle(
            f"{corridor.title()} — Distance vs Energy (colored by month)",
            fontsize=13,
            fontweight="bold",
        )
        cbar = fig.colorbar(sc, ax=axes, label="Month", ticks=range(1, 13))
        cbar.set_ticklabels(
            ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
        )

        out = fig_dir / f"distance_vs_energy_{corridor}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------


@app.command()
def main(
    input_dir: Path = typer.Option(  # noqa: B008
        "output/swopp3_rise",
        "--input-dir",
        "-i",
        help="Directory containing SWOPP3 output CSVs.",
    ),
    sample_step: int = typer.Option(  # noqa: B008
        1,
        "--sample-step",
        help="Plot every Nth departure in spaghetti plots (1=all).",
    ),
) -> None:
    """Generate all SWOPP3 route visualizations."""
    fig_dir = input_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SWOPP3 Route Visualizations")
    print("=" * 60)

    # 1. Spaghetti maps: all routes overlaid
    print("\n[1/5] Route spaghetti maps...")
    for corridor in ["atlantic", "pacific"]:
        for wps in [True, False]:
            plot_corridor_spaghetti(
                input_dir,
                corridor,
                wps,
                fig_dir,
                sample_step=sample_step,
            )

    # 2. Energy box plots
    print("\n[2/5] Energy comparison box plots...")
    plot_energy_comparison(input_dir, fig_dir)

    # 3. Energy time series
    print("\n[3/5] Energy time series...")
    plot_energy_timeseries(input_dir, fig_dir)

    # 4. Seasonal representative routes
    print("\n[4/5] Seasonal representative routes...")
    plot_seasonal_routes(input_dir, fig_dir)

    # 5. Distance vs Energy
    print("\n[5/5] Distance vs Energy scatter...")
    plot_distance_vs_energy(input_dir, fig_dir)

    print(f"\nDone — {len(list(fig_dir.glob('*.png')))} figures in {fig_dir}")


if __name__ == "__main__":
    app()
