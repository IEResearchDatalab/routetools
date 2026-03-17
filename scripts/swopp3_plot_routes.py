#!/usr/bin/env python
"""Visualize SWOPP3 routes from output CSV files.

Generates publication-quality maps comparing Great-Circle vs Optimised
routes for each corridor (Atlantic, Pacific) and WPS configuration.

Usage
-----
    python scripts/swopp3_plot_routes.py --input-dir output/swopp3_rise

Outputs PNG figures into ``<input-dir>/figures/``.
Missing or partially written outputs are skipped so plotting can run
against an in-progress SWOPP3 run.
"""

from __future__ import annotations

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import pandas as pd
import typer

from routetools.swopp3_output import read_file_a_dataframe, read_file_b_dataframe

app = typer.Typer(help="Visualize SWOPP3 route outputs.")

_FILE_A_REQUIRED_COLUMNS = {
    "departure_time_utc",
    "arrival_time_utc",
    "energy_cons_mwh",
    "max_wind_mps",
    "max_hs_m",
    "sailed_distance_nm",
    "details_filename",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _skip(message: str) -> None:
    """Print a concise skip message for partial or missing data."""
    print(f"  Skipping: {message}")


def _is_missing_or_partial_csv(exc: Exception) -> bool:
    """Return whether an exception likely came from partial output files."""
    return isinstance(
        exc,
        (FileNotFoundError | pd.errors.EmptyDataError | pd.errors.ParserError),
    )


def _load_summary(
    input_dir: Path,
    case_id: str,
    submission: int | None = None,
) -> pd.DataFrame | None:
    """Load a case summary table from File A CSV.

    Returns ``None`` when the case output is not available yet or the CSV is
    still being written.
    """
    try:
        df = read_file_a_dataframe(input_dir, case_id, submission=submission)
    except Exception as exc:
        if _is_missing_or_partial_csv(exc):
            _skip(f"summary for {case_id} is not available yet")
            return None
        raise

    missing_cols = _FILE_A_REQUIRED_COLUMNS.difference(df.columns)
    if missing_cols:
        _skip(
            "summary for "
            f"{case_id} is missing columns: {', '.join(sorted(missing_cols))}"
        )
        return None
    if df.empty:
        _skip(f"summary for {case_id} has no rows yet")
        return None

    return df.sort_values("departure_time_utc").reset_index(drop=True)


def _load_track(input_dir: Path, filename: str) -> pd.DataFrame | None:
    """Load a single track CSV.

    Returns ``None`` when the track is missing or still being written.
    """
    try:
        track = read_file_b_dataframe(input_dir, filename)
    except Exception as exc:
        if _is_missing_or_partial_csv(exc):
            return None
        raise

    if track.empty or not {"lon_deg", "lat_deg"}.issubset(track.columns):
        return None
    return track


def _load_tracks_from_summary(
    input_dir: Path,
    summary: pd.DataFrame,
) -> list[tuple[pd.Series, pd.DataFrame]]:
    """Load all readable tracks referenced by a summary table."""
    tracks: list[tuple[pd.Series, pd.DataFrame]] = []
    skipped = 0

    for _, row in summary.iterrows():
        filename = row["details_filename"]
        if not isinstance(filename, str) or not filename:
            skipped += 1
            continue
        track = _load_track(input_dir, filename)
        if track is None:
            skipped += 1
            continue
        tracks.append((row, track))

    if skipped:
        _skip(f"ignored {skipped} incomplete or missing track files")
    return tracks


def _align_case_summaries(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Align two case summaries on departure time for fair comparisons."""
    merged = pd.merge(
        left[["departure_time_utc", "energy_cons_mwh", "sailed_distance_nm"]],
        right[["departure_time_utc", "energy_cons_mwh", "sailed_distance_nm"]],
        on="departure_time_utc",
        how="inner",
        suffixes=("_left", "_right"),
    )
    return merged.sort_values("departure_time_utc").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Figure 1: All routes for one corridor (GC + Optimised, spaghetti)
# ---------------------------------------------------------------------------
def plot_corridor_spaghetti(
    input_dir: Path,
    corridor: str,
    wps: bool,
    fig_dir: Path,
    submission: int | None = None,
    n_departures: int = 366,
    sample_step: int = 1,
) -> None:
    """Plot all departure routes for one corridor overlaid on a map.

    GC routes in blue, optimised routes in red, with low alpha for the
    spaghetti and a thicker line for the median representative route.
    """
    wps_label = "WPS" if wps else "noWPS"
    gc_case = f"{'A' if corridor == 'atlantic' else 'P'}GC_{wps_label}"
    opt_case = f"{'A' if corridor == 'atlantic' else 'P'}O_{wps_label}"

    gc_summary = _load_summary(input_dir, gc_case, submission=submission)
    opt_summary = _load_summary(input_dir, opt_case, submission=submission)
    if gc_summary is None and opt_summary is None:
        _skip(f"no data available for {corridor} {wps_label}")
        return

    gc_tracks = (
        [] if gc_summary is None else _load_tracks_from_summary(input_dir, gc_summary)
    )
    opt_tracks = (
        [] if opt_summary is None else _load_tracks_from_summary(input_dir, opt_summary)
    )
    if not gc_tracks and not opt_tracks:
        _skip(f"no readable tracks available for {corridor} {wps_label}")
        return

    if corridor == "pacific":
        proj = ccrs.PlateCarree(central_longitude=180)
        transform = ccrs.PlateCarree()
        extent = [120, 260, 20, 60]
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

    endpoint_track: tuple[pd.Series, pd.Series] | None = None

    if gc_tracks:
        _, gc_track = gc_tracks[0]
        lons_gc = gc_track["lon_deg"]
        lats_gc = gc_track["lat_deg"]
        endpoint_track = (lons_gc, lats_gc)
        ax.plot(
            lons_gc.values,
            lats_gc.values,
            color="#2166ac",
            linewidth=2.5,
            alpha=0.9,
            transform=transform,
            label="Great Circle",
            zorder=5,
        )

    for _, track in opt_tracks[:n_departures:sample_step]:
        ax.plot(
            track["lon_deg"].values,
            track["lat_deg"].values,
            color="#b2182b",
            linewidth=0.4,
            alpha=0.15,
            transform=transform,
            zorder=3,
        )

    if opt_tracks and opt_summary is not None:
        _, med_track = min(
            opt_tracks,
            key=lambda item: abs(
                item[0]["energy_cons_mwh"] - opt_summary["energy_cons_mwh"].median()
            ),
        )
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
        if endpoint_track is None:
            endpoint_track = (med_track["lon_deg"], med_track["lat_deg"])

    if endpoint_track is not None:
        end_lons, end_lats = endpoint_track
        ax.plot(
            end_lons.iloc[0],
            end_lats.iloc[0],
            "o",
            color="#1a9850",
            markersize=8,
            transform=transform,
            zorder=10,
        )
        ax.plot(
            end_lons.iloc[-1],
            end_lats.iloc[-1],
            "s",
            color="#d73027",
            markersize=8,
            transform=transform,
            zorder=10,
        )

    title_lines = [f"{corridor.title()} Corridor - {wps_label}"]
    if gc_summary is not None:
        title_lines.append(
            "GC mean: "
            f"{gc_summary['energy_cons_mwh'].mean():.1f} MWh (n={len(gc_summary)})"
        )
    if opt_summary is not None:
        title_lines.append(
            "Optimised mean: "
            f"{opt_summary['energy_cons_mwh'].mean():.1f} MWh (n={len(opt_summary)})"
        )
    if gc_summary is not None and opt_summary is not None:
        overlap = _align_case_summaries(gc_summary, opt_summary)
        if not overlap.empty:
            savings_pct = (
                1
                - overlap["energy_cons_mwh_right"].mean()
                / overlap["energy_cons_mwh_left"].mean()
            ) * 100
            title_lines.append(
                f"Overlap savings: {savings_pct:.1f}% (n={len(overlap)})"
            )

    ax.set_title("\n".join(title_lines), fontsize=13, fontweight="bold")
    ax.legend(loc="lower left", fontsize=10)

    out = fig_dir / f"routes_{corridor}_{wps_label}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Figure 2: Energy distribution comparison (violin / box)
# ---------------------------------------------------------------------------
def plot_energy_comparison(
    input_dir: Path,
    fig_dir: Path,
    submission: int | None = None,
) -> None:
    """Box plots comparing energy across the available cases."""
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
    labels_present = []
    colors_present = []
    case_positions: dict[str, int] = {}
    for cid, label, color in zip(cases_order, labels, colors, strict=False):
        df = _load_summary(input_dir, cid, submission=submission)
        if df is None:
            continue
        data.append(df["energy_cons_mwh"].values)
        labels_present.append(label)
        colors_present.append(color)
        case_positions[cid] = len(data)

    if not data:
        _skip("no summaries available for energy comparison")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    bp = ax.boxplot(
        data,
        tick_labels=labels_present,
        patch_artist=True,
        widths=0.6,
        showfliers=True,
        flierprops=dict(marker=".", markersize=3, alpha=0.4),
    )
    for patch, color in zip(bp["boxes"], colors_present, strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

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
        "SWOPP3 Energy Consumption - Available Cases",
        fontsize=13,
        fontweight="bold",
    )

    atlantic_positions = [
        pos for cid, pos in case_positions.items() if cid.startswith("A")
    ]
    pacific_positions = [
        pos for cid, pos in case_positions.items() if cid.startswith("P")
    ]
    y_top = ax.get_ylim()[1] * 0.97
    if atlantic_positions:
        ax.text(
            sum(atlantic_positions) / len(atlantic_positions),
            y_top,
            "Atlantic",
            ha="center",
            fontsize=11,
            fontstyle="italic",
            alpha=0.7,
        )
    if pacific_positions:
        ax.text(
            sum(pacific_positions) / len(pacific_positions),
            y_top,
            "Pacific",
            ha="center",
            fontsize=11,
            fontstyle="italic",
            alpha=0.7,
        )
    if atlantic_positions and pacific_positions:
        ax.axvline(
            (max(atlantic_positions) + min(pacific_positions)) / 2,
            color="gray",
            linestyle="--",
            linewidth=0.8,
            alpha=0.5,
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
def plot_energy_timeseries(
    input_dir: Path,
    fig_dir: Path,
    submission: int | None = None,
) -> None:
    """Plot energy consumption over time for each corridor."""
    for corridor in ["atlantic", "pacific"]:
        prefix = "A" if corridor == "atlantic" else "P"

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        plotted = False
        for i, wps_label in enumerate(["noWPS", "WPS"]):
            ax = axes[i]
            gc_case = f"{prefix}GC_{wps_label}"
            opt_case = f"{prefix}O_{wps_label}"

            gc_df = _load_summary(input_dir, gc_case, submission=submission)
            opt_df = _load_summary(input_dir, opt_case, submission=submission)

            if gc_df is None and opt_df is None:
                ax.text(
                    0.5,
                    0.5,
                    "No data yet",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=11,
                    alpha=0.7,
                )
                ax.set_axis_off()
                continue

            if gc_df is not None:
                plotted = True
                ax.plot(
                    gc_df["departure_time_utc"],
                    gc_df["energy_cons_mwh"],
                    color="#2166ac",
                    linewidth=1,
                    alpha=0.8,
                    label=f"GC ({gc_df['energy_cons_mwh'].mean():.0f} MWh avg)",
                )
            if opt_df is not None:
                plotted = True
                ax.plot(
                    opt_df["departure_time_utc"],
                    opt_df["energy_cons_mwh"],
                    color="#b2182b",
                    linewidth=1,
                    alpha=0.8,
                    label=f"Opt ({opt_df['energy_cons_mwh'].mean():.0f} MWh avg)",
                )

            overlap = None
            if gc_df is not None and opt_df is not None:
                overlap = _align_case_summaries(gc_df, opt_df)
                if not overlap.empty:
                    ax.fill_between(
                        overlap["departure_time_utc"],
                        overlap["energy_cons_mwh_left"],
                        overlap["energy_cons_mwh_right"],
                        alpha=0.2,
                        color="#2ca02c",
                        label="Savings",
                    )

            ax.set_ylabel("Energy (MWh)", fontsize=11)
            if overlap is not None and not overlap.empty:
                savings = (
                    1
                    - overlap["energy_cons_mwh_right"].mean()
                    / overlap["energy_cons_mwh_left"].mean()
                ) * 100
                title = f"{wps_label} - Savings: {savings:.1f}% (n={len(overlap)})"
            else:
                counts = []
                if gc_df is not None:
                    counts.append(f"GC n={len(gc_df)}")
                if opt_df is not None:
                    counts.append(f"Opt n={len(opt_df)}")
                title = f"{wps_label} - {'  '.join(counts)}"
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.legend(loc="upper right", fontsize=9)
            ax.grid(alpha=0.3)

        if not plotted:
            plt.close(fig)
            _skip(f"no time-series data available for {corridor}")
            continue

        fig.suptitle(
            f"{corridor.title()} Corridor - Energy Consumption over 2024",
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
def plot_seasonal_routes(
    input_dir: Path,
    fig_dir: Path,
    submission: int | None = None,
) -> None:
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

            opt_df = _load_summary(input_dir, opt_case, submission=submission)
            gc_df = _load_summary(input_dir, gc_case, submission=submission)
            if opt_df is None:
                _skip(f"no optimised summary available for {corridor} {wps_label}")
                continue

            opt_tracks = _load_tracks_from_summary(input_dir, opt_df)
            if not opt_tracks:
                _skip(
                    f"no readable optimised tracks available for {corridor} {wps_label}"
                )
                continue

            gc_tracks = (
                [] if gc_df is None else _load_tracks_from_summary(input_dir, gc_df)
            )

            best_row, best_track = min(
                opt_tracks,
                key=lambda item: item[0]["energy_cons_mwh"],
            )
            worst_row, worst_track = max(
                opt_tracks,
                key=lambda item: item[0]["energy_cons_mwh"],
            )
            median_row, median_track = min(
                opt_tracks,
                key=lambda item: abs(
                    item[0]["energy_cons_mwh"] - opt_df["energy_cons_mwh"].median()
                ),
            )

            picks = {
                "Best": (best_row, best_track, "#1a9850"),
                "Median": (median_row, median_track, "#f46d43"),
                "Worst": (worst_row, worst_track, "#d73027"),
            }

            fig, ax = plt.subplots(figsize=(14, 7), subplot_kw={"projection": proj})
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.LAND, facecolor="#e8e8e8", edgecolor="none")
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color="#888888")
            ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)

            if gc_tracks:
                _, gc_track = gc_tracks[0]
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

            for label_name, (row, track, color) in picks.items():
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
                f"{corridor.title()} Optimised - {wps_label}\n"
                "Best / Median / Worst from available departures "
                f"(n={len(opt_tracks)})",
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
def plot_distance_vs_energy(
    input_dir: Path,
    fig_dir: Path,
    submission: int | None = None,
) -> None:
    """Scatter plot of sailed distance vs energy, colored by departure month."""
    for corridor in ["atlantic", "pacific"]:
        prefix = "A" if corridor == "atlantic" else "P"

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        scatter = None
        plotted = False

        for i, wps_label in enumerate(["noWPS", "WPS"]):
            ax = axes[i]
            opt_case = f"{prefix}O_{wps_label}"
            gc_case = f"{prefix}GC_{wps_label}"

            opt_df = _load_summary(input_dir, opt_case, submission=submission)
            gc_df = _load_summary(input_dir, gc_case, submission=submission)

            if opt_df is None and gc_df is None:
                ax.text(
                    0.5,
                    0.5,
                    "No data yet",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=11,
                    alpha=0.7,
                )
                ax.set_axis_off()
                continue

            if opt_df is not None:
                months = pd.to_datetime(opt_df["departure_time_utc"]).dt.month
                scatter = ax.scatter(
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
                plotted = True

            if gc_df is not None:
                gc_months = pd.to_datetime(gc_df["departure_time_utc"]).dt.month
                scatter = ax.scatter(
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
                plotted = True

            ax.set_xlabel("Sailed Distance (nm)", fontsize=11)
            ax.set_ylabel("Energy (MWh)", fontsize=11)
            ax.set_title(f"{wps_label}", fontsize=12, fontweight="bold")
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)

        if not plotted or scatter is None:
            plt.close(fig)
            _skip(f"no distance-vs-energy data available for {corridor}")
            continue

        fig.suptitle(
            f"{corridor.title()} - Distance vs Energy (colored by month)",
            fontsize=13,
            fontweight="bold",
        )
        cbar = fig.colorbar(scatter, ax=axes, label="Month", ticks=range(1, 13))
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
        "output/swopp3",
        "--input-dir",
        "-i",
        help="Directory containing SWOPP3 output CSVs.",
    ),
    submission: int | None = typer.Option(  # noqa: B008
        None,
        "--submission",
        "-s",
        help="Submission number to plot (default: auto-detect latest per case).",
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

    print("\n[1/5] Route spaghetti maps...")
    for corridor in ["atlantic", "pacific"]:
        for wps in [True, False]:
            plot_corridor_spaghetti(
                input_dir,
                corridor,
                wps,
                fig_dir,
                submission=submission,
                sample_step=sample_step,
            )

    print("\n[2/5] Energy comparison box plots...")
    plot_energy_comparison(input_dir, fig_dir, submission=submission)

    print("\n[3/5] Energy time series...")
    plot_energy_timeseries(input_dir, fig_dir, submission=submission)

    print("\n[4/5] Seasonal representative routes...")
    plot_seasonal_routes(input_dir, fig_dir, submission=submission)

    print("\n[5/5] Distance vs Energy scatter...")
    plot_distance_vs_energy(input_dir, fig_dir, submission=submission)

    print(f"\nDone: {len(list(fig_dir.glob('*.png')))} figures in {fig_dir}")


if __name__ == "__main__":
    app()
