#!/usr/bin/env python
"""Weather-routing optimization: comprehensive comparative analysis.

Generates the full SWOPP3 comparison figure set plus a summary table for
four experimental conditions across the SWOPP3 2024 competition:

  - CMA-ES (no weather penalty)
  - CMA-ES + FMS (no penalty, with FMS post-refinement)
  - CMA-ES + Penalty (wind/wave penalties in objective)
  - CMA-ES + Penalty + FMS (penalised + FMS post-refinement)

Usage
-----
    # Generate all figures with default paths
    uv run scripts/swopp3_analysis.py

    # Custom data and output directories
    uv run scripts/swopp3_analysis.py \
        --data-dir /path/to/output --output-dir /path/to/figs

    # Generate only selected figures (e.g. fig01 and fig05)
    uv run scripts/swopp3_analysis.py --figures 1 5

    # Higher resolution
    uv run scripts/swopp3_analysis.py --dpi 300

Options
-------
    --data-dir DIR      Root directory containing experiment output folders.
                        Default: <repo_root>/output
    --output-dir DIR    Directory where figures and tables are saved.
                        Default: <repo_root>/output/analysis
    --figures N [N...]  Space-separated list of figure numbers to generate
                        (1–12). Generates all figures if omitted.
    --dpi DPI           Figure resolution in DPI. Default: 180

Outputs
-------
    fig01_energy_overview.pdf / .png
    fig02_optimization_gains.pdf / .png
    fig03_penalty_tradeoff.pdf / .png
    fig04a_seasonality_no_penalty.pdf / .png
    fig04b_seasonality_penalty.pdf / .png
    fig13a_relative_gain_no_penalty.pdf / .png
    fig13b_relative_gain_penalty.pdf / .png
    fig05_wps_impact.pdf / .png
    fig06_fms_improvement.pdf / .png
    fig07_route_maps.pdf / .png
    fig08_risk_calendar.pdf / .png
    fig09_fms_delta_byseason.pdf / .png
    fig10_gc_victory_rate.pdf / .png
    fig11_gc_margin_heatmap.pdf / .png
    fig12_gc_violations.pdf / .png
    table01_summary.csv

Experimental conditions
-----------------------
    no_penalty       CMA-ES, no weather penalty, no FMS
    no_penalty_fms   CMA-ES, no weather penalty, with FMS post-refinement
    penalty          CMA-ES, wind/wave soft penalty in objective, no FMS
    penalty_fms      CMA-ES, wind/wave soft penalty, with FMS post-refinement

    Penalty thresholds: wind > 20 m/s, significant wave height > 7 m Hs.

Cases
-----
    AO_WPS    Atlantic (Santander → New York), vessel with Wind Propulsion System
    AO_noWPS  Atlantic, vessel without WPS
    PO_WPS    Pacific (Tokyo → Los Angeles), vessel with WPS
    PO_noWPS  Pacific, vessel without WPS

    Great-circle baselines (AGC_*, PGC_*) are included as reference.

Figure descriptions
-------------------
    fig01  Violin plots of energy consumption (MWh) per case × experiment,
           with great-circle baseline markers and median-savings annotations.
    fig02  Grouped bar chart of median % energy savings vs the great-circle
           baseline for every case × experiment combination.
    fig03  Three-panel safety/efficiency trade-off: wind-violation rate,
           wave-violation rate, and median energy cost across conditions.
    fig04a Seasonal energy lines — non-penalized (2 × 2 per case). Daily
           energy per departure plotted as a line for GC, CMA-ES, and
           CMA-ES + FMS (no weather penalty).
    fig04b Seasonal energy lines — penalized (2 × 2 per case). Same layout
           for CMA-ES + Penalty and CMA-ES + Penalty + FMS.
    fig13a Relative energy gain vs GC — non-penalized (2 × 2 per case).
           Two lines per panel: CMA-ES and CMA-ES + FMS, as % saving
           over the matched GC departure.
    fig13b Same layout as fig13a for the penalized experiments
           (CMA-ES + Penalty, CMA-ES + Penalty + FMS).
    fig05  Horizontal bars of absolute and relative energy savings from WPS
           (WPS vs no-WPS vessel, same route and experiment).
    fig06  Scatter of CMA-ES energy vs CMA-ES + FMS energy per departure;
           points below the diagonal confirm FMS always reduces energy.
    fig07  Cartopy maps of sampled vessel tracks coloured by experiment,
           showing how penalty routing avoids storm-prone corridors.
    fig08  2 × 2 heatmap (month × experiment) of any-violation rate per
           calendar month, revealing seasonal weather-risk patterns.
    fig09  Grouped bars of % energy reduction delivered by FMS, broken down
           by season and case.
    fig10  Monthly "victory rate" — % of departures that beat the GC energy
           for each case × experiment.
    fig11  Heatmap of median % margin over the great-circle route
           (rows = experiments, columns = months) per case.
    fig12  Side-by-side bars of any-violation rate for the great-circle route
           vs CMA-ES + Penalty + FMS, per month and case.

Data dependencies
-----------------
    The script reads CSV files from ``output/`` experiment folders (not
    tracked in Git). Expected layout for each experiment::

        output/swopp3_<experiment>/
            IEUniversity-1-<case>.csv   # one summary file per case
            tracks/
                <details_filename from summary CSV>  # per-voyage track

    Missing experiment folders or individual CSVs are silently skipped;
    all figures degrade gracefully to show only available data.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------------------
# Paths (defaults; overridden at runtime via CLI args in main())
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.parent
OUTPUT_DIR: Path = _REPO_ROOT / "output"
FIGS_DIR: Path = _REPO_ROOT / "output" / "analysis"

# ---------------------------------------------------------------------------
# Experiment metadata
# ---------------------------------------------------------------------------
EXPERIMENTS: dict[str, dict] = {
    "no_penalty": {
        "folder": "swopp3_no_penalty",
        "label": "CMA-ES",
        "short": "No Penalty",
        "color": "#F23333",  # IE law red — unconstrained
        "color_light": "#FF9B9B",
        "hatch": "",
        "order": 1,
    },
    "no_penalty_fms": {
        "folder": "swopp3_no_penalty_fms",
        "label": "CMA-ES + FMS",
        "short": "No Penalty + FMS",
        "color": "#007A3D",  # emerald green — high contrast with red
        "color_light": "#5CC28A",
        "hatch": "///",
        "order": 2,
    },
    "penalty": {
        "folder": "swopp3_penalty",
        "label": "CMA-ES + Penalty",
        "short": "Penalty",
        "color": "#000066",  # IE primary ocean-blue — constrained
        "color_light": "#6080CC",
        "hatch": "",
        "order": 3,
    },
    "penalty_fms": {
        "folder": "swopp3_penalty_fms",
        "label": "CMA-ES + Penalty + FMS",
        "short": "Penalty + FMS",
        "color": "#E09400",  # amber — high contrast with dark navy
        "color_light": "#FFCC66",
        "hatch": "///",
        "order": 4,
    },
}

# Optimised cases (what we are comparing)
OPT_CASES: dict[str, dict] = {
    "AO_WPS": {
        "label": "Atlantic\nWPS",
        "label_short": "Atl. WPS",
        "route": "atlantic",
        "wps": True,
        "gc": "AGC_WPS",
        "color": "#000066",  # IE ocean-blue
    },
    "AO_noWPS": {
        "label": "Atlantic\nno WPS",
        "label_short": "Atl. noWPS",
        "route": "atlantic",
        "wps": False,
        "gc": "AGC_noWPS",
        "color": "#0097DC",  # IE business blue
    },
    "PO_WPS": {
        "label": "Pacific\nWPS",
        "label_short": "Pac. WPS",
        "route": "pacific",
        "wps": True,
        "gc": "PGC_WPS",
        "color": "#6DC201",  # IE tech green
    },
    "PO_noWPS": {
        "label": "Pacific\nno WPS",
        "label_short": "Pac. noWPS",
        "route": "pacific",
        "wps": False,
        "gc": "PGC_noWPS",
        "color": "#47BFFF",  # IE sea-blue
    },
}

# Great-circle baselines (GC = fixed route, constant speed)
GC_CASES = ["AGC_WPS", "AGC_noWPS", "PGC_WPS", "PGC_noWPS"]

# Codabench thresholds
WIND_LIMIT = 20.0  # m/s
WAVE_LIMIT = 7.0  # m

# Season mapping and colours
_MONTH_TO_SEASON = {
    12: "Winter",
    1: "Winter",
    2: "Winter",
    3: "Spring",
    4: "Spring",
    5: "Spring",
    6: "Summer",
    7: "Summer",
    8: "Summer",
    9: "Autumn",
    10: "Autumn",
    11: "Autumn",
}
SEASON_ORDER = ["Winter", "Spring", "Summer", "Autumn"]
SEASON_COLORS = {
    "Winter": "#0097DC",  # IE business blue (cool)
    "Spring": "#6DC201",  # IE tech green
    "Summer": "#FF630F",  # IE humanities orange
    "Autumn": "#F23333",  # IE law red
}

MONTH_ABBR = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
def setup_style() -> None:
    """Configure IE Science & Technology branded matplotlib defaults."""
    mpl.rcParams.update(
        {
            "font.family": "Montserrat",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlepad": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": "#E5E5E5",
            "grid.linewidth": 0.7,
            "figure.facecolor": "#F8F8F8",
            "axes.facecolor": "#F8F8F8",
            "xtick.bottom": False,
            "ytick.left": False,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "legend.handlelength": 1.5,
            "savefig.bbox": "tight",
            "savefig.dpi": 180,
            "savefig.facecolor": "#F8F8F8",
            "figure.constrained_layout.use": True,
        }
    )


def add_source_note(
    fig: plt.Figure, note: str = "Source: SWOPP3 2024, IEResearchDatalab"
) -> None:
    """Add a small source note at the bottom-left of a figure."""
    fig.text(
        0.01,
        -0.01,
        note,
        ha="left",
        va="top",
        fontsize=7.5,
        color="#666666",
        style="italic",
    )


def _red_line_rule(ax: plt.Axes) -> None:
    """Draw the signature Economist top red rule on the axes title area."""
    x0, _, w, _ = ax.get_position().bounds
    # We draw a thin red rectangle across the top of the axes
    ax.axhline(
        y=ax.get_ylim()[1], color="#000066", linewidth=2.5, clip_on=False, zorder=10
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _outlier_cap(series: pd.Series, iqr_factor: float = 5.0) -> pd.Series:
    """Return a boolean mask of non-outliers using IQR rule."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return (series >= q1 - iqr_factor * iqr) & (series <= q3 + iqr_factor * iqr)


def load_summary_csv(exp_key: str, case_id: str) -> pd.DataFrame | None:
    """Load and annotate one experiment/case summary CSV."""
    folder = EXPERIMENTS[exp_key]["folder"]
    path = OUTPUT_DIR / folder / f"IEUniversity-1-{case_id}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["departure_time_utc", "arrival_time_utc"])
    df["experiment"] = exp_key
    df["case_id"] = case_id

    # Temporal features
    df["month"] = df["departure_time_utc"].dt.month
    df["season"] = df["month"].map(_MONTH_TO_SEASON)

    # Violation flags
    df["wind_viol"] = df["max_wind_mps"] > WIND_LIMIT
    df["wave_viol"] = df["max_hs_m"] > WAVE_LIMIT
    df["any_viol"] = df["wind_viol"] | df["wave_viol"]

    # Remove extreme outliers (FMS occasionally yields 10000+ MWh routes)
    mask = _outlier_cap(df["energy_cons_mwh"])
    if (~mask).any():
        print(f"  [!] Dropping {(~mask).sum()} outliers from {exp_key}/{case_id}")
    return df[mask].copy()


def load_gc_baselines() -> dict[str, float]:
    """Return mean energy for each GC case across both baseline folders."""
    baselines: dict[str, dict[str, list]] = {}
    for folder_key in ("no_penalty", "penalty"):
        folder = EXPERIMENTS[folder_key]["folder"]
        for gc_id in GC_CASES:
            path = OUTPUT_DIR / folder / f"IEUniversity-1-{gc_id}.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            baselines.setdefault(gc_id, []).append(df["energy_cons_mwh"].mean())
    return {k: float(np.mean(v)) for k, v in baselines.items()}


def load_all_data() -> pd.DataFrame:
    """Load all optimised-case summary rows across all experiments."""
    frames = []
    for exp_key in EXPERIMENTS:
        for case_id in OPT_CASES:
            df = load_summary_csv(exp_key, case_id)
            if df is not None:
                frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_tracks(
    exp_key: str, case_id: str, season_filter: str | None = None, n_sample: int = 8
) -> list[pd.DataFrame]:
    """Return sampled per-voyage tracks for one experiment/case pair.

    The sampling operates on the summary CSV first so that the selected track
    files inherit season and departure metadata from the same voyage rows.
    """
    folder = EXPERIMENTS[exp_key]["folder"]
    tracks_dir = OUTPUT_DIR / folder / "tracks"
    if not tracks_dir.exists():
        return []

    # Load the summary to know departure dates by season
    summary = load_summary_csv(exp_key, case_id)
    if summary is None:
        return []

    if season_filter:
        summary = summary[summary["season"] == season_filter]

    # Sample evenly
    sample = summary.sample(
        min(n_sample, len(summary)), replace=False, random_state=42
    ).sort_values("departure_time_utc")

    result = []
    for _, row in sample.iterrows():
        fname = row["details_filename"]
        fpath = tracks_dir / fname
        if fpath.exists():
            trk = pd.read_csv(fpath, parse_dates=["time_utc"])
            trk["experiment"] = exp_key
            trk["case_id"] = case_id
            trk["departure"] = row["departure_time_utc"]
            trk["season"] = row["season"]
            result.append(trk)
    return result


# ===========================================================================
# FIGURE 1 — Energy overview (violin plots)
# ===========================================================================
def fig_energy_overview(
    df: pd.DataFrame, gc: dict[str, float], gc_full: pd.DataFrame
) -> None:
    """Violin plot of energy distributions per case and experiment.

    The GC route is shown as its own violin (position 0) so its
    departure-to-departure variability is visible rather than a flat line.
    """
    setup_style()
    fig, axes = plt.subplots(1, 4, figsize=(14, 5), sharey=False)
    fig.suptitle(
        "Optimised routing cuts energy by 10–55 % versus the great-circle baseline",
        fontsize=13,
        fontweight="bold",
        x=0.02,
        ha="left",
    )

    GC_COLOR = "#878787"
    exp_order = list(EXPERIMENTS.keys())
    # Position 1 = GC, positions 2..5 = experiments
    gc_pos = 1
    exp_positions = np.arange(2, len(exp_order) + 2)
    width = 0.7

    def _draw_violin(ax, data, pos, color, alpha=0.80):
        vp = ax.violinplot(
            data,
            positions=[pos],
            widths=width,
            showmedians=True,
            showextrema=False,
        )
        for pc in vp["bodies"]:
            pc.set_facecolor(color)
            pc.set_edgecolor("none")
            pc.set_alpha(alpha)
        vp["cmedians"].set_color("white")
        vp["cmedians"].set_linewidth(2)
        return vp

    for ax, (case_id, case_meta) in zip(axes, OPT_CASES.items(), strict=False):
        gc_id = case_meta["gc"]
        gc_vals_raw = gc_full.loc[
            gc_full["case_id"] == case_id, "energy_cons_mwh"
        ].dropna()
        gc_vals = gc_vals_raw[_outlier_cap(gc_vals_raw)]
        gc_mean = gc_vals.median() if not gc_vals.empty else gc.get(gc_id, np.nan)

        # GC violin (position 1)
        if not gc_vals.empty:
            _draw_violin(ax, gc_vals.values, gc_pos, GC_COLOR, alpha=0.65)
            ax.text(
                gc_pos,
                gc_vals.quantile(0.05),
                "GC",
                ha="center",
                va="top",
                fontsize=7.5,
                color=GC_COLOR,
                fontweight="bold",
            )

        # Optimised experiment violins
        for i, exp_key in enumerate(exp_order):
            sub = df[(df["experiment"] == exp_key) & (df["case_id"] == case_id)][
                "energy_cons_mwh"
            ]
            if sub.empty:
                continue
            _draw_violin(
                ax, sub.values, exp_positions[i], EXPERIMENTS[exp_key]["color"]
            )

            # % savings vs GC median
            pct = (gc_mean - sub.median()) / gc_mean * 100
            ax.text(
                exp_positions[i],
                sub.quantile(0.05),
                f"−{pct:.0f}%",
                ha="center",
                va="top",
                fontsize=7.5,
                color=EXPERIMENTS[exp_key]["color"],
                fontweight="bold",
            )

        ax.set_title(
            case_meta["label"].replace("\n", " "), fontsize=10, fontweight="bold"
        )
        all_ticks = [gc_pos] + list(exp_positions)
        all_labels = ["GC"] + [
            EXPERIMENTS[k]["short"].replace(" + ", "\n+\n") for k in exp_order
        ]
        ax.set_xticks(all_ticks)
        ax.set_xticklabels(all_labels, fontsize=7.0)
        ax.set_ylabel("Energy (MWh)", fontsize=8)
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.7)
        ax.set_axisbelow(True)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=GC_COLOR, alpha=0.65, label="Great-circle baseline"),
    ] + [
        mpatches.Patch(
            facecolor=EXPERIMENTS[k]["color"], alpha=0.85, label=EXPERIMENTS[k]["label"]
        )
        for k in exp_order
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, -0.04),
        fontsize=8.5,
    )

    add_source_note(fig)
    out = FIGS_DIR / "fig01_energy_overview.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    print(f"  Saved {out.name}")
    plt.close(fig)


# ===========================================================================
# FIGURE 2 — Optimisation gains vs GC baseline
# ===========================================================================
def fig_optimization_gains(df: pd.DataFrame, gc: dict[str, float]) -> None:
    """Plot grouped bar chart of % energy savings vs GC baseline per experiment."""  # noqa: E501
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(
        "Weather-routing optimisation reduces energy consumption relative to the great-circle baseline",  # noqa: E501
        fontsize=12,
        fontweight="bold",
        x=0.02,
        ha="left",
    )

    route_groups = [
        ("atlantic", "Atlantic route (Santander → New York)", ["AO_WPS", "AO_noWPS"]),
        ("pacific", "Pacific route (Tokyo → Los Angeles)", ["PO_WPS", "PO_noWPS"]),
    ]

    bar_w = 0.18
    exp_order = list(EXPERIMENTS.keys())

    for ax, (_route, title, cases) in zip(axes, route_groups, strict=False):
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axhline(0, color="#444", linewidth=0.8)

        x_centers = np.arange(len(cases))
        offsets = np.linspace(
            -(len(exp_order) - 1) / 2 * bar_w,
            (len(exp_order) - 1) / 2 * bar_w,
            len(exp_order),
        )

        for j, exp_key in enumerate(exp_order):
            savings = []
            for case_id in cases:
                gc_id = OPT_CASES[case_id]["gc"]
                gc_mean = gc.get(gc_id, np.nan)
                sub = df[(df["experiment"] == exp_key) & (df["case_id"] == case_id)][
                    "energy_cons_mwh"
                ]
                if sub.empty or np.isnan(gc_mean):
                    savings.append(np.nan)
                else:
                    savings.append((gc_mean - sub.median()) / gc_mean * 100)

            xs = x_centers + offsets[j]
            bars = ax.bar(
                xs,
                savings,
                width=bar_w * 0.92,
                color=EXPERIMENTS[exp_key]["color"],
                alpha=0.88,
                label=EXPERIMENTS[exp_key]["label"],
                zorder=3,
            )
            for bar, val in zip(bars, savings, strict=False):
                if not np.isnan(val):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        val + 0.4 if val >= 0 else val - 0.4,
                        f"{val:.1f}%",
                        ha="center",
                        va="bottom" if val >= 0 else "top",
                        fontsize=7,
                        color=EXPERIMENTS[exp_key]["color"],
                        fontweight="bold",
                    )

        ax.set_xticks(x_centers)
        ax.set_xticklabels([OPT_CASES[c]["label_short"] for c in cases], fontsize=9)
        ax.set_ylabel("Energy saving vs GC baseline (%)")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.7)
        ax.set_axisbelow(True)

    # Shared legend
    handles = [
        mpatches.Patch(
            facecolor=EXPERIMENTS[k]["color"], alpha=0.85, label=EXPERIMENTS[k]["label"]
        )
        for k in exp_order
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.05),
        fontsize=8.5,
    )

    # Equalise y-axis across both route panels
    ymax = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(0, ymax)

    add_source_note(fig)
    out = FIGS_DIR / "fig02_optimization_gains.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    print(f"  Saved {out.name}")
    plt.close(fig)


# ===========================================================================
# FIGURE 3 — Penalty trade-off (safety vs efficiency)
# ===========================================================================
def fig_penalty_tradeoff(df: pd.DataFrame) -> None:
    """Side-by-side bars: violation rate reduction vs energy overhead from penalty."""
    setup_style()
    # Keep only no_penalty vs penalty pairs  (drop FMS variants for clarity)
    sub = df[df["experiment"].isin(["no_penalty", "penalty"])].copy()

    metrics = pd.DataFrame()
    for case_id in OPT_CASES:
        for exp_key in ["no_penalty", "penalty"]:
            piece = sub[(sub["experiment"] == exp_key) & (sub["case_id"] == case_id)]
            if piece.empty:
                continue
            wind_rate = piece["wind_viol"].mean() * 100
            wave_rate = piece["wave_viol"].mean() * 100
            any_rate = piece["any_viol"].mean() * 100
            mean_e = piece["energy_cons_mwh"].mean()
            metrics = pd.concat(
                [
                    metrics,
                    pd.DataFrame(
                        [
                            {
                                "case_id": case_id,
                                "experiment": exp_key,
                                "wind_violation_pct": wind_rate,
                                "wave_violation_pct": wave_rate,
                                "any_violation_pct": any_rate,
                                "mean_energy": mean_e,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.suptitle(
        "Weather penalties cut violation rates — and keep energy costs low",
        fontsize=12,
        fontweight="bold",
        x=0.02,
        ha="left",
    )

    cases_order = list(OPT_CASES.keys())
    x = np.arange(len(cases_order))
    bw = 0.35

    # Panel A — wind violation rate
    ax = axes[0]
    ax.set_title("Wind violations\n(% of departures above 20 m/s)", fontsize=9.5)
    for i, exp_key in enumerate(["no_penalty", "penalty"]):
        vals = [
            metrics.loc[
                (metrics.case_id == c) & (metrics.experiment == exp_key),
                "wind_violation_pct",
            ].values
            for c in cases_order
        ]
        vals = [v[0] if len(v) > 0 else np.nan for v in vals]
        offset = -bw / 2 if i == 0 else bw / 2
        bars = ax.bar(
            x + offset,
            vals,
            width=bw * 0.92,
            color=EXPERIMENTS[exp_key]["color"],
            alpha=0.88,
            label=EXPERIMENTS[exp_key]["label"],
            zorder=3,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([OPT_CASES[c]["label_short"] for c in cases_order], fontsize=8)
    ax.set_ylabel("Departures with wind violation (%)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))

    # Panel B — wave violation rate
    ax = axes[1]
    ax.set_title("Wave violations\n(% of departures above 7 m Hs)", fontsize=9.5)
    for i, exp_key in enumerate(["no_penalty", "penalty"]):
        vals = [
            metrics.loc[
                (metrics.case_id == c) & (metrics.experiment == exp_key),
                "wave_violation_pct",
            ].values
            for c in cases_order
        ]
        vals = [v[0] if len(v) > 0 else np.nan for v in vals]
        offset = -bw / 2 if i == 0 else bw / 2
        ax.bar(
            x + offset,
            vals,
            width=bw * 0.92,
            color=EXPERIMENTS[exp_key]["color"],
            alpha=0.88,
            zorder=3,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([OPT_CASES[c]["label_short"] for c in cases_order], fontsize=8)
    ax.set_ylabel("Departures with wave violation (%)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))

    # Panel C — mean energy
    ax = axes[2]
    ax.set_title("Mean energy consumption\n(MWh per voyage)", fontsize=9.5)
    for i, exp_key in enumerate(["no_penalty", "penalty"]):
        vals = [
            metrics.loc[
                (metrics.case_id == c) & (metrics.experiment == exp_key), "mean_energy"
            ].values
            for c in cases_order
        ]
        vals = [v[0] if len(v) > 0 else np.nan for v in vals]
        offset = -bw / 2 if i == 0 else bw / 2
        bars = ax.bar(
            x + offset,
            vals,
            width=bw * 0.92,
            color=EXPERIMENTS[exp_key]["color"],
            alpha=0.88,
            zorder=3,
        )
        for bar, val in zip(bars, vals, strict=False):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + 1,
                    f"{val:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    color=EXPERIMENTS[exp_key]["color"],
                    fontweight="bold",
                )
    ax.set_xticks(x)
    ax.set_xticklabels([OPT_CASES[c]["label_short"] for c in cases_order], fontsize=8)
    ax.set_ylabel("Mean energy consumption (MWh)")

    # Share y-axis between the two violation-rate panels
    ymax_viol = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
    axes[0].set_ylim(0, ymax_viol)
    axes[1].set_ylim(0, ymax_viol)

    for ax in axes:
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.7)
        ax.set_axisbelow(True)

    handles = [
        mpatches.Patch(
            facecolor=EXPERIMENTS["no_penalty"]["color"],
            alpha=0.85,
            label="CMA-ES (no penalty)",
        ),
        mpatches.Patch(
            facecolor=EXPERIMENTS["penalty"]["color"],
            alpha=0.85,
            label="CMA-ES + Penalty",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.04),
        fontsize=9,
    )

    add_source_note(fig)
    out = FIGS_DIR / "fig03_penalty_tradeoff.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    print(f"  Saved {out.name}")
    plt.close(fig)


# ===========================================================================
# FIGURE 4 — Seasonality (monthly mean energy)
# ===========================================================================
def _fig_seasonality_panel(
    df: pd.DataFrame,
    gc_full: pd.DataFrame,
    exp_keys: list[str],
    title: str,
    out_stem: str,
) -> None:
    """Shared implementation for fig04a and fig04b.

    Draws daily energy (line per experiment) vs day-of-year, one panel per
    SWOPP3 optimised case, for the subset of experiments given by *exp_keys*.
    GC is always included as a grey dashed reference line.
    """
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(title, fontsize=12, fontweight="bold", x=0.02, ha="left")

    cases_order = [
        ("AO_WPS", "Atlantic — with WPS (Santander → New York)"),
        ("AO_noWPS", "Atlantic — without WPS (Santander → New York)"),
        ("PO_WPS", "Pacific — with WPS (Tokyo → Los Angeles)"),
        ("PO_noWPS", "Pacific — without WPS (Tokyo → Los Angeles)"),
    ]

    _MONTH_STARTS = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    _SEASON_SPANS = [
        (0.5, 59.5, "Winter"),
        (59.5, 151.5, "Spring"),
        (151.5, 243.5, "Summer"),
        (243.5, 334.5, "Autumn"),
        (334.5, 365.5, "Winter"),
    ]

    for ax, (case_id, panel_title) in zip(axes.flat, cases_order, strict=False):
        ax.set_title(panel_title, fontsize=10, fontweight="bold")

        for exp_key in exp_keys:
            exp_meta = EXPERIMENTS[exp_key]
            piece = df[
                (df["experiment"] == exp_key) & (df["case_id"] == case_id)
            ].copy()
            if piece.empty:
                continue

            piece["doy"] = piece["departure_time_utc"].dt.dayofyear
            piece = piece.sort_values("doy")

            ax.plot(
                piece["doy"],
                piece["energy_cons_mwh"],
                color=exp_meta["color"],
                linewidth=1.4,
                alpha=0.85,
                label=exp_meta["label"],
                zorder=4,
            )

        # GC reference line
        gc_piece = gc_full[gc_full["case_id"] == case_id].copy()
        gc_piece = gc_piece[_outlier_cap(gc_piece["energy_cons_mwh"])]
        if not gc_piece.empty:
            gc_piece["doy"] = gc_piece["departure_time_utc"].dt.dayofyear
            gc_piece = gc_piece.sort_values("doy")
            ax.plot(
                gc_piece["doy"],
                gc_piece["energy_cons_mwh"],
                color="#878787",
                linewidth=1.4,
                linestyle="--",
                alpha=0.75,
                zorder=3,
            )

        # Season background shading
        for start, end, s in _SEASON_SPANS:
            ax.axvspan(start, end, alpha=0.06, color=SEASON_COLORS[s], zorder=1)

        ax.set_xlim(1, 365)
        ax.set_xticks(_MONTH_STARTS)
        ax.set_xticklabels(MONTH_ABBR, fontsize=8.5)
        ax.set_xlabel("Departure date")
        ax.set_ylabel("Energy consumption (MWh)")
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.7)
        ax.set_axisbelow(True)

    # Legend
    exp_handles = [
        mlines.Line2D(
            [],
            [],
            color=EXPERIMENTS[k]["color"],
            linewidth=2.0,
            alpha=0.85,
            label=EXPERIMENTS[k]["label"],
        )
        for k in exp_keys
    ]
    exp_handles.append(
        mlines.Line2D(
            [],
            [],
            color="#878787",
            linewidth=2.0,
            linestyle="--",
            alpha=0.75,
            label="Great-circle baseline",
        )
    )
    season_handles = [
        mpatches.Patch(facecolor=SEASON_COLORS[s], alpha=0.5, label=s)
        for s in SEASON_ORDER
    ]
    fig.legend(
        handles=exp_handles + season_handles,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.08),
        fontsize=8.5,
    )

    # Equalise y-axis across all four panels
    all_ylims = [ax.get_ylim() for ax in axes.flat]
    ymin_all = min(y[0] for y in all_ylims)
    ymax_all = max(y[1] for y in all_ylims)
    for ax in axes.flat:
        ax.set_ylim(ymin_all, ymax_all)

    add_source_note(fig)
    out = FIGS_DIR / f"{out_stem}.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    print(f"  Saved {out.name}")
    plt.close(fig)


def fig_seasonality_a(df: pd.DataFrame, gc_full: pd.DataFrame) -> None:
    """fig04a — seasonal energy lines for non-penalized experiments (CMA-ES and FMS)."""
    _fig_seasonality_panel(
        df,
        gc_full,
        exp_keys=["no_penalty", "no_penalty_fms"],
        title="Seasonal energy — non-penalized routing (GC · CMA-ES · CMA-ES + FMS)",
        out_stem="fig04a_seasonality_no_penalty",
    )


def fig_seasonality_b(df: pd.DataFrame, gc_full: pd.DataFrame) -> None:
    """fig04b — seasonal energy lines for penalized experiments.

    Shows CMA-ES + Penalty and CMA-ES + Penalty + FMS, plus GC baseline.
    """
    _fig_seasonality_panel(
        df,
        gc_full,
        exp_keys=["penalty", "penalty_fms"],
        title=(
            "Seasonal energy — penalized routing"
            " (GC · CMA-ES + Penalty · CMA-ES + Penalty + FMS)"
        ),
        out_stem="fig04b_seasonality_penalty",
    )


def _fig_relative_gain_panel(
    df: pd.DataFrame,
    gc_full: pd.DataFrame,
    exp_keys: list[str],
    title: str,
    out_stem: str,
) -> None:
    """Shared implementation for fig13a and fig13b.

    For each departure, computes the relative energy saving vs the matched
    GC departure: ``(gc_energy − exp_energy) / gc_energy × 100``.  The
    result is plotted as a connected daily line — two lines per panel, one
    per experiment.  A horizontal zero reference marks the break-even point.
    """
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(title, fontsize=12, fontweight="bold", x=0.02, ha="left")

    cases_order = [
        ("AO_WPS", "Atlantic — with WPS (Santander → New York)"),
        ("AO_noWPS", "Atlantic — without WPS (Santander → New York)"),
        ("PO_WPS", "Pacific — with WPS (Tokyo → Los Angeles)"),
        ("PO_noWPS", "Pacific — without WPS (Tokyo → Los Angeles)"),
    ]

    _MONTH_STARTS = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    _SEASON_SPANS = [
        (0.5, 59.5, "Winter"),
        (59.5, 151.5, "Spring"),
        (151.5, 243.5, "Summer"),
        (243.5, 334.5, "Autumn"),
        (334.5, 365.5, "Winter"),
    ]

    for ax, (case_id, panel_title) in zip(axes.flat, cases_order, strict=False):
        ax.set_title(panel_title, fontsize=10, fontweight="bold")

        gc_piece = gc_full[gc_full["case_id"] == case_id][
            ["departure_time_utc", "energy_cons_mwh"]
        ].rename(columns={"energy_cons_mwh": "gc_energy"})

        for exp_key in exp_keys:
            exp_meta = EXPERIMENTS[exp_key]
            piece = df[(df["experiment"] == exp_key) & (df["case_id"] == case_id)][
                ["departure_time_utc", "energy_cons_mwh"]
            ].copy()
            if piece.empty or gc_piece.empty:
                continue

            merged = piece.merge(gc_piece, on="departure_time_utc", how="inner")
            merged = merged[merged["gc_energy"] > 0]  # avoid division by zero
            merged["saving_pct"] = (
                (merged["gc_energy"] - merged["energy_cons_mwh"])
                / merged["gc_energy"]
                * 100
            )
            merged["doy"] = merged["departure_time_utc"].dt.dayofyear
            merged = merged.sort_values("doy")

            ax.plot(
                merged["doy"],
                merged["saving_pct"],
                color=exp_meta["color"],
                linewidth=1.4,
                alpha=0.85,
                label=exp_meta["label"],
                zorder=4,
            )

        # Zero break-even reference
        ax.axhline(0, color="#878787", linewidth=1.0, linestyle="--", zorder=3)

        # Season background shading
        for start, end, s in _SEASON_SPANS:
            ax.axvspan(start, end, alpha=0.06, color=SEASON_COLORS[s], zorder=1)

        ax.set_xlim(1, 365)
        ax.set_xticks(_MONTH_STARTS)
        ax.set_xticklabels(MONTH_ABBR, fontsize=8.5)
        ax.set_xlabel("Departure date")
        ax.set_ylabel("Energy saving vs GC (%)")
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.7)
        ax.set_axisbelow(True)

    # Legend
    exp_handles = [
        mlines.Line2D(
            [],
            [],
            color=EXPERIMENTS[k]["color"],
            linewidth=2.0,
            alpha=0.85,
            label=EXPERIMENTS[k]["label"],
        )
        for k in exp_keys
    ]
    exp_handles.append(
        mlines.Line2D(
            [],
            [],
            color="#878787",
            linewidth=1.0,
            linestyle="--",
            alpha=0.75,
            label="GC baseline (0 %)",
        )
    )
    season_handles = [
        mpatches.Patch(facecolor=SEASON_COLORS[s], alpha=0.5, label=s)
        for s in SEASON_ORDER
    ]
    fig.legend(
        handles=exp_handles + season_handles,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.08),
        fontsize=8.5,
    )

    # Equalise y-axis across all four panels
    all_ylims = [ax.get_ylim() for ax in axes.flat]
    ymin_all = min(y[0] for y in all_ylims)
    ymax_all = max(y[1] for y in all_ylims)
    for ax in axes.flat:
        ax.set_ylim(ymin_all, ymax_all)

    add_source_note(fig)
    out = FIGS_DIR / f"{out_stem}.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    print(f"  Saved {out.name}")
    plt.close(fig)


def fig_relative_gain_a(df: pd.DataFrame, gc_full: pd.DataFrame) -> None:
    """fig13a — relative energy gain vs GC for non-penalized experiments."""
    _fig_relative_gain_panel(
        df,
        gc_full,
        exp_keys=["no_penalty", "no_penalty_fms"],
        title=(
            "Relative energy saving vs GC"
            " — non-penalized routing (CMA-ES · CMA-ES + FMS)"
        ),
        out_stem="fig13a_relative_gain_no_penalty",
    )


def fig_relative_gain_b(df: pd.DataFrame, gc_full: pd.DataFrame) -> None:
    """fig13b — relative energy gain vs GC for penalized experiments."""
    _fig_relative_gain_panel(
        df,
        gc_full,
        exp_keys=["penalty", "penalty_fms"],
        title=(
            "Relative energy saving vs GC"
            " — penalized routing (CMA-ES + Penalty · CMA-ES + Penalty + FMS)"
        ),
        out_stem="fig13b_relative_gain_penalty",
    )


# ===========================================================================
# FIGURE 5 — WPS impact
# ===========================================================================
def fig_wps_impact(df: pd.DataFrame) -> None:
    """Bar chart of absolute and relative WPS energy savings per experiment."""
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(
        "Wind-propulsion systems (WPS) cut energy use by 30 % on the Atlantic, 55 % on the Pacific",  # noqa: E501
        fontsize=12,
        fontweight="bold",
        x=0.02,
        ha="left",
    )

    route_groups = [
        ("atlantic", "Trans-Atlantic WPS savings", "AO_WPS", "AO_noWPS"),
        ("pacific", "Trans-Pacific WPS savings", "PO_WPS", "PO_noWPS"),
    ]

    for ax, (_route, title, wps_case, nowps_case) in zip(
        axes, route_groups, strict=False
    ):
        ax.set_title(title, fontsize=10.5, fontweight="bold")

        exp_order = list(EXPERIMENTS.keys())
        x = np.arange(len(exp_order))
        bar_w = 0.55

        abs_savings = []
        rel_savings = []
        for exp_key in exp_order:
            wps_e = df[(df["experiment"] == exp_key) & (df["case_id"] == wps_case)][
                "energy_cons_mwh"
            ]
            nowps_e = df[(df["experiment"] == exp_key) & (df["case_id"] == nowps_case)][
                "energy_cons_mwh"
            ]
            if wps_e.empty or nowps_e.empty:
                abs_savings.append(np.nan)
                rel_savings.append(np.nan)
                continue
            savings = nowps_e.mean() - wps_e.mean()
            rel = savings / nowps_e.mean() * 100
            abs_savings.append(savings)
            rel_savings.append(rel)

        colors = [EXPERIMENTS[k]["color"] for k in exp_order]
        bars = ax.bar(x, abs_savings, width=bar_w, color=colors, alpha=0.88, zorder=3)

        # Annotate with % saving
        for bar, val, rel in zip(bars, abs_savings, rel_savings, strict=False):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + 0.5,
                    f"{rel:.0f}% less",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                    color=bar.get_facecolor(),
                )

        ax.set_xticks(x)
        ax.set_xticklabels([EXPERIMENTS[k]["short"] for k in exp_order], fontsize=8.5)
        ax.set_ylabel("WPS energy saving (MWh)")
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.7)
        ax.set_axisbelow(True)

    # Equalise y-axis so Atlantic vs Pacific savings are directly comparable
    ymax = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(0, ymax)

    add_source_note(fig)
    out = FIGS_DIR / "fig05_wps_impact.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    print(f"  Saved {out.name}")
    plt.close(fig)


# ===========================================================================
# FIGURE 6 — FMS improvement scatter
# ===========================================================================
def fig_fms_improvement(df: pd.DataFrame) -> None:
    """Scatter plot: CMA-ES energy vs CMA-ES+FMS energy (each point = one departure)."""
    setup_style()

    # Pairs: (base experiment, fms experiment)
    pairs = [
        ("no_penalty", "no_penalty_fms", "Without penalty"),
        ("penalty", "penalty_fms", "With penalty"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(
        "FMS refinement consistently reduces energy — gains are largest for low-energy routes",  # noqa: E501
        fontsize=12,
        fontweight="bold",
        x=0.02,
        ha="left",
    )

    # Pre-compute global axis limits so both panels share the same scale
    _fig6_all: list[float] = []
    for _b, _f, _ in pairs:
        for _c in OPT_CASES:
            _fig6_all += (
                df[(df["experiment"] == _b) & (df["case_id"] == _c)]["energy_cons_mwh"]
                .dropna()
                .tolist()
            )
            _fig6_all += (
                df[(df["experiment"] == _f) & (df["case_id"] == _c)]["energy_cons_mwh"]
                .dropna()
                .tolist()
            )
    glim_lo = np.nanmin(_fig6_all) * 0.9
    glim_hi = np.nanmax(_fig6_all) * 1.05

    for ax, (base_exp, fms_exp, panel_title) in zip(axes, pairs, strict=False):
        ax.set_title(panel_title, fontsize=10.5, fontweight="bold")

        for case_id, case_meta in OPT_CASES.items():
            base = df[
                (df["experiment"] == base_exp) & (df["case_id"] == case_id)
            ].set_index("departure_time_utc")
            fms = df[
                (df["experiment"] == fms_exp) & (df["case_id"] == case_id)
            ].set_index("departure_time_utc")
            joined = base[["energy_cons_mwh"]].join(
                fms[["energy_cons_mwh"]], lsuffix="_base", rsuffix="_fms", how="inner"
            )
            if joined.empty:
                continue

            ax.scatter(
                joined["energy_cons_mwh_base"],
                joined["energy_cons_mwh_fms"],
                color=case_meta["color"],
                alpha=0.35,
                s=12,
                label=case_meta["label_short"],
                zorder=3,
            )

        lim_lo, lim_hi = glim_lo, glim_hi

        # Diagonal x=y (no improvement)
        ax.plot(
            [lim_lo, lim_hi],
            [lim_lo, lim_hi],
            color="#444",
            linewidth=1,
            linestyle="--",
            alpha=0.6,
            label="No improvement",
            zorder=5,
        )

        # 5% improvement line
        ax.plot(
            [lim_lo, lim_hi],
            [lim_lo * 0.95, lim_hi * 0.95],
            color="#888",
            linewidth=0.8,
            linestyle=":",
            alpha=0.7,
            label="5% improvement",
            zorder=4,
        )

        ax.set_xlim(lim_lo, lim_hi)
        ax.set_ylim(lim_lo, lim_hi)
        ax.set_xlabel("CMA-ES energy (MWh)")
        ax.set_ylabel("CMA-ES + FMS energy (MWh)")
        ax.set_aspect("equal", adjustable="box")
        ax.legend(fontsize=8, loc="upper left", markerscale=1.5)

    add_source_note(fig)
    out = FIGS_DIR / "fig06_fms_improvement.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    print(f"  Saved {out.name}")
    plt.close(fig)


# ===========================================================================
# GC track loader (used by fig07)
# ===========================================================================
def load_gc_tracks(
    gc_case: str, season_filter: str | None = None, n_sample: int = 5
) -> list[pd.DataFrame]:
    """Load sample GC track DataFrames from the penalty output folder."""
    folder = EXPERIMENTS["penalty"]["folder"]
    tracks_dir = OUTPUT_DIR / folder / "tracks"
    summary_path = OUTPUT_DIR / folder / f"IEUniversity-1-{gc_case}.csv"
    if not summary_path.exists():
        return []
    gc_df = pd.read_csv(
        summary_path, parse_dates=["departure_time_utc", "arrival_time_utc"]
    )
    gc_df["season"] = gc_df["departure_time_utc"].dt.month.map(_MONTH_TO_SEASON)
    if season_filter:
        gc_df = gc_df[gc_df["season"] == season_filter]
    sample = gc_df.sample(
        min(n_sample, len(gc_df)), replace=False, random_state=7
    ).sort_values("departure_time_utc")
    result = []
    for _, row in sample.iterrows():
        fpath = tracks_dir / row["details_filename"]
        if fpath.exists():
            trk = pd.read_csv(fpath, parse_dates=["time_utc"])
            result.append(trk)
    return result


# ===========================================================================
# FIGURE 7 — Route maps
# ===========================================================================
def fig_route_maps() -> None:
    """Geographic maps showing representative routes for Atlantic and Pacific."""
    setup_style()

    # Cartopy/constrained_layout are incompatible — disable for this figure
    with mpl.rc_context({"figure.constrained_layout.use": False}):
        fig = plt.figure(figsize=(14, 6), facecolor="#FAFAF7")
        fig.suptitle(
            "Optimised routes diverge from great circles to exploit prevailing winds and avoid storms",  # noqa: E501
            fontsize=12,
            fontweight="bold",
            x=0.02,
            ha="left",
        )

        # Atlantic map
        ax_atl = fig.add_subplot(
            1,
            2,
            1,
            projection=ccrs.PlateCarree(central_longitude=-40),
        )
        # Pacific map (centred at 180° to avoid antimeridian split)
        ax_pac = fig.add_subplot(
            1,
            2,
            2,
            projection=ccrs.PlateCarree(central_longitude=180),
        )

        route_configs = [
            {
                "ax": ax_atl,
                "title": "Trans-Atlantic (Santander → New York)",
                "extent": [-80, 15, 25, 65],
                "cases": ["AO_WPS", "AO_noWPS"],
                "gc_case": "AGC_WPS",
                "seasons": ["Winter", "Summer"],
            },
            {
                "ax": ax_pac,
                "title": "Trans-Pacific (Tokyo → Los Angeles)",
                "extent": [115, 250, 20, 65],
                "cases": ["PO_WPS", "PO_noWPS"],
                "gc_case": "PGC_WPS",
                "seasons": ["Winter", "Summer"],
            },
        ]

        for cfg in route_configs:
            ax = cfg["ax"]
            ax.set_extent(cfg["extent"], crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.LAND, facecolor="#D9D0C3", zorder=1)
            ax.add_feature(cfeature.OCEAN, facecolor="#EFF5FF", zorder=0)
            ax.add_feature(
                cfeature.COASTLINE, linewidth=0.5, edgecolor="#7D7D7D", zorder=2
            )
            ax.add_feature(
                cfeature.BORDERS, linewidth=0.3, edgecolor="#BBBBBB", zorder=2
            )
            gl = ax.gridlines(
                draw_labels=True,
                linewidth=0.4,
                color="#CCCCCC",
                x_inline=False,
                y_inline=False,
            )
            gl.xlabel_style = {"size": 7}
            gl.ylabel_style = {"size": 7}
            ax.set_title(cfg["title"], fontsize=10, fontweight="bold", pad=6)

            for exp_key in ["no_penalty", "penalty"]:
                exp_meta = EXPERIMENTS[exp_key]
                for case_id in cfg["cases"]:
                    for season in cfg["seasons"]:
                        tracks = load_tracks(
                            exp_key, case_id, season_filter=season, n_sample=4
                        )
                        alpha = 0.55 if season == "Winter" else 0.35
                        for trk in tracks:
                            ax.plot(
                                trk["lon_deg"].values,
                                trk["lat_deg"].values,
                                transform=ccrs.PlateCarree(),
                                color=exp_meta["color"],
                                linewidth=0.85,
                                alpha=alpha,
                                zorder=3,
                            )

            # Great-circle reference routes — dashed dark grey
            gc_case = cfg.get("gc_case")
            if gc_case:
                for season in cfg["seasons"]:
                    alpha_gc = 0.80 if season == "Winter" else 0.45
                    for trk in load_gc_tracks(
                        gc_case, season_filter=season, n_sample=4
                    ):
                        ax.plot(
                            trk["lon_deg"].values,
                            trk["lat_deg"].values,
                            transform=ccrs.PlateCarree(),
                            color="#111111",
                            linewidth=1.5,
                            linestyle="--",
                            alpha=alpha_gc,
                            zorder=5,
                        )

        # Legend
        legend_elements = [
            mlines.Line2D(
                [],
                [],
                color="#111111",
                linewidth=1.5,
                linestyle="--",
                label="Great-circle route",
            ),
            mlines.Line2D(
                [],
                [],
                color=EXPERIMENTS["no_penalty"]["color"],
                linewidth=2,
                label="CMA-ES (no penalty)",
            ),
            mlines.Line2D(
                [],
                [],
                color=EXPERIMENTS["penalty"]["color"],
                linewidth=2,
                label="CMA-ES + Penalty",
            ),
            mlines.Line2D(
                [],
                [],
                color="#555",
                linewidth=2,
                alpha=0.75,
                label="Winter departures (bold)",
            ),
            mlines.Line2D(
                [],
                [],
                color="#555",
                linewidth=2,
                alpha=0.40,
                label="Summer departures (faint)",
            ),
        ]
        fig.legend(
            handles=legend_elements,
            loc="lower center",
            ncol=5,
            bbox_to_anchor=(0.5, -0.01),
            fontsize=8.5,
        )

        add_source_note(fig)
        fig.tight_layout(rect=[0, 0.05, 1, 0.93])
        out = FIGS_DIR / "fig07_route_maps.pdf"
        fig.savefig(out, bbox_inches="tight")
        fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
        print(f"  Saved {out.name}")
        plt.close(fig)


# ===========================================================================
# FIGURE 8 — Risk calendar (heatmap of violation rate)
# ===========================================================================
def fig_risk_calendar(df: pd.DataFrame) -> None:
    """Heatmap: departure month × case × experiment — violation rate."""
    setup_style()

    cases_order = list(OPT_CASES.keys())
    months = np.arange(1, 13)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13, 7),
        gridspec_kw={"height_ratios": [1, 1]},
    )
    fig.suptitle(
        "Weather-penalty optimisation substantially reduces high-risk departures, except in Pacific winter",  # noqa: E501
        fontsize=12,
        fontweight="bold",
        x=0.02,
        ha="left",
    )

    viol_titles = {
        "wind_viol": "Wind violations (> 20 m/s)",
        "wave_viol": "Wave violations (> 7 m Hs)",
    }
    viol_cmaps = {
        "wind_viol": "Reds",
        "wave_viol": "Blues",
    }

    for row_idx, viol_col in enumerate(["wind_viol", "wave_viol"]):
        for col_idx, exp_key in enumerate(["no_penalty", "penalty"]):
            ax = axes[row_idx][col_idx]
            exp_label = EXPERIMENTS[exp_key]["label"]
            ax.set_title(
                f"{viol_titles[viol_col]}\n{exp_label}",
                fontsize=9.5,
                fontweight="bold",
            )

            # Build heatmap matrix: rows=cases, cols=months
            matrix = np.full((len(cases_order), len(months)), np.nan)
            for i, case_id in enumerate(cases_order):
                piece = df[(df["experiment"] == exp_key) & (df["case_id"] == case_id)]
                if piece.empty:
                    continue
                monthly = piece.groupby("month")[viol_col].mean() * 100
                for j, m in enumerate(months):
                    matrix[i, j] = monthly.get(m, np.nan)

            cmap = plt.get_cmap(viol_cmaps[viol_col])
            im = ax.imshow(
                matrix,
                cmap=cmap,
                aspect="auto",
                vmin=0,
                vmax=50,
                origin="upper",
            )

            ax.set_xticks(np.arange(12))
            ax.set_xticklabels(MONTH_ABBR, fontsize=8)
            ax.set_yticks(np.arange(len(cases_order)))
            ax.set_yticklabels(
                [OPT_CASES[c]["label_short"] for c in cases_order],
                fontsize=8.5,
            )
            ax.tick_params(left=True, bottom=True)
            ax.grid(False)

            # Annotate cells
            for i in range(len(cases_order)):
                for j in range(12):
                    val = matrix[i, j]
                    if not np.isnan(val):
                        text_color = "white" if val > 25 else "#333333"
                        ax.text(
                            j,
                            i,
                            f"{val:.0f}%",
                            ha="center",
                            va="center",
                            fontsize=7.5,
                            color=text_color,
                            fontweight="bold",
                        )

            cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
            cb.set_label("Violation rate (%)", fontsize=8)
            cb.ax.tick_params(labelsize=7.5)

    add_source_note(fig)
    out = FIGS_DIR / "fig08_risk_calendar.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    print(f"  Saved {out.name}")
    plt.close(fig)


# ===========================================================================
# SUMMARY TABLE
# ===========================================================================
def generate_summary_table(df: pd.DataFrame, gc: dict[str, float]) -> pd.DataFrame:
    """Generate and save a summary statistics table."""
    rows = []
    for exp_key in EXPERIMENTS:
        for case_id in OPT_CASES:
            piece = df[(df["experiment"] == exp_key) & (df["case_id"] == case_id)]
            if piece.empty:
                continue
            gc_id = OPT_CASES[case_id]["gc"]
            gc_mean = gc.get(gc_id, np.nan)
            mean_e = piece["energy_cons_mwh"].mean()
            rows.append(
                {
                    "Experiment": EXPERIMENTS[exp_key]["label"],
                    "Case": OPT_CASES[case_id]["label"].replace("\n", " "),
                    "N departures": len(piece),
                    "Mean energy (MWh)": round(mean_e, 1),
                    "Median energy (MWh)": round(piece["energy_cons_mwh"].median(), 1),
                    "Std energy (MWh)": round(piece["energy_cons_mwh"].std(), 1),
                    "GC baseline (MWh)": round(gc_mean, 1),
                    "Saving vs GC (%)": round((gc_mean - mean_e) / gc_mean * 100, 1),
                    "Wind violation (%)": round(piece["wind_viol"].mean() * 100, 1),
                    "Wave violation (%)": round(piece["wave_viol"].mean() * 100, 1),
                    "Mean distance (nm)": round(piece["sailed_distance_nm"].mean(), 0),
                }
            )
    summary = pd.DataFrame(rows)
    out = FIGS_DIR / "table01_summary.csv"
    summary.to_csv(out, index=False)
    print(f"  Saved {out.name}")
    return summary


# ===========================================================================
# BONUS: FMS delta plot — per-voyage improvement
# ===========================================================================
def fig_fms_delta_byseason(df: pd.DataFrame) -> None:
    """Bar chart: median FMS improvement (%) by season and by case."""
    setup_style()

    pairs = [
        ("no_penalty", "no_penalty_fms"),
        ("penalty", "penalty_fms"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "FMS refinement is most effective in winter and for no-penalty runs",
        fontsize=12,
        fontweight="bold",
        x=0.02,
        ha="left",
    )

    bar_w = 0.18
    cases_order = list(OPT_CASES.keys())
    bar_positions = np.arange(len(SEASON_ORDER))

    for ax, (base_exp, fms_exp) in zip(axes, pairs, strict=False):
        ax.set_title(
            f"{EXPERIMENTS[base_exp]['label']} vs {EXPERIMENTS[fms_exp]['label']}",
            fontsize=10.5,
            fontweight="bold",
        )
        offsets = np.linspace(
            -(len(cases_order) - 1) / 2 * bar_w,
            (len(cases_order) - 1) / 2 * bar_w,
            len(cases_order),
        )

        for j, case_id in enumerate(cases_order):
            base = df[
                (df["experiment"] == base_exp) & (df["case_id"] == case_id)
            ].set_index("departure_time_utc")
            fms_d = df[
                (df["experiment"] == fms_exp) & (df["case_id"] == case_id)
            ].set_index("departure_time_utc")
            joined = base[["energy_cons_mwh", "season"]].join(
                fms_d[["energy_cons_mwh"]],
                lsuffix="_base",
                rsuffix="_fms",
                how="inner",
            )
            if joined.empty:
                continue
            joined["delta_pct"] = (
                (joined["energy_cons_mwh_base"] - joined["energy_cons_mwh_fms"])
                / joined["energy_cons_mwh_base"]
                * 100
            )

            medians = [
                joined.loc[joined["season"] == s, "delta_pct"].median()
                if s in joined["season"].values
                else np.nan
                for s in SEASON_ORDER
            ]

            xs = bar_positions + offsets[j]
            bars = ax.bar(
                xs,
                medians,
                width=bar_w * 0.9,
                color=OPT_CASES[case_id]["color"],
                alpha=0.85,
                label=OPT_CASES[case_id]["label_short"],
                zorder=3,
            )
            for bar, val in zip(bars, medians, strict=False):
                if not np.isnan(val):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        val + 0.2 if val >= 0 else val - 0.5,
                        f"{val:.1f}%",
                        ha="center",
                        va="bottom" if val >= 0 else "top",
                        fontsize=6.5,
                        color=OPT_CASES[case_id]["color"],
                        fontweight="bold",
                    )

        ax.axhline(0, color="#444", linewidth=0.8)
        ax.set_xticks(bar_positions)
        ax.set_xticklabels(SEASON_ORDER, fontsize=9)
        ax.set_ylabel("Median energy reduction from FMS (%)")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.7)
        ax.set_axisbelow(True)

    handles = [
        mpatches.Patch(
            facecolor=OPT_CASES[c]["color"],
            alpha=0.85,
            label=OPT_CASES[c]["label_short"],
        )
        for c in cases_order
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.04),
        fontsize=9,
    )

    # Equalise y-axis across both penalty-comparison panels
    ymin = min(ax.get_ylim()[0] for ax in axes)
    ymax = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(ymin, ymax)

    add_source_note(fig)
    out = FIGS_DIR / "fig09_fms_seasonal_delta.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    print(f"  Saved {out.name}")
    plt.close(fig)


# ===========================================================================
# GC per-departure data loader
# ===========================================================================
# Maps each GC case to the corresponding optimised case
_GC_TO_OPT = {
    "AGC_WPS": "AO_WPS",
    "AGC_noWPS": "AO_noWPS",
    "PGC_WPS": "PO_WPS",
    "PGC_noWPS": "PO_noWPS",
}


def load_gc_full() -> pd.DataFrame:
    """Load per-departure GC rows and map them onto optimised-case ids.

    GC summaries are read from the penalty output folder because that run
    contains the reference great-circle exports for all four cases.
    """
    folder = EXPERIMENTS["penalty"]["folder"]
    frames = []
    for gc_id, opt_id in _GC_TO_OPT.items():
        path = OUTPUT_DIR / folder / f"IEUniversity-1-{gc_id}.csv"
        if not path.exists():
            continue
        gc = pd.read_csv(path, parse_dates=["departure_time_utc", "arrival_time_utc"])
        gc["case_id"] = opt_id
        gc["gc_id"] = gc_id
        gc["month"] = gc["departure_time_utc"].dt.month
        gc["season"] = gc["month"].map(_MONTH_TO_SEASON)
        gc["wind_viol"] = gc["max_wind_mps"] > WIND_LIMIT
        gc["wave_viol"] = gc["max_hs_m"] > WAVE_LIMIT
        gc["any_viol"] = gc["wind_viol"] | gc["wave_viol"]
        frames.append(gc)
    return pd.concat(frames, ignore_index=True)


def _join_opt_to_gc(
    df: pd.DataFrame, gc_full: pd.DataFrame, exp_key: str, case_id: str
) -> pd.DataFrame:
    """Join one experiment/case slice against its matched GC departure rows.

    The join is keyed by ``departure_time_utc`` so every comparison uses the
    same calendar departure in the optimised and great-circle datasets.
    """
    opt = (
        df[(df["experiment"] == exp_key) & (df["case_id"] == case_id)]
        .set_index("departure_time_utc")[["energy_cons_mwh", "month", "season"]]
        .rename(columns={"energy_cons_mwh": "energy_opt"})
    )
    gc = (
        gc_full[gc_full["case_id"] == case_id]
        .set_index("departure_time_utc")[["energy_cons_mwh"]]
        .rename(columns={"energy_cons_mwh": "energy_gc"})
    )
    joined = opt.join(gc, how="inner")
    joined["margin_pct"] = (
        (joined["energy_gc"] - joined["energy_opt"]) / joined["energy_gc"] * 100
    )
    joined["beats_gc"] = joined["margin_pct"] > 0
    return joined.reset_index()


# ===========================================================================
# FIGURE 10 — Monthly "victory rate" over GC
# ===========================================================================
def fig_gc_victory_rate(df: pd.DataFrame, gc_full: pd.DataFrame) -> None:
    """Monthly % of departures that beat the GC energy for each case × experiment."""
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "How often do we beat the great-circle route? A month-by-month scorecard",
        fontsize=12,
        fontweight="bold",
        x=0.02,
        ha="left",
    )

    cases_order = [
        ("AO_WPS", "Atlantic — with WPS"),
        ("AO_noWPS", "Atlantic — without WPS"),
        ("PO_WPS", "Pacific — with WPS"),
        ("PO_noWPS", "Pacific — without WPS"),
    ]
    months = np.arange(1, 13)

    for ax, (case_id, panel_title) in zip(axes.flat, cases_order, strict=False):
        ax.set_title(panel_title, fontsize=10, fontweight="bold")

        # 50 % reference band
        ax.axhspan(45, 55, color="#E5E5E5", alpha=0.5, zorder=1)
        ax.axhline(
            50,
            color="#888",
            linewidth=1.0,
            linestyle="--",
            zorder=2,
            label="50% threshold",
        )

        for exp_key in EXPERIMENTS:
            joined = _join_opt_to_gc(df, gc_full, exp_key, case_id)
            if joined.empty:
                continue
            monthly_rate = (joined.groupby("month")["beats_gc"].mean() * 100).reindex(
                months
            )
            ax.plot(
                monthly_rate.index,
                monthly_rate.values,
                color=EXPERIMENTS[exp_key]["color"],
                linewidth=2.0,
                marker="o",
                markersize=4,
                label=EXPERIMENTS[exp_key]["label"],
                zorder=4,
                alpha=0.92,
            )

        # Season background
        for start, end, s in [
            (0.5, 2.5, "Winter"),
            (2.5, 5.5, "Spring"),
            (5.5, 8.5, "Summer"),
            (8.5, 11.5, "Autumn"),
            (11.5, 12.5, "Winter"),
        ]:
            ax.axvspan(start, end, alpha=0.05, color=SEASON_COLORS[s], zorder=0)

        ax.set_xticks(months)
        ax.set_xticklabels(MONTH_ABBR, fontsize=8)
        ax.set_xlabel("Departure month")
        ax.set_ylabel("Departures beating GC (%)")
        ax.set_ylim(40, 105)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.7)
        ax.set_axisbelow(True)

    exp_handles = [
        mlines.Line2D(
            [],
            [],
            color=EXPERIMENTS[k]["color"],
            linewidth=2,
            marker="o",
            markersize=4,
            label=EXPERIMENTS[k]["label"],
        )
        for k in EXPERIMENTS
    ]
    season_handles = [
        mpatches.Patch(facecolor=SEASON_COLORS[s], alpha=0.5, label=s)
        for s in SEASON_ORDER
    ]
    fig.legend(
        handles=exp_handles + season_handles,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.03),
        fontsize=8.5,
    )
    add_source_note(fig)
    out = FIGS_DIR / "fig10_gc_victory_rate.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    print(f"  Saved {out.name}")
    plt.close(fig)


# ===========================================================================
# FIGURE 11 — Margin-over-GC heatmap
# ===========================================================================
def fig_gc_margin_heatmap(df: pd.DataFrame, gc_full: pd.DataFrame) -> None:
    """Heatmap: median % margin over GC (rows=experiments, cols=months) per case."""
    setup_style()
    # 2×2 grid of cases; each subplot is an experiment×month heatmap
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "Energy margin over the great-circle route — darker green means a bigger win",
        fontsize=12,
        fontweight="bold",
        x=0.02,
        ha="left",
    )

    cases_order = [
        ("AO_WPS", "Atlantic — with WPS"),
        ("AO_noWPS", "Atlantic — without WPS"),
        ("PO_WPS", "Pacific — with WPS"),
        ("PO_noWPS", "Pacific — without WPS"),
    ]

    import matplotlib.colors as mcolors

    # Sequential green: since we always beat GC, show "how much we win"
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "gc_margin_win",
        [
            "#F8F8F8",  # near-zero margin → white
            "#C7E5A0",  # moderate margin → light green
            "#6DC201",  # large margin → IE tech green
        ],
        N=256,
    )

    # Collect all margin values — use 0 to 95th pct (positive range only)
    all_margins: list[float] = []
    for case_id, _ in cases_order:
        for exp_key in EXPERIMENTS:
            joined = _join_opt_to_gc(df, gc_full, exp_key, case_id)
            if not joined.empty:
                all_margins.extend(joined["margin_pct"].dropna().tolist())
    vabs = np.nanpercentile(all_margins, 95)  # upper bound

    for ax, (case_id, panel_title) in zip(axes.flat, cases_order, strict=False):
        ax.set_title(panel_title, fontsize=10, fontweight="bold")

        matrix = np.full((len(EXPERIMENTS), 12), np.nan)
        exp_labels = []
        for i, exp_key in enumerate(EXPERIMENTS):
            exp_labels.append(EXPERIMENTS[exp_key]["label"])
            joined = _join_opt_to_gc(df, gc_full, exp_key, case_id)
            if joined.empty:
                continue
            for m in range(1, 13):
                vals = joined.loc[joined["month"] == m, "margin_pct"]
                if len(vals) > 0:
                    matrix[i, m - 1] = vals.median()

        im = ax.imshow(
            matrix,
            aspect="auto",
            cmap=cmap,
            vmin=0,
            vmax=vabs,
            interpolation="nearest",
        )

        # Annotate cells
        for i in range(len(EXPERIMENTS)):
            for j in range(12):
                val = matrix[i, j]
                if not np.isnan(val):
                    txt_color = "white" if val > vabs * 0.60 else "#333"
                    ax.text(
                        j,
                        i,
                        f"{val:+.0f}%",
                        ha="center",
                        va="center",
                        fontsize=6.5,
                        color=txt_color,
                        fontweight="bold",
                    )

        ax.set_xticks(np.arange(12))
        ax.set_xticklabels(MONTH_ABBR, fontsize=8)
        ax.set_yticks(np.arange(len(EXPERIMENTS)))
        ax.set_yticklabels(exp_labels, fontsize=7.5)

        # Month-season separators
        for sep in [2.5, 5.5, 8.5, 11.5]:
            ax.axvline(sep, color="#888", linewidth=0.6, linestyle=":")

        plt.colorbar(
            im,
            ax=ax,
            shrink=0.7,
            label="Median margin over GC (%)",
            format=mticker.FuncFormatter(lambda x, _: f"{x:+.0f}%"),
        )

    add_source_note(fig)
    out = FIGS_DIR / "fig11_gc_margin_heatmap.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    print(f"  Saved {out.name}")
    plt.close(fig)


# ===========================================================================
# FIGURE 12 — GC's own violations: the unfair baseline
# ===========================================================================
def fig_gc_violations(df: pd.DataFrame, gc_full: pd.DataFrame) -> None:
    """Monthly 'any violation' rate — GC vs best optimised (Penalty + FMS)."""
    setup_style()
    best_exp = "penalty_fms"
    GC_COLOR = "#878787"
    OPT_COLOR = EXPERIMENTS[best_exp]["color"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "Optimised routing reduces dangerous weather exposure in the Atlantic — Pacific wind routes face an energy–safety tradeoff",  # noqa: E501
        fontsize=12,
        fontweight="bold",
        x=0.02,
        ha="left",
    )

    cases_order = [
        ("AO_WPS", "Atlantic — with WPS"),
        ("AO_noWPS", "Atlantic — without WPS"),
        ("PO_WPS", "Pacific — with WPS"),
        ("PO_noWPS", "Pacific — without WPS"),
    ]
    months = np.arange(1, 13)
    bar_w = 0.38

    for ax, (case_id, panel_title) in zip(axes.flat, cases_order, strict=False):
        ax.set_title(panel_title, fontsize=10, fontweight="bold")

        gc_piece = gc_full[gc_full["case_id"] == case_id]
        opt_piece = df[(df["experiment"] == best_exp) & (df["case_id"] == case_id)]

        gc_any = gc_piece.groupby("month")["any_viol"].mean() * 100
        opt_any = opt_piece.groupby("month")["any_viol"].mean() * 100

        x = months - 1  # 0-indexed

        ax.bar(
            x - bar_w / 2,
            gc_any.reindex(months, fill_value=0),
            width=bar_w * 0.92,
            color=GC_COLOR,
            alpha=0.75,
            label="Great-circle",
            zorder=3,
        )
        ax.bar(
            x + bar_w / 2,
            opt_any.reindex(months, fill_value=0),
            width=bar_w * 0.92,
            color=OPT_COLOR,
            alpha=0.88,
            label="CMA-ES + Penalty + FMS",
            zorder=3,
        )

        # Annotate the biggest reductions
        for m_idx in range(12):
            gc_val = gc_any.get(m_idx + 1, 0)
            opt_val = opt_any.get(m_idx + 1, 0)
            reduction = gc_val - opt_val
            if gc_val > 15 and reduction > 8:
                ax.text(
                    m_idx,
                    max(gc_val, opt_val) + 1.5,
                    f"\u2212{reduction:.0f}pp",
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    color="#444",
                )

        # Season background
        for start, end, s in [
            (-0.5, 1.5, "Winter"),
            (1.5, 4.5, "Spring"),
            (4.5, 7.5, "Summer"),
            (7.5, 10.5, "Autumn"),
            (10.5, 11.5, "Winter"),
        ]:
            ax.axvspan(start, end, alpha=0.05, color=SEASON_COLORS[s], zorder=0)

        ax.set_xticks(np.arange(12))
        ax.set_xticklabels(MONTH_ABBR, fontsize=8)
        ax.set_xlabel("Departure month")
        ax.set_ylabel("Departures with any weather violation (%)")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.7)
        ax.set_axisbelow(True)

        # In Pacific WPS panels, warn that WPS routes seek high-wind areas
        if "PO_WPS" in case_id:
            ax.text(
                0.97,
                0.96,
                "WPS routes seek\nwindy/wavy areas",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7.5,
                color="#555",
                style="italic",
            )

    # Shared y-axis
    ymax = max(ax.get_ylim()[1] for ax in axes.flat)
    for ax in axes.flat:
        ax.set_ylim(0, ymax)

    handles = [
        mpatches.Patch(facecolor=GC_COLOR, alpha=0.75, label="Great-circle route"),
        mpatches.Patch(
            facecolor=OPT_COLOR,
            alpha=0.88,
            label="CMA-ES + Penalty + FMS",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.04),
        fontsize=9,
    )
    add_source_note(fig)
    out = FIGS_DIR / "fig12_gc_violations.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    print(f"  Saved {out.name}")
    plt.close(fig)


# ===========================================================================
# CLI
# ===========================================================================
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="SWOPP3 2024 comparative analysis — generate figures and summary table."  # noqa: E501
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Root directory containing experiment output folders (default: <repo>/output).",  # noqa: E501
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Directory where figures and tables are saved (default: <repo>/output/analysis).",  # noqa: E501
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=180,
        metavar="DPI",
        help="Figure resolution in DPI (default: 180).",
    )
    p.add_argument(
        "--figures",
        nargs="+",
        type=int,
        metavar="N",
        default=None,
        help="Figure numbers to generate, e.g. --figures 1 5 10. Generates all if omitted.",  # noqa: E501
    )
    return p.parse_args()


# ===========================================================================
# MAIN
# ===========================================================================
def main() -> None:
    """Load datasets once, then generate the requested figures and summary."""
    global OUTPUT_DIR, FIGS_DIR

    args = parse_args()

    if args.data_dir is not None:
        OUTPUT_DIR = args.data_dir
    if args.output_dir is not None:
        FIGS_DIR = args.output_dir

    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    # Apply DPI setting
    import matplotlib as mpl  # noqa: PLC0415 — local import fine here

    mpl.rcParams["savefig.dpi"] = args.dpi

    want = set(args.figures) if args.figures else None

    def _want(n: int) -> bool:
        return want is None or n in want

    print("Loading data…")
    # Keep shared datasets in memory once so each figure function can focus on
    # presentation instead of repeating the same I/O and alignment work.
    gc_baselines = load_gc_baselines()
    df = load_all_data()
    gc_full = load_gc_full()
    print(
        f"  Loaded {len(df):,} voyage records across "
        f"{df['experiment'].nunique()} experiments and {df['case_id'].nunique()} cases."
    )

    print("\nGenerating figures…")
    if _want(1):
        fig_energy_overview(df, gc_baselines, gc_full)
    if _want(2):
        fig_optimization_gains(df, gc_baselines)
    if _want(3):
        fig_penalty_tradeoff(df)
    if _want(4):
        fig_seasonality_a(df, gc_full)
        fig_seasonality_b(df, gc_full)
    if _want(5):
        fig_wps_impact(df)
    if _want(6):
        fig_fms_improvement(df)
    if _want(7):
        fig_route_maps()
    if _want(8):
        fig_risk_calendar(df)
    if _want(9):
        fig_fms_delta_byseason(df)
    if _want(10):
        fig_gc_victory_rate(df, gc_full)
    if _want(11):
        fig_gc_margin_heatmap(df, gc_full)
    if _want(12):
        fig_gc_violations(df, gc_full)
    if _want(13):
        fig_relative_gain_a(df, gc_full)
        fig_relative_gain_b(df, gc_full)

    print("\nGenerating summary table…")
    summary = generate_summary_table(df, gc_baselines)
    print(summary.to_string(index=False))

    print(f"\nAll outputs saved to {FIGS_DIR}/")


if __name__ == "__main__":
    main()
