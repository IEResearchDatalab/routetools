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
                        (0–13). Generates all figures if omitted.
    --dpi DPI           Figure resolution in DPI. Default: 180

Outputs
-------
    fig00_teaser_routes.pdf / .png
    fig01_energy_overview.pdf / .png
    fig02_optimization_gains.pdf / .png
    fig03_penalty_tradeoff.pdf / .png
    fig04a_seasonality_sweep_combined.pdf / .png
    fig04b_seasonality_penalty.pdf / .png
    fig13a_relative_gain_sweep_combined.pdf / .png
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
    The active experiments are driven by the ``EXPERIMENT_PAIRS`` constant near
    the top of this file.  Each entry is a ``(base_key, fms_key)`` tuple that
    maps a base CMA-ES run to its FMS post-refinement counterpart.

    Built-in profiles (uncomment the desired one):

    Two-experiment profile (default — sweep combined):
        EXPERIMENT_PAIRS = [("sweep_combined", "sweep_combined_fms")]

    Four-experiment profile (penalty vs no-penalty comparison):
        EXPERIMENT_PAIRS = [
            ("no_penalty", "no_penalty_fms"),
            ("penalty", "penalty_fms"),
        ]

    All known experiment keys and their folder/colour/label metadata live in
    ``EXPERIMENTS_REGISTRY``.  ``ACTIVE_EXPERIMENTS`` is derived automatically
    from ``EXPERIMENT_PAIRS`` — do not edit it directly.

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
    fig00 Compact teaser map (Atlantic + Pacific) with one representative
          departure per corridor, overlaying GC (dashed grey), CMA-ES
          (orange), and final BERS route (blue).
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
            <team-prefix>-<case>.csv    # one summary file per case
            tracks/
                <details_filename from summary CSV>  # per-voyage track

    Missing experiment folders or individual CSVs are silently skipped;
    all figures degrade gracefully to show only available data.
"""

from __future__ import annotations

import argparse
import re
import warnings
from functools import cache
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from routetools.analysis_config import (
    EXPERIMENTS_REGISTRY,
    AnalysisPaths,
    _experiment_folder,
)
from routetools.violations import find_team_prefix

# ---------------------------------------------------------------------------
# Paths (defaults; overridden at runtime via CLI args in main())
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.parent


DEFAULT_PATHS = AnalysisPaths(
    output_dir=_REPO_ROOT / "output",
    figs_dir=_REPO_ROOT / "output" / "analysis",
    config_path=_REPO_ROOT / "config.toml",
)


@cache
def _team_prefix(folder_dir: Path) -> str:
    """Return the detected team prefix for one experiment output folder."""
    return find_team_prefix(folder_dir)


def _summary_csv_path(paths: AnalysisPaths, folder: str, case_id: str) -> Path | None:
    """Return the summary CSV path for one case when present."""
    folder_dir = paths.output_dir / folder
    if not folder_dir.exists():
        return None
    try:
        team_prefix = _team_prefix(folder_dir)
    except FileNotFoundError:
        return None
    return folder_dir / f"{team_prefix}-{case_id}.csv"


# ---------------------------------------------------------------------------
# Experiment registry is imported from routetools.analysis_config
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Analysis profile — THE only place to change for a different comparison
# ---------------------------------------------------------------------------
# Each tuple is (base_experiment_key, fms_experiment_key).  All figures and
# data loaders are derived from this list automatically.
# Two-experiment profile (sweep combined, no penalty distinction):
EXPERIMENT_PAIRS: list[tuple[str, str]] = [
    ("sweep_combined", "sweep_combined_fms_strict"),
]
# Four-experiment profile (no-penalty vs penalty comparison); uncomment to use:
# EXPERIMENT_PAIRS = [
#     ("no_penalty", "no_penalty_fms"),
#     ("penalty", "penalty_fms"),
# ]

# Active experiments for this run — derived from EXPERIMENT_PAIRS, do not edit
ACTIVE_EXPERIMENTS: dict[str, dict] = {
    k: EXPERIMENTS_REGISTRY[k] for pair in EXPERIMENT_PAIRS for k in pair
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

_CASE_FILE_SUFFIX = {
    "AO_WPS": "atlantic_wps",
    "AO_noWPS": "atlantic_no_wps",
    "PO_WPS": "pacific_wps",
    "PO_noWPS": "pacific_no_wps",
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
            "figure.facecolor": "none",
            "axes.facecolor": "none",
            "xtick.bottom": False,
            "ytick.left": False,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "legend.handlelength": 1.5,
            "savefig.bbox": "tight",
            "savefig.dpi": 180,
            "savefig.facecolor": "none",
            "savefig.transparent": True,
            "figure.constrained_layout.use": True,
        }
    )


def _save_figure_outputs(
    fig: plt.Figure,
    out: Path,
    **savefig_kwargs: object,
) -> None:
    """Write transparent PDF/PNG files for one figure."""
    fig.patch.set_alpha(0)
    for ax in fig.axes:
        ax.set_facecolor("none")
        ax.patch.set_alpha(0)

    save_kwargs = {"transparent": True, **savefig_kwargs}
    fig.savefig(out, **save_kwargs)

    hidden_items: list[tuple[object, bool]] = []
    if fig._suptitle is not None:
        hidden_items.append((fig._suptitle, fig._suptitle.get_visible()))
        fig._suptitle.set_visible(False)
    for text in fig.texts:
        if "source" in text.get_text().lower():
            hidden_items.append((text, text.get_visible()))
            text.set_visible(False)

    fig.savefig(out.with_suffix(".png"), **save_kwargs)

    for artist, was_visible in hidden_items:
        artist.set_visible(was_visible)

    out.with_suffix(".tikz").unlink(missing_ok=True)


def _slugify(value: str) -> str:
    """Return a filesystem-safe lowercase suffix for subplot filenames."""
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "panel"


def _save_subplot_outputs(
    fig: plt.Figure,
    out: Path,
    axes: list[plt.Axes],
    panel_suffixes: list[str],
    *,
    pad_inches: float = 0.06,
    **savefig_kwargs: object,
) -> None:
    """Save each axis as an individual cropped PNG file.

    The output names use ``<figure_stem>_<panel_suffix>.png``.
    """
    if len(axes) != len(panel_suffixes):
        raise ValueError("axes and panel_suffixes must have identical length")

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_w, fig_h = fig.get_size_inches()
    save_kwargs = {"transparent": True, **savefig_kwargs}
    save_kwargs.pop("bbox_inches", None)

    for ax, suffix in zip(axes, panel_suffixes, strict=False):
        bbox_display = ax.get_tightbbox(renderer)
        if bbox_display is None:
            continue

        bbox_inches = bbox_display.transformed(fig.dpi_scale_trans.inverted())
        x0 = max(0.0, bbox_inches.x0 - pad_inches)
        y0 = max(0.0, bbox_inches.y0 - pad_inches)
        x1 = min(fig_w, bbox_inches.x1 + pad_inches)
        y1 = min(fig_h, bbox_inches.y1 + pad_inches)
        if x1 <= x0 or y1 <= y0:
            continue

        panel_bbox = mpl.transforms.Bbox.from_extents(x0, y0, x1, y1)
        panel_out = out.with_name(f"{out.stem}_{_slugify(suffix)}{out.suffix}")
        fig.savefig(
            panel_out.with_suffix(".png"),
            bbox_inches=panel_bbox,
            **save_kwargs,
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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _outlier_mask(series: pd.Series, iqr_factor: float = 5.0) -> pd.Series:
    """Return a boolean mask of non-outliers using IQR rule."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return (series >= q1 - iqr_factor * iqr) & (series <= q3 + iqr_factor * iqr)


def load_summary_csv(
    exp_key: str,
    case_id: str,
    paths: AnalysisPaths = DEFAULT_PATHS,
) -> pd.DataFrame | None:
    """Load and annotate one experiment/case summary CSV."""
    folder = _experiment_folder(exp_key, paths)
    path = _summary_csv_path(paths, folder, case_id)
    if path is None or not path.exists():
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
    mask = _outlier_mask(df["energy_cons_mwh"])
    if (~mask).any():
        print(f"  [!] Dropping {(~mask).sum()} outliers from {exp_key}/{case_id}")
    return df[mask].copy()


def load_gc_baselines(paths: AnalysisPaths = DEFAULT_PATHS) -> dict[str, float]:
    """Return mean energy per GC case, averaged across all base experiment folders.

    Iterates over all pairs in ``EXPERIMENT_PAIRS`` and collects GC data from
    each base experiment folder.
    """
    baselines: dict[str, list] = {}
    for base_key, _fms_key in EXPERIMENT_PAIRS:
        folder = _experiment_folder(base_key, paths)
        for gc_id in GC_CASES:
            path = _summary_csv_path(paths, folder, gc_id)
            if path is None or not path.exists():
                continue
            df = pd.read_csv(path)
            baselines.setdefault(gc_id, []).append(df["energy_cons_mwh"].mean())
    return {k: float(np.mean(v)) for k, v in baselines.items()}


def load_all_data(paths: AnalysisPaths = DEFAULT_PATHS) -> pd.DataFrame:
    """Load all optimised-case summary rows across all experiments."""
    frames = []
    for exp_key in ACTIVE_EXPERIMENTS:
        for case_id in OPT_CASES:
            df = load_summary_csv(exp_key, case_id, paths)
            if df is not None:
                frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_tracks(
    exp_key: str,
    case_id: str,
    paths: AnalysisPaths = DEFAULT_PATHS,
    season_filter: str | None = None,
    n_sample: int = 8,
) -> list[pd.DataFrame]:
    """Return sampled per-voyage tracks for one experiment/case pair.

    The sampling operates on the summary CSV first so that the selected track
    files inherit season and departure metadata from the same voyage rows.
    """
    folder = _experiment_folder(exp_key, paths)
    tracks_dir = paths.output_dir / folder / "tracks"
    if not tracks_dir.exists():
        return []

    # Load the summary to know departure dates by season
    summary = load_summary_csv(exp_key, case_id, paths)
    if summary is None:
        return []

    if season_filter:
        summary = summary[summary["season"] == season_filter]

    # Keep figure selection deterministic so repeated runs save the same tracks.
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
    df: pd.DataFrame,
    gc: dict[str, float],
    gc_full: pd.DataFrame,
    paths: AnalysisPaths = DEFAULT_PATHS,
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
    exp_order = list(ACTIVE_EXPERIMENTS.keys())
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
        gc_vals = gc_vals_raw[_outlier_mask(gc_vals_raw)]
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
                ax, sub.values, exp_positions[i], ACTIVE_EXPERIMENTS[exp_key]["color"]
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
                color=ACTIVE_EXPERIMENTS[exp_key]["color"],
                fontweight="bold",
            )

        ax.set_title(
            case_meta["label"].replace("\n", " "), fontsize=10, fontweight="bold"
        )
        all_ticks = [gc_pos] + list(exp_positions)
        all_labels = ["GC"]
        if exp_order == ["sweep_combined", "sweep_combined_fms_strict"]:
            all_labels.extend(["CMA-ES", "BERS"])
        else:
            all_labels.extend(
                ACTIVE_EXPERIMENTS[k]["short"].replace(" + ", "\n+\n")
                for k in exp_order
            )
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
            facecolor=ACTIVE_EXPERIMENTS[k]["color"],
            alpha=0.85,
            label=ACTIVE_EXPERIMENTS[k]["label"],
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

    # Keep per-case panels directly comparable in both combined and per-panel exports.
    all_ylims = [ax.get_ylim() for ax in axes]
    ymin_all = min(y[0] for y in all_ylims)
    ymax_all = max(y[1] for y in all_ylims)
    for ax in axes:
        ax.set_ylim(ymin_all, ymax_all)

    add_source_note(fig)
    out = paths.figs_dir / "fig01_energy_overview.pdf"
    _save_subplot_outputs(
        fig,
        out,
        list(axes),
        [_CASE_FILE_SUFFIX[case_id] for case_id in OPT_CASES],
    )
    _save_figure_outputs(fig, out)
    print(f"  Saved {out.name}")
    plt.close(fig)


# ===========================================================================
# FIGURE 2 — Optimisation gains vs GC baseline
# ===========================================================================
def fig_optimization_gains(
    df: pd.DataFrame,
    gc: dict[str, float],
    paths: AnalysisPaths = DEFAULT_PATHS,
) -> None:
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
    exp_order = list(ACTIVE_EXPERIMENTS.keys())

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
                color=ACTIVE_EXPERIMENTS[exp_key]["color"],
                alpha=0.88,
                label=ACTIVE_EXPERIMENTS[exp_key]["label"],
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
                        color=ACTIVE_EXPERIMENTS[exp_key]["color"],
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
            facecolor=ACTIVE_EXPERIMENTS[k]["color"],
            alpha=0.85,
            label=ACTIVE_EXPERIMENTS[k]["label"],
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
    out = paths.figs_dir / "fig02_optimization_gains.pdf"
    _save_subplot_outputs(
        fig,
        out,
        list(axes),
        [route for route, _title, _cases in route_groups],
    )
    _save_figure_outputs(fig, out)
    print(f"  Saved {out.name}")
    plt.close(fig)


# ===========================================================================
# FIGURE 3 — Penalty trade-off (safety vs efficiency)
# ===========================================================================
def fig_penalty_tradeoff(
    df: pd.DataFrame,
    paths: AnalysisPaths = DEFAULT_PATHS,
) -> None:
    """Side-by-side bars: experiment comparison — violation rates and mean energy."""
    setup_style()
    active_keys = list(ACTIVE_EXPERIMENTS.keys())
    sub = df[df["experiment"].isin(active_keys)].copy()

    records = []
    for case_id in OPT_CASES:
        for exp_key in active_keys:
            piece = sub[(sub["experiment"] == exp_key) & (sub["case_id"] == case_id)]
            if piece.empty:
                continue
            records.append(
                {
                    "case_id": case_id,
                    "experiment": exp_key,
                    "wind_violation_pct": piece["wind_viol"].mean() * 100,
                    "wave_violation_pct": piece["wave_viol"].mean() * 100,
                    "any_violation_pct": piece["any_viol"].mean() * 100,
                    "mean_energy": piece["energy_cons_mwh"].mean(),
                }
            )
    metrics = pd.DataFrame(records)

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.suptitle(
        "Experiment comparison — violation rates and energy consumption",
        fontsize=12,
        fontweight="bold",
        x=0.02,
        ha="left",
    )

    cases_order = list(OPT_CASES.keys())
    x = np.arange(len(cases_order))
    bw = 0.7 / max(len(active_keys), 1)
    _offsets = np.linspace(
        -(len(active_keys) - 1) / 2 * bw,
        (len(active_keys) - 1) / 2 * bw,
        len(active_keys),
    )

    # Panel A — wind violation rate
    ax = axes[0]
    ax.set_title("Wind violations\n(% of departures above 20 m/s)", fontsize=9.5)
    for i, exp_key in enumerate(active_keys):
        vals = [
            metrics.loc[
                (metrics.case_id == c) & (metrics.experiment == exp_key),
                "wind_violation_pct",
            ].values
            for c in cases_order
        ]
        vals = [v[0] if len(v) > 0 else np.nan for v in vals]
        ax.bar(
            x + _offsets[i],
            vals,
            width=bw * 0.92,
            color=ACTIVE_EXPERIMENTS[exp_key]["color"],
            alpha=0.88,
            label=ACTIVE_EXPERIMENTS[exp_key]["label"],
            zorder=3,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([OPT_CASES[c]["label_short"] for c in cases_order], fontsize=8)
    ax.set_ylabel("Departures with wind violation (%)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))

    # Panel B — wave violation rate
    ax = axes[1]
    ax.set_title("Wave violations\n(% of departures above 7 m Hs)", fontsize=9.5)
    for i, exp_key in enumerate(active_keys):
        vals = [
            metrics.loc[
                (metrics.case_id == c) & (metrics.experiment == exp_key),
                "wave_violation_pct",
            ].values
            for c in cases_order
        ]
        vals = [v[0] if len(v) > 0 else np.nan for v in vals]
        ax.bar(
            x + _offsets[i],
            vals,
            width=bw * 0.92,
            color=ACTIVE_EXPERIMENTS[exp_key]["color"],
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
    for i, exp_key in enumerate(active_keys):
        vals = [
            metrics.loc[
                (metrics.case_id == c) & (metrics.experiment == exp_key), "mean_energy"
            ].values
            for c in cases_order
        ]
        vals = [v[0] if len(v) > 0 else np.nan for v in vals]
        bars = ax.bar(
            x + _offsets[i],
            vals,
            width=bw * 0.92,
            color=ACTIVE_EXPERIMENTS[exp_key]["color"],
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
                    color=ACTIVE_EXPERIMENTS[exp_key]["color"],
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
            facecolor=ACTIVE_EXPERIMENTS[k]["color"],
            alpha=0.85,
            label=ACTIVE_EXPERIMENTS[k]["label"],
        )
        for k in active_keys
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(active_keys),
        bbox_to_anchor=(0.5, -0.04),
        fontsize=9,
    )

    add_source_note(fig)
    out = paths.figs_dir / "fig03_penalty_tradeoff.pdf"
    _save_subplot_outputs(
        fig,
        out,
        list(axes),
        ["wind_violations", "wave_violations", "mean_energy"],
    )
    _save_figure_outputs(fig, out)
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
    paths: AnalysisPaths = DEFAULT_PATHS,
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
            exp_meta = EXPERIMENTS_REGISTRY[exp_key]
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
        gc_piece = gc_piece[_outlier_mask(gc_piece["energy_cons_mwh"])]
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
            color=EXPERIMENTS_REGISTRY[k]["color"],
            linewidth=2.0,
            alpha=0.85,
            label=EXPERIMENTS_REGISTRY[k]["label"],
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
    out = paths.figs_dir / f"{out_stem}.pdf"
    _save_subplot_outputs(
        fig,
        out,
        list(axes.flat),
        [_CASE_FILE_SUFFIX[case_id] for case_id, _ in cases_order],
        bbox_inches="tight",
    )
    _save_figure_outputs(fig, out, bbox_inches="tight")
    print(f"  Saved {out.name}")
    plt.close(fig)


def fig_seasonality_a(
    df: pd.DataFrame,
    gc_full: pd.DataFrame,
    paths: AnalysisPaths = DEFAULT_PATHS,
) -> None:
    """fig04a — seasonal energy lines for the first experiment pair."""
    _b, _f = EXPERIMENT_PAIRS[0]
    _fig_seasonality_panel(
        df,
        gc_full,
        exp_keys=list(EXPERIMENT_PAIRS[0]),
        title=(
            f"Seasonal energy \u2014 {ACTIVE_EXPERIMENTS[_b]['short']}"
            f" vs {ACTIVE_EXPERIMENTS[_f]['short']}"
            f" (GC \u00b7 {ACTIVE_EXPERIMENTS[_b]['label']} \u00b7 {ACTIVE_EXPERIMENTS[_f]['label']})"  # noqa: E501
        ),
        out_stem="fig04a_seasonality_sweep_combined",
        paths=paths,
    )


def fig_seasonality_b(
    df: pd.DataFrame,
    gc_full: pd.DataFrame,
    paths: AnalysisPaths = DEFAULT_PATHS,
) -> None:
    """fig04b — seasonal energy lines for the last experiment pair.

    Identical to fig04a when only one pair is active.
    """
    _b, _f = EXPERIMENT_PAIRS[-1]
    _fig_seasonality_panel(
        df,
        gc_full,
        exp_keys=list(EXPERIMENT_PAIRS[-1]),
        title=(
            f"Seasonal energy \u2014 {ACTIVE_EXPERIMENTS[_b]['short']}"
            f" vs {ACTIVE_EXPERIMENTS[_f]['short']}"
            f" (GC \u00b7 {ACTIVE_EXPERIMENTS[_b]['label']} \u00b7 {ACTIVE_EXPERIMENTS[_f]['label']})"  # noqa: E501
        ),
        out_stem="fig04b_seasonality_penalty",
        paths=paths,
    )


def _fig_relative_gain_panel(
    df: pd.DataFrame,
    gc_full: pd.DataFrame,
    exp_keys: list[str],
    title: str,
    out_stem: str,
    paths: AnalysisPaths = DEFAULT_PATHS,
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
            exp_meta = EXPERIMENTS_REGISTRY[exp_key]
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
            color=EXPERIMENTS_REGISTRY[k]["color"],
            linewidth=2.0,
            alpha=0.85,
            label=EXPERIMENTS_REGISTRY[k]["label"],
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
    out = paths.figs_dir / f"{out_stem}.pdf"
    _save_subplot_outputs(
        fig,
        out,
        list(axes.flat),
        [_CASE_FILE_SUFFIX[case_id] for case_id, _ in cases_order],
        bbox_inches="tight",
    )
    _save_figure_outputs(fig, out, bbox_inches="tight")
    print(f"  Saved {out.name}")
    plt.close(fig)


def fig_relative_gain_a(
    df: pd.DataFrame,
    gc_full: pd.DataFrame,
    paths: AnalysisPaths = DEFAULT_PATHS,
) -> None:
    """fig13a — relative energy gain vs GC for the first experiment pair."""
    _b, _f = EXPERIMENT_PAIRS[0]
    _fig_relative_gain_panel(
        df,
        gc_full,
        exp_keys=list(EXPERIMENT_PAIRS[0]),
        title=(
            f"Relative energy saving vs GC"
            f" — {ACTIVE_EXPERIMENTS[_b]['short']} vs {ACTIVE_EXPERIMENTS[_f]['short']}"
        ),
        out_stem="fig13a_relative_gain_sweep_combined",
        paths=paths,
    )


def fig_relative_gain_b(
    df: pd.DataFrame,
    gc_full: pd.DataFrame,
    paths: AnalysisPaths = DEFAULT_PATHS,
) -> None:
    """fig13b — relative energy gain vs GC for the last experiment pair.

    Identical to fig13a when only one pair is active.
    """
    _b, _f = EXPERIMENT_PAIRS[-1]
    _fig_relative_gain_panel(
        df,
        gc_full,
        exp_keys=list(EXPERIMENT_PAIRS[-1]),
        title=(
            f"Relative energy saving vs GC"
            f" — {ACTIVE_EXPERIMENTS[_b]['short']} vs {ACTIVE_EXPERIMENTS[_f]['short']}"
        ),
        out_stem="fig13b_relative_gain_penalty",
        paths=paths,
    )


# ===========================================================================
# FIGURE 5 — WPS impact
# ===========================================================================
def fig_wps_impact(
    df: pd.DataFrame,
    paths: AnalysisPaths = DEFAULT_PATHS,
) -> None:
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

        exp_order = list(ACTIVE_EXPERIMENTS.keys())

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

        colors = [ACTIVE_EXPERIMENTS[k]["color"] for k in exp_order]
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
        ax.set_xticklabels(
            [ACTIVE_EXPERIMENTS[k]["short"] for k in exp_order], fontsize=8.5
        )
        ax.set_ylabel("WPS energy saving (MWh)")
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.7)
        ax.set_axisbelow(True)

    # Equalise y-axis so Atlantic vs Pacific savings are directly comparable
    ymax = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(0, ymax)

    add_source_note(fig)
    out = paths.figs_dir / "fig05_wps_impact.pdf"
    _save_subplot_outputs(
        fig,
        out,
        list(axes),
        [route for route, _title, _wps_case, _nowps_case in route_groups],
    )
    _save_figure_outputs(fig, out)
    print(f"  Saved {out.name}")
    plt.close(fig)


# ===========================================================================
# FIGURE 6 — FMS improvement scatter
# ===========================================================================
def fig_fms_improvement(
    df: pd.DataFrame,
    paths: AnalysisPaths = DEFAULT_PATHS,
) -> None:
    """Scatter plot: CMA-ES energy vs CMA-ES+FMS energy (each point = one departure)."""
    setup_style()

    # Pairs: (base experiment, fms experiment, title); derived from EXPERIMENT_PAIRS
    pairs = [
        (
            p[0],
            p[1],
            f"{ACTIVE_EXPERIMENTS[p[0]]['short']}"
            f" vs {ACTIVE_EXPERIMENTS[p[1]]['short']}",
        )
        for p in EXPERIMENT_PAIRS
    ]

    n_pairs = len(pairs)
    fig, _axes_arr = plt.subplots(1, n_pairs, figsize=(7 * n_pairs, 5), squeeze=False)
    axes = list(_axes_arr.flat)
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
        ax.set_xlabel(f"{ACTIVE_EXPERIMENTS[base_exp]['label']} energy (MWh)")
        ax.set_ylabel(f"{ACTIVE_EXPERIMENTS[fms_exp]['label']} energy (MWh)")
        ax.set_aspect("equal", adjustable="box")
        ax.legend(fontsize=8, loc="upper left", markerscale=1.5)

    add_source_note(fig)
    out = paths.figs_dir / "fig06_fms_improvement.pdf"
    _save_subplot_outputs(
        fig,
        out,
        axes,
        [f"{base_exp}_vs_{fms_exp}" for base_exp, fms_exp, _title in pairs],
    )
    _save_figure_outputs(fig, out)
    print(f"  Saved {out.name}")
    plt.close(fig)


# ===========================================================================
# GC track loader (used by fig07)
# ===========================================================================
def load_gc_tracks(
    gc_case: str,
    paths: AnalysisPaths = DEFAULT_PATHS,
    season_filter: str | None = None,
    n_sample: int = 5,
) -> list[pd.DataFrame]:
    """Load sample GC track DataFrames from the first base experiment folder."""
    folder = _experiment_folder(EXPERIMENT_PAIRS[0][0], paths)
    tracks_dir = paths.output_dir / folder / "tracks"
    summary_path = _summary_csv_path(paths, folder, gc_case)
    if summary_path is None or not summary_path.exists():
        return []
    gc_df = pd.read_csv(
        summary_path, parse_dates=["departure_time_utc", "arrival_time_utc"]
    )
    gc_df["season"] = gc_df["departure_time_utc"].dt.month.map(_MONTH_TO_SEASON)
    if season_filter:
        gc_df = gc_df[gc_df["season"] == season_filter]
    # Keep GC track sampling deterministic so figure exports stay reproducible.
    sample = gc_df.sample(min(n_sample, len(gc_df)), replace=False, random_state=7)
    sample = sample.sort_values("departure_time_utc")
    result = []
    for _, row in sample.iterrows():
        fpath = tracks_dir / row["details_filename"]
        if fpath.exists():
            trk = pd.read_csv(fpath, parse_dates=["time_utc"])
            result.append(trk)
    return result


def _load_representative_route_triplet(
    case_id: str,
    gc_case: str,
    base_exp: str,
    final_exp: str,
    paths: AnalysisPaths,
) -> dict[str, object] | None:
    """Return aligned (GC, CMA-ES, final) tracks for one representative departure.

    Chooses the departure with the largest energy drop from base to final among
    rows that have all three track files available.
    """
    base_summary = load_summary_csv(base_exp, case_id, paths)
    final_summary = load_summary_csv(final_exp, case_id, paths)
    if base_summary is None or final_summary is None:
        return None

    base_folder = _experiment_folder(base_exp, paths)
    final_folder = _experiment_folder(final_exp, paths)
    gc_path = _summary_csv_path(paths, base_folder, gc_case)
    if gc_path is None or not gc_path.exists():
        return None
    gc_summary = pd.read_csv(gc_path, parse_dates=["departure_time_utc"])

    cols = [
        "departure_time_utc",
        "details_filename",
        "energy_cons_mwh",
        "max_wind_mps",
        "max_hs_m",
    ]
    base_cols = base_summary[cols].rename(
        columns={
            "details_filename": "base_details",
            "energy_cons_mwh": "base_energy",
            "max_wind_mps": "base_wind",
            "max_hs_m": "base_wave",
        }
    )
    final_cols = final_summary[cols].rename(
        columns={
            "details_filename": "final_details",
            "energy_cons_mwh": "final_energy",
            "max_wind_mps": "final_wind",
            "max_hs_m": "final_wave",
        }
    )
    gc_cols = gc_summary[cols].rename(
        columns={
            "details_filename": "gc_details",
            "energy_cons_mwh": "gc_energy",
            "max_wind_mps": "gc_wind",
            "max_hs_m": "gc_wave",
        }
    )

    merged = (
        base_cols.merge(final_cols, on="departure_time_utc", how="inner")
        .merge(gc_cols, on="departure_time_utc", how="inner")
        .dropna(subset=["base_details", "final_details", "gc_details"])
    )
    if merged.empty:
        return None

    merged["delta_mwh"] = merged["base_energy"] - merged["final_energy"]
    merged["gain_cma_vs_gc_pct"] = (
        (merged["gc_energy"] - merged["base_energy"]) / merged["gc_energy"] * 100
    )
    merged["gain_bers_vs_gc_pct"] = (
        (merged["gc_energy"] - merged["final_energy"]) / merged["gc_energy"] * 100
    )
    merged = merged.sort_values("delta_mwh", ascending=False)

    base_tracks_dir = paths.output_dir / base_folder / "tracks"
    final_tracks_dir = paths.output_dir / final_folder / "tracks"
    for _, row in merged.iterrows():
        gc_file = base_tracks_dir / row["gc_details"]
        base_file = base_tracks_dir / row["base_details"]
        final_file = final_tracks_dir / row["final_details"]
        if not (gc_file.exists() and base_file.exists() and final_file.exists()):
            continue

        gc_track = pd.read_csv(gc_file, parse_dates=["time_utc"])
        base_track = pd.read_csv(base_file, parse_dates=["time_utc"])
        final_track = pd.read_csv(final_file, parse_dates=["time_utc"])
        return {
            "gc_track": gc_track,
            "base_track": base_track,
            "final_track": final_track,
            "delta_mwh": float(row["delta_mwh"]),
            "gain_cma_vs_gc_pct": float(row["gain_cma_vs_gc_pct"]),
            "gain_bers_vs_gc_pct": float(row["gain_bers_vs_gc_pct"]),
            "base_wind": float(row["base_wind"]),
            "base_wave": float(row["base_wave"]),
            "final_wind": float(row["final_wind"]),
            "final_wave": float(row["final_wave"]),
        }
    return None


def _load_best_route_triplets_by_season(
    case_id: str,
    gc_case: str,
    base_exp: str,
    final_exp: str,
    paths: AnalysisPaths,
    n_scenarios: int = 3,
) -> list[dict[str, object]]:
    """Return up to ``n_scenarios`` best seasonal route triplets vs GC.

    Produces at most one representative departure per season, ranked by the
    final-route gain over GC (highest first).
    """
    base_summary = load_summary_csv(base_exp, case_id, paths)
    final_summary = load_summary_csv(final_exp, case_id, paths)
    if base_summary is None or final_summary is None:
        return []

    base_folder = _experiment_folder(base_exp, paths)
    final_folder = _experiment_folder(final_exp, paths)
    gc_path = _summary_csv_path(paths, base_folder, gc_case)
    if gc_path is None or not gc_path.exists():
        return []
    gc_summary = pd.read_csv(gc_path, parse_dates=["departure_time_utc"])

    cols = [
        "departure_time_utc",
        "details_filename",
        "energy_cons_mwh",
        "max_wind_mps",
        "max_hs_m",
    ]
    base_cols = base_summary[cols].rename(
        columns={
            "details_filename": "base_details",
            "energy_cons_mwh": "base_energy",
            "max_wind_mps": "base_wind",
            "max_hs_m": "base_wave",
        }
    )
    final_cols = final_summary[cols].rename(
        columns={
            "details_filename": "final_details",
            "energy_cons_mwh": "final_energy",
            "max_wind_mps": "final_wind",
            "max_hs_m": "final_wave",
        }
    )
    gc_cols = gc_summary[cols].rename(
        columns={
            "details_filename": "gc_details",
            "energy_cons_mwh": "gc_energy",
        }
    )

    merged = (
        base_cols.merge(final_cols, on="departure_time_utc", how="inner")
        .merge(gc_cols, on="departure_time_utc", how="inner")
        .dropna(subset=["base_details", "final_details", "gc_details"])
    )
    if merged.empty:
        return []

    merged["season"] = merged["departure_time_utc"].dt.month.map(_MONTH_TO_SEASON)
    merged["gain_cma_vs_gc_pct"] = (
        (merged["gc_energy"] - merged["base_energy"]) / merged["gc_energy"] * 100
    )
    merged["gain_bers_vs_gc_pct"] = (
        (merged["gc_energy"] - merged["final_energy"]) / merged["gc_energy"] * 100
    )

    base_tracks_dir = paths.output_dir / base_folder / "tracks"
    final_tracks_dir = paths.output_dir / final_folder / "tracks"
    seasonal_candidates: list[dict[str, object]] = []

    for season in SEASON_ORDER:
        season_rows = merged[merged["season"] == season].sort_values(
            "gain_bers_vs_gc_pct", ascending=False
        )
        if season_rows.empty:
            continue

        for _, row in season_rows.iterrows():
            gc_file = base_tracks_dir / row["gc_details"]
            base_file = base_tracks_dir / row["base_details"]
            final_file = final_tracks_dir / row["final_details"]
            if not (gc_file.exists() and base_file.exists() and final_file.exists()):
                continue

            seasonal_candidates.append(
                {
                    "season": season,
                    "gc_track": pd.read_csv(gc_file, parse_dates=["time_utc"]),
                    "base_track": pd.read_csv(base_file, parse_dates=["time_utc"]),
                    "final_track": pd.read_csv(
                        final_file,
                        parse_dates=["time_utc"],
                    ),
                    "gain_cma_vs_gc_pct": float(row["gain_cma_vs_gc_pct"]),
                    "gain_bers_vs_gc_pct": float(row["gain_bers_vs_gc_pct"]),
                    "base_wind": float(row["base_wind"]),
                    "base_wave": float(row["base_wave"]),
                    "final_wind": float(row["final_wind"]),
                    "final_wave": float(row["final_wave"]),
                }
            )
            break

    seasonal_candidates.sort(key=lambda d: d["gain_bers_vs_gc_pct"], reverse=True)
    return seasonal_candidates[:n_scenarios]


def _fig_teaser_seasonal_scenarios_for_ocean(
    cfg: dict[str, object],
    paths: AnalysisPaths,
    base_exp: str,
    final_exp: str,
    n_scenarios: int = 3,
) -> None:
    """Save a compact multi-panel teaser with the best seasonal scenarios."""
    scenarios = _load_best_route_triplets_by_season(
        case_id=str(cfg["case_id"]),
        gc_case=str(cfg["gc_case"]),
        base_exp=base_exp,
        final_exp=final_exp,
        paths=paths,
        n_scenarios=n_scenarios,
    )
    if not scenarios:
        print(f"  [!] Missing seasonal scenarios for {cfg['title']}; skipping")
        return

    def _plot_wrapped_track(
        ax: plt.Axes,
        lon_vals: np.ndarray,
        lat_vals: np.ndarray,
        *,
        central_longitude: float,
        **plot_kwargs: object,
    ) -> None:
        lon = np.asarray(lon_vals, dtype=float)
        lat = np.asarray(lat_vals, dtype=float)
        valid = np.isfinite(lon) & np.isfinite(lat)
        if not valid.any():
            return
        lon = lon[valid]
        lat = lat[valid]
        lon = ((lon - central_longitude + 180.0) % 360.0) - 180.0 + central_longitude

        split_idx = np.where(np.abs(np.diff(lon)) > 180.0)[0] + 1
        lon_segments = np.split(lon, split_idx)
        lat_segments = np.split(lat, split_idx)
        for lon_seg, lat_seg in zip(lon_segments, lat_segments, strict=False):
            if len(lon_seg) < 2:
                continue
            ax.plot(
                lon_seg,
                lat_seg,
                transform=ccrs.PlateCarree(),
                **plot_kwargs,
            )

    with mpl.rc_context({"figure.constrained_layout.use": False}):
        ncols = len(scenarios)
        fig = plt.figure(figsize=(4.4 * ncols, 4.2), facecolor="#FAFAF7")
        fig.suptitle(
            f"{cfg['title']}: best seasonal scenarios vs great-circle",
            fontsize=11,
            fontweight="bold",
            x=0.02,
            ha="left",
        )

        axes = []
        for i in range(ncols):
            ax = fig.add_subplot(1, ncols, i + 1, projection=cfg["projection"])
            axes.append(ax)
            ax.set_extent(cfg["extent"], crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.OCEAN, facecolor="#EFF5FF", zorder=0)
            ax.add_feature(cfeature.LAND, facecolor="#D9D0C3", zorder=1)
            ax.add_feature(
                cfeature.COASTLINE,
                linewidth=0.45,
                edgecolor="#808080",
                zorder=2,
            )
            ax.gridlines(
                draw_labels=False,
                linewidth=0.35,
                color="#C8C8C8",
                x_inline=False,
                y_inline=False,
            )

        GC_COLOR = "#7A7A7A"
        CMA_COLOR = "#FF8C42"
        BERS_COLOR = "#1C5DAA"

        for ax, scenario in zip(axes, scenarios, strict=False):
            gc_trk = scenario["gc_track"]
            base_trk = scenario["base_track"]
            final_trk = scenario["final_track"]

            _plot_wrapped_track(
                ax,
                gc_trk["lon_deg"].to_numpy(),
                gc_trk["lat_deg"].to_numpy(),
                central_longitude=float(cfg["central_longitude"]),
                color=GC_COLOR,
                linewidth=1.1,
                linestyle="--",
                alpha=0.95,
                zorder=3,
            )
            _plot_wrapped_track(
                ax,
                base_trk["lon_deg"].to_numpy(),
                base_trk["lat_deg"].to_numpy(),
                central_longitude=float(cfg["central_longitude"]),
                color=CMA_COLOR,
                linewidth=1.4,
                alpha=0.9,
                zorder=4,
            )
            _plot_wrapped_track(
                ax,
                final_trk["lon_deg"].to_numpy(),
                final_trk["lat_deg"].to_numpy(),
                central_longitude=float(cfg["central_longitude"]),
                color=BERS_COLOR,
                linewidth=1.8,
                alpha=0.95,
                zorder=5,
            )

            ax.text(
                0.02,
                0.02,
                (
                    f"CMA-ES vs GC: {float(scenario['gain_cma_vs_gc_pct']):+.1f}%\n"
                    f"BERS vs GC: {float(scenario['gain_bers_vs_gc_pct']):+.1f}%"
                ),
                transform=ax.transAxes,
                fontsize=6.5,
                color="#4D4D4D",
                ha="left",
                va="bottom",
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "fc": "white",
                    "ec": "none",
                    "alpha": 0.74,
                },
                zorder=6,
            )
            ax.text(
                0.98,
                0.02,
                (
                    "Max (CMAES / BERS)\n"
                    "Wind (m/s): "
                    f"{float(scenario['base_wind']):.1f}/{float(scenario['final_wind']):.1f}\n"
                    "Wave Hs (m): "
                    f"{float(scenario['base_wave']):.1f}/{float(scenario['final_wave']):.1f}"
                ),
                transform=ax.transAxes,
                fontsize=6.0,
                color="#4D4D4D",
                ha="right",
                va="bottom",
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "fc": "white",
                    "ec": "none",
                    "alpha": 0.74,
                },
                zorder=6,
            )

        legend_elements = [
            mlines.Line2D(
                [],
                [],
                color=GC_COLOR,
                linestyle="--",
                linewidth=1.2,
                label="Great-circle",
            ),
            mlines.Line2D([], [], color=CMA_COLOR, linewidth=1.5, label="CMA-ES"),
            mlines.Line2D([], [], color=BERS_COLOR, linewidth=1.9, label="BERS"),
        ]
        fig.legend(
            handles=legend_elements,
            loc="lower center",
            ncol=3,
            bbox_to_anchor=(0.5, -0.02),
            fontsize=8,
        )

        add_source_note(fig)
        fig.tight_layout(rect=[0, 0.08, 1, 0.9])
        out = paths.figs_dir / f"fig00_{str(cfg['slug'])}_seasonal_scenarios.pdf"
        _save_subplot_outputs(
            fig,
            out,
            axes,
            [str(s["season"]).lower() for s in scenarios],
            bbox_inches="tight",
        )
        _save_figure_outputs(fig, out, bbox_inches="tight")
        print(f"  Saved {out.name}")
        plt.close(fig)


def fig_teaser_seasonal_scenarios(paths: AnalysisPaths = DEFAULT_PATHS) -> None:
    """Generate 3 best-vs-GC seasonal scenarios for each ocean corridor."""
    setup_style()
    base_exp, final_exp = EXPERIMENT_PAIRS[0]
    ocean_cfgs = [
        {
            "slug": "atlantic",
            "title": "Trans-Atlantic",
            "projection": ccrs.PlateCarree(central_longitude=-40),
            "central_longitude": -40,
            "extent": [-80, 15, 25, 65],
            "case_id": "AO_WPS",
            "gc_case": "AGC_WPS",
        },
        {
            "slug": "pacific",
            "title": "Trans-Pacific",
            "projection": ccrs.PlateCarree(central_longitude=180),
            "central_longitude": 180,
            "extent": [115, 250, 20, 65],
            "case_id": "PO_WPS",
            "gc_case": "PGC_WPS",
        },
    ]

    for cfg in ocean_cfgs:
        _fig_teaser_seasonal_scenarios_for_ocean(
            cfg=cfg,
            paths=paths,
            base_exp=base_exp,
            final_exp=final_exp,
            n_scenarios=3,
        )


# ==========================================================================
# FIGURE 0 — Teaser route comparison
# ==========================================================================
def fig_teaser_routes(paths: AnalysisPaths = DEFAULT_PATHS) -> None:
    """Compact teaser map with one representative Atlantic and Pacific departure."""
    setup_style()

    def _plot_wrapped_track(
        ax: plt.Axes,
        lon_vals: np.ndarray,
        lat_vals: np.ndarray,
        *,
        central_longitude: float,
        **plot_kwargs: object,
    ) -> None:
        """Plot a track split at antimeridian crossings to avoid wrap artefacts."""
        lon = np.asarray(lon_vals, dtype=float)
        lat = np.asarray(lat_vals, dtype=float)
        valid = np.isfinite(lon) & np.isfinite(lat)
        if not valid.any():
            return

        lon = lon[valid]
        lat = lat[valid]
        lon = ((lon - central_longitude + 180.0) % 360.0) - 180.0 + central_longitude

        split_idx = np.where(np.abs(np.diff(lon)) > 180.0)[0] + 1
        lon_segments = np.split(lon, split_idx)
        lat_segments = np.split(lat, split_idx)
        for lon_seg, lat_seg in zip(lon_segments, lat_segments, strict=False):
            if len(lon_seg) < 2:
                continue
            ax.plot(
                lon_seg,
                lat_seg,
                transform=ccrs.PlateCarree(),
                **plot_kwargs,
            )

    base_exp, final_exp = EXPERIMENT_PAIRS[0]
    teaser_cfgs = [
        {
            "title": "Trans-Atlantic",
            "projection": ccrs.PlateCarree(central_longitude=-40),
            "central_longitude": -40,
            "extent": [-80, 15, 25, 65],
            "case_id": "AO_WPS",
            "gc_case": "AGC_WPS",
        },
        {
            "title": "Trans-Pacific",
            "projection": ccrs.PlateCarree(central_longitude=180),
            "central_longitude": 180,
            "extent": [115, 250, 20, 65],
            "case_id": "PO_WPS",
            "gc_case": "PGC_WPS",
        },
    ]

    route_triplets: list[dict[str, object]] = []
    for cfg in teaser_cfgs:
        triplet = _load_representative_route_triplet(
            cfg["case_id"], cfg["gc_case"], base_exp, final_exp, paths
        )
        if triplet is None:
            print(
                f"  [!] Missing representative tracks for {cfg['title']}; "
                "skipping fig00"
            )
            return
        route_triplets.append(triplet)

    with mpl.rc_context({"figure.constrained_layout.use": False}):
        fig = plt.figure(figsize=(11, 4.5), facecolor="#FAFAF7")
        fig.suptitle(
            "Two-stage weather routing: from coarse search to refined optimal tracks",
            fontsize=11.5,
            fontweight="bold",
            x=0.02,
            ha="left",
        )

        axes = []
        for i, cfg in enumerate(teaser_cfgs, start=1):
            ax = fig.add_subplot(1, 2, i, projection=cfg["projection"])
            axes.append(ax)
            ax.set_extent(cfg["extent"], crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.OCEAN, facecolor="#EFF5FF", zorder=0)
            ax.add_feature(cfeature.LAND, facecolor="#D9D0C3", zorder=1)
            ax.add_feature(
                cfeature.COASTLINE,
                linewidth=0.45,
                edgecolor="#808080",
                zorder=2,
            )
            ax.gridlines(
                draw_labels=False,
                linewidth=0.35,
                color="#C8C8C8",
                x_inline=False,
                y_inline=False,
            )
            ax.set_title(cfg["title"], fontsize=10.5, fontweight="bold", pad=6)

        GC_COLOR = "#7A7A7A"
        CMA_COLOR = "#FF8C42"
        BERS_COLOR = "#1C5DAA"

        for ax, cfg, route_data in zip(axes, teaser_cfgs, route_triplets, strict=False):
            gc_trk = route_data["gc_track"]
            base_trk = route_data["base_track"]
            final_trk = route_data["final_track"]
            gain_cma_vs_gc_pct = route_data["gain_cma_vs_gc_pct"]
            gain_bers_vs_gc_pct = route_data["gain_bers_vs_gc_pct"]
            base_wind = route_data["base_wind"]
            base_wave = route_data["base_wave"]
            final_wind = route_data["final_wind"]
            final_wave = route_data["final_wave"]

            _plot_wrapped_track(
                ax,
                gc_trk["lon_deg"].to_numpy(),
                gc_trk["lat_deg"].to_numpy(),
                central_longitude=cfg["central_longitude"],
                color=GC_COLOR,
                linewidth=1.3,
                linestyle="--",
                alpha=0.95,
                zorder=3,
            )
            _plot_wrapped_track(
                ax,
                base_trk["lon_deg"].to_numpy(),
                base_trk["lat_deg"].to_numpy(),
                central_longitude=cfg["central_longitude"],
                color=CMA_COLOR,
                linewidth=1.7,
                alpha=0.9,
                zorder=4,
            )
            _plot_wrapped_track(
                ax,
                final_trk["lon_deg"].to_numpy(),
                final_trk["lat_deg"].to_numpy(),
                central_longitude=cfg["central_longitude"],
                color=BERS_COLOR,
                linewidth=2.1,
                alpha=0.95,
                zorder=5,
            )

            ax.text(
                0.02,
                0.03,
                (
                    f"CMA-ES gain vs GC: {gain_cma_vs_gc_pct:+.1f}%\n"
                    f"BERS gain vs GC: {gain_bers_vs_gc_pct:+.1f}%"
                ),
                transform=ax.transAxes,
                fontsize=7.0,
                color="#555555",
                ha="left",
                va="bottom",
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "fc": "white",
                    "ec": "none",
                    "alpha": 0.7,
                },
                zorder=6,
            )
            ax.text(
                0.98,
                0.03,
                (
                    "Max met-ocean along route\n"
                    "Wind speed (m/s): "
                    f"CMA-ES {base_wind:.1f} | BERS {final_wind:.1f}\n"
                    "Wave height Hs (m): "
                    f"CMA-ES {base_wave:.1f} | BERS {final_wave:.1f}"
                ),
                transform=ax.transAxes,
                fontsize=6.6,
                color="#4D4D4D",
                ha="right",
                va="bottom",
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "fc": "white",
                    "ec": "none",
                    "alpha": 0.75,
                },
                zorder=6,
            )

        flow_note = (
            "CMA-ES coarse search  \\N{RIGHTWARDS ARROW}  "
            "FMS refinement  \\N{RIGHTWARDS ARROW}  final route"
        )
        fig.text(
            0.5,
            0.03,
            flow_note,
            ha="center",
            va="center",
            fontsize=8.2,
            color="#4D4D4D",
            bbox={
                "boxstyle": "round,pad=0.3",
                "fc": "white",
                "ec": "#D4D4D4",
                "alpha": 0.85,
            },
        )

        legend_elements = [
            mlines.Line2D(
                [],
                [],
                color=GC_COLOR,
                linestyle="--",
                linewidth=1.4,
                label="Great-circle",
            ),
            mlines.Line2D([], [], color=CMA_COLOR, linewidth=1.8, label="CMA-ES"),
            mlines.Line2D(
                [],
                [],
                color=BERS_COLOR,
                linewidth=2.2,
                label="BERS (final)",
            ),
        ]
        fig.legend(
            handles=legend_elements,
            loc="lower center",
            ncol=3,
            bbox_to_anchor=(0.5, -0.035),
            fontsize=8.2,
        )

        add_source_note(fig)
        fig.tight_layout(rect=[0, 0.08, 1, 0.9])
        out = paths.figs_dir / "fig00_teaser_routes.pdf"
        _save_subplot_outputs(
            fig,
            out,
            axes,
            ["atlantic", "pacific"],
            bbox_inches="tight",
        )
        _save_figure_outputs(fig, out, bbox_inches="tight")
        print(f"  Saved {out.name}")
        plt.close(fig)

    # Also export seasonal teaser scenarios: 3 best-vs-GC examples per ocean.
    fig_teaser_seasonal_scenarios(paths)


# ===========================================================================
# FIGURE 7 — Route maps
# ===========================================================================
def fig_route_maps(paths: AnalysisPaths = DEFAULT_PATHS) -> None:
    """Geographic maps showing all BERS routes and seasonal mean corridors."""
    setup_style()

    def _wrap_longitudes(
        lon_vals: np.ndarray,
        central_longitude: float,
    ) -> np.ndarray:
        """Wrap longitudes to the plotting frame centred on *central_longitude*."""
        lon = np.asarray(lon_vals, dtype=float)
        return ((lon - central_longitude + 180.0) % 360.0) - 180.0 + central_longitude

    def _plot_wrapped_track(
        ax: plt.Axes,
        lon_vals: np.ndarray,
        lat_vals: np.ndarray,
        *,
        central_longitude: float,
        **plot_kwargs: object,
    ) -> None:
        """Plot a track split at antimeridian crossings to avoid long wrap lines."""
        lon = np.asarray(lon_vals, dtype=float)
        lat = np.asarray(lat_vals, dtype=float)
        valid = np.isfinite(lon) & np.isfinite(lat)
        if not valid.any():
            return

        lon = lon[valid]
        lat = lat[valid]
        lon = _wrap_longitudes(lon, central_longitude)

        split_idx = np.where(np.abs(np.diff(lon)) > 180.0)[0] + 1
        lon_segments = np.split(lon, split_idx)
        lat_segments = np.split(lat, split_idx)

        for lon_seg, lat_seg in zip(lon_segments, lat_segments, strict=False):
            if len(lon_seg) < 2:
                continue
            ax.plot(
                lon_seg,
                lat_seg,
                transform=ccrs.PlateCarree(),
                **plot_kwargs,
            )

    def _load_all_bers_tracks(
        case_id: str,
        central_longitude: float,
    ) -> list[dict[str, object]]:
        """Load all available final-route tracks for one optimised case."""
        final_exp = EXPERIMENT_PAIRS[0][1]
        summary = load_summary_csv(final_exp, case_id, paths)
        if summary is None:
            return []

        tracks_dir = paths.output_dir / _experiment_folder(final_exp, paths) / "tracks"
        routes: list[dict[str, object]] = []
        for _, row in summary.sort_values("departure_time_utc").iterrows():
            fpath = tracks_dir / row["details_filename"]
            if not fpath.exists():
                continue
            trk = pd.read_csv(fpath, parse_dates=["time_utc"])
            routes.append(
                {
                    "season": row["season"],
                    "lon": _wrap_longitudes(
                        trk["lon_deg"].to_numpy(),
                        central_longitude,
                    ),
                    "lat": trk["lat_deg"].to_numpy(dtype=float),
                }
            )
        return routes

    def _load_best_bers_routes_by_month(
        case_id: str,
        gc_case: str,
        central_longitude: float,
    ) -> list[dict[str, object]]:
        """Load the best BERS route for each month, ranked by saving vs GC."""
        base_exp, final_exp = EXPERIMENT_PAIRS[0]
        summary = load_summary_csv(final_exp, case_id, paths)
        if summary is None:
            return []

        gc_path = _summary_csv_path(paths, _experiment_folder(base_exp, paths), gc_case)
        if gc_path is None or not gc_path.exists():
            return []
        gc_summary = pd.read_csv(gc_path, parse_dates=["departure_time_utc"])

        merged = summary.merge(
            gc_summary[["departure_time_utc", "energy_cons_mwh"]].rename(
                columns={"energy_cons_mwh": "gc_energy_cons_mwh"}
            ),
            on="departure_time_utc",
            how="inner",
        )
        if merged.empty:
            return []

        merged["gain_bers_vs_gc_pct"] = (
            (merged["gc_energy_cons_mwh"] - merged["energy_cons_mwh"])
            / merged["gc_energy_cons_mwh"]
            * 100
        )

        tracks_dir = paths.output_dir / _experiment_folder(final_exp, paths) / "tracks"
        routes: list[dict[str, object]] = []
        for month in range(1, 13):
            month_rows = merged[merged["month"] == month].sort_values(
                "gain_bers_vs_gc_pct",
                ascending=False,
            )
            if month_rows.empty:
                continue
            row = month_rows.iloc[0]
            fpath = tracks_dir / row["details_filename"]
            if not fpath.exists():
                continue
            trk = pd.read_csv(fpath, parse_dates=["time_utc"])
            routes.append(
                {
                    "month": int(month),
                    "season": row["season"],
                    "gain_bers_vs_gc_pct": float(row["gain_bers_vs_gc_pct"]),
                    "lon": _wrap_longitudes(
                        trk["lon_deg"].to_numpy(),
                        central_longitude,
                    ),
                    "lat": trk["lat_deg"].to_numpy(dtype=float),
                }
            )
        return routes

    def _load_gc_reference_track(
        gc_case: str,
        central_longitude: float,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Load one great-circle track to use as the route baseline."""
        base_exp = EXPERIMENT_PAIRS[0][0]
        summary_path = _summary_csv_path(
            paths,
            _experiment_folder(base_exp, paths),
            gc_case,
        )
        if summary_path is None or not summary_path.exists():
            return None

        summary = pd.read_csv(summary_path, parse_dates=["departure_time_utc"])
        if summary.empty:
            return None

        tracks_dir = paths.output_dir / _experiment_folder(base_exp, paths) / "tracks"
        details_name = summary.sort_values("departure_time_utc").iloc[0][
            "details_filename"
        ]
        fpath = tracks_dir / details_name
        if not fpath.exists():
            return None

        trk = pd.read_csv(fpath, parse_dates=["time_utc"])
        return (
            _wrap_longitudes(trk["lon_deg"].to_numpy(), central_longitude),
            trk["lat_deg"].to_numpy(dtype=float),
        )

    def _resample_track(
        lon_vals: np.ndarray,
        lat_vals: np.ndarray,
        n_points: int = 240,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Resample one route to a fixed number of points along cumulative length."""
        lon = np.asarray(lon_vals, dtype=float)
        lat = np.asarray(lat_vals, dtype=float)
        valid = np.isfinite(lon) & np.isfinite(lat)
        lon = lon[valid]
        lat = lat[valid]
        if len(lon) == 0:
            return np.array([]), np.array([])
        if len(lon) == 1:
            return np.full(n_points, lon[0]), np.full(n_points, lat[0])

        seg = np.hypot(np.diff(lon), np.diff(lat))
        dist = np.concatenate([[0.0], np.cumsum(seg)])
        if dist[-1] <= 0:
            return np.full(n_points, lon[0]), np.full(n_points, lat[0])

        target = np.linspace(0.0, dist[-1], n_points)
        return np.interp(target, dist, lon), np.interp(target, dist, lat)

    def _seasonal_route_stats(
        routes: list[dict[str, object]],
        season: str,
        n_points: int = 240,
    ) -> dict[str, object] | None:
        """Return mean route and percentile envelope for one season."""
        seasonal = [route for route in routes if route["season"] == season]
        if not seasonal:
            return None

        lon_stack = []
        lat_stack = []
        for route in seasonal:
            lon_res, lat_res = _resample_track(route["lon"], route["lat"], n_points)
            if len(lon_res) == 0:
                continue
            lon_stack.append(lon_res)
            lat_stack.append(lat_res)
        if not lon_stack:
            return None

        lon_arr = np.vstack(lon_stack)
        lat_arr = np.vstack(lat_stack)
        return {
            "season": season,
            "lon_mean": lon_arr.mean(axis=0),
            "lat_mean": lat_arr.mean(axis=0),
            "lat_p10": np.percentile(lat_arr, 10, axis=0),
            "lat_p90": np.percentile(lat_arr, 90, axis=0),
            "count": len(lon_stack),
        }

    def _setup_map(ax: plt.Axes, cfg: dict[str, object]) -> None:
        """Apply shared cartographic styling."""
        ax.set_extent(cfg["extent"], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor="#D9D0C3", zorder=1)
        ax.add_feature(cfeature.OCEAN, facecolor="#EFF5FF", zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="#7D7D7D", zorder=2)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="#BBBBBB", zorder=2)
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

    route_configs = [
        {
            "title": "Trans-Atlantic (Santander → New York)",
            "case_id": "AO_WPS",
            "gc_case": "AGC_WPS",
            "central_longitude": -40.0,
            "extent": [-80, 15, 25, 65],
            "projection": ccrs.PlateCarree(central_longitude=-40),
        },
        {
            "title": "Trans-Pacific (Tokyo → Los Angeles)",
            "case_id": "PO_WPS",
            "gc_case": "PGC_WPS",
            "central_longitude": 180.0,
            "extent": [115, 250, 20, 65],
            "projection": ccrs.PlateCarree(central_longitude=180),
        },
    ]
    season_colors = {
        "Winter": "#1C5DAA",
        "Spring": "#6DC201",
        "Summer": "#F23333",
        "Autumn": "#FF8C42",
    }

    route_data = {
        cfg["case_id"]: _load_all_bers_tracks(
            str(cfg["case_id"]),
            float(cfg["central_longitude"]),
        )
        for cfg in route_configs
    }

    # Cartopy/constrained_layout are incompatible — disable for this figure
    with mpl.rc_context({"figure.constrained_layout.use": False}):
        fig_all = plt.figure(figsize=(14, 6), facecolor="#FAFAF7")
        fig_all.suptitle(
            "Best BERS route of each month, coloured by departure season",
            fontsize=12,
            fontweight="bold",
            x=0.02,
            ha="left",
        )
        axes_all = [
            fig_all.add_subplot(1, 2, 1, projection=route_configs[0]["projection"]),
            fig_all.add_subplot(1, 2, 2, projection=route_configs[1]["projection"]),
        ]

        for ax, cfg in zip(axes_all, route_configs, strict=False):
            _setup_map(ax, cfg)
            gc_track = _load_gc_reference_track(
                str(cfg["gc_case"]),
                float(cfg["central_longitude"]),
            )
            if gc_track is not None:
                _plot_wrapped_track(
                    ax,
                    gc_track[0],
                    gc_track[1],
                    central_longitude=float(cfg["central_longitude"]),
                    color="#555555",
                    linewidth=1.6,
                    linestyle="--",
                    alpha=0.95,
                    zorder=2,
                )
            routes = _load_best_bers_routes_by_month(
                str(cfg["case_id"]),
                str(cfg["gc_case"]),
                float(cfg["central_longitude"]),
            )
            for route in routes:
                season = str(route["season"])
                _plot_wrapped_track(
                    ax,
                    route["lon"],
                    route["lat"],
                    central_longitude=float(cfg["central_longitude"]),
                    color=season_colors[season],
                    linewidth=1.2,
                    linestyle="-",
                    alpha=0.9,
                    zorder=3,
                )
            ax.text(
                0.02,
                0.02,
                f"{len(routes)} monthly best routes",
                transform=ax.transAxes,
                fontsize=7.5,
                color="#4D4D4D",
                ha="left",
                va="bottom",
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "fc": "white",
                    "ec": "none",
                    "alpha": 0.72,
                },
            )

        season_handles = [
            mlines.Line2D(
                [],
                [],
                color="#555555",
                linewidth=1.6,
                linestyle="--",
                label="Great-circle",
            ),
        ] + [
            mlines.Line2D([], [], color=season_colors[s], linewidth=2, label=s)
            for s in SEASON_ORDER
        ]
        fig_all.legend(
            handles=season_handles,
            loc="lower center",
            ncol=5,
            bbox_to_anchor=(0.5, -0.01),
            fontsize=8.5,
        )
        add_source_note(fig_all)
        fig_all.tight_layout(rect=[0, 0.05, 1, 0.93])
        out_all = paths.figs_dir / "fig07a_bers_routes_monthly_best.pdf"
        _save_subplot_outputs(
            fig_all,
            out_all,
            axes_all,
            ["atlantic", "pacific"],
            bbox_inches="tight",
        )
        _save_figure_outputs(fig_all, out_all, bbox_inches="tight")
        print(f"  Saved {out_all.name}")
        plt.close(fig_all)

        fig_avg = plt.figure(figsize=(14, 6), facecolor="#FAFAF7")
        fig_avg.suptitle(
            "Seasonal mean BERS routes with 10–90 percentile corridor",
            fontsize=12,
            fontweight="bold",
            x=0.02,
            ha="left",
        )
        axes_avg = [
            fig_avg.add_subplot(1, 2, 1, projection=route_configs[0]["projection"]),
            fig_avg.add_subplot(1, 2, 2, projection=route_configs[1]["projection"]),
        ]

        for ax, cfg in zip(axes_avg, route_configs, strict=False):
            _setup_map(ax, cfg)
            gc_track = _load_gc_reference_track(
                str(cfg["gc_case"]),
                float(cfg["central_longitude"]),
            )
            if gc_track is not None:
                _plot_wrapped_track(
                    ax,
                    gc_track[0],
                    gc_track[1],
                    central_longitude=float(cfg["central_longitude"]),
                    color="#555555",
                    linewidth=1.6,
                    linestyle="--",
                    alpha=0.95,
                    zorder=2,
                )
            routes = route_data[str(cfg["case_id"])]
            for season in SEASON_ORDER:
                stats = _seasonal_route_stats(routes, season)
                if stats is None:
                    continue
                ax.fill_between(
                    stats["lon_mean"],
                    stats["lat_p10"],
                    stats["lat_p90"],
                    transform=ccrs.PlateCarree(),
                    color=season_colors[season],
                    alpha=0.14,
                    zorder=3,
                )
                _plot_wrapped_track(
                    ax,
                    stats["lon_mean"],
                    stats["lat_mean"],
                    central_longitude=float(cfg["central_longitude"]),
                    color=season_colors[season],
                    linewidth=2.2,
                    linestyle="-",
                    alpha=0.95,
                    zorder=4,
                )

        mean_handles = [
            mlines.Line2D(
                [],
                [],
                color="#555555",
                linewidth=1.6,
                linestyle="--",
                label="Great-circle",
            ),
        ] + [
            mlines.Line2D([], [], color=season_colors[s], linewidth=2.2, label=s)
            for s in SEASON_ORDER
        ]
        mean_handles.append(
            mpatches.Patch(
                facecolor="#888888",
                alpha=0.14,
                label="10-90 percentile band",
            )
        )
        fig_avg.legend(
            handles=mean_handles,
            loc="lower center",
            ncol=5,
            bbox_to_anchor=(0.5, -0.01),
            fontsize=8.5,
        )
        add_source_note(fig_avg)
        fig_avg.tight_layout(rect=[0, 0.05, 1, 0.93])
        out_avg = paths.figs_dir / "fig07b_bers_routes_average.pdf"
        _save_subplot_outputs(
            fig_avg,
            out_avg,
            axes_avg,
            ["atlantic", "pacific"],
            bbox_inches="tight",
        )
        _save_figure_outputs(fig_avg, out_avg, bbox_inches="tight")
        print(f"  Saved {out_avg.name}")
        plt.close(fig_avg)


# ===========================================================================
# FIGURE 8 — Risk calendar (heatmap of violation rate)
# ===========================================================================
def fig_risk_calendar(
    df: pd.DataFrame,
    paths: AnalysisPaths = DEFAULT_PATHS,
) -> None:
    """Heatmap: departure month × case × experiment — violation rate."""
    setup_style()

    cases_order = list(OPT_CASES.keys())
    months = np.arange(1, 13)
    n_exp = len(ACTIVE_EXPERIMENTS)

    fig, axes = plt.subplots(
        2,
        n_exp,
        figsize=(6.5 * n_exp, 7),
        gridspec_kw={"height_ratios": [1, 1]},
        squeeze=False,
    )
    fig.suptitle(
        "Experiment comparison — monthly weather violation rates",  # noqa: E501
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
        for col_idx, exp_key in enumerate(ACTIVE_EXPERIMENTS):
            ax = axes[row_idx][col_idx]
            exp_label = ACTIVE_EXPERIMENTS[exp_key]["label"]
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
    out = paths.figs_dir / "fig08_risk_calendar.pdf"
    exp_keys = list(ACTIVE_EXPERIMENTS.keys())
    _save_subplot_outputs(
        fig,
        out,
        list(axes.flat),
        [
            *[f"wind_violations_{key}" for key in exp_keys],
            *[f"wave_violations_{key}" for key in exp_keys],
        ],
    )
    _save_figure_outputs(fig, out)
    print(f"  Saved {out.name}")
    plt.close(fig)


# ===========================================================================
# SUMMARY TABLE
# ===========================================================================
def generate_summary_table(
    df: pd.DataFrame,
    gc: dict[str, float],
    paths: AnalysisPaths = DEFAULT_PATHS,
) -> pd.DataFrame:
    """Generate and save a summary statistics table."""
    rows = []
    for exp_key in ACTIVE_EXPERIMENTS:
        for case_id in OPT_CASES:
            piece = df[(df["experiment"] == exp_key) & (df["case_id"] == case_id)]
            if piece.empty:
                continue
            gc_id = OPT_CASES[case_id]["gc"]
            gc_mean = gc.get(gc_id, np.nan)
            mean_e = piece["energy_cons_mwh"].mean()
            rows.append(
                {
                    "Experiment": ACTIVE_EXPERIMENTS[exp_key]["label"],
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
    out = paths.figs_dir / "table01_summary.csv"
    summary.to_csv(out, index=False)
    print(f"  Saved {out.name}")
    return summary


# ===========================================================================
# BONUS: FMS delta plot — per-voyage improvement
# ===========================================================================
def fig_fms_delta_byseason(
    df: pd.DataFrame,
    paths: AnalysisPaths = DEFAULT_PATHS,
) -> None:
    """Bar chart: median FMS improvement (%) by season and by case."""
    setup_style()

    pairs = list(EXPERIMENT_PAIRS)
    n_pairs = len(pairs)
    fig, _axes_arr = plt.subplots(1, n_pairs, figsize=(8 * n_pairs, 5), squeeze=False)
    axes = list(_axes_arr.flat)
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
            f"{ACTIVE_EXPERIMENTS[base_exp]['label']}"
            f" vs {ACTIVE_EXPERIMENTS[fms_exp]['label']}",
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

    # Equalise y-axis (single panel)
    ymin = axes[0].get_ylim()[0]
    ymax = axes[0].get_ylim()[1]
    axes[0].set_ylim(ymin, ymax)

    add_source_note(fig)
    out = paths.figs_dir / "fig09_fms_seasonal_delta.pdf"
    _save_subplot_outputs(
        fig,
        out,
        axes,
        [f"{base_exp}_vs_{fms_exp}" for base_exp, fms_exp in pairs],
    )
    _save_figure_outputs(fig, out)
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


def load_gc_full(paths: AnalysisPaths = DEFAULT_PATHS) -> pd.DataFrame:
    """Load per-departure GC rows and map them onto optimised-case ids.

    GC summaries are read from the first base experiment folder because that run
    contains the reference great-circle exports for all four cases.
    """
    folder = _experiment_folder(EXPERIMENT_PAIRS[0][0], paths)
    frames = []
    for gc_id, opt_id in _GC_TO_OPT.items():
        path = _summary_csv_path(paths, folder, gc_id)
        if path is None or not path.exists():
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
def fig_gc_victory_rate(
    df: pd.DataFrame,
    gc_full: pd.DataFrame,
    paths: AnalysisPaths = DEFAULT_PATHS,
) -> None:
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

        for exp_key in ACTIVE_EXPERIMENTS:
            joined = _join_opt_to_gc(df, gc_full, exp_key, case_id)
            if joined.empty:
                continue
            monthly_rate = (joined.groupby("month")["beats_gc"].mean() * 100).reindex(
                months
            )
            ax.plot(
                monthly_rate.index,
                monthly_rate.values,
                color=ACTIVE_EXPERIMENTS[exp_key]["color"],
                linewidth=2.0,
                marker="o",
                markersize=4,
                label=ACTIVE_EXPERIMENTS[exp_key]["label"],
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
            color=ACTIVE_EXPERIMENTS[k]["color"],
            linewidth=2,
            marker="o",
            markersize=4,
            label=ACTIVE_EXPERIMENTS[k]["label"],
        )
        for k in ACTIVE_EXPERIMENTS
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
    out = paths.figs_dir / "fig10_gc_victory_rate.pdf"
    _save_subplot_outputs(
        fig,
        out,
        list(axes.flat),
        [_CASE_FILE_SUFFIX[case_id] for case_id, _ in cases_order],
    )
    _save_figure_outputs(fig, out)
    print(f"  Saved {out.name}")
    plt.close(fig)


# ===========================================================================
# FIGURE 11 — Margin-over-GC heatmap
# ===========================================================================
def fig_gc_margin_heatmap(
    df: pd.DataFrame,
    gc_full: pd.DataFrame,
    paths: AnalysisPaths = DEFAULT_PATHS,
) -> None:
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
        for exp_key in ACTIVE_EXPERIMENTS:
            joined = _join_opt_to_gc(df, gc_full, exp_key, case_id)
            if not joined.empty:
                all_margins.extend(joined["margin_pct"].dropna().tolist())
    vabs = np.nanpercentile(all_margins, 95)  # upper bound

    for ax, (case_id, panel_title) in zip(axes.flat, cases_order, strict=False):
        ax.set_title(panel_title, fontsize=10, fontweight="bold")

        matrix = np.full((len(ACTIVE_EXPERIMENTS), 12), np.nan)
        exp_labels = []
        for i, exp_key in enumerate(ACTIVE_EXPERIMENTS):
            exp_labels.append(ACTIVE_EXPERIMENTS[exp_key]["label"])
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
        for i in range(len(ACTIVE_EXPERIMENTS)):
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
        ax.set_yticks(np.arange(len(ACTIVE_EXPERIMENTS)))
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
    out = paths.figs_dir / "fig11_gc_margin_heatmap.pdf"
    _save_subplot_outputs(
        fig,
        out,
        list(axes.flat),
        [_CASE_FILE_SUFFIX[case_id] for case_id, _ in cases_order],
    )
    _save_figure_outputs(fig, out)
    print(f"  Saved {out.name}")
    plt.close(fig)


# ===========================================================================
# FIGURE 12 — GC's own violations: the unfair baseline
# ===========================================================================
def fig_gc_violations(
    df: pd.DataFrame,
    gc_full: pd.DataFrame,
    paths: AnalysisPaths = DEFAULT_PATHS,
) -> None:
    """Monthly 'any violation' rate — GC vs best optimised (Penalty + FMS)."""
    setup_style()
    best_exp = EXPERIMENT_PAIRS[-1][1]
    GC_COLOR = "#878787"
    OPT_COLOR = ACTIVE_EXPERIMENTS[best_exp]["color"]

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
            label=ACTIVE_EXPERIMENTS[best_exp]["label"],
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
    out = paths.figs_dir / "fig12_gc_violations.pdf"
    _save_subplot_outputs(
        fig,
        out,
        list(axes.flat),
        [_CASE_FILE_SUFFIX[case_id] for case_id, _ in cases_order],
    )
    _save_figure_outputs(fig, out)
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
        help="Figure numbers to generate, e.g. --figures 0 1 5 10. Generates all if omitted.",  # noqa: E501
    )
    return p.parse_args()


# ===========================================================================
# MAIN
# ===========================================================================
def main() -> None:
    """Load datasets once, then generate the requested figures and summary."""
    args = parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)
    paths = AnalysisPaths(
        output_dir=args.data_dir
        if args.data_dir is not None
        else DEFAULT_PATHS.output_dir,
        figs_dir=args.output_dir
        if args.output_dir is not None
        else DEFAULT_PATHS.figs_dir,
        config_path=DEFAULT_PATHS.config_path,
    )

    paths.figs_dir.mkdir(parents=True, exist_ok=True)

    # Apply DPI setting
    import matplotlib as mpl  # noqa: PLC0415 — local import fine here

    mpl.rcParams["savefig.dpi"] = args.dpi

    want = set(args.figures) if args.figures else None

    def _want(n: int) -> bool:
        return want is None or n in want

    print("Loading data…")
    # Keep shared datasets in memory once so each figure function can focus on
    # presentation instead of repeating the same I/O and alignment work.
    gc_baselines = load_gc_baselines(paths)
    df = load_all_data(paths)
    gc_full = load_gc_full(paths)
    print(
        f"  Loaded {len(df):,} voyage records across "
        f"{df['experiment'].nunique()} experiments and {df['case_id'].nunique()} cases."
    )

    print("\nGenerating figures…")
    if _want(0):
        fig_teaser_routes(paths)
    if _want(1):
        fig_energy_overview(df, gc_baselines, gc_full, paths)
    if _want(2):
        fig_optimization_gains(df, gc_baselines, paths)
    if _want(3):
        fig_penalty_tradeoff(df, paths)
    if _want(4):
        fig_seasonality_a(df, gc_full, paths)
        fig_seasonality_b(df, gc_full, paths)
    if _want(5):
        fig_wps_impact(df, paths)
    if _want(6):
        fig_fms_improvement(df, paths)
    if _want(7):
        fig_route_maps(paths)
    if _want(8):
        fig_risk_calendar(df, paths)
    if _want(9):
        fig_fms_delta_byseason(df, paths)
    if _want(10):
        fig_gc_victory_rate(df, gc_full, paths)
    if _want(11):
        fig_gc_margin_heatmap(df, gc_full, paths)
    if _want(12):
        fig_gc_violations(df, gc_full, paths)
    if _want(13):
        fig_relative_gain_a(df, gc_full, paths)
        fig_relative_gain_b(df, gc_full, paths)

    print("\nGenerating summary table…")
    summary = generate_summary_table(df, gc_baselines, paths)
    print(summary.to_string(index=False))

    print(f"\nAll outputs saved to {paths.figs_dir}/")


if __name__ == "__main__":
    main()
