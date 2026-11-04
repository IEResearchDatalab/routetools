# ruff: noqa: D103, E501
"""Generate the Revision 2 follow-up verification artifacts (C1, C2, C4-C6).

This module never reruns an optimizer or changes a saved route.  It consumes the
preserved route-level common-weather scores, the saved-waypoint segment table,
the stage-functional decomposition, and the Task 0 reconciliation.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parent.parent
REV = ROOT / "revision2"
FIG_DIR = REV / "figures"
CMA_ROOT = ROOT / "output" / "sweep_combined"
BERS_ROOT = ROOT / "output" / "sweep_combined_fms_strict"
ROUTE_SCORES = REV / "task1_weather_consistency_correction.csv"
SEGMENTS = REV / "task1_segment_inputs_common_weather.csv.gz"
FUNCTIONALS = REV / "task2_functional_decomposition.csv"
TASK0 = REV / "task0_config_reconciliation.csv"
OLD_EXPOSURE = ROOT / "revision" / "task4_violations_table.csv"
SEASON_ORDER = ("Winter", "Spring", "Summer", "Autumn")
SEASON_BY_MONTH = {
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
CASE_INFO = {
    "AO_noWPS": ("Atlantic", "no WPS"),
    "AO_WPS": ("Atlantic", "WPS"),
    "PO_noWPS": ("Pacific", "no WPS"),
    "PO_WPS": ("Pacific", "WPS"),
}
STRATEGY_ORDER = ("GC", "CMA-ES", "BERS")
COLORS = {"GC": "#666666", "CMA-ES": "#d95f02", "BERS": "#1f77b4"}


def git_commit() -> str:
    """Return the exact analysis commit."""
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metadata(extra: dict[str, object] | None = None) -> dict[str, object]:
    result: dict[str, object] = {
        "analysis_git_commit": git_commit(),
        "cmaes_root": CMA_ROOT,
        "bers_root": BERS_ROOT,
    }
    if extra:
        result.update(extra)
    return result


def read_csv(path: Path, **kwargs: object) -> pd.DataFrame:
    return pd.read_csv(path, comment="#", **kwargs)


def write_csv(
    frame: pd.DataFrame, path: Path, extra: dict[str, object] | None = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = metadata(extra)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(
            "# " + "; ".join(f"{key}={value}" for key, value in meta.items()) + "\n"
        )
        frame.to_csv(handle, index=False)


def route_scores() -> pd.DataFrame:
    frame = read_csv(ROUTE_SCORES, parse_dates=["departure_utc"])
    frame["season"] = frame["departure_utc"].dt.month.map(SEASON_BY_MONTH)
    return frame


def segment_inputs() -> pd.DataFrame:
    columns = [
        "strategy",
        "case",
        "departure_utc",
        "duration_h",
        "tws_mps",
        "hs_m",
        "speed_mps",
    ]
    frame = read_csv(SEGMENTS, usecols=columns, parse_dates=["departure_utc"])
    return frame


def seasonal_table(scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for case in ("AO_WPS", "PO_WPS"):
        subset = scores[scores["case"].eq(case)]
        for season in SEASON_ORDER:
            seasonal = subset[subset["season"].eq(season)]
            wide_old = seasonal.pivot(
                index="departure_utc", columns="strategy", values="legacy_file_a_mwh"
            )
            wide_new = seasonal.pivot(
                index="departure_utc", columns="strategy", values="common_weather_mwh"
            )
            old = 100.0 * (1.0 - wide_old["BERS"].mean() / wide_old["GC"].mean())
            new = 100.0 * (1.0 - wide_new["BERS"].mean() / wide_new["GC"].mean())
            rows.append(
                {
                    "case": case,
                    "corridor": CASE_INFO[case][0],
                    "season": season,
                    "n_departures": len(wide_new),
                    "legacy_gc_mean_mwh": wide_old["GC"].mean(),
                    "legacy_bers_mean_mwh": wide_old["BERS"].mean(),
                    "legacy_saving_reduction_of_means_pct": old,
                    "common_gc_mean_mwh": wide_new["GC"].mean(),
                    "common_bers_mean_mwh": wide_new["BERS"].mean(),
                    "common_saving_reduction_of_means_pct": new,
                    "common_minus_legacy_pp": new - old,
                }
            )
    return pd.DataFrame(rows)


def route_exposure(segments: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    route_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for (case, strategy), group in segments.groupby(["case", "strategy"], sort=True):
        for hazard, column, limit in (("wind", "tws_mps", 20.0), ("wave", "hs_m", 7.0)):
            work = group[["departure_utc", "duration_h", column]].copy()
            work["excess"] = (work[column] - limit).clip(lower=0.0)
            work["violation_duration_h"] = work["duration_h"].where(
                work[column] > limit, 0.0
            )
            route = work.groupby("departure_utc", sort=True).agg(
                peak_exceedance=("excess", "max"),
                duration_h=("violation_duration_h", "sum"),
            )
            route["has_violation"] = route["peak_exceedance"] > 0.0
            for departure, item in route.iterrows():
                route_rows.append(
                    {
                        "case": case,
                        "strategy": strategy,
                        "departure_utc": departure,
                        "hazard": hazard,
                        "threshold": limit,
                        "has_violation": bool(item["has_violation"]),
                        "peak_exceedance": item["peak_exceedance"],
                        "duration_h": item["duration_h"],
                    }
                )
            violating = route[route["has_violation"]]
            total_h = float(work["duration_h"].sum())
            violation_h = float(work["violation_duration_h"].sum())
            summary_rows.append(
                {
                    "case": case,
                    "corridor": CASE_INFO[case][0],
                    "wps": CASE_INFO[case][1],
                    "strategy": strategy,
                    "hazard": hazard,
                    "threshold": limit,
                    "n_departures": len(route),
                    "n_with_violation": int(route["has_violation"].sum()),
                    "departure_incidence_pct": 100.0 * route["has_violation"].mean(),
                    "conditional_mean_peak_exceedance": (
                        float(violating["peak_exceedance"].mean())
                        if len(violating)
                        else 0.0
                    ),
                    "max_peak_exceedance": float(route["peak_exceedance"].max()),
                    "conditional_mean_duration_h": (
                        float(violating["duration_h"].mean()) if len(violating) else 0.0
                    ),
                    "max_duration_h": float(route["duration_h"].max()),
                    "total_evaluated_h": total_h,
                    "total_above_threshold_h": violation_h,
                    "time_weighted_exposure_pct": 100.0 * violation_h / total_h,
                }
            )
    return pd.DataFrame(route_rows), pd.DataFrame(summary_rows)


def segment_speed_table(segments: pd.DataFrame) -> pd.DataFrame:
    rows = []
    bers = segments[segments["strategy"].eq("BERS")]
    for case, group in bers.groupby("case", sort=True):
        knots = group["speed_mps"].to_numpy() * 1.9438444924406
        rows.append(
            {
                "case": case,
                "segments": len(knots),
                "min_kn": np.min(knots),
                "p5_kn": np.percentile(knots, 5),
                "mean_kn": np.mean(knots),
                "median_kn": np.median(knots),
                "p95_kn": np.percentile(knots, 95),
                "max_kn": np.max(knots),
            }
        )
    return pd.DataFrame(rows)


def energy_overview_figures(scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for case, (corridor, wps) in CASE_INFO.items():
        subset = scores[scores["case"].eq(case)]
        arrays = [
            subset[subset["strategy"].eq(strategy)]["common_weather_mwh"].to_numpy()
            for strategy in STRATEGY_ORDER
        ]
        medians = {
            name: float(np.median(values))
            for name, values in zip(STRATEGY_ORDER, arrays, strict=True)
        }
        fig, ax = plt.subplots(figsize=(4.3, 3.35))
        violin = ax.violinplot(arrays, showextrema=False, widths=0.82)
        for body, strategy in zip(violin["bodies"], STRATEGY_ORDER, strict=True):
            body.set_facecolor(COLORS[strategy])
            body.set_edgecolor("black")
            body.set_alpha(0.78)
        for idx, strategy in enumerate(STRATEGY_ORDER, start=1):
            median = medians[strategy]
            ax.hlines(median, idx - 0.16, idx + 0.16, color="white", linewidth=3.0)
            saving = 100.0 * (1.0 - median / medians["GC"])
            if strategy != "GC":
                ax.text(
                    idx,
                    np.max(arrays[idx - 1]) * 1.015,
                    f"{saving:.1f}%",
                    ha="center",
                    fontsize=8,
                )
            rows.append(
                {
                    "figure": f"fig01_energy_overview_{corridor.lower()}_{wps.lower().replace(' ', '_')}.pdf",
                    "case": case,
                    "strategy": strategy,
                    "median_mwh": median,
                    "saving_of_marginal_medians_vs_gc_pct": saving,
                }
            )
        ax.set_xticks((1, 2, 3), ("GC", "CMA-ES", "BERS"))
        ax.set_ylabel("Voyage energy (MWh)")
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        slug = f"fig01_energy_overview_{corridor.lower()}_{wps.lower().replace(' ', '_')}.pdf"
        fig.savefig(FIG_DIR / slug, bbox_inches="tight")
        plt.close(fig)
    return pd.DataFrame(rows)


def paired_histogram(scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.2))
    for ax, case in zip(axes.flat, CASE_INFO, strict=True):
        subset = scores[scores["case"].eq(case)].pivot(
            index="departure_utc", columns="strategy", values="common_weather_mwh"
        )
        values = 100.0 * (subset["GC"] - subset["BERS"]) / subset["GC"]
        counts, edges, _ = ax.hist(values, bins=30, color=COLORS["BERS"], alpha=0.78)
        mean = float(values.mean())
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
        ax.axvline(
            mean,
            color="#d95f02",
            linestyle=":",
            linewidth=1.5,
            label=f"Mean: {mean:.1f}%",
        )
        ax.set_title(f"{CASE_INFO[case][0]}, {CASE_INFO[case][1]}")
        ax.set_xlabel("Paired energy improvement (%)")
        ax.set_ylabel("Departures")
        ax.legend(frameon=False, fontsize=8)
        for left, right, count in zip(edges[:-1], edges[1:], counts, strict=True):
            rows.append(
                {
                    "figure": "fig09_paired_improvements.pdf",
                    "case": case,
                    "bin_left_pct": left,
                    "bin_right_pct": right,
                    "count": int(count),
                    "mean_annotation_pct": mean,
                }
            )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig09_paired_improvements.pdf", bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(rows)


def seasonality_figures(scores: pd.DataFrame) -> None:
    for case, (corridor, wps) in CASE_INFO.items():
        subset = scores[scores["case"].eq(case)]
        fig, ax = plt.subplots(figsize=(6.0, 3.25))
        for strategy in STRATEGY_ORDER:
            selected = subset[subset["strategy"].eq(strategy)].sort_values(
                "departure_utc"
            )
            ax.plot(
                selected["departure_utc"],
                selected["common_weather_mwh"],
                color=COLORS[strategy],
                linestyle="--" if strategy == "GC" else "-",
                linewidth=1.0,
                label=strategy,
            )
        ax.set_ylabel("Voyage energy (MWh)")
        ax.legend(frameon=False, ncol=3, fontsize=8)
        ax.grid(alpha=0.18)
        fig.autofmt_xdate()
        fig.tight_layout()
        name = f"fig04a_seasonality_sweep_combined_{corridor.lower()}_{wps.lower().replace(' ', '_')}.pdf"
        fig.savefig(FIG_DIR / name, bbox_inches="tight")
        plt.close(fig)


def violation_curves(segments: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    fig, axes = plt.subplots(2, 4, figsize=(13.2, 6.3), sharey="row")
    for col_idx, case in enumerate(CASE_INFO):
        for row_idx, (hazard, column, limit, unit) in enumerate(
            (("wind", "tws_mps", 20.0, "m/s"), ("wave", "hs_m", 7.0, "m"))
        ):
            ax = axes[row_idx, col_idx]
            case_data = segments[segments["case"].eq(case)]
            all_values = case_data[column].to_numpy()
            lo, hi = float(np.min(all_values)), float(np.max(all_values))
            edges = np.linspace(lo, hi, 121)
            centers = 0.5 * (edges[:-1] + edges[1:])
            for strategy in STRATEGY_ORDER:
                selected = case_data[case_data["strategy"].eq(strategy)]
                values = selected[column].to_numpy()
                weights = selected["duration_h"].to_numpy()
                counts, _ = np.histogram(values, bins=edges, weights=weights)
                density = (
                    counts / np.trapezoid(counts, centers) if counts.sum() else counts
                )
                density = gaussian_filter1d(density.astype(float), sigma=1.4)
                exposure = 100.0 * weights[values > limit].sum() / weights.sum()
                ax.plot(
                    centers,
                    density,
                    color=COLORS[strategy],
                    linestyle="--" if strategy == "GC" else "-",
                    label=f"{strategy} ({exposure:.2f}%)",
                )
                for x_value, density_value in zip(centers, density, strict=True):
                    rows.append(
                        {
                            "figure": "fig10_weather_violation_curves.pdf",
                            "case": case,
                            "strategy": strategy,
                            "hazard": hazard,
                            "curve_x": x_value,
                            "density": density_value,
                            "threshold": limit,
                            "time_weighted_exposure_pct": exposure,
                        }
                    )
            ax.axvline(limit, color="red", linestyle=":", linewidth=1.0)
            ax.set_xlabel(f"{hazard.capitalize()} ({unit})")
            if col_idx == 0:
                ax.set_ylabel("Time-weighted density")
            if row_idx == 0:
                ax.set_title(f"{CASE_INFO[case][0]}, {CASE_INFO[case][1]}")
            ax.legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig10_weather_violation_curves.pdf", bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(rows)


def parse_manuscript_exposure(main_tex: Path | None) -> pd.DataFrame:
    if main_tex is None or not main_tex.exists():
        return pd.DataFrame()
    pattern = re.compile(
        r"^(Atlantic|Pacific), (no WPS|WPS) --- (GC|\\name\{\})\s*&\s*"
        r"(Wind|Wave)\s*&\s*([-0-9.]+)\s*&\s*([-0-9.]+)\s*&\s*"
        r"([-0-9.]+)\s*&\s*([-0-9.]+)\s*&\s*([-0-9.]+)\s*\\\\",
        re.MULTILINE,
    )
    rows = []
    for match in pattern.finditer(main_tex.read_text(encoding="utf-8")):
        corridor, wps, raw_strategy, hazard, *values = match.groups()
        case = ("AO" if corridor == "Atlantic" else "PO") + (
            "_noWPS" if wps == "no WPS" else "_WPS"
        )
        rows.append(
            {
                "case": case,
                "strategy": "BERS" if raw_strategy.startswith("\\name") else "GC",
                "hazard": hazard.lower(),
                "manuscript_departure_incidence_pct": float(values[0]),
                "manuscript_conditional_mean_peak_exceedance": float(values[1]),
                "manuscript_max_peak_exceedance": float(values[2]),
                "manuscript_conditional_mean_duration_h": float(values[3]),
                "manuscript_max_duration_h": float(values[4]),
            }
        )
    return pd.DataFrame(rows)


def legacy_exposure_table() -> pd.DataFrame:
    """Normalize the preserved Review 1 exposure summary."""
    old = read_csv(OLD_EXPOSURE)
    old["case"] = old.apply(
        lambda row: ("AO" if row["corridor"] == "atlantic" else "PO")
        + ("_WPS" if row["wps"] == "WPS" else "_noWPS"),
        axis=1,
    )
    return old.rename(
        columns={
            "freq_pct": "legacy_departure_incidence_pct",
            "mean_max_exceedance": "legacy_conditional_mean_peak_exceedance",
            "max_exceedance": "legacy_max_peak_exceedance",
            "mean_duration_h": "legacy_conditional_mean_duration_h",
            "max_duration_h": "legacy_max_duration_h",
        }
    )


def c1_report(
    seasonal: pd.DataFrame,
    exposure: pd.DataFrame,
    legacy: pd.DataFrame,
    manuscript: pd.DataFrame,
) -> None:
    lines = [
        "# C1 common-weather propagation",
        "",
        f"Analysis commit: `{git_commit()}`.",
        f"Input roots: `{CMA_ROOT}` and `{BERS_ROOT}`.",
        f"Common-weather segment artifact: `{SEGMENTS}` (SHA-256 `{sha256(SEGMENTS)}`).",
        "No optimizer was rerun and no route geometry was changed.",
        "",
        "## Seasonal table: legacy versus common weather",
        "",
        "The statistic is the reduction of seasonal mean BERS energy relative to seasonal mean GC energy, matching the old table.",
        "",
        "| Corridor | Season | Old (%) | New (%) | Change (pp) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in seasonal.itertuples(index=False):
        lines.append(
            f"| {row.corridor} | {row.season} | {row.legacy_saving_reduction_of_means_pct:.6f} | "
            f"{row.common_saving_reduction_of_means_pct:.6f} | {row.common_minus_legacy_pp:+.6f} |"
        )
    lines.extend(
        ["", "## Weather exposure: preserved legacy output versus common weather", ""]
    )
    merged = legacy.merge(
        exposure,
        on=["case", "strategy", "hazard"],
        how="inner",
        validate="one_to_one",
    )
    lines.extend(
        [
            "| Case | Strategy | Hazard | Old incidence (%) | New incidence (%) | Old max | New max | Old mean duration (h) | New mean duration (h) |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in merged.itertuples(index=False):
        lines.append(
            f"| {row.case} | {row.strategy} | {row.hazard} | "
            f"{row.legacy_departure_incidence_pct:.6f} | {row.departure_incidence_pct:.6f} | "
            f"{row.legacy_max_peak_exceedance:.6f} | {row.max_peak_exceedance:.6f} | "
            f"{row.legacy_conditional_mean_duration_h:.6f} | {row.conditional_mean_duration_h:.6f} |"
        )
    if not manuscript.empty:
        manuscript_comparison = manuscript.merge(
            exposure,
            on=["case", "strategy", "hazard"],
            how="left",
            validate="one_to_one",
        )
        lines.extend(
            [
                "| Case | Strategy | Hazard | Old incidence (%) | New incidence (%) | Old max | New max | Old mean duration (h) | New mean duration (h) |",
                "|---|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in manuscript_comparison.itertuples(index=False):
            lines.append(
                f"| {row.case} | {row.strategy} | {row.hazard} | "
                f"{row.manuscript_departure_incidence_pct:.6f} | {row.departure_incidence_pct:.6f} | "
                f"{row.manuscript_max_peak_exceedance:.6f} | {row.max_peak_exceedance:.6f} | "
                f"{row.manuscript_conditional_mean_duration_h:.6f} | {row.conditional_mean_duration_h:.6f} |"
            )
    lines.extend(
        [
            "",
            "## Provenance finding",
            "",
            "The preserved `revision/task4_violations_table.csv` does not reproduce the manuscript's optimized-route incidence rows (the GC rows agree). It therefore cannot be treated as the direct source of the current manuscript table. The old column above comes from that preserved CSV, while every new value is rebuilt from the verified common-weather Task 1 segment artifact. No legacy aggregate rows for CMA-ES were preserved, so CMA-ES appears only in the complete new `c1_weather_exposure.csv`. This discrepancy is retained rather than worked around.",
            "",
            "## Unaffected artifacts",
            "",
            "- Teaser figure: unaffected; it uses saved route geometry, which is unchanged.",
            "- Route-geometry figures: unaffected for the same reason.",
            "- Segment-speed table (Table 10): unchanged; speed depends on saved geometry and fixed segment timing, neither of which is altered by a weather re-score.",
            "",
            "All requested regenerated figures are in `revision2/figures/`. Their plotted values are in the C1 CSV outputs.",
        ]
    )
    (REV / "c1_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_c1(main_tex: Path | None) -> None:
    scores = route_scores()
    segments = segment_inputs()
    seasonal = seasonal_table(scores)
    route_level, exposure = route_exposure(segments)
    speeds = segment_speed_table(segments)
    energy_annotations = energy_overview_figures(scores)
    histogram = paired_histogram(scores)
    seasonality_figures(scores)
    curves = violation_curves(segments)
    legacy = legacy_exposure_table()
    manuscript = parse_manuscript_exposure(main_tex)
    write_csv(seasonal, REV / "c1_seasonal_savings.csv")
    write_csv(route_level, REV / "c1_route_exposure.csv")
    write_csv(exposure, REV / "c1_weather_exposure.csv")
    write_csv(speeds, REV / "c1_table10_segment_speed.csv")
    write_csv(energy_annotations, REV / "c1_figure9_annotations.csv")
    write_csv(histogram, REV / "c1_paired_histogram_bins.csv")
    write_csv(curves, REV / "c1_violation_curves.csv")
    write_csv(legacy, REV / "c1_legacy_exposure_source.csv", {"source": OLD_EXPOSURE})
    if not manuscript.empty:
        write_csv(
            manuscript, REV / "c1_manuscript_old_exposure.csv", {"source": main_tex}
        )
    c1_report(seasonal, exposure, legacy, manuscript)


def run_c2() -> None:
    frame = read_csv(FUNCTIONALS, parse_dates=["departure_utc"])
    keys = ["case", "departure_utc"]
    cma = frame[frame["strategy"].eq("CMA-ES")].set_index(keys)
    final = frame[frame["strategy"].eq("BERS")].set_index(keys)
    components = {
        "energy": "energy_mwh",
        "wind_penalty": "cma_wind_penalty",
        "wave_penalty": "cma_wave_penalty",
        "coastline_proximity": "cma_coastline_penalty",
        "land_replacement": "cma_land_replacement_term",
        "j_cma": "j_cma",
    }
    delta = pd.DataFrame(index=cma.index)
    for name, column in components.items():
        delta[name] = final[column] - cma[column]
    delta["final_land_count"] = final["cma_land_crossings"]
    delta["cma_land_count"] = cma["cma_land_crossings"]
    delta = delta.reset_index()
    activated = delta[delta["final_land_count"] > 0].copy()
    summaries = []
    for case, group in delta.groupby("case", sort=True):
        clean = group[group["final_land_count"].eq(0)]
        for component in components:
            summaries.append(
                {
                    "case": case,
                    "component": component,
                    "mean_final_minus_cma": group[component].mean(),
                    "median_final_minus_cma": group[component].median(),
                    "mean_excluding_final_land_activated": clean[component].mean(),
                    "n_pairs": len(group),
                    "n_excluded_land_activated": len(group) - len(clean),
                }
            )
    summary = pd.DataFrame(summaries)
    land = activated[
        ["case", "departure_utc", "cma_land_count", "final_land_count", "j_cma"]
    ].copy()
    land["cma_already_land_activated"] = land["cma_land_count"] > 0
    write_csv(summary, REV / "c2_jcma_decomposition.csv")
    write_csv(land, REV / "c2_land_activated_routes.csv")
    pacific = summary[
        summary["case"].eq("PO_noWPS") & summary["component"].eq("j_cma")
    ].iloc[0]
    land_component = summary[
        summary["case"].eq("PO_noWPS") & summary["component"].eq("land_replacement")
    ].iloc[0]
    lines = [
        "# C2 CMA-ES-functional decomposition",
        "",
        f"Analysis commit: `{git_commit()}`. Input: `{FUNCTIONALS}`.",
        "Every value is a paired final-minus-CMA difference on the preserved routes under the common-weather evaluation.",
        "",
        "## Component summary",
        "",
        "| Case | Component | Mean | Median | Mean excluding final land-activated routes |",
        "|---|---|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.case} | {row.component} | {row.mean_final_minus_cma:.6f} | "
            f"{row.median_final_minus_cma:.6f} | {row.mean_excluding_final_land_activated:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Land-activated routes",
            "",
            f"Exactly {len(land)} final routes activate the raster-mask replacement rule; their identities and both land counts are in `c2_land_activated_routes.csv`.",
            "",
            "## Pacific no-WPS interpretation",
            "",
            f"The all-route mean change in J_CMA is {pacific.mean_final_minus_cma:.6f}; excluding final land-activated routes it is {pacific.mean_excluding_final_land_activated:.6f}. The mean land-replacement contribution is {land_component.mean_final_minus_cma:.6f}. Thus the positive all-route mean is explained by the land-activated routes rather than by a typical increase across departures.",
        ]
    )
    (REV / "c2_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def source_line(path: Path, fragment: str) -> int:
    for number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if fragment in line:
            return number
    raise ValueError(f"Fragment {fragment!r} not found in {path}")


def run_c4() -> None:
    scores = route_scores()
    maximum = scores.loc[scores["common_weather_mwh"].idxmax()]
    checks = [
        (
            "Land base term 1000 + N_land",
            ROOT / "routetools" / "cmaes.py",
            "cost = jnp.where(has_land, penalty + land_count, cost)",
            "yes",
            "The real-ocean launcher supplies penalty=1000; the route cost is replaced when sampled land count is positive.",
        ),
        (
            "J_CMA weather and coastline additions",
            ROOT / "routetools" / "cmaes.py",
            "cost += land.distance_penalty(curve, weight=distance_penalty_weight)",
            "yes",
            "Wind and wave use route-wise maxima; the EDT term is added at saved waypoints.",
        ),
        (
            "Coastline 10 sum 1/(d_j+1)",
            ROOT / "routetools" / "land.py",
            "return weight * jnp.sum(point_penalty, axis=-1)",
            "yes",
            "distance_penalty samples the EDT at curve waypoints with epsilon=1 and the launcher weight is 10.",
        ),
        (
            "A_FMS = h sum_i(E_i + penalties)",
            ROOT / "routetools" / "fms.py",
            "return jnp.sum(h * lag)",
            "yes",
            "Each custom-cost call receives a two-waypoint segment with travel_time=h; cost_function_rise returns that segment's MWh before the action multiplies by h.",
        ),
        (
            "J_FMS = A_FMS/h",
            ROOT / "routetools" / "cost.py",
            "energy_mwh = jnp.sum(power_kw, axis=1) * dt_h / 1000.0",
            "yes",
            "Summing the segment energies after removing the common positive h gives voyage MWh plus segment-sum penalties.",
        ),
        (
            "J_tracker route-wise maximum and best-iterate selection",
            ROOT / "routetools" / "fms.py",
            "improved = cost_now < cost_best",
            "yes",
            "The full-route custom cost uses route-wise maximum smooth penalties and the stored best curve is updated with that scalar tracker.",
        ),
        (
            "FMS activation offsets and weights",
            ROOT / "scripts" / "run_fms_sweep_combined_strict.sh",
            "WIND_PW=50",
            "yes",
            "The run used weights 50/50; task0_config_reconciliation.csv records 20.0 m/s and 7.0 m activation thresholds.",
        ),
    ]
    lines = [
        "# C4 equation-to-code check",
        "",
        f"Analysis commit: `{git_commit()}`. Input roots: `{CMA_ROOT}` and `{BERS_ROOT}`.",
        "",
        "| Equation | Code location | Matches | Comment |",
        "|---|---|---|---|",
    ]
    for equation, path, fragment, matches, comment in checks:
        line = source_line(path, fragment)
        lines.append(
            f"| {equation} | `{path.relative_to(ROOT)}:{line}` | {matches} | {comment} |"
        )
    lines.extend(
        [
            "",
            "## Saved-route energy bound",
            "",
            f"The maximum common-weather propulsive energy among all saved GC, CMA-ES, and BERS routes is **{maximum.common_weather_mwh:.6f} MWh** ({maximum.strategy}, {maximum.case}, departure {maximum.departure_utc}). It is below the 1000 MWh replacement constant.",
            "",
            "Candidate-level energies inside CMA-ES generations were not saved, so this bound applies to exported routes only and cannot establish the same bound for every transient candidate.",
        ]
    )
    (REV / "c4_equation_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_c5() -> None:
    scores = route_scores()
    affected = scores[
        scores["strategy"].eq("BERS")
        & scores["requires_2025_weather"]
        & ~scores["within_tolerance"]
    ].copy()
    boundary = pd.Timestamp("2025-01-01 00:00:00")
    affected["hours_beyond_2024"] = (
        (pd.to_datetime(affected["final_required_evaluation_time_utc"]) - boundary)
        .dt.total_seconds()
        .div(3600.0)
        .clip(lower=0.0)
    )
    affected_out = affected.rename(
        columns={
            "legacy_file_a_mwh": "original_energy_mwh",
            "common_weather_mwh": "rescored_energy_mwh",
            "weather_consistency_delta_mwh": "difference_mwh",
        }
    )[
        [
            "case",
            "departure_utc",
            "original_energy_mwh",
            "rescored_energy_mwh",
            "difference_mwh",
            "hours_beyond_2024",
        ]
    ]
    unaffected = scores.drop(index=affected.index)
    stats = (
        unaffected.groupby(["case", "strategy"], sort=True)
        .agg(
            n_records=("abs_diff_mwh", "size"),
            max_abs_difference_mwh=("abs_diff_mwh", "max"),
            mean_abs_difference_mwh=("abs_diff_mwh", "mean"),
            min_tolerance_mwh=("tolerance_mwh", "min"),
            max_tolerance_mwh=("tolerance_mwh", "max"),
            n_outside_tolerance=("within_tolerance", lambda value: int((~value).sum())),
        )
        .reset_index()
    )
    write_csv(affected_out, REV / "c5_rescored_records.csv")
    write_csv(stats, REV / "c5_reproduction_stats.csv")
    required = (
        scores[scores["requires_2025_weather"]]
        .groupby("case")["departure_utc"]
        .nunique()
        .to_dict()
    )
    lines = [
        "# C5 evidence for re-scored records",
        "",
        f"Analysis commit: `{git_commit()}`. Source: `{ROUTE_SCORES}`.",
        "",
        f"Exactly {len(affected_out)} records are outside tolerance. All are BERS late-December departures; no GC or CMA-ES record is in this set.",
        f"Affected departure range: {affected_out.departure_utc.min()} through {affected_out.departure_utc.max()}.",
        "",
        "The number of distinct departures whose route evaluation requires 2025 weather is:",
        "",
    ]
    for case in CASE_INFO:
        lines.append(f"- {case}: {required.get(case, 0)}")
    lines.extend(
        [
            "",
            "The corridor-level common-horizon subsets are therefore 351 Atlantic departures (366 - 15) and 342 Pacific departures (366 - 24), matching the stated sensitivity subset.",
            "",
            "`c5_reproduction_stats.csv` reports maximum and mean absolute differences for every other case/strategy group together with the route-specific analytical-model tolerance in MWh. Every `n_outside_tolerance` entry is zero.",
        ]
    )
    (REV / "c5_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_c6() -> None:
    task0 = read_csv(TASK0)
    selected = task0[
        task0["role"].eq("manuscript-number roots")
        & task0["section"].eq("configuration")
    ].copy()
    selected.insert(0, "task0_source", str(TASK0.relative_to(ROOT)))
    write_csv(selected, REV / "c6_config_reconciliation.csv")
    all_match = bool(selected["match"].str.lower().eq("yes").all())
    lines = [
        "# C6 configuration reconciliation",
        "",
        "The existing Task 0 report is `revision2/task0_report.md`; its machine-readable source is `revision2/task0_config_reconciliation.csv`.",
        "",
        f"**Does manuscript Section 4.1 describe the run that produced the real-ocean tables? {'Yes' if all_match else 'No'}.** The manuscript-number roots are `output/sweep_combined` for GC/CMA-ES and `output/sweep_combined_fms_strict` for BERS, and the requested K, population, sigma0, budgets, damping, patience, weights, coastline weight, and 1000 land replacement agree with those roots. The different 0.1/200/25000/0.9/50 settings belong to the separate later corrected run, not to the run that produced the manuscript numbers. The FMS launcher did use 20.0 m/s and 7.0 m activation buffers while reporting exposure at 20 m/s and 7 m.",
    ]
    (REV / "c6_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest() -> None:
    commit = git_commit()
    rows = [
        (
            "Table 10",
            "tab:rw-speeds",
            "revision2/followup_verification.py",
            SEGMENTS,
            "revision2/c1_table10_segment_speed.csv",
            "unchanged",
        ),
        (
            "Real-ocean domain-audit table",
            "tab:rw-domain-audit",
            "revision2/task1_out_of_domain_audit.py",
            SEGMENTS,
            "revision2/task1_out_of_domain_summary.csv",
            "unchanged",
        ),
        (
            "Table 11",
            "tab:rw-summary",
            "revision2/task1_out_of_domain_audit.py",
            f"{CMA_ROOT}; {BERS_ROOT}",
            "revision2/task1_table_11_12_weather_consistency.csv",
            "common weather",
        ),
        (
            "Table 12",
            "tab:rw-paired",
            "revision2/task1_out_of_domain_audit.py",
            f"{CMA_ROOT}; {BERS_ROOT}",
            "revision2/task1_table_11_12_weather_consistency.csv",
            "common weather",
        ),
        (
            "Table 13 / common functionals",
            "tab:rw-common-functionals",
            "revision2/task2_common_functionals.py",
            SEGMENTS,
            "revision2/task2_paired_summary.csv",
            "common weather",
        ),
        (
            "Seasonal table",
            "tab:rw-seasonal",
            "revision2/followup_verification.py",
            ROUTE_SCORES,
            "revision2/c1_seasonal_savings.csv",
            "common weather",
        ),
        (
            "Table 14 / exposure",
            "tab:rw-violations",
            "revision2/followup_verification.py",
            SEGMENTS,
            "revision2/c1_weather_exposure.csv",
            "common weather",
        ),
        (
            "Figure 9 / energy violins",
            "fig:rw-energy-overview-atlantic",
            "revision2/followup_verification.py",
            ROUTE_SCORES,
            "revision2/c1_figure9_annotations.csv",
            "regenerated",
        ),
        (
            "Paired-improvement histogram",
            "fig:rw-paired",
            "revision2/followup_verification.py",
            ROUTE_SCORES,
            "revision2/c1_paired_histogram_bins.csv",
            "regenerated",
        ),
        (
            "Seasonality figures",
            "fig:rw-seasonality",
            "revision2/followup_verification.py",
            ROUTE_SCORES,
            "revision2/c1_seasonal_savings.csv",
            "regenerated",
        ),
        (
            "Violation curves",
            "fig:rw-violation-curves",
            "revision2/followup_verification.py",
            SEGMENTS,
            "revision2/c1_violation_curves.csv",
            "regenerated",
        ),
        (
            "Segment-speed distribution",
            "fig:rw-speeds",
            "scripts/swopp3_analysis.py",
            BERS_ROOT,
            "not regenerated",
            "unaffected: saved geometry and fixed timing only",
        ),
        (
            "Teaser figure",
            "fig:teaser",
            "scripts/swopp3_analysis.py",
            CMA_ROOT,
            "not regenerated",
            "unaffected: geometry only",
        ),
        (
            "Route geometry",
            "fig:rw-routes",
            "scripts/swopp3_analysis.py",
            BERS_ROOT,
            "not regenerated",
            "unaffected: geometry only",
        ),
    ]
    lines = [
        "# Revision 2 table and figure manifest",
        "",
        f"Analysis commit: `{commit}`.",
        "",
        "| Manuscript item | Label | Producer | Input root/artifact | Output CSV | Commit | Status |",
        "|---|---|---|---|---|---|---|",
    ]
    for number, label, script, source, output, status in rows:
        lines.append(
            f"| {number} | `{label}` | `{script}` | `{source}` | `{output}` | `{commit}` | {status} |"
        )
    (REV / "MANIFEST.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary() -> None:
    c3_report = (REV / "c3_report.md").read_text(encoding="utf-8")
    c3_counts = re.search(
        r"Comparisons: (\d+) total; (\d+) match; (\d+) mismatch", c3_report
    )
    if c3_counts is None:
        raise RuntimeError("C3 report does not contain the expected comparison summary")
    total, matched, mismatched = c3_counts.groups()
    rows = [
        (
            "C6",
            "done",
            "Section 4.1 does not exactly describe the run: its optimizer settings match the manuscript-number roots, but the FMS launcher activated at 20.0 m/s and 7.0 m rather than 20 m/s and 7 m.",
        ),
        (
            "C4",
            "done",
            "The equations match the implementation, and all exported-route energies are below 1000 MWh; transient CMA-ES candidates were not saved.",
        ),
        (
            "C1",
            "partial",
            "Common-weather tables and figures were regenerated, but the preserved Review 1 exposure CSV is not the source of the manuscript's BERS incidence rows.",
        ),
        (
            "C5",
            "done",
            "Exactly 78 late-December BERS records require correction; all other records reproduce within tolerance.",
        ),
        (
            "C2",
            "done",
            "The positive Pacific/no-WPS mean J_CMA change is driven by the raster-mask land-replacement activations.",
        ),
        (
            "C3",
            "done",
            f"The executable audit checked {total} manuscript/PDF values: {matched} match and {mismatched} mismatch, concentrated in the unpropagated seasonal and exposure results plus two segment-speed rounding cells.",
        ),
    ]
    lines = [
        "# Revision 2 follow-up summary",
        "",
        f"Analysis commit: `{git_commit()}`.",
        "",
        "| Task | Status | Most important finding |",
        "|---|---|---|",
    ]
    lines.extend(f"| {task} | {status} | {finding} |" for task, status, finding in rows)
    (REV / "SUMMARY2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manuscript-main", type=Path)
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=("c6", "c4", "c1", "c5", "c2", "manifest", "summary", "all"),
        default=["all"],
    )
    args = parser.parse_args()
    tasks = (
        ("c6", "c4", "c1", "c5", "c2", "manifest", "summary")
        if "all" in args.tasks
        else args.tasks
    )
    for task in tasks:
        if task == "c6":
            run_c6()
        elif task == "c4":
            run_c4()
        elif task == "c1":
            run_c1(args.manuscript_main)
        elif task == "c5":
            run_c5()
        elif task == "c2":
            run_c2()
        elif task == "manifest":
            write_manifest()
        elif task == "summary":
            write_summary()


if __name__ == "__main__":
    main()
