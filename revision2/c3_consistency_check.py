# ruff: noqa: D101, D102, D103, E501
"""Audit real-ocean manuscript numbers against the preserved-route evidence.

This script is intentionally read-only with respect to the manuscript.  Expected
values are derived from revision2 CSV files; only table structure and display
precision are encoded here.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

CASE_LABEL = {
    "AO_noWPS": "Atlantic, no WPS",
    "AO_WPS": "Atlantic, WPS",
    "PO_noWPS": "Pacific, no WPS",
    "PO_WPS": "Pacific, WPS",
}
CASE_ORDER = tuple(CASE_LABEL)
STRATEGIES = ("GC", "CMA-ES", "BERS")


@dataclass(frozen=True)
class Check:
    location: str
    manuscript: str
    evidence: str
    match: bool
    source: str


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, comment="#")


def git_commit(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def metadata_line(path: Path) -> str:
    first = path.read_text(encoding="utf-8").splitlines()[0]
    return first.removeprefix("# ") if first.startswith("#") else "no CSV metadata"


def table_rows(tex: str, label: str) -> list[list[str]]:
    marker = rf"\label{{{label}}}"
    label_at = tex.index(marker)
    start = tex.rfind(r"\begin{table", 0, label_at)
    end = tex.find(r"\end{table", label_at)
    block = tex[start:end]
    rows: list[list[str]] = []
    for line in block.splitlines():
        stripped = line.strip()
        if "&" not in stripped or not stripped.endswith(r"\\"):
            continue
        if any(token in stripped for token in (r"\textbf", r"\multicolumn")):
            continue
        rows.append([cell.strip() for cell in stripped[:-2].split("&")])
    return rows


def plain(value: str) -> str:
    value = value.replace(r"\name{}", "BERS")
    value = value.replace(r"\textsubscript{same}", "same")
    value = value.replace(r"\textsubscript{noWPS}", "noWPS")
    value = value.replace("$", "").replace("{", "").replace("}", "")
    value = value.replace(r"\%", "%").replace("~", " ")
    value = re.sub(r"\\(?:textbf|textit|mathrm)\b", "", value)
    return re.sub(r"\s+", " ", value).strip()


def first_number(value: str) -> float | None:
    cleaned = plain(value).replace(",", "")
    match = re.search(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?", cleaned)
    return float(match.group()) if match else None


def formatted(value: float | int, decimals: int, signed: bool = False) -> str:
    prefix = "+" if signed and float(value) > 0 else ""
    return f"{prefix}{float(value):.{decimals}f}"


class Audit:
    def __init__(self) -> None:
        self.checks: list[Check] = []

    def numeric(
        self,
        location: str,
        manuscript: str,
        expected: float | int,
        decimals: int,
        source: str,
        *,
        signed: bool = False,
    ) -> None:
        observed = first_number(manuscript)
        rounded = round(float(expected), decimals)
        match = observed is not None and abs(observed - rounded) < 0.5 * 10 ** (
            -decimals
        )
        self.checks.append(
            Check(
                location,
                plain(manuscript),
                formatted(expected, decimals, signed),
                match,
                source,
            )
        )

    def integer(
        self, location: str, manuscript: str, expected: int, source: str
    ) -> None:
        self.numeric(location, manuscript, expected, 0, source)

    def fraction(
        self,
        location: str,
        manuscript: str,
        numerator: int,
        denominator: int,
        source: str,
    ) -> None:
        cleaned = plain(manuscript).replace(",", "")
        found = re.search(r"(\d+)\s*/\s*(\d+)", cleaned)
        match = bool(
            found
            and int(found.group(1)) == numerator
            and int(found.group(2)) == denominator
        )
        self.checks.append(
            Check(
                location, plain(manuscript), f"{numerator}/{denominator}", match, source
            )
        )

    def text_token(self, location: str, text: str, expected: str, source: str) -> None:
        self.checks.append(
            Check(
                location,
                expected if expected in text else "missing",
                expected,
                expected in text,
                source,
            )
        )


def row_by_prefix(rows: list[list[str]], *prefixes: str) -> list[str]:
    for row in rows:
        joined = " | ".join(plain(cell) for cell in row)
        if all(prefix in joined for prefix in prefixes):
            return row
    raise ValueError(f"missing table row containing {prefixes}")


def audit_speed_table(audit: Audit, tex: str, rev: Path) -> None:
    rows = table_rows(tex, "tab:rw-speeds")
    data = read_csv(rev / "c1_table10_segment_speed.csv").set_index("case")
    columns = (
        ("segments", 0),
        ("min_kn", 2),
        ("p5_kn", 2),
        ("mean_kn", 2),
        ("median_kn", 2),
        ("p95_kn", 2),
        ("max_kn", 2),
    )
    for case, label in CASE_LABEL.items():
        row = row_by_prefix(rows, label)
        for index, (column, decimals) in enumerate(columns, start=1):
            audit.numeric(
                f"Table rw-speeds/{label}/{column}",
                row[index],
                data.loc[case, column],
                decimals,
                "c1_table10_segment_speed.csv",
            )


def audit_domain_table(audit: Audit, tex: str, rev: Path) -> None:
    rows = table_rows(tex, "tab:rw-domain-audit")
    data = read_csv(rev / "task1_out_of_domain_summary.csv")
    all_data = data[data["sample"].eq("all departures")]
    for corridor, case in (("Atlantic", "AO_noWPS"), ("Pacific", "PO_noWPS")):
        row = row_by_prefix(rows, "GC", corridor)
        any_row = all_data[
            all_data["case"].eq(case)
            & all_data["strategy"].eq("GC")
            & all_data["variable"].eq("any")
        ].iloc[0]
        audit.fraction(
            f"Table rw-domain-audit/GC {corridor}/segments",
            row[2],
            int(any_row.n_out),
            int(any_row.n_segments),
            "task1_out_of_domain_summary.csv",
        )
        audit.numeric(
            f"Table rw-domain-audit/GC {corridor}/segment percent",
            row[3],
            any_row.pct_segments_out,
            3,
            "task1_out_of_domain_summary.csv",
        )
        audit.fraction(
            f"Table rw-domain-audit/GC {corridor}/departures",
            row[4],
            int(any_row.n_departures_out),
            int(any_row.n_departures),
            "task1_out_of_domain_summary.csv",
        )
        audit.numeric(
            f"Table rw-domain-audit/GC {corridor}/departure percent",
            row[4].split("(")[-1],
            any_row.pct_departures_out,
            2,
            "task1_out_of_domain_summary.csv",
        )
        audit.numeric(
            f"Table rw-domain-audit/GC {corridor}/time",
            row[5],
            any_row.hours_out,
            1,
            "task1_out_of_domain_summary.csv",
        )
        hazard_rows = all_data[
            all_data["case"].eq(case)
            & all_data["strategy"].eq("GC")
            & all_data["variable"].isin(["hs", "tws"])
            & all_data["n_out"].gt(0)
        ]
        for hazard in hazard_rows.itertuples(index=False):
            token = "SWH" if hazard.variable == "hs" else "TWS"
            match = re.search(rf"{token}\s+([0-9.]+)", plain(row[6]))
            manuscript = match.group(1) if match else "missing"
            audit.numeric(
                f"Table rw-domain-audit/GC {corridor}/{token} max excess",
                manuscript,
                hazard.max_exceedance,
                3,
                "task1_out_of_domain_summary.csv",
            )

    for strategy in ("CMA-ES", "BERS"):
        manuscript_name = "CMA-ES" if strategy == "CMA-ES" else "BERS"
        row = row_by_prefix(rows, manuscript_name)
        subset = all_data[
            all_data["strategy"].eq(strategy) & all_data["variable"].eq("any")
        ]
        audit.fraction(
            f"Table rw-domain-audit/{strategy}/segments",
            row[2],
            int(subset.n_out.sum()),
            int(subset.n_segments.sum()),
            "task1_out_of_domain_summary.csv",
        )
        audit.integer(
            f"Table rw-domain-audit/{strategy}/segment percent",
            row[3],
            0,
            "task1_out_of_domain_summary.csv",
        )
        audit.fraction(
            f"Table rw-domain-audit/{strategy}/departures",
            row[4],
            int(subset.n_departures_out.sum()),
            int(subset.n_departures.sum()),
            "task1_out_of_domain_summary.csv",
        )
        audit.integer(
            f"Table rw-domain-audit/{strategy}/time",
            row[5],
            0,
            "task1_out_of_domain_summary.csv",
        )


def audit_summary_table(audit: Audit, tex: str, rev: Path) -> None:
    rows = table_rows(tex, "tab:rw-summary")
    data = read_csv(rev / "task1_table_11_12_weather_consistency.csv")
    data = data[data.table.eq("Table 11")]
    subjects = {"Great Circle": "GC", "CMA-ES": "CMA-ES", "BERS": "BERS"}
    for case, label in CASE_LABEL.items():
        corridor, wps = label.split(", ")
        wps_tex = "Enabled" if wps == "WPS" else "Disabled"
        for manuscript_strategy, subject in subjects.items():
            row = next(
                row
                for row in rows
                if plain(row[0]) == corridor
                and plain(row[1]) == wps_tex
                and plain(row[2]) == manuscript_strategy
            )
            subset = data[data.case.eq(case) & data.subject.eq(subject)].set_index(
                "statistic"
            )
            for idx, stat in enumerate(
                (
                    "mean_mwh",
                    "std_mwh",
                    "saving_vs_gc_same_mode_pct",
                    "saving_vs_gc_nowps_pct",
                ),
                start=3,
            ):
                expected = subset.loc[stat, "common_weather_rescore"]
                if "---" in row[idx]:
                    expected_zero = abs(float(expected)) < 5e-12
                    audit.checks.append(
                        Check(
                            f"Table rw-summary/{label}/{subject}/{stat}",
                            "---",
                            "---" if expected_zero else formatted(expected, 1),
                            expected_zero,
                            "task1_table_11_12_weather_consistency.csv",
                        )
                    )
                else:
                    audit.numeric(
                        f"Table rw-summary/{label}/{subject}/{stat}",
                        row[idx],
                        expected,
                        1,
                        "task1_table_11_12_weather_consistency.csv",
                    )


def audit_paired_table(audit: Audit, tex: str, rev: Path) -> None:
    rows = table_rows(tex, "tab:rw-paired")
    data = read_csv(rev / "task1_table_11_12_weather_consistency.csv")
    data = data[data.table.eq("Table 12")]
    for case, label in CASE_LABEL.items():
        row = row_by_prefix(rows, label)
        gc = data[data.case.eq(case) & data.subject.eq("BERS vs GC")].set_index(
            "statistic"
        )
        cma = data[data.case.eq(case) & data.subject.eq("BERS vs CMA-ES")].set_index(
            "statistic"
        )
        expected = [
            *(
                gc.loc[s, "common_weather_rescore"]
                for s in (
                    "mean_pct",
                    "std_pct",
                    "median_pct",
                    "p5_pct",
                    "p95_pct",
                    "min_pct",
                )
            ),
            *(
                cma.loc[s, "common_weather_rescore"]
                for s in ("mean_pct", "std_pct", "p5_pct", "p95_pct")
            ),
        ]
        names = (
            "mean",
            "std",
            "median",
            "p5",
            "p95",
            "min",
            "FMS mean",
            "FMS std",
            "FMS p5",
            "FMS p95",
        )
        for idx, (name, value) in enumerate(zip(names, expected, strict=True), start=1):
            audit.numeric(
                f"Table rw-paired/{label}/{name}",
                row[idx],
                value,
                1,
                "task1_table_11_12_weather_consistency.csv",
                signed=name == "min" and float(value) > 0,
            )


def audit_functional_table(audit: Audit, tex: str, rev: Path) -> None:
    rows = table_rows(tex, "tab:rw-common-functionals")
    data = read_csv(rev / "task2_paired_summary.csv").set_index("case")
    fields = (
        "bers_minus_cma_energy_mean",
        "bers_minus_cma_j_cma_mean",
        "bers_minus_cma_j_fms_mean",
        "mean_j_fms_change_from_energy",
        "mean_j_fms_change_from_penalties",
    )
    for case, label in CASE_LABEL.items():
        row = row_by_prefix(rows, label)
        for idx, field in enumerate(fields, start=1):
            value = data.loc[case, field]
            audit.numeric(
                f"Table rw-common-functionals/{label}/{field}",
                row[idx],
                value,
                3,
                "task2_paired_summary.csv",
                signed=float(value) > 0,
            )
        audit.fraction(
            f"Table rw-common-functionals/{label}/n_lower",
            row[6],
            int(data.loc[case, "n_j_fms_decreases"]),
            int(data.loc[case, "n_departures"]),
            "task2_paired_summary.csv",
        )


def audit_seasonal_table(audit: Audit, tex: str, rev: Path) -> None:
    rows = table_rows(tex, "tab:rw-seasonal")
    data = read_csv(rev / "c1_seasonal_savings.csv")
    for season in ("Winter", "Spring", "Summer", "Autumn"):
        row = row_by_prefix(rows, season)
        for idx, corridor in enumerate(("Atlantic", "Pacific"), start=1):
            expected = (
                data[data.season.eq(season) & data.corridor.eq(corridor)]
                .iloc[0]
                .common_saving_reduction_of_means_pct
            )
            audit.numeric(
                f"Table rw-seasonal/{season}/{corridor}",
                row[idx],
                expected,
                1,
                "c1_seasonal_savings.csv",
            )


def audit_exposure_table(audit: Audit, tex: str, rev: Path) -> None:
    rows = table_rows(tex, "tab:rw-violations")
    data = read_csv(rev / "c1_weather_exposure.csv")
    fields = (
        ("departure_incidence_pct", 1),
        ("conditional_mean_peak_exceedance", 3),
        ("max_peak_exceedance", 2),
        ("conditional_mean_duration_h", 1),
        ("max_duration_h", 1),
    )
    for case, label in CASE_LABEL.items():
        for strategy in ("GC", "BERS"):
            for hazard in ("wind", "wave"):
                row = next(
                    row
                    for row in rows
                    if plain(row[0]).startswith(label)
                    and strategy in plain(row[0])
                    and plain(row[1]).lower() == hazard
                )
                expected = data[
                    data.case.eq(case)
                    & data.strategy.eq(strategy)
                    & data.hazard.eq(hazard)
                ].iloc[0]
                for idx, (field, decimals) in enumerate(fields, start=2):
                    audit.numeric(
                        f"Table rw-violations/{label}/{strategy}/{hazard}/{field}",
                        row[idx],
                        getattr(expected, field),
                        decimals,
                        "c1_weather_exposure.csv",
                    )


def pdf_text(path: Path) -> str:
    """Extract annotations with pypdf, falling back to Poppler's pdftotext."""
    try:
        from pypdf import PdfReader
    except ModuleNotFoundError:
        result = subprocess.run(
            ["pdftotext", "-layout", str(path), "-"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout
    return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)


def audit_figures(audit: Audit, rev: Path) -> None:
    figures = rev / "figures"
    annotations = read_csv(rev / "c1_figure9_annotations.csv")
    for figure, group in annotations.groupby("figure", sort=True):
        text = pdf_text(figures / figure)
        for row in group[group.strategy.ne("GC")].itertuples(index=False):
            expected = f"{row.saving_of_marginal_medians_vs_gc_pct:.1f}%"
            audit.text_token(
                f"Figure {figure}/{row.strategy} median saving",
                text,
                expected,
                "c1_figure9_annotations.csv + PDF text",
            )

    paired = read_csv(rev / "task1_table_11_12_weather_consistency.csv")
    paired = paired[
        paired.table.eq("Table 12")
        & paired.subject.eq("BERS vs GC")
        & paired.statistic.eq("mean_pct")
    ].set_index("case")
    text = pdf_text(figures / "fig09_paired_improvements.pdf")
    tokens = re.findall(r"Mean:\s*([-+]?\d+(?:\.\d+)?)%", text)
    for case, actual in zip(CASE_ORDER, tokens, strict=False):
        expected = paired.loc[case, "common_weather_rescore"]
        audit.numeric(
            f"Figure fig09_paired_improvements.pdf/{CASE_LABEL[case]}/mean",
            actual,
            expected,
            1,
            "task1_table_11_12_weather_consistency.csv + PDF text",
        )
    if len(tokens) != len(CASE_ORDER):
        audit.checks.append(
            Check(
                "Figure fig09/annotation count",
                str(len(tokens)),
                str(len(CASE_ORDER)),
                False,
                "PDF text",
            )
        )

    curves = read_csv(rev / "c1_violation_curves.csv")
    unique = curves[
        ["case", "strategy", "hazard", "time_weighted_exposure_pct"]
    ].drop_duplicates()
    text = pdf_text(figures / "fig10_weather_violation_curves.pdf")
    actual_tokens = re.findall(r"(?:GC|CMA-ES|BERS)\s*\(([-+]?\d+(?:\.\d+)?)%\)", text)
    expected_rows = []
    for hazard in ("wind", "wave"):
        for case in CASE_ORDER:
            for strategy in STRATEGIES:
                expected_rows.append(
                    unique[
                        unique.case.eq(case)
                        & unique.strategy.eq(strategy)
                        & unique.hazard.eq(hazard)
                    ].iloc[0]
                )
    for expected, actual in zip(expected_rows, actual_tokens, strict=False):
        audit.numeric(
            f"Figure fig10/{CASE_LABEL[expected.case]}/{expected.strategy}/{expected.hazard}",
            actual,
            expected.time_weighted_exposure_pct,
            2,
            "c1_violation_curves.csv + PDF text",
        )
    if len(actual_tokens) != len(expected_rows):
        audit.checks.append(
            Check(
                "Figure fig10/annotation count",
                str(len(actual_tokens)),
                str(len(expected_rows)),
                False,
                "PDF text",
            )
        )

    for path in sorted(figures.glob("fig04a_*.pdf")):
        text = pdf_text(path)
        for token in ("GC", "CMA-ES", "BERS", "2024-01", "2025-01"):
            audit.text_token(
                f"Figure {path.name}/structural label", text, token, "PDF text"
            )


def section(tex: str, start_label: str, next_label: str | None = None) -> str:
    start = tex.index(rf"\label{{{start_label}}}")
    end = tex.index(rf"\label{{{next_label}}}", start) if next_label else len(tex)
    return tex[start:end]


def audit_quoted_text(audit: Audit, tex: str, highlights: str, rev: Path) -> None:
    summary = read_csv(rev / "task1_table_11_12_weather_consistency.csv")
    table11 = summary[summary.table.eq("Table 11")]
    table12 = summary[summary.table.eq("Table 12")]
    paired = read_csv(rev / "task2_paired_summary.csv").set_index("case")
    exposure = read_csv(rev / "c1_weather_exposure.csv")
    seasonal = read_csv(rev / "c1_seasonal_savings.csv")
    scores = read_csv(rev / "task1_weather_consistency_correction.csv")

    bers_same = table11[
        table11.subject.eq("BERS") & table11.statistic.eq("saving_vs_gc_same_mode_pct")
    ].common_weather_rescore
    bers_nowps = table11[
        table11.subject.eq("BERS") & table11.statistic.eq("saving_vs_gc_nowps_pct")
    ].common_weather_rescore
    headline_range = f"{bers_same.min():.0f}--{bers_same.max():.0f}"
    combined = f"{bers_nowps.max():.0f}"
    n_lower = int(paired.n_j_fms_decreases.min())
    n_departures = int(paired.n_departures.min())
    lower_phrase = f"{n_lower} of {n_departures}"
    for location, body in (
        ("Abstract", tex[: tex.index(r"\section{Introduction}")]),
        ("Highlights", highlights),
    ):
        for expected, source in (
            (headline_range, "task1_table_11_12_weather_consistency.csv"),
            (combined + r"\%", "task1_table_11_12_weather_consistency.csv"),
            (lower_phrase, "task2_paired_summary.csv"),
        ):
            audit.text_token(f"{location}/{plain(expected)}", body, expected, source)

    setup = section(tex, "sec:rw-setup", "sec:rw-routes")
    c5 = read_csv(rev / "c5_rescored_records.csv")
    audit.text_token(
        "§4.1/affected records", setup, str(len(c5)), "c5_rescored_records.csv"
    )
    for corridor, case in (("Atlantic", "AO_noWPS"), ("Pacific", "PO_noWPS")):
        n_need = scores[
            scores.case.eq(case) & scores.requires_2025_weather
        ].departure_utc.nunique()
        expected = str(366 - n_need)
        audit.text_token(
            f"§4.1/common horizon {corridor}",
            setup,
            expected,
            "task1_weather_consistency_correction.csv",
        )

    energy = section(tex, "sec:rw-energy", "sec:rw-paired")
    cma_same = table11[
        table11.subject.eq("CMA-ES")
        & table11.statistic.eq("saving_vs_gc_same_mode_pct")
    ].common_weather_rescore
    for expected in (
        f"{cma_same.min():.0f}--{cma_same.max():.0f}",
        headline_range,
        combined + r"\%",
    ):
        audit.text_token(
            f"§4 energy/{plain(expected)}",
            energy,
            expected,
            "task1_table_11_12_weather_consistency.csv",
        )

    paired_text = section(tex, "sec:rw-paired", "sec:rw-local-opt")
    wide = scores.pivot(
        index=["case", "departure_utc"], columns="strategy", values="common_weather_mwh"
    )
    for case in CASE_ORDER:
        wide.loc[(case,), "bers_vs_gc_pct"] = (
            100
            * (wide.loc[(case,), "GC"] - wide.loc[(case,), "BERS"])
            / wide.loc[(case,), "GC"]
        )
    losses = wide[wide.bers_vs_gc_pct.lt(0)]
    for case in ("PO_noWPS", "PO_WPS"):
        expected = str(int((losses.reset_index().case == case).sum()))
        audit.text_token(
            f"§4 paired/{CASE_LABEL[case]} losses",
            paired_text,
            expected,
            "task1_weather_consistency_correction.csv",
        )
    december_losses = (
        pd.to_datetime(losses.reset_index().departure_utc).dt.month.eq(12).sum()
    )
    audit.text_token(
        "§4 paired/December exceptions",
        paired_text,
        str(int(december_losses)),
        "task1_weather_consistency_correction.csv",
    )
    fms_means = table12[
        table12.subject.eq("BERS vs CMA-ES") & table12.statistic.eq("mean_pct")
    ].common_weather_rescore
    for endpoint, value in (("minimum", fms_means.min()), ("maximum", fms_means.max())):
        audit.text_token(
            f"§4 paired/FMS mean range {endpoint}",
            paired_text,
            f"{value:.1f}\\%",
            "task1_table_11_12_weather_consistency.csv",
        )
    for case in CASE_ORDER:
        case_wide = wide.loc[(case,)]
        count = int((case_wide["CMA-ES"] > case_wide.BERS).sum())
        audit.text_token(
            f"§4 paired/{CASE_LABEL[case]} FMS energy decreases",
            paired_text,
            str(count),
            "task1_weather_consistency_correction.csv",
        )

    functional_text = section(tex, "sec:rw-local-opt", "sec:rw-wps")
    functional_range = (
        f"{int(paired.n_j_fms_decreases.min())}--{int(paired.n_j_fms_decreases.max())}"
    )
    audit.text_token(
        "§4 common functionals/lower-count range",
        functional_text,
        functional_range,
        "task2_paired_summary.csv",
    )

    seasonal_text = section(tex, "sec:rw-seasonal", "sec:rw-violations")
    seasonal_match = re.search(
        r"Winter delivers.*?([0-9.]+)\\% for the Atlantic and ([0-9.]+)\\% for the Pacific.*?Summer savings.*?\(([0-9.]+)\\% and ([0-9.]+)\\%\)",
        seasonal_text,
        re.DOTALL,
    )
    seasonal_actual = seasonal_match.groups() if seasonal_match else ("missing",) * 4
    for index, (season_name, corridor) in enumerate(
        (
            ("Winter", "Atlantic"),
            ("Winter", "Pacific"),
            ("Summer", "Atlantic"),
            ("Summer", "Pacific"),
        )
    ):
        value = (
            seasonal[seasonal.corridor.eq(corridor) & seasonal.season.eq(season_name)]
            .iloc[0]
            .common_saving_reduction_of_means_pct
        )
        audit.numeric(
            f"§4 seasonal text/{corridor}/{season_name}",
            seasonal_actual[index],
            value,
            1,
            "c1_seasonal_savings.csv",
        )

    weather_text = section(tex, "sec:rw-violations", "sec:discussion")
    distribution_match = re.search(
        r"great-circle route spends ([0-9.]+)--([0-9.]+)\\%.*?wind threshold and ([0-9.]+)--([0-9.]+)\\%.*?wave threshold.*?CMA-ES reduces those ranges to ([0-9.]+)--([0-9.]+)\\% and ([0-9.]+)--([0-9.]+)\\%.*?exposure is ([0-9.]+)\\% for wind and ([0-9.]+)--([0-9.]+)\\% for waves",
        weather_text,
        re.DOTALL,
    )
    actual_values = (
        distribution_match.groups() if distribution_match else ("missing",) * 11
    )
    expected_ranges: list[tuple[str, str, float, float]] = []
    for strategy, manuscript_name in (("GC", "great-circle"), ("CMA-ES", "CMA-ES")):
        for hazard in ("wind", "wave"):
            values = exposure[
                exposure.strategy.eq(strategy) & exposure.hazard.eq(hazard)
            ].time_weighted_exposure_pct
            expected_ranges.append(
                (manuscript_name, hazard, values.min(), values.max())
            )
    bers_wind = exposure[
        exposure.strategy.eq("BERS") & exposure.hazard.eq("wind")
    ].time_weighted_exposure_pct
    bers_wave = exposure[
        exposure.strategy.eq("BERS") & exposure.hazard.eq("wave")
    ].time_weighted_exposure_pct
    expected_ranges.extend(
        (
            ("BERS", "wind", bers_wind.min(), bers_wind.max()),
            ("BERS", "wave", bers_wave.min(), bers_wave.max()),
        )
    )
    actual_ranges = (
        actual_values[0:2],
        actual_values[2:4],
        actual_values[4:6],
        actual_values[6:8],
        (actual_values[8], actual_values[8]),
        actual_values[9:11],
    )
    for (manuscript_name, hazard, minimum, maximum), actual in zip(
        expected_ranges, actual_ranges, strict=True
    ):
        audit.numeric(
            f"§4 weather/{manuscript_name}/{hazard} time minimum",
            actual[0],
            minimum,
            2,
            "c1_weather_exposure.csv",
        )
        audit.numeric(
            f"§4 weather/{manuscript_name}/{hazard} time maximum",
            actual[1],
            maximum,
            2,
            "c1_weather_exposure.csv",
        )

    ratios_peak: list[float] = []
    ratios_duration: list[float] = []
    for (_case, _hazard), group in exposure.groupby(["case", "hazard"]):
        by_strategy = group.set_index("strategy")
        if by_strategy.loc["BERS", "conditional_mean_peak_exceedance"] > 0:
            ratios_peak.append(
                by_strategy.loc["GC", "conditional_mean_peak_exceedance"]
                / by_strategy.loc["BERS", "conditional_mean_peak_exceedance"]
            )
            ratios_duration.append(
                by_strategy.loc["GC", "conditional_mean_duration_h"]
                / by_strategy.loc["BERS", "conditional_mean_duration_h"]
            )
    factor_match = re.search(
        r"peak exceedance falls by factors of approximately ([0-9.]+)--([0-9.]+).*?duration falls by factors of ([0-9.]+)--([0-9.]+)",
        weather_text,
        re.DOTALL,
    )
    factor_actual = factor_match.groups() if factor_match else ("missing",) * 4
    for index, (name, value) in enumerate(
        (
            ("peak minimum", min(ratios_peak)),
            ("peak maximum", max(ratios_peak)),
            ("duration minimum", min(ratios_duration)),
            ("duration maximum", max(ratios_duration)),
        )
    ):
        audit.numeric(
            f"§4 weather/conditional {name}",
            factor_actual[index],
            value,
            1,
            "c1_weather_exposure.csv",
        )
    maxima_match = re.search(
        r"largest optimized-route exceedance is ([0-9.]+)~m/s for wind and ([0-9.]+)~m for waves; the great-circle maxima are ([0-9.]+)~m/s and ([0-9.]+)~m",
        weather_text,
    )
    maxima_actual = maxima_match.groups() if maxima_match else ("missing",) * 4
    maxima_expected = (
        exposure[
            exposure.strategy.eq("BERS") & exposure.hazard.eq("wind")
        ].max_peak_exceedance.max(),
        exposure[
            exposure.strategy.eq("BERS") & exposure.hazard.eq("wave")
        ].max_peak_exceedance.max(),
        exposure[
            exposure.strategy.eq("GC") & exposure.hazard.eq("wind")
        ].max_peak_exceedance.max(),
        exposure[
            exposure.strategy.eq("GC") & exposure.hazard.eq("wave")
        ].max_peak_exceedance.max(),
    )
    for name, actual, expected in zip(
        ("BERS wind", "BERS wave", "GC wind", "GC wave"),
        maxima_actual,
        maxima_expected,
        strict=True,
    ):
        audit.numeric(
            f"§4 weather/{name} maximum exceedance",
            actual,
            expected,
            2,
            "c1_weather_exposure.csv",
        )
    pacific_wps_wave = exposure[
        exposure.case.eq("PO_WPS")
        & exposure.strategy.eq("BERS")
        & exposure.hazard.eq("wave")
    ].iloc[0]
    event_match = re.search(
        r"only ([A-Za-z]+|\d+) departures exceed 7~m, although their conditional mean duration is ([0-9.]+)~h",
        weather_text,
    )
    event_actual = event_match.groups() if event_match else ("missing", "missing")
    number_words = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
        "thirteen": 13,
    }
    event_count = number_words.get(event_actual[0].lower(), event_actual[0])
    audit.integer(
        "§4 weather/Pacific WPS wave departures",
        str(event_count),
        int(pacific_wps_wave.n_with_violation),
        "c1_weather_exposure.csv",
    )
    audit.numeric(
        "§4 weather/Pacific WPS conditional duration",
        event_actual[1],
        pacific_wps_wave.conditional_mean_duration_h,
        1,
        "c1_weather_exposure.csv",
    )

    discussion = section(tex, "sec:discussion", "sec:conclusion")
    conclusion = section(tex, "sec:conclusion")
    for expected, source in (
        (f"{bers_same.max():.0f}\\%", "task1_table_11_12_weather_consistency.csv"),
        (combined + r"\%", "task1_table_11_12_weather_consistency.csv"),
        (functional_range, "task2_paired_summary.csv"),
        (
            f"{fms_means.min():.1f}--{fms_means.max():.1f}",
            "task1_table_11_12_weather_consistency.csv",
        ),
    ):
        audit.text_token(
            f"§5 Discussion/{plain(expected)}", discussion, expected, source
        )
    for expected, source in (
        (headline_range, "task1_table_11_12_weather_consistency.csv"),
        (combined + r"\%", "task1_table_11_12_weather_consistency.csv"),
        (lower_phrase, "task2_paired_summary.csv"),
    ):
        audit.text_token(
            f"§6 Conclusions/{plain(expected)}", conclusion, expected, source
        )


def write_report(path: Path, audit: Audit, commit: str, sources: list[Path]) -> None:
    counts = Counter(check.match for check in audit.checks)
    mismatches = [check for check in audit.checks if not check.match]
    lines = [
        "# C3 manuscript number-consistency audit",
        "",
        f"Analysis commit: `{commit}`.",
        "",
        f"Comparisons: {len(audit.checks)} total; {counts[True]} match; {counts[False]} mismatch.",
        "",
        "The script reads the manuscript and evidence only; it does not edit either. A mismatch means the displayed manuscript/PDF value does not equal the evidence value at the displayed precision.",
        "",
        "## Mismatches",
        "",
        "| Location | Manuscript/PDF | CSV evidence | Source | Match |",
        "|---|---:|---:|---|:---:|",
    ]
    for check in mismatches:
        lines.append(
            f"| {check.location} | {check.manuscript} | {check.evidence} | `{check.source}` | no |"
        )
    if not mismatches:
        lines.append("| none | --- | --- | --- | yes |")
    lines.extend(
        [
            "",
            "## Complete comparison table",
            "",
            "| Location | Manuscript/PDF | CSV evidence | Source | Match |",
            "|---|---:|---:|---|:---:|",
        ]
    )
    for check in audit.checks:
        lines.append(
            f"| {check.location} | {check.manuscript} | {check.evidence} | `{check.source}` | {'yes' if check.match else 'no'} |"
        )
    lines.extend(["", "## Input metadata", ""])
    for source in sources:
        lines.append(f"- `{source.name}`: {metadata_line(source)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manuscript-root", type=Path, required=True)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parent.parent
    )
    parser.add_argument("--analysis-commit")
    args = parser.parse_args()
    rev = args.repo_root / "revision2"
    tex = (args.manuscript_root / "main.tex").read_text(encoding="utf-8")
    highlights = (args.manuscript_root / "highlights.tex").read_text(encoding="utf-8")
    sources = [
        rev / "c1_table10_segment_speed.csv",
        rev / "task1_out_of_domain_summary.csv",
        rev / "task1_table_11_12_weather_consistency.csv",
        rev / "task2_paired_summary.csv",
        rev / "c1_seasonal_savings.csv",
        rev / "c1_weather_exposure.csv",
        rev / "c1_figure9_annotations.csv",
        rev / "c1_violation_curves.csv",
        rev / "c5_rescored_records.csv",
        rev / "task1_weather_consistency_correction.csv",
    ]
    missing = [str(path) for path in sources if not path.exists()]
    if missing:
        raise FileNotFoundError("missing evidence:\n" + "\n".join(missing))

    audit = Audit()
    audit_speed_table(audit, tex, rev)
    audit_domain_table(audit, tex, rev)
    audit_summary_table(audit, tex, rev)
    audit_paired_table(audit, tex, rev)
    audit_functional_table(audit, tex, rev)
    audit_seasonal_table(audit, tex, rev)
    audit_exposure_table(audit, tex, rev)
    audit_figures(audit, rev)
    audit_quoted_text(audit, tex, highlights, rev)
    commit = args.analysis_commit or git_commit(args.repo_root)
    write_report(rev / "c3_report.md", audit, commit, sources)
    print(
        f"C3: {sum(check.match for check in audit.checks)} matches, {sum(not check.match for check in audit.checks)} mismatches"
    )
    for check in audit.checks:
        if not check.match:
            print(
                f"NO | {check.location} | {check.manuscript} | {check.evidence} | {check.source}"
            )


if __name__ == "__main__":
    main()
