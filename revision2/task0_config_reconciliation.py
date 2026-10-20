"""Reconcile the recorded 2024 final run with the manuscript reference values."""

from __future__ import annotations

import csv
import json
import subprocess
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUN_TAG = "bers_revision_2024_final_20260923"
RUN_ROOT = ROOT / "output" / RUN_TAG
BERS_ROOT = RUN_ROOT / "bers"
CMAES_ROOT = RUN_ROOT / "cmaes"
MANIFEST_PATH = BERS_ROOT / "experiment_manifest.json"
CONFIG_PATH = RUN_ROOT / "resolved_config.toml"
OUTPUT_CSV = ROOT / "revision2" / "task0_config_reconciliation.csv"
OUTPUT_REPORT = ROOT / "revision2" / "task0_report.md"

CASES = {
    "AO_noWPS": {"corridor": "Atlantic", "mode": "noWPS", "gc": "AGC_noWPS"},
    "AO_WPS": {"corridor": "Atlantic", "mode": "WPS", "gc": "AGC_WPS"},
    "PO_noWPS": {"corridor": "Pacific", "mode": "noWPS", "gc": "PGC_noWPS"},
    "PO_WPS": {"corridor": "Pacific", "mode": "WPS", "gc": "PGC_WPS"},
}

TABLE_11 = {
    ("AO_noWPS", "GC"): 310.3,
    ("AO_noWPS", "CMA-ES"): 220.2,
    ("AO_noWPS", "BERS"): 206.2,
    ("AO_WPS", "GC"): 253.3,
    ("AO_WPS", "CMA-ES"): 159.9,
    ("AO_WPS", "BERS"): 139.5,
    ("PO_noWPS", "GC"): 251.7,
    ("PO_noWPS", "CMA-ES"): 200.2,
    ("PO_noWPS", "BERS"): 194.8,
    ("PO_WPS", "GC"): 158.0,
    ("PO_WPS", "CMA-ES"): 87.3,
    ("PO_WPS", "BERS"): 64.1,
}

TABLE_12 = {
    ("AO_noWPS", "BERS over GC"): 27.5,
    ("AO_WPS", "BERS over GC"): 38.9,
    ("PO_noWPS", "BERS over GC"): 17.5,
    ("PO_WPS", "BERS over GC"): 55.2,
    ("AO_noWPS", "BERS over CMA-ES"): 6.0,
    ("AO_WPS", "BERS over CMA-ES"): 12.5,
    ("PO_noWPS", "BERS over CMA-ES"): 2.6,
    ("PO_WPS", "BERS over CMA-ES"): 30.6,
}

TABLE_14 = {
    ("GC", "Atlantic", "wind"): 6.57,
    ("GC", "Atlantic", "wave"): 4.99,
    ("GC", "Pacific", "wind"): 10.64,
    ("GC", "Pacific", "wave"): 6.17,
    ("BERS", "All", "wind"): 0.32,
    ("BERS", "All", "wave"): 2.16,
}


def _git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _read_manifest() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    resolved = tomllib.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    profile = resolved["swopp3"]["experiments"]["bers_revision_final"]
    defaults = profile["defaults"]

    if manifest["experiment"] != "bers_revision_final":
        raise ValueError(f"Unexpected run experiment: {manifest['experiment']}")
    if Path(manifest["output_dir"]).resolve() != BERS_ROOT.resolve():
        raise ValueError(
            "The run manifest output directory does not match this run root."
        )
    if Path(manifest["config_path"]).resolve() != CONFIG_PATH.resolve():
        raise ValueError("The manifest does not point to this resolved configuration.")

    for key in (
        "dt_eval_minutes",
        "wind_penalty_weight",
        "wave_penalty_weight",
        "distance_penalty_weight",
        "land_distance_weight",
        "land_distance_epsilon",
        "land_crossing_penalty",
        "cmaes_k",
        "sigma0",
        "popsize",
        "maxfevals",
        "fms_patience",
        "fms_damping",
        "fms_maxfevals",
    ):
        if defaults[key] != manifest["runs"][0][key]:
            raise ValueError(f"Config/manifest mismatch for {key}.")

    return manifest, profile, defaults


def _read_summary(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required route summary is missing: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    departures = {row["departure_time_utc"]: row for row in rows}
    if len(rows) != 366 or len(departures) != 366:
        raise ValueError(
            f"{path} contains {len(rows)} rows and {len(departures)} unique "
            "departures; expected 366 of each."
        )
    for departure, row in departures.items():
        for column in ("energy_cons_mwh", "max_wind_mps", "max_hs_m"):
            try:
                row[column] = float(row[column])
            except (KeyError, ValueError) as exc:
                raise ValueError(f"Invalid {column} for {departure} in {path}") from exc
    return departures


def _assert_same_gc(
    bers_rows: Mapping[str, Mapping[str, Any]],
    cmaes_rows: Mapping[str, Mapping[str, Any]],
    case: str,
) -> None:
    if bers_rows.keys() != cmaes_rows.keys():
        raise ValueError(f"GC departure keys differ between stage roots for {case}.")
    for departure in bers_rows:
        for column in ("energy_cons_mwh", "max_wind_mps", "max_hs_m"):
            if bers_rows[departure][column] != cmaes_rows[departure][column]:
                raise ValueError(
                    f"GC data differs between stage roots for {case}, "
                    f"{departure}, {column}."
                )


def _matches(actual: float, expected: float, decimals: int) -> bool:
    return abs(actual - expected) <= 0.5 * 10**-decimals + 1e-12


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _add_row(
    rows: list[dict[str, str]],
    *,
    section: str,
    scope: str,
    parameter: str,
    manuscript_value: Any,
    actual_value: Any,
    decimals: int | None,
    notes: str = "",
) -> None:
    if decimals is None:
        matched = str(manuscript_value) == str(actual_value)
    else:
        matched = _matches(float(actual_value), float(manuscript_value), decimals)
    rows.append(
        {
            "section": section,
            "scope": scope,
            "parameter": parameter,
            "manuscript_value": _fmt(manuscript_value),
            "actual_value": _fmt(actual_value),
            "match": "yes" if matched else "no",
            "notes": notes,
        }
    )


def _write_csv(
    rows: list[dict[str, str]], run_commit: str, analysis_commit: str
) -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    input_roots = f"{RUN_ROOT}; {BERS_ROOT}; {CMAES_ROOT}"
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as stream:
        stream.write(
            f"# run_tag={RUN_TAG}; run_git_commit={run_commit}; "
            f"analysis_git_commit={analysis_commit}; input_run_roots={input_roots}\n"
        )
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "section",
                "scope",
                "parameter",
                "manuscript_value",
                "actual_value",
                "match",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Generate the configuration and table reconciliation outputs."""
    manifest, _profile, defaults = _read_manifest()
    run_commit = (RUN_ROOT / "git_commit.txt").read_text(encoding="utf-8").strip()
    analysis_commit = _git_output("rev-parse", "HEAD")
    if not (RUN_ROOT / "COMPLETED").is_file():
        raise FileNotFoundError(f"Run completion marker is missing under {RUN_ROOT}.")

    run_by_corridor = {item["name"]: item for item in manifest["runs"]}
    rows: list[dict[str, str]] = []

    manuscript = {
        "Atlantic": {"K": 10, "P": 400, "wind": 50, "wave": 50, "points": 178},
        "Pacific": {"K": 15, "P": 400, "wind": 50, "wave": 200, "points": 293},
    }
    for corridor, run in (
        ("Atlantic", run_by_corridor["atlantic"]),
        ("Pacific", run_by_corridor["pacific"]),
    ):
        expected = manuscript[corridor]
        config_values = (
            ("CMA-ES K", expected["K"], defaults["cmaes_k"], 0, "Control-point count."),
            (
                "CMA-ES population P",
                expected["P"],
                defaults["popsize"],
                0,
                "Population size.",
            ),
            ("CMA-ES sigma0", 0.3, defaults["sigma0"], 1, ""),
            (
                "CMA-ES maximum evaluations",
                50000,
                defaults["maxfevals"],
                0,
                "Configured evaluation cap.",
            ),
            ("Sampled-land penalty", 1000, defaults["land_crossing_penalty"], 0, ""),
            ("Distance-to-land weight", 10, defaults["land_distance_weight"], 0, ""),
            ("CMA-ES wind weight", expected["wind"], run["wind_penalty_weight"], 0, ""),
            ("CMA-ES wave weight", expected["wave"], run["wave_penalty_weight"], 0, ""),
            (
                "FMS wind weight",
                50,
                run["wind_penalty_weight"],
                0,
                "Shared wind weight recorded in the run manifest.",
            ),
            (
                "FMS wave weight",
                50,
                run["wave_penalty_weight"],
                0,
                "Shared wave weight recorded in the run manifest.",
            ),
            ("FMS damping", 0.95, defaults["fms_damping"], 2, ""),
            ("FMS patience", 200, defaults["fms_patience"], 0, ""),
            (
                "FMS maximum sweeps",
                10000,
                defaults["fms_maxfevals"],
                0,
                "Recorded as fms_maxfevals, not as sweeps.",
            ),
            ("Waypoints", expected["points"], run["n_points"], 0, ""),
            ("Evaluation grid (minutes)", 30, defaults["dt_eval_minutes"], 0, ""),
            (
                "Wind threshold (m/s)",
                20,
                20.0,
                0,
                "Default at the run's recorded source commit.",
            ),
            (
                "Wave threshold (m)",
                7,
                7.0,
                0,
                "Default at the run's recorded source commit.",
            ),
        )
        for parameter, expected_value, actual, decimals, notes in config_values:
            _add_row(
                rows,
                section="configuration",
                scope=corridor,
                parameter=parameter,
                manuscript_value=expected_value,
                actual_value=actual,
                decimals=decimals,
                notes=notes,
            )

    summary: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for case, details in CASES.items():
        mode_suffix = details["mode"]
        gc_name = details["gc"]
        bers_gc = _read_summary(BERS_ROOT / f"IEUniversity-1-{gc_name}.csv")
        cmaes_gc = _read_summary(CMAES_ROOT / f"IEUniversity-1-{gc_name}.csv")
        _assert_same_gc(bers_gc, cmaes_gc, case)
        stage_rows = {
            "GC": bers_gc,
            "CMA-ES": _read_summary(CMAES_ROOT / f"IEUniversity-1-{case}.csv"),
            "BERS": _read_summary(BERS_ROOT / f"IEUniversity-1-{case}.csv"),
        }
        departure_sets = {frozenset(data) for data in stage_rows.values()}
        if len(departure_sets) != 1:
            raise ValueError(f"Departure keys differ across strategies for {case}.")
        summary[(case, mode_suffix)] = stage_rows

        for strategy, data in stage_rows.items():
            mean_energy = sum(row["energy_cons_mwh"] for row in data.values()) / len(
                data
            )
            _add_row(
                rows,
                section="Table 11",
                scope=case,
                parameter=f"{strategy} mean voyage energy (MWh)",
                manuscript_value=TABLE_11[(case, strategy)],
                actual_value=mean_energy,
                decimals=1,
                notes=(
                    "Arithmetic mean over 366 departure_time_utc-matched summary rows."
                ),
            )

        paired_cases = (
            ("BERS over GC", "GC", "BERS"),
            ("BERS over CMA-ES", "CMA-ES", "BERS"),
        )
        for metric, baseline_name, candidate_name in paired_cases:
            baseline = stage_rows[baseline_name]
            candidate = stage_rows[candidate_name]
            improvement = [
                (
                    baseline[departure]["energy_cons_mwh"]
                    - candidate[departure]["energy_cons_mwh"]
                )
                / baseline[departure]["energy_cons_mwh"]
                * 100
                for departure in baseline
            ]
            mean_improvement = sum(improvement) / len(improvement)
            _add_row(
                rows,
                section="Table 12",
                scope=case,
                parameter=f"{metric} mean paired improvement (%)",
                manuscript_value=TABLE_12[(case, metric)],
                actual_value=mean_improvement,
                decimals=1,
                notes=f"Mean per-departure percentage; baseline={baseline_name}.",
            )

    for corridor in ("Atlantic", "Pacific"):
        gc_cases = [
            case for case, detail in CASES.items() if detail["corridor"] == corridor
        ]
        gc_rows = [
            row
            for case in gc_cases
            for row in summary[(case, CASES[case]["mode"])]["GC"].values()
        ]
        for variable, column, threshold in (
            ("wind", "max_wind_mps", 20.0),
            ("wave", "max_hs_m", 7.0),
        ):
            actual = max(row[column] for row in gc_rows) - threshold
            _add_row(
                rows,
                section="Table 14",
                scope=f"GC {corridor}",
                parameter=f"Maximum {variable} exceedance",
                manuscript_value=TABLE_14[("GC", corridor, variable)],
                actual_value=actual,
                decimals=2,
                notes=f"Maximum of route-summary maxima minus {threshold:g} threshold.",
            )

    for variable, column, threshold in (
        ("wind", "max_wind_mps", 20.0),
        ("wave", "max_hs_m", 7.0),
    ):
        bers_rows = [
            row
            for (case, _mode), strategies in summary.items()
            for row in strategies["BERS"].values()
        ]
        actual = max(row[column] for row in bers_rows) - threshold
        _add_row(
            rows,
            section="Table 14",
            scope="BERS All",
            parameter=f"Maximum {variable} exceedance",
            manuscript_value=TABLE_14[("BERS", "All", variable)],
            actual_value=actual,
            decimals=2,
            notes=f"Maximum of route-summary maxima minus {threshold:g} threshold.",
        )

    _write_csv(rows, run_commit, analysis_commit)
    mismatches = [row for row in rows if row["match"] == "no"]
    table_11_rows = [row for row in rows if row["section"] == "Table 11"]
    table_12_rows = [row for row in rows if row["section"] == "Table 12"]
    table_14_rows = [row for row in rows if row["section"] == "Table 14"]
    report_lines = [
        "# Task 0: Configuration reconciliation",
        "",
        f"- Run tag: `{RUN_TAG}`",
        f"- Run Git commit: `{run_commit}`",
        f"- Analysis Git commit: `{analysis_commit}`",
        f"- Input run root: `{RUN_ROOT}`",
        f"- BERS root: `{BERS_ROOT}`",
        f"- CMA-ES root: `{CMAES_ROOT}`",
        f"- Resolved config and manifest: `{CONFIG_PATH}`, `{MANIFEST_PATH}`",
        "- Completion marker: present; 366 unique departures per case and strategy.",
        "- GC summaries in the BERS and CMA-ES roots were identical for energy and "
        "weather maxima.",
        "- Run identity is unverified: this authorized candidate does not reproduce "
        "the manuscript's Table 11 or Table 12 values.",
        "- Expected config note path: "
        "`coding_agent/bers_revision_reference/final_run_configuration.md` "
        "(absent).",
        "- Specified run-output parent: "
        "`coding_agent/bers_revision_outputs/` (absent).",
        "- The completed local run above was used only after author confirmation.",
        "",
        "## What was done",
        "",
        "Compared the run manifest and resolved TOML with the reference settings. "
        "Recomputed Table 11 means, Table 12 paired mean improvements, and Table 14 "
        "maximum exceedances from the per-departure summary CSVs. No optimizer was "
        "run and no route was changed.",
        "",
        "## Configuration mismatches",
        "",
    ]
    config_mismatches = [row for row in mismatches if row["section"] == "configuration"]
    table_11_mismatches = sum(row["match"] != "yes" for row in table_11_rows)
    table_12_mismatches = sum(row["match"] != "yes" for row in table_12_rows)
    table_14_mismatches = sum(row["match"] != "yes" for row in table_14_rows)
    if config_mismatches:
        report_lines.extend(
            [
                "| Corridor | Parameter | Manuscript | Run |",
                "|---|---|---:|---:|",
            ]
        )
        for row in config_mismatches:
            report_lines.append(
                f"| {row['scope']} | {row['parameter']} | "
                f"{row['manuscript_value']} | {row['actual_value']} |"
            )
    else:
        report_lines.append("All compared configuration settings match.")

    for title, section_rows in (
        ("Table 11 mean energies (MWh)", table_11_rows),
        ("Table 12 paired improvements (%)", table_12_rows),
        ("Table 14 maximum exceedances", table_14_rows),
    ):
        report_lines.extend(["", f"## {title}", ""])
        report_lines.extend(
            [
                "| Scope | Measure | Manuscript | Recomputed | Match |",
                "|---|---|---:|---:|:---:|",
            ]
        )
        for row in section_rows:
            report_lines.append(
                f"| {row['scope']} | {row['parameter']} | {row['manuscript_value']} | "
                f"{float(row['actual_value']):.3f} | {row['match']} |"
            )

    report_lines.extend(
        [
            "",
            "## CSV columns",
            "",
            "`section`, `scope`, `parameter`, `manuscript_value`, `actual_value`, "
            "`match`, and `notes`. The CSV begins with a metadata comment recording "
            "both commits, the run tag, "
            "and all input run roots.",
            "",
            "## Decisions needed",
            "",
            f"- {len(config_mismatches)} configuration mismatches; "
            f"{table_11_mismatches} Table 11, {table_12_mismatches} Table 12, and "
            f"{table_14_mismatches} Table 14 values also fail rounding "
            "agreement. The manuscript run identity/numbers must be resolved before "
            "proceeding to Tasks 1–4; no alternate data were substituted.",
            "- No corrections or manuscript edits were made.",
        ]
    )
    OUTPUT_REPORT.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(
        f"Wrote {OUTPUT_CSV} ({len(rows)} rows) and {OUTPUT_REPORT}; "
        f"{len(mismatches)} of {len(rows)} comparisons differ."
    )


if __name__ == "__main__":
    main()
