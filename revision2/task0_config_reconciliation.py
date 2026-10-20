"""Reconcile the manuscript tables and configuration with the run roots.

The primary audit targets the roots that numerically reproduce the manuscript
(CMA-ES/GC in ``output/sweep_combined`` and BERS/GC in
``output/sweep_combined_fms_strict``). A later, fully documented run can be
audited separately as a comparison; it is never substituted for the primary roots.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import subprocess
import tomllib
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CMAES_ROOT = ROOT / "output" / "sweep_combined"
DEFAULT_BERS_ROOT = ROOT / "output" / "sweep_combined_fms_strict"
DEFAULT_LATER_ROOT = ROOT / "output" / "bers_revision_2024_final_20260923"
OUTPUT_CSV = ROOT / "revision2" / "task0_config_reconciliation.csv"
OUTPUT_REPORT = ROOT / "revision2" / "task0_report.md"

ROLE_PRIMARY = "manuscript-number roots"
ROLE_LATER = "later corrected run"
N_DEPARTURES = 366
VALUE_COLUMNS = ("energy_cons_mwh", "max_wind_mps", "max_hs_m")
REPORT_THRESHOLDS = {"wind": 20.0, "wave": 7.0}

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

# Exact means supplied by the authors for numerical identification (6 decimals).
EXACT_MEANS = {
    ("AO_noWPS", "GC"): 310.346441,
    ("AO_noWPS", "CMA-ES"): 220.241450,
    ("AO_noWPS", "BERS"): 206.159138,
    ("AO_WPS", "GC"): 253.329296,
    ("AO_WPS", "CMA-ES"): 159.936026,
    ("AO_WPS", "BERS"): 139.489065,
    ("PO_noWPS", "GC"): 251.725395,
    ("PO_noWPS", "CMA-ES"): 200.205757,
    ("PO_noWPS", "BERS"): 194.758570,
    ("PO_WPS", "GC"): 157.995371,
    ("PO_WPS", "CMA-ES"): 87.265266,
    ("PO_WPS", "BERS"): 64.135670,
}

# Paired improvements supplied by the authors (2 decimals).
EXACT_TABLE_12 = {
    ("AO_noWPS", "BERS over GC"): 27.53,
    ("AO_WPS", "BERS over GC"): 38.90,
    ("PO_noWPS", "BERS over GC"): 17.52,
    ("PO_WPS", "BERS over GC"): 55.17,
    ("AO_noWPS", "BERS over CMA-ES"): 5.97,
    ("AO_WPS", "BERS over CMA-ES"): 12.49,
    ("PO_noWPS", "BERS over CMA-ES"): 2.62,
    ("PO_WPS", "BERS over CMA-ES"): 30.58,
}

# Provenance basis of a configuration claim:
#   direct        literal in preserved source/launcher committed BEFORE the
#                 output it is associated with, or in an embedded manifest;
#   reconstructed inferred from material committed AFTER the output, or from
#                 code defaults whose use in the run is not recorded;
#   unverified    no preserved evidence either way.
L_CMA = "scripts/swopp3_slurm_atlantic_k10.sh @ 4742156e"
P_CMA = "scripts/swopp3_slurm_pacific_k15_p400.sh @ 40c91c02"
RUNNER = "routetools/swopp3_runner.py defaults @ 4742156e (identical @ 66192104)"
FMS_CODE = "scripts/swopp3_apply_fms.py @ 66192104 (unchanged since d2bcc51f)"
STRICT = (
    "scripts/run_fms_sweep_combined_strict.sh @ 66192104 (committed 2026-05-22; "
    "outputs dated 2026-04-06/07)"
)
MANIFEST8 = "revision/task8_sweep_combined_fms_strict_manifest.json (2026-09-24)"


def _cma_claims() -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    spec = {
        "Atlantic": (L_CMA, 10, 50, 178),
        "Pacific": (P_CMA, 15, 200, 293),
    }
    for corridor, (src, k, wave, points) in spec.items():
        for parameter, manuscript, value, decimals in (
            ("CMA-ES K", k, k, 0),
            ("CMA-ES population P", 400, 400, 0),
            ("CMA-ES sigma0", 0.3, 0.3, 1),
            ("CMA-ES maximum evaluations", 50000, 50000, 0),
            ("CMA-ES wind weight", 50, 50, 0),
            ("CMA-ES wave weight", wave, wave, 0),
            ("Distance-to-land weight (distance_penalty_weight)", 10, 10, 0),
            ("Waypoints", points, points, 0),
            ("Evaluation grid (minutes)", 30, 30, 0),
        ):
            claims.append(
                {
                    "scope": corridor,
                    "parameter": parameter,
                    "manuscript": manuscript,
                    "value": value,
                    "decimals": decimals,
                    "provenance": "direct",
                    "source": src,
                    "notes": "Launcher committed before the 2026-04-02 outputs.",
                }
            )
        claims.append(
            {
                "scope": corridor,
                "parameter": "Sampled-land penalty",
                "manuscript": 1000,
                "value": 1000,
                "decimals": 0,
                "provenance": "direct",
                "source": RUNNER,
                "notes": "Runner default penalty=1000; launcher passes no override.",
            }
        )
    return claims


def _fms_claims() -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    for corridor in ("Atlantic", "Pacific"):
        base = {"scope": corridor, "decimals": 0}
        claims.extend(
            [
                {
                    **base,
                    "parameter": "FMS patience",
                    "manuscript": 200,
                    "value": 200,
                    "provenance": "reconstructed",
                    "source": FMS_CODE,
                    "notes": "Code default; launcher passes no --fms-* flag.",
                },
                {
                    **base,
                    "parameter": "FMS damping",
                    "manuscript": 0.95,
                    "value": 0.95,
                    "decimals": 2,
                    "provenance": "reconstructed",
                    "source": FMS_CODE,
                    "notes": "Code default; launcher passes no --fms-* flag.",
                },
                {
                    **base,
                    "parameter": "FMS maximum iterations (code name: fms_maxfevals)",
                    "manuscript": 10000,
                    "value": 10000,
                    "provenance": "reconstructed",
                    "source": FMS_CODE,
                    "notes": "Named maxfevals in code; 'sweeps' wording unverified.",
                },
                {
                    **base,
                    "parameter": "FMS wind weight",
                    "manuscript": 50,
                    "value": 50,
                    "provenance": "reconstructed",
                    "source": STRICT,
                    "notes": "Also stated in " + MANIFEST8,
                },
                {
                    **base,
                    "parameter": "FMS wave weight",
                    "manuscript": 50,
                    "value": 50,
                    "provenance": "reconstructed",
                    "source": STRICT,
                    "notes": "Also stated in " + MANIFEST8,
                },
                {
                    **base,
                    "parameter": "Wind threshold used by FMS (m/s)",
                    "manuscript": 20,
                    "value": 19.9,
                    "decimals": 1,
                    "provenance": "reconstructed",
                    "source": STRICT,
                    "notes": "DISCREPANCY: strict launcher uses 19.9, manuscript 20.",
                },
                {
                    **base,
                    "parameter": "Wave threshold used by FMS (m)",
                    "manuscript": 7,
                    "value": 6.9,
                    "decimals": 1,
                    "provenance": "reconstructed",
                    "source": STRICT,
                    "notes": "DISCREPANCY: strict launcher uses 6.9, manuscript 7.",
                },
                {
                    **base,
                    "parameter": "FMS land handling",
                    "manuscript": None,
                    "value": "sampled-land rollback to the CMA-ES curve",
                    "provenance": "reconstructed",
                    "source": FMS_CODE,
                    "notes": "No differentiated land penalty in the FMS objective.",
                },
                {
                    **base,
                    "parameter": "Coastline-distance term in FMS objective",
                    "manuscript": None,
                    "value": "absent",
                    "provenance": "reconstructed",
                    "source": FMS_CODE + "; " + MANIFEST8,
                    "notes": "costfun kwargs contain no land-distance entry.",
                },
            ]
        )
    return claims


UNVERIFIED_CLAIMS = [
    (
        "All",
        "Exact checked-out/dirty source state of the CMA-ES runs",
        "No embedded manifest, git_commit.txt or resolved config in the root.",
    ),
    (
        "All",
        "Exact checked-out/dirty source state of the strict FMS run",
        "Outputs predate the launcher; no reflog entry for the run was found.",
    ),
    (
        "All",
        "Combination of the K10 and K15 launcher outputs into sweep_combined",
        "The combination step is not documented in the repository.",
    ),
    (
        "All",
        "FMS invoked with the code defaults (no explicit --fms-* flags)",
        "Only the later launcher shows an invocation.",
    ),
    (
        "All",
        "FMS bound counts sweeps (manuscript) or evaluations (code name)",
        "The code exposes fms_maxfevals; the manuscript says sweeps.",
    ),
]


def legacy_config_claims() -> list[dict[str, Any]]:
    """Return the reconstructed legacy configuration claims."""
    return _cma_claims() + _fms_claims()


def root_provenance_basis(root: Path) -> str:
    """Classify a run root as ``manifest`` (embedded provenance) or ``legacy``."""
    markers = (
        root / "experiment_manifest.json",
        root / "git_commit.txt",
        root.parent / "resolved_config.toml",
        root.parent / "git_commit.txt",
    )
    return "manifest" if any(path.is_file() for path in markers) else "legacy"


def _git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def read_summary(path: Path, expected: int = N_DEPARTURES) -> dict[str, dict]:
    """Read one route-summary CSV keyed by departure timestamp, failing loudly."""
    if not path.is_file():
        raise FileNotFoundError(f"Required route summary is missing: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = {"departure_time_utc", *VALUE_COLUMNS} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} lacks columns {sorted(missing)}.")
        rows = list(reader)
    keys = [row["departure_time_utc"] for row in rows]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise ValueError(f"{path} has duplicate departure dates: {duplicates[:3]}.")
    if len(rows) != expected:
        raise ValueError(f"{path} contains {len(rows)} rows; expected {expected}.")
    out: dict[str, dict] = {}
    for row in rows:
        key = row["departure_time_utc"]
        try:
            datetime.fromisoformat(key)
            out[key] = {col: float(row[col]) for col in VALUE_COLUMNS}
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid row for {key!r} in {path}.") from exc
    return out


def assert_same_gc(a: Mapping, b: Mapping, case: str) -> None:
    """Fail unless two GC summaries have identical dates and values."""
    if a.keys() != b.keys():
        raise ValueError(f"GC departure dates differ between roots for {case}.")
    for key in a:
        for col in VALUE_COLUMNS:
            if a[key][col] != b[key][col]:
                raise ValueError(f"GC data differs for {case}, {key}, {col}.")


def load_summaries(cmaes_root: Path, bers_root: Path) -> dict[str, dict]:
    """Load GC/CMA-ES/BERS summaries per case with strict inventory checks."""
    out: dict[str, dict] = {}
    for case, detail in CASES.items():
        name = "IEUniversity-1-{}.csv"
        gc_c = read_summary(cmaes_root / name.format(detail["gc"]))
        gc_b = read_summary(bers_root / name.format(detail["gc"]))
        assert_same_gc(gc_c, gc_b, case)
        strategies = {
            "GC": gc_c,
            "CMA-ES": read_summary(cmaes_root / name.format(case)),
            "BERS": read_summary(bers_root / name.format(case)),
        }
        for strategy, data in strategies.items():
            if data.keys() != gc_c.keys():
                raise ValueError(f"Departure dates for {strategy} differ in {case}.")
        out[case] = strategies
    return out


def mean_energy(data: Mapping[str, Mapping[str, float]]) -> float:
    """Return the mean voyage energy in MWh."""
    return sum(row["energy_cons_mwh"] for row in data.values()) / len(data)


def paired_improvement(
    baseline: Mapping[str, Mapping[str, float]],
    candidate: Mapping[str, Mapping[str, float]],
) -> float:
    """Mean per-departure percent improvement, paired by departure timestamp."""
    if baseline.keys() != candidate.keys():
        raise ValueError("Cannot pair routes with different departure dates.")
    values = [
        (baseline[k]["energy_cons_mwh"] - candidate[k]["energy_cons_mwh"])
        / baseline[k]["energy_cons_mwh"]
        * 100
        for k in baseline
    ]
    return sum(values) / len(values)


def matches(actual: float, expected: float, decimals: int) -> bool:
    """Compare after rounding: within half a unit of the last displayed digit."""
    return abs(actual - expected) <= 0.5 * 10**-decimals + 1e-12


def _fmt(value: Any) -> str:
    return f"{value:.6f}" if isinstance(value, float) else str(value)


def make_row(
    role: str,
    section: str,
    scope: str,
    parameter: str,
    manuscript: Any,
    actual: Any,
    decimals: int | None,
    provenance: str,
    source: str = "",
    notes: str = "",
) -> dict[str, str]:
    """Build one CSV row, deriving the match flag."""
    numeric = isinstance(manuscript, int | float) and isinstance(actual, int | float)
    if manuscript is None:
        match = "n/a"
    elif actual is None:
        match = "unverified"
    elif numeric and decimals is not None:
        match = "yes" if matches(float(actual), float(manuscript), decimals) else "no"
    else:
        match = "yes" if str(manuscript) == str(actual) else "no"
    return {
        "role": role,
        "section": section,
        "scope": scope,
        "parameter": parameter,
        "manuscript_value": "n/a" if manuscript is None else _fmt(manuscript),
        "actual_value": "unknown" if actual is None else _fmt(actual),
        "match": match,
        "provenance": provenance,
        "source": source,
        "notes": notes,
    }


def legacy_config_rows() -> list[dict[str, str]]:
    """Rows for the reconstructed legacy configuration (primary roots)."""
    rows = [
        make_row(
            ROLE_PRIMARY,
            "configuration",
            claim["scope"],
            claim["parameter"],
            claim["manuscript"],
            claim["value"],
            claim["decimals"],
            claim["provenance"],
            claim["source"],
            claim["notes"],
        )
        for claim in legacy_config_claims()
    ]
    rows.extend(
        make_row(
            ROLE_PRIMARY,
            "configuration",
            scope,
            param,
            "n/a",
            None,
            None,
            "unverified",
            "none preserved",
            note,
        )
        for scope, param, note in UNVERIFIED_CLAIMS
    )
    return rows


def table_rows(
    role: str,
    summaries: Mapping[str, Mapping[str, Mapping]],
    provenance: str,
    source: str,
) -> list[dict[str, str]]:
    """Recompute Tables 11, 12 and 14 against the manuscript values."""
    rows: list[dict[str, str]] = []
    add = rows.append
    for case, strategies in summaries.items():
        for strategy, data in strategies.items():
            add(
                make_row(
                    role,
                    "Table 11",
                    case,
                    f"{strategy} mean energy (MWh)",
                    TABLE_11[(case, strategy)],
                    mean_energy(data),
                    1,
                    provenance,
                    source,
                    "Mean over 366 matched departures.",
                )
            )
        for metric, base in (("BERS over GC", "GC"), ("BERS over CMA-ES", "CMA-ES")):
            value = paired_improvement(strategies[base], strategies["BERS"])
            add(
                make_row(
                    role,
                    "Table 12",
                    case,
                    f"{metric} paired improvement (%)",
                    TABLE_12[(case, metric)],
                    value,
                    1,
                    provenance,
                    source,
                    "Mean of per-departure percentages, paired by timestamp.",
                )
            )
    for corridor in ("Atlantic", "Pacific"):
        gc = [
            row
            for case, st in summaries.items()
            if CASES[case]["corridor"] == corridor
            for row in st["GC"].values()
        ]
        for var, col in (("wind", "max_wind_mps"), ("wave", "max_hs_m")):
            thr = REPORT_THRESHOLDS[var]
            add(
                make_row(
                    role,
                    "Table 14",
                    f"GC {corridor}",
                    f"Maximum {var} exceedance",
                    TABLE_14[("GC", corridor, var)],
                    max(r[col] for r in gc) - thr,
                    2,
                    provenance,
                    source,
                    f"Max route maximum minus manuscript threshold {thr:g}.",
                )
            )
    bers = [row for st in summaries.values() for row in st["BERS"].values()]
    for var, col in (("wind", "max_wind_mps"), ("wave", "max_hs_m")):
        thr = REPORT_THRESHOLDS[var]
        add(
            make_row(
                role,
                "Table 14",
                "BERS All",
                f"Maximum {var} exceedance",
                TABLE_14[("BERS", "All", var)],
                max(r[col] for r in bers) - thr,
                2,
                provenance,
                source,
                f"Max route maximum minus manuscript threshold {thr:g}.",
            )
        )
    return rows


def identification_rows(
    summaries: Mapping[str, Mapping[str, Mapping]],
) -> list[dict[str, str]]:
    """Compare means (6 dp) and paired improvements (2 dp) with author values."""
    src = "author-supplied exact values"
    rows = []
    for (case, strategy), expected in EXACT_MEANS.items():
        rows.append(
            make_row(
                ROLE_PRIMARY,
                "Identification",
                case,
                f"{strategy} mean energy (MWh, 6 dp)",
                expected,
                mean_energy(summaries[case][strategy]),
                6,
                "direct",
                src,
                "Numerical identification of the roots.",
            )
        )
    for (case, metric), expected in EXACT_TABLE_12.items():
        base = "GC" if metric.endswith("GC") else "CMA-ES"
        rows.append(
            make_row(
                ROLE_PRIMARY,
                "Identification",
                case,
                f"{metric} paired improvement (%, 2 dp)",
                expected,
                paired_improvement(summaries[case][base], summaries[case]["BERS"]),
                2,
                "direct",
                src,
                "Numerical identification of the roots.",
            )
        )
    return rows


def manifest_config_rows(run_root: Path) -> list[dict[str, str]]:
    """Rows for a run with embedded provenance (later corrected run)."""
    manifest_path = run_root / "bers" / "experiment_manifest.json"
    config_path = run_root / "resolved_config.toml"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    resolved = tomllib.loads(config_path.read_text(encoding="utf-8"))
    defaults = resolved["swopp3"]["experiments"][manifest["experiment"]]["defaults"]
    for key in (
        "cmaes_k",
        "sigma0",
        "popsize",
        "maxfevals",
        "fms_patience",
        "fms_damping",
        "fms_maxfevals",
        "land_crossing_penalty",
        "land_distance_weight",
        "dt_eval_minutes",
    ):
        for item in manifest["runs"]:
            if defaults[key] != item[key]:
                raise ValueError(
                    f"Config/manifest mismatch for {key} in run {item['name']}."
                )
    names = [item["name"] for item in manifest["runs"]]
    if sorted(names) != ["atlantic", "pacific"]:
        raise ValueError(f"Expected one atlantic and one pacific run, got {names}.")
    runs = {item["name"]: item for item in manifest["runs"]}
    src = f"{manifest_path.name} + resolved_config.toml"
    rows: list[dict[str, str]] = []
    expected = {
        "Atlantic": {"K": 10, "wind": 50, "wave": 50, "points": 178},
        "Pacific": {"K": 15, "wind": 50, "wave": 200, "points": 293},
    }
    for corridor, key in (("Atlantic", "atlantic"), ("Pacific", "pacific")):
        run, exp = runs[key], expected[corridor]
        items = (
            ("CMA-ES K", exp["K"], defaults["cmaes_k"], 0),
            ("CMA-ES population P", 400, defaults["popsize"], 0),
            ("CMA-ES sigma0", 0.3, defaults["sigma0"], 1),
            ("CMA-ES maximum evaluations", 50000, defaults["maxfevals"], 0),
            ("Sampled-land penalty", 1000, defaults["land_crossing_penalty"], 0),
            ("Distance-to-land weight", 10, defaults["land_distance_weight"], 0),
            ("CMA-ES wind weight", exp["wind"], run["wind_penalty_weight"], 0),
            ("CMA-ES wave weight", exp["wave"], run["wave_penalty_weight"], 0),
            ("FMS wind weight", 50, run["wind_penalty_weight"], 0),
            ("FMS wave weight", 50, run["wave_penalty_weight"], 0),
            ("FMS damping", 0.95, defaults["fms_damping"], 2),
            ("FMS patience", 200, defaults["fms_patience"], 0),
            (
                "FMS maximum iterations (fms_maxfevals)",
                10000,
                defaults["fms_maxfevals"],
                0,
            ),
            ("Waypoints", exp["points"], run["n_points"], 0),
            ("Evaluation grid (minutes)", 30, defaults["dt_eval_minutes"], 0),
        )
        rows.extend(
            make_row(ROLE_LATER, "configuration", corridor, p, m, a, d, "direct", src)
            for p, m, a, d in items
        )
        for param, value, man in (
            ("Wind threshold (m/s)", 20.0, 20),
            ("Wave threshold (m)", 7.0, 7),
        ):
            rows.append(
                make_row(
                    ROLE_LATER,
                    "configuration",
                    corridor,
                    param,
                    man,
                    value,
                    0,
                    "reconstructed",
                    "code default at the commit in git_commit.txt",
                    "Not recorded in the manifest or resolved config.",
                )
            )
    return rows


def _write_csv(rows: list[dict[str, str]], meta: Mapping[str, str]) -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as stream:
        stream.write("# " + "; ".join(f"{k}={v}" for k, v in meta.items()) + "\n")
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _md_table(rows: list[dict[str, str]], extra: bool = False) -> list[str]:
    head = "| Scope | Measure | Manuscript | Value | Match |"
    sep = "|---|---|---:|---:|:---:|"
    if extra:
        head += " Basis |"
        sep += "---|"
    lines = [head, sep]
    for r in rows:
        value = r["actual_value"]
        with contextlib.suppress(ValueError):
            value = (
                f"{float(value):.6g}"
                if r["section"] != "Identification"
                else (f"{float(value):.6f}")
            )
        line = (
            f"| {r['scope']} | {r['parameter']} | {r['manuscript_value']} | "
            f"{value} | {r['match']} |"
        )
        if extra:
            line += f" {r['provenance']}: {r['source']} |"
        lines.append(line)
    return lines


def _report(rows: list[dict[str, str]], meta: Mapping[str, str]) -> str:
    def pick(role: str, *sections: str) -> list[dict[str, str]]:
        return [r for r in rows if r["role"] == role and r["section"] in sections]

    def bad(items: list[dict[str, str]]) -> int:
        return sum(r["match"] == "no" for r in items)

    prim = pick(ROLE_PRIMARY, "Identification")
    t11 = pick(ROLE_PRIMARY, "Table 11")
    t12 = pick(ROLE_PRIMARY, "Table 12")
    t14 = pick(ROLE_PRIMARY, "Table 14")
    cfg = pick(ROLE_PRIMARY, "configuration")
    later_cfg = pick(ROLE_LATER, "configuration")
    later_tab = pick(ROLE_LATER, "Table 11", "Table 12", "Table 14")
    out = [
        "# Task 0: Configuration reconciliation",
        "",
        f"- Analysis code commit: `{meta['analysis_git_commit']}` "
        f"(tree clean: {meta['analysis_tree_clean']})",
        f"- CMA-ES and GC root: `{meta['cmaes_root']}`",
        f"- BERS and GC root: `{meta['bers_root']}`",
        f"- Later corrected run: `{meta['later_run_root']}`",
        "",
        "## 1. Numerical identification of the manuscript roots",
        "",
        "All eight summary CSVs per root were read; each has exactly 366 unique "
        "departure timestamps, identical across GC, CMA-ES and BERS, and the GC "
        "summaries of the two roots are identical in energy and weather maxima. "
        f"{len(prim) - bad(prim)} of {len(prim)} comparisons with the exact "
        "author-supplied means (6 dp) and paired improvements (2 dp) agree. "
        f"Against the manuscript tables: Table 11 {len(t11) - bad(t11)}/{len(t11)}, "
        f"Table 12 {len(t12) - bad(t12)}/{len(t12)}, Table 14 "
        f"{len(t14) - bad(t14)}/{len(t14)} agree after rounding. Table 14 is "
        "computed against the manuscript thresholds 20 m/s and 7 m.",
        "",
        "### Identification",
        "",
        *_md_table(prim),
        "",
        "### Table 11 (MWh)",
        "",
        *_md_table(t11),
        "",
        "### Table 12 (%)",
        "",
        *_md_table(t12),
        "",
        "### Table 14 (exceedance over 20 m/s and 7 m)",
        "",
        *_md_table(t14),
        "",
        "## 2. Reconstructed configuration evidence",
        "",
        "Basis: `direct` = literal in preserved source committed before the "
        "output (or an embedded manifest); `reconstructed` = inferred from "
        "material committed after the output or from code defaults whose use is "
        "not recorded; `unverified` = no preserved evidence. The legacy roots "
        "predate embedded manifests. The strict launcher and the task-8 manifest "
        "are reconstructions, not contemporaneous manifests. Atlantic and Pacific "
        "rows are identical for the FMS settings.",
        "",
        *_md_table([r for r in cfg if r["provenance"] != "unverified"], extra=True),
        "",
        "## 3. Unresolved provenance limitations",
        "",
        *[
            f"- {r['parameter']}: {r['notes']}"
            for r in cfg
            if r["provenance"] == "unverified"
        ],
        "- The strict FMS output directory is dated April 2026 while the launcher "
        "was committed on 2026-05-22 (`66192104`); no repository source file "
        "relevant to FMS or CMA-ES changed between the outputs and that commit, "
        "but the checked-out and dirty state at run time cannot be proven.",
        "- No exact run commit is asserted.",
        "",
        "## 4. Later corrected-run comparison (separate)",
        "",
        "`output/bers_revision_2024_final_20260923` is complete and has strong "
        "embedded provenance (resolved config, manifest, git commit). It does "
        "not reproduce Tables 11 or 12, so it was not the source of the "
        "manuscript numbers and is not substituted here.",
        "",
        f"Configuration: {bad(later_cfg)} of {len(later_cfg)} claims differ.",
        "",
        *_md_table([r for r in later_cfg if r["match"] == "no"]),
        "",
        f"Tables: {bad(later_tab)} of {len(later_tab)} comparisons differ.",
        "",
        *_md_table([r for r in later_tab if r["section"] != "Table 14"]),
        "",
        "## 5. Decisions required from the authors",
        "",
        "1. Confirm that the manuscript describes the legacy roots, and decide "
        "whether the later corrected run is to be reported separately.",
        "2. The strict FMS used 19.9 m/s and 6.9 m while the manuscript states "
        "20 m/s and 7 m: amend the text, or rerun, as a deliberate decision.",
        "3. Decide whether the FMS bound is described as sweeps or as "
        "evaluations (the code name is `fms_maxfevals`).",
        "4. Document how the K=10 and K=15 launcher outputs were combined into "
        "`sweep_combined`, or state the limitation.",
        "5. Decide whether to accept the reconstructed provenance or to "
        "re-execute under an embedded manifest.",
        "",
        "No optimizer was run, no route or run output was modified, and the "
        "manuscript was not edited.",
    ]
    return "\n".join(out) + "\n"


def run(
    cmaes_root: Path,
    bers_root: Path,
    later_root: Path | None,
    analysis_commit: str,
    tree_clean: bool,
) -> tuple[list[dict[str, str]], dict[str, str]]:
    """Run the full audit and return CSV rows and metadata."""
    rows = legacy_config_rows()
    basis = root_provenance_basis(bers_root)
    summaries = load_summaries(cmaes_root, bers_root)
    rows += identification_rows(summaries)
    source = f"{cmaes_root.name} + {bers_root.name} summaries ({basis} provenance)"
    rows += table_rows(ROLE_PRIMARY, summaries, "direct", source)
    later_commit = "not audited"
    if later_root is not None:
        rows += manifest_config_rows(later_root)
        later = load_summaries(later_root / "cmaes", later_root / "bers")
        rows += table_rows(ROLE_LATER, later, "direct", later_root.name)
        later_commit = (later_root / "git_commit.txt").read_text().strip()
    meta = {
        "analysis_git_commit": analysis_commit,
        "analysis_tree_clean": str(tree_clean).lower(),
        "cmaes_root": str(cmaes_root),
        "bers_root": str(bers_root),
        "later_run_root": str(later_root) if later_root else "none",
        "later_run_git_commit": later_commit,
        "legacy_run_commit": "unverified (no embedded provenance)",
    }
    return rows, meta


def main(argv: list[str] | None = None) -> None:
    """Generate the configuration and table reconciliation outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cmaes-root",
        type=Path,
        default=DEFAULT_CMAES_ROOT,
        help="CMA-ES and GC root (manuscript numbers).",
    )
    parser.add_argument(
        "--bers-root",
        type=Path,
        default=DEFAULT_BERS_ROOT,
        help="Final BERS and GC root (manuscript numbers).",
    )
    parser.add_argument(
        "--later-run-root",
        type=Path,
        default=DEFAULT_LATER_ROOT,
        help="Optional later run with cmaes/ and bers/ subfolders.",
    )
    parser.add_argument(
        "--no-later-run",
        action="store_true",
        help="Skip the later corrected-run comparison.",
    )
    args = parser.parse_args(argv)
    later = None if args.no_later_run else args.later_run_root
    clean = not _git_output(
        "status", "--porcelain", "--", "revision2/*.py", "routetools"
    )
    rows, meta = run(
        args.cmaes_root, args.bers_root, later, _git_output("rev-parse", "HEAD"), clean
    )
    _write_csv(rows, meta)
    OUTPUT_REPORT.write_text(_report(rows, meta), encoding="utf-8")
    diffs = sum(r["match"] == "no" for r in rows)
    print(
        f"Wrote {OUTPUT_CSV} ({len(rows)} rows) and {OUTPUT_REPORT}; "
        f"{diffs} comparisons differ."
    )


if __name__ == "__main__":
    main()
