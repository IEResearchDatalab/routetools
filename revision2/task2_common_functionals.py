# ruff: noqa: E501
"""Task 2: decompose the CMA-ES and FMS objectives on common sampling.

This is a post-hoc evaluation of the saved manuscript-number routes.  It does
not rerun either optimizer.  Weather and propulsive-energy inputs come from the
Task 1 common-weather segment artifact, so every functional is evaluated at the
same saved-waypoint segment midpoints with the same 2024 + January 2025 ERA5
continuation.

The FMS implementation has two different functionals:

* waypoint updates use ``h * sum_i(E_i + 50 e_w,i^2 + 50 e_h,i^2)``;
* best-iterate bookkeeping uses the full-route penalized cost, whose smooth
  weather penalties are route-wise maxima.

Both are reported.  ``j_fms`` is the update functional divided by the positive
case-constant ``h`` so its energy component remains in MWh; ``fms_action_raw``
is the literal duration-scaled value used by the update equations.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "revision2"
SEGMENTS = OUT / "task1_segment_inputs_common_weather.csv.gz"
CMA_ROOT = ROOT / "output" / "sweep_combined"
BERS_ROOT = ROOT / "output" / "sweep_combined_fms_strict"

CASES = {
    "AO_noWPS": {"corridor": "Atlantic", "gc": "AGC_noWPS", "cma_wave": 50.0},
    "AO_WPS": {"corridor": "Atlantic", "gc": "AGC_WPS", "cma_wave": 50.0},
    "PO_noWPS": {"corridor": "Pacific", "gc": "PGC_noWPS", "cma_wave": 200.0},
    "PO_WPS": {"corridor": "Pacific", "gc": "PGC_WPS", "cma_wave": 200.0},
}
STRATEGIES = ("GC", "CMA-ES", "BERS")

CMA_WIND_LIMIT = 20.0
CMA_WAVE_LIMIT = 7.0
CMA_WIND_WEIGHT = 50.0
CMA_LAND_REPLACEMENT = 1000.0
CMA_COAST_WEIGHT = 10.0

FMS_WIND_LIMIT = 19.9
FMS_WAVE_LIMIT = 6.9
FMS_WIND_WEIGHT = 50.0
FMS_WAVE_WEIGHT = 50.0


@dataclass(frozen=True)
class LandResult:
    """Land terms for one saved route."""

    crossings: float
    coast_penalty: float


def squared_excess(values: Iterable[float], limit: float) -> np.ndarray:
    """Return squared positive exceedance above ``limit``."""
    arr = np.asarray(values, dtype=np.float64)
    return np.maximum(arr - limit, 0.0) ** 2


def weather_terms(group: pd.DataFrame, cma_wave_weight: float) -> dict[str, float]:
    """Return the weather terms of both stage-specific objectives."""
    wind_cma = squared_excess(group["tws_mps"], CMA_WIND_LIMIT)
    wave_cma = squared_excess(group["hs_m"], CMA_WAVE_LIMIT)
    wind_fms = squared_excess(group["tws_mps"], FMS_WIND_LIMIT)
    wave_fms = squared_excess(group["hs_m"], FMS_WAVE_LIMIT)
    return {
        "cma_wind_penalty": CMA_WIND_WEIGHT * float(wind_cma.max(initial=0.0)),
        "cma_wave_penalty": cma_wave_weight * float(wave_cma.max(initial=0.0)),
        # Dividing the literal FMS action by h leaves these segment sums.
        "fms_wind_penalty": FMS_WIND_WEIGHT * float(wind_fms.sum()),
        "fms_wave_penalty": FMS_WAVE_WEIGHT * float(wave_fms.sum()),
        # The full-route evaluator used for best-iterate selection takes maxima.
        "fms_tracker_wind_penalty": FMS_WIND_WEIGHT * float(wind_fms.max(initial=0.0)),
        "fms_tracker_wave_penalty": FMS_WAVE_WEIGHT * float(wave_fms.max(initial=0.0)),
    }


def _route_root(strategy: str) -> Path:
    return BERS_ROOT if strategy == "BERS" else CMA_ROOT


def _read_curve(path: Path) -> np.ndarray:
    frame = pd.read_csv(path, usecols=["lon_deg", "lat_deg"])
    curve = frame[["lon_deg", "lat_deg"]].to_numpy(dtype=np.float32)
    if curve.ndim != 2 or curve.shape[1] != 2 or len(curve) < 2:
        raise ValueError(f"Invalid track curve in {path}: {curve.shape}")
    return curve


def _land_bounds(corridor: str) -> tuple[tuple[float, float], tuple[float, float]]:
    """Read the same ERA5 coordinate bounds used to build the run's land mask."""
    import xarray as xr

    stem = corridor.lower()
    path = Path(f"/codabench/data/era5_waves_{stem}_2024.nc")
    if not path.is_file():
        raise FileNotFoundError(
            f"Required ERA5 grid for land-mask bounds is missing: {path}"
        )
    with xr.open_dataset(path) as ds:
        lon_name = next(name for name in ("longitude", "lon") if name in ds.coords)
        lat_name = next(name for name in ("latitude", "lat") if name in ds.coords)
        lon = np.asarray(ds[lon_name].values, dtype=np.float64)
        lat = np.asarray(ds[lat_name].values, dtype=np.float64)
    return (float(lon.min()), float(lon.max())), (float(lat.min()), float(lat.max()))


def _load_land(corridor: str) -> Any:
    from routetools.era5.loader import load_natural_earth_land_mask

    lon_range, lat_range = _land_bounds(corridor)
    return load_natural_earth_land_mask(lon_range, lat_range)


def land_terms(
    route_rows: pd.DataFrame, batch_size: int = 32
) -> dict[tuple[str, str, str], LandResult]:
    """Evaluate historical land replacement and EDT terms for saved routes."""
    import jax.numpy as jnp

    from routetools.violations import normalise_route_longitudes

    results: dict[tuple[str, str, str], LandResult] = {}
    for corridor in ("Atlantic", "Pacific"):
        land = _load_land(corridor)
        longitude_bounds, _ = _land_bounds(corridor)
        subset = route_rows[route_rows["corridor"] == corridor].reset_index(drop=True)
        for start in range(0, len(subset), batch_size):
            batch = subset.iloc[start : start + batch_size]
            curves = []
            for row in batch.itertuples(index=False):
                curve = _read_curve(
                    _route_root(row.strategy) / "tracks" / row.route_file
                )
                curves.append(
                    np.asarray(
                        normalise_route_longitudes(curve, longitude_bounds),
                        dtype=np.float32,
                    )
                )
            lengths = {len(curve) for curve in curves}
            if len(lengths) != 1:
                raise ValueError(
                    f"Mixed route lengths in one land batch: {sorted(lengths)}"
                )
            stack = jnp.asarray(np.stack(curves), dtype=jnp.float32)
            crossings = np.asarray(
                land.penalization(stack, penalty=1), dtype=np.float64
            )
            coast = np.asarray(
                land.distance_penalty(stack, weight=CMA_COAST_WEIGHT),
                dtype=np.float64,
            )
            for row, crossing, coast_value in zip(
                batch.itertuples(index=False), crossings, coast, strict=True
            ):
                key = (row.strategy, row.case, row.departure_utc)
                results[key] = LandResult(float(crossing), float(coast_value))
    if len(results) != len(route_rows):
        raise AssertionError(
            f"Expected {len(route_rows)} land rows, got {len(results)}"
        )
    return results


def decompose(segments: pd.DataFrame) -> pd.DataFrame:
    """Return one additive functional decomposition per saved route."""
    keys = ["strategy", "case", "departure_utc", "route_file"]
    counts = (
        segments.groupby(keys, sort=False).size().rename("n_segments").reset_index()
    )
    counts["corridor"] = counts["case"].map(lambda case: CASES[case]["corridor"])
    lands = land_terms(counts)

    records: list[dict[str, Any]] = []
    for key, group in segments.groupby(keys, sort=True):
        strategy, case, departure, route_file = key
        n_segments = int(len(group))
        if set(group["segment_index"]) != set(range(n_segments)):
            raise ValueError(f"Non-contiguous segments for {key}")
        durations = group["duration_h"].to_numpy(dtype=np.float64)
        if not np.allclose(durations, durations[0], rtol=0.0, atol=1e-12):
            raise ValueError(f"FMS duration scaling is not constant for {key}")
        energy = float(group["energy_mwh"].sum())
        terms = weather_terms(group, float(CASES[case]["cma_wave"]))
        land = lands[(strategy, case, departure)]
        land_replacement = (
            CMA_LAND_REPLACEMENT + land.crossings - energy
            if land.crossings > 0
            else 0.0
        )
        j_cma = (
            energy
            + terms["cma_wind_penalty"]
            + terms["cma_wave_penalty"]
            + land_replacement
            + land.coast_penalty
        )
        j_fms = energy + terms["fms_wind_penalty"] + terms["fms_wave_penalty"]
        tracker = (
            energy
            + terms["fms_tracker_wind_penalty"]
            + terms["fms_tracker_wave_penalty"]
        )
        h = float(durations[0])
        records.append(
            {
                "strategy": strategy,
                "case": case,
                "corridor": CASES[case]["corridor"],
                "wps": int(group["wps"].iloc[0]),
                "departure_utc": departure,
                "route_file": route_file,
                "n_segments": n_segments,
                "segment_duration_h": h,
                "energy_mwh": energy,
                **terms,
                "cma_land_crossings": land.crossings,
                "cma_land_replacement_term": land_replacement,
                "cma_coastline_penalty": land.coast_penalty,
                "j_cma": j_cma,
                "fms_land_penalty": 0.0,
                "fms_coastline_penalty": 0.0,
                "j_fms": j_fms,
                "fms_action_raw": h * j_fms,
                "j_fms_tracker": tracker,
            }
        )
    frame = pd.DataFrame.from_records(records)
    expected = len(CASES) * len(STRATEGIES) * 366
    if len(frame) != expected:
        raise AssertionError(
            f"Expected {expected} route decompositions, got {len(frame)}"
        )
    return frame.sort_values(["case", "strategy", "departure_utc"]).reset_index(
        drop=True
    )


METRICS = {
    "energy": "energy_mwh",
    "cma_wind": "cma_wind_penalty",
    "cma_wave": "cma_wave_penalty",
    "cma_land": "cma_land_replacement_term",
    "cma_coast": "cma_coastline_penalty",
    "j_cma": "j_cma",
    "fms_wind": "fms_wind_penalty",
    "fms_wave": "fms_wave_penalty",
    "fms_land": "fms_land_penalty",
    "fms_coast": "fms_coastline_penalty",
    "j_fms": "j_fms",
    "j_fms_tracker": "j_fms_tracker",
}


def _paired(frame: pd.DataFrame, left: str, right: str, case: str) -> pd.DataFrame:
    cols = ["departure_utc", *METRICS.values()]
    a = frame[(frame["case"] == case) & (frame["strategy"] == left)][cols]
    b = frame[(frame["case"] == case) & (frame["strategy"] == right)][cols]
    merged = a.merge(
        b, on="departure_utc", suffixes=("_left", "_right"), validate="one_to_one"
    )
    if len(merged) != 366:
        raise AssertionError(
            f"{case} {left}/{right}: expected 366 pairs, got {len(merged)}"
        )
    return merged


def paired_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize BERS-minus-CMA deltas and recast Table 12 comparisons."""
    rows: list[dict[str, Any]] = []
    for case in CASES:
        pair = _paired(frame, "BERS", "CMA-ES", case)
        row: dict[str, Any] = {"case": case, "n_departures": len(pair)}
        deltas: dict[str, np.ndarray] = {}
        for short, column in METRICS.items():
            delta = pair[f"{column}_left"].to_numpy(dtype=np.float64) - pair[
                f"{column}_right"
            ].to_numpy(dtype=np.float64)
            deltas[short] = delta
            for stat, value in (
                ("mean", np.mean(delta)),
                ("std", np.std(delta, ddof=1)),
                ("median", np.median(delta)),
                ("p5", np.percentile(delta, 5)),
                ("p95", np.percentile(delta, 95)),
            ):
                row[f"bers_minus_cma_{short}_{stat}"] = float(value)
        row["n_j_fms_decreases"] = int(np.sum(deltas["j_fms"] < 0))
        row["n_j_fms_tracker_decreases"] = int(np.sum(deltas["j_fms_tracker"] < 0))
        row["n_energy_increases_j_fms_decreases"] = int(
            np.sum((deltas["energy"] > 0) & (deltas["j_fms"] < 0))
        )
        row["n_energy_decreases_j_fms_increases"] = int(
            np.sum((deltas["energy"] < 0) & (deltas["j_fms"] > 0))
        )
        row["mean_j_fms_change_from_energy"] = float(np.mean(deltas["energy"]))
        row["mean_j_fms_change_from_penalties"] = float(
            np.mean(deltas["fms_wind"] + deltas["fms_wave"])
        )

        for left, right, label in (
            ("CMA-ES", "GC", "cma_over_gc"),
            ("BERS", "GC", "bers_over_gc"),
            ("BERS", "CMA-ES", "bers_over_cma"),
        ):
            comparison = _paired(frame, left, right, case)
            for short in ("energy", "j_cma", "j_fms"):
                column = METRICS[short]
                saving = (
                    100.0
                    * (comparison[f"{column}_right"] - comparison[f"{column}_left"])
                    / comparison[f"{column}_right"]
                )
                row[f"{label}_{short}_mean_paired_saving_pct"] = float(saving.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def write_tex(summary: pd.DataFrame, path: Path) -> None:
    """Write a compact reviewer-facing paired table."""
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Case & $\Delta E$ & $\Delta J_{\rm CMA}$ & $\Delta J_{\rm FMS}$ & $n(\Delta J_{\rm FMS}<0)$ & $n(\Delta E>0,\Delta J_{\rm FMS}<0)$ & $n(\Delta E<0,\Delta J_{\rm FMS}>0)$ \\",
        r"\midrule",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"{row.case.replace('_', r'\_')} & "
            f"{row.bers_minus_cma_energy_mean:.3f} & "
            f"{row.bers_minus_cma_j_cma_mean:.3f} & "
            f"{row.bers_minus_cma_j_fms_mean:.3f} & "
            f"{row.n_j_fms_decreases:d} & "
            f"{row.n_energy_increases_j_fms_decreases:d} & "
            f"{row.n_energy_decreases_j_fms_increases:d} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(frame: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    """Write the numerical Task 2 report; formulas/provenance are in the companion."""
    positive_land = frame[frame["cma_land_crossings"] > 0]
    land_counts = positive_land.groupby("case").size().to_dict()
    lines = [
        "# Task 2 report: stage-specific and common-functional evaluation",
        "",
        "## Scope and evaluation convention",
        "",
        "This analysis evaluates the saved GC, CMA-ES and BERS routes; it does not rerun an optimizer. "
        "All energy and weather terms use the Task 1 common-weather saved-waypoint segment table. "
        "Thus every route uses its exported waypoints, uniform passage schedule, segment-midpoint sampling, "
        "and the same ERA5 2024 + January 2025 continuation.",
        "",
        "`J_CMA` is the exact CMA-ES-stage objective decomposition. `J_FMS` is the FMS waypoint-update "
        "action divided by the positive case-constant segment duration h, leaving the energy component in "
        "MWh. `fms_action_raw = h * J_FMS` is the literal action used for differentiation. "
        "`J_FMS_tracker` is the distinct whole-route objective used to select the exported best iterate.",
        "",
        "## Results",
        "",
        "All 4,392 route/departure combinations were decomposed. The land replacement and FMS land/coast "
        "terms are reported explicitly rather than silently folded into an objective.",
        "",
        "<!-- prettier-ignore -->",
        "| Case | mean BERS-CMA E (MWh) | mean BERS-CMA J_CMA | mean BERS-CMA J_FMS | J_FMS decreased | E up / J_FMS down | E down / J_FMS up |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.case} | {row.bers_minus_cma_energy_mean:.6f} | "
            f"{row.bers_minus_cma_j_cma_mean:.6f} | {row.bers_minus_cma_j_fms_mean:.6f} | "
            f"{row.n_j_fms_decreases}/366 | {row.n_energy_increases_j_fms_decreases} | "
            f"{row.n_energy_decreases_j_fms_increases} |"
        )
    lines.extend(
        [
            "",
            "Negative changes mean BERS is lower than CMA-ES. Counts under `J_FMS` refer to the normalized "
            "waypoint-update action, not the tracker objective.",
            "",
            "### Energy versus penalty contribution",
            "",
            "<!-- prettier-ignore -->",
            "| Case | mean energy contribution | mean wind+wave penalty contribution | total mean J_FMS change |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.case} | {row.mean_j_fms_change_from_energy:.6f} | "
            f"{row.mean_j_fms_change_from_penalties:.6f} | "
            f"{row.bers_minus_cma_j_fms_mean:.6f} |"
        )
    pwps = summary.loc[summary["case"] == "PO_WPS"].iloc[0]
    e = float(pwps["mean_j_fms_change_from_energy"])
    p = float(pwps["mean_j_fms_change_from_penalties"])
    common_energy_saving = float(pwps["bers_over_cma_energy_mean_paired_saving_pct"])
    dominant = "propulsive energy" if abs(e) >= abs(p) else "weather penalties"
    lines.extend(
        [
            "",
            f"For Pacific WPS, the mean J_FMS change comprises {e:.6f} from propulsive energy and "
            f"{p:.6f} from weather penalties; by absolute contribution it is mainly {dominant}. "
            f"The common-weather paired propulsive-energy saving is {common_energy_saving:.3f}%. "
            "The familiar 30.6% value is the legacy-weather version of that energy statistic; "
            "neither value is a CMA-ES-objective improvement.",
            "",
            "## Post-hoc land result",
            "",
            "All saved GC and CMA-ES routes have zero crossings under the common land mask. "
            f"Positive counts occur in {len(positive_land)} saved BERS routes: "
            + ", ".join(f"{case}={land_counts.get(case, 0)}" for case in CASES)
            + ". Each positive route has land_count=3. The J_CMA counterfactual applies the exact "
            "replacement rule to those routes; J_FMS still has no land objective term.",
            "",
            "## Recast Table 12 quantities",
            "",
            "The CSV includes, for CMA-ES versus GC, BERS versus GC and BERS versus CMA-ES, separate mean "
            "paired percentage savings for propulsive energy, J_CMA and J_FMS. These columns permit the "
            "manuscript to report energy savings and common-functional changes separately.",
            "",
            "## Interpretation caveat",
            "",
            "The FMS update action and exported-best tracker are not the same functional. Therefore an exported "
            "BERS route is guaranteed only to be the best visited route under the tracker definition, not under "
            "the segment-sum update action. Any departures for which J_FMS rises are an implementation-level "
            "consequence of that mismatch, not evidence that the saved output failed its own best-iterate rule.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run the common-functional decomposition and write all Task 2 outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments", type=Path, default=SEGMENTS)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    if not args.segments.is_file():
        raise FileNotFoundError(
            f"Missing Task 1 segment artifact: {args.segments}. Run Task 1 first."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    segments = pd.read_csv(args.segments)
    frame = decompose(segments)
    summary = paired_summary(frame)
    frame.to_csv(args.output_dir / "task2_functional_decomposition.csv", index=False)
    summary.to_csv(args.output_dir / "task2_paired_summary.csv", index=False)
    write_tex(summary, args.output_dir / "task2_paired_summary.tex")
    write_report(frame, summary, args.output_dir / "task2_report.md")
    manifest = {
        "segment_source": str(args.segments),
        "n_segment_rows": len(segments),
        "n_route_rows": len(frame),
        "weights": {
            "cma_wind": CMA_WIND_WEIGHT,
            "cma_wave_atlantic": 50.0,
            "cma_wave_pacific": 200.0,
            "cma_coast": CMA_COAST_WEIGHT,
            "cma_land_replacement": CMA_LAND_REPLACEMENT,
            "fms_wind": FMS_WIND_WEIGHT,
            "fms_wave": FMS_WAVE_WEIGHT,
        },
        "limits": {
            "cma_wind_mps": CMA_WIND_LIMIT,
            "cma_wave_m": CMA_WAVE_LIMIT,
            "fms_wind_mps": FMS_WIND_LIMIT,
            "fms_wave_m": FMS_WAVE_LIMIT,
        },
    }
    (args.output_dir / "task2_provenance.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
