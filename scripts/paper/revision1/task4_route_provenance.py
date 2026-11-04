"""Audit whether exported BERS routes are CMA-ES refinements or GC fallbacks.

The audit matches every optimized departure to the corresponding great-circle
track and CMA-ES precursor, then compares the exported coordinates directly.
It is intentionally independent of energy values and weather interpolation.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import typer

CASE_PAIRS = {
    "AO_noWPS": "AGC_noWPS",
    "AO_WPS": "AGC_WPS",
    "PO_noWPS": "PGC_noWPS",
    "PO_WPS": "PGC_WPS",
}


def _summary_rows(folder: Path, case_id: str) -> dict[str, dict[str, str]]:
    path = folder / f"IEUniversity-1-{case_id}.csv"
    with path.open(newline="") as handle:
        return {row["departure_time_utc"]: row for row in csv.DictReader(handle)}


def _track(folder: Path, filename: str) -> np.ndarray:
    path = folder / "tracks" / filename
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return np.asarray(
        [[float(row["lon_deg"]), float(row["lat_deg"])] for row in rows],
        dtype=float,
    )


def _wrapped_lon_delta_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a - b + 180.0) % 360.0 - 180.0


def _mean_separation_km(a: np.ndarray, b: np.ndarray) -> float:
    lat1 = np.deg2rad(a[:, 1])
    lat2 = np.deg2rad(b[:, 1])
    dlon = np.deg2rad(_wrapped_lon_delta_deg(a[:, 0], b[:, 0]))
    dlat = lat2 - lat1
    h = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(
        dlon / 2.0
    ) ** 2
    return float(np.mean(6371.0088 * 2.0 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))))


def _max_coordinate_delta_deg(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return float("inf")
    lon_delta = np.abs(_wrapped_lon_delta_deg(a[:, 0], b[:, 0]))
    lat_delta = np.abs(a[:, 1] - b[:, 1])
    return float(max(np.max(lon_delta), np.max(lat_delta)))


def audit(cmaes_dir: Path, bers_dir: Path) -> list[dict[str, object]]:
    """Return coordinate-level provenance metrics for every route group."""
    results: list[dict[str, object]] = []
    for optimised_case, gc_case in CASE_PAIRS.items():
        cmaes_rows = _summary_rows(cmaes_dir, optimised_case)
        gc_rows = _summary_rows(cmaes_dir, gc_case)
        bers_rows = _summary_rows(bers_dir, optimised_case)

        exact_gc = exact_cmaes = near_gc = near_cmaes = 0
        closer_gc = closer_cmaes = equal_distance = 0
        bers_gc_km: list[float] = []
        bers_cmaes_km: list[float] = []
        cmaes_gc_km: list[float] = []

        common = sorted(set(cmaes_rows) & set(gc_rows) & set(bers_rows))
        for departure in common:
            cmaes = _track(cmaes_dir, cmaes_rows[departure]["details_filename"])
            gc = _track(cmaes_dir, gc_rows[departure]["details_filename"])
            bers = _track(bers_dir, bers_rows[departure]["details_filename"])

            d_bers_gc = _max_coordinate_delta_deg(bers, gc)
            d_bers_cmaes = _max_coordinate_delta_deg(bers, cmaes)
            exact_gc += int(d_bers_gc == 0.0)
            exact_cmaes += int(d_bers_cmaes == 0.0)
            near_gc += int(d_bers_gc <= 1e-6)
            near_cmaes += int(d_bers_cmaes <= 1e-6)

            sep_bers_gc = _mean_separation_km(bers, gc)
            sep_bers_cmaes = _mean_separation_km(bers, cmaes)
            sep_cmaes_gc = _mean_separation_km(cmaes, gc)
            bers_gc_km.append(sep_bers_gc)
            bers_cmaes_km.append(sep_bers_cmaes)
            cmaes_gc_km.append(sep_cmaes_gc)
            if np.isclose(sep_bers_gc, sep_bers_cmaes, atol=1e-9, rtol=0):
                equal_distance += 1
            elif sep_bers_gc < sep_bers_cmaes:
                closer_gc += 1
            else:
                closer_cmaes += 1

        results.append(
            {
                "case": optimised_case,
                "n": len(common),
                "bers_exact_gc": exact_gc,
                "bers_exact_cmaes": exact_cmaes,
                "bers_near_gc_1e-6deg": near_gc,
                "bers_near_cmaes_1e-6deg": near_cmaes,
                "bers_closer_gc": closer_gc,
                "bers_closer_cmaes": closer_cmaes,
                "equal_distance": equal_distance,
                "mean_bers_gc_km": float(np.mean(bers_gc_km)),
                "mean_bers_cmaes_km": float(np.mean(bers_cmaes_km)),
                "mean_cmaes_gc_km": float(np.mean(cmaes_gc_km)),
            }
        )
    return results


def main(
    cmaes_dir: Path = Path("output/sweep_combined"),
    bers_dir: Path = Path("output/sweep_combined_fms_strict"),
    output_csv: Path | None = None,
) -> None:
    """Compare all BERS routes with their CMA-ES and great-circle tracks."""
    results = audit(cmaes_dir, bers_dir)
    for row in results:
        print(
            f"{row['case']}: n={row['n']}  "
            f"BERS==GC {row['bers_exact_gc']}  "
            f"BERS==CMA-ES {row['bers_exact_cmaes']}  "
            f"closer-to-GC {row['bers_closer_gc']}  "
            f"closer-to-CMA-ES {row['bers_closer_cmaes']}  "
            f"mean separation BERS/GC={row['mean_bers_gc_km']:.2f} km, "
            f"BERS/CMA-ES={row['mean_bers_cmaes_km']:.2f} km"
        )

    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
        print(f"Wrote {output_csv}")


if __name__ == "__main__":
    typer.run(main)
