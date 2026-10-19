"""Create and verify a polygon-safe copy of the final real-ocean routes.

The optimiser uses a densely sampled raster land mask.  This final export step
checks every polyline segment against Natural Earth 1:10m polygons and, when a
short coastline corner is detected, moves the smallest possible interior
waypoint by a local angular search.  All unflagged tracks are copied unchanged.

The output directory contains a JSON manifest with every coordinate change.
The script aborts unless the final segment--polygon audit finds zero crossings.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import typer
from shapely.geometry import LineString
from shapely.prepared import prep

try:
    from revision.task7_land_verification import (
        OPTIMISED_CASES,
        _load_tiled_natural_earth_land,
        _unwrap_route_longitudes,
        verify_real_ocean_routes,
    )
except ModuleNotFoundError:  # Direct execution: python revision/<script>.py
    from task7_land_verification import (  # type: ignore[no-redef]
        OPTIMISED_CASES,
        _load_tiled_natural_earth_land,
        _unwrap_route_longitudes,
        verify_real_ocean_routes,
    )
from routetools.swopp3_output import sailed_distance_nm
from routetools.violations import find_team_prefix, read_track_curve

REPO_ROOT = Path(__file__).resolve().parent.parent


def _normalise_longitude(longitude: float) -> float:
    """Map a longitude to the conventional [-180, 180) interval."""
    return (longitude + 180.0) % 360.0 - 180.0


def _crossing_segments(route: np.ndarray, prepared_land) -> list[int]:
    """Return segment indices that intersect the prepared polygon geometry."""
    return [
        index
        for index in range(len(route) - 1)
        if prepared_land.intersects(LineString([route[index], route[index + 1]]))
    ]


def _local_segments_are_clear(
    route: np.ndarray,
    point_index: int,
    candidate: np.ndarray,
    prepared_land,
) -> bool:
    """Check both segments adjacent to one proposed interior waypoint."""
    return not prepared_land.intersects(
        LineString([route[point_index - 1], candidate])
    ) and not prepared_land.intersects(
        LineString([candidate, route[point_index + 1]])
    )


def repair_route(
    route: np.ndarray,
    prepared_land,
    *,
    max_nudge_deg: float = 0.1,
    radii: int = 101,
    angles: int = 720,
) -> tuple[np.ndarray, list[dict[str, float | int]]]:
    """Repair exact polygon intersections with minimal one-waypoint nudges."""
    repaired = np.asarray(route, dtype=float).copy()
    changes: list[dict[str, float | int]] = []

    for _ in range(len(repaired)):
        crossings = _crossing_segments(repaired, prepared_land)
        if not crossings:
            return repaired, changes

        segment_index = crossings[0]
        best: tuple[float, int, np.ndarray] | None = None
        for point_index in (segment_index, segment_index + 1):
            if point_index <= 0 or point_index >= len(repaired) - 1:
                continue
            for radius in np.geomspace(1e-6, max_nudge_deg, radii):
                found_at_radius = False
                for angle in np.linspace(0.0, 2.0 * np.pi, angles, endpoint=False):
                    candidate = repaired[point_index] + radius * np.array(
                        [np.cos(angle), np.sin(angle)]
                    )
                    if _local_segments_are_clear(
                        repaired, point_index, candidate, prepared_land
                    ):
                        if best is None or radius < best[0]:
                            best = (float(radius), point_index, candidate)
                        found_at_radius = True
                        break
                if found_at_radius:
                    break

        if best is None:
            raise RuntimeError(
                f"Could not repair crossing segment {segment_index} within "
                f"{max_nudge_deg} degrees"
            )

        radius, point_index, candidate = best
        original = repaired[point_index].copy()
        repaired[point_index] = candidate
        changes.append(
            {
                "segment_index": segment_index,
                "point_index": point_index,
                "old_lon_deg": float(original[0]),
                "old_lat_deg": float(original[1]),
                "new_lon_deg": float(candidate[0]),
                "new_lat_deg": float(candidate[1]),
                "nudge_deg": radius,
                "approx_nudge_km": radius * 111.0,
            }
        )

    raise RuntimeError("Land repair did not converge")


def _rewrite_track(path: Path, repaired: np.ndarray, changed_indices: set[int]) -> None:
    """Update only repaired coordinates while preserving the time column."""
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(repaired):
        raise ValueError(f"Track length changed unexpectedly for {path}")

    for index in changed_indices:
        # Nine decimal places preserve sub-metre repairs used for hairline
        # coastline contacts; six places can round a repaired segment back
        # onto the polygon boundary.
        rows[index]["lon_deg"] = f"{_normalise_longitude(repaired[index, 0]):.9f}"
        rows[index]["lat_deg"] = f"{repaired[index, 1]:.9f}"

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["time_utc", "lat_deg", "lon_deg"])
        writer.writeheader()
        writer.writerows(rows)


def main(
    input_dir: Path = Path("output/sweep_combined_fms"),
    output_dir: Path = Path("output/sweep_combined_fms_landfree"),
    clearance_deg: float = 1e-4,
) -> None:
    """Copy, repair, and verify the final optimized real-ocean route set."""
    source = (REPO_ROOT / input_dir).resolve()
    destination = (REPO_ROOT / output_dir).resolve()
    if destination.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output directory: {destination}"
        )
    shutil.copytree(source, destination)

    land = _load_tiled_natural_earth_land()
    prepared_land = prep(land)
    prepared_repair_land = prep(land.buffer(clearance_deg))
    team_prefix = find_team_prefix(destination)
    repairs: list[dict[str, object]] = []

    for case_id in OPTIMISED_CASES:
        summary_path = destination / f"{team_prefix}-{case_id}.csv"
        with summary_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            summary_rows = list(reader)
        case_modified = False
        for summary_row in summary_rows:
            track_path = destination / "tracks" / summary_row["details_filename"]
            route = np.asarray(read_track_curve(track_path), dtype=float)
            route[:, 0] = _unwrap_route_longitudes(route[:, 0])
            if not _crossing_segments(route, prepared_land):
                continue
            # Repair against a very small buffered polygon.  The margin makes
            # the exported decimal coordinates robust to float32 readers and
            # avoids leaving a route exactly tangent to the polygon boundary.
            repaired, changes = repair_route(route, prepared_repair_land)
            _rewrite_track(
                track_path,
                repaired,
                {int(change["point_index"]) for change in changes},
            )
            exported_route = read_track_curve(track_path)
            summary_row["sailed_distance_nm"] = (
                f"{sailed_distance_nm(exported_route):.4f}"
            )
            case_modified = True
            repairs.append(
                {
                    "case": case_id,
                    "track": summary_row["details_filename"],
                    "changes": changes,
                }
            )
        if case_modified:
            with summary_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(summary_rows)

    audit_rows = verify_real_ocean_routes(destination, land)
    remaining = [row for row in audit_rows if row["crosses_land"]]
    manifest = {
        "source": str(source),
        "output": str(destination),
        "natural_earth_resolution": "1:10m",
        "repair_clearance_deg": clearance_deg,
        "routes_checked": len(audit_rows),
        "routes_repaired": len(repairs),
        "remaining_crossings": len(remaining),
        "distance_metrics_refreshed": True,
        "weather_and_energy_metrics_inherited": True,
        "repairs": repairs,
    }
    (destination / "land_repair_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    if remaining:
        raise RuntimeError(f"Final audit still found {len(remaining)} crossings")
    typer.echo(
        f"Verified {len(audit_rows)} routes; repaired {len(repairs)}; "
        f"remaining crossings: 0. Output: {destination}"
    )


if __name__ == "__main__":
    typer.run(main)
