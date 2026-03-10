#!/usr/bin/env python
"""Validate SWOPP3 route submissions for land intersection.

Checks every segment of every track file for land crossings using
Natural Earth 1:10m polygons via Shapely geometric intersection.
This is more precise than the rasterized Land mask used during
optimization, and can be applied to any team's submission.

Usage
-----
Validate a single track file::

    uv run scripts/validate_routes.py tracks/MyTeam-1-AO_WPS-20240101.csv

Validate all tracks in a directory::

    uv run scripts/validate_routes.py output/swopp3_ne_land/tracks/

Validate with a specific interpolation density (points per segment)::

    uv run scripts/validate_routes.py output/swopp3_ne_land/tracks/ --density 100

Exclude great-circle baselines (filenames containing ``GC``)::

    uv run scripts/validate_routes.py output/swopp3_ne_land/tracks/ --exclude-gc

Print only per-file pass/fail (no per-segment details)::

    uv run scripts/validate_routes.py output/swopp3_ne_land/tracks/ --summary-only

"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


def _load_track(path: Path) -> np.ndarray:
    """Load a track CSV and return an (N, 2) array of (lon, lat)."""
    lons, lats = [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lats.append(float(row["lat_deg"]))
            lons.append(float(row["lon_deg"]))
    return np.column_stack([lons, lats])


def _build_land_geometries(
    bounds: tuple[float, float, float, float] | None = None,
) -> list:
    """Load Natural Earth 1:10m land polygons, clipped to bounds.

    Parameters
    ----------
    bounds : (lon_min, lat_min, lon_max, lat_max), optional
        If given, only geometries intersecting this box are returned.

    Returns
    -------
    list of shapely geometries
    """
    import cartopy.io.shapereader as shpreader
    from shapely.geometry import box
    from shapely.ops import unary_union
    from shapely.validation import make_valid

    shp_path = shpreader.natural_earth(
        resolution="10m", category="physical", name="land"
    )
    reader = shpreader.Reader(shp_path)

    if bounds is not None:
        clip = box(*bounds).buffer(1.0)
        geoms = []
        for g in reader.geometries():
            if g is None or g.is_empty:
                continue
            g = make_valid(g) if not g.is_valid else g
            inter = g.intersection(clip)
            if not inter.is_empty:
                geoms.append(inter)
        return [unary_union(geoms)] if geoms else []
    else:
        geoms = []
        for g in reader.geometries():
            if g is None or g.is_empty:
                continue
            g = make_valid(g) if not g.is_valid else g
            geoms.append(g)
        return [unary_union(geoms)] if geoms else []


def _interpolate_segment(
    p1: np.ndarray, p2: np.ndarray, density: int
) -> np.ndarray:
    """Linearly interpolate between two points, returning (density+1, 2)."""
    t = np.linspace(0, 1, density + 1)[:, None]
    return p1 + t * (p2 - p1)


def validate_track(
    waypoints: np.ndarray,
    land_union,
    density: int = 50,
) -> list[dict]:
    """Check each segment for land intersection.

    Parameters
    ----------
    waypoints : np.ndarray
        Shape (N, 2) with (lon, lat).
    land_union : shapely geometry
        Union of all land polygons.
    density : int
        Number of sub-points per segment for intersection check.

    Returns
    -------
    list of dict
        One entry per violating segment: ``{segment, from_idx, to_idx,
        from_coord, to_coord, land_points, land_fraction}``.
    """
    from shapely.geometry import LineString, Point

    violations = []
    n = len(waypoints)

    for i in range(n - 1):
        p1, p2 = waypoints[i], waypoints[i + 1]
        segment_line = LineString([p1, p2])

        if not segment_line.intersects(land_union):
            continue

        # Count how many sub-points fall on land
        sub_pts = _interpolate_segment(p1, p2, density)
        on_land = sum(
            1 for pt in sub_pts if land_union.contains(Point(pt[0], pt[1]))
        )
        fraction = on_land / len(sub_pts)

        violations.append({
            "segment": i,
            "from_idx": i,
            "to_idx": i + 1,
            "from_coord": (float(p1[0]), float(p1[1])),
            "to_coord": (float(p2[0]), float(p2[1])),
            "land_points": on_land,
            "total_points": len(sub_pts),
            "land_fraction": round(fraction, 4),
        })

    return violations


def validate_file(
    track_path: Path,
    land_union,
    density: int = 50,
) -> list[dict]:
    """Validate a single track file. Returns list of violations."""
    waypoints = _load_track(track_path)
    return validate_track(waypoints, land_union, density=density)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate SWOPP3 routes for land intersection.",
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Track CSV file or directory of track CSVs.",
    )
    parser.add_argument(
        "--density",
        type=int,
        default=50,
        help="Sub-points per segment for intersection check (default: 50).",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print per-file pass/fail, not individual segments.",
    )
    parser.add_argument(
        "--exclude-gc",
        action="store_true",
        help="Skip great-circle baseline files (names containing 'GC').",
    )
    args = parser.parse_args()

    # Collect track files
    if args.path.is_dir():
        track_files = sorted(args.path.glob("*.csv"))
    elif args.path.is_file():
        track_files = [args.path]
    else:
        print(f"Error: {args.path} not found.", file=sys.stderr)
        sys.exit(1)

    if not track_files:
        print("No CSV files found.", file=sys.stderr)
        sys.exit(1)

    # Filter out great-circle baselines if requested
    if args.exclude_gc:
        before = len(track_files)
        track_files = [f for f in track_files if "GC" not in f.stem]
        excluded = before - len(track_files)
        if excluded:
            print(f"Excluded {excluded} great-circle file(s).")
        if not track_files:
            print("No files remaining after excluding GC baselines.")
            sys.exit(0)

    # Compute a bounding box from all track files for efficient clipping
    all_lons, all_lats = [], []
    for tf in track_files:
        wp = _load_track(tf)
        all_lons.extend(wp[:, 0].tolist())
        all_lats.extend(wp[:, 1].tolist())
    bounds = (
        min(all_lons) - 2, min(all_lats) - 2,
        max(all_lons) + 2, max(all_lats) + 2,
    )

    print(f"Loading Natural Earth land polygons for bbox {bounds} ...")
    land_geoms = _build_land_geometries(bounds=bounds)
    if not land_geoms:
        print("Warning: no land geometries found in bounding box.")
        sys.exit(0)
    land_union = land_geoms[0]
    print("Land polygons loaded.\n")

    # Validate each file
    total_files = len(track_files)
    files_with_violations = 0
    total_violations = 0

    for tf in track_files:
        violations = validate_file(tf, land_union, density=args.density)
        if violations:
            files_with_violations += 1
            total_violations += len(violations)
            status = f"FAIL ({len(violations)} segment(s))"
        else:
            status = "PASS"

        if args.summary_only:
            print(f"  {status}  {tf.name}")
        else:
            print(f"[{status}] {tf.name}")
            for v in violations:
                print(
                    f"    seg {v['segment']:3d}: "
                    f"({v['from_coord'][0]:8.3f}, {v['from_coord'][1]:7.3f}) → "
                    f"({v['to_coord'][0]:8.3f}, {v['to_coord'][1]:7.3f})  "
                    f"land={v['land_fraction']:.1%} "
                    f"({v['land_points']}/{v['total_points']} sub-points)"
                )

    # Summary
    print(f"\n{'='*60}")
    print(f"Files checked:    {total_files}")
    print(f"Files with land:  {files_with_violations}")
    print(f"Total violations: {total_violations}")
    if files_with_violations == 0:
        print("Result: ALL CLEAR")
    else:
        print("Result: LAND DETECTED")
    sys.exit(1 if files_with_violations > 0 else 0)


if __name__ == "__main__":
    main()
