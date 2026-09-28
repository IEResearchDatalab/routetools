"""Task 7 (R1-3) — exact geometric land-crossing verification.

For every **final BERS route** (real-ocean SWOPP3 submissions and synthetic
benchmark instances), test every route segment against exact land polygons
using ``shapely`` geometry intersection (no waypoint subsampling).

Real-ocean routes
------------------
Uses the FMS-refined (BERS) SWOPP3 tracks in ``output/sweep_combined_fms``
for the four optimised (non great-circle) cases: ``AO_WPS``, ``AO_noWPS``,
``PO_WPS``, ``PO_noWPS``. Land is the Natural Earth 10m land shapefile,
tiled at -360/0/+360 degrees longitude so that Pacific tracks expressed in
the 0-360 convention are tested correctly across the antimeridian.

Synthetic routes
-----------------
For each of the 5 literature vector fields (circular, fourvortices,
doublegyre, techy, swirlys), runs BERS (CMA-ES + FMS) with land avoidance
enabled over a small grid of synthetic land scenarios (varying water level
and random seed). The raster land mask is polygonized with
``rasterio.features.shapes`` and tested the same way.

Outputs
-------
- ``revision/task7_route_results.csv``: one row per tested route.
- ``revision/task7_verification_report.txt``: summary statement for the paper.
"""

from __future__ import annotations

import csv
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import rasterio.features
import shapely.affinity
import typer
from shapely.geometry import LineString, shape
from shapely.ops import unary_union
from shapely.prepared import prep

from routetools.cmaes import optimize
from routetools.fms import optimize_fms
from routetools.land import Land
from routetools.violations import CASE_ORDER, find_team_prefix, read_track_curve

REPO_ROOT = Path(__file__).resolve().parent.parent

OPTIMISED_CASES = ["AO_WPS", "AO_noWPS", "PO_WPS", "PO_noWPS"]

SYNTHETIC_FIELDS = ["circular", "fourvortices", "doublegyre", "techy", "swirlys"]
SYNTHETIC_WATER_LEVELS = [0.9, 0.8, 0.7]
SYNTHETIC_SEEDS = [0, 1]

# BERS configuration used throughout the paper's synthetic experiments.
CMAES_KWARGS = dict(
    K=9,
    L=200,
    num_pieces=1,
    popsize=500,
    sigma0=2.0,
    tolfun=1e-3,
    damping=1.0,
    maxfevals=500_000,
)
FMS_KWARGS = dict(patience=50, damping=0.5, maxfevals=500_000)
LAND_PENALTY = 1e10


def _load_tiled_natural_earth_land() -> shapely.geometry.base.BaseGeometry:
    """Load Natural Earth 10m land polygons tiled at -360/0/+360 longitude."""
    import cartopy.io.shapereader as shpreader
    import shapefile as shp

    shapefile_path = shpreader.natural_earth(
        resolution="10m", category="physical", name="land"
    )
    reader = shp.Reader(str(shapefile_path))
    land = unary_union([shape(record) for record in reader.shapes()])
    tiled = unary_union(
        [
            shapely.affinity.translate(land, xoff=-360),
            land,
            shapely.affinity.translate(land, xoff=360),
        ]
    )
    return tiled


def _route_crosses_land(
    lon: np.ndarray, lat: np.ndarray, land_polygon, prepared_land=None
) -> tuple[bool, float]:
    """Return (crosses_land, intersection_extent) for a route vs ``land_polygon``.

    ``intersection_extent`` is the bounding-box diagonal of the intersection
    geometry, in the same units as the input coordinates (degrees for
    real-ocean routes, field units for synthetic routes). It quantifies how
    large a crossing is, distinguishing a hard failure from a hairline
    corner-cutting artifact.
    """
    if len(lon) < 2:
        return False, 0.0
    line = LineString(list(zip(lon.tolist(), lat.tolist(), strict=False)))
    if prepared_land is not None and not prepared_land.intersects(line):
        return False, 0.0
    inter = land_polygon.intersection(line)
    if inter.is_empty:
        return False, 0.0
    minx, miny, maxx, maxy = inter.bounds
    extent = float(np.hypot(maxx - minx, maxy - miny))
    return True, extent


def _unwrap_route_longitudes(lon: np.ndarray) -> np.ndarray:
    """Return a continuous longitude sequence across the antimeridian.

    Stored tracks may wrap from +180 to -180 degrees.  Treating those raw
    coordinates as a planar line creates a false segment across almost the
    whole world.  The Natural Earth geometry is tiled at +/-360 degrees, so a
    continuous, unwrapped route can be checked directly against it.
    """
    longitude = np.asarray(lon, dtype=np.float64)
    if longitude.size < 2:
        return longitude.copy()
    return np.rad2deg(np.unwrap(np.deg2rad(longitude)))


def verify_real_ocean_routes(
    input_dir: Path,
    land_polygon,
) -> list[dict[str, object]]:
    """Verify every BERS-optimised real-ocean track for land crossings."""
    team_prefix = find_team_prefix(input_dir)
    tracks_dir = input_dir / "tracks"
    prepared_land = prep(land_polygon)
    rows: list[dict[str, object]] = []

    for case_id in OPTIMISED_CASES:
        assert case_id in CASE_ORDER
        summary_path = input_dir / f"{team_prefix}-{case_id}.csv"
        if not summary_path.exists():
            print(f"  [skip] missing summary CSV: {summary_path}")
            continue

        with summary_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                track_path = tracks_dir / row["details_filename"]
                curve = read_track_curve(track_path)
                lon = _unwrap_route_longitudes(np.asarray(curve[:, 0]))
                lat = np.asarray(curve[:, 1])
                crosses, extent_deg = _route_crosses_land(
                    lon, lat, land_polygon, prepared_land
                )
                # Rough deg-to-km conversion for reporting only (not for testing).
                extent_km = extent_deg * 111.0
                rows.append(
                    {
                        "source": "real_ocean",
                        "scenario": case_id,
                        "route_id": row["details_filename"],
                        "n_waypoints": len(lon),
                        "crosses_land": crosses,
                        "crossing_extent_km": round(extent_km, 4),
                    }
                )
    return rows


def _land_to_polygon(land: Land) -> shapely.geometry.base.BaseGeometry:
    """Polygonize a synthetic ``Land`` raster mask into a shapely geometry."""
    mask = np.asarray(land.array, dtype=np.uint8)  # shape (lenx, leny)
    lenx, leny = mask.shape
    x0, x1 = float(land.xmin), float(land.xmax)
    y0, y1 = float(land.ymin), float(land.ymax)
    dx = (x1 - x0) / max(lenx - 1, 1)
    dy = (y1 - y0) / max(leny - 1, 1)

    # rasterio expects array[row, col]; row -> y axis, col -> x axis.
    array_rc = mask.T  # shape (leny, lenx)
    transform = rasterio.transform.Affine(
        dx,
        0,
        x0 - dx / 2,
        0,
        dy,
        y0 - dy / 2,
    )
    polygons = [
        shape(geom)
        for geom, value in rasterio.features.shapes(array_rc, transform=transform)
        if value == 1
    ]
    if not polygons:
        return shapely.geometry.GeometryCollection()
    return unary_union(polygons)


def _build_land_avoiding_scenario(
    vf_name: str,
    vf_config: dict,
    water_level: float,
    seed: int,
) -> Land | None:
    """Build a Land instance for a synthetic scenario, or None if src/dst blocked."""
    xlim = tuple(vf_config["xlim"])
    ylim = tuple(vf_config["ylim"])
    src = jnp.array(vf_config["src"])
    dst = jnp.array(vf_config["dst"])

    land = Land(
        xlim,
        ylim,
        water_level=water_level,
        resolution=5,
        random_seed=seed,
        outbounds_is_land=False,
    )
    if bool(land(src)) or bool(land(dst)):
        return None
    return land


def verify_synthetic_routes(config_path: Path) -> list[dict[str, object]]:
    """Run BERS with land avoidance over a grid of synthetic scenarios and verify."""
    import tomllib

    with config_path.open("rb") as handle:
        config = tomllib.load(handle)

    rows: list[dict[str, object]] = []

    for vf_name in SYNTHETIC_FIELDS:
        vf_config = config["vectorfield"][vf_name]
        vectorfield_module = __import__(
            "routetools.vectorfield", fromlist=[f"vectorfield_{vf_name}"]
        )
        vectorfield_fun = getattr(vectorfield_module, f"vectorfield_{vf_name}")
        src = jnp.array(vf_config["src"])
        dst = jnp.array(vf_config["dst"])
        travel_stw = vf_config.get("travel_stw")
        travel_time = vf_config.get("travel_time")

        for water_level in SYNTHETIC_WATER_LEVELS:
            for seed in SYNTHETIC_SEEDS:
                land = _build_land_avoiding_scenario(
                    vf_name, vf_config, water_level, seed
                )
                if land is None:
                    print(
                        f"  [skip] {vf_name} wl={water_level} seed={seed}: "
                        "src/dst on land"
                    )
                    continue

                print(f"  running {vf_name} wl={water_level} seed={seed} ...")
                curve_cmaes, _ = optimize(
                    vectorfield_fun,
                    src,
                    dst,
                    land=land,
                    penalty=LAND_PENALTY,
                    travel_stw=travel_stw,
                    travel_time=travel_time,
                    seed=seed,
                    verbose=False,
                    **CMAES_KWARGS,
                )
                curve_bers, _ = optimize_fms(
                    vectorfield_fun,
                    curve=curve_cmaes,
                    land=land,
                    penalty=LAND_PENALTY,
                    travel_stw=travel_stw,
                    travel_time=travel_time,
                    seed=seed,
                    verbose=False,
                    **FMS_KWARGS,
                )
                final_route = np.asarray(curve_bers[0])
                land_polygon = _land_to_polygon(land)
                crosses, extent_native = _route_crosses_land(
                    final_route[:, 0], final_route[:, 1], land_polygon
                )
                rows.append(
                    {
                        "source": "synthetic",
                        "scenario": f"{vf_name}_wl{water_level}_seed{seed}",
                        "route_id": f"{vf_name}_wl{water_level}_seed{seed}",
                        "n_waypoints": final_route.shape[0],
                        "crosses_land": crosses,
                        # Field-native units (not km); xlim/ylim span O(1-6).
                        "crossing_extent_km": round(extent_native, 6),
                    }
                )
    return rows


def main(
    real_ocean_dir: str = "output/sweep_combined_fms",
    config_path: str = "config.toml",
    output_dir: str = "revision",
    include_synthetic: bool = True,
) -> None:
    """Run Task 7 and optionally skip synthetic route generation."""
    out_dir = REPO_ROOT / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading tiled Natural Earth land polygons...")
    land_polygon = _load_tiled_natural_earth_land()

    print(f"Verifying real-ocean BERS routes in {real_ocean_dir} ...")
    real_rows = verify_real_ocean_routes(REPO_ROOT / real_ocean_dir, land_polygon)

    if include_synthetic:
        print("Verifying synthetic BERS routes ...")
        synthetic_rows = verify_synthetic_routes(REPO_ROOT / config_path)
    else:
        print("Skipping synthetic route generation; checking stored routes only.")
        synthetic_rows = []

    all_rows = real_rows + synthetic_rows
    csv_path = out_dir / "task7_route_results.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source",
                "scenario",
                "route_id",
                "n_waypoints",
                "crosses_land",
                "crossing_extent_km",
            ],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    n_real = len(real_rows)
    n_synth = len(synthetic_rows)
    n_total = len(all_rows)
    n_crossing = sum(1 for r in all_rows if r["crosses_land"])
    n_crossing_real = sum(1 for r in real_rows if r["crosses_land"])
    n_crossing_synth = sum(1 for r in synthetic_rows if r["crosses_land"])

    report_lines = [
        "Task 7 — Geometric Land Verification (R1-3)",
        "=" * 60,
        f"Real-ocean BERS routes tested: {n_real} "
        f"(cases: {', '.join(OPTIMISED_CASES)})",
        (
            f"Synthetic BERS routes tested: {n_synth} "
            f"(fields: {', '.join(SYNTHETIC_FIELDS)}; "
            f"water levels: {SYNTHETIC_WATER_LEVELS}; seeds: {SYNTHETIC_SEEDS})"
            if include_synthetic
            else "Synthetic BERS routes tested: skipped by request"
        ),
        f"Total routes tested: {n_total}",
        "",
        f"Land-crossing routes found (real-ocean): {n_crossing_real}",
        f"Land-crossing routes found (synthetic): {n_crossing_synth}",
        f"Land-crossing routes found (total): {n_crossing}",
        "",
    ]
    if n_crossing == 0:
        report_lines.append(
            f"All {n_total} final routes ({n_synth} synthetic + {n_real} "
            "real-ocean) were validated by exact segment-polygon intersection "
            "tests; zero land-crossing solutions were found."
        )
    else:
        offending = [
            (r["route_id"], r["crossing_extent_km"])
            for r in all_rows
            if r["crosses_land"]
        ]
        report_lines.append(
            f"WARNING: {n_crossing}/{n_total} routes have a nonzero exact "
            "geometric intersection with land. 'crossing_extent_km' is the "
            "bounding-box diagonal of the intersection (real-ocean: km via a "
            "111 km/deg approximation; synthetic: field-native units, NOT km). "
            "Very small values indicate hairline corner-cutting between "
            "sparse waypoints/grid cells rather than a large detour through "
            "land.\n"
            f"Offending routes (route_id, crossing_extent_km): {offending}"
        )

    report_text = "\n".join(report_lines)
    print("\n" + report_text)
    (out_dir / "task7_verification_report.txt").write_text(
        report_text + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    typer.run(main)
