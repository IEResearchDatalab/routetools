#!/usr/bin/env python
"""Animate wind and wave forecasts for one fixed target time, without boats.

The initialization advances every12hours while the target time stays fixed.
Acquisition uses permanent all51member shards from acquire_weather.py.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import acquire_weather as acquisition
import numpy as np
import xarray as xr

LOG = logging.getLogger("forecast_revisions")
U = "10m_u_component_of_wind"
V = "10m_v_component_of_wind"
HS = "significant_height_of_combined_wind_waves_and_swell"


def schedule(target: datetime) -> list[tuple[datetime, int]]:
    """Hold valid time fixed while advancing initialization over15days."""
    if (
        target.hour not in (0, 12)
        or target.minute
        or target.second
        or target.microsecond
        or target.tzinfo
    ):
        raise ValueError("target must be a naive UTC00/12hour timestamp")
    return [(target - timedelta(hours=lead), lead) for lead in range(360, -1, -12)]


def shard(root: Path, issue: datetime, stream: str, lead: int) -> Path:
    """Return the permanent shared path for an ensemble lead."""
    return (
        root
        / "ifs/v1/atlantic"
        / issue.strftime("%Y/%m/%Y%m%dT%H%MZ")
        / stream
        / f"lead_{lead:03d}.nc"
    )


def fetch(root: Path, target: datetime, stream: str, workers: int) -> None:
    """Extend the existing dataset with each required complete member shard."""
    catalog = json.loads((root / "ifs/v1/dataset.json").read_text())
    if catalog["schema"] != acquisition.SCHEMA:
        raise ValueError("unrecognized shared dataset schema")
    for issue, lead in schedule(target):
        acquisition.ifs(
            SimpleNamespace(
                start=issue.date().isoformat(),
                end=issue.date().isoformat(),
                hours=[issue.hour],
                leads=[lead],
                streams=[stream],
                root=root,
                corridor="atlantic",
                workers=workers,
            )
        )


def read_forecast(
    path: Path,
    target: datetime,
    members: list[int],
    issue: datetime,
    lead: int,
    stream: str,
) -> xr.Dataset:
    """Require a verified all51member source before selecting display members."""
    expected = dict(
        corridor="atlantic",
        initialization=issue.isoformat(),
        lead=lead,
        stream=stream,
        ensemble=True,
        variables=list(acquisition.PARAMS[stream].values()),
    )
    if not acquisition.completed(path, expected):
        raise ValueError(f"unverified forecast shard: {path}")
    with xr.open_dataset(path) as raw:
        np.testing.assert_array_equal(raw.valid_time.values, [np.datetime64(target)])
        return raw.sel(number=members).isel(step=0).load()


def read_reference(root: Path, target: datetime) -> xr.Dataset:
    """Read the same exact ERA5 valid time for both wind and wave fields."""
    fields = []
    for field in ("wind", "waves"):
        path = root / f"era5/v1/atlantic/{target:%Y/%m/%Y%m%d}_{field}.nc"
        receipt = json.loads(path.with_suffix(".json").read_text())
        if not acquisition.completed(path, receipt["request"]):
            raise ValueError(f"unverified reference: {path}")
        with xr.open_dataset(path) as raw:
            acquisition.require_final_era5(
                xr.Dataset(attrs=receipt["provenance"]["source_attributes"]),
                target.date(),
            )
            fields.append(raw.sel(valid_time=np.datetime64(target)).load())
    return xr.merge(fields, compat="no_conflicts")


def arrays(
    ds: xr.Dataset,
    land_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepare marine display/score arrays without modifying source values."""
    hs = ds[HS].values
    mask = np.isfinite(hs)
    if land_mask is not None:
        mask &= ~land_mask
    return (
        np.where(mask, hs, np.nan),
        np.where(mask, ds[U].values, np.nan),
        np.where(mask, ds[V].values, np.nan),
    )


def cell_edges(centres: np.ndarray) -> np.ndarray:
    """Locate cell boundaries halfway between the actual source coordinates."""
    if len(centres) < 2:
        raise ValueError("at least two coordinate centres are required")
    return np.r_[
        centres[0] - (centres[1] - centres[0]) / 2,
        (centres[:-1] + centres[1:]) / 2,
        centres[-1] + (centres[-1] - centres[-2]) / 2,
    ]


def detailed_land(latitude: np.ndarray, longitude: np.ndarray):
    """Use the earlier SWOPP Natural Earth 10m geometry at exact grid centres."""
    from cartopy.io import shapereader
    from shapely import contains_xy, make_valid, prepare
    from shapely.geometry import box
    from shapely.ops import unary_union

    source = Path(
        shapereader.natural_earth(resolution="10m", category="physical", name="land")
    )
    region = box(
        float(longitude.min()) - 1,
        float(latitude.min()) - 1,
        float(longitude.max()) + 1,
        float(latitude.max()) + 1,
    )
    geometries = []
    for geometry in shapereader.Reader(source).geometries():
        if geometry.intersects(region):
            part = make_valid(geometry).intersection(region)
            if not part.is_empty:
                geometries.append(part)
    land = unary_union(geometries)
    prepare(land)
    xx, yy = np.meshgrid(longitude, latitude)
    return geometries, contains_xy(land, xx, yy), source


def rmse(hs: np.ndarray, ref: np.ndarray, latitude: np.ndarray) -> float:
    """Calculate descriptive area-weighted wave error on paired ocean cells."""
    mask = np.isfinite(hs) & np.isfinite(ref)
    weights = np.broadcast_to(np.cos(np.deg2rad(latitude))[:, None], hs.shape)
    if not np.any(mask):
        raise ValueError("no common ocean cells")
    return float(
        np.sqrt(
            np.sum(weights[mask] * (hs[mask] - ref[mask]) ** 2) / np.sum(weights[mask])
        )
    )


def render(
    root: Path, target: datetime, output: Path, seed: int, preview: bool
) -> None:
    """Render the requested two-by-two ocean animation with one fixed scale."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.animation as animation
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from PIL import Image

    class FixedPaletteWriter(animation.PillowWriter):
        """Keep identical map colours identical throughout the GIF."""

        def finish(self):
            """Quantize every frame against one common colour palette."""
            palette = (
                self._frames[0]
                .convert("RGB")
                .quantize(colors=256, dither=Image.Dither.NONE)
            )
            frames = [
                frame.convert("RGB").quantize(palette=palette, dither=Image.Dither.NONE)
                for frame in self._frames
            ]
            frames[0].save(
                self.outfile,
                save_all=True,
                append_images=frames[1:],
                duration=round(1000 / self.fps),
                loop=0,
            )

    output.mkdir(parents=True, exist_ok=True)
    members = [
        0,
        *map(
            int, np.random.default_rng(seed).choice(np.arange(1, 51), 2, replace=False)
        ),
    ]
    times = schedule(target)[:1] if preview else schedule(target)
    reference = read_reference(root, target)
    lat, lon = reference.latitude.values, reference.longitude.values
    land_geometries, land_mask, land_source = detailed_land(lat, lon)
    frames = []
    source_records = []
    for issue, lead in times:
        paths = [shard(root, issue, stream, lead) for stream in ("enfo", "waef")]
        wind, wave = (
            read_forecast(path, target, members, issue, lead, stream)
            for path, stream in zip(paths, ("enfo", "waef"), strict=True)
        )
        np.testing.assert_array_equal(wind.number, wave.number)
        np.testing.assert_array_equal(wind.latitude, wave.latitude)
        np.testing.assert_array_equal(wind.longitude, wave.longitude)
        ds = xr.merge([wind, wave], compat="no_conflicts")
        np.testing.assert_array_equal(ds.latitude, lat)
        np.testing.assert_array_equal(ds.longitude, lon)
        frames.append(
            [arrays(ds.sel(number=m), land_mask) for m in (members[1], members[2], 0)]
        )
        source_records.append(
            dict(
                issue=issue.isoformat(),
                lead=lead,
                sources=[
                    dict(path=str(p), sha256=acquisition.digest(p)) for p in paths
                ],
            )
        )
    ref_arrays = arrays(reference, land_mask)
    maximum = max(float(np.nanmax(a[0])) for f in frames for a in f)
    maximum = max(maximum, float(np.nanmax(ref_arrays[0])))
    vmax = float(np.ceil(maximum / 2) * 2)
    projection = ccrs.PlateCarree()
    fig, axes = plt.subplots(
        2, 2, figsize=(14, 9), dpi=100, subplot_kw=dict(projection=projection)
    )
    fig.subplots_adjust(
        left=0.045, right=0.985, bottom=0.20, top=0.76, hspace=0.24, wspace=0.08
    )
    fig.suptitle(
        f"Forecast revisions for {target:%d %B %Y, %H:%M} UTC", fontsize=18, y=0.98
    )
    issue_text = fig.text(0.5, 0.925, "", ha="center", fontsize=14)
    timeline = fig.add_axes([0.12, 0.81, 0.76, 0.075])
    timeline.set(xlim=(-12, 372), ylim=(-0.8, 0.9))
    timeline.axis("off")
    timeline.plot([0, 360], [0, 0], color="#b6bdc4", lw=2)
    for elapsed in range(0, 361, 12):
        timeline.plot([elapsed, elapsed], [-0.06, 0.06], color="#b6bdc4", lw=0.8)
    for elapsed in range(0, 361, 72):
        date = target - timedelta(hours=360 - elapsed)
        label = date.strftime("%d %b") + ("\nTarget" if elapsed == 360 else "")
        timeline.text(elapsed, -0.22, label, ha="center", va="top", fontsize=9)
    (progress,) = timeline.plot([0, 0], [0, 0], color="#246a91", lw=3)
    current = timeline.annotate(
        "",
        xy=(0, 0),
        xytext=(0, 0.6),
        arrowprops=dict(arrowstyle="-|>", color="#246a91", lw=1.6),
    )
    labels = [
        f"Perturbed member {members[1]}",
        f"Perturbed member {members[2]}",
        "Control (deterministic forecast)",
        "ERA5 reanalysis · fixed reference",
    ]
    meshes, quivers = [], []
    arrow_slice = (slice(None, None, 10), slice(None, None, 15))
    xx, yy = np.meshgrid(lon[::15], lat[::10])
    for i, ax in enumerate(axes.flat):
        ax.set_extent([-80, 10, 25, 60], crs=projection)
        ax.set_facecolor("#cfdfec")
        ax.add_feature(
            cfeature.ShapelyFeature(land_geometries, projection),
            facecolor="#e7e9eb",
            edgecolor="#5f666b",
            linewidth=0.35,
            zorder=3,
        )
        gridlines = ax.gridlines(
            draw_labels=True, linewidth=0.3, alpha=0.3, color="#5f666b"
        )
        gridlines.top_labels = False
        gridlines.right_labels = False
        gridlines.bottom_labels = i >= 2
        gridlines.left_labels = i % 2 == 0
        gridlines.xlabel_style = dict(size=9)
        gridlines.ylabel_style = dict(size=9)
        data = frames[0][i] if i < 3 else ref_arrays
        mesh = ax.pcolormesh(
            cell_edges(lon),
            cell_edges(lat),
            data[0],
            shading="flat",
            cmap="viridis",
            vmin=0,
            vmax=vmax,
            transform=projection,
            zorder=1,
        )
        q = ax.quiver(
            xx,
            yy,
            data[1][arrow_slice],
            data[2][arrow_slice],
            color="white",
            width=0.0021,
            scale=550,
            alpha=0.9,
            transform=projection,
            zorder=5,
        )
        q.set_path_effects([pe.withStroke(linewidth=0.65, foreground="#20262c")])
        ax.set_title(labels[i], fontsize=12, pad=9)
        meshes.append(mesh)
        quivers.append(q)
    fig.legend(
        handles=[
            Patch(
                facecolor="#e7e9eb",
                edgecolor="#5f666b",
                label="Land (detailed coastline)",
            ),
            Patch(facecolor="#cfdfec", label="Ocean: no wave data"),
        ],
        loc="center",
        bbox_to_anchor=(0.5, 0.16),
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    cax = fig.add_axes([0.18, 0.125, 0.52, 0.018])
    bar = fig.colorbar(meshes[0], cax=cax, orientation="horizontal")
    bar.set_label(
        "Significant wave height (m) · same colour scale for every panel and frame",
        fontsize=10,
    )
    axes.flat[3].quiverkey(
        quivers[-1],
        0.85,
        0.135,
        10,
        "10 m/s wind",
        coordinates="figure",
        color="#20262c",
        labelpos="E",
        fontproperties=dict(size=10),
    )
    fig.text(
        0.5,
        0.045,
        "Arrows point toward wind motion. Member IDs are local to each issue; "
        "these are not persistent scenario tracks.",
        ha="center",
        fontsize=10,
    )
    fig.text(
        0.5,
        0.018,
        "Target time stays fixed. ERA5 is a retrospective reference. "
        "Initialization is earlier than publication. No time interpolation.",
        ha="center",
        fontsize=9,
        color="#505860",
    )
    error_rows = []
    for (issue, lead), data in zip(times, frames, strict=True):
        common = np.isfinite(ref_arrays[0])
        for a in data:
            common &= np.isfinite(a[0])
        error_rows.append(
            dict(
                issue=issue.isoformat(),
                lead=lead,
                target=target.isoformat(),
                common_ocean_cells=int(common.sum()),
                **{
                    f"member_{member}_wave_rmse_m": rmse(
                        np.where(common, a[0], np.nan), ref_arrays[0], lat
                    )
                    for member, a in zip((members[1], members[2], 0), data, strict=True)
                },
            )
        )

    def update(index):
        issue, lead = times[index]
        days, hours = divmod(lead, 24)
        issue_text.set_text(
            f"Forecast initialization {issue:%d %b %Y, %H:%M} UTC"
            f"   |   Target in {days} d {hours:02d} h"
        )
        elapsed = 360 - lead
        progress.set_data([0, elapsed], [0, 0])
        current.xy = (elapsed, 0)
        current.set_position((elapsed, 0.6))
        for i, data in enumerate(frames[index]):
            meshes[i].set_array(data[0].ravel())
            quivers[i].set_UVC(data[1][arrow_slice], data[2][arrow_slice])
        return [*meshes[:3], *quivers[:3], issue_text, progress, current]

    update(0)
    stem = f"fixed_target_{target:%Y%m%dT%H%MZ}"
    fig.savefig(output / f"{stem}_first.png", dpi=140)
    if not preview:
        # Discrete archived issues only; no invented smoothing between forecasts.
        indices = list(range(len(times))) + [len(times) - 1] * 3
        anim = animation.FuncAnimation(
            fig, update, frames=indices, interval=800, blit=False, repeat=False
        )
        anim.save(output / f"{stem}.gif", writer=FixedPaletteWriter(fps=1.25))
        update(len(times) // 2)
        fig.savefig(output / f"{stem}_middle.png", dpi=140)
        update(len(times) - 1)
        fig.savefig(output / f"{stem}_last.png", dpi=140)
        import csv

        with (output / f"{stem}_errors.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=error_rows[0].keys())
            writer.writeheader()
            writer.writerows(error_rows)
    acquisition.atomic_json(
        output / f"{stem}_manifest.json",
        dict(
            target=target.isoformat(),
            issue_count=len(times),
            fixed_valid_time=True,
            members=members,
            random_seed=seed,
            wave_vmin=0,
            wave_vmax=vmax,
            color="significant wave height (m)",
            arrows="10m u/v (m/s), toward direction, fixed scale",
            reference_hashes={
                field: acquisition.digest(
                    root / f"era5/v1/atlantic/{target:%Y/%m/%Y%m%d}_{field}.nc"
                )
                for field in ("wind", "waves")
            },
            coastline=dict(
                source=str(land_source),
                sha256=acquisition.digest(land_source),
                resolution="Natural Earth 1:10 million (10m)",
                coordinate_shift_applied=False,
                rendering=(
                    "explicit source-grid cell edges; polygon land "
                    "separate from no-wave-data ocean"
                ),
            ),
            score_scope=(
                "area-weighted wave RMSE on the common finite cells of all "
                "three forecasts and ERA5, excluding Natural Earth land; "
                "cell counts are recorded per initialization"
            ),
            reference_axes_bounds=list(axes.flat[3].get_position().bounds),
            source_records=source_records,
            script_sha256=acquisition.digest(Path(__file__)),
            caveats=[
                "member numbers do not identify continuous scenarios across issues",
                "no monotonic improvement is assumed",
                "ERA5 is not exact observed truth",
            ],
        ),
    )
    plt.close(fig)


def main() -> None:
    """Acquire exact-target slices or render their four-panel animation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("/home/SHARED/data/weather-routing")
    )
    parser.add_argument("--target", default="2025-01-30T00:00:00")
    parser.add_argument("--fetch", choices=("enfo", "waef"))
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=4)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--seed", type=int, default=20250130)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    target = datetime.fromisoformat(args.target)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    if args.fetch:
        fetch(args.root, target, args.fetch, args.workers)
    else:
        output = (
            args.output
            or args.root / "diagnostics" / f"fixed_target_{target:%Y%m%dT%H%MZ}"
        )
        render(args.root, target, output, args.seed, args.preview)


if __name__ == "__main__":
    main()
