#!/usr/bin/env python
"""Animate a single SWOPP3 optimised departure as a GIF.

Example
-------
    uv run scripts/swopp3_animate_single_departure.py \
        --case-id AO_WPS \
        --departure-number 7
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated

import jax
import jax.numpy as jnp
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import typer
import xarray as xr
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from routetools.cmaes import optimize as cmaes_optimize
from routetools.cost import cost_function_rise
from routetools.era5.loader import (
    load_dataset_epoch,
    load_era5_wavefield,
    load_era5_windfield,
    load_natural_earth_land_mask,
)
from routetools.fms import optimize_fms
from routetools.swopp3 import (
    SWOPP3_CASES,
    case_endpoints,
    departures_2024,
    great_circle_route,
)
from routetools.swopp3_output import sailed_distance_nm
from routetools.swopp3_runner import evaluate_energy
from routetools.weather import (
    DEFAULT_HS_LIMIT,
    DEFAULT_TWS_LIMIT,
    weather_penalty_smooth,
)

app = typer.Typer(help="Animate a single optimised SWOPP3 departure.")

_ERA5_FILE_RE = re.compile(
    r"^(?P<prefix>era5_[^_]+_[^_]+_)(?P<year>\d{4})(?:_(?P<suffix>\d{2}(?:-\d{2})?))?\.nc$"
)


@dataclass(frozen=True)
class CaseContext:
    """Resolved SWOPP3 inputs for one animated departure."""

    case_id: str
    departure_number: int
    departure: datetime
    departure_offset_h: float
    passage_hours: float
    wps: bool
    src: jnp.ndarray
    dst: jnp.ndarray
    gc_curve: jnp.ndarray
    vectorfield: object
    windfield: object
    wavefield: object
    land: object


@dataclass(frozen=True)
class CmaesFrame:
    """One rendered CMA-ES iteration with population and best-route data."""

    iteration: int
    population_curves: np.ndarray
    generation_best_curve: np.ndarray
    generation_best_objective: float
    generation_best_distance_nm: float
    best_curve: np.ndarray
    best_objective: float
    best_distance_nm: float


@dataclass(frozen=True)
class CmaesPauseFrame:
    """Hold frame shown between the CMA-ES and FMS stages."""

    hold_index: int
    hold_total: int
    best_curve: np.ndarray
    best_objective: float
    best_distance_nm: float


@dataclass(frozen=True)
class FmsFrame:
    """One rendered FMS iteration with current and best-so-far routes."""

    iteration: int
    current_curve: np.ndarray
    current_objective: float
    current_distance_nm: float
    best_curve: np.ndarray
    best_objective: float
    best_distance_nm: float


type AnimationFrame = CmaesFrame | CmaesPauseFrame | FmsFrame


def _loadable_era5_paths(path: Path) -> list[Path]:
    """Return the base ERA5 file plus any next-year continuation files."""
    match = _ERA5_FILE_RE.match(path.name)
    if match is None:
        return [path]

    prefix = match.group("prefix")
    next_year = int(match.group("year")) + 1
    exact_next_year = path.with_name(f"{prefix}{next_year}.nc")
    if exact_next_year.exists():
        return [path, exact_next_year]

    continuation_paths = sorted(path.parent.glob(f"{prefix}{next_year}_*.nc"))
    return [path, *continuation_paths]


def _default_era5_path(corridor: str, field: str) -> Path:
    return Path(f"data/era5/era5_{field}_{corridor}_2024.nc")


def _to_numpy(value) -> np.ndarray:
    return np.asarray(jax.device_get(value), dtype=float)


def _distance_nm(curve: np.ndarray) -> float:
    return float(sailed_distance_nm(jnp.asarray(curve)))


def _sample_land_extent(path: Path) -> tuple[tuple[float, float], tuple[float, float]]:
    with xr.open_dataset(path) as ds:
        for coord_name in ("longitude", "lon"):
            if coord_name in ds.coords:
                lons = ds[coord_name].values
                break
        else:
            raise KeyError(f"No longitude coordinate found in {path}")

        for coord_name in ("latitude", "lat"):
            if coord_name in ds.coords:
                lats = ds[coord_name].values
                break
        else:
            raise KeyError(f"No latitude coordinate found in {path}")

    return (float(np.min(lons)), float(np.max(lons))), (
        float(np.min(lats)),
        float(np.max(lats)),
    )


def _load_case_context(
    *,
    case_id: str,
    departure_number: int,
    n_points: int,
    wind_path: Path | None,
    wave_path: Path | None,
) -> CaseContext:
    if case_id not in SWOPP3_CASES:
        raise typer.BadParameter(f"Unknown case_id: {case_id}")

    case = SWOPP3_CASES[case_id]
    if case["strategy"] != "optimised":
        raise typer.BadParameter(
            f"Case {case_id} uses strategy={case['strategy']!r}; "
            "choose an optimised case."
        )

    departures = departures_2024()
    if departure_number < 1 or departure_number > len(departures):
        raise typer.BadParameter(
            f"departure_number must be between 1 and {len(departures)}"
        )

    corridor = str(case["route"])
    resolved_wind_path = wind_path or _default_era5_path(corridor, "wind")
    resolved_wave_path = wave_path or _default_era5_path(corridor, "waves")
    if not resolved_wind_path.exists():
        raise FileNotFoundError(f"Wind dataset not found: {resolved_wind_path}")
    if not resolved_wave_path.exists():
        raise FileNotFoundError(f"Wave dataset not found: {resolved_wave_path}")

    wind_paths = _loadable_era5_paths(resolved_wind_path)
    wave_paths = _loadable_era5_paths(resolved_wave_path)
    wind_target = wind_paths if len(wind_paths) > 1 else wind_paths[0]
    wave_target = wave_paths if len(wave_paths) > 1 else wave_paths[0]

    typer.echo("Loading wind field from " + ", ".join(str(path) for path in wind_paths))
    windfield = load_era5_windfield(wind_target)
    dataset_epoch = load_dataset_epoch(wind_target)

    typer.echo("Loading wave field from " + ", ".join(str(path) for path in wave_paths))
    wavefield = load_era5_wavefield(wave_target)

    lon_range, lat_range = _sample_land_extent(wave_paths[0])
    typer.echo(f"Building Natural Earth land mask for lon={lon_range}, lat={lat_range}")
    land = load_natural_earth_land_mask(lon_range, lat_range)

    departure = departures[departure_number - 1]
    departure_offset_h = (
        departure.replace(tzinfo=None) - dataset_epoch.replace(tzinfo=None)
    ).total_seconds() / 3600.0

    src, dst = case_endpoints(case_id)
    gc_curve = great_circle_route(src, dst, n_points=n_points)
    return CaseContext(
        case_id=case_id,
        departure_number=departure_number,
        departure=departure,
        departure_offset_h=departure_offset_h,
        passage_hours=float(case["passage_hours"]),
        wps=bool(case["wps"]),
        src=src,
        dst=dst,
        gc_curve=gc_curve,
        vectorfield=windfield,
        windfield=windfield,
        wavefield=wavefield,
        land=land,
    )


def _bounds_from_frames(
    *,
    land,
    gc_curve: np.ndarray,
    cma_frames: list[CmaesFrame],
    fms_frames: list[FmsFrame],
    cma_best_curve: np.ndarray,
) -> tuple[tuple[float, float], tuple[float, float]]:
    xmin = float(np.min(gc_curve[:, 0]))
    xmax = float(np.max(gc_curve[:, 0]))
    ymin = float(np.min(gc_curve[:, 1]))
    ymax = float(np.max(gc_curve[:, 1]))

    def update(curve: np.ndarray) -> None:
        nonlocal xmin, xmax, ymin, ymax
        xmin = min(xmin, float(np.min(curve[..., 0])))
        xmax = max(xmax, float(np.max(curve[..., 0])))
        ymin = min(ymin, float(np.min(curve[..., 1])))
        ymax = max(ymax, float(np.max(curve[..., 1])))

    update(cma_best_curve)
    for frame in cma_frames:
        update(frame.population_curves)
        update(frame.best_curve)
    for frame in fms_frames:
        update(frame.current_curve)
        update(frame.best_curve)

    xpad = max(1.5, 0.06 * (xmax - xmin))
    ypad = max(1.0, 0.08 * (ymax - ymin))
    return (
        max(float(land.xmin), xmin - xpad),
        min(float(land.xmax), xmax + xpad),
    ), (
        max(float(land.ymin), ymin - ypad),
        min(float(land.ymax), ymax + ypad),
    )


def _downsample_fms_frames(frames: list[FmsFrame], step: int) -> list[FmsFrame]:
    if step <= 1 or len(frames) <= 2:
        return frames

    selected = [frames[0]]
    selected.extend(frames[idx] for idx in range(step, len(frames) - 1, step))
    if selected[-1] is not frames[-1]:
        selected.append(frames[-1])
    return selected


def _render_gif(
    *,
    context: CaseContext,
    cma_frames: list[CmaesFrame],
    fms_frames: list[FmsFrame],
    cma_best_curve: np.ndarray,
    cma_best_objective: float,
    cma_best_distance_nm: float,
    output_path: Path,
    fps: int,
    cma_hold_seconds: float,
) -> None:
    hold_frames = max(1, int(round(cma_hold_seconds * fps)))
    all_frames: list[AnimationFrame] = list(cma_frames)
    all_frames.extend(
        CmaesPauseFrame(
            hold_index=hold_idx + 1,
            hold_total=hold_frames,
            best_curve=cma_best_curve,
            best_objective=cma_best_objective,
            best_distance_nm=cma_best_distance_nm,
        )
        for hold_idx in range(hold_frames)
    )
    all_frames.extend(fms_frames)

    gc_curve = _to_numpy(context.gc_curve)
    xlim, ylim = _bounds_from_frames(
        land=context.land,
        gc_curve=gc_curve,
        cma_frames=cma_frames,
        fms_frames=fms_frames,
        cma_best_curve=cma_best_curve,
    )

    fig = plt.figure(figsize=(13.5, 7.5), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[3.0, 1.35])
    ax_route = fig.add_subplot(grid[0, 0])
    ax_info = fig.add_subplot(grid[0, 1])

    ax_route.contourf(
        context.land.x,
        context.land.y,
        context.land.array.T,
        levels=[0.0, 0.5, 1.0],
        colors=["#f8f9fa", "#adb5bd"],
        alpha=1.0,
        zorder=0,
    )
    (gc_line,) = ax_route.plot(
        gc_curve[:, 0],
        gc_curve[:, 1],
        linestyle="--",
        color="#868e96",
        linewidth=1.5,
        alpha=0.9,
        zorder=1,
    )
    population_collection = LineCollection([], colors="#74c0fc", linewidths=0.8)
    population_collection.set_alpha(0.14)
    population_collection.set_zorder(2)
    ax_route.add_collection(population_collection)
    (cma_line,) = ax_route.plot([], [], color="#f08c00", linewidth=2.6, zorder=4)
    (fms_line,) = ax_route.plot([], [], color="#1971c2", linewidth=2.6, zorder=5)
    ax_route.scatter(
        [float(gc_curve[0, 0])],
        [float(gc_curve[0, 1])],
        color="#2f9e44",
        s=60,
        zorder=6,
    )
    ax_route.scatter(
        [float(gc_curve[-1, 0])],
        [float(gc_curve[-1, 1])],
        color="#c92a2a",
        s=60,
        zorder=6,
    )
    title = ax_route.set_title("")
    ax_route.set_xlabel("Longitude [deg]")
    ax_route.set_ylabel("Latitude [deg]")
    ax_route.set_xlim(*xlim)
    ax_route.set_ylim(*ylim)
    ax_route.set_aspect("equal", adjustable="box")
    ax_route.grid(alpha=0.25)
    ax_route.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color="#868e96",
                linestyle="--",
                linewidth=1.5,
                label="Great-circle init",
            ),
            Line2D(
                [0],
                [0],
                color="#74c0fc",
                linewidth=1.5,
                alpha=0.45,
                label="CMA-ES population",
            ),
            Line2D(
                [0],
                [0],
                color="#f08c00",
                linewidth=2.6,
                label="Highlighted CMA-ES route",
            ),
            Line2D([0], [0], color="#1971c2", linewidth=2.6, label="FMS route"),
        ],
        loc="lower right",
        framealpha=0.95,
    )

    ax_info.axis("off")
    info_text = ax_info.text(
        0.0,
        1.0,
        "",
        va="top",
        ha="left",
        family="monospace",
        fontsize=11,
    )

    header = (
        f"Case: {context.case_id}\n"
        f"Departure: #{context.departure_number:03d}  {context.departure:%Y-%m-%d}\n"
        f"Passage: {context.passage_hours:.0f} h\n"
        f"WPS: {'on' if context.wps else 'off'}\n"
    )

    cma_total = cma_frames[-1].iteration if cma_frames else 0
    fms_total = fms_frames[-1].iteration if fms_frames else 0

    def animate(frame_index: int):
        frame = all_frames[frame_index]
        if isinstance(frame, CmaesFrame):
            population_collection.set_segments(list(frame.population_curves))
            cma_line.set_data(
                frame.generation_best_curve[:, 0],
                frame.generation_best_curve[:, 1],
            )
            fms_line.set_data([], [])
            title.set_text(
                f"SWOPP3 single departure | CMA-ES {frame.iteration}/{cma_total}"
            )
            info_text.set_text(
                header
                + "\n"
                + "Stage: CMA-ES\n"
                + f"Iteration: {frame.iteration}/{cma_total}\n"
                + f"Population: {frame.population_curves.shape[0]} routes\n"
                + "\n"
                + "Highlighted route\n"
                + f"  Objective: {frame.generation_best_objective:.3f}\n"
                + f"  Distance : {frame.generation_best_distance_nm:.1f} nm\n"
                + "\n"
                + "Best so far\n"
                + f"  Objective: {frame.best_objective:.3f}\n"
                + f"  Distance : {frame.best_distance_nm:.1f} nm\n"
            )
        elif isinstance(frame, CmaesPauseFrame):
            population_collection.set_segments([])
            cma_line.set_data(frame.best_curve[:, 0], frame.best_curve[:, 1])
            fms_line.set_data([], [])
            title.set_text("SWOPP3 single departure | CMA-ES best route")
            info_text.set_text(
                header
                + "\n"
                + "Stage: CMA-ES complete\n"
                + f"Hold: {frame.hold_index}/{frame.hold_total}\n"
                + "\n"
                + "Best CMA-ES route\n"
                + f"  Objective: {frame.best_objective:.3f}\n"
                + f"  Distance : {frame.best_distance_nm:.1f} nm\n"
            )
        else:
            population_collection.set_segments([])
            cma_line.set_data(cma_best_curve[:, 0], cma_best_curve[:, 1])
            fms_line.set_data(frame.current_curve[:, 0], frame.current_curve[:, 1])
            title.set_text(
                f"SWOPP3 single departure | FMS {frame.iteration}/{fms_total}"
            )
            info_text.set_text(
                header
                + "\n"
                + "Stage: FMS\n"
                + f"Iteration: {frame.iteration}/{fms_total}\n"
                + "\n"
                + "CMA-ES baseline\n"
                + f"  Objective: {cma_best_objective:.3f}\n"
                + f"  Distance : {cma_best_distance_nm:.1f} nm\n"
                + "\n"
                + "FMS current\n"
                + f"  Objective: {frame.current_objective:.3f}\n"
                + f"  Distance : {frame.current_distance_nm:.1f} nm\n"
                + "\n"
                + "FMS best so far\n"
                + f"  Objective: {frame.best_objective:.3f}\n"
                + f"  Distance : {frame.best_distance_nm:.1f} nm\n"
            )

        return [gc_line, population_collection, cma_line, fms_line, info_text]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=len(all_frames),
        interval=1000 / fps,
        blit=False,
    )
    anim.save(output_path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)


@app.command()
def main(
    case_id: str = typer.Option(
        "AO_WPS", "--case-id", help="Optimised SWOPP3 case ID."
    ),
    departure_number: int = typer.Option(
        1,
        "--departure-number",
        min=1,
        help="1-based departure number within the 2024 schedule.",
    ),
    output_path: Annotated[
        Path | None,
        typer.Option(
            "--output-path",
            help="GIF output path. Defaults to output/figures/<case>-dep<nnn>.gif",
        ),
    ] = None,
    wind_path: Annotated[
        Path | None,
        typer.Option(
            "--wind-path",
            help="Override the default ERA5 wind NetCDF path for the case corridor.",
        ),
    ] = None,
    wave_path: Annotated[
        Path | None,
        typer.Option(
            "--wave-path",
            help="Override the default ERA5 wave NetCDF path for the case corridor.",
        ),
    ] = None,
    n_points: int = typer.Option(100, "--n-points", help="Number of route waypoints."),
    fps: int = typer.Option(8, "--fps", min=1, help="GIF frame rate."),
    cma_hold_seconds: float = typer.Option(
        1.0,
        "--cma-hold-seconds",
        min=0.1,
        help="Seconds to hold the best CMA-ES route before starting FMS.",
    ),
    fms_frame_step: int = typer.Option(
        5,
        "--fms-frame-step",
        min=1,
        help="Render every Nth FMS iteration to keep the GIF manageable.",
    ),
    cma_popsize: int = typer.Option(200, "--cma-popsize", min=2),
    cma_sigma0: float = typer.Option(1.0, "--cma-sigma0", min=0.01),
    cma_tolfun: float = typer.Option(1e-4, "--cma-tolfun", min=0.0),
    cma_damping: float = typer.Option(1.0, "--cma-damping", min=0.0),
    cma_maxfevals: int = typer.Option(25000, "--cma-maxfevals", min=1),
    fms_patience: int = typer.Option(200, "--fms-patience", min=1),
    fms_damping: float = typer.Option(0.95, "--fms-damping", min=0.0),
    fms_maxfevals: int = typer.Option(10000, "--fms-maxfevals", min=1),
    tws_limit: float = typer.Option(DEFAULT_TWS_LIMIT, "--tws-limit", min=0.1),
    hs_limit: float = typer.Option(DEFAULT_HS_LIMIT, "--hs-limit", min=0.1),
) -> None:
    """Run one SWOPP3 optimised departure and export an animation GIF."""
    context = _load_case_context(
        case_id=case_id,
        departure_number=departure_number,
        n_points=n_points,
        wind_path=wind_path,
        wave_path=wave_path,
    )

    if output_path is None:
        output_path = Path(
            f"output/figures/swopp3_single_{case_id}_dep{departure_number:03d}.gif"
        )

    travel_time = float(context.passage_hours)
    gc_init = context.gc_curve
    src_opt = jnp.array([gc_init[0, 0], gc_init[0, 1]])
    dst_opt = jnp.array([gc_init[-1, 0], gc_init[-1, 1]])

    def cma_cost(curve_batch: jnp.ndarray) -> jnp.ndarray:
        return cost_function_rise(
            curve=curve_batch,
            windfield=context.windfield,
            wavefield=context.wavefield,
            travel_time=travel_time,
            wps=context.wps,
            time_offset=context.departure_offset_h,
        ) + weather_penalty_smooth(
            curve_batch,
            windfield=context.windfield,
            wavefield=context.wavefield,
            travel_time=travel_time,
            spherical_correction=True,
            time_offset=context.departure_offset_h,
        )

    cma_frames: list[CmaesFrame] = []

    def on_cma(snapshot) -> None:
        population_curves = _to_numpy(snapshot["population_curves"])
        generation_best_curve = _to_numpy(snapshot["generation_best_curve"])
        best_curve = _to_numpy(snapshot["best_curve"])
        cma_frames.append(
            CmaesFrame(
                iteration=int(snapshot["iteration"]),
                population_curves=population_curves,
                generation_best_curve=generation_best_curve,
                generation_best_objective=float(snapshot["generation_best_cost"]),
                generation_best_distance_nm=_distance_nm(generation_best_curve),
                best_curve=best_curve,
                best_objective=float(snapshot["best_cost"]),
                best_distance_nm=_distance_nm(best_curve),
            )
        )

    typer.echo(
        "Running CMA-ES for "
        f"{case_id} departure #{departure_number:03d} "
        f"({context.departure:%Y-%m-%d})"
    )
    curve_cmaes, dict_cmaes = cmaes_optimize(
        vectorfield=context.vectorfield,
        src=src_opt,
        dst=dst_opt,
        curve0=gc_init,
        land=context.land,
        cost_fn=cma_cost,
        penalty=1e6,
        land_margin=2,
        windfield=context.windfield,
        wavefield=context.wavefield,
        spherical_correction=True,
        travel_time=travel_time,
        time_offset=context.departure_offset_h,
        land_distance_weight=50.0,
        L=n_points,
        popsize=cma_popsize,
        sigma0=cma_sigma0,
        tolfun=cma_tolfun,
        damping=cma_damping,
        maxfevals=cma_maxfevals,
        verbose=False,
        snapshot_callback=on_cma,
    )
    cma_best_curve = _to_numpy(curve_cmaes)
    cma_best_objective = float(dict_cmaes["cost"])
    cma_best_distance_nm = _distance_nm(cma_best_curve)

    fms_frames_all: list[FmsFrame] = []

    def on_fms(snapshot) -> None:
        current_curve = _to_numpy(snapshot["curve"])[0]
        best_curve = _to_numpy(snapshot["best_curve"])[0]
        current_cost = float(_to_numpy(snapshot["cost"])[0])
        best_cost = float(_to_numpy(snapshot["best_cost"])[0])
        fms_frames_all.append(
            FmsFrame(
                iteration=int(snapshot["iteration"]),
                current_curve=current_curve,
                current_objective=current_cost,
                current_distance_nm=_distance_nm(current_curve),
                best_curve=best_curve,
                best_objective=best_cost,
                best_distance_nm=_distance_nm(best_curve),
            )
        )

    typer.echo("Running FMS refinement")
    curve_fms_batch, dict_fms = optimize_fms(
        vectorfield=context.vectorfield,
        curve=curve_cmaes,
        land=context.land,
        windfield=context.windfield,
        wavefield=context.wavefield,
        travel_time=travel_time,
        spherical_correction=True,
        time_offset=context.departure_offset_h,
        enforce_weather_limits=True,
        tws_limit=tws_limit,
        hs_limit=hs_limit,
        costfun=cost_function_rise,
        costfun_kwargs={
            "windfield": context.windfield,
            "wavefield": context.wavefield,
            "wps": context.wps,
        },
        patience=fms_patience,
        damping=fms_damping,
        maxfevals=fms_maxfevals,
        verbose=False,
        snapshot_callback=on_fms,
    )
    curve_fms = _to_numpy(curve_fms_batch)[0]
    fms_frames = _downsample_fms_frames(fms_frames_all, fms_frame_step)
    fms_best_curve = fms_frames_all[-1].best_curve if fms_frames_all else curve_fms
    fms_best_objective = (
        fms_frames_all[-1].best_objective
        if fms_frames_all
        else float(np.asarray(dict_fms["cost"])[0])
    )
    fms_best_distance_nm = _distance_nm(fms_best_curve)

    typer.echo(f"Rendering GIF to {output_path}")
    _render_gif(
        context=context,
        cma_frames=cma_frames,
        fms_frames=fms_frames,
        cma_best_curve=cma_best_curve,
        cma_best_objective=cma_best_objective,
        cma_best_distance_nm=cma_best_distance_nm,
        output_path=output_path,
        fps=fps,
        cma_hold_seconds=cma_hold_seconds,
    )

    cma_energy_mwh, _, _ = evaluate_energy(
        jnp.asarray(cma_best_curve),
        context.departure,
        context.passage_hours,
        wps=context.wps,
        windfield=context.windfield,
        wavefield=context.wavefield,
        departure_offset_h=context.departure_offset_h,
    )
    fms_energy_mwh, _, _ = evaluate_energy(
        jnp.asarray(fms_best_curve),
        context.departure,
        context.passage_hours,
        wps=context.wps,
        windfield=context.windfield,
        wavefield=context.wavefield,
        departure_offset_h=context.departure_offset_h,
    )

    typer.echo(
        f"Saved {output_path}\n"
        "CMA-ES: "
        f"objective={cma_best_objective:.3f}, "
        f"distance={cma_best_distance_nm:.1f} nm, "
        f"energy={cma_energy_mwh:.2f} MWh\n"
        "FMS best: "
        f"objective={fms_best_objective:.3f}, "
        f"distance={fms_best_distance_nm:.1f} nm, "
        f"energy={fms_energy_mwh:.2f} MWh"
    )


if __name__ == "__main__":
    app()
