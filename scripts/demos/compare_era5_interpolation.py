#!/usr/bin/env python
"""Animate ERA5 wind forecasts at several temporal resolutions.

The native 3-hour field is the reference.  Coarser fields are created by
retaining every 24-, 12-, 6-, or 3-hour timestep and cubic-spline interpolation
is used to evaluate all fields at every 3-hour frame.

Example
-------
uv run scripts/demos/compare_era5_interpolation.py \
    --wind-path data/era5/era5_wind_atlantic_2024.nc \
    --departure 2024-01-01T00:00:00
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import PillowWriter

from routetools.era5.loader import (
    load_era5_windfield,
    temporal_stride_for_resolution,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare 24-, 12-, 6-, and 3-hour ERA5 interpolation."
    )
    parser.add_argument("--wind-path", type=Path, required=True)
    parser.add_argument("--departure", default="2024-01-01T00:00:00")
    parser.add_argument("--lon-min", type=float, default=-80.0)
    parser.add_argument("--lon-max", type=float, default=10.0)
    parser.add_argument("--lat-min", type=float, default=25.0)
    parser.add_argument("--lat-max", type=float, default=60.0)
    parser.add_argument("--grid-size", type=int, default=100)
    parser.add_argument("--voyage-hours", type=float, default=72.0)
    parser.add_argument("--frame-step-hours", type=float, default=3.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/era5_interpolation.gif"),
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=Path("output/era5_interpolation_rmse.csv"),
    )
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--order", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    """Generate the interpolation comparison animation and RMSE table."""
    args = _parse_args()
    if not args.wind_path.exists():
        raise FileNotFoundError(
            f"Missing wind file: {args.wind_path}\n"
            "Download ERA5 data first with scripts/data/download_era5.py."
        )

    resolution_h = (24, 12, 6, 3)
    fields = {}
    for target_h in resolution_h:
        stride = temporal_stride_for_resolution(3.0, float(target_h))
        fields[target_h] = load_era5_windfield(
            args.wind_path,
            departure_time=args.departure,
            voyage_hours=args.voyage_hours,
            temporal_stride=stride,
            order=args.order if target_h > 3 else 1,
        )

    lon = np.linspace(args.lon_min, args.lon_max, args.grid_size)
    lat = np.linspace(args.lat_min, args.lat_max, args.grid_size)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    lon_jax = jnp.asarray(lon_grid)
    lat_jax = jnp.asarray(lat_grid)

    frame_times = np.arange(
        0.0,
        args.voyage_hours + 0.1,
        args.frame_step_hours,
        dtype=float,
    )
    all_magnitude = {}
    rmse = {target_h: [] for target_h in resolution_h if target_h > 3}
    for target_h in resolution_h:
        target_magnitudes = []
        for frame_h in frame_times:
            u, v = fields[target_h](
                lon_jax,
                lat_jax,
                jnp.asarray(frame_h, dtype=jnp.float32),
            )
            magnitude = np.asarray(jnp.sqrt(u**2 + v**2) * 1.94384)
            target_magnitudes.append(magnitude)
        all_magnitude[target_h] = np.asarray(target_magnitudes)

    reference = all_magnitude[3]
    for target_h in rmse:
        difference = all_magnitude[target_h] - reference
        rmse[target_h] = np.sqrt(np.mean(difference**2, axis=(1, 2))).tolist()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_output.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["time_h", "resolution_h", "rmse_knots"])
        for target_h in rmse:
            for frame_h, value in zip(frame_times, rmse[target_h], strict=True):
                writer.writerow([frame_h, target_h, value])

    vmax = max(
        float(np.nanpercentile(reference, 99)),
        *(
            float(np.nanpercentile(all_magnitude[target_h], 99))
            for target_h in resolution_h
        ),
    )
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 9),
        constrained_layout=True,
        subplot_kw={"aspect": "equal"},
    )
    axes = axes.ravel()
    titles = [
        "Real 3-hour forecast",
        "24-hour",
        "12-hour",
        "6-hour",
        "3-hour field",
        "RMSE",
    ]

    images = []
    for ax, target_h, title in zip(
        axes[:5],
        (3, 24, 12, 6, 3),
        titles[:5],
        strict=True,
    ):
        image = ax.pcolormesh(
            lon_grid,
            lat_grid,
            all_magnitude[target_h][0],
            shading="auto",
            cmap="viridis",
            vmin=0,
            vmax=vmax,
        )
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_xlim(args.lon_min, args.lon_max)
        ax.set_ylim(args.lat_min, args.lat_max)
        images.append(image)
    fig.colorbar(images[0], ax=axes[:5], label="Wind speed (kn)", shrink=0.86)

    rmse_ax = axes[5]
    rmse_lines = []
    for target_h in rmse:
        line, = rmse_ax.plot([], [], marker="o", label=f"{target_h} h")
        rmse_lines.append(line)
    rmse_ax.set_xlim(frame_times[0], frame_times[-1])
    rmse_ax.set_ylim(0, max(1.0, max(max(values) for values in rmse.values()) * 1.1))
    rmse_ax.set_xlabel("Forecast time (h)")
    rmse_ax.set_ylabel("RMSE vs real forecast (kn)")
    rmse_ax.set_title("Interpolation difference")
    rmse_ax.grid(alpha=0.3)
    rmse_ax.legend()

    def update(frame_index: int):
        frame_h = frame_times[frame_index]
        for image, target_h in zip(
            images,
            (3, 24, 12, 6, 3),
            strict=True,
        ):
            image.set_array(all_magnitude[target_h][frame_index].ravel())
        fig.suptitle(f"ERA5 wind comparison | departure + {frame_h:.0f} h")
        for line, target_h in zip(rmse_lines, rmse, strict=True):
            line.set_data(
                frame_times[: frame_index + 1],
                rmse[target_h][: frame_index + 1],
            )
        return [*images, *rmse_lines]

    writer = PillowWriter(fps=args.fps)
    from matplotlib.animation import FuncAnimation

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frame_times),
        interval=1000 / args.fps,
        blit=False,
    )
    animation.save(args.output, writer=writer)
    plt.close(fig)
    print(f"Saved animation: {args.output}")
    print(f"Saved RMSE table: {args.csv_output}")


if __name__ == "__main__":
    main()
