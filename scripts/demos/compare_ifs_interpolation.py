#!/usr/bin/env python
"""Compare native and temporally interpolated IFS ensemble forecasts.

Example
-------
Wind comparison::

uv run scripts/demos/compare_ifs_interpolation.py \
    --wind-path data/pf/atlantic/enfo/20250119_00z.nc

Significant wave height comparison::

uv run scripts/demos/compare_ifs_interpolation.py \
    --wave-path data/pf/atlantic/waef/20250119_00z.nc
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.artist import Artist

from routetools.ifs.interpolation import (
    load_ensemble_waves,
    load_ensemble_wind,
    relative_rmse_percent,
    resample_field_at_resolutions,
    resample_wind_at_resolutions,
)


def _parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare IFS ensemble wind or significant wave height at "
            "24-, 12-, 6-, and 3-hour temporal resolutions."
        )
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--wind-path", type=Path)
    input_group.add_argument("--wave-path", type=Path)
    parser.add_argument("--ensemble-member", type=int)
    parser.add_argument("--lon-min", type=float, default=-80.0)
    parser.add_argument("--lon-max", type=float, default=10.0)
    parser.add_argument("--lat-min", type=float, default=25.0)
    parser.add_argument("--lat-max", type=float, default=60.0)
    parser.add_argument("--max-hours", type=float, default=144.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
    )
    parser.add_argument("--fps", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Generate the IFS interpolation animation and RMSE table."""
    args = _parse_args().parse_args(argv)
    is_wave = args.wave_path is not None
    input_path = args.wave_path if is_wave else args.wind_path
    if input_path is None:
        raise ValueError("Provide either --wind-path or --wave-path")
    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing IFS ensemble forecast file: {input_path}"
        )

    load_options = {
        "ensemble_member": args.ensemble_member,
        "lon_bounds": (args.lon_min, args.lon_max),
        "lat_bounds": (args.lat_min, args.lat_max),
        "max_hours": args.max_hours,
    }
    if is_wave:
        hours, lats, lons, native_field = load_ensemble_waves(
            input_path,
            **load_options,
        )
        field_by_resolution = resample_field_at_resolutions(hours, native_field)
        field_label = "Significant wave height"
        unit = "m"
        variable_label = "wave height"
        rmse_column = "rmse_meters"
        default_stem = "ifs_wave_interpolation"
    else:
        hours, lats, lons, u, v = load_ensemble_wind(
            input_path,
            **load_options,
        )
        wind_fields = resample_wind_at_resolutions(hours, u, v)
        field_by_resolution = {
            resolution_h: np.hypot(u_field, v_field) * 1.94384
            for resolution_h, (u_field, v_field) in wind_fields.items()
        }
        field_label = "Wind speed"
        unit = "kn"
        variable_label = "wind"
        rmse_column = "rmse_knots"
        default_stem = "ifs_interpolation"

    reference = field_by_resolution[3.0]
    comparison_resolutions = (24.0, 12.0, 6.0)
    common_valid = np.isfinite(reference)
    for resolution_h in comparison_resolutions:
        common_valid &= np.isfinite(field_by_resolution[resolution_h])
    valid_grid_cells = np.count_nonzero(common_valid, axis=(1, 2))
    if np.any(valid_grid_cells == 0):
        raise ValueError(
            "No grid cells have valid data across all temporal resolutions"
        )

    reference_mean = np.sum(
        np.where(common_valid, reference, 0.0),
        axis=(1, 2),
    ) / valid_grid_cells
    rmse = {}
    relative_rmse = {}
    for resolution_h in comparison_resolutions:
        difference = np.where(
            common_valid,
            field_by_resolution[resolution_h] - reference,
            0.0,
        )
        rmse[resolution_h] = np.sqrt(
            np.sum(difference**2, axis=(1, 2)) / valid_grid_cells
        )
        relative_rmse[resolution_h] = relative_rmse_percent(
            rmse[resolution_h],
            reference_mean,
        )

    output_path = args.output or Path(f"output/{default_stem}.gif")
    csv_output_path = args.csv_output or Path(f"output/{default_stem}_rmse.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_output_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "lead_time_h",
                "resolution_h",
                rmse_column,
                "relative_rmse_percent",
                "valid_grid_cells",
            ]
        )
        for resolution_h in rmse:
            for lead_time_h, value, relative_value, valid_grid_cell_count in zip(
                hours,
                rmse[resolution_h],
                relative_rmse[resolution_h],
                valid_grid_cells,
                strict=True,
            ):
                writer.writerow(
                    [
                        lead_time_h,
                        int(resolution_h),
                        value,
                        relative_value,
                        valid_grid_cell_count,
                    ]
                )

    vmax = float(np.nanpercentile(reference, 99))
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=(1, 1.15))
    map_axes = [
        fig.add_subplot(grid[0, col], aspect="equal") for col in range(3)
    ]
    map_axes.append(fig.add_subplot(grid[1, 0], aspect="equal"))
    chart_grid = grid[1, 1:].subgridspec(
        2,
        1,
        height_ratios=(1, 0.85),
        hspace=0.4,
    )
    rmse_ax = fig.add_subplot(chart_grid[0])
    relative_rmse_ax = fig.add_subplot(chart_grid[1], sharex=rmse_ax)
    titles = [
        "Real forecast (3-hour reference)",
        "24-hour interpolation",
        "12-hour interpolation",
        "6-hour interpolation",
    ]
    panel_resolutions = (3.0, 24.0, 12.0, 6.0)

    images = []
    for ax, resolution_h, title in zip(
        map_axes,
        panel_resolutions,
        titles,
        strict=True,
    ):
        image = ax.pcolormesh(
            lon_grid,
            lat_grid,
            field_by_resolution[resolution_h][0],
            shading="auto",
            cmap="viridis",
            vmin=0,
            vmax=vmax,
        )
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        images.append(image)
    fig.colorbar(
        images[0],
        ax=map_axes[:3],
        label=f"{field_label} ({unit})",
        shrink=0.9,
    )

    resolution_colors = {
        24.0: "tab:blue",
        12.0: "tab:orange",
        6.0: "tab:green",
    }
    rmse_lines = {}
    relative_rmse_lines = {}
    for resolution_h in rmse:
        line, = rmse_ax.plot(
            [],
            [],
            marker="o",
            color=resolution_colors[resolution_h],
            label=f"{int(resolution_h)} h",
        )
        relative_line, = relative_rmse_ax.plot(
            [],
            [],
            marker="o",
            linestyle="--",
            color=resolution_colors[resolution_h],
            label=f"{int(resolution_h)} h",
        )
        rmse_lines[resolution_h] = line
        relative_rmse_lines[resolution_h] = relative_line
    rmse_ax.set_xlim(hours[0], hours[-1])
    rmse_max = max(float(np.max(values)) for values in rmse.values())
    rmse_ax.set_ylim(0, max(1.0, rmse_max * 1.1))
    rmse_ax.tick_params(labelbottom=False)
    rmse_ax.set_ylabel(f"RMSE vs 3-hour forecast ({unit})")
    rmse_ax.set_title("Absolute interpolation error")
    rmse_ax.grid(alpha=0.3)
    rmse_ax.legend()
    relative_max = max(float(np.max(values)) for values in relative_rmse.values())
    relative_rmse_ax.set_ylim(0, max(1.0, relative_max * 1.1))
    relative_rmse_ax.set_xlabel("Forecast lead time (h)")
    relative_rmse_ax.set_ylabel("Relative RMSE (%)")
    relative_rmse_ax.set_title(f"Error relative to mean reference {variable_label}")
    relative_rmse_ax.grid(alpha=0.3)

    forecast_kind = (
        f"member {args.ensemble_member}"
        if args.ensemble_member is not None
        else "ensemble mean"
    )

    def update(frame_index: int) -> list[Artist]:
        lead_time_h = hours[frame_index]
        for image, resolution_h in zip(
            images,
            panel_resolutions,
            strict=True,
        ):
            image.set_array(field_by_resolution[resolution_h][frame_index].ravel())
        fig.suptitle(
            f"IFS {forecast_kind} {variable_label} | "
            f"{lead_time_h:.0f} h lead time"
        )
        for resolution_h, line in rmse_lines.items():
            line.set_data(
                hours[: frame_index + 1],
                rmse[resolution_h][: frame_index + 1],
            )
        for resolution_h, line in relative_rmse_lines.items():
            line.set_data(
                hours[: frame_index + 1],
                relative_rmse[resolution_h][: frame_index + 1],
            )
        return [
            *images,
            *rmse_lines.values(),
            *relative_rmse_lines.values(),
        ]

    animation = FuncAnimation(
        fig,
        update,
        frames=len(hours),
        interval=1000 / args.fps,
        blit=False,
    )
    animation.save(output_path, writer=PillowWriter(fps=args.fps))
    plt.close(fig)
    print(f"Saved animation: {output_path}")
    print(f"Saved RMSE table: {csv_output_path}")


if __name__ == "__main__":
    main()
