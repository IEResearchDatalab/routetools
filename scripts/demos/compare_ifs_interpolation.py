#!/usr/bin/env python
"""Compare native and temporally interpolated IFS ensemble wind forecasts.

Example
-------
uv run scripts/demos/compare_ifs_interpolation.py \
    --wind-path data/pf/atlantic/enfo/20250119_00z.nc
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from routetools.ifs.interpolation import (
    SUPPORTED_INTERPOLATION_RESOLUTIONS_H,
    load_ensemble_wind,
    resample_wind_at_resolutions,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare 24-, 12-, 6-, and 3-hour IFS ensemble wind forecasts."
    )
    parser.add_argument("--wind-path", type=Path, required=True)
    parser.add_argument("--ensemble-member", type=int)
    parser.add_argument("--lon-min", type=float, default=-80.0)
    parser.add_argument("--lon-max", type=float, default=10.0)
    parser.add_argument("--lat-min", type=float, default=25.0)
    parser.add_argument("--lat-max", type=float, default=60.0)
    parser.add_argument("--max-hours", type=float, default=144.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/ifs_interpolation.gif"),
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=Path("output/ifs_interpolation_rmse.csv"),
    )
    parser.add_argument("--fps", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Generate the IFS interpolation animation and RMSE table."""
    args = _parse_args().parse_args(argv)
    if not args.wind_path.exists():
        raise FileNotFoundError(
            f"Missing IFS ensemble wind file: {args.wind_path}"
        )

    hours, lats, lons, u, v = load_ensemble_wind(
        args.wind_path,
        ensemble_member=args.ensemble_member,
        lon_bounds=(args.lon_min, args.lon_max),
        lat_bounds=(args.lat_min, args.lat_max),
        max_hours=args.max_hours,
    )
    fields = resample_wind_at_resolutions(hours, u, v)

    magnitudes = {}
    for resolution_h, (u_field, v_field) in fields.items():
        magnitudes[resolution_h] = np.sqrt(u_field**2 + v_field**2) * 1.94384

    reference = magnitudes[3.0]
    rmse = {}
    for resolution_h in (24.0, 12.0, 6.0):
        difference = magnitudes[resolution_h] - reference
        rmse[resolution_h] = np.sqrt(
            np.mean(difference**2, axis=(1, 2))
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_output.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["lead_time_h", "resolution_h", "rmse_knots"])
        for resolution_h in rmse:
            for lead_time_h, value in zip(hours, rmse[resolution_h], strict=True):
                writer.writerow([lead_time_h, int(resolution_h), value])

    vmax = max(
        float(np.nanpercentile(reference, 99)),
        *(
            float(np.nanpercentile(magnitudes[resolution_h], 99))
            for resolution_h in SUPPORTED_INTERPOLATION_RESOLUTIONS_H
        ),
    )
    lon_grid, lat_grid = np.meshgrid(lons, lats)
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
        "24-hour sampling",
        "12-hour sampling",
        "6-hour sampling",
        "3-hour sampling",
        "RMSE",
    ]
    panel_resolutions = (3.0, 24.0, 12.0, 6.0, 3.0)

    images = []
    for ax, resolution_h, title in zip(
        axes[:5],
        panel_resolutions,
        titles[:5],
        strict=True,
    ):
        image = ax.pcolormesh(
            lon_grid,
            lat_grid,
            magnitudes[resolution_h][0],
            shading="auto",
            cmap="viridis",
            vmin=0,
            vmax=vmax,
        )
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        images.append(image)
    fig.colorbar(images[0], ax=axes[:5], label="Wind speed (kn)", shrink=0.86)

    rmse_ax = axes[5]
    rmse_lines = []
    for resolution_h in rmse:
        line, = rmse_ax.plot([], [], marker="o", label=f"{int(resolution_h)} h")
        rmse_lines.append(line)
    rmse_ax.set_xlim(hours[0], hours[-1])
    rmse_max = max(float(np.max(values)) for values in rmse.values())
    rmse_ax.set_ylim(0, max(1.0, rmse_max * 1.1))
    rmse_ax.set_xlabel("Forecast lead time (h)")
    rmse_ax.set_ylabel("RMSE vs 3-hour forecast (kn)")
    rmse_ax.set_title("Interpolation difference")
    rmse_ax.grid(alpha=0.3)
    rmse_ax.legend()

    forecast_kind = (
        f"member {args.ensemble_member}"
        if args.ensemble_member is not None
        else "ensemble mean"
    )

    def update(frame_index: int):
        lead_time_h = hours[frame_index]
        for image, resolution_h in zip(
            images,
            panel_resolutions,
            strict=True,
        ):
            image.set_array(magnitudes[resolution_h][frame_index].ravel())
        fig.suptitle(
            f"IFS {forecast_kind} wind | {lead_time_h:.0f} h lead time"
        )
        for line, resolution_h in zip(rmse_lines, rmse, strict=True):
            line.set_data(
                hours[: frame_index + 1],
                rmse[resolution_h][: frame_index + 1],
            )
        return [*images, *rmse_lines]

    animation = FuncAnimation(
        fig,
        update,
        frames=len(hours),
        interval=1000 / args.fps,
        blit=False,
    )
    animation.save(args.output, writer=PillowWriter(fps=args.fps))
    plt.close(fig)
    print(f"Saved animation: {args.output}")
    print(f"Saved RMSE table: {args.csv_output}")


if __name__ == "__main__":
    main()
