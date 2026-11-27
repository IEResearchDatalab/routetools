#!/usr/bin/env python
"""Compare native and temporally interpolated IFS ensemble forecasts.

Wind maps show downsampled U/V vectors and component-wise errors. Wave maps
show significant wave height and scalar-field errors.

Example
-------
Wind comparison::

uv run scripts/demos/compare_ifs_interpolation.py --field wind \
    --date 20250101 --cycle 00

Significant wave height comparison::

uv run scripts/demos/compare_ifs_interpolation.py --field waves \
    --date 20250101 --cycle 00

The shared GRIB root defaults to ``/home/SHARED/weather-routing-data/ensambles``.
Use ``--lon-min``, ``--lon-max``, ``--lat-min``, and ``--lat-max`` to crop the
global forecast before loading its fields. Add ``uv sync --extra ifs`` once to
install the GRIB decoder.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.artist import Artist

from routetools.ifs.interpolation import (
    relative_rmse_percent,
    resample_field_at_resolutions,
    resample_wind_at_resolutions,
)
from routetools.ifs.loader import load_ensemble_waves, load_ensemble_wind


def _parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare IFS ensemble wind or significant wave height at "
            "24-, 12-, 6-, and 3-hour temporal resolutions using both "
            "routing linear and cubic interpolation."
        )
    )
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--wind-path", type=Path)
    input_group.add_argument("--wave-path", type=Path)
    parser.add_argument("--v-wind-path", type=Path)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/SHARED/weather-routing-data/ensambles"),
        help="Root of the shared GRIB archive.",
    )
    parser.add_argument(
        "--field",
        choices=("wind", "waves"),
        help="Field to load from --data-root; defaults to wind.",
    )
    parser.add_argument("--date", default="20250101", help="Run date (YYYYMMDD).")
    parser.add_argument("--cycle", choices=("00", "12"), default="00")
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
    parser = _parse_args()
    args = parser.parse_args(argv)
    try:
        run_date = datetime.strptime(args.date, "%Y%m%d").date()
    except ValueError:
        parser.error("--date must use YYYYMMDD format")

    if args.wind_path is not None or args.wave_path is not None:
        is_wave = args.wave_path is not None
        if args.field is not None and (args.field == "waves") != is_wave:
            parser.error("--field must match the selected --wind-path/--wave-path")
        if args.v_wind_path is not None and args.wind_path is None:
            parser.error("--v-wind-path requires --wind-path")
        input_path = args.wave_path if is_wave else args.wind_path
        v_wind_path = args.v_wind_path
    else:
        if args.v_wind_path is not None:
            parser.error("--v-wind-path requires --wind-path")
        is_wave = args.field == "waves"
        filename = f"{run_date:%Y%m%d}_{args.cycle}z.grib2"
        if is_wave:
            input_path = args.data_root / "waef" / "swh" / filename
            v_wind_path = None
        else:
            input_path = args.data_root / "enfo" / "10u" / filename
            v_wind_path = args.data_root / "enfo" / "10v" / filename

    if input_path is None:
        parser.error("Provide --wind-path, --wave-path, or select a shared --field")
    paths_to_check = [input_path]
    if v_wind_path is not None:
        paths_to_check.append(v_wind_path)
    missing_paths = [path for path in paths_to_check if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "Missing IFS forecast file(s): "
            + ", ".join(str(path) for path in missing_paths)
        )

    load_options = {
        "ensemble_member": args.ensemble_member,
        "lon_bounds": (args.lon_min, args.lon_max),
        "lat_bounds": (args.lat_min, args.lat_max),
        "max_hours": args.max_hours,
    }
    interpolation_methods = ("linear", "cubic")
    if is_wave:
        hours, lats, lons, native_field = load_ensemble_waves(
            input_path,
            **load_options,
        )
        components_by_method = {
            method: {
                "Hs": resample_field_at_resolutions(
                    hours,
                    native_field,
                    method=method,
                )
            }
            for method in interpolation_methods
        }
        component_names: tuple[str, ...] = ("Hs",)
        unit = "m"
        variable_label = "wave height"
        relative_error_title = "RMSE relative to mean reference wave height"
        default_stem = "ifs_wave_interpolation"
    else:
        hours, lats, lons, u, v = load_ensemble_wind(
            input_path,
            v_path=v_wind_path,
            **load_options,
        )
        wind_fields_by_method = {
            method: resample_wind_at_resolutions(
                hours,
                u,
                v,
                method=method,
            )
            for method in interpolation_methods
        }
        components_by_method = {
            method: {
                "U": {
                    resolution_h: fields[0]
                    for resolution_h, fields in wind_fields.items()
                },
                "V": {
                    resolution_h: fields[1]
                    for resolution_h, fields in wind_fields.items()
                },
            }
            for method, wind_fields in wind_fields_by_method.items()
        }
        component_names = ("U", "V")
        unit = "m/s"
        variable_label = "wind components"
        relative_error_title = "Vector RMSE relative to reference wind RMS"
        default_stem = "ifs_interpolation"

    comparison_resolutions = (24.0, 12.0, 6.0)
    common_valid = np.ones_like(
        components_by_method["linear"][component_names[0]][3.0],
        dtype=bool,
    )
    for method in interpolation_methods:
        for component_name in component_names:
            component_fields = components_by_method[method][component_name]
            common_valid &= np.isfinite(component_fields[3.0])
            for resolution_h in comparison_resolutions:
                common_valid &= np.isfinite(component_fields[resolution_h])
    valid_grid_cells = np.count_nonzero(common_valid, axis=(1, 2))
    if np.any(valid_grid_cells == 0):
        raise ValueError("No grid cells have valid data across methods and resolutions")

    if is_wave:
        reference_hs = components_by_method["linear"]["Hs"][3.0]
        reference_scale = (
            np.sum(
                np.where(common_valid, reference_hs, 0.0),
                axis=(1, 2),
            )
            / valid_grid_cells
        )
    else:
        reference_u = components_by_method["linear"]["U"][3.0]
        reference_v = components_by_method["linear"]["V"][3.0]
        reference_scale = np.sqrt(
            np.sum(
                np.where(
                    common_valid,
                    reference_u**2 + reference_v**2,
                    0.0,
                ),
                axis=(1, 2),
            )
            / valid_grid_cells
        )
        if np.any(reference_scale <= 0):
            raise ValueError(
                "Cannot normalize vector RMSE: reference wind RMS "
                "must be greater than zero"
            )

    rmse: dict[str, dict[float, dict[str, np.ndarray]]] = {}
    relative_rmse: dict[str, dict[float, dict[str, np.ndarray]]] = {}
    for method in interpolation_methods:
        rmse[method] = {}
        relative_rmse[method] = {}
        for resolution_h in comparison_resolutions:
            rmse[method][resolution_h] = {}
            relative_rmse[method][resolution_h] = {}
            for component_name in component_names:
                reference_component = components_by_method[method][component_name][3.0]
                comparison_component = components_by_method[method][component_name][
                    resolution_h
                ]
                difference = np.where(
                    common_valid,
                    comparison_component - reference_component,
                    0.0,
                )
                rmse_value = np.sqrt(
                    np.sum(difference**2, axis=(1, 2)) / valid_grid_cells
                )
                rmse[method][resolution_h][component_name] = rmse_value

            if is_wave:
                relative_rmse[method][resolution_h]["Hs"] = relative_rmse_percent(
                    rmse[method][resolution_h]["Hs"],
                    reference_scale,
                )
            else:
                vector_rmse = np.sqrt(
                    rmse[method][resolution_h]["U"] ** 2
                    + rmse[method][resolution_h]["V"] ** 2
                )
                relative_rmse[method][resolution_h]["vector"] = (
                    vector_rmse / reference_scale * 100.0
                )

    output_path = args.output or Path(f"output/{default_stem}.gif")
    csv_output_path = args.csv_output or Path(f"output/{default_stem}_rmse.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_output_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        if is_wave:
            writer.writerow(
                [
                    "lead_time_h",
                    "resolution_h",
                    *(
                        name
                        for method in interpolation_methods
                        for name in (
                            f"rmse_meters_{method}",
                            f"relative_rmse_percent_{method}",
                        )
                    ),
                    "valid_grid_cells",
                ]
            )
        else:
            writer.writerow(
                [
                    "lead_time_h",
                    "resolution_h",
                    *(
                        name
                        for method in interpolation_methods
                        for name in (
                            f"u_rmse_mps_{method}",
                            f"v_rmse_mps_{method}",
                        )
                    ),
                    *(
                        f"vector_relative_rmse_percent_{method}"
                        for method in interpolation_methods
                    ),
                    "valid_grid_cells",
                ]
            )
        for resolution_h in comparison_resolutions:
            for frame_index, lead_time_h in enumerate(hours):
                if is_wave:
                    values = [
                        lead_time_h,
                        int(resolution_h),
                        *(
                            metric[method][resolution_h]["Hs"][frame_index]
                            for method in interpolation_methods
                            for metric in (rmse, relative_rmse)
                        ),
                        valid_grid_cells[frame_index],
                    ]
                else:
                    values = [
                        lead_time_h,
                        int(resolution_h),
                        *(
                            rmse[method][resolution_h][component_name][frame_index]
                            for method in interpolation_methods
                            for component_name in component_names
                        ),
                        *(
                            relative_rmse[method][resolution_h]["vector"][frame_index]
                            for method in interpolation_methods
                        ),
                        valid_grid_cells[frame_index],
                    ]
                writer.writerow(values)

    lon_grid, lat_grid = np.meshgrid(lons, lats)
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=(1, 1.15))
    map_axes = [fig.add_subplot(grid[0, col], aspect="equal") for col in range(3)]
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
        "24-hour cubic B-spline",
        "12-hour cubic B-spline",
        "6-hour cubic B-spline",
    ]
    panel_resolutions = (3.0, 24.0, 12.0, 6.0)

    images = []
    wind_vectors = []
    if is_wave:
        reference = components_by_method["cubic"]["Hs"][3.0]
        vmax = float(np.nanpercentile(reference, 99))
        for ax, resolution_h, title in zip(
            map_axes,
            panel_resolutions,
            titles,
            strict=True,
        ):
            image = ax.pcolormesh(
                lon_grid,
                lat_grid,
                components_by_method["cubic"]["Hs"][resolution_h][0],
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
            label=f"Significant wave height ({unit})",
            shrink=0.9,
        )
    else:
        vector_stride = max(1, int(np.ceil(max(len(lats), len(lons)) / 28)))
        reference_u = components_by_method["cubic"]["U"][3.0]
        reference_v = components_by_method["cubic"]["V"][3.0]
        reference_speed = np.hypot(reference_u, reference_v)
        reference_speed_scale = float(np.nanpercentile(reference_speed, 95))
        if not np.isfinite(reference_speed_scale) or reference_speed_scale <= 0:
            raise ValueError(
                "Reference wind speed must have a positive 95th percentile"
            )
        quiver_scale = reference_speed_scale / 0.04
        for ax, resolution_h, title in zip(
            map_axes,
            panel_resolutions,
            titles,
            strict=True,
        ):
            vector = ax.quiver(
                lon_grid[::vector_stride, ::vector_stride],
                lat_grid[::vector_stride, ::vector_stride],
                components_by_method["cubic"]["U"][resolution_h][
                    0, ::vector_stride, ::vector_stride
                ],
                components_by_method["cubic"]["V"][resolution_h][
                    0, ::vector_stride, ::vector_stride
                ],
                angles="uv",
                scale_units="width",
                scale=quiver_scale,
                color="tab:blue",
                width=0.003,
            )
            ax.set_xlim(float(lons[0]), float(lons[-1]))
            ax.set_ylim(float(np.min(lats)), float(np.max(lats)))
            ax.set_title(title)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            wind_vectors.append(vector)
        map_axes[0].quiverkey(
            wind_vectors[0],
            0.72,
            0.08,
            10.0,
            "10 m/s",
            labelpos="E",
            coordinates="axes",
        )

    resolution_colors = {
        24.0: "tab:blue",
        12.0: "tab:orange",
        6.0: "tab:green",
    }
    method_styles = {"linear": "--", "cubic": "-"}
    component_markers = {"Hs": "o", "U": "o", "V": "^"}
    rmse_lines = {}
    relative_rmse_lines = {}
    for method in interpolation_methods:
        for resolution_h in comparison_resolutions:
            for component_name in component_names:
                label = (
                    f"{int(resolution_h)} h {method}"
                    if is_wave
                    else f"{int(resolution_h)} h {component_name} {method}"
                )
                (line,) = rmse_ax.plot(
                    [],
                    [],
                    marker=component_markers[component_name],
                    color=resolution_colors[resolution_h],
                    linestyle=method_styles[method],
                    label=label,
                )
                rmse_lines[(method, resolution_h, component_name)] = line
            relative_label = f"{int(resolution_h)} h {method}"
            (relative_line,) = relative_rmse_ax.plot(
                [],
                [],
                marker="o",
                color=resolution_colors[resolution_h],
                linestyle=method_styles[method],
                label=relative_label,
            )
            relative_rmse_lines[(method, resolution_h)] = relative_line
    rmse_ax.set_xlim(hours[0], hours[-1])
    rmse_max = max(
        float(np.max(values))
        for method_values in rmse.values()
        for resolution_values in method_values.values()
        for values in resolution_values.values()
    )
    rmse_ax.set_ylim(0, max(1.0, rmse_max * 1.1))
    rmse_ax.tick_params(labelbottom=False)
    rmse_ax.set_ylabel(f"Component RMSE vs 3-hour forecast ({unit})")
    rmse_ax.set_title("Absolute component interpolation error")
    rmse_ax.grid(alpha=0.3)
    rmse_ax.legend(ncol=2, fontsize="small")
    relative_max = max(
        float(np.max(values))
        for method_values in relative_rmse.values()
        for resolution_values in method_values.values()
        for values in resolution_values.values()
    )
    relative_rmse_ax.set_ylim(0, max(1.0, relative_max * 1.1))
    relative_rmse_ax.set_xlabel("Forecast lead time (h)")
    relative_rmse_ax.set_ylabel(
        "Relative RMSE (%)" if is_wave else "Relative vector RMSE (%)"
    )
    relative_rmse_ax.set_title(relative_error_title)
    relative_rmse_ax.grid(alpha=0.3)
    relative_rmse_ax.legend(ncol=2, fontsize="small")

    forecast_kind = (
        f"member {args.ensemble_member}"
        if args.ensemble_member is not None
        else "ensemble mean"
    )

    def update(frame_index: int) -> list[Artist]:
        lead_time_h = hours[frame_index]
        if is_wave:
            for image, resolution_h in zip(
                images,
                panel_resolutions,
                strict=True,
            ):
                image.set_array(
                    components_by_method["cubic"]["Hs"][resolution_h][
                        frame_index
                    ].ravel()
                )
        else:
            for vector, resolution_h in zip(
                wind_vectors,
                panel_resolutions,
                strict=True,
            ):
                vector.set_UVC(
                    components_by_method["cubic"]["U"][resolution_h][
                        frame_index, ::vector_stride, ::vector_stride
                    ],
                    components_by_method["cubic"]["V"][resolution_h][
                        frame_index, ::vector_stride, ::vector_stride
                    ],
                )
        fig.suptitle(
            f"IFS {forecast_kind} {variable_label} | {lead_time_h:.0f} h lead time"
        )
        for (method, resolution_h, component_name), line in rmse_lines.items():
            line.set_data(
                hours[: frame_index + 1],
                rmse[method][resolution_h][component_name][: frame_index + 1],
            )
        for (method, resolution_h), line in relative_rmse_lines.items():
            relative_name = "Hs" if is_wave else "vector"
            line.set_data(
                hours[: frame_index + 1],
                relative_rmse[method][resolution_h][relative_name][: frame_index + 1],
            )
        return [
            *images,
            *wind_vectors,
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
