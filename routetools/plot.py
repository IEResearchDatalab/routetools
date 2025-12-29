import json
from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from routetools.cost import (
    cost_function_constant_speed_time_variant,
    haversine_distance_from_curve,
)
from routetools.land import Land

DICT_COLOR = {
    "BERS": "blue",
    "CMA-ES": "orange",
    "FMS": "green",
}

DICT_VF_NAMES = {
    "circular": "Circular",
    "fourvortices": "Four Vortices",
    "doublegyre": "Double Gyre",
    "techy": "Techy",
    "swirlys": "Swirlys",
}


def plot_curve(
    vectorfield: Callable[
        [jnp.ndarray, jnp.ndarray, float], tuple[jnp.ndarray, jnp.ndarray]
    ],
    ls_curve: list[jnp.ndarray],
    ls_name: list[str] | None = None,
    ls_cost: list[float] | None = None,
    land: Land | None = None,
    xlim: tuple[float, float] = (jnp.inf, -jnp.inf),
    ylim: tuple[float, float] = (jnp.inf, -jnp.inf),
    gridstep: float = 0.25,
    figsize: tuple[float, float] = (4, 4),
    cost: str = "cost",
    legend_outside: bool = False,
    color_currents: bool = False,
) -> tuple[Figure, Axes]:
    """Plot the vectorfield and the curves.

    Parameters
    ----------
    vectorfield : Callable
        Vectorfield function
    ls_curve : list[jnp.ndarray]
        List of curves to plot
    ls_name : list[str] | None, optional
        List of names for each curve, by default None
    ls_cost : list[float] | None, optional
        List of costs for each curve, by default None
    land_array : jnp.ndarray | None, optional
        Array of land, by default None
    xlnd : jnp.ndarray | None, optional
        x values of the land array, by default None
    ylnd : jnp.ndarray | None, optional
        y values of the land array, by default None
    xlim : tuple | None, optional
        x limits, by default None
    ylim : tuple | None, optional
        y limits, by default None
    gridstep : float, optional
        Grid step for the vectorfield, by default 0.25
    figsize : tuple, optional
        Figure size, by default (4, 4)
    cost : str, optional
        Cost function, by default "cost"
    legend_outside : bool, optional
        Place the legend outside the plot, by default False

    Returns
    -------
    tuple[Figure, Axes]
        Figure and Axes objects
    """
    # Set default parameters
    if ls_name is None:
        ls_name = []
    if ls_cost is None:
        ls_cost = []

    fig = plt.figure(figsize=figsize)
    ax = plt.gca()

    # Plot the land
    if land is not None:
        # Land is a boolean array, so we need to use contourf
        ax.contourf(
            land.x,
            land.y,
            land.array.T,
            levels=[0, 0.5, 1],
            colors=["white", "black", "black"],
            origin="lower",
            zorder=0,
        )

    if color_currents:
        # Plot the vectorfield as a background color map
        xvf = jnp.arange(xlim[0] - gridstep * 2, xlim[1] + gridstep * 2, gridstep)
        yvf = jnp.arange(ylim[0] - gridstep * 2, ylim[1] + gridstep * 2, gridstep)
        t = 0
        X, Y = jnp.meshgrid(xvf, yvf)
        U, V = vectorfield(X, Y, t)
        # Compute the magnitude
        mag = jnp.sqrt(U**2 + V**2)
        # When currents are zero, set them to NaN to avoid plotting
        mag = jnp.where(mag == 0, jnp.nan, mag)
        # Plot the magnitude as a colormap (ensure array orientation matches X,Y)
        pcm = ax.pcolormesh(
            X,
            Y,
            mag,
            shading="auto",
            cmap="Reds",
            alpha=1.0,
            zorder=0,
        )
        # Check if vectorfield is more horizontal or vertical for colorbar orientation
        if (xlim[1] - xlim[0]) >= (ylim[1] - ylim[0]):
            orientation = "horizontal"
        else:
            orientation = "vertical"
        # Plot colorbar horizontally below the plot
        cbar = fig.colorbar(pcm, ax=ax, orientation=orientation)
        # Set it to knots (1 m/s = 1.94384 knots)
        cbar.set_label("Current magnitude (m/s)")
    else:
        # Plot the vectorfield
        xvf = jnp.arange(xlim[0] - gridstep * 2, xlim[1] + gridstep * 2, gridstep)
        yvf = jnp.arange(ylim[0] - gridstep * 2, ylim[1] + gridstep * 2, gridstep)
        t = 0
        X, Y = jnp.meshgrid(xvf, yvf)
        U, V = vectorfield(X, Y, t)
        # Scale U and V for better visualization
        mag = jnp.sqrt(U**2 + V**2)
        U, V = U / mag, V / mag
        # Skip if all is 0
        if not jnp.all(U == 0) or not jnp.all(V == 0):
            ax.quiver(X, Y, U, V, zorder=1)

    # Plot the curves
    for idx, curve in enumerate(ls_curve):
        label = ""
        if len(ls_name) == len(ls_curve):
            label = ls_name[idx]
        if len(ls_cost) == len(ls_curve):
            cost_val = ls_cost[idx]
            label += f" ({cost} = {cost_val:.3f})"
        # Assign a color based on the label
        color: str | None = None
        for key, col in DICT_COLOR.items():
            if label.startswith(key):
                color = col
                break
        # Plot the curve
        ax.plot(
            curve[:, 0],
            curve[:, 1],
            marker="o",
            markersize=1,
            label=label,
            zorder=2,
            color=color,
        )
        # Update limits according to the curve
        # Ensure xlim/ylim keep float types (convert jnp types to float)
        x0 = float(min(xlim[0], float(min(curve[:, 0]))))
        x1 = float(max(xlim[1], float(max(curve[:, 0]))))
        y0 = float(min(ylim[0], float(min(curve[:, 1]))))
        y1 = float(max(ylim[1], float(max(curve[:, 1]))))
        xlim = (x0, x1)
        ylim = (y0, y1)

    # Plot the start and end points
    src = curve[0]
    dst = curve[-1]
    ax.plot(src[0], src[1], "o", color="blue", zorder=3)
    ax.plot(dst[0], dst[1], "o", color="green", zorder=3)

    if legend_outside:
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    else:
        ax.legend()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    # Make sure the aspect ratio is correct
    ax.set_aspect("equal", adjustable="box")

    # Adjust the layout
    fig.tight_layout()

    return fig, ax


def plot_route_from_json(path_json: str) -> tuple[Figure, Axes]:
    """Plot the route from a json file.

    Parameters
    ----------
    path_json : str
        Path to the json file with the route

    Returns
    -------
    tuple[Figure, Axes]
        Figure and Axes objects
    """
    with open(path_json) as file:
        data: dict[str, Any] = json.load(file)

    # Get the data
    ls_curve = [jnp.array(data["curve_cmaes"]), jnp.array(data["curve_b"])]
    ls_name = ["CMA-ES", "BERS"]
    ls_cost = [data["cost_cmaes"], data["cost_fms"]]

    # Load the vectorfield function
    vfname = data["vectorfield"]
    vectorfield_module = __import__(
        "routetools.vectorfield", fromlist=["vectorfield_" + vfname]
    )
    vectorfield = getattr(vectorfield_module, "vectorfield_" + vfname)

    # Load the land parameters
    water_level = data["water_level"]
    resolution = data.get("resolution", 0)
    random_seed = data.get("random_seed", 0)
    k = data.get("K")
    sigma0 = data.get("sigma0")

    # Generate the land
    if resolution != 0:
        land = Land(
            xlim=data["xlim"],
            ylim=data["ylim"],
            water_level=water_level,
            resolution=resolution,
            interpolate=data.get("interpolate", 100),
            outbounds_is_land=data["outbounds_is_land"],
            random_seed=random_seed,
        )
    else:
        land = None

    # Identify the cost function
    if "travel_stw" in data:
        cost = "dist" if data["vectorfield"] == "zero" else "time"
    else:
        cost = "fuel"

    fig, ax = plot_curve(
        vectorfield,
        ls_curve,
        ls_name=ls_name,
        ls_cost=ls_cost,
        land=land,
        xlim=data["xlim"],
        ylim=data["ylim"],
        cost=cost,
    )
    # Set the title and tight layout
    if water_level == 1:
        vf = DICT_VF_NAMES.get(vfname, vfname)
        title = f"{vf} | K = {int(k)} | " + r"$\sigma_0$ = " + f"{sigma0:.1f}"
    else:
        title = f"Water level: {water_level:.1f} | Resolution: {int(resolution)}"
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def plot_table_aggregated(
    df: pd.DataFrame,
    value_column: str,
    index_columns: list[str],
    column_columns: list[str],
    agg: str = "mean",
    vmin: float | None = None,
    vmax: float | None = None,
    round_decimals: int = 2,
    cmap: str = "coolwarm",
    colorbar_label: str = "",
    title: str = "",
    figsize: tuple[float, float] = (12, 12),
) -> tuple[Figure, Axes]:
    """
    Plot a heatmap for a given metric with mean ± standard deviation.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the data to be visualized.
    mask : np.ndarray
        A boolean mask to filter the DataFrame.
    value_column : str
        The name of the column containing the values to be aggregated.
    index_columns : list
        List of column names to use as row indices (e.g., ["sigma0", "popsize"]).
    column_columns : list
        List of column names to use as column indices (e.g., ["K", "L"]).
    agg : str, optional
        Aggregation function to use, default is "mean".
    vmin : float, optional
        Minimum value for the heatmap color scale, default is None.
    vmax : float, optional
        Maximum value for the heatmap color scale, default is None.
    round_decimals : int, optional
        Number of decimals to round the values, default is 2.
    cmap : str, optional
        Colormap for the heatmap, default is "coolwarm".
    colorbar_label : str, optional
        Label for the colorbar, default is an empty string.
    title : str, optional
        Title of the heatmap, default is an empty string.
    figsize : tuple, optional
        Size of the figure, default is (12, 12).

    Returns
    -------
    tuple[Figure, Axes]
        Figure and Axes objects for the heatmap.
    """

    def _create_pivot_table(aggfunc: str) -> pd.DataFrame:
        """Auxiliary function to create a pivot table with aggregated values."""
        return (
            df.pivot_table(
                values=value_column,
                index=index_columns,
                columns=column_columns,
                aggfunc=aggfunc,
            )
            .round(round_decimals)
            .astype(float if round_decimals > 0 else int)
        )

    if agg == "mean":
        # Create pivot tables for mean and standard deviation
        pivot_table_values = _create_pivot_table("mean")
        pivot_table_std = _create_pivot_table("std")

        # Combine mean and std into a single pivot table for annotation
        pivot_table_annot = pivot_table_values.copy()
        for col in pivot_table_annot.columns:
            pivot_table_annot[col] = (
                pivot_table_values[col].astype(str)
                + " ± "
                + pivot_table_std[col].astype(str)
            )

    elif agg == "sum":
        # Create pivot table for sum
        pivot_table_values = _create_pivot_table("sum")
        pivot_table_annot = pivot_table_values.copy()
    else:
        raise ValueError(f"Invalid aggregation function: {agg}")

    # Plot heatmap
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        pivot_table_values,
        annot=pivot_table_annot,
        vmin=vmin,
        vmax=vmax,
        fmt="",
        cmap=cmap,
        cbar_kws={"label": colorbar_label},
        annot_kws={"ha": "center", "va": "center"},
        ax=ax,
        cbar=False,
    )

    # Set labels and title
    ax.set_xlabel(" - ".join(column_columns))
    ax.set_ylabel(" - ".join(index_columns))
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def plot_distance_to_end_vs_time(
    vectorfield: Callable[
        [jnp.ndarray, jnp.ndarray, float], tuple[jnp.ndarray, jnp.ndarray]
    ],
    ls_curve: list[jnp.ndarray],
    ls_name: list[str] | None = None,
    name: str = "",
    vel_ship: float = 10.0,
) -> tuple[Figure, Axes]:
    """Plot distance to the end point vs time for two curves.

    Parameters
    ----------
    curve_a : jnp.ndarray
        First curve (lon, lat)
    curve_b : jnp.ndarray
        Second curve (lon, lat)
    vectorfield : Callable
        Vectorfield function
    name : str, optional
        Instance name for the title, by default ""
    vel_ship : float, optional
        Ship speed in meters per second, by default 10.0

    Returns
    -------
    tuple[Figure, Axes]
        Figure and Axes objects
    """
    ls_dist: list[jnp.ndarray] = []
    ls_cost: list[jnp.ndarray] = []

    for curve in ls_curve:
        # Compute distance traversed between points (L-1)
        d_curve = haversine_distance_from_curve(curve) / 1000  # in km
        # Compute the cumulative sum, backwards from the end point
        d_curve = jnp.cumsum(d_curve[::-1])[::-1]
        # Include a 0 at the end (L)
        d_curve = jnp.concatenate([d_curve, jnp.array([0.0])])
        ls_dist.append(d_curve)

        # Compute time vector (L-1)
        t_curve = cost_function_constant_speed_time_variant(
            vectorfield=vectorfield,
            curve=curve[jnp.newaxis, :, :],
            travel_stw=vel_ship,
            spherical_correction=True,
        )
        t_curve = t_curve[0] / 3600  # in hours
        # Append a first time as 0 (L)
        t_curve = jnp.concatenate([jnp.array([0.0]), t_curve])
        # Compute cumulative time to have a proper x-axis
        t_curve = jnp.cumsum(t_curve)
        ls_cost.append(t_curve)

    # Plot distance to end vs time
    fig = plt.figure(figsize=(6, 4))
    # Create two axes, one on top of the other
    ax1, ax2 = fig.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for idx, d_curve in enumerate(ls_dist):
        t_curve = ls_cost[idx]
        label = ""
        if ls_name is not None and len(ls_name) == len(ls_curve):
            label = ls_name[idx]
            dist = d_curve[0]
            cost = t_curve[-1]
            label += f" ({int(dist)} km, {cost:.1f} h)"
        ax1.plot(
            t_curve,
            d_curve,
            marker="o",
            markersize=2,
            label=label,
        )
    ax1.set_xlabel("Time (hours)")
    ax1.set_ylabel("Distance to traverse (km)")
    ax1.set_title(f"{name} | {int(vel_ship * 2)} knots")
    ax1.legend()
    ax1.grid()
    # In the second axis, plot the difference between each pair of curves
    if len(ls_dist) == 2:
        d_curve_a = ls_dist[0]
        d_curve_b = ls_dist[1]
        # Interpolate the shorter curve to the longer one
        if ls_cost[0][-1] < ls_cost[1][-1]:
            d_curve_a = jnp.interp(ls_cost[1], ls_cost[0], ls_dist[0])  # type: ignore
            d_curve_a = jnp.maximum(d_curve_a, 0)
            t_curve = ls_cost[1]
        else:
            d_curve_b = jnp.interp(ls_cost[0], ls_cost[1], ls_dist[1])  # type: ignore
            d_curve_b = jnp.maximum(d_curve_b, 0)
            t_curve = ls_cost[0]
        d_diff = d_curve_a - d_curve_b
        ax2.plot(
            t_curve,
            d_diff,
            marker="o",
            markersize=2,
            color="red",
            label="Distance difference",
        )
        ax2.set_xlabel("Time (hours)")
        ax2.set_ylabel("Difference (km)")
        ax2.legend()
        ax2.grid()
    else:
        # Remove the second axis if there are not exactly two curves
        fig.delaxes(ax2)
    plt.tight_layout()
    return fig, (ax1, ax2)
