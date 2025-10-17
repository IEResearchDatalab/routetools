import json
from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

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
    figsize: tuple[float, float] = (4, 4),
    cost: str = "cost",
    legend_outside: bool = False,
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

    # Plot the vectorfield
    xvf = jnp.arange(xlim[0] - 0.5, xlim[1] + 0.5, 0.25)
    yvf = jnp.arange(ylim[0] - 0.5, ylim[1] + 0.5, 0.25)
    t = 0
    X, Y = jnp.meshgrid(xvf, yvf)
    U, V = vectorfield(X, Y, t)
    # Skip if all is 0
    if not jnp.all(U == 0) or not jnp.all(V == 0):
        ax.quiver(X, Y, U, V, zorder=1)

    if legend_outside:
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    else:
        ax.legend()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    # Make sure the aspect ratio is correct
    ax.set_aspect("equal", adjustable="box")

    # Adjust the layout
    fig.tight_layout(pad=2.5)

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
    ls_curve = [jnp.array(data["curve_cmaes"]), jnp.array(data["curve_fms"])]
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
