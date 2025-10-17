import tomllib

import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer

from routetools.cmaes import optimize
from routetools.fms import optimize_fms
from routetools.plot import DICT_VF_NAMES, plot_curve, plot_route_from_json

DICT_COST_LITERATURE = {
    "circular": 1.98,
    "fourvortices": 8.95,
    "doublegyre": 1.01,
    "techy": 1.03,
    "swirlys": 5.73,
}


def plot_viable_area(vectorfield: str, config: str = "config.toml", t: float = 0):
    """Plot the viable area for the given vector field.

    Parameters
    ----------
    vectorfield : str
        The name of the vector field function to use.
    """
    # Load the vectorfield function
    vectorfield_module = __import__(
        "routetools.vectorfield", fromlist=["vectorfield_" + vectorfield]
    )
    vectorfield_fun = getattr(vectorfield_module, "vectorfield_" + vectorfield)

    # Load the config file as a dictionary
    with open(config, "rb") as f:
        config = tomllib.load(f)
    # Extract the vectorfield parameters
    vfparams = config["vectorfield"][vectorfield]

    xlim = vfparams.get("xlim", (-1, 1))
    ylim = vfparams.get("ylim", (-1, 1))

    # Create a meshgrid for the vector field
    x = jnp.linspace(xlim[0], xlim[1], 1000)
    y = jnp.linspace(ylim[0], ylim[1], 1000)
    X, Y = jnp.meshgrid(x, y)
    T = jnp.zeros_like(X) + t
    # Compute the vector field
    U, V = vectorfield_fun(X, Y, T)
    # Compute the module of the vector field
    module = jnp.sqrt(U**2 + V**2)
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))
    # Plot the mask as a colormap
    # We use a colormap that goes from blue (low values) to red (high values)
    ax.imshow(
        module,
        extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
        origin="lower",
        cmap="coolwarm",
        vmin=0,
        vmax=2,
    )
    # Plot the vector field
    # We will see an arrow every 0.25 units in both directions
    step = 0.25
    x = jnp.arange(xlim[0] - step, xlim[1] + step, step)
    y = jnp.arange(ylim[0] - step, ylim[1] + step, step)
    X, Y = jnp.meshgrid(x, y)
    T = jnp.zeros_like(X) + t
    U, V = vectorfield_fun(X, Y, T)
    ax.quiver(X, Y, U, V)
    # Plot source and destination
    src = jnp.array(vfparams["src"])
    dst = jnp.array(vfparams["dst"])
    ax.plot(src[0], src[1], "ro", markersize=10, label="Source")
    ax.plot(dst[0], dst[1], "go", markersize=10, label="Destination")
    # Set the axis limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    # Set the axis labels
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    # Set the title
    ax.set_title(f"Viable Area for {vectorfield} Vector Field")
    # Show the plot
    plt.tight_layout()
    t = str(round(t, 2)).replace(".", "_")  # Replace dot with underscore for filename
    plt.savefig(f"./output/area_{vectorfield}_{t}.png", dpi=300)
    plt.close()


def run_single_simulation(
    vectorfield: str = "fourvortices",
    cmaes_K: int = 6,
    cmaes_L: int = 200,
    cmaes_numpieces: int = 1,
    cmaes_popsize: int = 500,
    cmaes_sigma: float = 2,
    cmaes_tolfun: float = 1e-3,
    cmaes_damping: float = 1.0,
    cmaes_maxfevals: int = 500000,
    cmaes_seed: int = 0,
    fms_tolfun: float = 1e-6,
    fms_damping: float = 0.5,
    fms_maxfevals: int = 500000,
    path_img: str = "./output",
    path_config: str = "config.toml",
    verbose: bool = True,
):
    """
    Run a single simulation to find an optimal path from source to destination.

    Parameters
    ----------
    vectorfield : str, optional
        The name of the vector field function to use, by default "zero".
    cmaes_K : int, optional
        The number of control points for CMA-ES, by default 6.
    cmaes_L : int, optional
        The number of segments for CMA-ES, by default 64.
    cmaes_numpieces : int, optional
        The number of pieces for CMA-ES, by default 1.
    cmaes_popsize : int, optional
        The population size for CMA-ES, by default 2000.
    cmaes_sigma : float, optional
        The initial standard deviation for CMA-ES, by default 1.
    cmaes_tolfun : float, optional
        The tolerance for the function value in CMA-ES, by default 0.0001.
    fms_tolfun : float, optional
        The tolerance for the function value in FMS, by default 1e-6.
    fms_damping : float, optional
        The damping factor for FMS, by default 0.9.
    fms_maxfevals : int, optional
        The maximum number of iterations for FMS, by default 50000.
    path_img : str, optional
        The path to save output images, by default "./output".
    """
    # Load the config file as a dictionary
    with open(path_config, "rb") as f:
        config = tomllib.load(f)

    # Extract the vectorfield parameters
    vfparams = config["vectorfield"][vectorfield]
    src = jnp.array(vfparams["src"])
    dst = jnp.array(vfparams["dst"])
    travel_stw = vfparams.get("travel_stw", None)
    travel_time = vfparams.get("travel_time", None)
    land_xlim = vfparams.get("xlim", None)
    land_ylim = vfparams.get("ylim", None)

    # Load the vectorfield function
    vectorfield_module = __import__(
        "routetools.vectorfield", fromlist=["vectorfield_" + vectorfield]
    )
    vectorfield_fun = getattr(vectorfield_module, "vectorfield_" + vectorfield)

    # CMA-ES optimization algorithm
    curve_cmaes, dict_cmaes = optimize(
        vectorfield_fun,
        src,
        dst,
        travel_stw=travel_stw,
        travel_time=travel_time,
        K=cmaes_K,
        L=cmaes_L,
        num_pieces=cmaes_numpieces,
        popsize=cmaes_popsize,
        sigma0=cmaes_sigma,
        tolfun=cmaes_tolfun,
        damping=cmaes_damping,
        maxfevals=cmaes_maxfevals,
        seed=cmaes_seed,
        verbose=verbose,
    )
    cost_cmaes = dict_cmaes["cost"]

    # Create a straight line from source to destination,
    # with as many points as the CMA-ES curve
    num_points = curve_cmaes.shape[0]
    curve_straight = jnp.linspace(src, dst, num_points)

    # FMS variational algorithm (refinement)
    curve_fms, dict_fms = optimize_fms(
        vectorfield_fun,
        curve=curve_straight,
        travel_stw=travel_stw,
        travel_time=travel_time,
        tolfun=fms_tolfun,
        damping=fms_damping,
        maxfevals=fms_maxfevals,
        verbose=verbose,
    )
    # FMS returns an extra dimensions, we ignore that
    curve_fms = curve_fms[0]
    cost_fms = dict_fms["cost"][0]  # FMS returns a list of costs

    # FMS after CMA-ES
    curve_bers, dict_bers = optimize_fms(
        vectorfield_fun,
        curve=curve_cmaes,
        travel_stw=travel_stw,
        travel_time=travel_time,
        tolfun=fms_tolfun,
        damping=fms_damping,
        maxfevals=fms_maxfevals,
        verbose=verbose,
    )
    # FMS returns an extra dimensions, we ignore that
    curve_bers = curve_bers[0]
    cost_bers = dict_bers["cost"][0]  # FMS returns a list of costs

    # Plot them
    fig, ax = plot_curve(
        vectorfield_fun,
        [curve_cmaes, curve_fms, curve_bers],
        ls_name=["CMA-ES", "FMS", "BERS"],
        ls_cost=[cost_cmaes, cost_fms, cost_bers],
        xlim=land_xlim,
        ylim=land_ylim,
    )
    ax.set_title(f"{vectorfield}")
    fig.savefig(f"{path_img}/literature_{vectorfield}.png")
    plt.close(fig)


def plot_best_values(
    path_csv: str = "./output/results_noland.csv",
    folder: str = "./output/",
    col: str = "gain_fms",
    ascending: bool = False,
    size: int = 2,
):
    """Generate plots for the examples with the highest values.

    Parameters
    ----------
    path_csv : str, optional
        Path to the CSV file containing the results of the experiments,
        by default "./output/results_noland.csv"
    folder : str, optional
        The directory containing the results CSV file and JSON files,
        by default "output".
    col : str, optional
        The column to sort by, by default "gain_fms".
    ascending : bool, optional
        Whether to sort the values in ascending order, by default False.
    size : int, optional
        The number of top examples to plot per vectorfield, by default 2.
    """
    df = pd.read_csv(path_csv)

    # Filter the rows with highest col, grouped by vectorfield
    df_filtered = df.groupby("vectorfield")[["vectorfield", "json", col]].apply(
        lambda x: x.nsmallest(size, col) if ascending else x.nlargest(size, col)
    )

    # Plot the top examples
    for multiidx, row in df_filtered.iterrows():
        vf, idx = multiidx
        json_id = int(row["json"])
        print(f"Best {col}: processing {json_id}...")
        fig, ax = plot_route_from_json(f"{folder}/noland/{json_id:06d}.json")
        fig.savefig(f"{folder}/{vf}_{json_id}.png")
        plt.close(fig)


def plot_land_avoidance(
    path_csv: str = "./output/results_land.csv",
    folder: str = "./output/",
    size: int = 3,
):
    """
    Generate and save plots for land avoidance analysis based on simulation results.

    This function reads simulation results from a CSV file, identifies the worst
    examples based on the cost function, and generates plots to visualize the land
    avoidance behavior for different water levels, resolutions, and random seeds.

    Parameters
    ----------
    folder : str, optional
        The directory containing the results CSV file and JSON files,
        by default "output".
    """
    # Read the results CSV file
    df_land = pd.read_csv(path_csv)
    # Mask the three cases we are working with
    mask_easy = (df_land["resolution"] == 3) & (df_land["water_level"] == 0.9)
    mask_medium = (df_land["resolution"] == 4) & (df_land["water_level"] == 0.8)
    mask_hard = (df_land["resolution"] == 5) & (df_land["water_level"] == 0.7)

    # Create a new column "complexity" to categorize the experiments
    df_land["complexity"] = np.nan
    df_land.loc[mask_easy, "complexity"] = 1
    df_land.loc[mask_medium, "complexity"] = 2
    df_land.loc[mask_hard, "complexity"] = 3

    # Drop the experiments not belonging to any of these categories
    df_land = df_land.dropna(subset=["complexity"])

    # Generate plots for the worst ten examples
    for _, df_sub in df_land.groupby("complexity"):
        # Sort by gain
        df_sub = df_sub.sort_values("gain_fms", ascending=True)
        # Take the worst three examples
        df_worst = df_sub.tail(3)
        for _, row in df_worst.iterrows():
            # Extract the configuration parameters

            # Load the JSON file for the identified example
            json_id = int(row["json"])
            print(f"Land avoidance: processing {json_id}...")

            # Print what was the CMA-ES configuration
            fig, ax = plot_route_from_json(f"{folder}/land/{json_id:06d}.json")
            fig.savefig(f"{folder}/land_{json_id}.png")
            plt.close(fig)


def experiment_parameter_sensitivity(
    path_csv: str = "./output/results_noland.csv", folder: str = "./output/"
):
    """Plot the results of the BERS experiments by parameter sensitivity.

    Parameters
    ----------
    path_csv : str, optional
        Path to the CSV file containing the results of the experiments,
        by default "./output/results_noland.csv"
    folder : str, optional
        Path to the folder where the plots will be saved, by default "./output/"
    """
    # Read the CSV file containing the results of the experiments
    df_noland = pd.read_csv(path_csv)

    # Assign literature cost using "vectorfield"
    df_noland["cost_reference"] = df_noland["vectorfield"].map(DICT_COST_LITERATURE)

    # Choose only the following vectorfields
    ls_vf = ["circular", "doublegyre", "fourvortices", "swirlys", "techy"]
    df_noland = df_noland[df_noland["vectorfield"].isin(ls_vf)]

    # Compute the difference with the literature cost
    df_noland["diff_cmaes"] = 1 - df_noland["cost_cmaes"] / df_noland["cost_reference"]
    df_noland["diff_fms"] = 1 - df_noland["cost_fms"] / df_noland["cost_reference"]
    df_noland["niter"] = df_noland["niter_cmaes"] + df_noland["niter_fms"]

    # Compute the mean of all these values, grouped by vectorfield, K, and sigma0
    df_noland = df_noland.groupby(
        ["vectorfield", "K", "sigma0"],
        as_index=False,
    ).mean(numeric_only=True)

    def _helper_experiment_parameter_sensitivity(
        col1: str,
        col2: str,
        title: str = "",
        limits2: tuple = (0, 500),  # Limits for col2 heatmap
    ):
        # Plot multiple heatmaps in a single figure:
        # First row: col1 for each vectorfield
        # Second row: col2 for each vectorfield
        # In each heatmap:
        # x-axis: "K" (number of control points for Bézier curve)
        # y-axis: "sigma0" (standard deviation of the CMA-ES distribution)
        # color: col1 or col2 (depending on the heatmap)
        num_cols = len(ls_vf)
        fig, axs = plt.subplots(
            nrows=2, ncols=num_cols, figsize=(num_cols * 2, 4), sharex=True, sharey=True
        )
        fig.suptitle(title, fontsize=16)
        for i, vf in enumerate(ls_vf):
            vfname = DICT_VF_NAMES.get(vf, vf)
            # Filter the DataFrame for the current vectorfield
            df_vf = df_noland[df_noland["vectorfield"] == vf]

            # Pivot the DataFrame to create a heatmap
            heatmap_data = df_vf.pivot(index="sigma0", columns="K", values=col1)
            # Plot the heatmap for col1
            cmap = matplotlib.cm.bwr_r
            cmap.set_bad("black", 1.0)
            im1 = axs[0, i].imshow(
                heatmap_data,
                cmap=cmap,
                aspect="equal",
                # Center the map around zero
                vmin=-0.5,
                vmax=0.5,
            )
            axs[0, i].set_title(vfname)
            if i == 0:
                axs[0, i].set_ylabel(r"$\sigma_0$")
            axs[0, i].set_xticks(np.arange(len(heatmap_data.columns)))
            axs[0, i].set_xticklabels(heatmap_data.columns)
            axs[0, i].set_yticks(np.arange(len(heatmap_data.index)))
            axs[0, i].set_yticklabels(heatmap_data.index)

            # Plot the heatmap for col2
            heatmap_data = df_vf.pivot(index="sigma0", columns="K", values=col2)
            cmap = matplotlib.cm.Reds
            cmap.set_bad("black", 1.0)
            im2 = axs[1, i].imshow(
                heatmap_data,
                cmap=cmap,
                aspect="equal",
                vmin=limits2[0],
                vmax=limits2[1],  # Set limits for col2 heatmap
            )
            axs[1, i].set_xlabel("K")
            if i == 0:
                axs[1, i].set_ylabel(r"$\sigma_0$")
            axs[1, i].set_xticks(np.arange(len(heatmap_data.columns)))
            axs[1, i].set_xticklabels(heatmap_data.columns)
            axs[1, i].set_yticks(np.arange(len(heatmap_data.index)))
            axs[1, i].set_yticklabels(heatmap_data.index)
        # Add colorbars to im1 and im2
        cbar1 = fig.colorbar(
            im1,
            ax=axs[0, :],
            orientation="vertical",
            fraction=0.02,
            location="right",
        )
        cbar1.set_label("Cost reduction factor")
        cbar2 = fig.colorbar(
            im2,
            ax=axs[1, :],
            orientation="vertical",
            fraction=0.02,
            location="right",
        )
        cbar2.set_label("Number of iterations")

    # GRAPH 1
    _helper_experiment_parameter_sensitivity(
        col1="diff_cmaes", col2="niter_cmaes", limits2=[0, 200]
    )
    plt.savefig(folder + "parameter_sensitivity_cmaes.png", dpi=300)
    plt.close()

    # GRAPH 2
    _helper_experiment_parameter_sensitivity(
        col1="diff_fms", col2="niter", limits2=[0, 5000]
    )
    plt.savefig(folder + "parameter_sensitivity_bers.png", dpi=300)
    plt.close()


def experiment_land_complexity(
    path_csv: str = "./output/results_land.csv", folder: str = "./output/"
) -> None:
    """Plot the results of the BERS experiments by land complexity level.

    The land complexity is defined by the resolution and water level:
    - Easy: resolution 3, water level 0.9
    - Medium: resolution 4, water level 0.8
    - Hard: resolution 5, water level 0.7

    Parameters
    ----------
    path_csv : str, optional
        Path to the CSV file containing the results of the experiments,
        by default "./output/results_land.csv"
    folder : str, optional
        Path to the folder where the plots will be saved, by default "./output/"
    """
    # Read the CSV file containing the results of the experiments
    df_land = pd.read_csv(path_csv)

    # Define the land complexity levels based on resolution and water level
    mask_easy = (df_land["resolution"] == 3) & (df_land["water_level"] == 0.9)
    mask_medium = (df_land["resolution"] == 4) & (df_land["water_level"] == 0.8)
    mask_hard = (df_land["resolution"] == 5) & (df_land["water_level"] == 0.7)

    # Create a new column "complexity" to categorize the experiments
    df_land["complexity"] = np.nan
    df_land.loc[mask_easy, "complexity"] = 1
    df_land.loc[mask_medium, "complexity"] = 2
    df_land.loc[mask_hard, "complexity"] = 3

    # Drop the experiments not belonging to any of these categories
    df_land = df_land.dropna(subset=["complexity"])

    # Print the number of examples for each case
    print(
        "Number of examples for each complexity level:\n",
        df_land["complexity"].value_counts(),
    )

    # Boxplot: one box for each complexity level
    # Vertical axis: "cost_fms"
    # We also overlay a line showing the average "comp_time" for each complexity level
    # Use a second y-axis for the average "comp_time" line
    plt.figure(figsize=(6, 3))

    ax = plt.gca()  # Get current axes
    ax = df_land.boxplot(
        column="cost_fms", by="complexity", grid=False, showfliers=False, ax=ax
    )
    ax.set_ylim(8.5, 12)
    ax.set_xlabel("Land Complexity Level (affects resolution and water level)")
    ax.set_ylabel("Travel time")
    ax.set_title("Results of BERS by Land Complexity Level")

    # Overlay average comp_time
    avg_comp_time = df_land.groupby("complexity")["comp_time"].mean()
    avg_comp_time_cmaes = df_land.groupby("complexity")["comp_time_cmaes"].mean()
    avg_comp_time_fms = df_land.groupby("complexity")["comp_time_fms"].mean()
    ax2 = ax.twinx()
    # Draw each computation time in three tones of red
    avg_comp_time_cmaes.plot(
        ax=ax2, color="orange", marker="o", linestyle="-", label="CMA-ES step"
    )
    avg_comp_time_fms.plot(
        ax=ax2, color="red", marker="o", linestyle="-", label="FMS step"
    )
    avg_comp_time.plot(
        ax=ax2, color="darkred", marker="o", linestyle="-", label="Total"
    )
    ax2.legend(loc="upper left")
    ax2.set_ylabel("Average Computation Time (s)", color="red")
    ax2.tick_params(axis="y", labelcolor="red")
    ax2.set_ylim(0, 30)

    # Add a grid
    # ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    plt.suptitle("")
    plt.xticks([1, 2, 3], ["Easy", "Medium", "Hard"])
    plt.tight_layout()
    plt.savefig(folder + "land_complexity.png", dpi=300)
    plt.close()


def main(folder: str = "./output/"):
    """Run the experiments and plot the results."""
    print("---\nSINGLE SIMULATION\n---")
    run_single_simulation("fourvortices", path_img=folder)
    run_single_simulation("swirlys", path_img=folder)
    print("\n---\nPARAMETER SENSITIVITY EXPERIMENTS\n---")
    experiment_parameter_sensitivity(folder=folder)
    print("\n---\nLAND COMPLEXITY EXPERIMENTS\n---")
    experiment_land_complexity(folder=folder)
    print("\n---\nBIGGEST FMS GAINS\n---")
    plot_best_values(folder=folder, col="gain_fms", ascending=False, size=4)
    print("\n---\nBIGGEST BERS SAVINGS\n---")
    plot_best_values(folder=folder, col="cost_fms", ascending=True, size=4)
    print("\n---\nLAND AVOIDANCE ANALYSIS\n---")
    plot_land_avoidance(folder=folder, size=10)
    for t in [0, 0.2, 0.5, 0.7, 1.0]:
        print(f"\n---\nVIABLE AREA FOR TECHY VECTOR FIELD AT t={t}\n---")
        plot_viable_area("techy", t=t)
    plot_viable_area("fourvortices")
    plot_viable_area("circular")
    plot_viable_area("doublegyre")


if __name__ == "__main__":
    typer.run(main)  # Use Typer to run the main function
