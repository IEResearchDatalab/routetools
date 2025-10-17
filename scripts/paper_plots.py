import tomllib

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer

from routetools.cmaes import optimize
from routetools.fms import optimize_fms
from routetools.plot import plot_curve, plot_route_from_json

COST_LITERATURE = {
    "circular": 1.98,
    "fourvortices": 8.95,
    "doublegyre": 1.01,
    "techy": 1.03,
    "swirlys": 5.73,
}


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
    weight_l1: float = 1.0,
    weight_l2: float = 0.0,
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
    weight_l1 : float, optional
        Weight for the L1 norm in the combined cost. Default is 1.0.
    weight_l2 : float, optional
        Weight for the L2 norm in the combined cost. Default is 0.0.
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
        weight_l1=weight_l1,
        weight_l2=weight_l2,
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
        weight_l1=weight_l1,
        weight_l2=weight_l2,
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
        weight_l1=weight_l1,
        weight_l2=weight_l2,
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


def plot_best_no_land(
    path_csv: str = "./output/results_noland.csv", folder: str = "./output/"
):
    """Generate plots for the best examples without land avoidance.

    Parameters
    ----------
    folder : str, optional
        The directory containing the results CSV file and JSON files,
        by default "output".
    """
    df = pd.read_csv(path_csv)

    # Filter the rows with highest "gain_fms", grouped by vectorfield
    df_filtered = (
        df.groupby("vectorfield")[["vectorfield", "json", "gain_fms"]]
        .apply(lambda x: x.nlargest(1, "gain_fms"))
        .reset_index(drop=True)
        .sort_values("gain_fms", ascending=False)
    )

    # Plot the top examples
    for idx in df_filtered.index:
        row = df_filtered.iloc[idx]
        vf = row["vectorfield"]
        json_id = int(row["json"])
        print(f"Best without land avoidance: processing {json_id}...")
        fig, ax = plot_route_from_json(f"{folder}/noland/{json_id:06d}.json")
        fig.savefig(f"{folder}/best_{vf}.png")
        plt.close(fig)


def plot_biggest_difference(
    path_csv: str = "./output/results_noland.csv", folder: str = "./output/"
):
    """Generate plots for the examples with the biggest FMS savings.

    Parameters
    ----------
    folder : str, optional
        The directory containing the results CSV file and JSON files,
        by default "output".
    """
    df = pd.read_csv(path_csv)

    # Filter the rows with highest "gain_fms", grouped by vectorfield
    df_filtered = (
        df.groupby("vectorfield")[["json", "vectorfield", "gain_fms"]]
        .apply(lambda x: x.nlargest(2, "gain_fms"))
        .reset_index(drop=True)
        .sort_values("gain_fms", ascending=False)
    )

    # Plot the top examples
    for idx in df_filtered.index:
        row = df_filtered.iloc[idx]
        json_id = int(row["json"])
        print(f"Biggest FMS savings: processing {json_id}...")
        fig, ax = plot_route_from_json(f"{folder}/noland/{json_id:06d}.json")
        fig.savefig(f"{folder}/biggest_fms_{idx}.png")
        plt.close(fig)


def plot_land_avoidance(
    path_csv: str = "./output/results_land.csv", folder: str = "./output/"
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
    idx = 0
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
            fig.savefig(f"{folder}/land_avoidance_{idx}.png")
            plt.close(fig)
            idx += 1


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
    df_noland["cost_reference"] = df_noland["vectorfield"].map(COST_LITERATURE)

    # Choose only the following vectorfields
    ls_vf = ["fourvortices", "swirlys"]
    df_noland = df_noland[df_noland["vectorfield"].isin(ls_vf)]

    # Compute percentage errors, clip at 0%
    df_noland["percterr_cmaes"] = (
        df_noland["cost_cmaes"] / df_noland["cost_reference"] * 100 - 100
    ).clip(lower=0)
    df_noland["percterr_fms"] = (
        df_noland["cost_fms"] / df_noland["cost_reference"] * 100 - 100
    ).clip(lower=0)

    df_noland["gain_fms"] = 100 - df_noland["cost_fms"] / df_noland["cost_cmaes"] * 100

    # If any "percterr_cmaes" is equal or higher than 1e10, we warn the user and drop it
    if (df_noland["percterr_cmaes"] >= 1e10).any():
        print(
            "Warning: Some percentage errors for CMA-ES are equal or higher than 1e10. "
            "These will be dropped from the analysis."
        )
        df_noland = df_noland[df_noland["cost_cmaes"] < 1e10]
        # Same with "percterr_fms"
        df_noland = df_noland[df_noland["cost_fms"] < 1e10]

    # We will group results by "K", "sigma0" and compute their average "percterr_cmaes"
    df_noland = (
        df_noland.groupby(["K", "sigma0"])
        .agg(
            avg_percterr_cmaes=("percterr_cmaes", "mean"),
            avg_comp_time_cmaes=("comp_time_cmaes", "mean"),
            avg_gain_fms=("gain_fms", "mean"),
            avg_comp_time_fms=("comp_time_fms", "mean"),
            avg_percterr_fms=("percterr_fms", "mean"),
            avg_comp_time=("comp_time", "mean"),
        )
        .reset_index()
    )

    def _helper_experiment_parameter_sensitivity(
        col1: str,
        col2: str,
        cmap: str,
        vmin: float = 0,
        vmax: float = 100,
        title: str = "",
    ):
        # Plot a heatmap where:
        # x-axis: "K" (number of control points for Bézier curve)
        # y-axis: "sigma0" (standard deviation of the CMA-ES distribution)
        # color: col1
        # We place the number on each cell of the heatmap, using white letters
        # Below each number, we add the computation time too (col2)
        plt.figure(figsize=(8, 6))
        ax = plt.gca()  # Get current axes
        heatmap = ax.pcolor(
            df_noland.pivot(index="sigma0", columns="K", values=col1),
            cmap=cmap,
            edgecolors="k",
            linewidths=0.5,
            vmin=vmin,
            vmax=vmax,  # Set limits for the color scale
        )
        vmean = (vmax + vmin) / 2
        # Add the numbers in each cell
        for (i, j), val in np.ndenumerate(
            df_noland.pivot(index="sigma0", columns="K", values=col1)
        ):
            ax.text(
                j + 0.5,
                i + 0.5,
                f"{val:.2f}%",
                ha="center",
                va="center",
                color="black" if val < vmean else "white",
                fontsize=10,
            )
            # Add computation time below the percentage
            ct = df_noland.loc[
                (df_noland["K"] == df_noland["K"].unique()[j])
                & (df_noland["sigma0"] == df_noland["sigma0"].unique()[i]),
                col2,
            ].values[0]
            ax.text(
                j + 0.5,
                i + 0.3,
                f"CT: {ct:.2f}s",
                ha="center",
                va="center",
                color="black" if val < vmean else "white",
                fontsize=8,
            )
        # Set the ticks and labels for the axes
        ax.set_xticks(np.arange(len(df_noland["K"].unique())) + 0.5)
        ax.set_xticklabels(df_noland["K"].unique())
        ax.set_yticks(np.arange(len(df_noland["sigma0"].unique())) + 0.5)
        ax.set_yticklabels(df_noland["sigma0"].unique())
        ax.set_xlabel("Number of Control Points (K)")
        ax.set_ylabel(r"Standard Deviation of CMA-ES ($\sigma_0$)")
        ax.set_title(title)
        plt.colorbar(heatmap, label="Loss w.r.t. Literature (%)")
        plt.tight_layout()

    # GRAPH 1
    # color: "avg_percterr_cmaes" (average percentage of error)
    _helper_experiment_parameter_sensitivity(
        col1="avg_percterr_cmaes",
        col2="avg_comp_time_cmaes",
        cmap="Reds",
        vmin=0,
        vmax=50,
        title="Parameter Sensitivity of CMA-ES",
    )
    plt.savefig(folder + "parameter_sensitivity_cmaes.png", dpi=300)
    plt.close()

    # GRAPH 2
    # color: "gain_fms" (percentage of gain by FMS compared to CMA-ES)
    _helper_experiment_parameter_sensitivity(
        col1="avg_gain_fms",
        col2="avg_comp_time_fms",
        cmap="Reds",
        vmin=0,
        vmax=20,
        title="Average Percentage of Gain by FMS Compared to CMA-ES",
    )
    plt.savefig(folder + "parameter_sensitivity_fms.png", dpi=300)
    plt.close()

    # GRAPH 3
    # color: "avg_percterr_fms" (average percentage of error for BERS)
    _helper_experiment_parameter_sensitivity(
        col1="avg_percterr_fms",
        col2="avg_comp_time",
        cmap="Reds",
        vmin=0,
        vmax=50,
        title="Average Percentage of Error for BERS",
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
    plt.figure(figsize=(8, 4))

    ax = plt.gca()  # Get current axes
    ax = df_land.boxplot(
        column="cost_fms", by="complexity", grid=False, showfliers=False, ax=ax
    )
    ax.set_ylim(8.5, 11.5)
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
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    plt.suptitle("")
    plt.xticks([1, 2, 3], ["Easy", "Medium", "Hard"])
    plt.tight_layout()
    plt.savefig(folder + "land_complexity.png", dpi=300)
    plt.close()


def main(folder: str = "./output/"):
    """Run the experiments and plot the results."""
    print("---\nSINGLE SIMULATION\n---")
    # run_single_simulation(path_img=folder)
    print("\n---\nBEST EXAMPLES WITHOUT LAND AVOIDANCE\n---")
    # plot_best_no_land(folder=folder)
    print("\n---\nBIGGEST FMS SAVINGS\n---")
    plot_biggest_difference(folder=folder)
    print("\n---\nLAND AVOIDANCE ANALYSIS\n---")
    plot_land_avoidance(folder=folder)
    print("\n---\nPARAMETER SENSITIVITY EXPERIMENTS\n---")
    experiment_parameter_sensitivity(folder=folder)
    print("\n---\nLAND COMPLEXITY EXPERIMENTS\n---")
    experiment_land_complexity(folder=folder)


if __name__ == "__main__":
    typer.run(main)  # Use Typer to run the main function
