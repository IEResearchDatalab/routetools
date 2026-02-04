import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from routetools.benchmark import load_benchmark_instance
from routetools.cost import cost_function
from routetools.plot import plot_curve, plot_distance_to_end_vs_time

MODEL = "BERS"


def season(week: int) -> str:
    """Return the season name based on the week number.

    Parameters
    ----------
    week : int
        Week number (1-52).

    Returns
    -------
    str
        Season name ("winter", "spring", "summer", "autumn").
    """
    if week <= 11:
        return "winter"
    elif week <= 24:
        return "spring"
    elif week <= 37:
        return "summer"
    elif week <= 50:
        return "autumn"
    else:
        return "winter"


def generate_dataframe(path_output: Path) -> pd.DataFrame:
    """Generate a combined DataFrame from BERS and orthodromic results."""
    # First check if the dataframe already exists inside the folder
    pth_df = path_output / "dataframe.csv"
    ls_data = []
    # Loop through all JSON files in the folder and extract relevant data
    for path_json in path_output.glob("*.json"):
        with open(path_json) as file:
            data: dict[str, Any] = json.load(file)
            # Drop any keys that are lists
            data = {k: v for k, v in data.items() if not isinstance(v, list)}
            ls_data.append(data)
    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(ls_data)

    # Sanity check: compute mean dn std of cost using all columns:
    # cost_circ, cost_cmaes, cost_fms
    # Then drop the rows where the cost - mean is larger than 3*std
    cost_mean = df[["cost_circ", "cost_cmaes", "cost_fms"]].values.mean()
    cost_std = df[["cost_circ", "cost_cmaes", "cost_fms"]].values.std()
    mask_valid = np.all(
        np.abs(df[["cost_circ", "cost_cmaes", "cost_fms"]] - cost_mean) < 3 * cost_std,
        axis=1,
    )
    df = df[mask_valid].copy()
    # If there was some invalid data, print how many rows were dropped
    if len(mask_valid) != len(df):
        print(f"Dropped {len(mask_valid) - len(df)} rows due to invalid costs.")
        # Save the invalid rows to a separate CSV for inspection
        df_invalid = df[~mask_valid]
        path_invalid = path_output / "dataframe_invalid.csv"
        df_invalid.to_csv(path_invalid, index=False)

    # Calculate the time savings as a percentage
    df["gain"] = 100 * (df["cost_circ"] - df["cost_fms"]) / df["cost_circ"]
    # Calculate relative distance increase
    df["relative_distance"] = (
        100 * (df["distance_fms"] - df["distance_circ"]) / df["distance_circ"]
    )
    # Create week column from date_start
    df["date_start"] = pd.to_datetime(df["date_start"])
    df["week"] = df["date_start"].dt.isocalendar().week
    # Create season column from week
    df["season"] = df["week"].apply(season)
    # Sort by: instance_name, vel_ship, date_start
    df = df.sort_values(by=["instance_name", "vel_ship", "date_start"])
    # Save the DataFrame to a CSV file for future use
    df.to_csv(pth_df, index=False)
    return df


def plot_bers_vs_circumnavigation(
    df: pd.DataFrame, path_output: Path, data_path: Path = Path("data")
):
    """Plot BERS vs Orthodromic travel times for all benchmark instances.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing benchmark results.
    path_output : Path
        Path to the output folder where the plot will be saved.
    """
    # List all instances and speeds
    ls_instances = df["instance_name"].unique()
    ls_speeds = df["vel_ship"].unique()

    for instance_name in ls_instances:
        mask_instance = df["instance_name"] == instance_name
        for vel_ship in ls_speeds:
            mask_vel = df["vel_ship"] == vel_ship
            df_sub: pd.DataFrame = df[mask_instance & mask_vel].copy()
            # Find the biggest gain and take its IDs
            idx_max_gain = df_sub["gain"].idxmax()
            instance_name, date_start, vel_ship = df_sub.loc[
                idx_max_gain, ["instance_name", "date_start", "vel_ship"]
            ]
            name: str = instance_name.replace("-", "")
            # Turn date into "YYMMDD"
            date_str: str = date_start.strftime("%y%m%d")
            unique_name = f"{name}_{date_str}_{vel_ship}"
            # Load the curves from the JSON files
            with open(Path("output/json_benchmark") / f"{unique_name}.json") as f:
                data_bers = json.load(f)
            # Extract the curves
            curve_cmaes = jnp.array(data_bers["curve_cmaes"])
            curve_fms = jnp.array(data_bers["curve_fms"])
            curve_circ = jnp.array(data_bers["curve_circ"])
            # Extract the vectorfield and land data
            dict_instance = load_benchmark_instance(
                instance_name,
                date_start=data_bers["date_start"],
                vel_ship=vel_ship,
                data_path=data_path,
            )

            dict_costs = {}

            for name, curve in [
                ("Orthodromic", curve_circ),
                ("CMA-ES", curve_cmaes),
                ("FMS", curve_fms),
            ]:
                cost = cost_function(
                    vectorfield=dict_instance["vectorfield"],
                    curve=curve[jnp.newaxis, :, :],
                    wavefield=dict_instance["wavefield"],
                    travel_stw=vel_ship,
                    travel_time=None,
                    spherical_correction=True,
                )
                # Cost is in seconds, convert to hours
                dict_costs[name] = int(cost[0] / 3600.0)

            # ------------------------------
            # Plot the curve
            # ------------------------------

            vectorfield = dict_instance["vectorfield"]
            land = dict_instance["land"]
            fig, ax = plot_curve(
                vectorfield=vectorfield,
                ls_curve=[curve_circ, curve_cmaes, curve_fms],
                ls_name=[
                    f"Orthodromic ({dict_costs['Orthodromic']} hrs)",
                    f"CMA-ES ({dict_costs['CMA-ES']} hrs)",
                    f"FMS ({dict_costs['FMS']} hrs)",
                ],
                land=land,
                gridstep=1 / 12,
                figsize=(6, 6),
                xlim=(land.xmin, land.xmax),
                ylim=(land.ymin, land.ymax),
                color_currents=True,
            )
            # Include date and velocity in the title
            ax.set_title(
                f"{instance_name} | {data_bers['date_start']} | {int(2 * vel_ship)} "
                "knots"
            )
            fig.tight_layout()
            # We use redundant naming to avoid too many images
            fig.savefig(
                path_output / f"benchmark_{instance_name}_{int(vel_ship)}.jpg", dpi=300
            )
            plt.close()

            # ------------------------------
            # Plot distance to end vs time
            # ------------------------------
            fig2, ax2 = plot_distance_to_end_vs_time(
                vectorfield=vectorfield,
                ls_curve=[curve_circ, curve_fms],
                ls_name=["Orthodromic", "FMS"],
                name=instance_name,
                vel_ship=vel_ship,
            )
            fig2.tight_layout()
            fig2.savefig(
                path_output / f"distance_{instance_name}_{int(vel_ship)}.jpg", dpi=300
            )
            plt.close()


def fullyear_savings_speed(df: pd.DataFrame, path_output: Path):
    """Plot the time savings distribution over ship speeds.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing benchmark results.
    path_output : Path
        Path to the output folder where the plot will be saved.
    """
    # Initialize matplotlib colors
    colors = ["red", "blue", "green"]
    xmin, xmax = -2, 10
    plt.figure(figsize=(6, 3))

    for idx, (v, df_sub) in enumerate(df.groupby("vel_ship")):
        plt.hist(
            df_sub["gain"],
            range=(xmin, xmax),
            bins=140,
            alpha=0.5,
            label=f"Speed: {int(v * 2)} knots",
            color=colors[idx],
        )
        # Fit a curve to the data using kde
        dens = sm.nonparametric.KDEUnivariate(df_sub["gain"])
        dens.fit()
        x = np.linspace(xmin, xmax, 1000)
        y = dens.evaluate(x)
        plt.plot(x, y * 100, color=colors[idx])

    plt.grid()
    plt.xlabel(f"Time savings [%] of {MODEL} over orthodromic")
    plt.ylabel("Count (no. instances)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path_output / "fullyear_speed.png", dpi=300)
    plt.show()


def fullyear_savings_odp(df: pd.DataFrame, path_output: Path):
    """Plot the time savings over the weeks of the year for each ODP benchmark.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing benchmark results.
    path_output : Path
        Path to the output folder where the plots will be saved.
    """
    dict_color = {
        "winter": "lightblue",
        "spring": "lightgreen",
        "summer": "yellow",
        "autumn": "orange",
    }
    color = df["season"].map(dict_color)

    ls_savings = []
    ls_negative = []

    # Iterate over each benchmark and create a plot
    for benchmark, df2 in df.groupby("instance_name"):
        for vel, df3 in df2.groupby("vel_ship"):
            _ = plt.figure(figsize=(10, 5))
            ax = plt.gca()
            # Set the x-axis and y-axis data
            x = df3["week"]
            y1 = df3["cost_circ"]
            y1 = 100 * (y1 - y1.mean()) / y1.mean()
            y2 = df3["gain"]
            # y3 = df3["rel_dist"]

            # Plot the time as a bar
            ax.bar(x, y1, color=color, label="Orthodromic Time w.r.t. Average")
            # ax.plot(x, y3, color="darkgreen", label=f"{MODEL} Extra Distance")
            # Sort the x and y2 based on x to have a proper line plot
            sorted_indices = np.argsort(x)
            x = x.iloc[sorted_indices]
            y2 = y2.iloc[sorted_indices]
            ax.plot(x, y2, color="red", label=f"{MODEL} Savings")
            ax.axhline(0, color="black", linestyle="--")
            ax.set_xlabel("Week")
            ax.set_ylabel("Percentage Difference [%]")
            ax.set_title(f"ODP: {benchmark} | Speed: {int(2 * vel)} knots")
            ax.legend()
            plt.savefig(path_output / f"fullyear_{vel:02d}_{benchmark}.png")
            plt.close()

            ls_savings.append(
                {
                    "odp": benchmark,
                    "velocity": vel,
                    "mean": y2.mean(),
                    "median": y2.median(),
                    "std": y2.std(),
                }
            )

            if df3["gain"].min() < 0:
                # Add a row per negative gain
                for _idx, row in df3[df3["gain"] < 0].iterrows():
                    ls_negative.append(
                        {
                            "benchmark": benchmark,
                            "velocity": vel,
                            "week": row["week"],
                            "gain": row["gain"],
                            "time": row["cost_fms"],
                            "time_circ": row["cost_circ"],
                            # "rel_dist": row["rel_dist"],
                        }
                    )

    # Create a DataFrame with the savings
    df_savings = pd.DataFrame(ls_savings)
    df_savings.to_csv(
        path_output / "fullyear_odp.csv", index=False, float_format="%.2f"
    )
    if len(ls_negative) > 0:
        df_neg = pd.DataFrame(ls_negative)
        df_neg.sort_values(["velocity", "benchmark"]).to_csv(
            path_output / "fullyear_odp_negative.csv", index=False, float_format="%.2f"
        )


def format_instance_name(name: str) -> str:
    """Format instance name to include an arrow pointing to the destination.

    Parameters
    ----------
    name : str
        Original instance name in the format 'START-END'.

    Returns
    -------
    str
        Formatted instance name with arrow, e.g., 'START > END' or 'END < START'.
    """
    parts = name.split("-")
    if len(parts) != 2:
        return name
    start, end = parts
    # Simple heuristic: if start port code is lexicographically smaller than end,
    # use '>'
    if start < end:
        return f"{start} > {end}"
    else:
        return f"{end} < {start}"


def boxplot_gains_per_instance(df: pd.DataFrame, path_output: Path, vel_ship: float):
    """Create boxplots of time savings per benchmark instance.

    Percent reduction in travel time achieved by the algorithm compared to the
    orthodromic route at 6 knots. ODPs are sorted by decreasing distance along the
    X-axis, and formatted such that the arrow '>' always points to the destination,
    i.e., 'D < O', 'O > D'.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing benchmark results.
    path_output : Path
        Path to the output folder where the plot will be saved.
    vel_ship : float
        Ship velocity to filter the DataFrame.
    """
    df_sub = df[df["vel_ship"] == vel_ship].copy()

    # New instance names with arrows
    df_sub["instance_name"] = df_sub["instance_name"].apply(format_instance_name)
    # Sort instances by mean distance
    order_instances = (
        df_sub.groupby("instance_name")["distance_circ"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    # Create boxplot
    box = ax.boxplot(
        [df_sub[df_sub["instance_name"] == inst]["gain"] for inst in order_instances],
        positions=np.arange(len(order_instances)),
        patch_artist=True,
        medianprops=dict(color="black"),
    )
    # Color the boxes
    for patch in box["boxes"]:
        patch.set_facecolor("lightblue")
    # Set x-ticks and labels
    ax.set_xticks(np.arange(len(order_instances)))
    ax.set_xticklabels(order_instances, rotation=45, ha="right")
    ax.set_ylabel(f"Time Savings [%] of {MODEL} over Orthodromic")
    ax.set_title(f"Time Savings per Benchmark Instance at {int(2 * vel_ship)} knots")
    ax.axhline(0, color="red", linestyle="--")
    plt.tight_layout()
    plt.savefig(path_output / f"boxplot_gains_{int(2 * vel_ship)}knots.png", dpi=300)
    plt.close()


def main(
    path_output: str = "output/json_benchmark",
    data_path: str = "../weather-routing-benchmarks/data",
):
    """Generate the figures for the paper from benchmark results.

    Requires to run first:
    - scripts/realworld/results.py
    """
    try:
        df = generate_dataframe(Path(path_output))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Results not found. Please run first scripts/realworld/results.py"
        ) from exc
    plot_bers_vs_circumnavigation(
        df, path_output=Path("output"), data_path=Path(data_path)
    )
    fullyear_savings_speed(df, Path("output"))
    fullyear_savings_odp(df, Path("output"))
    boxplot_gains_per_instance(df, Path("output"), vel_ship=3)


if __name__ == "__main__":
    main()
