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
from routetools.plot import plot_curve

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


def orthodromic_distance(instance_name: str) -> float:
    """Hardcoded orthodromic distances for benchmark instances.

    Parameters
    ----------
    instance_name : str
        Name of the benchmark instance.

    Returns
    -------
    float
        Orthodromic distance in nautical miles.
    """
    dict_dist = {
        "DEHAM-USNYC": 6248,
        "USNYC-DEHAM": 6248,
        "EGPSD-ESALG": 3533,
        "ESALG-EGPSD": 3533,
        "PABLB-PECLL": 2474,
        "PECLL-PABLB": 2474,
        "PAONX-USNYC": 3617,
        "USNYC-PAONX": 3617,
    }
    return dict_dist.get(instance_name, 0.0)


def generate_individual_dataframe(folder: Path) -> pd.DataFrame:
    """Generate a pandas DataFrame from all JSON files in the given folder.

    Parameters
    ----------
    folder : Path
        Path to the folder containing JSON files.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the extracted data from the JSON files.
    """
    # First check if the dataframe already exists inside the folder
    pth_df = folder / "dataframe.csv"
    ls_data = []
    # Loop through all JSON files in the folder and extract relevant data
    for path_json in folder.glob("*.json"):
        with open(path_json) as file:
            data: dict[str, Any] = json.load(file)
            # Drop any keys that are lists
            data = {k: v for k, v in data.items() if not isinstance(v, list)}
            # Include the ID of the JSON file
            data["id"] = path_json.stem
            ls_data.append(data)
    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(ls_data)
    # Save the DataFrame to a CSV file for future use
    df.to_csv(pth_df, index=False)
    return df


def generate_dataframe(
    path_bers: str = "output/json_benchmark",
    path_ortho: str = "output/json_orthodromic",
):
    """Generate a combined DataFrame from BERS and orthodromic results."""
    df_bers = generate_individual_dataframe(Path(path_bers))
    df_ortho = generate_individual_dataframe(Path(path_ortho))
    # Merge the two DataFrames on instance_name, date_start, vel_ship
    df = pd.merge(
        df_bers,
        df_ortho,
        on=["instance_name", "date_start", "vel_ship"],
        suffixes=("", "_ortho"),
    )
    # Rename cost to cost_ortho
    df = df.rename(columns={"cost": "cost_ortho"})
    # Calculate the time savings as a percentage
    df["gain"] = 100 * (df["cost_ortho"] - df["cost_fms"]) / df["cost_ortho"]
    # Calculate relative distance increase
    # df["rel_dist"] = 100 * (df["dist_fms"] - df["dist_ortho"]) / df["dist_ortho"]
    # Create week column from date_start
    df["date_start"] = pd.to_datetime(df["date_start"])
    df["week"] = df["date_start"].dt.isocalendar().week
    # Create season column from week
    df["season"] = df["week"].apply(season)
    # Add orthodromic distance column
    df["dist_ortho"] = df["instance_name"].apply(orthodromic_distance)
    # Sort by: instance_name, vel_ship, date_start
    df = df.sort_values(by=["instance_name", "vel_ship", "date_start"])
    return df


def plot_bers_vs_orthodromic(
    df: pd.DataFrame, instance_name: str, vel_ship: float, path_output: Path
):
    """Plot BERS vs Orthodromic travel times for a given benchmark and ship speed.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing benchmark results.
    instance_name : str
        Name of the benchmark instance.
    vel_ship : float
        Ship velocity to filter the DataFrame.
    path_output : Path
        Path to the output folder where the plot will be saved.
    """
    df_sub = df[
        (df["instance_name"] == instance_name) & (df["vel_ship"] == vel_ship)
    ].copy()
    # Find the biggest gain and take its IDs
    idx_max_gain = df_sub["gain"].idxmax()
    ids_max_gain = df_sub.loc[idx_max_gain, ["id", "id_ortho"]]
    # Load the curves from the JSON files
    with open(Path("output/json_benchmark") / f"{ids_max_gain['id']}.json") as f:
        data_bers = json.load(f)
    with open(
        Path("output/json_orthodromic") / f"{ids_max_gain['id_ortho']}.json"
    ) as f:
        data_ortho = json.load(f)
    # Extract the curves
    curve_cmaes = jnp.array(data_bers["curve_cmaes"])
    curve_fms = jnp.array(data_bers["curve_fms"])
    curve_ortho = jnp.array(data_ortho["curve"])
    # Extract the vectorfield and land data
    dict_instance = load_benchmark_instance(
        instance_name, date_start=data_bers["date_start"], vel_ship=vel_ship
    )

    dict_costs = {}

    for name, curve in [
        ("Orthodromic", curve_ortho),
        ("CMA-ES", curve_cmaes),
        ("FMS", curve_fms),
    ]:
        cost = cost_function(
            vectorfield=dict_instance["vectorfield"],
            curve=curve[jnp.newaxis, :, :],
            travel_stw=vel_ship,
            travel_time=None,
            spherical_correction=True,
        )
        # Cost is in seconds, convert to hours
        dict_costs[name] = int(cost[0] / 3600.0)

    # Plot the curve
    vectorfield = dict_instance["vectorfield"]
    land = dict_instance["land"]
    plot_curve(
        vectorfield=vectorfield,
        ls_curve=[curve_ortho, curve_cmaes, curve_fms],
        ls_name=[
            f"Orthodromic ({dict_costs['Orthodromic']} hrs)",
            f"CMA-ES ({dict_costs['CMA-ES']} hrs)",
            f"FMS ({dict_costs['FMS']} hrs)",
        ],
        land=land,
        gridstep=0.5,
        figsize=(6, 6),
        xlim=(land.xmin, land.xmax),
        ylim=(land.ymin, land.ymax),
    )
    # Include date and velocity in the title
    plt.title(
        f"{instance_name} | {data_bers['date_start']} | {int(2 * vel_ship)} knots"
    )
    plt.tight_layout()
    # We use redundant naming to avoid too many images
    plt.savefig(f"output/benchmark_{instance_name}.jpg", dpi=300)
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
    xmin, xmax = -50, 50
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
    plt.ylabel("Count (no. benchmarks)")
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
            y1 = df3["cost_ortho"]
            y1 = 100 * (y1 - y1.mean()) / y1.mean()
            y2 = df3["gain"]
            # y3 = df3["rel_dist"]

            # Plot the time as a bar
            ax.bar(x, y1, color=color, label="Orthodromic Time w.r.t. Average")
            # ax.plot(x, y3, color="darkgreen", label=f"{MODEL} Extra Distance")
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
                            "time_circ": row["cost_ortho"],
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
        df_sub.groupby("instance_name")["dist_ortho"]
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


def table_gains_per_season(df: pd.DataFrame, path_output: Path):
    """Generate a LaTeX table of average gains per season.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing benchmark results.
    path_output : Path
        Path to the output folder where the LaTeX table will be saved.
    """
    latex_code = r"""
\begin{table}[htbp]
    \centering
    \caption{Gains of \name{} algorithm over orthodromic route,
    grouped by each season of 2023 in the Northern Hemisphere.}
    \label{tab:seasons}
    \begin{tabular}{lrrr}
    \toprule
    {\textbf{Season}} & \multicolumn{3}{c}{ \textbf{Avg. Gain \% (Std.)}}\\
     & \textbf{6 kn} & \textbf{12 kn} & \textbf{24 kn} \\ \midrule
    Winter & x.x (x.x) & x.x (x.x) & x.x (x.x)\\
    Spring & x.x (x.x) & x.x (x.x) & x.x (x.x)\\
    Summer & x.x (x.x) & x.x (x.x) & x.x (x.x)\\
    Autumn & x.x (x.x) & x.x (x.x) & x.x (x.x)\\ \bottomrule
    \end{tabular}
\end{table}
    """
    # Fill in the table with actual data
    seasons = ["winter", "spring", "summer", "autumn"]
    speeds = [3, 6, 12]  # Corresponding to 6, 12, 24 knots
    for season_name in seasons:
        row = f"{season_name.capitalize()} & "
        for speed in speeds:
            df_sub = df[(df["season"] == season_name) & (df["vel_ship"] == speed)]
            mean_gain = df_sub["gain"].mean()
            std_gain = df_sub["gain"].std()
            row += f"{mean_gain:.1f} ({std_gain:.1f}) & "
        row = row.removesuffix(" & ") + r"\\"
        latex_code = latex_code.replace(
            f"{season_name.capitalize()} & x.x (x.x) & x.x (x.x) & x.x (x.x)\\\\", row
        )
    # Save the LaTeX code to a .tex file
    with open(path_output / "table_gains_per_season.tex", "w") as f:
        f.write(latex_code)


def main(
    path_bers: str = "output/json_benchmark",
    path_ortho: str = "output/json_orthodromic",
):
    """Generate the figures for the paper from benchmark results.

    Requires to run first:
    - scripts/paper_results_benchmark.py
    - scripts/paper_results_orthodromic.py
    """
    df = generate_dataframe(
        path_bers=path_bers,
        path_ortho=path_ortho,
    )
    plot_bers_vs_orthodromic(
        df, instance_name="USNYC-PAONX", vel_ship=3, path_output=Path("output")
    )
    fullyear_savings_speed(df, Path("output"))
    fullyear_savings_odp(df, Path("output"))
    boxplot_gains_per_instance(df, Path("output"), vel_ship=3)
    table_gains_per_season(df, Path("output"))


if __name__ == "__main__":
    main()
