import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

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
    if pth_df.exists():
        df = pd.read_csv(pth_df)
        return df
    ls_data = []
    # Loop through all JSON files in the folder and extract relevant data
    for path_json in folder.glob("*.json"):
        with open(path_json) as file:
            data: dict[str, Any] = json.load(file)
            # Drop any keys that are lists
            data = {k: v for k, v in data.items() if not isinstance(v, list)}
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
    # Sort by: instance_name, vel_ship, date_start
    df = df.sort_values(by=["instance_name", "vel_ship", "date_start"])
    return df


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
    fullyear_savings_speed(df, Path("output"))
    fullyear_savings_odp(df, Path("output"))


if __name__ == "__main__":
    main()
