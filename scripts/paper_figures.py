import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


def generate_dataframe(folder: Path) -> pd.DataFrame:
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
    xmin, xmax = -5, 15
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
    plt.xlabel("Time savings [%] of BERS over orthodromic")
    plt.ylabel("Count (no. benchmarks)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path_output / "fullyear_speed.png", dpi=300)
    plt.show()


def main(path_json: str = "output/json_benchmark"):
    """Run a single benchmark instance and save the results to output/."""
    df = generate_dataframe(Path(path_json))
    fullyear_savings_speed(df, Path("output"))


if __name__ == "__main__":
    main()
