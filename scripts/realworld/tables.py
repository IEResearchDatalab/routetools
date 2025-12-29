from pathlib import Path

import pandas as pd

MODEL = "BERS"


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


def main(path_output: str = "output/json_benchmark"):
    """Generate the figures for the paper from benchmark results.

    Requires to run first:
    - scripts/realworld/results.py
    - scripts/realworld/figures.py
    """
    try:
        df = pd.read_csv(Path(path_output) / "dataframe.csv")
    except FileNotFoundError:
        print(
            f"Dataframe not found in {path_output}. "
            "Please run scripts/realworld/figures.py first."
        )
        return
    table_gains_per_season(df, Path("output"))


if __name__ == "__main__":
    main()
