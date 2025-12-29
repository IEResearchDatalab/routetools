import os

import pandas as pd
import typer

LATEX_CORRELATION = r"""
\begin{table}[htbp]
\caption{Pearson correlation coefficient (PCC) between the percentage error (PE)
produced by  \name{} (compared with the minimum distance), its computation time
and the different parameters of this algorithm.}
\label{tab:correlation}
\begin{tabular}{lrr}
\textbf{Configuration Parameters} & \textbf{PE} & \textbf{Compute time} \\
\toprule
Population size, $P$ & $POPLOSS$ & $POPTIME$ \\
Standard deviation, $\sigma_0$ & $SIGMALOSS$ & $SIGMATIME$ \\
Control points, $K$ & $KLOSS$ & $KTIME$ \\
Number of pieces, $C$ & $NPLOSS$ & $NPTIME$ \\
Waypoints, $L$ & $LLOSS$ & $LTIME$ \\ \bottomrule
\end{tabular}
\end{table}
"""

LATEX_COMPUTATION_TIMES = r"""
\begin{table}[htbp]
\centering
\caption{Mean computation times for different land configurations.}
\label{tab:computation_times}
\begin{tabular}{lllrr}
\textbf{Cost} & \textbf{Time} & \textbf{Water}
& \textbf{CMA-ES} & \textbf{FMS}\\
\textbf{function} & \textbf{dependent} & \textbf{level}
& \textbf{time (s)} & \textbf{time (s)}\\
\toprule
Time & No & 1.0 & $10TNC$ & $10TNF$ \\
& & 0.9 & $9TNC$ & $9TNF$ \\
& & 0.8 & $8TNC$ & $8TNF$ \\
& & 0.7 & $7TNC$ & $7TNF$ \\
\midrule
Time & Yes & 1.0 & $10TYC$ & $10TYF$ \\
& & 0.9 & $9TYC$ & $9TYF$ \\
& & 0.8 & $8TYC$ & $8TYF$ \\
& & 0.7 & $7TYC$ & $7TYF$ \\
\midrule
Fuel & & 1.0 & $10FNC$ & $10FNF$ \\
& & 0.9 & $9FNC$ & $9FNF$ \\
& & 0.8 & $8FNC$ & $8FNF$ \\
& & 0.7 & $7FNC$ & $7FNF$ \\
\bottomrule
\end{tabular}
\end{table}
"""


def table_parameter_search_correlation(
    path_csv: str = "./output/results_noland.csv", folder: str = "output"
):
    """Generate a LaTeX table with the correlation between loss and parameters.

    Parameters
    ----------
    folder : str, optional
        The directory containing the results CSV file, by default "output".
    """
    df = pd.read_csv(path_csv)

    dict_masks = {
        "land_avoidance": (df["vectorfield"] == "zero") & (df["water_level"] < 1.0),
        "vectorfields": df["vectorfield"] != "zero",
    }

    for name, mask in dict_masks.items():
        df_filtered = df[mask]

        # Define columns of interest
        cols_param = ["popsize", "sigma0", "K", "num_pieces", "L"]
        cols_target = ["percterr_fms", "comp_time"]

        # Compute correlation matrix
        corr: pd.DataFrame = df_filtered[cols_param + cols_target].corr()
        corr = corr.loc[cols_param, cols_target]

        # LaTeX table template
        latex_template = LATEX_CORRELATION

        # Replace placeholders with computed correlation values
        replacements = {
            "$POPLOSS$": f"{corr.loc['popsize', 'percterr_fms']:.3f}",
            "$POPTIME$": f"{corr.loc['popsize', 'comp_time']:.3f}",
            "$SIGMALOSS$": f"{corr.loc['sigma0', 'percterr_fms']:.3f}",
            "$SIGMATIME$": f"{corr.loc['sigma0', 'comp_time']:.3f}",
            "$KLOSS$": f"{corr.loc['K', 'percterr_fms']:.3f}",
            "$KTIME$": f"{corr.loc['K', 'comp_time']:.3f}",
            "$NPLOSS$": f"{corr.loc['num_pieces', 'percterr_fms']:.3f}",
            "$NPTIME$": f"{corr.loc['num_pieces', 'comp_time']:.3f}",
            "$LLOSS$": f"{corr.loc['L', 'percterr_fms']:.3f}",
            "$LTIME$": f"{corr.loc['L', 'comp_time']:.3f}",
            "tab:correlation": f"tab:correlation-{name}".replace("_", "-"),
        }

        for key, value in replacements.items():
            latex_template = latex_template.replace(key, value)

        # Save to a text file - keep the LaTeX format
        filename = f"{folder}/parameter_search_{name}.tex"
        with open(filename, "w") as file:
            file.write(latex_template)

        print(f"LaTeX table saved to {filename}")


def table_computation_times(
    path_csv: str = "./output/results_land.csv", folder: str = "output"
):
    """Generate a LaTeX table with the computation times for different configurations.

    Parameters
    ----------
    folder : str, optional
        The directory containing the results CSV file, by default "output".
    """
    df = pd.read_csv(path_csv)

    # Identify time dependent vector fields
    df["time_dependent"] = df["vectorfield"] == "techy"
    # Identify cost function via the input parameters
    df["cost_function"] = df["travel_stw"].isna().replace({False: "time", True: "fuel"})

    # Initialize dictionary for replacing placeholders
    replacements = {}

    for water_level in [1.0, 0.9, 0.8, 0.7]:
        for cost_function in ["time", "fuel"]:
            mask = (df["water_level"] == water_level) & (
                df["cost_function"] == cost_function
            )

            df_filtered = df[mask]
            mask_td = df_filtered["time_dependent"]

            wl = int(water_level * 10)
            cf = "T" if cost_function == "time" else "F"

            # Compute mean computation time
            df_nc = df_filtered.loc[~mask_td, "comp_time_cmaes"]
            replacements[f"${wl}{cf}NC$"] = rf"{df_nc.mean():.0f} \pm {df_nc.std():.0f}"
            df_yc = df_filtered.loc[mask_td, "comp_time_cmaes"]
            replacements[f"${wl}{cf}YC$"] = rf"{df_yc.mean():.0f} \pm {df_yc.std():.0f}"
            df_nf = df_filtered.loc[~mask_td, "comp_time_fms"]
            replacements[f"${wl}{cf}NF$"] = rf"{df_nf.mean():.0f} \pm {df_nf.std():.0f}"
            df_yf = df_filtered.loc[mask_td, "comp_time_fms"]
            replacements[f"${wl}{cf}YF$"] = rf"{df_yf.mean():.0f} \pm {df_yf.std():.0f}"

    # LaTeX table template
    latex_template = LATEX_COMPUTATION_TIMES

    # Replace placeholders with computed correlation values
    for key, value in replacements.items():
        latex_template = latex_template.replace(key, rf"${value}$")

    # Save to a text file - keep the LaTeX format
    filename = f"{folder}/computation_times.tex"
    with open(filename, "w") as file:
        file.write(latex_template)

    print(f"LaTeX table saved to {filename}")


def main(folder: str = "output"):
    """Generate LaTeX tables for parameter search correlation and computation times.

    Parameters
    ----------
    folder : str, optional
        The directory to save the LaTeX files, by default "output".
    """
    # Ensure the output folder exists
    os.makedirs(folder, exist_ok=True)

    # Generate the correlation table
    table_parameter_search_correlation(folder=folder)

    # Generate the computation times table
    table_computation_times(folder=folder)


if __name__ == "__main__":
    typer.run(main)
