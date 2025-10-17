import gc
import json
import os

import jax
import pandas as pd
import typer

from routetools.cmaes import optimize
from routetools.config import list_config_combinations
from routetools.fms import optimize_fms
from routetools.land import Land


def run_param_configuration(
    params: dict, path_jsons: str = "json", idx: int = 0, verbose: bool = False
) -> None:
    """Run the optimization algorithm with the given parameters.

    Parameters
    ----------
    params : dict
        Dictionary with the parameters
    path_jsons : str, optional
        Path to the folder where the JSON files will be saved, by default "json"
    idx : int, optional
        JSON number, by default 0
    """
    # Path to the JSON file
    path_json = f"{path_jsons}/{idx:06d}.json"
    # If the file already exists, skip
    if os.path.exists(path_json):
        return

    # Initialize the results dictionary with the parameters
    results = {**params}
    # src and dst are jnp arrays, convert them to lists
    results["src"] = params["src"].tolist()
    results["dst"] = params["dst"].tolist()

    # Load the vectorfield function
    vfname = params["vectorfield"]
    vectorfield_module = __import__(
        "routetools.vectorfield", fromlist=["vectorfield_" + vfname]
    )
    vectorfield = getattr(vectorfield_module, "vectorfield_" + vfname)

    # Load the land
    land = Land(
        params["xlim"],
        params["ylim"],
        water_level=params["water_level"],
        resolution=params.get("resolution", 1),
        random_seed=params.get("random_seed", 0),
        outbounds_is_land=params["outbounds_is_land"],
    )

    # Check if the land is valid
    if land(params["src"]) or land(params["dst"]):
        print(f"{idx}: Source or destination is on land.")
        # Save the results in a JSON file
        with open(path_json, "w") as f:
            json.dump(results, f, indent=4)
        return

    # CMA-ES optimization algorithm
    curve, dict_cmaes = optimize(
        vectorfield,
        params["src"],
        params["dst"],
        land=land,
        penalty=params["penalty"],
        travel_stw=params.get("travel_stw"),
        travel_time=params.get("travel_time"),
        K=params["K"],
        L=params["L"],
        num_pieces=params.get("num_pieces", 1),
        popsize=params["popsize"],
        sigma0=params["sigma0"],
        tolfun=params["tolfun"],
        damping=params["damping"],
        maxfevals=params["maxfevals"],
        seed=params.get("cmaes_seed", 0),
        verbose=verbose,
    )
    cost_cmaes = dict_cmaes["cost"]

    # Update the results dictionary with the optimization results
    results.update(
        {
            "cost_cmaes": cost_cmaes,
            "comp_time_cmaes": dict_cmaes["comp_time"],
            "niter_cmaes": dict_cmaes["niter"],
            "curve_cmaes": curve.tolist(),
        }
    )

    # Check if the route crosses land
    if land(curve).any() or cost_cmaes < 0:
        print(f"{idx}: CMA-ES solution crosses land.")
        # Store NaN as cost
        results["cost_cmaes"] = float("nan")
        # Save the results in a JSON file
        with open(path_json, "w") as f:
            json.dump(results, f, indent=4)
        # Stop here, no need to run FMS
        return

    # FMS variational algorithm (refinement)
    curve_fms, dict_fms = optimize_fms(
        vectorfield,
        curve=curve,
        land=land,
        travel_stw=params.get("travel_stw"),
        travel_time=params.get("travel_time"),
        tolfun=params["refiner_tolfun"],
        damping=params["refiner_damping"],
        maxfevals=params["refiner_maxfevals"],
        verbose=verbose,
    )
    # FMS returns an extra dimension, we ignore that
    curve_fms = curve_fms[0]
    cost_fms = dict_fms["cost"][0]

    if round(cost_fms, 3) > round(cost_cmaes, 3):
        # The FMS went wrong
        cost_fms = float("nan")

    # Update the results dictionary with the optimization results
    results.update(
        {
            "cost_fms": cost_fms,  # FMS returns a list of costs
            "comp_time_fms": dict_fms["comp_time"],
            "niter_fms": dict_fms["niter"],
            "curve_fms": curve_fms.tolist(),
        }
    )

    # Save the results in a JSON file
    with open(path_json, "w") as f:
        json.dump(results, f, indent=4)

    # Delete the results variable to free up memory
    results.clear()
    del results
    # Force garbage collection to free up memory
    # This is important to avoid memory leaks
    gc.collect()
    # Clear the cache to free up memory
    jax.clear_caches()


def build_dataframe(
    path_jsons: str = "json",
    path_results: str | None = None,
    experiment: str = "noland",
) -> pd.DataFrame:
    """Build a dataframe with the results.

    Parameters
    ----------
    path_jsons : str, optional
        Path to the folder where the JSON files are stored, by default "json"
    path_results : str, optional
        Path to the output folder, by default None

    Returns
    -------
    pd.DataFrame
        Dataframe with the results
    """
    # Read the results as dictionaries and store them in a list
    ls_files = os.listdir(path_jsons)
    ls_results = []
    for file in ls_files:
        with open(f"{path_jsons}/{file}") as f:
            d: dict = json.load(f)
            # Drop curves
            d.pop("curve_cmaes", None)
            d.pop("curve_fms", None)
            # Add the json file number
            d["json"] = int(file.split(".")[0])
            ls_results.append(d)

    # Build the dataframe
    df = pd.DataFrame(ls_results)

    # Any negative cost is turned into NaN
    df["cost_cmaes"] = df["cost_cmaes"].where(df["cost_cmaes"] >= 0, float("nan"))
    df["cost_fms"] = df["cost_fms"].where(df["cost_fms"] >= 0, float("nan"))

    # Any cost higher than 1e4 is considered an error and turned into NaN
    df["cost_cmaes"] = df["cost_cmaes"].where(df["cost_cmaes"] < 1e4, float("nan"))
    df["cost_fms"] = df["cost_fms"].where(df["cost_fms"] < 1e4, float("nan"))

    # Drop rows with NaN costs
    df = df.dropna(subset=["cost_cmaes", "cost_fms"], how="any")

    # Land generation check: under the same conditions, the land should be the same
    # When the land makes source or destination not reachable, the cost is NaN
    # Thus, when a cost is NaN, it should be NaN for all the rows with the same
    # land configuration
    df["is_nan"] = df["cost_cmaes"].isna()
    col_land = ["vectorfield", "water_level", "resolution", "random_seed"]
    # Drop columns not in df
    col_land = [col for col in col_land if col in df.columns]
    df_group = df.groupby(col_land)["is_nan"].mean()
    mask_wrong = ~((df_group == 0) | (df_group == 1))
    if mask_wrong.any():
        raise ValueError("Land generation is not consistent")
    else:
        # Remove the column
        df = df.drop(columns="is_nan")

    # We need to fill NaNs in resolution and random_seed with -1
    # so we can group by them
    for col in col_land:
        df[col] = df[col].fillna(-1)

    # --------------------------------------------------------
    # EXTRA COLUMNS
    # --------------------------------------------------------

    # Total computation time
    df["comp_time"] = df["comp_time_cmaes"] + df["comp_time_fms"]

    # FMS gains w.r.t. CMA-ES
    df["gain_fms"] = 100 * ((df["cost_cmaes"] - df["cost_fms"]) / df["cost_cmaes"])

    # Group by "water_level", "resolution" and "random_seed"
    # Get the lowest "cost_fms" for each group
    df_best = df.sort_values("cost_fms").groupby(col_land).first().reset_index()
    # Add that best cost to the original dataframe
    df_best = df_best.rename(columns={"cost_fms": "cost_best"})
    df = df.merge(
        df_best[col_land + ["cost_best"]],
        on=col_land,
        how="left",
    )

    # Compare CMAES cost with best (percentage error)
    df["percterr_cmaes"] = 100 * (df["cost_cmaes"] / df["cost_best"] - 1)
    df["percterr_fms"] = 100 * (df["cost_fms"] / df["cost_best"] - 1)

    # If the percentage error < 0.1% we assume the best solution was found
    df["isoptimal_cmaes"] = df["percterr_cmaes"] <= 0.1
    df["isoptimal_fms"] = df["percterr_fms"] <= 0.1

    if path_results:
        df.to_csv(
            path_results + f"/results_{experiment}.csv",
            index=False,
            float_format="%.6f",
        )
    return df


def main(
    experiment: str = "noland", path_results: str = "output", verbose: bool = False
):
    """Run the results.

    Parameters
    ----------
    experiment : str, optional
        Name of the experiment, by default "noland"
    path_results : str, optional
        Path to the output folder, by default "output"
    """
    path_config = f"config_{experiment}.toml"

    # Generate the list of parameters
    ls_params = list_config_combinations(path_config)

    # Ensure the output folder exists
    os.makedirs(path_results, exist_ok=True)
    path_jsons = path_results + "/" + experiment
    os.makedirs(path_jsons, exist_ok=True)

    # We cannot multiprocess with JAX, because JAX uses a threadpool
    for idx, params in enumerate(ls_params):
        run_param_configuration(params, path_jsons, idx, verbose=verbose)

    # Build the dataframe once at the end
    build_dataframe(path_jsons, path_results=path_results, experiment=experiment)


if __name__ == "__main__":
    typer.run(main)
