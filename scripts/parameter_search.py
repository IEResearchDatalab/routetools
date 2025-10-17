import gc
import json
import logging
import os
import shutil
from typing import Any

import jax
import pandas as pd
import psutil
import typer

from routetools.cmaes import optimize
from routetools.config import list_config_combinations
from routetools.fms import optimize_fms
from routetools.land import Land

# ---------------------------------------------------------------------------
# Setup logging
# ---------------------------------------------------------------------------
# This will log INFO-level messages to both console and the file "system_stats.log"
logging.basicConfig(
    filename="system_stats.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
# Also log INFO to stdout
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)


def log_system_info():
    """Log CPU usage, memory usage, and disk usage."""
    # CPU usage
    cpu_percent = psutil.cpu_percent(interval=0.1)
    cpu_freq = psutil.cpu_freq()
    # Memory usage
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    # Disk usage
    total, used, free = shutil.disk_usage("/")  # returns exactly 3 values
    percent = used / total * 100.0  # compute usage % manually

    logger.info("--- System Resources ---")
    logger.info(f"CPU Usage: {cpu_percent:.1f}% (freq: {cpu_freq.current:.0f} MHz)")
    logger.info(f"Memory Usage (RSS): {mem_info.rss / (1024**3):.2f} GB")
    logger.info(
        f"Disk Usage: {used / (1024**3):.2f} GB / {total / (1024**3):.2f} GB "
        f"({percent:.2f}% used)"
    )
    logger.info("----------------------------------------")


def run_param_configuration(
    params: dict, path_jsons: str = "json", idx: int = 0
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

    logger.info(f"{idx}: Running configuration...")

    land = Land(
        params["xlim"],
        params["ylim"],
        water_level=params["water_level"],
        resolution=params.get("resolution", 1),
        random_seed=params.get("random_seed", 0),
        outbounds_is_land=params["outbounds_is_land"],
    )

    # Is source or destination on land?
    if land(params["src"]) or land(params["dst"]):
        logger.info(f"{idx}: Source or destination is on land.")
        return
    else:
        # Load the vectorfield function
        vfname = params["vectorfield"]
        vectorfield_module = __import__(
            "routetools.vectorfield", fromlist=["vectorfield_" + vfname]
        )
        vectorfield = getattr(vectorfield_module, "vectorfield_" + vfname)

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
            weight_l1=params.get("weight_l1", 1.0),
            weight_l2=params.get("weight_l2", 0.0),
            verbose=False,
        )
        if land(curve).any():
            logger.info(f"{idx}: CMA-ES curve is on land")

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
            weight_l1=params.get("weight_l1", 1.0),
            weight_l2=params.get("weight_l2", 0.0),
            verbose=False,
        )
        # FMS returns an extra dimension, we ignore that
        curve_fms = curve_fms[0]
        if land(curve_fms).any():
            logger.info(f"{idx}: FMS curve is on land")

        # Store the results
        results = {
            **params,
            "cost_cmaes": dict_cmaes["cost"],
            "comp_time_cmaes": dict_cmaes["comp_time"],
            "niter_cmaes": dict_cmaes["niter"],
            "cost_fms": dict_fms["cost"][0],  # FMS returns a list of costs
            "comp_time_fms": dict_fms["comp_time"],
            "niter_fms": dict_fms["niter"],
            "curve_cmaes": curve.tolist(),
            "curve_fms": curve_fms.tolist(),
        }

        # src and dst are jnp arrays, convert them to lists
        results["src"] = params["src"].tolist()
        results["dst"] = params["dst"].tolist()

    # Save the results in a JSON file
    with open(path_json, "w") as f:
        json.dump(results, f, indent=4)

    logger.info(f"{idx}: Done!")
    logger.info("------------------")

    # Delete the results variable to free up memory
    results.clear()
    del results
    # Force garbage collection to free up memory
    # This is important to avoid memory leaks
    gc.collect()
    # Clear the cache to free up memory
    jax.clear_caches()

    # Output system information
    log_system_info()


def build_dataframe(
    path_jsons: str = "json", path_results: str | None = None
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

    # Land generation check: under the same conditions, the land should be the same
    # When the land makes source or destination not reachable, the cost is NaN
    # Thus, when a cost is NaN, it should be NaN for all the rows with the same
    # land configuration
    df["is_nan"] = df["cost_cmaes"].isna()
    col_land = ["vectorfield", "water_level", "resolution", "random_seed"]
    df_group = df.groupby(col_land)["is_nan"].mean()
    mask_wrong = ~((df_group == 0) | (df_group == 1))
    if mask_wrong.any():
        raise ValueError("Land generation is not consistent")
    else:
        # Remove the column
        df = df.drop(columns="is_nan")

    # We need to fill NaNs in resolution and random_seed with -1
    # so we can group by them
    df["resolution"] = df["resolution"].fillna(-1)
    df["random_seed"] = df["random_seed"].fillna(-1)

    # --------------------------------------------------------
    # EXTRA COLUMNS
    # --------------------------------------------------------

    # Total computation time
    df["comp_time"] = df["comp_time_cmaes"] + df["comp_time_fms"]

    # FMS gains w.r.t. CMA-ES
    df["gain_fms"] = 100 * ((df["cost_cmaes"] - df["cost_fms"]) / df["cost_cmaes"])

    # Group by "water_level", "resolution" and "random_seed"
    # Get the lowest "cost_fms" for each group
    df_best = (
        df.sort_values("cost_fms")
        .groupby(["vectorfield", "water_level", "resolution", "random_seed"])
        .first()
        .reset_index()
    )
    # Add that best cost to the original dataframe
    df_best = df_best.rename(columns={"cost_fms": "cost_best"})
    df = df.merge(
        df_best[
            ["vectorfield", "water_level", "resolution", "random_seed", "cost_best"]
        ],
        on=["vectorfield", "water_level", "resolution", "random_seed"],
        how="left",
    )

    # Compare CMAES cost with best (percentage error)
    df["percterr_cmaes"] = 100 * (df["cost_cmaes"] / df["cost_best"] - 1)
    df["percterr_fms"] = 100 * (df["cost_fms"] / df["cost_best"] - 1)

    # If the percentage error < 0.1% we assume the best solution was found
    df["isoptimal_cmaes"] = df["percterr_cmaes"] <= 0.1
    df["isoptimal_fms"] = df["percterr_fms"] <= 0.1

    if path_results:
        df.to_csv(path_results + "/results.csv", index=False, float_format="%.6f")
    return df


def main(
    path_config: str = "config.toml",
    path_results: str = "output",
    batch_start: int = 0,
    batch_end: int = -1,
):
    """Run the results.

    Parameters
    ----------
    path_config : str, optional
        Path to the configuration file, by default "config.toml"
    path_results : str, optional
        Path to the output folder, by default "output"
    """
    # Generate the list of parameters
    ls_params = list_config_combinations(path_config)

    # If batch_end is -1, interpret it as "run until the end"
    if batch_end == -1:
        batch_end = len(ls_params)

    # Slice the parameters
    subset_params = ls_params[batch_start:batch_end]

    # Ensure the output folder exists
    os.makedirs(path_results, exist_ok=True)
    path_jsons = path_results + "/json"
    os.makedirs(path_jsons, exist_ok=True)

    # Wrap the function in a try-except block to catch any error
    def run_param_configuration_try(params: dict[str, Any], idx: int):
        try:
            run_param_configuration(params, path_jsons, idx)
            # Throttle the optimization loop
            # time.sleep(0.1)
        except Exception as e:
            logger.error(f"{idx}: Error! {e}")
            logger.error("------------------")

    # We cannot multiprocess with JAX, because JAX uses a threadpool
    for idx, params in enumerate(subset_params, start=batch_start):
        run_param_configuration_try(params, idx)

    # Build the dataframe once at the end
    build_dataframe(path_jsons, path_results=path_results)


if __name__ == "__main__":
    typer.run(main)
