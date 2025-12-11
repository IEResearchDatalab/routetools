import os
import time
import tomllib

import jax.numpy as jnp
import pandas as pd
import typer

from routetools.cmaes import optimize
from routetools.fms import optimize_fms
from routetools.land import Land


def run_single_simulation(
    vectorfield: str = "zero",
    land_waterlevel: float = 0.7,
    land_resolution: int = 20,
    land_seed: int = 0,
    land_penalty: float = 100,
    outbounds_is_land: bool = False,
    cmaes_K: int = 7,
    cmaes_L: int = 211,
    cmaes_numpieces: int = 1,
    cmaes_popsize: int = 500,
    cmaes_sigma: float = 1,
    cmaes_tolfun: float = 1e-6,
    cmaes_damping: float = 1.0,
    cmaes_maxfevals: int = 200000,
    cmaes_seed: int = 0,
    fms_patience: int = 50,
    fms_damping: float = 0.5,
    fms_maxfevals: int = 50000,
    weight_l1: float = 1.0,
    weight_l2: float = 0.0,
    path_config: str = "config.toml",
):
    """
    Run a single simulation to find an optimal path from source to destination.

    Parameters
    ----------
    vectorfield : str, optional
        The name of the vector field function to use, by default "zero".
    land_xlim : tuple[float, float], optional
        The x-axis limits for the land, by default None.
    land_ylim : tuple[float, float], optional
        The y-axis limits for the land, by default None.
    land_waterlevel : float, optional
        The water level for the land, by default 0.7.
    land_resolution : int, optional
        The resolution of the land grid, by default 3.
    land_seed : int, optional
        The random seed for land generation, by default 0.
    land_penalty : float, optional
        The penalty for traveling over land, by default 10.
    outbounds_is_land : bool, optional
        Whether out-of-bounds areas are considered land, by default False.
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
    fms_patience : int, optional
        The number of iterations without improvement before stopping for FMS,
        by default 50.
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

    land = Land(
        land_xlim,
        land_ylim,
        water_level=land_waterlevel,
        resolution=land_resolution,
        random_seed=land_seed,
        outbounds_is_land=outbounds_is_land,
    )

    # Is source or destination on land?
    if land(src) or land(dst):
        print("Source or destination is on land.")
        return None, None

    # CMA-ES optimization algorithm
    start = time.time()

    curve_cmaes, _ = optimize(
        vectorfield_fun,
        src,
        dst,
        land=land,
        penalty=land_penalty,
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
    )
    time_cmaes = time.time() - start

    if land(curve_cmaes).any():
        print("The curve is on land")

    # FMS variational algorithm (refinement)
    start = time.time()

    curve_fms, dict_fms = optimize_fms(
        vectorfield_fun,
        curve=curve_cmaes,
        land=land,
        travel_stw=travel_stw,
        travel_time=travel_time,
        patience=fms_patience,
        damping=fms_damping,
        maxfevals=fms_maxfevals,
        weight_l1=weight_l1,
        weight_l2=weight_l2,
        verbose=True,
    )

    time_fms = time.time() - start

    # FMS returns an extra dimensions, we ignore that
    curve_fms = curve_fms[0]

    if land(curve_fms).any():
        print("The curve is on land")

    return time_cmaes, time_fms


def main(path_output: str = "./output", path_config: str = "config.toml"):
    """Run the simulation."""
    # Make the output directory if it does not exist
    os.makedirs(path_output, exist_ok=True)

    ls_dict = []

    for water_level in [0.7, 0.8, 0.9]:
        for land_resolution in [3, 4, 5]:
            for land_seed in range(6):
                time_cmaes, time_fms = run_single_simulation(
                    land_waterlevel=water_level,
                    land_resolution=land_resolution,
                    land_seed=land_seed,
                    path_config=path_config,
                )

                ls_dict.append(
                    {
                        "water_level": water_level,
                        "land_resolution": land_resolution,
                        "land_seed": land_seed,
                        "time_cmaes": time_cmaes,
                        "time_fms": time_fms,
                    }
                )

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(ls_dict)

    # Output the following table:
    # Rows: water level
    # Columns: land resolution
    # Values: average time for CMA-ES and FMS, including +- standard deviation
    df_pivot = df.pivot_table(
        index="water_level",
        columns="land_resolution",
        values=["time_cmaes", "time_fms"],
        aggfunc=["mean"],
    )
    df_pivot.to_csv(
        os.path.join(path_output, "results-land-avoidance.csv"),
        float_format="%.2f",
    )
    print(df_pivot)


if __name__ == "__main__":
    typer.run(main)
