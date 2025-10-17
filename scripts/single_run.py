import tomllib

import jax.numpy as jnp
import matplotlib.pyplot as plt
import typer

from routetools.cmaes import optimize
from routetools.fms import optimize_fms
from routetools.land import Land
from routetools.plot import plot_curve


def run_single_simulation(
    vectorfield: str = "techy",
    land_waterlevel: float = 1.0,
    land_resolution: int = 5,
    land_seed: int = 0,
    land_penalty: float = 100,
    outbounds_is_land: bool = False,
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
    path_img: str = "./output",
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
    fms_tolfun : float, optional
        The tolerance for the function value in FMS, by default 1e-6.
    fms_damping : float, optional
        The damping factor for FMS, by default 0.9.
    fms_maxfevals : int, optional
        The maximum number of iterations for FMS, by default 50000.
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
        return

    # CMA-ES optimization algorithm
    curve_cmaes, dict_cmaes = optimize(
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
        seed=cmaes_seed,
    )

    if land(curve_cmaes).any():
        print("The curve is on land")
        cost_cmaes = jnp.inf
    else:
        cost_cmaes = dict_cmaes["cost"]

    # FMS variational algorithm (refinement)
    curve_fms, dict_fms = optimize_fms(
        vectorfield_fun,
        curve=curve_cmaes,
        land=land,
        travel_stw=travel_stw,
        travel_time=travel_time,
        tolfun=fms_tolfun,
        damping=fms_damping,
        maxfevals=fms_maxfevals,
        verbose=True,
    )
    # FMS returns an extra dimensions, we ignore that
    curve_fms = curve_fms[0]

    if land(curve_fms).any():
        print("The curve is on land")
        cost_fms = jnp.inf
    else:
        cost_fms = dict_fms["cost"][0]  # FMS returns a list of costs

    # Plot them
    fig, ax = plot_curve(
        vectorfield_fun,
        [curve_cmaes, curve_fms],
        ls_name=["CMA-ES", "BERS"],
        ls_cost=[cost_cmaes, cost_fms],
        land=land,
        xlim=land_xlim,
        ylim=land_ylim,
    )
    ax.set_title(f"{vectorfield}")
    fig.savefig(f"{path_img}/single_run.png")
    plt.close(fig)


if __name__ == "__main__":
    typer.run(run_single_simulation)
