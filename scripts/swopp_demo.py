import jax.numpy as jnp
import matplotlib.pyplot as plt

from routetools.benchmark import load_benchmark_instance, optimize_benchmark_instance
from routetools.cost import haversine_distance_from_curve
from routetools.fms import optimize_fms
from routetools.plot import plot_curve


def main(
    instance_name: str = "ESSDR-USNYC",
    date_start: str = "2023-01-08",
    vel_ship: int = 4,
    data_path: str = "./data",
    penalty: float = 1e6,
    K: int = 10,
    L: int = 320,
    num_pieces: int = 3,
    popsize: int = 500,
    sigma0: int = 1,
    keep_top: float = 0.002,
    tolfun_cmaes: float = 60,
    damping_cmaes: float = 1,
    maxfevals_cmaes: int = int(1e8),
    patience_fms: int = 100,
    damping_fms: float = 0.9,
    maxfevals_fms: int = int(1e6),
    seed: int = 42,
    verbose: bool = True,
):
    """Test the benchmark."""
    # Extract relevant information from the problem instance
    dict_instance = load_benchmark_instance(
        instance_name,
        date_start=date_start,
        vel_ship=vel_ship,
        data_path=data_path,
    )

    curve_cmaes, dict_cmaes = optimize_benchmark_instance(
        dict_instance,
        penalty=penalty,
        K=K,
        L=L,  # One segment per hour approx
        num_pieces=num_pieces,
        popsize=popsize,
        sigma0=sigma0,
        tolfun=tolfun_cmaes,
        damping=damping_cmaes,
        maxfevals=maxfevals_cmaes,
        init_circumnavigate=True,
        keep_top=keep_top,
        seed=seed,
        verbose=verbose,
    )
    print("CMA-ES optimization details:", dict_cmaes)

    curve_fms, dict_fms = optimize_fms(
        vectorfield=dict_instance["vectorfield"],
        curve=curve_cmaes,
        land=dict_instance["land"],
        travel_stw=dict_instance["travel_stw"],
        travel_time=dict_instance["travel_time"],
        patience=patience_fms,
        damping=damping_fms,
        maxfevals=maxfevals_fms,
        weight_l1=1.0,
        weight_l2=0.0,
        spherical_correction=True,
        seed=seed,
        verbose=verbose,
    )

    print("FMS optimization details:", dict_fms)

    # Compute distances
    dist_cmaes = jnp.sum(haversine_distance_from_curve(curve_cmaes)) / 1000
    dist_fms = jnp.sum(haversine_distance_from_curve(curve_fms[0])) / 1000

    # Plot
    vectorfield = dict_instance["vectorfield"]
    land = dict_instance["land"]
    fig, ax = plot_curve(
        vectorfield=vectorfield,
        ls_curve=[curve_cmaes, curve_fms[0]],
        ls_name=[
            f"CMA-ES ({dict_cmaes['cost'] / 3600:.0f} hrs, {dist_cmaes:.0f} km)",
            f"FMS ({sum(dict_fms['cost']) / 3600:.0f} hrs, {dist_fms:.0f} km)",
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
        f"{instance_name} | {dict_instance['date_start']} | {int(2 * vel_ship)} knots"
    )
    fig.tight_layout()
    # We use redundant naming to avoid too many images
    fig.savefig(f"output/benchmark_{instance_name}_{int(vel_ship)}.jpg", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
