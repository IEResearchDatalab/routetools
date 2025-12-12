import datetime as dt
import json
import os

import matplotlib.pyplot as plt
import typer

from routetools.benchmark import load_benchmark_instance, optimize_benchmark_instance
from routetools.fms import optimize_fms
from routetools.plot import plot_curve


def single_run(
    instance_name: str,
    date_start: str = "2023-01-08",
    vel_ship: int = 6,
    data_path: str = "./data",
    penalty: float = 1e8,
    K: int = 12,
    L: int = 256,
    num_pieces: int = 1,
    popsize: int = 5000,
    sigma0: int = 1,
    tolfun_cmaes: float = 1e-4,
    damping_cmaes: float = 1,
    maxfevals_cmaes: int = int(1e5),
    patience_fms: int = 100,
    damping_fms: float = 0.9,
    maxfevals_fms: int = int(1e6),
    path_jsons: str = "output/json",
    idx: int = 0,
    seed: int = 42,
    verbose: bool = True,
):
    """Run a single benchmark instance and save the result to output/."""
    # Path to the JSON file
    path_json = f"{path_jsons}/{idx:06d}.json"
    # If the file already exists, skip
    if os.path.exists(path_json):
        return

    # Initialize the results dictionary with the parameters
    results = {
        "instance_name": instance_name,
        "date_start": date_start,
        "vel_ship": vel_ship,
        "penalty": penalty,
        "K": K,
        "L": L,
        "num_pieces": num_pieces,
        "popsize": popsize,
        "sigma0": sigma0,
        "tolfun_cmaes": tolfun_cmaes,
        "damping_cmaes": damping_cmaes,
        "maxfevals_cmaes": maxfevals_cmaes,
        "patience_fms": patience_fms,
        "damping_fms": damping_fms,
        "maxfevals_fms": maxfevals_fms,
        "seed": seed,
    }

    # Extract relevant information from the problem instance
    dict_instance = load_benchmark_instance(
        instance_name,
        date_start=date_start,
        vel_ship=vel_ship,
        data_path=data_path,
    )

    print("The problem instance contains the following information:")
    print(", ".join(list(dict_instance.keys())))

    curve_cmaes, dict_cmaes = optimize_benchmark_instance(
        dict_instance,
        penalty=penalty,
        K=K,
        L=L,
        num_pieces=num_pieces,
        popsize=popsize,
        sigma0=sigma0,
        tolfun=tolfun_cmaes,
        damping=damping_cmaes,
        maxfevals=maxfevals_cmaes,
        init_circumnavigate=False,
        seed=seed,
        verbose=verbose,
    )
    cost_cmaes = dict_cmaes["cost"]

    # Update the results dictionary with the optimization results
    results.update(
        {
            "cost_cmaes": cost_cmaes,
            "comp_time_cmaes": dict_cmaes["comp_time"],
            "niter_cmaes": dict_cmaes["niter"],
            "curve_cmaes": curve_cmaes.tolist(),
        }
    )

    # FMS
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
    # FMS adds an extra batch dimension, remove it
    curve_fms = curve_fms[0]
    cost_fms = dict_fms["cost"][0]

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

    # Plot the curve
    vectorfield = dict_instance["vectorfield"]
    land = dict_instance["land"]
    plot_curve(
        vectorfield=vectorfield,
        ls_curve=[curve_cmaes, curve_fms],
        ls_name=[f"CMA-ES (cost={cost_cmaes:.2e})", f"FMS (cost={cost_fms:.2e})"],
        land=land,
        gridstep=0.5,
        figsize=(6, 6),
        xlim=(land.xmin, land.xmax),
        ylim=(land.ymin, land.ymax),
    )
    plt.tight_layout()
    # We use redundant naming to avoid too many images
    plt.savefig(f"output/benchmark_{instance_name}.jpg", dpi=300)
    plt.close()


def main(path_jsons: str = "output/json_benchmark"):
    """Run benchmark instances and save the results to output/.

    Change the parameters in single_run() as needed.
    """
    ls_instances = [
        "DEHAM-USNYC",
        "USNYC-DEHAM",
        # "EGHRG-MYKUL",  # Suez canal is impossible
        # "MYKUL-EGHRG",  # Suez canal is impossible
        "EGPSD-ESALG",
        "ESALG-EGPSD",
        "PABLB-PECLL",
        "PECLL-PABLB",
        "PAONX-USNYC",
        "USNYC-PAONX",
    ]

    # Initialize index for JSON filenames
    idx = 0

    # Make sure the output/json directory exists
    os.makedirs(path_jsons, exist_ok=True)

    # Loop over each week of 2023
    date = dt.datetime(2023, 1, 1)
    ls_weeks = [date.strftime("%Y-%m-%d")]
    ls_velships = [3, 6, 12]
    while date.year == 2023:
        date += dt.timedelta(weeks=1)
        ls_weeks.append(date.strftime("%Y-%m-%d"))

    for date_start in ls_weeks:
        print(f"[INFO] Starting benchmarks for week starting {date_start}")
        for instance in ls_instances:
            for vel_ship in ls_velships:
                print(
                    f"[INFO] Running benchmark for instance {instance}"
                    f" and date {date_start}"
                    f" and ship velocity {vel_ship}"
                )
                try:
                    single_run(
                        instance,
                        date_start=date_start,
                        vel_ship=vel_ship,
                        path_jsons=path_jsons,
                        idx=idx,
                    )

                except IndexError as e:
                    print(
                        f"[ERROR] Benchmark for instance {instance} couldn't find "
                        f"circumnavigation: {e}"
                    )
                except FileNotFoundError as e:
                    print(
                        f"[ERROR] Benchmark for instance {instance} couldn't find "
                        f"data files: {e}"
                    )
                # Increment index
                idx += 1


if __name__ == "__main__":
    typer.run(main)
    typer.run(main)
    typer.run(main)
