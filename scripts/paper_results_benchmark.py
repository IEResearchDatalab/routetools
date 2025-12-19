import datetime as dt
import json
import os

import jax.numpy as jnp
import typer

from routetools.benchmark import (
    circumnavigate,
    load_benchmark_instance,
    optimize_benchmark_instance,
)
from routetools.cost import cost_function, haversine_distance_from_curve
from routetools.fms import optimize_fms

YEAR = 2023
WEEKS = 52
LS_VELOCITIES = [3, 6, 12]
LS_INSTANCES = [
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


def single_run(
    instance_name: str,
    date_start: str = "2023-01-08",
    vel_ship: int = 6,
    data_path: str = "./data",
    penalty: float = 1e6,
    K: int = 10,
    L: int = 256,
    num_pieces: int = 3,
    popsize: int = 5000,
    sigma0: int = 2,
    keep_top: float = 0.02,
    tolfun_cmaes: float = 60,
    damping_cmaes: float = 1,
    maxfevals_cmaes: int = int(1e8),
    patience_fms: int = 100,
    damping_fms: float = 0.9,
    maxfevals_fms: int = int(1e6),
    path_jsons: str = "output/json_benchmark",
    path_jsons_circ: str = "output/json_circumnavigation",
    seed: int = 42,
    overwrite: bool = False,
    verbose: bool = True,
):
    """Run a single benchmark instance and save the result to output/."""
    # Path to the JSON file
    # Create a unique name based on the parameters
    # Remove "-" from instance name for filename compatibility
    name = instance_name.replace("-", "")
    # Turn date into "YYMMDD"
    date_str = dt.datetime.strptime(date_start, "%Y-%m-%d").strftime("%y%m%d")
    # Ensure vel_ship is string and integer
    vel_ship = int(vel_ship)
    unique_name = f"{name}_{date_str}_{vel_ship}"
    path_json = f"{path_jsons}/{unique_name}.json"

    # If the file already exists, skip
    if os.path.exists(path_json) and not overwrite:
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

    # ----------------------------------------------------------------------
    # Initialize the circumnavigation route
    # ----------------------------------------------------------------------

    # Find if the circumnavigation curve already exists
    # The name is the instance name, sorted alphabetically to avoid duplicates
    port1, port2 = instance_name.split("-")
    name_circ = "".join(sorted([port1, port2]))
    path_json_circ = f"{path_jsons_circ}/{name_circ}.json"
    if os.path.exists(path_json_circ):
        with open(path_json_circ) as f:
            data = json.load(f)
            curve = data["curve"]
            curve = [[pt[0], pt[1]] for pt in curve]  # (lon, lat)
            # Convert to jax array
            curve_circ = jnp.array(curve)
            # Check the start and end points match, else reverse
            if not (
                jnp.isclose(curve_circ[0, 1], dict_instance["lat_start"])
                and jnp.isclose(curve_circ[0, 0], dict_instance["lon_start"])
                and jnp.isclose(curve_circ[-1, 1], dict_instance["lat_end"])
                and jnp.isclose(curve_circ[-1, 0], dict_instance["lon_end"])
            ):
                curve_circ = curve_circ[::-1, :]
    else:
        # Compute the circumnavigation curve
        curve_circ = circumnavigate(
            lat_start=dict_instance["lat_start"],
            lon_start=dict_instance["lon_start"],
            lat_end=dict_instance["lat_end"],
            lon_end=dict_instance["lon_end"],
            ocean=dict_instance["data"],
            land=dict_instance["land"],
            date_start=dict_instance["date_start"],
            date_end=dict_instance.get("date_end"),
            vel_ship=dict_instance.get("vel_ship"),
            grid_resolution=4,
            neighbour_disk_size=3,
            land_dilation=1,
            fms_damping=0.0,
            verbose=verbose,
        )
        # Save the circumnavigation curve
        os.makedirs(path_jsons_circ, exist_ok=True)
        with open(path_json_circ, "w") as f:
            json.dump(
                {
                    "instance_name": instance_name,
                    "curve": curve_circ.tolist(),
                },
                f,
                indent=4,
            )

    cost_circ = cost_function(
        vectorfield=dict_instance["vectorfield"],
        curve=curve_circ[jnp.newaxis, :, :],
        wavefield=dict_instance["wavefield"],
        travel_stw=vel_ship,
        travel_time=None,
        spherical_correction=True,
    )
    cost_circ = int(cost_circ[0])

    # Compute distance too
    dist_circ_km = jnp.sum(haversine_distance_from_curve(curve_circ)) / 1000

    # Update the results dictionary with the optimization results
    results.update(
        {
            "cost_circ": cost_circ,
            "distance_circ": float(dist_circ_km),
            "curve_circ": curve_circ.tolist(),
        }
    )

    # ----------------------------------------------------------------------
    # Optimize with CMA-ES
    # ----------------------------------------------------------------------

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
        curve0=curve_circ,
        init_circumnavigate=False,
        keep_top=keep_top,
        seed=seed,
        verbose=verbose,
    )
    cost_cmaes = dict_cmaes["cost"]

    # Compute distance too
    dist_cmaes_km = jnp.sum(haversine_distance_from_curve(curve_cmaes)) / 1000

    # Update the results dictionary with the optimization results
    results.update(
        {
            "cost_cmaes": cost_cmaes,
            "distance_cmaes": float(dist_cmaes_km),
            "comp_time_cmaes": dict_cmaes["comp_time"],
            "niter_cmaes": dict_cmaes["niter"],
            "curve_cmaes": curve_cmaes.tolist(),
        }
    )

    # ----------------------------------------------------------------------
    # FMS
    # ----------------------------------------------------------------------

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

    # Compute distance too
    dist_fms_km = jnp.sum(haversine_distance_from_curve(curve_fms)) / 1000

    # Update the results dictionary with the optimization results
    results.update(
        {
            "cost_fms": cost_fms,  # FMS returns a list of costs
            "distance_fms": float(dist_fms_km),
            "comp_time_fms": dict_fms["comp_time"],
            "niter_fms": dict_fms["niter"],
            "curve_fms": curve_fms.tolist(),
        }
    )

    # Move the "curve_" keys to the bottom for better readability
    dict_curves = {k: v for k, v in results.items() if k.startswith("curve_")}
    keys_curves = list(dict_curves.keys())
    for k in keys_curves:
        results.pop(k)
    results.update(dict_curves)

    # Save the results in a JSON file
    with open(path_json, "w") as f:
        json.dump(results, f, indent=4)

    # Delete the results variable to free up memory
    results.clear()
    del results


def main(path_jsons: str = "output/json_benchmark"):
    """Run benchmark instances and save the results to output/.

    Change the parameters in single_run() as needed.
    """
    # Make sure the output/json directory exists
    os.makedirs(path_jsons, exist_ok=True)

    # Loop over each week of target year
    ls_weeks = []
    for week in range(WEEKS):
        date = dt.datetime(YEAR, 1, 1) + dt.timedelta(weeks=week)
        # If the date is above the year, stop
        if date.year > YEAR:
            break
        ls_weeks.append(date.strftime("%Y-%m-%d"))

    for instance in LS_INSTANCES:
        for date_start in ls_weeks:
            for vel_ship in LS_VELOCITIES:
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
                    )

                except IndexError as e:
                    print(f"[ERROR] Benchmark for instance {instance} failed: {e}")
                except FileNotFoundError as e:
                    print(f"[ERROR] Benchmark for instance {instance} failed: {e}")


if __name__ == "__main__":
    typer.run(main)
