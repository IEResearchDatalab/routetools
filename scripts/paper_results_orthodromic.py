import datetime as dt
import json
import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
import typer

from routetools.benchmark import circumnavigate, load_benchmark_instance
from routetools.cost import cost_function
from routetools.plot import plot_curve


def single_run(
    instance_name: str,
    curve: jnp.ndarray | None = None,
    date_start: str = "2023-01-08",
    vel_ship: int = 6,
    data_path: str = "./data",
    path_jsons: str = "output/json",
    idx: int = 0,
    verbose: bool = True,
):
    """Run a single benchmark instance and save the result to output/."""
    # Path to the JSON file
    path_json = f"{path_jsons}/{idx:06d}.json"
    # If the file already exists, skip
    if os.path.exists(path_json):
        # Load existing curve if provided
        if curve is None:
            with open(path_json) as f:
                data = json.load(f)
                curve = jnp.array(data["curve"])
        return curve

    # Initialize the results dictionary with the parameters
    results = {
        "instance_name": instance_name,
        "date_start": date_start,
        "vel_ship": vel_ship,
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

    if curve is None:
        # Initialize the circumnavigation route
        curve0 = circumnavigate(
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
    else:
        curve0 = curve

    cost = cost_function(
        vectorfield=dict_instance["vectorfield"],
        curve=curve0[jnp.newaxis, :, :],
        travel_stw=vel_ship,
        travel_time=None,
        spherical_correction=True,
    )
    cost = int(cost[0])

    # Update the results dictionary with the optimization results
    results.update(
        {
            "cost": cost,
            "curve": curve0.tolist(),
        }
    )

    # Save the results in a JSON file
    with open(path_json, "w") as f:
        json.dump(results, f, indent=4)

    # Delete the results variable to free up memory
    results.clear()
    del results

    if curve is None:
        # Plot the curve
        vectorfield = dict_instance["vectorfield"]
        land = dict_instance["land"]
        plot_curve(
            vectorfield=vectorfield,
            ls_curve=[curve0],
            ls_name=[f"Orthodromic (cost={cost:.2e})"],
            land=land,
            gridstep=0.5,
            figsize=(6, 6),
            xlim=(land.xmin, land.xmax),
            ylim=(land.ymin, land.ymax),
        )
        plt.tight_layout()
        # We use redundant naming to avoid too many images
        plt.savefig(f"output/benchmark_{instance_name}_orthodromic.jpg", dpi=300)
        plt.close()

    return curve0


def main(path_jsons: str = "output/json_orthodromic"):
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

    for instance in ls_instances:
        # Reset curve for each instance
        curve = None
        # Loop through weeks and ship velocities
        for date_start in ls_weeks:
            for vel_ship in ls_velships:
                print(
                    f"[INFO] Running benchmark for instance {instance}"
                    f" and date {date_start}"
                    f" and ship velocity {vel_ship}"
                )
                try:
                    curve = single_run(
                        instance,
                        curve,
                        date_start=date_start,
                        vel_ship=vel_ship,
                        path_jsons=path_jsons,
                        idx=idx,
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
