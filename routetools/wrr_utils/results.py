import contextlib
import itertools
import json
import os
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from routetools.wrr_utils import BaseOptimizer
from routetools.wrr_utils.benchmark import load_from_config
from routetools.wrr_utils.consumption import FuelConsumption, Joessel, Towage
from routetools.wrr_utils.route import Route
from routetools.wrr_utils.utils.aggregate_data import aggregate_data
from routetools.wrr_utils.utils.config import load_config


def create_folder(root: Path, name: str) -> Path:
    """Create a folder named ``name`` under ``root`` and return its Path.

    This is safe for parallel runs: concurrent creation collisions are
    suppressed.
    """
    folder = root / name
    if not folder.exists():
        with contextlib.suppress(FileExistsError):
            os.mkdir(folder)
    return folder


def pool_process(func: callable, ls_params: list, njobs: int):
    """Run ``func`` over ``ls_params`` using a process pool when ``njobs`` > 1.

    ``ls_params`` is an iterable of argument tuples fed to ``func``.
    """
    if njobs == 1:
        for params in ls_params:
            func(*params)
        return
    with Pool(processes=njobs, maxtasksperchild=1) as p:
        p.starmap(func, ls_params)


def optimize_benchmark(
    benchmark: str,
    ship_params: dict,
    vel: float,
    week: int,
    opt_params: dict,
    ls_refiners: list[dict],
    config: dict,
    replace: bool,
):
    """Optimize a single benchmark configuration and store results.

    Performs optimization, optional reparametrization and writes result
    files and summaries under the configured results path.
    """
    # Load information from the config file
    path_results: str = Path(config.get("path_results", "."))
    path_base_route: str = config.get("path_base_route")
    dict_reparam = config.get("reparametrization")

    # Make a copy to not modify the originals
    ship_params = ship_params.copy()
    opt_params = opt_params.copy()
    ship: str = ship_params.pop("name")
    opt: str = opt_params.pop("name")

    # Create the folder structure
    folder0 = create_folder(path_results, benchmark)
    folder1 = create_folder(folder0, ship)
    folder2 = create_folder(folder1, f"vel_{vel:02d}")
    folder3 = create_folder(folder2, f"week_{week + 1:02d}")
    file = folder3 / f"{opt}.csv"

    # If the file already exists, skip it
    if file.exists() and not replace:
        print(f"\nFile {file} already exists. Skipping...")
        return
    else:
        print(
            f"\nOptimizing for {benchmark} | {ship} | V: {vel} | W: {week + 1} | {opt}"
        )

    # Define the consumption model
    ship_type = ship_params.pop("type")
    if ship_type == "single":
        res_obj = Joessel(**ship_params)
    elif ship_type == "towage":
        resistances = []
        for ship in ship_params["ships"].values():
            resistances.append(Joessel(**ship))

        res_obj = Towage(resistances)
    # Build the fuel consumption class
    consumption = FuelConsumption(
        res_obj, engine_efficiency=ship_params.get("engine_efficiency", 0.6)
    )

    # Load the benchmark
    dict_benchmark = load_from_config(config, benchmark, week=week)
    date_start = dict_benchmark["date_start"]

    if "vel_ship" in dict_benchmark:
        dict_benchmark.pop("vel_ship")

    # Initialize dictionary with extra info for the output route
    dict_extra = {}

    # Initialize the optimizer
    comp = opt_params.get("optimizer", "Bezier")
    opt_params.pop("optimizer")
    mod = getattr(__import__("wrr_utils").optimization, comp)
    optimizer: BaseOptimizer = mod(**opt_params)

    if opt == "min_distance":
        # Optimize the route of minimum distance only if it is not there already
        file_min_dist = folder0 / "ref_min_dist.csv"
        if replace or not file_min_dist.exists():
            # Check if the route is precomputed
            if path_base_route is not None:
                df = pd.read_csv(path_base_route)
                df.to_csv(file_min_dist, index=False)
                dict_extra["comp_time"] = df["comp_time"].values[0]
            else:
                start = time.time()
                # Generate the route (of minimum distance) for that benchmark
                route: Route = optimizer.optimize(**dict_benchmark, vel_ship=vel)
                dict_extra["comp_time"] = time.time() - start

                route.export_to_csv(file_min_dist, all_data=True, extra=dict_extra)

        # Load the route of minimum distance and adapt to the current conditions
        route = Route.from_csv_file(
            path_to_file=file_min_dist,
            ocean_data=dict_benchmark["data"],
            vel_ship=vel,
        )

        route.recompute_given_time_start(date_start)

        fuel = consumption.required_fuel_from_route(route, 1e10)
        dict_extra["fuel"] = np.concatenate([[0], fuel.flatten()])
    else:
        start = time.time()
        # Generate the route for that benchmark
        route: Route = optimizer.optimize(
            **dict_benchmark,
            vel_ship=vel,
            consumption=consumption,
        )
        dict_extra["comp_time"] = time.time() - start

    # Some optimizers will output None when they do not find a route
    # To avoid errors, we do not store the route in that case
    if route is None:
        warnings.warn(
            (
                f"Route not found for {benchmark} | {ship} | V: {vel} | "
                f"W: {week + 1} | {opt}"
            ),
            stacklevel=2,
        )
        # If the route is None, skip the reparametrization and refining step
        dict_reparam = None
        ls_refiners = []
    else:
        route.export_to_csv(file, all_data=True, extra=dict_extra)

    dict_opt_params = {
        "benchmark": benchmark,
        "ship": ship,
        "velocity": vel,
        "week": week + 1,
        "optimizer": opt,
    }

    dict_opt_params["optimizer"] = optimizer.last_optimization_summary()
    dict_opt_params["optimizer"]["optimizer"] = comp

    # Reparametrize the route if needed
    if dict_reparam is not None:
        rep_type = dict_reparam.get("type", "none")
        # Find the type of reparametrization
        if rep_type == "cost":
            if "space_step" in dict_reparam:
                time_step = dict_reparam["space_step"] / route.vel_ship
            else:
                time_step = dict_reparam.get("time_step", 3600)
            n_iter = dict_reparam.get("n_iter", 10)
            route = route.reparametrize_to_fixed_cost(time_step, n_iter=n_iter)
        elif rep_type == "points":
            n_points = dict_reparam.get("n_points", 1000)
            n_iter = dict_reparam.get("n_iter", 10)
            route = route.reparametrize_to_num_waypoints(n_points, n_iter=n_iter)
        elif rep_type == "distance":
            max_dist = dict_reparam.get("max_dist", 5000)
            route = route.reparametrize_to_maximum_segment_length(max_dist)
        else:
            warnings.warn(
                f"Reparametrization type {rep_type} not recognized",
                stacklevel=2,
            )
        # Recompute fuel
        fuel = consumption.required_fuel_from_route(route, 1e10)
        dict_extra["fuel"] = np.concatenate([[0], fuel.flatten()])
        # Store the reparametrized route
        file_rep = file if opt == "min_distance" else folder3 / f"{opt}_reparam.csv"
        route.export_to_csv(file_rep, all_data=True, extra=dict_extra)

    # Minimum distance route does not need to be refined
    if opt == "min_distance":
        ls_refiners = []

    refiners_sumaries = []
    # Refine the route
    for ref_params in ls_refiners:
        # Make a copy to not modify the originals
        ref_params = ref_params.copy()
        ref_name: str = ref_params.pop("name")
        print(f"Refining with {ref_name}")

        # Initialize the optimizer
        comp = ref_params.get("optimizer", "DNJ")
        ref_params.pop("optimizer")
        mod = getattr(__import__("wrr_utils").optimization, comp)
        optimizer: BaseOptimizer = mod(**ref_params)

        # Refine the route
        ts = time.time()
        route_ref: Route = optimizer.optimize(route)
        extra = {"comp_time": int(time.time() - ts)}
        file_ref = folder3 / f"{opt}_{ref_name}.csv"
        route_ref.export_to_csv(file_ref, all_data=True, extra=extra)

        dict_ref_summary = optimizer.last_optimization_summary()
        dict_ref_summary["refiner"] = comp
        refiners_sumaries.append(dict_ref_summary)

    # Store the summary of the optimization
    dict_opt_params["refiner"] = refiners_sumaries
    with open(folder3 / f"{opt}_summary.json", "w") as f:
        json.dump(dict_opt_params, f, indent=4)

    aggregate_data(path_results)


def optimize_benchmark_ignore_errors(*args):
    """Run ``optimize_benchmark`` and print exceptions instead of raising.

    Useful when parallelizing many runs where a single failure should not
    abort the whole batch.
    """
    try:
        optimize_benchmark(*args)
    except Exception as e:
        print(repr(e))


def optimize_routes_and_store(
    config: dict, replace: bool = False, ignore_errors: bool = False
) -> int:
    """Optimize routes and compute all statistics used in the demo.

    Parameters
    ----------
    config : dict
        Configuration dictionary
    replace : bool, optional
        Replace existing results, by default False
    ignore_errors : bool, optional
        If True, ignore errors, by default False

    Returns
    -------
    int
        The number of simulations that have been done
    """
    list_benchmarks = config["benchmark"]
    list_ship: list[dict] = config["ship"]
    list_velocities = config["velocity"]
    list_weeks = list(range(int(config.get("weeks", 1))))
    list_optimizers: list[dict] = config["optimizers"]
    list_refiners: list[dict] = config.get("refiners", [])

    njobs: int = config["njobs"]

    # Initialize the output folder
    root = Path(config["path_results"])
    if not root.exists():
        os.mkdir(root)

    # Choose the function to use
    fun = optimize_benchmark_ignore_errors if ignore_errors else optimize_benchmark

    # Loop through all the combinations of ship, velocity and optimizer
    list_params = list(
        itertools.product(
            list_benchmarks,
            list_ship,
            list_velocities,
            list_weeks,
            list_optimizers,
            [list_refiners],
            [config],
            [replace],
        )
    )
    pool_process(fun, list_params, njobs)


def main(
    path_config: str = "config/results/astar.json",
    replace: bool = False,
    ignore_errors: bool = False,
):
    """Run the benchmark suite described in `path_config` and aggregate results.

    Parameters
    ----------
    path_config : str
        Path to the results configuration file.
    replace : bool, optional
        Replace existing results, by default False.
    ignore_errors : bool, optional
        If True, continue on individual run errors, by default False.
    """
    config = load_config(path_config)
    optimize_routes_and_store(config, replace=replace, ignore_errors=ignore_errors)
    aggregate_data(config["path_results"])
    print("\n\nDone!")


if __name__ == "__main__":
    typer.run(main)
