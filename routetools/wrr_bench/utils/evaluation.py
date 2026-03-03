import os

from routetools.wrr_bench.benchmark import load
from routetools.wrr_bench.route import Route
from routetools.wrr_bench.utils.dataset import date_from_week


def evaluate_csv(csv_path: str, data_path: str) -> dict:
    """
    Evaluate a csv file.

    Parameters
    ----------
    csv_path : str
        Path to the csv file containing the route.
    data_path : str
        Path to the data.

    Returns
    -------
    dict
        Dictionary containing the results.
    """
    csv_file = os.path.basename(csv_path)
    assert csv_file.endswith(".csv"), "File must be a csv file"
    try:
        benchmark_name, vel, week = csv_file.replace(".csv", "").split("_")
    except ValueError:
        raise ValueError(
            "File name must be in the format: [benchmark]_[vel]_[week].csv"
        ) from None

    vel = float(vel)

    date_start = date_from_week(int(week))

    bench_dict = load(
        benchmark_name,
        date_start=date_start,
        vel_ship=vel,
        data_path=data_path,
    )

    error = None

    try:
        r = Route.from_csv_file(csv_path, vel_ship=vel, ocean_data=bench_dict["data"])
    except Exception as e:
        print(e)
        error = "Route cannot be loaded. Reason: " + str(e)

    if r.time_stamps[0] != date_start:
        error = "Route start date is not equal to week start date"
        print(error)

    if not r.feasibility_check():
        error = "Route is not feasible"
        print(error)

    return {
        "id": csv_file.replace(".csv", ""),
        "benchmark": benchmark_name,
        "vel": vel,
        "week": week,
        "time": float(r.total_time) if error is None else None,
        "distance": float(r.total_distance) if error is None else None,
        "error": error,
    }
