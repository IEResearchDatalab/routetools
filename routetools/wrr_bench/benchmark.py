import datetime as dt
import re
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from routetools.wrr_bench.ocean import Ocean

DICT_PORTS = {
    "CAVAN": {"lat": 48.50, "lon": -124.82, "city": "Vancouver", "country": "Canada"},
    "CNSHA": {"lat": 31.28, "lon": 121.98, "city": "Shanghai", "country": "China"},
    "DEHAM": {"lat": 53.99, "lon": 8.63, "city": "Hamburg", "country": "Germany"},
    "EGHRG": {
        "lat": 27.66,
        "lon": 33.88,
        "city": "Suez",
        "country": "Egypt",
        "canal": "Suez",
        "ocean": "Indian",
    },
    "EGPSD": {
        "lat": 31.53,
        "lon": 32.32,
        "city": "Suez",
        "country": "Egypt",
        "canal": "Suez",
        "ocean": "Mediterranean",
    },
    "ESALG": {"lat": 36.07, "lon": -5.38, "city": "Algeciras", "country": "Spain"},
    "FRLEH": {"lat": 49.49, "lon": 0.06, "city": "Le Havre", "country": "France"},
    "JPTYO": {"lat": 34.86, "lon": 139.72, "city": "Tokyo", "country": "Japan"},
    "LKCMB": {"lat": 6.93, "lon": 79.79, "city": "Colombo", "country": "Sri Lanka"},
    "PABLB": {
        "lat": 8.89,
        "lon": -79.53,
        "city": "Balboa",
        "country": "Panama",
        "canal": "Panama",
        "ocean": "Pacific",
    },
    "PAONX": {
        "lat": 9.43,
        "lon": -79.92,
        "city": "Colon",
        "country": "Panama",
        "canal": "Panama",
        "ocean": "Atlantic",
    },
    "PECLL": {
        "lat": -12.19,
        "lon": -77.23,
        "city": "Callao",
        "country": "Peru",
    },
    "USHOU": {
        "lat": 29.30,
        "lon": -94.63,
        "city": "Houston",
        "country": "United States",
    },
    "USLBH": {
        "lat": 33.73,
        "lon": -118.17,
        "city": "Long Beach",
        "country": "United States",
    },
    "USNYC": {
        "lat": 40.53,
        "lon": -73.80,
        "city": "New York",
        "country": "United States",
    },
    "USSAV": {
        "lat": 31.99,
        "lon": -80.76,
        "city": "Savannah",
        "country": "United States",
    },
    "MYKUL": {
        "lat": 2.968,
        "lon": 100.899,
        "city": "Kuala Lumpur",
        "country": "Malaysia",
    },
}
DICT_BENCHMARKS = {
    "route_days": 5,
    "benchmarks": {
        "DEHAM-USNYC": {"date_start": "2023-01-01T00:00:00"},
        "USNYC-DEHAM": {"date_start": "2023-01-01T00:00:00"},
        "EGHRG-MYKUL": {"date_start": "2023-01-01T00:00:00"},
        "MYKUL-EGHRG": {"date_start": "2023-01-01T00:00:00"},
        "EGPSD-ESALG": {"date_start": "2023-01-01T00:00:00"},
        "ESALG-EGPSD": {"date_start": "2023-01-01T00:00:00"},
        "PABLB-PECLL": {"date_start": "2023-01-01T00:00:00"},
        "PECLL-PABLB": {"date_start": "2023-01-01T00:00:00"},
        "PAONX-USNYC": {"date_start": "2023-01-01T00:00:00"},
        "USNYC-PAONX": {"date_start": "2023-01-01T00:00:00"},
    },
}


def load_files(
    list_dates: Iterable[str],
    data_path: str = "./data",
    weather_variables: Iterable[str] = ["currents", "waves"],
) -> dict[xr.Dataset]:
    """
    Load multiple files and return a xarray dataset.

    Parameters
    ----------
    list_dates : Iterable[str]
        List of dates to load in format "%Y-%m-%d".
    data_path : str, optional
        Path to the folder containing the files, by default "./data".
    weather_variables : Iterable[str], optional
        List of weather variables to load, by default ("currents").

    Returns
    -------
    dict[xr.Dataset]
        Dataset dictionary containing the data from the files.
    """
    ocean_datasets = dict()
    data_path = Path(data_path)

    # In the future we will have more than one ocean dataset
    for string_ocean in weather_variables:
        files = []
        for date in list_dates:
            file_path = data_path / f"{string_ocean}/{date}.nc"
            files.append(file_path)

        ds = xr.open_mfdataset(files, concat_dim="time", combine="nested")

        # if string_ocean == "currents":
        #    ds["land"] = ds.isnull()["vo"][0]

        # ds = ds.fillna(0)
        ocean_datasets[string_ocean] = ds

    return ocean_datasets


def load(
    name_benchmark: str,
    date_start: np.datetime64 | str | None = None,
    data_path: str = "./data",
    bounding_box: Iterable[float] | None = None,
    bounding_border: float = 5.0,
    land_resolution: float = "2km",
    vel_ship: float = None,
    use_currents: bool = True,
    use_waves: bool = True,
) -> dict:
    """
    Load benchmark configuration and prepare data and parameters to optimize.

    Parameters
    ----------
    name_benchmark : str
        Name of the benchmark used.
    date_start : np.datetime64 | str, optional
        Starting date. If not given, takes the one given by the benchmark.
        Be careful, because the benchmark doesn't have the date by default.
    data_path : str, optional
        Path to the folder containing the files, by default "./data"
    bounding_box : Optional[List[float]], optional
        Bounding box of the area to optimize, by default None
        It is a list of 4 elements: [bottom, left, up, right]
    bounding_border : float, optional
        Border to add to the bounding box around the route, by default 5.0
    land_resolution : float, optional
        Resolution of the land polygons, by default "2km"
        It can be "1km" or "2km".
    vel_ship : float, optional
        Speed of the ship in m/s, by default None
        If it is None will try to use the one given by the benchmark,
        if not it will use this value.
    use_currents : bool, optional
        Whether to use ocean currents data, by default True
    use_waves : bool, optional
        Whether to use ocean waves data, by default True

    Returns
    -------
    dict
        Dictionary containing the benchmark configuration.
    """
    dict_all_benchmarks: dict = DICT_BENCHMARKS["benchmarks"]

    assert (
        name_benchmark in dict_all_benchmarks
    ), f"Benchmark {name_benchmark} not found"

    data_path = Path(data_path)

    # TODO: Add the valid land file to the data_path
    if land_resolution == "1km":
        land_file_name = "earth-seas-1km-valid.geo.json"
    else:
        land_file_name = "earth-seas-2km5-valid.geo.json"

    route_days = DICT_BENCHMARKS["route_days"]

    benchmark_dict = {}

    # Check if the benchmark name is a port-to-port code "XXXXX-YYYYY"
    if re.match(r"^[A-Z]{5}-[A-Z]{5}$", name_benchmark):
        # If it does, take the port information
        port_start = name_benchmark[:5]
        port_end = name_benchmark[6:]
        # Reinitiliaze the dictionary with the port coordinates
        dict_add = {
            "lat_start": DICT_PORTS[port_start]["lat"],
            "lon_start": DICT_PORTS[port_start]["lon"],
            "lat_end": DICT_PORTS[port_end]["lat"],
            "lon_end": DICT_PORTS[port_end]["lon"],
        }
        benchmark_dict.update(dict_add)
        # In addition, make sure the ODP information exists in either direction
        # This will be useful if we have, for instance, defined a limit or starting date
        name_alt = f"{port_end}-{port_start}"
    else:
        name_alt = name_benchmark

    # Find if the benchmark is defined inside the dictionary, as is
    if name_benchmark in dict_all_benchmarks:
        benchmark_dict.update(dict_all_benchmarks[name_benchmark])
    # Else, find if the reverse name works
    elif name_alt in dict_all_benchmarks:
        benchmark_dict.update(dict_all_benchmarks[name_alt])

    # Initialize the dictionary containing the benchmark configuration
    # Adds default parameters to avoid missing information
    assert (
        vel_ship is not None or benchmark_dict.get("vel_ship") is not None
    ), "Velocity of the ship not found. Must be defined on benchmark or as parameter."

    if vel_ship is not None:
        benchmark_dict["vel_ship"] = vel_ship

    # Fill the date, if provided
    if date_start is None:
        # If date_start is not provided, take the one from the benchmark
        date_start = np.datetime64(benchmark_dict["date_start"])
    else:
        # If it is provided, update the dictionary
        benchmark_dict["date_start"] = pd.to_datetime(str(date_start)).strftime(
            "%Y-%m-%dT%H:%M:%S"
        )

    # Convert date_start to np.datetime64 if it is a string
    if isinstance(date_start, str):
        # If it is a string, convert to np.datetime64
        date_start = np.datetime64(date_start)

    string_date_start = date_start.astype(dt.datetime).strftime("%Y-%m-%d")

    list_string_date = [string_date_start]
    for day in range(1, route_days):
        current_date = date_start + np.timedelta64(day, "D")
        list_string_date.append(current_date.astype(dt.datetime).strftime("%Y-%m-%d"))

    # Load the ocean data choosing the variables
    weather_variables = []
    if use_currents:
        weather_variables.append("currents")
    if use_waves:
        weather_variables.append("waves")
    ocean_datasets = load_files(
        list_string_date, data_path, weather_variables=weather_variables
    )

    if bounding_box is not None:
        benchmark_dict["bounding_box"] = bounding_box

    # Slice the data chunk
    if benchmark_dict.get("bounding_box") is None:
        bb = bounding_border
        bottom = min(benchmark_dict["lat_start"], benchmark_dict["lat_end"]) - bb
        up = max(benchmark_dict["lat_start"], benchmark_dict["lat_end"]) + bb
        left = min(benchmark_dict["lon_start"], benchmark_dict["lon_end"]) - bb
        right = max(benchmark_dict["lon_start"], benchmark_dict["lon_end"]) + bb
        benchmark_dict["bounding_box"] = [bottom, left, up, right]

    benchmark_dict["data"] = Ocean(
        currents_data=ocean_datasets.get("currents", None),
        waves_data=ocean_datasets.get("waves", None),
        wind_data=ocean_datasets.get("wind", None),
        bounding_box=benchmark_dict["bounding_box"],
        land_file=data_path / land_file_name,
    )

    benchmark_dict["date_start"] = date_start

    return benchmark_dict
