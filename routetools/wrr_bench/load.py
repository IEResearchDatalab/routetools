import datetime as dt
import re
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from routetools._ports import DICT_INSTANCES, DICT_PORTS
from routetools.wrr_bench.ocean import Ocean


def load_files(
    list_dates: Iterable[str],
    data_path: str = "./data",
    weather_variables: Iterable[str] = ["currents", "waves"],
) -> dict[str, xr.Dataset]:
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
    data_Path = Path(data_path)

    # In the future we will have more than one ocean dataset
    for string_ocean in weather_variables:
        files = []
        for date in list_dates:
            file_path = data_Path / f"{string_ocean}/{date}.nc"
            files.append(file_path)

        ds = xr.open_mfdataset(files, concat_dim="time", combine="nested")

        ocean_datasets[string_ocean] = ds

    return ocean_datasets


def load_real_instance(
    name_instance: str,
    date_start: np.datetime64 | str | None = None,
    data_path: str = "./data",
    bounding_box: Iterable[float] | None = None,
    bounding_border: float = 5.0,
    land_resolution: str = "2km",
    vel_ship: float | None = None,
    use_currents: bool = True,
    use_waves: bool = True,
    route_days: int = 10,
) -> dict:
    """
    Load instance configuration and prepare data and parameters to optimize.

    Parameters
    ----------
    name_instance : str
        Name of the instance used.
    date_start : np.datetime64 | str, optional
        Starting date. If not given, takes the one given by the instance.
        Be careful, because the instance doesn't have the date by default.
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
        If it is None will try to use the one given by the instance,
        if not it will use this value.
    use_currents : bool, optional
        Whether to use ocean currents data, by default True
    use_waves : bool, optional
        Whether to use ocean waves data, by default True
    route_days : int, optional
        Number of days worth of data. If the route goes for longer,
        it will repeat the last day, by default 10

    Returns
    -------
    dict
        Dictionary containing the instance configuration.
    """
    # TODO: Add the valid land file to the data_path
    if land_resolution == "1km":
        land_file_name = "earth-seas-1km-valid.geo.json"
    else:
        land_file_name = "earth-seas-2km5-valid.geo.json"

    dict_instance = {}

    # Check if the instance name is a port-to-port code "XXXXX-YYYYY"
    if re.match(r"^[A-Z]{5}-[A-Z]{5}$", name_instance):
        # If it does, take the port information
        port_start = name_instance[:5]
        port_end = name_instance[6:]
        # Reinitiliaze the dictionary with the port coordinates
        dict_add = {
            "lat_start": DICT_PORTS[port_start]["lat"],
            "lon_start": DICT_PORTS[port_start]["lon"],
            "lat_end": DICT_PORTS[port_end]["lat"],
            "lon_end": DICT_PORTS[port_end]["lon"],
        }
        dict_instance.update(dict_add)
        # In addition, make sure the ODP information exists in either direction
        # This will be useful if we have, for instance, defined a limit or starting date
        name_alt = f"{port_end}-{port_start}"
    else:
        name_alt = name_instance

    # Find if the instance is defined inside the dictionary, as is
    if name_instance in DICT_INSTANCES:
        dict_instance.update(DICT_INSTANCES[name_instance])
    # Else, find if the reverse name works
    elif name_alt in DICT_INSTANCES:
        dict_instance.update(DICT_INSTANCES[name_alt])

    # If neither the direct name nor the alternate (reversed) name exist,
    # and the provided name is not a port-to-port code, it's an error.
    if not (
        name_instance in DICT_INSTANCES
        or name_alt in DICT_INSTANCES
        or re.match(r"^[A-Z]{5}-[A-Z]{5}$", name_instance)
    ):
        raise KeyError(f"Instance {name_instance} not found")

    # Initialize the dictionary containing the instance configuration
    # Adds default parameters to avoid missing information
    assert (
        vel_ship is not None or dict_instance.get("vel_ship") is not None
    ), "Velocity of the ship not found. Must be defined on instance or as parameter."

    if vel_ship is not None:
        dict_instance["vel_ship"] = vel_ship

    # Fill the date, if provided
    if date_start is None:
        # If date_start is not provided, take the one from the instance
        date_start = np.datetime64(dict_instance["date_start"])
    else:
        # If it is provided, update the dictionary
        dict_instance["date_start"] = pd.to_datetime(str(date_start)).strftime(
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
        dict_instance["bounding_box"] = bounding_box

    # Slice the data chunk
    if dict_instance.get("bounding_box") is None:
        bb = bounding_border
        bottom = min(dict_instance["lat_start"], dict_instance["lat_end"]) - bb
        up = max(dict_instance["lat_start"], dict_instance["lat_end"]) + bb
        left = min(dict_instance["lon_start"], dict_instance["lon_end"]) - bb
        right = max(dict_instance["lon_start"], dict_instance["lon_end"]) + bb
        dict_instance["bounding_box"] = [bottom, left, up, right]

    dict_instance["data"] = Ocean(
        currents_data=ocean_datasets.get("currents", None),
        waves_data=ocean_datasets.get("waves", None),
        wind_data=ocean_datasets.get("wind", None),
        bounding_box=dict_instance["bounding_box"],
        land_file=data_path + "/" + land_file_name,
    )

    dict_instance["date_start"] = date_start

    return dict_instance
