import datetime as dt
import json
import os
import re
from collections.abc import Iterable

import numpy as np
import pandas as pd
import s3fs
import xarray as xr
from tqdm import tqdm

from routetools.wrr_bench.ocean import Ocean, get_data_chunk


def list_benchmarks_names(
    path_config: str = "config/benchmarks.json", include_reverse: bool = True
) -> list[str]:
    """Return the list of benchmark names from a config file.

    Parameters
    ----------
    path_config : str, optional
        Benchmarks configuration file, by default "config/benchmarks.json".
    include_reverse : bool, optional
        If True include reversed benchmark names (e.g. "A-B" -> "B-A").

    Returns
    -------
    list[str]
        All benchmark names found in the config file.
    """
    with open(path_config) as f:
        benchmarks = json.load(f)

    names = list(benchmarks["benchmarks"].keys())

    if include_reverse:
        names += [("-").join(b.split("-")[::-1]) for b in names]

    return names


def load_files(
    list_dates: Iterable[str],
    local_data_path: str | None = None,
    path_s3_bucket: str | None = None,
    weather_variables: Iterable[str] = ("uvDepth0", "Waves", "Wind10m"),
) -> dict[xr.Dataset]:
    """Load datasets for the requested dates and weather variables.

    Behavior depends on the combination of ``local_data_path`` and
    ``path_s3_bucket``:

    - If both are provided, local files are used when present and missing
      files are downloaded from S3.
    - If only ``local_data_path`` is provided, missing files cause an error.
    - If only ``path_s3_bucket`` is provided, files are read from S3 and not
      saved locally (useful for serverless processing).

    Parameters
    ----------
    list_dates : Iterable[str]
        Dates to load in "%Y-%m-%d" format.
    local_data_path : str, optional
        Folder to store or read local files, by default None.
    path_s3_bucket : str, optional
        S3 bucket base path containing the files, by default None.
    weather_variables : Iterable[str], optional
        Variables to load, by default ("uvDepth0", "Waves", "Wind10m").

    Returns
    -------
    dict[xr.Dataset]
        Mapping from variable name to the loaded xarray Dataset.
    """
    assert (
        local_data_path is not None or path_s3_bucket is not None
    ), "local_data_path and path_s3_bucket cannot be None at the same time."

    ocean_datasets = dict()

    for string_ocean in weather_variables:
        if local_data_path is not None:
            if not os.path.exists(local_data_path):
                os.makedirs(local_data_path)

            folder_ocean = local_data_path + "/" + string_ocean

            # Check if data exists in local_data_path folder
            if not os.path.exists(folder_ocean):
                os.makedirs(folder_ocean)
        else:
            folder_ocean = "tmp"

        files = []
        files_to_download = []
        fs_s3 = None
        for date in list_dates:
            file_name = f"{date}.nc"
            file_path = f"{folder_ocean}/{file_name}"

            files.append(file_path)

            # Check which files need to be downloaded
            # If local_data_path is None, all files need to be downloaded
            if local_data_path is None or not os.path.exists(file_path):
                files_to_download.append((file_name, file_path))

        if len(files_to_download) > 0 and path_s3_bucket is None:
            raise ValueError(
                "Some files are missing in the local_data_path and "
                "path_s3_bucket is None."
            )

        # This code is split in two loops to ensure legibility using tqdm
        if len(files_to_download) > 0:
            fs_s3 = s3fs.S3FileSystem()

            all_ds = []

            for file_name, file_path in tqdm(
                files_to_download, desc=f"Downloading {string_ocean} files"
            ):
                # Download file from S3 bucket
                aws_url = f"{path_s3_bucket}/{string_ocean}/{file_name}"

                s3_file_obj = fs_s3.open(aws_url, mode="rb")
                ds = xr.open_dataset(s3_file_obj)

                if local_data_path is not None:
                    ds.to_netcdf(file_path)
                else:
                    all_ds.append(ds)

        if local_data_path is not None:
            ds = xr.open_mfdataset(files, concat_dim="time", combine="nested")
        else:
            ds = xr.concat(all_ds, dim="time")

        if string_ocean == "uvDepth0":
            ds["land"] = ds.isnull()["vo"][0]
        elif string_ocean == "Waves":
            ds["land"] = ds.isnull()["height"][0]

        # This can be also fixed with the `utils.correct_ds_coordinates` function
        if "lat" in ds.coords:
            ds = ds.rename({"lat": "latitude"})

        if "lon" in ds.coords:
            ds = ds.rename({"lon": "longitude"})

        # ds = ds.fillna(0)
        ocean_datasets[string_ocean] = ds

    return ocean_datasets


def load(
    name_benchmark: str,
    date_start: np.datetime64 | str | None = None,
    route_days: int | None = None,
    config_file: str = "config/benchmarks.json",
    local_data_path: str = "./data",
    bounding_box: Iterable[float] | None = None,
    land_file: str = "static_data/geojson/earth-seas-2km5-valid.geo.json",
    use_currents: bool = True,
    use_waves: bool = True,
    use_wind: bool = True,
    interp_method: str = "EvenLinearInterpolator",
    add_raw_data: bool = False,
    **kwargs,
) -> dict:
    """
    Load benchmark configuration and prepare data and parameters to optimize.

    Parameters
    ----------
    name_benchmark : str
        Name of the benchmark used.
    date_start : Union[np.datetime64, str], optional
        Starting date. If not given, takes the one given by the benchmark
    route_days : int, optional
        Number of days to load the data. If not given, takes the one given
        by the benchmark
    config_file : str, optional
        Path to JSON config file, by default "config/benchmarks.json"
    local_data_path : str, optional
        Path to the folder containing the files, by default "./data"
    bounding_box : Optional[List[float]], optional
        Bounding box of the area to optimize, by default None
        It is a list of 4 elements: [bottom, left, up, right]
    land_file : str, optional
        Path to the geojson file containing the land, by default
        "static_data/geojson/earth-seas-2km5-valid.geo.json"
    interp_method : str, optional
        Method used to interpolate the data, by default "EvenLinearInterpolator"
    add_raw_data : bool, optional
        If True, adds the raw data to the dictionary, by default False

    Returns
    -------
    dict
        Dictionary containing the benchmark configuration.
    """
    with open(config_file) as f:
        config_dict: dict = json.load(f)

    s3_bucket_path = config_dict["s3_bucket_path"]

    dict_all_benchmarks: dict = config_dict["benchmarks"]
    dict_ports: dict = config_dict.get("ports", {})

    # Initialize the dictionary containing the benchmark configuration
    # Adds default parameters to avoid missing information
    benchmark_dict = {"vel_ship": 10}

    # Check if the benchmark name is a port-to-port code "XXXXX-YYYYY"
    if re.match(r"^[A-Z]{5}-[A-Z]{5}$", name_benchmark):
        # If it does, take the port information
        port_start = name_benchmark[:5]
        port_end = name_benchmark[6:]
        # Reinitiliaze the dictionary with the port coordinates
        dict_add = {
            "lat_start": dict_ports[port_start]["lat"],
            "lon_start": dict_ports[port_start]["lon"],
            "lat_end": dict_ports[port_end]["lat"],
            "lon_end": dict_ports[port_end]["lon"],
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

    # Fill the date, if provided
    if date_start is None:
        # If date_start is not provided, take the one from the benchmark
        date_start = np.datetime64(benchmark_dict["date_start"])
    elif isinstance(date_start, np.datetime64):
        # If it is provided, update the dictionary
        date_str = pd.to_datetime(str(date_start)).strftime("%Y-%m-%dT%H:%M:%S")
        benchmark_dict["date_start"] = date_str
    elif isinstance(date_start, str):
        # If it is provided as a string, update the dictionary
        benchmark_dict["date_start"] = date_start
        date_start = np.datetime64(date_start)

    # Fill the route days, if provided
    if route_days is None:
        route_days = config_dict["route_days"]
    elif route_days < 1:
        raise ValueError("route_days must be greater than 0.")
    else:
        route_days = int(route_days)

    list_string_date = [date_start.astype(dt.datetime).strftime("%Y-%m-%d")]
    for day in range(1, route_days):
        current_date = date_start + np.timedelta64(day, "D")
        list_string_date.append(current_date.astype(dt.datetime).strftime("%Y-%m-%d"))

    # Sets default ocean variables loading with Waves
    benchmark_dict.update(
        {"use_currents": use_currents, "use_waves": use_waves, "use_wind": use_wind}
    )
    weather_variables = []
    if use_currents is True:
        weather_variables.append("uvDepth0")

    if use_waves is True:
        weather_variables.append("Waves")

    if use_wind is True:
        weather_variables.append("Wind10m")

    ocean_datasets = load_files(
        list_string_date,
        local_data_path,
        s3_bucket_path,
        weather_variables=weather_variables,
    )

    if bounding_box is not None:
        benchmark_dict["bounding_box"] = bounding_box

    # Slice the data chunk
    if benchmark_dict.get("bounding_box") is None:
        bottom = min(benchmark_dict["lat_start"], benchmark_dict["lat_end"]) - 5
        up = max(benchmark_dict["lat_start"], benchmark_dict["lat_end"]) + 5
        left = min(benchmark_dict["lon_start"], benchmark_dict["lon_end"]) - 5
        right = max(benchmark_dict["lon_start"], benchmark_dict["lon_end"]) + 5
        benchmark_dict["bounding_box"] = [bottom, left, up, right]

    benchmark_dict["data"] = Ocean(
        currents_data=ocean_datasets.get("uvDepth0", None),
        waves_data=ocean_datasets.get("Waves", None),
        wind_data=ocean_datasets.get("Wind10m", None),
        bounding_box=benchmark_dict["bounding_box"],
        land_file=land_file,
        interp_method=interp_method,
    )

    benchmark_dict["date_start"] = date_start

    if add_raw_data:
        raw_data = {}
        for key, value in ocean_datasets.items():
            raw_data[key] = get_data_chunk(value, *benchmark_dict["bounding_box"])

        benchmark_dict["raw_data"] = raw_data

    return benchmark_dict


def load_from_config(config: dict, benchmark: str, week: int = 0) -> dict:
    """Load benchmark configuration and prepare data and parameters to optimize.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing the parameters to load the benchmark.
    benchmark : str
        Name of the benchmark used.
    week : int, optional
        Number of the week to load the data, by default 0.

    Returns
    -------
    dict
        Dictionary containing the benchmark configuration.
    """
    # Copy the dictionary to not modify the original
    config = config.copy()
    # Compute the first date of the benchmark
    date_first = np.datetime64(config.pop("date_start"))
    date_start = date_first + np.timedelta64(week, "W")
    return load(benchmark, date_start=date_start, **config)
