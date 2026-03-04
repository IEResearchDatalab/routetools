import numpy as np
import xarray as xr


def get_data_chunk(
    xarray_data: xr.Dataset, bottom: float, left: float, up: float, right: float
) -> xr.Dataset:
    """Return the specified frame from the full dataset.

    Parameters
    ----------
    xarray_data : xr.Dataset
        The full dataset

    Returns
    -------
    xr.Dataset
        The set of data within the rectangle frame specified
    """
    data_chunk = xarray_data.sel(
        latitude=slice(min(bottom, up), max(bottom, up)),
        longitude=slice(min(left, right), max(left, right)),
    )
    return data_chunk


def correct_ds_coordinates(ds: xr.Dataset) -> xr.Dataset:
    """Fix dataset coordinates and add spacing attributes.

    The function fixes coordinate names and ordering. It also adds
    `grid_thickness` and `time_spacing` attributes to the dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to fix.

    Returns
    -------
    xr.Dataset
        Dataset with fixed coordinates.
    """
    # TODO: move this function and the calculation of the grid thickness to
    # the data_client and store in s3 with these corrections

    if "lat" in ds.coords and "lon" in ds.coords:
        # Fix the coordinates names
        ds = ds.rename({"lat": "latitude", "lon": "longitude"})

    # Fix the coordinates order
    ds = ds.sortby(["latitude", "longitude", "time"])

    # Add grid_thickness attribute
    try:
        ds.attrs["grid_thickness"] = ds.latitude.attrs["step"]
    except KeyError:
        # step is not defined in the attributes
        differences = ds.latitude.values[1:] - ds.latitude.values[:-1]

        assert np.allclose(differences, differences[0]), "Dataset is not evenly spaced"
        ds.attrs["grid_thickness"] = differences[0]

    # Add time_spacing attribute
    time_values = ds.time.values

    if len(time_values) == 1:
        ds.attrs["time_spacing"] = 24
    else:
        differences = (
            (time_values[1:] - time_values[:-1]) / np.timedelta64(1, "h")
        ).astype(int)

        assert np.allclose(
            differences, differences[0]
        ), "Dataset is not evenly spaced in time dimension"

        ds.attrs["time_spacing"] = differences[0]

    return ds


def date_from_week(week, year=2023) -> np.datetime64:
    """Return the date for the first day of the given ISO week and year.

    Parameters
    ----------
    week : int
        Week number. It is consider that the count of weeks goes from 1 to 52.
    year : int, optional
        Year, by default 2023.

    Returns
    -------
    np.datetime64
        Date of the first day of the week.
    """
    return np.datetime64(f"{year}-01-01") + np.timedelta64((week - 1) * 7, "D")
