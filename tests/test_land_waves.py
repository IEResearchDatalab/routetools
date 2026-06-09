import numpy as np
import xarray as xr

from routetools.wrr_bench.ocean import Ocean


def test_waves_create_land_when_geojson_missing():
    # Create a small synthetic waves dataset with a NaN blob
    times = np.array(["2024-01-01", "2024-01-02"]).astype("datetime64[ns]")
    lat = np.linspace(0, 3, 4)
    lon = np.linspace(0, 4, 5)

    # shape (time, lat, lon)
    height = np.ones((len(times), len(lat), len(lon)), dtype=float)
    # Insert NaNs in a small rectangular region (time x lat x lon)
    height[:, 1:3, 2:4] = np.nan

    # Build dataset by assigning DataArrays directly
    ds = xr.Dataset()
    ds["height"] = xr.DataArray(
        height,
        dims=("time", "latitude", "longitude"),
        coords={"time": times, "latitude": lat, "longitude": lon},
    )
    ds["direction"] = xr.DataArray(
        np.zeros_like(height),
        dims=("time", "latitude", "longitude"),
        coords={"time": times, "latitude": lat, "longitude": lon},
    )

    # Instantiate Ocean with no land_file so it should use waves NaN mask
    ocean = Ocean(
        currents_data=None,
        waves_data=ds,
        wind_data=None,
        currents_interpolator=object(),
        waves_interpolator=object(),
        wind_interpolator=object(),
        land_file=None,
        use_ice=True,
        erode_ice=0,
        prepare_geom=False,
    )

    # pick a point inside the NaN rectangle by mapping pixel center to lat/lon
    height_n = len(lat)
    width_n = len(lon)
    y_center = 1.5  # center between pixel rows 1 and 2
    x_center = 2.5  # center between pixel cols 2 and 3
    lat_pt = lat[0] + (lat[-1] - lat[0]) * (y_center / height_n)
    lon_pt = lon[0] + (lon[-1] - lon[0]) * (x_center / width_n)

    # Inside land
    land_flag = ocean.get_land(np.array([lat_pt]), np.array([lon_pt]))[0]
    assert land_flag == 1

    # Outside land (pixel center between row 0 and 1, col 0 and 1)
    y_out = 0.5
    x_out = 0.5
    lat_pt_out = lat[0] + (lat[-1] - lat[0]) * (y_out / height_n)
    lon_pt_out = lon[0] + (lon[-1] - lon[0]) * (x_out / width_n)
    land_flag_out = ocean.get_land(np.array([lat_pt_out]), np.array([lon_pt_out]))[0]
    assert land_flag_out == 0
