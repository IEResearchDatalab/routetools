import importlib
import json
from collections.abc import Iterable

import cv2
import numpy as np
import pandas as pd
import shapely
import xarray as xr
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, shape
from shapely.ops import unary_union
from shapely.validation import make_valid

from routetools._cost.waves import beaufort_scale
from routetools.wrr_bench.dataset import correct_ds_coordinates, get_data_chunk
from routetools.wrr_bench.interpolate import Interpolator
from routetools.wrr_bench.polygons import (
    crop_polygon,
    invert_polygon,
    relative_to_latlon,
)

EARTH_RADIUS = 6378137
DEG2M = np.deg2rad(1) * EARTH_RADIUS


def data_zero(
    bounding_box: tuple | None = None, data_vars: tuple[str] = ("vo", "uo")
) -> xr.Dataset:
    """Create a fake current dataset with zeros.

    Parameters
    ----------
    bounding_box : Tuple
        Bounding box of the region (lat_min, lon_min, lat_max, lon_max)

    Returns
    -------
    xr.Dataset
        Dataset with zeros.
    """
    if bounding_box is None:
        lat_min, lon_min, lat_max, lon_max = (-90, -180, 90, 180)
    else:
        lat_min, lon_min, lat_max, lon_max = bounding_box
    # Space coordinates evenly across the bounding box
    # The step does not matter because the interpolator always picks the closest
    # point and here the data is always zero
    lat_values = np.linspace(lat_min, lat_max, 10)
    lon_values = np.linspace(lon_min, lon_max, 10)
    # Choose two timestamps - again which ones does not matter for the interpolator
    # as it will pick the closest one
    timestamp_values = np.array(["2024-01-01", "2024-01-02"]).astype("datetime64[ns]")

    # Fill the current velocities with zeros
    array_zero = np.zeros((len(lat_values), len(lon_values), len(timestamp_values)))

    # Create the fake current data in the format expected by the interpolator
    dict_vars = {}
    for k in data_vars:
        dict_vars[k] = (["latitude", "longitude", "time"], array_zero)
    data = xr.Dataset(
        dict_vars,
        coords={
            "latitude": lat_values,
            "longitude": lon_values,
            "time": timestamp_values,
        },
    )

    return data


class Ocean:
    """Container for ocean, wave and wind data with interpolation helpers."""

    def __init__(
        self,
        currents_data: xr.Dataset = None,
        waves_data: xr.Dataset = None,
        wind_data: xr.Dataset = None,
        currents_interpolator: Interpolator = None,
        waves_interpolator: Interpolator = None,
        wind_interpolator: Interpolator = None,
        radius: float = EARTH_RADIUS,
        bounding_box: Iterable = None,
        time_spacing: float = 8,
        min_thickness: float = 0.08333,
        land_file: str = "static_data/geojson/earth-seas-2km5-valid.geo.json",
        interp_method: str = "EvenLinearInterpolator",
        prepare_geom: bool = True,
        use_ice: bool = True,
        erode_ice: int = 1,
    ):
        """
        Create an Ocean object to manage meteo data.

        Parameters
        ----------
        currents_data : xr.Dataset, optional
            Currents dataset, by default None
        waves_data : xr.Dataset, optional
            Waves dataset, by default None
        wind_data : xr.Dataset, optional
            Wind dataset, by default None
        currents_interpolator : Interpolator, optional
            Interpolator object to gather currents data, by default None
            If None, a new Interpolator object will be created
        waves_interpolator : Interpolator, optional
            Interpolator object to gather waves data, by default None
            If None, a new Interpolator object will be created
        wind_interpolator : Interpolator, optional
            Interpolator object to gather wind data, by default None
            If None, a new Interpolator object will be created
        radius : float, optional
            Earth radius in meters, by default 6378137
        bounding_box : Iterable, optional
            Bounding box of the computation, by default None.
            The order of the coordinates is:
            (lat_bottom_left, lon_bottom_left, lat_up_right, lon_up_right).
            If no data is provided, the bounding box will be used to create
            synthetic data.
        time_spacing : float, optional
            Delta time between different values in hours, by default 8.
            It is used to reduce memory usage.
        min_thickness : float, optional
            Minimum thickness of the grid in degrees, by default 0.08333.
            This value is used to ensure that the data chunk is big enough.
        land_file : str, optional
            Path to the geojson file containing the land data, by default
            "static_data/geojson/earth-seas-2km5-valid.geo.json"
            If it is None, land will be always 0.
            This is mainly for testing.
        interp_method : str, optional
            Interpolation method to use, if not defined by the interpolator objects.
            By default, "EvenLinearInterpolator".
        prepare_geom : bool, optional
            If True, the geometry will be prepared to speed up the intersection and
            contain checks, by default True.
            A prepared geometry can't be pickled, so if you want to parallelize multiple
            runs it won't work.
        use_ice : bool, optional
            If True, the ice data will be used, by default True.
            Ice is consider and land and can be used only if waves_data is not None.
        erode_ice : int, optional
            Number of pixels to erode the ice mask, by default 1.
        """
        self.time_spacing = time_spacing
        self.min_thickness = min_thickness
        self.land_file = land_file
        self.interp_method = interp_method
        self.radius = radius  # in meters

        if bounding_box is not None:
            self.bounding_box = bounding_box
        else:
            if currents_data is not None:
                latitudes = currents_data.latitude
                longitudes = currents_data.longitude
            elif waves_data is not None:
                latitudes = waves_data.latitude
                longitudes = waves_data.longitude
            elif wind_data is not None:
                latitudes = wind_data.latitude
                longitudes = wind_data.longitude
            else:
                raise AttributeError("No data provided and no bounding box defined.")

            self.bounding_box = (
                latitudes.min().values,
                longitudes.min().values,
                latitudes.max().values,
                longitudes.max().values,
            )

        try:
            interp_obj: Interpolator = getattr(
                importlib.import_module("wrr_bench.interpolate"), interp_method
            )
            if not isinstance(Interpolator, type):
                raise AttributeError(f"{interp_method} is not a class.")
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Failed to import {interp_method}: {e}") from e

        bottom, left, up, right = self.bounding_box

        ocean_datasets = [currents_data, waves_data, wind_data]
        for i, ds in enumerate(ocean_datasets):
            if ds is not None:
                ds = correct_ds_coordinates(ds)
                if ds.grid_thickness > min_thickness:
                    # This is made to ensure that when the interpolation is made all
                    # the latitudes and longitdes are inside the chunked data.
                    ocean_datasets[i] = get_data_chunk(
                        ds,
                        bottom - ds.grid_thickness,
                        left - ds.grid_thickness,
                        up + ds.grid_thickness,
                        right + ds.grid_thickness,
                    )

                if ds.time_spacing < time_spacing:
                    # Go from start time to end time in the step defined by time_spacing
                    start_time = ocean_datasets[i].time.values[0]
                    end_time = ocean_datasets[i].time.values[-1]
                    time_index = pd.date_range(
                        start=start_time, end=end_time, freq=f"{time_spacing}h"
                    )
                    ocean_datasets[i] = ocean_datasets[i].sel(
                        time=time_index, method="nearest"
                    )
                    ocean_datasets[i] = ocean_datasets[i].assign_coords(time=time_index)

                    ocean_datasets[i].attrs["time_spacing"] = time_spacing

        currents_data, waves_data, wind_data = ocean_datasets

        if land_file is None:
            self.shapely_ocean = MultiPolygon([Polygon()])
        else:
            self.shapely_ocean = self.create_land_mask(
                use_ice=use_ice,
                erode_ice=erode_ice,
                waves_data=waves_data,
                land_file=land_file,
                bounding_box=self.bounding_box,
            )

        if prepare_geom:
            shapely.prepare(self.shapely_ocean)

        if currents_interpolator is None:
            if currents_data is None:
                currents_data = data_zero(self.bounding_box, data_vars=("vo", "uo"))
            if "land" in list(currents_data.keys()):
                currents_data = currents_data.drop_vars("land")
            # Next line can go inside the interpolator
            currents_data = currents_data.fillna(0)
            self.currents_interpolator = interp_obj(
                currents_data, vars=list(currents_data.keys())
            )
        else:
            self.currents_interpolator = currents_interpolator

        if waves_interpolator is None:
            if waves_data is None:
                waves_data = data_zero(
                    self.bounding_box, data_vars=("height", "direction")
                )
            if "land" in list(waves_data.keys()):
                waves_data = waves_data.drop_vars("land")
            waves_data = waves_data.fillna(0)
            self.waves_interpolator = interp_obj(
                waves_data, vars=list(waves_data.keys())
            )
        else:
            self.waves_interpolator = waves_interpolator

        if wind_interpolator is None:
            if wind_data is None:
                wind_data = data_zero(
                    self.bounding_box, data_vars=("vgrd10m", "ugrd10m")
                )
            wind_data = wind_data.fillna(0)
            self.wind_interpolator = interp_obj(wind_data, vars=list(wind_data.keys()))
        else:
            self.wind_interpolator = wind_interpolator

    def create_land_mask(
        self,
        use_ice: bool,
        erode_ice: int,
        waves_data: xr.Dataset,
        land_file: str,
        bounding_box: Iterable[float],
    ) -> np.ndarray:
        """
        Create a mask with the ice and land data.

        Parameters
        ----------
        use_ice : bool
            If True, the ice data will be used.
        erode_ice : int
            Number of pixels to erode the ice mask.
        waves_data : xr.Dataset
            Waves dataset.
        land_file : str
            Path to the geojson file containing the land data.
        bounding_box : Iterable[float]
            Bounding box of the computation.
            The order of the coordinates is:
            (lat_bottom_left, lon_bottom_left, lat_up_right, lon_up_right).

        Returns
        -------
        np.ndarray
            Mask with the land data.
        """
        with open(land_file) as f:
            land_data = json.load(f)

        shapely_ocean = shape(land_data["geometries"][0])
        # This way it can be used inside A*
        shapely_ocean = crop_polygon(shapely_ocean, bounding_box)
        shapely_ocean = invert_polygon(shapely_ocean, bounding_box)

        if use_ice and waves_data is not None:
            # Extract waves NaN mask (Land + Ice)
            # It will take only the first timestamp
            waves_mask = np.max(
                np.isnan(waves_data["height"].values).astype(np.uint8), axis=0
            )

            height, width = waves_mask.shape

            if erode_ice > 0:
                # Erode `erode-ice` pixels the mask to avoid losing land resolution
                kernel = np.ones((2, 2), np.uint8)
                waves_mask = cv2.erode(waves_mask, kernel, iterations=erode_ice)

            # Find contours
            contours, _ = cv2.findContours(
                waves_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Create polygons from contours
            polygons = []
            for c in contours:
                if c.shape[0] > 2:
                    coords = [
                        relative_to_latlon(y, x, height, width, bounding_box)
                        for x, y in c.reshape(-1, 2)
                    ]
                    polygons.append(Polygon(coords))

            multipolygon = MultiPolygon(polygons)
            multipolygon = make_valid(multipolygon)

            # This step is not needed, but it is done to ensure that the
            # multipolygon is inside the bounding box
            ice_multipolygon = crop_polygon(multipolygon, bounding_box)

            shapely_ocean = unary_union([ice_multipolygon, shapely_ocean])

        if isinstance(shapely_ocean, Polygon):
            shapely_ocean = MultiPolygon([shapely_ocean])
        else:
            # Clean the geometry
            shapely_ocean = MultiPolygon(
                [p for p in shapely_ocean.geoms if isinstance(p, Polygon)]
            )

        assert shapely_ocean.is_valid, "The geometry is not valid."

        return shapely_ocean

    def unprepare_geom(self):
        """Unprepare the geometry to allow pickling the object."""
        if shapely.is_prepared(self.shapely_ocean):
            shapely.destroy_prepared(self.shapely_ocean)

    def prepare_geom(self):
        """Prepare the geometry to speed up intersection checks."""
        if not shapely.is_prepared(self.shapely_ocean):
            shapely.prepare(self.shapely_ocean)

    def get_land_edge(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray[int]:
        """Return an array indicating which route edges intersect land.

        A value of 1 means an adjacent edge crosses land.

        Parameters
        ----------
        lat : np.ndarray
            Latitude array in degrees.
        lon : np.ndarray
            Longitude array in degrees.

        Returns
        -------
        np.ndarray[int]
            Array of 0's and 1's, where 1 is land and 0 is sea.
        """
        intersects = np.zeros(len(lat), dtype=int)

        for i in range(len(lat) - 1):
            line = LineString([(lon[i], lat[i]), (lon[i + 1], lat[i + 1])])

            if self.shapely_ocean.intersects(line):
                intersects[i] = 1
                intersects[i + 1] = 1

        return intersects

    def get_land(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray[int]:
        """Return an array of 0's and 1's where 1 indicates land.

        Parameters
        ----------
        lat : np.ndarray
            Latitude array in degrees.
        lon : np.ndarray
            Longitude array in degrees.

        Returns
        -------
        np.ndarray[int]
            Array of 0's and 1's, where 1 is land and 0 is sea.
        """
        lat = np.atleast_1d(lat)
        lon = np.atleast_1d(lon)

        points = [Point(lon[i], lat[i]) for i in range(len(lat))]

        return self.shapely_ocean.contains(points).astype(int)

    def get_currents(
        self, lat: np.ndarray, lon: np.ndarray, time: np.ndarray, derivate: bool = False
    ) -> np.ndarray:
        """Interpolate currents at the given positions and times."""
        assert self.currents_interpolator is not None, "No currents data available."

        assert len(lon) == len(lat) and len(lat) == len(
            time
        ), "The length of the longitude, latitude and time arrays must be the same."

        derivate = (
            hasattr(self.currents_interpolator, "interpolate_derivates") and derivate
        )

        if not derivate:
            data = self.currents_interpolator.interpolate(lat=lat, lon=lon, ts=time)
        else:
            data = self.currents_interpolator.interpolate_derivates(
                lat=lat, lon=lon, ts=time
            )

        return data

    def get_waves(
        self, lat: np.ndarray, lon: np.ndarray, time: np.ndarray, derivate: bool = False
    ) -> np.ndarray:
        """Interpolate waves data at the given positions and times."""
        assert self.waves_interpolator is not None, "No waves data available."

        assert len(lon) == len(lat) and len(lat) == len(
            time
        ), "The length of the longitude, latitude and time arrays must be the same."

        derivate = (
            hasattr(self.waves_interpolator, "interpolate_derivates") and derivate
        )

        if not derivate:
            data = self.waves_interpolator.interpolate(lat=lat, lon=lon, ts=time)
        else:
            data = self.waves_interpolator.interpolate_derivates(
                lat=lat, lon=lon, ts=time
            )

        return data

    def get_wind(
        self, lat: np.ndarray, lon: np.ndarray, time: np.ndarray, derivate: bool = False
    ) -> np.ndarray:
        """Interpolate wind data at the given positions and times."""
        assert self.wind_interpolator is not None, "No wind data available."

        assert len(lon) == len(lat) and len(lat) == len(
            time
        ), "The length of the longitude, latitude and time arrays must be the same."

        derivate = hasattr(self.wind_interpolator, "interpolate_derivates") and derivate

        if not derivate:
            data = self.wind_interpolator.interpolate(lat=lat, lon=lon, ts=time)
        else:
            data = self.wind_interpolator.interpolate_derivates(
                lat=lat, lon=lon, ts=time
            )

        return data

    def get_beaufort(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        time: np.ndarray,
        use_wind: bool = True,
        asfloat: bool = False,
    ):
        """Return the beaufort scale at positions using wind or waves."""
        if use_wind:
            return beaufort_scale(
                wind_speed=np.sqrt(
                    (self.get_wind(lat=lat, lon=lon, time=time) ** 2).sum(axis=0)
                ),
                asfloat=asfloat,
            )
        else:
            return beaufort_scale(
                wave_height=self.get_waves(lat=lat, lon=lon, time=time)[0],
                asfloat=asfloat,
            )

    def _to_dict(
        self,
        data: tuple[np.ndarray] | np.ndarray,
        var_keys: Iterable[str],
        derivate: bool,
    ) -> dict:
        """Convert interpolator output into a dict keyed by variable names."""
        dict_data = {}
        if derivate:
            for i, sub_name in enumerate(["", "_dlat", "_dlon"]):
                for j, key in enumerate(var_keys):
                    dict_data[key + sub_name] = data[i][j, :]
        else:
            for i, key in enumerate(var_keys):
                dict_data[key] = data[i, :]

        return dict_data

    def get_data(
        self, lat: np.ndarray, lon: np.ndarray, time: np.ndarray, derivate: bool = False
    ) -> np.ndarray:
        """Return a dictionary with interpolated data for the given positions."""
        dict_data = {}
        dict_data["land"] = self.get_land(lat=lat, lon=lon)
        if self.currents_interpolator is not None:
            currents_data = self.get_currents(
                lat=lat, lon=lon, time=time, derivate=derivate
            )
            dict_data.update(
                self._to_dict(
                    currents_data, self.currents_interpolator.vars, derivate=derivate
                )
            )
        if self.waves_interpolator is not None:
            waves_data = self.get_waves(lat=lat, lon=lon, time=time, derivate=derivate)
            dict_data.update(
                self._to_dict(
                    waves_data, self.waves_interpolator.vars, derivate=derivate
                )
            )
        if self.wind_interpolator is not None:
            wind_data = self.get_wind(lat=lat, lon=lon, time=time, derivate=derivate)
            dict_data.update(
                self._to_dict(wind_data, self.wind_interpolator.vars, derivate=derivate)
            )
            wind: np.ndarray = wind_data[0] if derivate else wind_data
            dict_data["beaufort"] = beaufort_scale(np.sqrt((wind**2).sum(axis=0)))
        return dict_data
