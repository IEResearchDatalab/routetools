from collections.abc import Iterable

import numpy as np
import pandas as pd

from routetools.wrr_bench.ocean import Ocean
from routetools.wrr_utils.simulation import (
    angle_from_points,
    compute_currents_projection,
    compute_times_given_coordinates_and_start,
    compute_times_linalg,
    distance_from_points,
    even_reparametrization,
    split_route_segments,
)


class Route:
    """Represent a maritime route and provide utilities to manipulate it.

    Instances store coordinates, timestamps, ship speeds and ocean data used
    by the various route operations.
    """

    def __init__(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        time_stamps: np.ndarray,
        vel_ship: float | np.ndarray,
        ocean_data: Ocean,
    ):
        """Initialize a Route.

        Parameters
        ----------
        lats : np.ndarray
            The list of latitudes on the route
        lons : np.ndarray
            The list of longitudes on the route
        time_stamps : np.ndarray
            The list of timestamps for each pair of lats and lons,
        vel_ship : Union[float, np.array]
            Speed of the ship. If it is a float, then it is a constant speed and
            the same speed is used for all the points
        ocean_data : Ocean
            Ocean object, to compute the currents projection.
        """
        assert lats.shape == lons.shape, (
            "The dimensions of the latitude array and longitude array are not of the"
            " same dimension"
        )
        assert lons.shape == time_stamps.shape, (
            "The dimensions of the longitude array and time_stamp array are not of the"
            " same dimension"
        )
        self.lats = np.array(lats, dtype="float64")
        self.lons = np.array(lons, dtype="float64")
        self.time_stamps = np.array(time_stamps, dtype="datetime64")
        # Ensure the array of velocities has the appropriate length
        if not isinstance(vel_ship, np.ndarray):
            vel_ship = np.full(len(self.lats) - 1, vel_ship)
        elif len(vel_ship) > len(self.lats) - 1:
            vel_ship = vel_ship[: len(self.lats) - 1]
        elif len(vel_ship) < len(self.lats) - 1:
            diff = len(self.lats) - 1 - len(vel_ship)
            vel_ship = np.append(vel_ship, [np.nan] * diff)

        self.vel_ship = vel_ship
        self.ocean_data = ocean_data
        self.land = ocean_data.get_land(lats, lons).astype(bool)
        self.filename = None

    @classmethod
    def from_start_time(
        cls,
        lats: np.ndarray,
        lons: np.ndarray,
        time_start: np.datetime64,
        vel_ship: float | np.ndarray,
        ocean_data: Ocean,
        land_penalization: float = 0,
    ):
        """Route object constructor with unknown intermediate timestamps.

        Parameters
        ----------
        lats : np.ndarray
            The list of latitudes on the route
        lons : np.ndarray
            The list of longitudes on the route
        time_start : np.datetime64
            The start time of the journey
        ocean_data : Ocean
            Ocean object, to compute the currents projection.
        vel_ship : float
            Speed of the ship. If it is a float, then it is a constant speed and
            the same speed is used for all the points.
        land_penalization : float
            Extra distance penalization for every point in the route
            that crosses land (in meters), by default 0.

        Returns
        -------
        Route Object
            Invokes the default constructor to create Route Object
        """
        assert lats.shape == lons.shape, (
            "The dimensions of the latitude array and longitude array are not of the"
            " same dimension"
        )
        if not isinstance(vel_ship, np.ndarray):
            vel_ship = np.full(len(lats) - 1, vel_ship)
        ts = compute_times_given_coordinates_and_start(
            latitudes=lats,
            longitudes=lons,
            ts_start=time_start,
            vel_ship=vel_ship,
            ocean_data=ocean_data,
            land_penalization=land_penalization,
        )[0]
        return cls(lats, lons, ts, vel_ship, ocean_data)

    @classmethod
    def from_csv_file(
        cls,
        path_to_file: str,
        ocean_data: Ocean,
        vel_ship: float = None,
        lat_column_name: str = "lat",
        lon_column_name: str = "lon",
        time_stamp_column_name: str = "ts",
        vel_ship_column_name: str = "vel_ship",
    ):
        """Read a CSV file and store the data to attributes.

        Parameters
        ----------
        path_to_file : str
            Path to the csv file
        ocean_data : Ocean
            Ocean object, to compute the currents projection.
        vel_ship : float, optional
            Constant speed of the ship, by default None.
            If it is None, then the speed is read from the csv file.
        lat_column_name : str, optional
            Name of the latitude column, by default "lat"
        lon_column_name : str, optional
            Name of the longitude column, by default "lon"
        time_stamp_column_name : str, optional
            Name of the time stamps column, by default "ts"
        vel_ship_column_name : str, optional
            Name of the ship velocity column, by default "vel_ship"
            If vel_ship is not None, then this parameter is ignored.
        """
        dict_route_dataframe = pd.read_csv(path_to_file)

        route = cls.from_dataframe(
            df=dict_route_dataframe,
            vel_ship=vel_ship,
            ocean_data=ocean_data,
            lat_column_name=lat_column_name,
            lon_column_name=lon_column_name,
            time_stamp_column_name=time_stamp_column_name,
            vel_ship_column_name=vel_ship_column_name,
        )
        route.filename = path_to_file
        return route

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        ocean_data: Ocean,
        vel_ship: float = None,
        lat_column_name: str = "lat",
        lon_column_name: str = "lon",
        time_stamp_column_name: str = "ts",
        vel_ship_column_name: str = "vel_ship",
    ):
        """Read a DataFrame and store the data to attributes.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe containing the route data
        ocean_data : Ocean
            Ocean object, to compute the currents projection.
        vel_ship : float, optional
            Constant speed of the ship, by default None.
            If it is None, then the speed is read from the csv file.
        lat_column_name : str, optional
            Name of the latitude column, by default "lat"
        lon_column_name : str, optional
            Name of the longitude column, by default "lon"
        time_stamp_column_name : str, optional
            Name of the time stamps column, by default "ts"
        vel_ship_column_name : str, optional
            Name of the ship velocity column, by default "vel_ship"
            If vel_ship is not None, then this parameter is ignored.
        """
        lat_column_name = lat_column_name.strip()
        lon_column_name = lon_column_name.strip()
        time_stamp_column_name = time_stamp_column_name.strip()

        df.columns = df.columns.str.strip()

        lons = np.array(df[lon_column_name].values)
        lats = np.array(df[lat_column_name].values)
        time_stamps = np.array(df[time_stamp_column_name].values)

        if vel_ship is None:
            vel_ship_column_name = vel_ship_column_name.strip()
            vel_ship = np.array(df[vel_ship_column_name].values)
        return cls(lats, lons, time_stamps, vel_ship, ocean_data)

    @classmethod
    def concatenate(cls, routes: Iterable["Route"], remove_mid_point: bool = False):
        """Concatenate a list of routes into a single route.

        Parameters
        ----------
        routes : Iterable[Route]
            List of routes to be concatenated
        remove_mid_point : bool, optional
            Remove the point between two routes, by default True

        Returns
        -------
        Route
            A new route object.
        """
        # TODO: Create a similar assert for ocean_data

        if remove_mid_point:
            lats = np.concatenate([route.lats[:-1] for route in routes])
            lons = np.concatenate([route.lons[:-1] for route in routes])
            time_stamps = np.concatenate([route.time_stamps[:-1] for route in routes])
            vel_ship = np.concatenate([route.vel_ship[:-1] for route in routes])

            lats = np.append(lats, routes[-1].lats[-1])
            lons = np.append(lons, routes[-1].lons[-1])
            time_stamps = np.append(time_stamps, routes[-1].time_stamps[-1])
            vel_ship = np.append(vel_ship, routes[-1].vel_ship[-1])
        else:
            lats = np.concatenate([route.lats for route in routes])
            lons = np.concatenate([route.lons for route in routes])
            time_stamps = np.concatenate([route.time_stamps for route in routes])
            vel_ship = np.concatenate([route.vel_ship for route in routes])

        ocean_data = routes[0].ocean_data
        return cls(lats, lons, time_stamps, vel_ship, ocean_data)

    def __len__(self):
        """Return number of points in the route."""
        return len(self.lats)

    def __getitem__(self, idx: int) -> "Route":
        """Return a Route or element(s) accessed by index."""
        return Route(
            self.lats[idx],
            self.lons[idx],
            self.time_stamps[idx],
            self.vel_ship,
            self.ocean_data,
        )

    def split_by_coords(self, lat: float, lon: float) -> tuple["Route"]:
        """Split the route into two routes from the closest point to coords.

        The route is split at the point closest to (lat, lon).
        """
        distances = [
            distance_from_points(lat, lon, _lat, _lon, land_penalization=0)
            for _lat, _lon in zip(self.lats, self.lons, strict=False)
        ]
        idx = np.argmin(distances)

        route1 = Route(
            self.lats[: idx + 1],
            self.lons[: idx + 1],
            self.time_stamps[: idx + 1],
            self.vel_ship,
            self.ocean_data,
        )
        route2 = Route(
            self.lats[idx:],
            self.lons[idx:],
            self.time_stamps[idx:],
            self.vel_ship,
            self.ocean_data,
        )

        return route1, route2

    def recompute_given_time_start(
        self, time_start: np.datetime64, land_penalization: float = 0
    ):
        """Recompute the time stamps given a new starting time.

        Parameters
        ----------
        time_start : np.datetime64
            The new starting time
        land_penalization : float, optional
            Extra distance penalization for every point in the route
            that crosses land (in meters), by default 0.
        """
        ts = compute_times_given_coordinates_and_start(
            latitudes=self.lats,
            longitudes=self.lons,
            ts_start=time_start,
            vel_ship=self.vel_ship,
            ocean_data=self.ocean_data,
            land_penalization=land_penalization,
        )[0]
        self.time_stamps = ts

    @property
    def distance_per_segment(self) -> np.ndarray:
        """Compute the distance per segment on a route.

        Distances use the Haversine formula between each segment.

        Returns
        -------
        np.array(float)
            Distance per segment in meters.
        """
        distances = distance_from_points(
            lat_start=self.lats[:-1],
            lon_start=self.lons[:-1],
            lat_end=self.lats[1:],
            lon_end=self.lons[1:],
            ocean_data=self.ocean_data,
        )
        return distances

    @property
    def total_distance(self) -> float:
        """Compute the total distance of a route.

        Returns
        -------
        float
            Total distance on the path in meters.
        """
        return np.sum(self.distance_per_segment)

    @property
    def time_per_segment(self) -> np.ndarray:
        """Return the time per segment in seconds."""
        cum_diff = self.time_stamps[1:] - self.time_stamps[:-1]
        return cum_diff / np.timedelta64(1, "us") / 1e6

    @property
    def total_time(self) -> float:
        """Computes the total time taken on a given route.

        Returns
        -------
        float
            Total time taken on the route.
        """
        time_stamps = self.time_stamps
        time = (time_stamps[-1] - time_stamps[0]) / np.timedelta64(1, "us") / 1e6
        return time

    @property
    def ground_vel_per_segment(self) -> np.ndarray:
        """Compute the ground velocity per segment on a route.

        Returns
        -------
        np.array(float)
            Ground velocity per segment.
        """
        distances = self.distance_per_segment
        time_per_segment = self.time_per_segment
        return distances / time_per_segment

    @property
    def heading(self) -> np.ndarray:
        """Compute the ground heading per segment on a route.

        Returns
        -------
        np.array(float)
            Ground heading per segment.
        """
        return angle_from_points(
            self.lats[:-1],
            self.lons[:-1],
            self.lats[1:],
            self.lons[1:],
            radius=self.ocean_data.radius,
        )

    def insert_point(self, lat: float, lon: float, time_stamp: np.datetime64 = None):
        """Insert a specified point (lat, lon, time_stamp) into the route.

        If the timestamp of the point is not specified, then a timestamp is
        generated from the route end and the point is appended.

        Parameters
        ----------
        lat : float
            The latitude of the point to be added
        lon : float
            The longitude of the point to be added
        time_stamp : np.datetime64
            The timestamp of the point to be added
        """
        if time_stamp is None:
            routes = compute_times_linalg(
                lat_start=self.lats[-1],
                lon_start=self.lons[-1],
                lat_end=lat,
                lon_end=lon,
                timestamps=self.time_stamps[-1],
                vel_ship=self.vel_ship[-1],
                ocean_data=self.ocean_data,
            )
            time = (
                self.time_stamps[-1] + float(routes[0]) * np.timedelta64(1, "us") / 1e6
            )
            self.lats = np.append(self.lats, lat)
            self.lons = np.append(self.lons, lon)
            self.time_stamps = np.append(self.time_stamps, time)
            self.vel_ship = np.append(self.vel_ship, self.vel_ship[-1])
        else:
            relative_time_stamp = self.time_stamps - time_stamp
            time_stamp_idx = np.argmin(np.abs(relative_time_stamp))
            self.time_stamps = np.insert(self.time_stamps, time_stamp_idx, time_stamp)
            self.lats = np.insert(self.lats, time_stamp_idx, lat)
            self.lons = np.insert(self.lons, time_stamp_idx, lon)
            self.vel_ship = np.insert(self.vel_ship, time_stamp_idx, self.vel_ship[0])

    def get_ocean_data(self) -> dict:
        """Return ocean data for the route coordinates."""
        return self.ocean_data.get_data(
            lon=self.lons, lat=self.lats, time=self.time_stamps
        )

    @property
    def beaufort_value_per_segment(self) -> np.ndarray:
        """Compute the Beaufort value per segment on the route.

        Returns
        -------
        np.ndarray
            An array of Beaufort values of the same shape as the route.
        """
        return self.ocean_data.get_beaufort(
            self.lons, self.lats, self.time_stamps, use_wind=False
        )

    def as_dict(self, all_data: bool = False, serializable: bool = False) -> dict:
        """Return the route data as a dictionary.

        Parameters
        ----------
        all_data : bool, optional
            If True, the dictionary will contain all the data, by default False.
            If False, the dictionary will only contain the coordinates and the
            timestamps.
        serializable : bool, optional
            If True the dictionary will be serializable to send it over through
            the network, by default False.

        Returns
        -------
        dict
            Dictionary containing the route data.
        """
        vel_sh = np.concatenate([self.vel_ship, [np.nan]])
        dict_route = {
            "lat": self.lats,
            "lon": self.lons,
            "ts": self.time_stamps,
            "vel_ship": vel_sh,
            "land": self.land,
        }
        if all_data:
            # Time per segment and velocity over ground and
            time = np.concatenate([[np.nan], self.time_per_segment])
            vel_gr = np.concatenate([self.ground_vel_per_segment, [np.nan]])
            heading = np.concatenate([self.heading, [np.nan]])
            distance = np.concatenate([[np.nan], self.distance_per_segment.flatten()])
            dict_data = {
                "time": time,
                "vel_ground": vel_gr,
                "heading": heading,
                "distance": distance,
            }
            dict_route = dict_route | dict_data
            # Ocean data
            dict_ocean = self.ocean_data.get_data(
                lon=self.lons, lat=self.lats, time=self.time_stamps
            )
            dict_route = dict_route | dict_ocean

        if serializable:
            dict_route = {k: v.tolist() for k, v in dict_route.items()}
            dict_route["ts"] = [str(ts) for ts in dict_route["ts"]]

        return dict_route

    def as_dataframe(self, all_data: bool = False) -> pd.DataFrame:
        """Return a pandas DataFrame representing the route."""
        return pd.DataFrame(self.as_dict(all_data=all_data))

    def export_to_csv(
        self,
        path_to_file: str,
        all_data: bool = False,
        extra: dict | None = None,
        decimals: int = 3,
    ):
        """Write the list of lats, lons, timestamps to a CSV file.

        Parameters
        ----------
        path_to_file : str
            Path to file.
        """
        df = self.as_dataframe(all_data=all_data)
        if extra is not None:
            df = df.assign(**extra)
        df.to_csv(path_to_file, index=False, float_format=f"%.{decimals}f")

    def _even_reparametrization(
        self,
        cost_per_segment: float = None,  # In seconds
        n_points: int = None,
        n_iter: int = 5,
        verbose: bool = False,
    ):
        """Wrap `even_reparametrization` to use a dataframe-based cost.

        The cost uses `compute_times_given_coordinates_and_start`.
        """
        stw = np.concatenate([self.vel_ship, [np.nan]])
        curve = np.stack((self.lons, self.lats, stw), axis=1)
        date_start = self.time_stamps[0]

        def cost(curve):
            latitudes = curve[:, 1]
            longitudes = curve[:, 0]
            stw = curve[:-1, 2]
            ts = compute_times_given_coordinates_and_start(
                latitudes,
                longitudes,
                date_start,
                stw,
                self.ocean_data,
                land_penalization=0,
            )[0]

            # Make sure the difference is in seconds, with 6 decimal places
            cum_diff = (ts[1:] - ts[:-1]).astype("timedelta64[us]").astype(float) / 1e6

            return cum_diff

        new_curve, _ = even_reparametrization(
            curve,
            cost,
            n_points=n_points,
            cost_per_segment=cost_per_segment,
            n_iter=n_iter,
            verbose=verbose,
        )

        lats = new_curve[:, 1]
        lons = new_curve[:, 0]
        stw = new_curve[:-1, 2]
        return self.from_start_time(
            lats,
            lons,
            self.time_stamps[0],
            vel_ship=stw,
            ocean_data=self.ocean_data,
            land_penalization=0,
        )

    def reparametrize_to_fixed_cost(
        self, cost_per_segment: float, n_iter: int = 5, verbose: bool = False
    ) -> "Route":
        """
        Given a route, reparameterize it so that the time between each point is equal.

        Parameters
        ----------
        cost_per_segment : float, optional
            The target cost of each segment.
        n_iter : int, optional
            Number of iterations, by default 5
        verbose : bool, optional
            Print the standard deviation of the cost at each iteration

        Returns
        -------
        Route
            A new Route object after reparametrization
        """
        return self._even_reparametrization(
            cost_per_segment=cost_per_segment,
            n_iter=n_iter,
            verbose=verbose,
        )

    def reparametrize_to_num_waypoints(
        self, n_points: int, n_iter: int = 5, verbose: bool = False
    ) -> "Route":
        """
        Given a route, reparameterize it so that the number of points is the same.

        Parameters
        ----------
        n_points : int, optional
            Number of desired points for the new polyline.
        n_iter : int, optional
            Number of iterations, by default 5
        verbose : bool, optional
            Print the standard deviation of the cost at each iteration

        Returns
        -------
        Route
            A new Route object after reparametrization
        """
        return self._even_reparametrization(
            n_points=n_points,
            n_iter=n_iter,
            verbose=verbose,
        )

    def reparametrize_to_maximum_segment_length(self, threshold: float) -> "Route":
        """
        Given a route, reparameterize it.

        New points are added in between the original points until the distance between
        each point is less than a threshold.
        This reparametrization will never change the original points, but will add
        new points in between.

        Parameters
        ----------
        threshold : float
            The maximum distance between two points, in meters.

        Returns
        -------
        Route
            A new Route object after reparametrization
        """
        lats, lons = split_route_segments(
            self.lats, self.lons, threshold=threshold, ocean_data=self.ocean_data
        )
        return self.from_start_time(
            lats,
            lons,
            self.time_stamps[0],
            vel_ship=self.vel_ship[0],
            ocean_data=self.ocean_data,
            land_penalization=0,
        )

    def feasibility_check(
        self,
        vel_expected: float = None,
        tol: float = 0.1,
    ) -> bool:
        """
        Check if the given trajectory is feasible.

        Parameters
        ----------
        vel_expected : float
            Expected velocity of the ship in m/s.
        data : xr.Dataset
            Dataset containing the ocean data.
        tol : float
            Tolerance in percentage for the feasibility check, by default 0.1.

        Returns
        -------
        bool
            True if the trajectory is feasible, False otherwise.
        """
        if vel_expected is None:
            vel_expected = self.vel_ship
        lat_start = self.lats[:-1]
        lon_start = self.lons[:-1]
        lat_end = self.lats[1:]
        lon_end = self.lons[1:]
        ts_start = self.time_stamps[:-1]

        distance = distance_from_points(
            lat_start,
            lon_start,
            lat_end,
            lon_end,
            ocean_data=self.ocean_data,
            land_penalization=0,
        )
        current_v_proj, current_u_proj = compute_currents_projection(
            lat_start,
            lon_start,
            lat_end,
            lon_end,
            ts_start,
            ocean_data=self.ocean_data,
        )

        # Compute the time
        times = np.diff(self.time_stamps).astype("timedelta64[us]").astype(float) / 1e6

        vel_real = distance / times

        vel_ship = np.sqrt((vel_real - current_u_proj) ** 2 + current_v_proj**2)

        return np.all(np.abs(vel_ship - vel_expected) < (tol * vel_expected))
