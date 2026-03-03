from collections.abc import Iterable

import numpy as np
import pandas as pd

from routetools.wrr_bench.ocean import Ocean
from routetools.wrr_bench.simulation import (
    angle_from_points,
    compute_currents_projection,
    compute_times_given_coordinates_and_start,
    compute_times_linalg,
    distance_from_points,
    even_reparametrization,
    split_route_segments,
)


class Route:
    """Representation of a route (lat, lon, timestamp) with utilities."""

    def __init__(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        time_stamps: np.ndarray,
        vel_ship: float,
        ocean_data: Ocean,
    ):
        """Create a Route object.

        Parameters
        ----------
        lats : np.ndarray
            The list of latitudes on the route
        lons : np.ndarray
            The list of longitudes on the route
        time_stamps : np.ndarray
            The list of timestamps for each pair of lats and lons,
        vel_ship : float
            Constant speed of the ship
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
        self.lats = np.array(lats, dtype="float32")
        self.lons = np.array(lons, dtype="float32")
        self.time_stamps = np.array(time_stamps, dtype="datetime64")
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
        vel_ship: float,
        ocean_data: Ocean,
        land_penalization: float = 1e6,
    ):
        """Create a Route object when intermediate timestamps are unknown.

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
            Constant speed of the ship
        land_penalization : float
            Time penalization (in seconds) for every point of the route
            that is located in land, by default 1e6.

        Returns
        -------
        Route Object
            Invokes the default constructor to create Route Object
        """
        assert lats.shape == lons.shape, (
            "The dimensions of the latitude array and longitude array are not of the"
            " same dimension"
        )
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
        vel_ship: float,
        ocean_data: Ocean,
        lat_column_name: str = "lat",
        lon_column_name: str = "lon",
        time_stamp_column_name: str = "ts",
    ):
        """Read a CSV file and store the data to attributes.

        Parameters
        ----------
        path_to_file : str
            Path to the csv file
        ocean_data : Ocean
            Ocean object, to compute the currents projection.
        vel_ship : float
            Constant speed of the ship.
        lat_column_name : str, optional
            Name of the latitude column, by default "lat"
        lon_column_name : str, optional
            Name of the longitude column, by default "lon"
        time_stamp_column_name : str, optional
            Name of the time stamps column, by default "ts"
        """
        dict_route_dataframe = pd.read_csv(path_to_file)

        route = cls.from_dataframe(
            dict_route_dataframe,
            vel_ship,
            ocean_data,
            lat_column_name,
            lon_column_name,
            time_stamp_column_name,
        )
        route.filename = path_to_file
        return route

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        vel_ship: float,
        ocean_data: Ocean,
        lat_column_name: str = "lat",
        lon_column_name: str = "lon",
        time_stamp_column_name: str = "ts",
    ):
        """Read a dataframe and store the data to attributes.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe containing the route data
        ocean_data : Ocean
            Ocean object, to compute the currents projection.
        vel_ship : float
            Constant speed of the ship.
        lat_column_name : str, optional
            Name of the latitude column, by default "lat"
        lon_column_name : str, optional
            Name of the longitude column, by default "lon"
        time_stamp_column_name : str, optional
            Name of the time stamps column, by default "ts"
        """
        lat_column_name = lat_column_name.strip()
        lon_column_name = lon_column_name.strip()
        time_stamp_column_name = time_stamp_column_name.strip()

        df.columns = df.columns.str.strip()

        lons = np.array(df[lon_column_name].values)
        lats = np.array(df[lat_column_name].values)
        time_stamps = np.array(df[time_stamp_column_name].values)

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
            A new route object
        """
        assert all(
            [r.vel_ship == routes[0].vel_ship for r in routes]
        ), "The velocities of the routes are not the same"

        # TODO: Create a similar assert for ocean_data

        if remove_mid_point:
            lats = np.concatenate([route.lats[:-1] for route in routes])
            lons = np.concatenate([route.lons[:-1] for route in routes])
            time_stamps = np.concatenate([route.time_stamps[:-1] for route in routes])

            lats = np.append(lats, routes[-1].lats[-1])
            lons = np.append(lons, routes[-1].lons[-1])
            time_stamps = np.append(time_stamps, routes[-1].time_stamps[-1])
        else:
            lats = np.concatenate([route.lats for route in routes])
            lons = np.concatenate([route.lons for route in routes])
            time_stamps = np.concatenate([route.time_stamps for route in routes])

        vel_ship = routes[0].vel_ship
        ocean_data = routes[0].ocean_data
        return cls(lats, lons, time_stamps, vel_ship, ocean_data)

    def __len__(self):
        """Return the number of points in the route."""
        return len(self.lats)

    def __getitem__(self, idx: int) -> "Route":
        """Return a sub-route or point selection by index."""
        return Route(
            self.lats[idx],
            self.lons[idx],
            self.time_stamps[idx],
            self.vel_ship,
            self.ocean_data,
        )

    def split_by_coords(self, lat: float, lon: float) -> tuple["Route"]:
        """Split the route into two routes at the point nearest to coords.

        Parameters
        ----------
        lat : float
            Latitude to split at.
        lon : float
            Longitude to split at.
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
        """Recompute timestamps given a new start time.

        Parameters
        ----------
        time_start : np.datetime64
            The new starting time
        land_penalization : float, optional
            Time penalization (in seconds) for every point of the route
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
        """Compute the distance per segment using the haversine formula.

        Returns
        -------
        np.array(float)
            Distance per segment in meters
        """
        distances = distance_from_points(
            lat_start=self.lats[:-1],
            lon_start=self.lons[:-1],
            lat_end=self.lats[1:],
            lon_end=self.lons[1:],
        )
        return distances

    @property
    def total_distance(self) -> float:
        """Compute the total distance of the route.

        Returns
        -------
        float
            Total distance on the path in meters
        """
        return np.sum(self.distance_per_segment)

    @property
    def time_per_segment(self) -> np.ndarray:
        """Return time per segment in seconds."""
        cum_diff = self.time_stamps[1:] - self.time_stamps[:-1]
        return cum_diff / np.timedelta64(1, "s")

    @property
    def total_time(self) -> float:
        """Compute the total travel time of the route in seconds."""
        time_stamps = self.time_stamps
        time = (time_stamps[-1] - time_stamps[0]) / np.timedelta64(1, "s")
        return time

    @property
    def ground_vel_per_segment(self) -> np.ndarray:
        """Compute ground velocity per segment.

        Returns
        -------
        np.array(float)
            Ground velocity per segment
        """
        distances = self.distance_per_segment
        time_per_segment = self.time_per_segment
        return distances / time_per_segment

    @property
    def heading(self) -> np.ndarray:
        """Compute ground heading per segment in degrees.

        Returns
        -------
        np.array(float)
            Ground heading per segment in degrees.
        """
        return angle_from_points(
            self.lats[:-1], self.lons[:-1], self.lats[1:], self.lons[1:]
        )

    def insert_point(self, lat: float, lon: float, time_stamp: np.datetime64 = None):
        """Insert a point into the route, optionally at a specified timestamp.

        If `time_stamp` is None the timestamp is computed from the last point.

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
                vel_ship=self.vel_ship,
                ocean_data=self.ocean_data,
            )
            time = self.time_stamps[-1] + float(routes[0]) * np.timedelta64(1, "s")
            self.lats = np.append(self.lats, lat)
            self.lons = np.append(self.lons, lon)
            self.time_stamps = np.append(self.time_stamps, time)
        else:
            relative_time_stamp = self.time_stamps - time_stamp
            time_stamp_idx = np.argmin(np.abs(relative_time_stamp))
            self.time_stamps = np.insert(self.time_stamps, time_stamp_idx, time_stamp)
            self.lats = np.insert(self.lats, time_stamp_idx, lat)
            self.lons = np.insert(self.lons, time_stamp_idx, lon)

    def get_ocean_data(self) -> dict:
        """Return interpolated ocean data for this route."""
        return self.ocean_data.get_data(
            lon=self.lons, lat=self.lats, time=self.time_stamps
        )

    @property
    def beaufort_value_per_segment(self) -> np.ndarray:
        """Compute the beaufort value per segment for the route.

        Returns
        -------
        np.ndarray
            An array of beaufort values for the route segments.
        """
        return self.ocean_data.get_beaufort(
            self.lons, self.lats, self.time_stamps, use_wind=False
        )

    def as_dict(self, all_data: bool = False, serializable: bool = False) -> dict:
        """Return the route data as a dictionary.

        Parameters
        ----------
        all_data : bool, optional
            If True include full computed data; if False include coords and
            timestamps only.
        serializable : bool, optional
            If True convert arrays and timestamps into serializable types.

        Returns
        -------
        dict
            Dictionary containing the route data.
        """
        dict_route = {
            "lat": self.lats,
            "lon": self.lons,
            "ts": self.time_stamps,
            "land": self.land,
        }
        if all_data:
            # Time per segment and velocity over ground and
            time = np.concatenate([[0], self.time_per_segment]).astype(int)
            vel_gr = np.concatenate([self.ground_vel_per_segment, [0]])
            heading = np.concatenate([self.heading, [0]])
            distance = np.concatenate([[0], self.distance_per_segment.flatten()])
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
        """Return the route as a pandas DataFrame."""
        return pd.DataFrame(self.as_dict(all_data=all_data))

    def export_to_csv(
        self, path_to_file: str, all_data: bool = False, extra: dict | None = None
    ):
        """Write the route to a CSV file.

        Parameters
        ----------
        path_to_file : str
            Path to file
        """
        df = self.as_dataframe(all_data=all_data)
        if extra is not None:
            df = df.assign(**extra)
        df.to_csv(path_to_file, index=False, float_format="%.3f")

    def _even_reparametrization(
        self,
        cost_per_segment: float = None,
        n_points: int = None,
        n_iter: int = 5,
        verbose: bool = False,
    ):
        """Wrap even_reparametrization to use route times as cost function.

        The wrapper adapts the route to the expected input of
        `even_reparametrization` and uses the route time computation as the
        cost function.
        """
        curve = np.stack((self.lons, self.lats), axis=1)
        date_start = self.time_stamps[0]

        def cost(curve):
            latitudes = curve[:, 1]
            longitudes = curve[:, 0]
            ts = compute_times_given_coordinates_and_start(
                latitudes,
                longitudes,
                date_start,
                self.vel_ship,
                self.ocean_data,
                land_penalization=0,
            )[0]

            cum_diff = ts[1:] - ts[:-1]

            return (cum_diff).astype(float)

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
        return self.from_start_time(
            lats,
            lons,
            self.time_stamps[0],
            vel_ship=self.vel_ship,
            ocean_data=self.ocean_data,
            land_penalization=0,
        )

    def reparametrize_to_fixed_cost(
        self, cost_per_segment: float, n_iter: int = 5, verbose: bool = False
    ) -> "Route":
        """Reparameterize the route so segment times are equal.

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
        """Reparameterize the route to have a fixed number of waypoints.

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
        """Reparameterize the route so segments are below a distance threshold.

        The reparameterization never removes original points; it only inserts
        intermediate points where needed.

        Parameters
        ----------
        threshold : float
            The maximum distance between two points, in meters.

        Returns
        -------
        Route
            A new Route object after reparametrization
        """
        lats, lons = split_route_segments(self.lats, self.lons, threshold=threshold)
        return self.from_start_time(
            lats,
            lons,
            self.time_stamps[0],
            vel_ship=self.vel_ship,
            ocean_data=self.ocean_data,
            land_penalization=0,
        )

    def feasibility_check(
        self,
        vel_expected: float = None,
        tol: float = 0.1,
    ) -> bool:
        """Check whether the trajectory is feasible.

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

        distance = distance_from_points(lat_start, lon_start, lat_end, lon_end)
        current_v_proj, current_u_proj = compute_currents_projection(
            lat_start,
            lon_start,
            lat_end,
            lon_end,
            ts_start,
            ocean_data=self.ocean_data,
        )

        # Compute the time
        times = np.diff(self.time_stamps).astype("timedelta64[s]").astype(float)

        vel_real = distance / times

        vel_ship = np.sqrt((vel_real - current_u_proj) ** 2 + current_v_proj**2)

        return np.all(np.abs(vel_ship - vel_expected) < (tol * vel_expected))
