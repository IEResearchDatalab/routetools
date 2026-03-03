from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np

from routetools.wrr_bench.ocean import Ocean
from routetools.wrr_utils.route import Route


class FuelConsumption:  # type: ignore
    """Placeholder FuelConsumption used when the real module is absent."""

    pass


class Interpolator:  # type: ignore
    """Placeholder Interpolator used when the real module is absent."""

    def __init__(self):
        pass


def vectorfield_circular(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    intensity: float = -0.9,
    centre: tuple[float, float] = (0, 0),
) -> tuple[np.ndarray, np.ndarray]:
    """Circular vector field derived from Techy 2011.

    doi.org/10.1007/s11370-011-0092-9

    Parameters
    ----------
    x : np.ndarray
        x-coordinates of the points
    y : np.ndarray
        y-coordinates of the points
    data : Ocean
        Ocean data

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple of the x and y components of the vector field
    """
    x0, y0 = centre
    u = -intensity * (y - y0)
    v = intensity * (x - x0)

    return u, v


def vectorfield_fourvortices(
    x: np.ndarray, y: np.ndarray, t: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vector field with four vortices.

    Source:
    https://doi.org/10.1016/j.ifacol.2021.11.097

    Parameters
    ----------
    x : np.ndarray
        x-coordinates of the points
    y : np.ndarray
        y-coordinates of the points
    data : Ocean
        Ocean data

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple of the x and y components of the vector field
    """

    def Ru(a: float, b: float) -> np.ndarray:
        return 1 / (3 * ((x - a) ** 2 + (y - b) ** 2) + 1) * -(y - b)

    u = 1.7 * (-Ru(2, 2) - Ru(4, 4) - Ru(2, 5) + Ru(5, 1))

    def Rv(a: float, b: float) -> np.ndarray:
        return 1 / (3 * ((x - a) ** 2 + (y - b) ** 2) + 1) * (x - a)

    v = 1.7 * (-Rv(2, 2) - Rv(4, 4) - Rv(2, 5) + Rv(5, 1))
    return u, v


def vectorfield_swirlys(
    x: np.ndarray, y: np.ndarray, t: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vector field with periodic behaviour.

    Source:
    https://doi.org/10.1016/j.ifacol.2021.11.097
    """
    u = np.cos(2 * x - y - 6)
    v = 2 / 3 * np.sin(y) + x - 3
    return u, v


def vectorfield_techy(
    x: np.ndarray, y: np.ndarray, t: np.ndarray, sink: float = -0.3
) -> tuple[np.ndarray, np.ndarray]:
    """Time-varying flow with a sink and a time-varying vortex.

    References
    ----------
    "Optimal navigation in planar time-varying flow: Zermelo's problem revisited"
    Techy 2011, Fig 12a-c

    "Visir-1.b: ocean surface gravity waves and currents for energy-efficient
    navigation" Mannarini 2019, Fig 2b
    """
    vortex = t - 0.5
    u = sink * x - vortex * y
    v = vortex * x + sink * y
    return u, v


class InterpolatorVF(Interpolator):
    """Interpolator adapter for synthetic vector fields."""

    def __init__(
        self,
        fun: callable,
        t0: np.datetime64 | None = None,
        **kwargs,
    ):
        """Create an interpolator for a vector field.

        Parameters
        ----------
        fun : callable
            The function to interpolate
        t0 : Optional[np.datetime64]
            Reference timestamp for interpolation. If None, a default date is used.
        **kwargs
            Additional arguments to be passed to the function
        """
        self.fun = lambda x, y, t: fun(x, y, t, **kwargs)
        self.t0 = np.datetime64("2023-01-01", "ns") if t0 is None else t0
        self.vars = ["vo", "uo"]

    def interpolate(
        self, lat: np.ndarray, lon: np.ndarray, ts: np.ndarray
    ) -> np.ndarray:
        """
        Interpolate the data at the specified (lat, lon, ts) positions.

        Parameters
        ----------
        lat : np.ndarray
            Array of latitudes (Y-coordinates)
        lon : np.ndarray
            Array of longitudes (X-coordinates)
        ts : np.ndarray
            Array of timestamps

        Returns
        -------
        np.ndarray
            Array of the interpolated vector field at the specified (lat, lon)
        """
        # Turn timestamps into float
        t = ((ts - self.t0) / np.timedelta64(1, "s")).astype(float)
        u, v = self.fun(lon, lat, t)

        return np.array([v, u])


class OceanSynthetic(Ocean):
    """Synthetic Ocean that exposes `get_currents` for testing and plotting."""

    def __init__(
        self,
        vectorfield: str = "circular",
        date_start: np.datetime64 | None = None,
        bounding_box: Iterable = None,
        **kwargs,
    ):
        """Create a synthetic ocean with a given vector field.

        Parameters
        ----------
        vectorfield : str, optional
            The vector field to use, by default "circular"
        date_start : Optional[np.datetime64], optional
            Reference start date for the vector field, by default None.
        bounding_box : Iterable, optional
            The bounding box of the ocean (ymin, xmin, ymax, xmax), by default None
        **kwargs
            Additional arguments to be passed to the vector field function
        """
        if date_start is None:
            date_start = np.datetime64("2023-01-01", "ns")
        fun: callable = eval("vectorfield_" + vectorfield)
        currents_interpolator = InterpolatorVF(fun, t0=date_start, **kwargs)
        self.date_start = date_start

        if bounding_box is None:
            bounding_box = (-10, 10, -10, 10)

        super().__init__(
            currents_interpolator=currents_interpolator,
            radius=None,
            bounding_box=bounding_box,
            land_file=None,
            use_ice=False,
        )

    def plot(self, t: np.datetime64 | None = None, step: float = 1, order: str = "xy"):
        """Plot the vector field.

        Parameters
        ----------
        step : float, optional
            Step size for the grid, by default 1
        """
        ymin, xmin, ymax, xmax = self.bounding_box
        x = np.arange(xmin, xmax, step)
        y = np.arange(ymin, ymax, step)
        X, Y = np.meshgrid(x, y)
        t = self.date_start if t is None else t
        T = np.tile(t, X.shape).astype("datetime64[ns]")

        V, U = self.get_currents(Y, X, T)

        if order == "xy":
            plt.quiver(X, Y, U, V)
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
        else:
            plt.quiver(Y, X, V, U)
            plt.xlim(ymin, ymax)
            plt.ylim(xmin, xmax)

        # Axis proportions
        plt.gca().set_aspect("equal", adjustable="box")


class FuelConsumptionSynthetic(FuelConsumption):
    """Toy FuelConsumption implementation for synthetic tests."""

    def __init__(self):
        pass

    def required_fuel_from_coordinates(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        times: np.ndarray,
        vel_ship: float | np.ndarray,
        ocean_data: Ocean,
        land_penalization: float,
    ) -> np.ndarray:
        """Compute required fuel given arrays of coordinates and times.

        Returns an array of fuel costs per segment.
        """
        assert (
            lats.shape == lons.shape == times.shape
        ), f"{lats.shape}, {lons.shape}, {times.shape}"

        dt = np.diff(times).astype("timedelta64[us]").astype(float) / 1e6
        assert (dt > 0).all(), "Time stamps must be increasing"
        # Compute cost from currents
        u, v = ocean_data.get_currents(lats[:, :-1], lons[:, :-1], times[:, :-1])
        sogx = np.diff(lons, axis=1) / dt
        sogy = np.diff(lats, axis=1) / dt
        vel_ship = np.power(sogx - u, 2) + np.power(sogy - v, 2)
        cost = 1 / 2 * vel_ship * dt
        return cost

    def required_fuel_from_route(self, route_obj: Route, land_penalization: float = 0):
        """Compute required fuel for a `Route` object.

        Wrapper around `required_fuel_from_coordinates`.
        """
        return self.required_fuel_from_coordinates(
            route_obj.lats[None, :],
            route_obj.lons[None, :],
            route_obj.time_stamps[None, :],
            route_obj.vel_ship[None, :],
            route_obj.ocean_data,
            land_penalization,
        )
