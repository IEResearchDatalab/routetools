from abc import ABC, abstractmethod

import numpy as np
import scipy
import xarray as xr


class Interpolator(ABC):
    """Abstract base class for dataset interpolators."""

    @abstractmethod
    def __init__(
        self,
        ds: xr.Dataset,
        vars: tuple[str] = ("vo", "uo"),
        land_penalization: float = 0,
        *kwargs,
    ):
        """Prepare an interpolator for the given dataset.

        Parameters
        ----------
        ds : xr.Dataset
            The dataset containing the data to interpolate
        vars : Tuple[str]
            The variables to be interpolated
        land_penalization : float, optional
            The value to be used to penalize land cells, by default 0
            If the value is 0, then the land cells are not penalized.
            land_penalization can be used only if the dataset has the
            `land` variable.
        """
        pass

    @abstractmethod
    def interpolate(
        self, lat: np.ndarray, lon: np.ndarray, ts: np.ndarray
    ) -> np.ndarray:
        """
        Interpolate the data at the specified (lat, lon, ts) positions.

        Parameters
        ----------
        lat : float
            The specified latitude of the location of the ship
        lon : float
            The specified longitude of the location of the ship
        ts: np.ndarray
            An array of timestamps in of type np.datetime64 format

        Returns
        -------
        Tuple[Tuple[np.ndarray]]
            An array of length 3, with interpolated data at the specified (lat, lon)
            and respective deriviatives in pairs.
        """
        pass


class EvenLinearInterpolator(Interpolator):
    """Even-spacing linear interpolator implementation."""

    def __init__(
        self,
        ds: xr.Dataset,
        vars: tuple[str] = ("vo", "uo"),
        order: int = 0,
        land_penalization: float = 0,
    ):
        """Prepare an even-spacing interpolator for the given dataset.

        Parameters
        ----------
        ds : xr.Dataset
            The dataset containing the data to interpolate
        vars : Tuple[str]
            The variables to be interpolated
        order : int, optional
            The order of the spline interpolation, by default 1
            This value must be between 0 and 5
        land_penalization : float, optional
            The value to be used to penalize land cells, by default 0
            If the value is 0, then the land cells are not penalized.
            land_penalization can be used only if the dataset has the
            `land` variable.
        """
        data = []
        for var in vars:
            if land_penalization > 0:
                ds[var] = ds[var].where(ds["land"] == 0, land_penalization)

            data.append(ds[var].values)

        self.vars = vars

        self.date_start = ds.coords["time"].values[0]

        axes = [
            (ds.coords["time"] - self.date_start) / np.timedelta64(1, "s"),
            ds.coords["latitude"].values.astype(np.float32),
            ds.coords["longitude"].values.astype(np.float32),
        ]

        assert len(axes) == data[0].ndim  # FIXME check all axes

        self.order = order
        self.data = data
        self.begin = np.zeros(len(axes))
        self.spacing = np.zeros(len(axes))
        for i, var in enumerate(axes):
            if len(var) > 1 and not np.allclose(np.std(np.diff(var)), 0, atol=1e-4):
                raise ValueError(f"{var} is not regularly spaced")

            self.begin[i] = var[0]
            self.spacing[i] = (var[-1] - var[0]) / (len(var) - 1)

            if np.isnan(self.spacing[i]):
                self.spacing[i] = self.begin[i]

    def interpolate(
        self, lat: np.ndarray, lon: np.ndarray, ts: np.ndarray
    ) -> np.ndarray:
        """Interpolate the data at the specified (lat, lon, ts) positions.

        Parameters
        ----------
        lat : float
            The specified latitude of the location of the ship
        lon : float
            The specified longitude of the location of the ship
        ts: np.ndarray
            An array of timestamps in of type np.datetime64 format

        Returns
        -------
        Tuple[Tuple[np.ndarray]]
            An array of length 3, with interpolated data at the specified (lat, lon)
            and respective deriviatives in pairs.
        """
        # If ts is datetime, convert to seconds
        if np.issubdtype(ts.dtype, np.datetime64):
            ts = (ts - self.date_start) / np.timedelta64(1, "s")
        else:
            # If it is float, assume it is in seconds
            ts = ts.astype(np.float32)
        x = np.array([ts, lat, lon]).T

        assert x.ndim == 2
        assert x.shape[1] == self.data[0].ndim

        coords = (x - self.begin[None, :]) / self.spacing[None, :]

        output = []
        for d in self.data:
            output.append(
                scipy.ndimage.map_coordinates(
                    d, coords.T, order=self.order, mode="wrap"
                )
            )

        return np.array(output)
