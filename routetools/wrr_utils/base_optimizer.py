from abc import ABC, abstractmethod
from datetime import datetime

import xarray as xr

from routetools.wrr_utils.route import Route


class BaseOptimizer(ABC):
    """Base class for optimizers."""

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def optimize(
        self,
        lat_start: float,
        lon_start: float,
        lat_end: float,
        lon_end: float,
        data: xr.Dataset,
        date_start: datetime,
        vel_ship: float,
    ) -> Route:
        """Optimize the route from start to end coordinates.

        Parameters
        ----------
        lat_start : float
            Starting latitude of the route
        lon_start : float
            Starting longitude of the route
        lat_end : float
            Ending latitude of the route
        lon_end : float
            Ending longitude of the route
        data : xr.Dataset
            Ocean data to be used for optimization
        date_start : datetime
            Starting date of the route
        vel_ship : float
            Speed of the ship in m/s
        """
        pass

    @abstractmethod
    def last_optimization_summary(self) -> dict:
        """Return a summary of the last optimization."""
        return {}
