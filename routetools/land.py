from functools import partial
from math import ceil

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from jax.scipy.ndimage import map_coordinates
from perlin_numpy import generate_perlin_noise_2d as pn2d


class Land:
    """Class to check if points on a curve are on land."""

    def __init__(
        self,
        xlim: tuple[float, float],
        ylim: tuple[float, float],
        water_level: float = 0.7,
        resolution: int | tuple[int, int] | None = None,
        interpolate: int = 100,
        outbounds_is_land: bool = False,
        random_seed: int | None = None,
        land_array: jnp.ndarray | None = None,
    ):
        """Class to check if points on a curve are on land.

        Parameters
        ----------
        x : jnp.ndarray
            array of x axis values
        y : jnp.ndarray
            array of y axis values
        water_level : float, optional
            the threshold value to determine land from water, by default 0.7
        resolution : int | tuple, optional
            resolution of the noise, or density of the land. Each entry must be divisors
            of the length of x and y respectively. Higher resolution generates more
            detailed land, by default (1, 1)
        interpolate : int, optional
            The number of times to interpolate the curve, by default 100
        outbounds_is_land : bool, optional
            if True, points outside the limits are considered land, by default False
        random_seed : int, optional
            random seed for reproducibility, by default None
        """
        # Ensure resolution is 2D
        if resolution is None:
            resolution = (1, 1)
        elif isinstance(resolution, int):
            resolution = (resolution, resolution)
        elif len(resolution) != 2:
            raise ValueError(
                f"""
                Resolution must be a tuple of length 2, not {len(resolution)}
                """
            )

        # Random seed
        rng = np.random.default_rng(random_seed) if random_seed is not None else None

        # Generate land
        lenx = ceil(xlim[1] - xlim[0]) * resolution[0]
        leny = ceil(ylim[1] - ylim[0]) * resolution[1]
        if land_array is None:
            land = pn2d((lenx, leny), res=resolution, rng=rng)
        else:
            land = jnp.array(land_array)
            if land.shape != (lenx, leny):
                raise ValueError(
                    f"""
                    The provided land array has shape {land.shape}, but the expected
                    shape is {(lenx, leny)}. Please provide an array with the correct
                    shape or set resolution to None.
                    """
                )
        # Normalize land between 0 and 1
        land = (land - jnp.min(land)) / (jnp.max(land) - jnp.min(land))
        # No land should be absolutely 0
        land = jnp.clip(land, 1e-6, 1)

        # Store the class properties
        self._array = jnp.array(land)
        self.x = jnp.linspace(*xlim, lenx)
        self.y = jnp.linspace(*ylim, leny)
        self.xmin = xlim[0]
        self.xmax = xlim[1]
        self.xnorm = (self._array.shape[0] - 1) / (self.xmax - self.xmin)
        self.ymin = ylim[0]
        self.ymax = ylim[1]
        self.ynorm = (self._array.shape[1] - 1) / (self.ymax - self.ymin)
        self.resolution = resolution
        self.random_seed = random_seed
        self.water_level = water_level
        self.shape = self._array.shape
        self.interpolate = interpolate
        self.outbounds_is_land = outbounds_is_land

    @property
    def array(self) -> jnp.ndarray:
        """Return a boolean array indicating land presence."""
        return jnp.asarray((self._array > self.water_level).astype(int))

    @partial(jit, static_argnums=(0,))
    def _check_nointerp(self, curve: jnp.ndarray) -> jnp.ndarray:
        """
        Check if points on a curve are on land using bilinear interpolation.

        Parameters
        ----------
        curve : jnp.ndarray
            a batch of curves (an array of shape 2 or L x 2 or W x L x 2)

        Returns
        -------
        jnp.ndarray
            a boolean array of shape (1,) or (L,) or (W, L) indicating
            if each point is on land
        """
        # Extract x and y coordinates from the curve
        x_coords = curve[..., 0]
        y_coords = curve[..., 1]

        # Shift the coordinates to start at the limits
        x_norm = (x_coords - self.xmin) * self.xnorm
        y_norm = (y_coords - self.ymin) * self.ynorm

        # Use bilinear interpolation to check if the points are on land
        land_values = map_coordinates(
            self._array, [x_norm, y_norm], order=0, mode="nearest"
        )

        # Return a boolean array where land_values > 0 indicates land
        is_land = jnp.asarray(land_values > self.water_level)

        # Find points outside the limits
        if self.outbounds_is_land:
            is_out = (
                (x_coords < self.xmin)
                | (x_coords > self.xmax)
                | (y_coords < self.ymin)
                | (y_coords > self.ymax)
            )
            is_land = is_land | is_out
        return jnp.clip(is_land, 0, 1).astype(bool)

    @partial(jit, static_argnums=(0,))
    def _check_interp(self, curve: jnp.ndarray) -> jnp.ndarray:
        """
        Check if points on a curve are on land using bilinear interpolation.

        Parameters
        ----------
        curve : jnp.ndarray
            a batch of curves (an array of shape L x 2 or W x L x 2)

        Returns
        -------
        jnp.ndarray
            a boolean array of shape (L,) or (W, L) indicating if each point is on land
        """
        n = self.interpolate

        # Interpolate x times to check if the curve passes through land
        curve_new = jnp.repeat(curve, n + 1, axis=0)
        left = jnp.concatenate([jnp.arange(n + 2, 1, -1)] * (curve.shape[0] - 1))
        right = jnp.concatenate([jnp.arange(0, n + 1, 1)] * (curve.shape[0] - 1))
        left = curve_new[: -(n + 1), :] * left[:, None]
        right = curve_new[(n + 1) :, :] * right[:, None]
        interp = (left + right) / (n + 2)
        curve_new = curve_new.at[: -(n + 1)].set(interp)[:-n, :]

        # Extract x and y coordinates from the curve
        x_coords = curve_new[..., 0]
        y_coords = curve_new[..., 1]

        # Shift the coordinates to start at the limits
        x_norm = (x_coords - self.xmin) * self.xnorm
        y_norm = (y_coords - self.ymin) * self.ynorm

        # Use bilinear interpolation to check if the points are on land
        land_values = map_coordinates(
            self._array, [x_norm, y_norm], order=0, mode="nearest"
        )

        # Return a boolean array where land_values > 0 indicates land
        is_land = jnp.asarray(land_values > self.water_level)

        # Find points outside the limits
        if self.outbounds_is_land:
            is_out = (
                (x_coords < self.xmin)
                | (x_coords > self.xmax)
                | (y_coords < self.ymin)
                | (y_coords > self.ymax)
            )
            is_land = is_land | is_out

        # Interpolate back to the original size
        is_land = jnp.convolve(is_land, jnp.ones(n + 1), mode="full")[:: n + 1]
        # When a point is on land, mark neighbors too
        is_land = jnp.convolve(is_land, jnp.ones(3), mode="same") > 0
        return jnp.clip(is_land, 0, 1).astype(bool)

    @partial(jit, static_argnums=(0,))
    def __call__(self, curve: jnp.ndarray) -> jnp.ndarray:
        """
        Check if points on a curve are on land.

        Parameters
        ----------
        curve : jnp.ndarray
            a batch of curves (an array of shape W x L x 2)

        Returns
        -------
        jnp.ndarray
            a boolean array of shape (W, L) indicating if each point is on land
        """
        is_land: jnp.ndarray
        if curve.ndim == 1:
            is_land = self._check_nointerp(curve)
        elif curve.ndim == 2:
            if self.interpolate == 0:
                is_land = self._check_nointerp(curve)
            else:
                is_land = self._check_interp(curve)
        else:
            if self.interpolate == 0:
                is_land = jax.vmap(self._check_nointerp)(curve)
            else:
                is_land = jax.vmap(self._check_interp)(curve)
        return is_land

    def penalization(self, curve: jnp.ndarray, penalty: float) -> jnp.ndarray:
        """
        Return an array indicating land presence, in one of two versions.

        (A) (no penalty) A boolean array indicating if the curve passes through land.
        (B) (penalty) the sum of the number of points on land times the penalty.

        Parameters
        ----------
        land_function : Callable[[jnp.ndarray], jnp.ndarray] | None, optional
            A function that checks if points on a curve are on land, by default None
        curve : jnp.ndarray
            A batch of curves (an array of shape W x L x 2)
        penalty : float
            The penalty for passing through land.
        """
        # Check if the curve passes through land
        is_land = self(curve)

        # Consecutive points on land count as one
        is_land = jnp.diff(is_land, axis=1) != 0

        # Return the sum of the number of land intersections times the penalty
        return jnp.sum(is_land, axis=1) * penalty
