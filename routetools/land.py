from functools import partial
from math import ceil

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from jax.scipy.ndimage import map_coordinates
from perlin_numpy import generate_perlin_noise_2d as pn2d

from routetools._cost.haversine import haversine_meters_components


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
        penalize_segments: bool = True,
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
        penalize_segments : bool, optional
            If True, penalize segments, not points, by default True
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
            # Normalize land between 0 and 1
            land = (land - jnp.min(land)) / (jnp.max(land) - jnp.min(land))
            # No land should be absolutely 0
            land = jnp.clip(land, 1e-6, 1)
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
        self.penalize_segments = penalize_segments

        land_indices = jnp.argwhere(self._array)
        self._lats = self.y[land_indices[:, 1]]
        self._lons = self.x[land_indices[:, 0]]

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
        if self.penalize_segments:
            is_land = jnp.diff(is_land, axis=1) != 0

        # Return the sum of the number of land intersections times the penalty
        return jnp.sum(is_land, axis=1) * penalty

    def distance_to_land(
        self, curve: jnp.ndarray, haversine: bool = False
    ) -> jnp.ndarray:
        """
        Compute the distance from each point on the curve to the nearest land point.

        Parameters
        ----------
        curve : jnp.ndarray
            A batch of curves (an array of shape W x L x 2)
        haversine : bool, optional
            If True, compute distances using the haversine formula, by default False

        Returns
        -------
        jnp.ndarray
            An array of shape (W, L) with the distance to the nearest land point.
        """

        # Define a function to compute the distance of a single point to land
        def point_distance_to_land(point: jnp.ndarray) -> jnp.ndarray:
            # Compute the haversine distance from the point to all land points
            if haversine:
                dx, dy = haversine_meters_components(
                    point[1], point[0], self._lats, self._lons
                )
            else:
                dx = self._lats - point[1]
                dy = self._lons - point[0]
            dists = jnp.sqrt(dx**2 + dy**2)
            # Find the minimum distance
            dist = jnp.min(dists)
            # Check inside land
            is_in = self(point)
            dist = jnp.where(is_in, 0.0, dist)
            # Check out of bounds
            if self.outbounds_is_land:
                is_out = (
                    (point[0] < self.xmin)
                    | (point[0] > self.xmax)
                    | (point[1] < self.ymin)
                    | (point[1] > self.ymax)
                )
                dist = jnp.where(is_out, 0.0, dist)
            # Return the minimum distance
            return dist

        # Vectorize the function over the curve points
        vectorized_distance = jax.vmap(
            jax.vmap(point_distance_to_land, in_axes=0), in_axes=0
        )
        # If curve has shape (L, 2), add a batch dimension
        if curve.ndim == 2:
            curve = curve[None, :, :]
            return vectorized_distance(curve)[0]
        else:
            return vectorized_distance(curve)

    def cost_function(
        self,
        vectorfield: None,
        curve: jnp.ndarray,
        max_distance: float = 50000,  # meters
        neighbor_penalty: float = 0.0,
        **kwargs,
    ) -> jnp.ndarray:
        """Penalizes being close to land."""
        # Compute distance to land
        distance = self.distance_to_land(curve, haversine=True)
        # Set a maximum distance for cost calculation
        distance = jnp.clip(distance, a_min=0.0, a_max=max_distance)
        # Cost is inverse of distance
        cost = max_distance / (distance + 1e-6)

        # Also penalize distance between points (in km)
        lats = curve[:, :, 1]
        lons = curve[:, :, 0]
        dx, dy = haversine_meters_components(
            lats[:, 1:], lons[:, 1:], lats[:, :-1], lons[:, :-1]
        )
        dist = jnp.sqrt(dx**2 + dy**2) / max_distance

        return jnp.sum(cost, axis=1) + neighbor_penalty * jnp.sum(dist, axis=1)


def move_curve_away_from_land(
    curve: jnp.ndarray,
    land: Land,
    step_size: float = 1000.0,
    distance_from_land: float = 0,
    spherical_correction: bool = False,
) -> jnp.ndarray:
    """Move a curve away from land by a specified step size.

    This function assumes the curve is marginally on land
    and tries to move it away iteratively.
    It will not work well if the curve is deeply on land,
    as it may get stuck in a local minimum. In that case,
    consider using a more sophisticated optimization algorithm.
    """
    # Ensure we are working with a single curve of shape (L, 2)
    if curve.ndim == 3 and curve.shape[0] == 1:
        curve = curve[0]
    elif curve.ndim != 2 or curve.shape[1] != 2:
        raise ValueError(
            f"""
            Curve must be of shape (L, 2) or (1, L, 2), but got {curve.shape}
            """
        )
    # Find all points of the curve that are on land
    is_land = land(curve)
    # Also the points close to land, if distance_from_land > 0
    if distance_from_land > 0:
        distance = land.distance_to_land(curve, haversine=spherical_correction)
        is_close = distance < distance_from_land
        is_land = is_land | is_close
    if not jnp.any(is_land):
        return curve  # No points on land, return the original curve
    # We will start an iterative process to move points away from land
    idx_land = jnp.argwhere(is_land)
    curve_new = curve.copy()
    for idx in idx_land:
        # Open a radius around the point and find the nearest point
        # not on land
        point = curve[idx]
        radius = step_size  # in meters
        niter = 0
        while niter < 10:
            # Create a grid of points around the current point
            angles = jnp.linspace(0, 2 * jnp.pi, num=16, endpoint=False)
            offsets = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1) * radius
            candidates = point + offsets
            # Check which candidates are on land
            is_candidate_land = land(candidates)
            # Also check distance to land for candidates, if distance_from_land > 0
            if distance_from_land > 0:
                distance = land.distance_to_land(
                    candidates, haversine=spherical_correction
                )
                is_candidate_close = distance < distance_from_land
                is_candidate_land = is_candidate_land | is_candidate_close
            if not jnp.any(is_candidate_land):
                # If no candidates are on land, move to the nearest candidate
                distances = jnp.linalg.norm(candidates - point, axis=1)
                nearest_idx = jnp.argmin(distances)
                curve_new = curve_new.at[idx].set(candidates[nearest_idx])
                break
            else:
                # If some candidates are on land, increase the radius and try again
                radius += step_size
            niter += 1
    return curve_new
