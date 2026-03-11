import heapq
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
        map_mode: str = "nearest",
        map_order: int = 0,
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
        map_mode : str, optional
            The mode to use for `scipy.ndimage.map_coordinates`, by default "nearest".
            This determines how points outside the array are handled.
        map_order : int, optional
            The order of the spline interpolation,
            by default 0. 0 for nearest neighbor, 1 for bilinear.
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

        self._map_mode = map_mode
        self._map_order = map_order

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
            self._array, [x_norm, y_norm], order=self._map_order, mode=self._map_mode
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
            self._array, [x_norm, y_norm], order=self._map_order, mode=self._map_mode
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
    step_size: float = 0.01,
) -> jnp.ndarray:
    """Move a curve away from land by a specified step size.

    Curve must be of shape (L, 2) or (1, L, 2), expressed in degrees.
    The land function must be an instance of the Land class.

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
    if not jnp.any(is_land):
        return curve  # No points on land, return the original curve
    # We will start an iterative process to move points away from land
    idx_land = jnp.argwhere(is_land)
    curve_new = curve.copy()
    for idx in idx_land:
        # Open a radius around the point and find the nearest point
        # not on land
        idx = idx[0]
        point = curve[idx]
        radius = step_size  # in degrees
        niter = 0
        while niter < 10:
            # Create a grid of points around the current point
            angles = jnp.linspace(0, 2 * jnp.pi, 16, endpoint=False)
            candidates = point + radius * jnp.stack(
                [jnp.cos(angles), jnp.sin(angles)], axis=1
            )
            # Shape of candidates is (16, 2). Turn into (16, 1, 2)
            # and then append the previous point to make it (16, 2, 2)
            candidates_extended = candidates[:, None, :]
            if idx > 0:
                point_prev = curve_new[idx - 1]
                # Replicate point_prev to match the shape of candidates
                point_prev = jnp.tile(point_prev, (candidates.shape[0], 1))
                candidates_extended = jnp.concatenate(
                    [point_prev[:, None, :], candidates_extended], axis=1
                )
            # Check which candidates are on land
            is_candidate_land = land(candidates_extended)
            is_candidate_land = jnp.any(is_candidate_land, axis=1)
            if not jnp.all(is_candidate_land):
                # If there are candidates not on land, move to the nearest one
                candidates_not_land = candidates[~is_candidate_land]
                # Compute distances to the original point
                dists = jnp.sqrt(jnp.sum((candidates_not_land - point) ** 2, axis=1))
                nearest_idx = jnp.argmin(dists)
                curve_new = curve_new.at[idx].set(candidates_not_land[nearest_idx])
                break
            else:
                # If some candidates are on land, increase the radius and try again
                radius += step_size
            niter += 1
    return curve_new


def _point_is_land(point: np.ndarray, land: Land) -> bool:
    """Return True if a single point lies on land according to ``land``."""
    return bool(np.asarray(land(np.asarray(point, dtype=float))).item())


def _segment_crosses_land(
    a: np.ndarray,
    b: np.ndarray,
    land: Land,
    oversample: int = 6,
) -> bool:
    """Return True if the segment from ``a`` to ``b`` intersects land."""
    dx = (land.xmax - land.xmin) / max(1, land.shape[0] - 1)
    dy = (land.ymax - land.ymin) / max(1, land.shape[1] - 1)
    steps = max(
        abs((b[0] - a[0]) / max(dx, 1e-12)), abs((b[1] - a[1]) / max(dy, 1e-12))
    )
    # Use dense sampling to avoid false negatives near complex coastlines.
    n_samples = max(101, int(np.ceil(oversample * steps)) + 1)

    samples = np.linspace(a, b, n_samples)
    is_land = np.asarray(land(samples), dtype=bool).reshape(-1)
    return bool(is_land[1:-1].any())


def _build_astar_grid(
    land: Land,
    astar_resolution_scale: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build an A* land-mask grid (optionally upsampled)."""
    if astar_resolution_scale < 1:
        raise ValueError("astar_resolution_scale must be >= 1")

    is_land = np.asarray(land.array, dtype=bool)
    if astar_resolution_scale > 1:
        is_land = np.kron(
            is_land,
            np.ones((astar_resolution_scale, astar_resolution_scale), dtype=bool),
        )

    x_axis = np.linspace(land.xmin, land.xmax, is_land.shape[0])
    y_axis = np.linspace(land.ymin, land.ymax, is_land.shape[1])
    return is_land, x_axis, y_axis


def _dilate_land_mask(is_land: np.ndarray, radius: int) -> np.ndarray:
    """Dilate boolean land mask by ``radius`` cells using 8-neighbour growth."""
    if radius <= 0:
        return is_land

    dilated = is_land.copy()
    for _ in range(radius):
        padded = np.pad(dilated, 1, mode="edge")
        expanded = np.zeros_like(dilated)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                expanded |= padded[
                    1 + dx : 1 + dx + dilated.shape[0],
                    1 + dy : 1 + dy + dilated.shape[1],
                ]
        dilated = expanded
    return dilated


def _point_to_cell(
    point: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
) -> tuple[int, int]:
    """Map (lon, lat) to nearest A* grid index."""
    ix = int(np.clip(np.searchsorted(x_axis, point[0]), 0, len(x_axis) - 1))
    iy = int(np.clip(np.searchsorted(y_axis, point[1]), 0, len(y_axis) - 1))

    if ix > 0 and abs(x_axis[ix - 1] - point[0]) < abs(x_axis[ix] - point[0]):
        ix -= 1
    if iy > 0 and abs(y_axis[iy - 1] - point[1]) < abs(y_axis[iy] - point[1]):
        iy -= 1

    return ix, iy


def _nearest_water_cell(
    cell: tuple[int, int],
    water: np.ndarray,
) -> tuple[int, int] | None:
    """Return nearest water cell, searching outward in square rings."""
    x0, y0 = cell
    if water[x0, y0]:
        return cell

    max_radius = max(water.shape)
    for radius in range(1, max_radius):
        xmin = max(0, x0 - radius)
        xmax = min(water.shape[0] - 1, x0 + radius)
        ymin = max(0, y0 - radius)
        ymax = min(water.shape[1] - 1, y0 + radius)

        candidates: list[tuple[int, int]] = []
        for ix in range(xmin, xmax + 1):
            for iy in range(ymin, ymax + 1):
                if ix not in (xmin, xmax) and iy not in (ymin, ymax):
                    continue
                if water[ix, iy]:
                    candidates.append((ix, iy))

        if candidates:
            return min(candidates, key=lambda c: (c[0] - x0) ** 2 + (c[1] - y0) ** 2)

    return None


def _astar_cells(
    start: tuple[int, int],
    goal: tuple[int, int],
    water: np.ndarray,
) -> list[tuple[int, int]] | None:
    """Run A* on the water grid and return the cell path from start to goal."""

    def heuristic(cell: tuple[int, int]) -> float:
        return float(np.hypot(cell[0] - goal[0], cell[1] - goal[1]))

    neighbour_steps = [
        (-1, -1, np.sqrt(2.0)),
        (-1, 0, 1.0),
        (-1, 1, np.sqrt(2.0)),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (1, -1, np.sqrt(2.0)),
        (1, 0, 1.0),
        (1, 1, np.sqrt(2.0)),
    ]

    open_heap: list[tuple[float, float, tuple[int, int]]] = []
    heapq.heappush(open_heap, (heuristic(start), 0.0, start))
    best_g: dict[tuple[int, int], float] = {start: 0.0}
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    closed: set[tuple[int, int]] = set()

    while open_heap:
        _, g_now, current = heapq.heappop(open_heap)
        if current in closed:
            continue

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        closed.add(current)

        for dx, dy, move_cost in neighbour_steps:
            nx = current[0] + dx
            ny = current[1] + dy
            if nx < 0 or ny < 0 or nx >= water.shape[0] or ny >= water.shape[1]:
                continue
            if not water[nx, ny]:
                continue

            neighbour = (nx, ny)
            tentative_g = g_now + move_cost
            if tentative_g >= best_g.get(neighbour, np.inf):
                continue

            best_g[neighbour] = tentative_g
            came_from[neighbour] = current
            heapq.heappush(
                open_heap,
                (tentative_g + heuristic(neighbour), tentative_g, neighbour),
            )

    return None


def _resample_polyline(polyline: np.ndarray, n_points: int) -> np.ndarray:
    """Resample a polyline to exactly ``n_points`` interior waypoints."""
    if n_points <= 0:
        return np.empty((0, 2), dtype=float)

    if len(polyline) < 2:
        return np.repeat(polyline[[0]], n_points, axis=0)

    seg_len = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(cumulative[-1])
    if total <= 0:
        return np.repeat(polyline[[0]], n_points, axis=0)

    targets = np.linspace(0.0, total, n_points + 2)[1:-1]
    out = np.empty((n_points, 2), dtype=float)

    j = 1
    for i, target in enumerate(targets):
        while j < len(cumulative) and cumulative[j] < target:
            j += 1
        if j >= len(cumulative):
            out[i] = polyline[-1]
            continue

        left = j - 1
        right = j
        d0 = cumulative[left]
        d1 = cumulative[right]
        if d1 <= d0:
            out[i] = polyline[right]
            continue

        alpha = (target - d0) / (d1 - d0)
        out[i] = (1.0 - alpha) * polyline[left] + alpha * polyline[right]

    return out


def _polyline_crosses_land(polyline: np.ndarray, land: Land) -> bool:
    """Return True when any segment of ``polyline`` crosses land."""
    for i in range(len(polyline) - 1):
        if _segment_crosses_land(polyline[i], polyline[i + 1], land):
            return True
    return False


def _astar_segment_replacement(
    a: np.ndarray,
    b: np.ndarray,
    n_removed: int,
    is_land: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    land: Land,
    max_safety_cells: int = 6,
) -> np.ndarray | None:
    """Compute A* replacement waypoints between two water anchors."""
    if n_removed <= 0:
        return np.empty((0, 2), dtype=float)

    for safety_cells in range(max_safety_cells + 1):
        safe_land = _dilate_land_mask(is_land, safety_cells)
        water = ~safe_land

        start_cell = _nearest_water_cell(_point_to_cell(a, x_axis, y_axis), water)
        end_cell = _nearest_water_cell(_point_to_cell(b, x_axis, y_axis), water)

        if start_cell is None or end_cell is None:
            continue

        cells = _astar_cells(start_cell, end_cell, water)
        if cells is None:
            continue

        path = np.array([[x_axis[ix], y_axis[iy]] for ix, iy in cells], dtype=float)
        path[0] = a
        path[-1] = b

        replacement = _resample_polyline(path, n_removed)
        candidate = np.vstack([a, replacement, b])
        if not _polyline_crosses_land(candidate, land):
            return replacement

    return None


def reroute_around_land(
    route: np.ndarray,
    land: Land,
    astar_resolution_scale: int = 2,
    max_passes: int = 4,
    max_anchor_expansion: int = 12,
) -> np.ndarray:
    """Replace land-crossing route runs using A* on a high-resolution land grid.

    The function keeps the original route length and edits only runs of segments
    that intersect land.

    Parameters
    ----------
    route : np.ndarray
        Waypoints as an array of shape ``(N, 2)`` with ``[lon, lat]`` columns.
    land : Land
        The original ``routetools.land.Land`` object used for land/water checks.
    astar_resolution_scale : int, optional
        Integer upsampling factor for the A* occupancy grid. Values greater than
        1 increase search resolution. Defaults to 2.
    max_passes : int, optional
        Maximum correction passes across the route. Defaults to 4.
    max_anchor_expansion : int, optional
        Maximum number of outward anchor-expansion attempts per crossing run.
        Defaults to 12.

    Returns
    -------
    np.ndarray
        Corrected route with the same shape as ``route``.
    """
    route = np.asarray(route, dtype=float)
    if route.ndim != 2 or route.shape[1] != 2:
        raise ValueError(f"route must have shape (N, 2), got {route.shape}")
    if not isinstance(land, Land):
        raise TypeError("land must be an instance of routetools.land.Land")

    n_points = len(route)
    if n_points < 2:
        return route.copy()

    is_land, x_axis, y_axis = _build_astar_grid(land, astar_resolution_scale)
    result = route.copy()

    for _ in range(max_passes):
        crossing_seg = np.zeros(n_points - 1, dtype=bool)
        for i in range(n_points - 1):
            crossing_seg[i] = _segment_crosses_land(result[i], result[i + 1], land)

        if not crossing_seg.any():
            break

        runs: list[tuple[int, int]] = []
        idx = 0
        while idx < len(crossing_seg):
            if not crossing_seg[idx]:
                idx += 1
                continue
            end = idx
            while end + 1 < len(crossing_seg) and crossing_seg[end + 1]:
                end += 1
            runs.append((idx, end))
            idx = end + 1

        for seg_start, seg_end in runs:
            base_a_idx = seg_start
            base_b_idx = seg_end + 1

            while base_a_idx > 0 and _point_is_land(result[base_a_idx], land):
                base_a_idx -= 1
            while base_b_idx < n_points - 1 and _point_is_land(
                result[base_b_idx], land
            ):
                base_b_idx += 1

            if base_a_idx >= base_b_idx:
                continue
            if _point_is_land(result[base_a_idx], land):
                continue
            if _point_is_land(result[base_b_idx], land):
                continue

            applied = False
            for expand in range(max_anchor_expansion + 1):
                anchor_a_idx = max(0, base_a_idx - expand)
                anchor_b_idx = min(n_points - 1, base_b_idx + expand)

                while anchor_a_idx > 0 and _point_is_land(result[anchor_a_idx], land):
                    anchor_a_idx -= 1
                while anchor_b_idx < n_points - 1 and _point_is_land(
                    result[anchor_b_idx], land
                ):
                    anchor_b_idx += 1

                if anchor_a_idx >= anchor_b_idx:
                    continue
                if _point_is_land(result[anchor_a_idx], land):
                    continue
                if _point_is_land(result[anchor_b_idx], land):
                    continue

                n_removed = anchor_b_idx - anchor_a_idx - 1
                if n_removed <= 0:
                    continue

                a = result[anchor_a_idx]
                b = result[anchor_b_idx]
                replacement = _astar_segment_replacement(
                    a,
                    b,
                    n_removed,
                    is_land,
                    x_axis,
                    y_axis,
                    land,
                )
                if replacement is None:
                    continue

                result[anchor_a_idx + 1 : anchor_b_idx] = replacement
                applied = True
                break

            if not applied:
                n_removed = base_b_idx - base_a_idx - 1
                if n_removed > 0:
                    result[base_a_idx + 1 : base_b_idx] = np.linspace(
                        result[base_a_idx],
                        result[base_b_idx],
                        n_removed + 2,
                    )[1:-1]

    return result
