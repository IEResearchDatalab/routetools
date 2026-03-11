"""Load ERA5 NetCDF data into JAX-compatible field closures.

The closures returned by the ``load_*`` functions conform to the interface
expected by :func:`routetools.cost.cost_function`:

- **vectorfield** ``(lon, lat, t) -> (u, v)`` — ocean current components.
  For ERA5 wind data this returns 10-m wind components, which can be used
  by a ship performance / polar model.
- **wavefield** ``(lon, lat, t) -> (height, direction)`` — significant wave
  height (m) and mean wave direction (degrees from North).
- **windfield** ``(lon, lat, t) -> (u10, v10)`` — alias for the wind loader,
  separated from "vectorfield" to make the distinction explicit.

The ``t`` coordinate represents *hours elapsed since a reference epoch*
(typically the departure time).  The loader precomputes the mapping from
absolute NetCDF timestamps to a ``[0, N_t)`` integer index so that
interpolation via ``jax.scipy.ndimage.map_coordinates`` works correctly.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from math import ceil
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr

from routetools.vectorfield import time_variant

if TYPE_CHECKING:
    from routetools.land import Land

logger = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────


def _load_dataset(path: str | Path) -> xr.Dataset:
    """Open a NetCDF file with xarray.

    Tries the ``scipy`` engine first (pure-Python, no C-library
    compatibility issues), then falls back to ``netcdf4``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ERA5 data file not found: {path}")
    last_exc: Exception | None = None
    for engine in ("scipy", "netcdf4", None):
        try:
            return xr.open_dataset(path, engine=engine)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.debug(
                "Failed to open %s with engine %r", path, engine, exc_info=True
            )
            continue
    raise RuntimeError(
        f"Failed to open ERA5 data file {path!s} with engines "
        "('scipy', 'netcdf4', None)"
    ) from last_exc


def _get_coord_name(ds: xr.Dataset, candidates: list[str]) -> str:
    """Return the first coordinate name that exists in *ds*."""
    for name in candidates:
        if name in ds.coords or name in ds.dims:
            return name
    raise KeyError(
        f"None of {candidates} found in dataset coordinates: {list(ds.coords)}"
    )


def _prepare_grid(
    ds: xr.Dataset,
    departure_time: datetime | str | np.datetime64 | None = None,
) -> dict:
    """Extract grid metadata and convert to JAX arrays.

    Returns a dict with keys:
        - ``lat``: 1-D numpy array of latitudes  (ascending)
        - ``lon``: 1-D numpy array of longitudes (ascending)
        - ``begin``: ``jnp.array([t0, lat0, lon0])`` — grid origin
        - ``spacing``: ``jnp.array([dt, dlat, dlon])`` — grid step sizes
        - ``departure_offset_h``: hours from first NetCDF timestamp to
          *departure_time* (0 if *departure_time* is ``None``).
    """
    lat_name = _get_coord_name(ds, ["latitude", "lat"])
    lon_name = _get_coord_name(ds, ["longitude", "lon"])
    time_name = _get_coord_name(ds, ["time", "valid_time"])

    lats = ds[lat_name].values.astype(np.float64)
    lons = ds[lon_name].values.astype(np.float64)
    times = ds[time_name].values  # numpy datetime64 array

    # Ensure ascending latitude
    if lats[0] > lats[-1]:
        lats = lats[::-1]
        ds = ds.isel({lat_name: slice(None, None, -1)})

    # Ensure ascending longitude
    if lons[0] > lons[-1]:
        lons = lons[::-1]
        ds = ds.isel({lon_name: slice(None, None, -1)})

    # Convert times to hours since first timestamp
    t0_np = times[0]
    times_hours = (times - t0_np) / np.timedelta64(1, "h")
    times_hours = times_hours.astype(np.float64)

    # Compute departure offset
    if departure_time is not None:
        if isinstance(departure_time, str | datetime):
            departure_time = np.datetime64(departure_time)
        departure_offset_h = float((departure_time - t0_np) / np.timedelta64(1, "h"))
    else:
        departure_offset_h = 0.0

    # Grid parameters
    dt = float(times_hours[1] - times_hours[0]) if len(times_hours) > 1 else 1.0
    dlat = float(lats[1] - lats[0]) if len(lats) > 1 else 1.0
    dlon = float(lons[1] - lons[0]) if len(lons) > 1 else 1.0

    begin = jnp.array([times_hours[0], lats[0], lons[0]], dtype=jnp.float32)[None, :]
    spacing = jnp.array([dt, dlat, dlon], dtype=jnp.float32)[None, :]

    return {
        "lat": lats,
        "lon": lons,
        "times_hours": times_hours,
        "begin": begin,
        "spacing": spacing,
        "departure_offset_h": departure_offset_h,
        "ds": ds,
        "lat_name": lat_name,
        "lon_name": lon_name,
        "time_name": time_name,
    }


def load_dataset_epoch(path: str | Path) -> datetime:
    """Return the first ERA5 timestamp as a naive UTC datetime.

    Parameters
    ----------
    path : str or Path
        Path to an ERA5 NetCDF dataset.

    Returns
    -------
    datetime
        First dataset timestamp as timezone-naive UTC datetime.
    """
    ds = _load_dataset(path)
    try:
        time_name = _get_coord_name(ds, ["time", "valid_time"])
        epoch_np = ds[time_name].values[0]
    finally:
        ds.close()

    ts = (epoch_np - np.datetime64("1970-01-01T00:00:00")) / np.timedelta64(1, "s")
    return datetime.fromtimestamp(float(ts), tz=UTC).replace(tzinfo=None)


def _build_field_closure(
    data_a: jnp.ndarray,
    data_b: jnp.ndarray,
    begin: jnp.ndarray,
    spacing: jnp.ndarray,
    departure_offset_h: float,
    order: int = 1,
    mode: str = "nearest",
    add_time_variant_attr: bool = False,
) -> Callable[
    [jnp.ndarray, jnp.ndarray, jnp.ndarray],
    tuple[jnp.ndarray, jnp.ndarray],
]:
    """Build a JAX-interpolated field closure.

    Parameters
    ----------
    data_a, data_b : jnp.ndarray
        3-D arrays of shape ``(T, lat, lon)`` for the two field components.
    begin, spacing : jnp.ndarray
        Grid origin and step size, each of shape ``(1, 3)`` — ``[t, lat, lon]``.
    departure_offset_h : float
        Offset in hours to add to the ``t`` argument so that ``t=0`` maps to
        the departure time within the dataset.
    order : int
        Interpolation order (1 = linear).
    mode : str
        Boundary mode for ``map_coordinates``.
    add_time_variant_attr : bool
        If ``True``, mark the returned function as ``time_variant``.

    Returns
    -------
    Callable
        ``(lon, lat, t) -> (a, b)`` closure.
    """
    dep_offset = jnp.float32(departure_offset_h)

    def _field(
        lon: jnp.ndarray,
        lat: jnp.ndarray,
        ts: jnp.ndarray | int | float,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        # Normalise ts
        if isinstance(ts, int | float):
            ts = jnp.array([ts], dtype=jnp.float32)
        ts = jnp.atleast_1d(ts)

        # Handle mismatched lengths (same pattern as benchmark.py)
        diff = lat.shape[0] - ts.shape[0]
        if diff > 0:
            ts_full = jnp.concatenate([ts, jnp.full(diff, ts[-1])])
        elif diff < 0:
            ts_full = ts[: lat.shape[0]]
        else:
            ts_full = ts

        # Offset: t=0 means departure time
        ts_full = ts_full + dep_offset

        # Handle 2D inputs
        if lat.ndim > 1:
            shape = lat.shape
            if ts_full.ndim < lat.ndim:
                ts_full = jnp.repeat(ts_full[:, None], lat.shape[1], axis=1)
            ts_full = ts_full.flatten()
            lat = lat.flatten()
            lon = lon.flatten()
        else:
            shape = None

        # Build coordinates: [t, lat, lon]
        x = jnp.stack([ts_full, lat, lon], axis=-1)

        # Normalise to grid indices
        coords = (x - begin) / spacing  # shape (N, 3)

        # Interpolate
        a = jax.scipy.ndimage.map_coordinates(data_a, coords.T, order=order, mode=mode)
        b = jax.scipy.ndimage.map_coordinates(data_b, coords.T, order=order, mode=mode)

        # Reshape if needed
        if shape is not None:
            a = a.reshape(shape)
            b = b.reshape(shape)

        return a, b

    if add_time_variant_attr:
        _field = time_variant(_field)

    return _field


# ── public API ────────────────────────────────────────────────────────────


def load_era5_windfield(
    path: str | Path,
    departure_time: datetime | str | np.datetime64 | None = None,
    order: int = 1,
    u_var: str | None = None,
    v_var: str | None = None,
) -> Callable[
    [jnp.ndarray, jnp.ndarray, jnp.ndarray],
    tuple[jnp.ndarray, jnp.ndarray],
]:
    """Load ERA5 10-m wind data and return a windfield closure.

    The returned function has the signature
    ``(lon, lat, t) -> (u10, v10)``
    where ``u10`` and ``v10`` are the 10-m wind components in m/s.

    Parameters
    ----------
    path : str or Path
        Path to the ERA5 wind NetCDF file.
    departure_time : datetime or str or np.datetime64, optional
        The voyage departure time.  When provided, ``t = 0`` in the returned
        closure corresponds to this datetime; otherwise ``t = 0`` maps to the
        first timestamp in the dataset.
    order : int
        Interpolation order (1 = linear, default).
    u_var : str, optional
        Name of the u-wind variable in the NetCDF file.
    v_var : str, optional
        Name of the v-wind variable in the NetCDF file.

    Returns
    -------
    Callable
        ``(lon, lat, t) -> (u10, v10)`` with ``.is_time_variant = True``.
    """
    ds = _load_dataset(path)

    # Auto-detect variable names
    if u_var is None:
        for candidate in ["u10", "10m_u_component_of_wind", "U10"]:
            if candidate in ds.data_vars:
                u_var = candidate
                break
        if u_var is None:
            ds.close()
            raise KeyError(
                f"Cannot find u-wind variable in {path}. "
                f"Available: {list(ds.data_vars)}"
            )
    if v_var is None:
        for candidate in ["v10", "10m_v_component_of_wind", "V10"]:
            if candidate in ds.data_vars:
                v_var = candidate
                break
        if v_var is None:
            ds.close()
            raise KeyError(
                f"Cannot find v-wind variable in {path}. "
                f"Available: {list(ds.data_vars)}"
            )

    grid = _prepare_grid(ds, departure_time)
    ds = grid["ds"]  # use the reindexed dataset from _prepare_grid

    # Extract data as JAX arrays: shape (T, lat, lon)
    udata = ds[u_var]
    vdata = ds[v_var]

    umat = jnp.array(udata.values, dtype=jnp.float32)
    vmat = jnp.array(vdata.values, dtype=jnp.float32)

    ds.close()  # release file handles; data is now in JAX arrays

    logger.info(
        "Loaded ERA5 wind: shape=%s, lat=[%.1f, %.1f], lon=[%.1f, %.1f], "
        "t=[%.0f, %.0f] h",
        umat.shape,
        grid["lat"][0],
        grid["lat"][-1],
        grid["lon"][0],
        grid["lon"][-1],
        grid["times_hours"][0],
        grid["times_hours"][-1],
    )

    return _build_field_closure(
        umat,
        vmat,
        grid["begin"],
        grid["spacing"],
        grid["departure_offset_h"],
        order=order,
        mode="nearest",
        add_time_variant_attr=True,
    )


def load_era5_vectorfield(
    path: str | Path,
    departure_time: datetime | str | np.datetime64 | None = None,
    order: int = 1,
    u_var: str | None = None,
    v_var: str | None = None,
) -> Callable[
    [jnp.ndarray, jnp.ndarray, jnp.ndarray],
    tuple[jnp.ndarray, jnp.ndarray],
]:
    """Load ERA5 wind data and return a vectorfield-compatible closure.

    This is identical to :func:`load_era5_windfield` but named to align with
    the existing ``vectorfield`` interface used by ``cost_function``.

    The 10-m wind components are returned as ``(u, v)`` in m/s.  For SWOPP3,
    these are passed to the RISE polar model rather than being used directly
    as ocean currents, but the function signature is the same.

    See :func:`load_era5_windfield` for parameter docs.
    """
    return load_era5_windfield(
        path,
        departure_time=departure_time,
        order=order,
        u_var=u_var,
        v_var=v_var,
    )


def load_era5_wavefield(
    path: str | Path,
    departure_time: datetime | str | np.datetime64 | None = None,
    order: int = 1,
    hs_var: str | None = None,
    dir_var: str | None = None,
) -> Callable[
    [jnp.ndarray, jnp.ndarray, jnp.ndarray],
    tuple[jnp.ndarray, jnp.ndarray],
]:
    """Load ERA5 wave data and return a wavefield closure.

    The returned function has the signature
    ``(lon, lat, t) -> (hs, mwd)``
    where ``hs`` is significant wave height in metres and ``mwd`` is mean wave
    direction in degrees from North.

    Parameters
    ----------
    path : str or Path
        Path to the ERA5 wave NetCDF file.
    departure_time : datetime or str or np.datetime64, optional
        The voyage departure time.  When provided, ``t = 0`` in the returned
        closure corresponds to this datetime.
    order : int
        Interpolation order (1 = linear, default).
    hs_var : str, optional
        Name of the significant wave height variable.
    dir_var : str, optional
        Name of the mean wave direction variable.

    Returns
    -------
    Callable
        ``(lon, lat, t) -> (hs, mwd)``.
    """
    ds = _load_dataset(path)

    # Auto-detect variable names
    if hs_var is None:
        for candidate in [
            "swh",
            "significant_height_of_combined_wind_waves_and_swell",
            "Hs",
            "hs",
        ]:
            if candidate in ds.data_vars:
                hs_var = candidate
                break
        if hs_var is None:
            ds.close()
            raise KeyError(
                f"Cannot find wave height variable in {path}. "
                f"Available: {list(ds.data_vars)}"
            )

    if dir_var is None:
        for candidate in ["mwd", "mean_wave_direction", "MWD"]:
            if candidate in ds.data_vars:
                dir_var = candidate
                break
        if dir_var is None:
            ds.close()
            raise KeyError(
                f"Cannot find wave direction variable in {path}. "
                f"Available: {list(ds.data_vars)}"
            )

    grid = _prepare_grid(ds, departure_time)
    ds = grid["ds"]  # use the reindexed dataset from _prepare_grid

    hsdata = ds[hs_var]
    dirdata = ds[dir_var]

    hmat = jnp.array(hsdata.values, dtype=jnp.float32)
    dmat = jnp.array(dirdata.values, dtype=jnp.float32)

    ds.close()  # release file handles; data is now in JAX arrays

    # Replace NaN with 0 (land points in wave data)
    hmat = jnp.nan_to_num(hmat, nan=0.0)
    dmat = jnp.nan_to_num(dmat, nan=0.0)

    logger.info(
        "Loaded ERA5 waves: shape=%s, lat=[%.1f, %.1f], lon=[%.1f, %.1f], "
        "t=[%.0f, %.0f] h",
        hmat.shape,
        grid["lat"][0],
        grid["lat"][-1],
        grid["lon"][0],
        grid["lon"][-1],
        grid["times_hours"][0],
        grid["times_hours"][-1],
    )

    return _build_field_closure(
        hmat,
        dmat,
        grid["begin"],
        grid["spacing"],
        grid["departure_offset_h"],
        order=order,
        mode="nearest",
        add_time_variant_attr=False,
    )


def load_era5_land_mask(
    wave_path: str | Path,
    hs_var: str | None = None,
) -> Land:
    """Create a :class:`~routetools.land.Land` mask from ERA5 wave NaN values.

    ERA5 wave variables (SWH, MWD) are NaN over land.  This function
    reads the first timestep, marks every grid cell that is NaN as land,
    and returns a ``Land`` object that can be passed to CMA-ES.

    Parameters
    ----------
    wave_path : str or Path
        Path to an ERA5 wave NetCDF file.
    hs_var : str, optional
        Name of the significant wave height variable.  Auto-detected if
        ``None``.

    Returns
    -------
    Land
        A ``Land`` object whose grid covers the ERA5 file extent.
    """
    from routetools.land import Land

    ds = _load_dataset(wave_path)

    # Auto-detect wave-height variable
    if hs_var is None:
        for candidate in [
            "swh",
            "significant_height_of_combined_wind_waves_and_swell",
            "Hs",
            "hs",
        ]:
            if candidate in ds.data_vars:
                hs_var = candidate
                break
        if hs_var is None:
            ds.close()
            raise KeyError(
                f"Cannot find wave height variable in {wave_path}. "
                f"Available: {list(ds.data_vars)}"
            )

    # Get grid coordinates
    lat_name = _get_coord_name(ds, ["latitude", "lat"])
    lon_name = _get_coord_name(ds, ["longitude", "lon"])

    lats = ds[lat_name].values.astype(np.float64)
    lons = ds[lon_name].values.astype(np.float64)

    # Ensure ascending
    if lats[0] > lats[-1]:
        lats = lats[::-1]
        ds = ds.isel({lat_name: slice(None, None, -1)})
    if lons[0] > lons[-1]:
        lons = lons[::-1]
        ds = ds.isel({lon_name: slice(None, None, -1)})

    # First timestep of wave height: NaN → land
    hs_t0 = ds[hs_var].isel({_get_coord_name(ds, ["time", "valid_time"]): 0}).values

    ds.close()

    # hs_t0 shape is (lat, lon).  Land class expects (x, y) = (lon, lat).
    is_land_latlon = np.isnan(hs_t0)  # (lat, lon)
    is_land_lonlat = is_land_latlon.T  # (lon, lat)

    # Land class: values > water_level → land.
    # We use 1.0 for land, 0.0 for water, water_level = 0.5.
    land_array = is_land_lonlat.astype(np.float32)

    n_lon, n_lat = land_array.shape
    lon_min, lon_max = float(lons[0]), float(lons[-1])
    lat_min, lat_max = float(lats[0]), float(lats[-1])

    # Land.__init__ computes lenx = ceil(xlim[1] - xlim[0]) * resolution[0]
    # and expects land_array.shape == (lenx, leny).
    # To match the ERA5 grid exactly we construct the Land object directly,
    # bypassing the Perlin-noise generation path entirely.
    land = Land.__new__(Land)
    land._array = jnp.array(land_array)
    land.x = jnp.linspace(lon_min, lon_max, n_lon)
    land.y = jnp.linspace(lat_min, lat_max, n_lat)
    land.xmin = lon_min
    land.xmax = lon_max
    land.xnorm = (n_lon - 1) / (lon_max - lon_min) if lon_max > lon_min else 1.0
    land.ymin = lat_min
    land.ymax = lat_max
    land.ynorm = (n_lat - 1) / (lat_max - lat_min) if lat_max > lat_min else 1.0
    land.resolution = (1, 1)
    land.random_seed = None
    land.water_level = 0.5
    land.shape = land_array.shape
    land.interpolate = 10  # insert 10 sub-points between waypoints to catch narrow land
    land.outbounds_is_land = True
    land.penalize_segments = False
    land._map_mode = "nearest"
    land._map_order = 0

    land_indices = jnp.argwhere(land._array > land.water_level)
    if land_indices.size > 0:
        land._lats = land.y[land_indices[:, 1]]
        land._lons = land.x[land_indices[:, 0]]
    else:
        land._lats = jnp.array([])
        land._lons = jnp.array([])

    n_land = int(np.sum(is_land_lonlat))
    n_total = int(np.prod(is_land_lonlat.shape))
    logger.info(
        "ERA5 land mask: %d/%d cells are land (%.1f%%), "
        "lon=[%.1f, %.1f], lat=[%.1f, %.1f], shape=%s",
        n_land,
        n_total,
        100 * n_land / n_total,
        lon_min,
        lon_max,
        lat_min,
        lat_max,
        land_array.shape,
    )

    return land


def load_natural_earth_land_mask(
    lon_range: tuple[float, float],
    lat_range: tuple[float, float],
    resolution: float = 0.01,
    ne_resolution: str = "10m",
    interpolate: int = 50,
) -> Land:
    """Create a high-resolution land mask from Natural Earth shapefiles.

    Uses cartopy's Natural Earth 1:10m land polygons rasterized onto a
    regular grid.  This is much more accurate than the ERA5 wave-NaN
    approach for narrow land features (Cape Cod, Aleutian Islands,
    Channel Islands, narrow straits, etc.).

    Parameters
    ----------
    lon_range : tuple[float, float]
        ``(lon_min, lon_max)`` — longitude extent of the corridor in
        degrees East.  Values > 180 are supported (unwrapped antimeridian).
    lat_range : tuple[float, float]
        ``(lat_min, lat_max)`` — latitude extent in degrees North.
    resolution : float
        Grid cell size in degrees.  Default 0.01° ≈ 1.1 km at the equator.
    ne_resolution : str
        Natural Earth resolution: ``"10m"``, ``"50m"``, or ``"110m"``.
    interpolate : int
        Number of sub-points inserted between waypoints for segment
        checking (passed to ``Land``).  Default 50 gives ~1 km spacing
        with L=100 waypoints, matching the 0.01° mask resolution.

    Returns
    -------
    Land
        A ``Land`` object compatible with CMA-ES.
    """
    try:
        import cartopy.io.shapereader as shpreader
        from shapely.affinity import translate
        from shapely.geometry import box
        from shapely.validation import make_valid
    except ImportError as exc:
        raise ImportError(
            "Natural Earth land mask requires cartopy, shapely, and rasterio. "
            "Install them with:\n  pip install cartopy shapely rasterio"
        ) from exc

    try:
        import rasterio  # noqa: F401 — validate availability early
    except ImportError as exc:
        raise ImportError(
            "Natural Earth land mask requires rasterio. "
            "Install it with:\n  pip install rasterio"
        ) from exc

    from routetools.land import Land

    # Load Natural Earth land polygons
    shp_path = shpreader.natural_earth(
        resolution=ne_resolution, category="physical", name="land"
    )
    reader = shpreader.Reader(shp_path)
    land_geoms = list(reader.geometries())

    lon_min, lon_max = float(lon_range[0]), float(lon_range[1])
    lat_min, lat_max = float(lat_range[0]), float(lat_range[1])

    # Build a bounding box with a small buffer for clipping
    buf = 1.0  # degree buffer
    clip_box = box(lon_min - buf, lat_min - buf, lon_max + buf, lat_max + buf)

    # If the corridor uses longitudes > 180 (unwrapped antimeridian),
    # we need shifted copies of the NE geometries (which are in [-180, 180]).
    need_shift = lon_max > 180 or lon_min > 180

    # Clip geometries to the region of interest for efficiency
    clipped = []
    for geom in land_geoms:
        if geom is None or geom.is_empty:
            continue

        # Original geometry (NE coordinates in [-180, 180])
        try:
            g = make_valid(geom) if not geom.is_valid else geom
            intersection = g.intersection(clip_box)
            if not intersection.is_empty:
                clipped.append(intersection)
        except Exception as exc:
            logger.debug("NE geom intersection failed: %s", exc)

        # For antimeridian-crossing corridors, also shift by +360°
        # so that NE lon -180..-120 becomes 180..240, etc.
        if need_shift:
            try:
                shifted = translate(geom, xoff=360)
                shifted = make_valid(shifted) if not shifted.is_valid else shifted
                intersection = shifted.intersection(clip_box)
                if not intersection.is_empty:
                    clipped.append(intersection)
            except Exception as exc:
                logger.debug("NE shifted geom intersection failed: %s", exc)

    if not clipped:
        clipped = []

    # Build the raster grid dimensions
    n_lon = int(ceil((lon_max - lon_min) / resolution)) + 1
    n_lat = int(ceil((lat_max - lat_min) / resolution)) + 1

    # Rasterize using rasterio scan-line fill (O(pixels), very fast)
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds

    # rasterio transform: maps pixel (col, row) → (lon, lat).
    # rows = lat axis (n_lat), cols = lon axis (n_lon).
    # The Land object expects array shape (n_lon, n_lat) with
    # axis-0 = lon, axis-1 = lat, so we rasterize with
    # width=n_lon, height=n_lat, then transpose.
    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, n_lon, n_lat)
    shapes = [(geom, 1) for geom in clipped if geom is not None and not geom.is_empty]

    if shapes:
        # rasterize returns (height=n_lat, width=n_lon) uint8 array
        # Row 0 = lat_max (north) in rasterio convention.
        raster = rasterize(
            shapes,
            out_shape=(n_lat, n_lon),
            transform=transform,
            fill=0,
            dtype=np.uint8,
            all_touched=True,  # mark cells touched by polygon boundary
        )
        # Flip latitude so row 0 = lat_min (south), matching
        # Land.y = linspace(lat_min, lat_max, n_lat).
        raster = raster[::-1, :]
        # Transpose to (n_lon, n_lat) to match Land's (x=lon, y=lat) convention
        land_array = raster.T.astype(np.float32)
    else:
        land_array = np.zeros((n_lon, n_lat), dtype=np.float32)

    # Construct the Land object (same bypass pattern as load_era5_land_mask)
    land = Land.__new__(Land)
    land._array = jnp.array(land_array)
    land.x = jnp.linspace(lon_min, lon_max, n_lon)
    land.y = jnp.linspace(lat_min, lat_max, n_lat)
    land.xmin = lon_min
    land.xmax = lon_max
    land.xnorm = (n_lon - 1) / (lon_max - lon_min) if lon_max > lon_min else 1.0
    land.ymin = lat_min
    land.ymax = lat_max
    land.ynorm = (n_lat - 1) / (lat_max - lat_min) if lat_max > lat_min else 1.0
    land.resolution = (1, 1)
    land.random_seed = None
    land.water_level = 0.5
    land.shape = land_array.shape
    land.interpolate = interpolate
    land.outbounds_is_land = True
    land.penalize_segments = False
    land._map_mode = "nearest"
    land._map_order = 0

    land_indices = jnp.argwhere(land._array > land.water_level)
    if land_indices.size > 0:
        land._lats = land.y[land_indices[:, 1]]
        land._lons = land.x[land_indices[:, 0]]
    else:
        land._lats = jnp.array([])
        land._lons = jnp.array([])

    n_land = int(np.sum(land_array > 0.5))
    n_total = n_lon * n_lat
    logger.info(
        "Natural Earth land mask (%s): %d/%d cells are land (%.1f%%), "
        "lon=[%.1f, %.1f], lat=[%.1f, %.1f], res=%.3f°, shape=%s",
        ne_resolution,
        n_land,
        n_total,
        100 * n_land / n_total,
        lon_min,
        lon_max,
        lat_min,
        lat_max,
        resolution,
        land_array.shape,
    )

    return land
