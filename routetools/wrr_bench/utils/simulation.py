import numpy as np
import scipy
import searoute as sr

from routetools.wrr_bench.ocean import Ocean, beaufort_scale

EARTH_RADIUS = 6378137
DEG2M = np.deg2rad(1) * EARTH_RADIUS


def auto_bounding_box(
    lat_start: float,
    lon_start: float,
    lat_end: float,
    lon_end: float,
    padding: float = 5,
) -> tuple[float]:
    """Compute a bounding box that contains the searoute between two points.

    Parameters
    ----------
    lat_start : float
        Latitude of the starting point.
    lon_start : float
        Longitude of the starting point.
    lat_end : float
        Latitude of the ending point.
    lon_end : float
        Longitude of the ending point.
    padding : float
        Padding to be added to the bounding box in degrees, by default 5.

    Returns
    -------
    tuple[float]
        Tuple with (min_lat, min_lon, max_lat, max_lon).
    """
    origin = [lon_start, lat_start]
    destination = [lon_end, lat_end]

    coordinates_dict = sr.searoute(origin, destination)

    coords = coordinates_dict["geometry"]["coordinates"]

    lats = [x[1] for x in coords]
    lons = [x[0] for x in coords]
    min_lat = min(lats) - padding
    max_lat = max(lats) + padding
    min_lon = min(lons) - padding
    max_lon = max(lons) + padding

    return (min_lat, min_lon, max_lat, max_lon)


def unit_vector(lat: np.array, lon: np.array) -> np.array:
    """Compute the 3D unit vector for given lat/lon coordinates.

    Parameters
    ----------
    lat : np.array
        Array containing the latitude of the points.
    lon : np.array
        Array containing the longitude of the points.

    Returns
    -------
    np.array
        Array with components (x, y, z) for each input point.
    """
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)

    return np.array([x, y, z])


def distance_from_points(
    lat_start: np.array,
    lon_start: np.array,
    lat_end: np.array,
    lon_end: np.array,
    ocean_data: Ocean = None,
    radius: float = EARTH_RADIUS,
    land_penalization: float = 1e9,
) -> np.array:
    """Compute great-circle distances (meters) between point pairs.

    Parameters
    ----------
    lat_start : np.array
        Array containing the latitude of the starting points.
    lon_start : np.array
        Array containing the longitude of the starting points.
    lat_end : np.array
        Array containing the latitude of the ending points.
    lon_end : np.array
        Array containing the longitude of the ending points.
    radius : float
        Radius of the sphere. Default is Earth's radius in meters.
    land_penalization : float
        Value added when route touches land, by default 1e9.

    Returns
    -------
    np.array
        Distances in meters between the two points.
    """
    unit_vector_start = unit_vector(lat_start, lon_start)
    unit_vector_end = unit_vector(lat_end, lon_end)

    # Compute the angle between the two unit vectors
    dot_product = np.sum(unit_vector_start * unit_vector_end, axis=0)
    angle = np.arccos(dot_product)  # radians
    dist = radius * angle
    # Checks if the start and ends points crosses land, and if it does, we add
    # 1e9 to penalize it
    if ocean_data is not None:
        dist += ocean_data.get_land(lat_start, lon_start) * land_penalization

    # Compute the distance between the two points
    return dist


def angle_from_points(
    lat_start: np.array,
    lon_start: np.array,
    lat_end: np.array,
    lon_end: np.array,
) -> np.array:
    """Compute azimuth angles (degrees) between point pairs.

    Parameters
    ----------
    lat_start : np.array
        Array containing the latitude of the starting points in degrees.
    lon_start : np.array
        Array containing the longitude of the starting points in degrees.
    lat_end : np.array
        Array containing the latitude of the ending points in degrees.
    lon_end : np.array
        Array containing the longitude of the ending points in degrees.

    Returns
    -------
    np.array
        Angles in degrees between the two points.
    """
    x1, y1, z1 = unit_vector(lat_start, lon_start)
    x2, y2, z2 = unit_vector(lat_end, lon_end)

    return np.rad2deg(
        np.arctan2(-x2 * y1 + x1 * y2, -(x1 * x2 + y1 * y2) * z1 + (x1**2 + y1**2) * z2)
    )


def module_angle_from_components(v: np.array, u: np.array) -> tuple[np.array]:
    """Return magnitude and azimuth angle (degrees) for vector components.

    Parameters
    ----------
    v : np.array
        Array containing the v (latitude) component of the vector.
    u : np.array
        Array containing the u (longitude) component of the vector.

    Returns
    -------
    Tuple[np.array]
        Tuple with (module, angle) for each input.
    """
    module = np.sqrt(v**2 + u**2)
    angle = np.rad2deg(np.arctan2(u, v))

    return module, angle


def components_from_module_angle(module: np.array, angle: np.array) -> tuple[np.array]:
    """Compute v and u components from module and azimuth angle.

    Parameters
    ----------
    module : np.array
        Array containing the module of the vector.
    angle : np.array
        Array containing the angle in degrees (with respect to latitude).

    Returns
    -------
    Tuple[np.array]
        Tuple with (v, u) components for each input.
    """
    angle = np.deg2rad(angle)
    v = module * np.cos(angle)  # When parallel to latitude: 0º -> cos(0º) = 1
    u = module * np.sin(angle)  # When perpendicular to latitude: 90º -> sin(90º) = 1

    return v, u


def compute_currents_projection(
    lat_start: np.array | float,
    lon_start: np.array | float,
    lat_end: np.array | float,
    lon_end: np.array | float,
    timestamps: np.array | float,
    ocean_data: Ocean,
) -> tuple[np.array]:
    """Compute current components projected parallel and perpendicular to course.

    Parameters
    ----------
    lat_start : Union[np.array, float]
        Array containing the latitude of the starting points in degrees.
    lon_start : Union[np.array, float]
        Array containing the longitude of the starting points in degrees.
    lat_end : Union[np.array, float]
        Array containing the latitude of the ending points in degrees.
    lon_end : Union[np.array, float]
        Array containing the longitude of the ending points in degrees.
    timestamps : Union[np.array, float]
        Array containing the timestamps of the starting and ending points.
    ocean_data: Ocean
        Ocean object containing the ocean data.

    Returns
    -------
    Tuple[np.array]
        Tuple containing (u_proj, v_proj) components.
    """
    # Compute the distance between the two points
    angle = angle_from_points(lat_start, lon_start, lat_end, lon_end)

    current_v, current_u = ocean_data.get_currents(lat_start, lon_start, timestamps)
    current_module, current_angle = module_angle_from_components(current_v, current_u)

    current_angle_proj = current_angle - angle

    # u is the components perpendicular to the ship direction over ground
    # v is the components parallel to the ship direction over ground
    current_v_proj, current_u_proj = components_from_module_angle(
        current_module, current_angle_proj
    )

    return current_u_proj, current_v_proj


def _coefficient_direction_reduction(
    wave_incidence_angle: np.ndarray, beaufort: np.ndarray
) -> np.ndarray:
    """Compute directional reduction coefficient from wave angle and beaufort.

    Parameters
    ----------
    wave_incidence_angle : np.ndarray
        Relative angle between ship and incoming waves in degrees.
    beaufort : np.ndarray
        Sliding Beaufort level, converted from wave height

    Returns
    -------
    np.ndarray
        Direction reduction coefficient, C_beta or mu
    """
    rad = np.radians(wave_incidence_angle)
    a = 6 * np.sin(rad / 2) ** (2 / 3) + 2
    r = 1.205 * rad
    b = 0.06 * (1 + (np.sin(r) - np.cos(r))) / (1 + np.sqrt(2))
    c = 2 - 1.6 * np.sin(rad / 2)
    return (c - b * (beaufort - a) ** 2) / 2


def _coefficient_speed_reduction(
    beaufort: np.ndarray,
    displacement: float,
) -> np.ndarray:
    """Compute speed reduction coefficient from beaufort and ship displacement.

    Parameters
    ----------
    beaufort : np.ndarray
        Sliding Beaufort level
    displacement : float
        Ship displacement in m^3

    Returns
    -------
    np.ndarray
        Speed reduction coefficient, C_u or alpha.
    """
    return (0.7 * beaufort + beaufort ** (6.5)) / (22 * displacement ** (2 / 3))


def _correction_factor(froude_number: np.ndarray, coef_block: float) -> np.ndarray:
    """Taken from Table 3.12 of Ship Resistance and Propulsion by Anthony F. Molland.

    The range of block coefficient is from 0.55 - 0.85.

    Parameters
    ----------
    froude_number : np.ndarray
        froude number as list of float
    coef_block : float
        block coefficient as list of float

    Returns
    -------
    np.ndarray
        correction factor, C_u or alpha

    Raises
    ------
    ValueError
        Block coefficient too low, less than 0.55
    ValueError
        Block coefficient too high, greater than 0.85
    """
    if coef_block < 0.55:
        raise ValueError("Block coefficient too low")
    elif coef_block == 0.55:
        # Normal
        a, b, c = 1.7, 1.4, 7.4
    elif coef_block <= 0.6:
        # Normal
        a, b, c = 2.2, 2.5, 9.7
    elif coef_block <= 0.65:
        # Normal
        a, b, c = 2.6, 3.7, 11.6
    elif coef_block <= 0.7:
        # Normal
        a, b, c = 3.1, 5.3, 12.4
    elif coef_block <= 0.75:
        # Normal or laden
        a, b, c = 2.4, 10.6, 9.5
    elif coef_block <= 0.8:
        # Normal or laden
        a, b, c = 2.6, 13.1, 15.1
    elif coef_block <= 0.85:
        # Normal or laden
        a, b, c = 3.1, 18.7, 28
    else:
        raise ValueError("Block coefficient too high")
    return a - b * froude_number - c * froude_number**2


def _speed_loss_involuntary(
    beaufort: np.ndarray,
    wave_incidence_angle: np.ndarray,
    vel_ship: float,
    coef_block: float = 0.6,
    length: float = 220,
    displacement: float = 36500,
) -> np.ndarray:
    """Estimate speed loss percentage due to waves and wind.

    Parameters
    ----------
    beaufort : np.ndarray
        Sliding scale of Beaufort levels
    wave_incidence_angle : np.ndarray
        Relative angle between ship and incoming waves in degrees
    vel_ship : float
        Ship velocity over water
    coef_block : float, optional
        Block coefficient of the ship, by default 0.6
    length : float, optional
        Length of the ship, in m, by default 220
    displacement : float, optional
        Displacement of the ship, in m^3, by default 36500

    Returns
    -------
    np.ndarray
        Speed reduction coefficient taking into account of all coefficients.
    """
    c_beta = _coefficient_direction_reduction(wave_incidence_angle, beaufort)
    c_u = _coefficient_speed_reduction(beaufort, displacement)

    # Computes froude number, given the ship's velocity and length,
    # g = 9.81 m/s^2, the gravitational acceleration
    froude_number = vel_ship / (length * 9.81) ** 0.5
    alpha = _correction_factor(froude_number, coef_block)
    return c_beta * c_u * alpha


def _cost_function_linalg(
    lat: np.ndarray,
    lon: np.ndarray,
    ts: np.ndarray,
    angle: np.ndarray,
    distance: np.ndarray,
    vel_ship: float,
    ocean_data: Ocean,
    clip_p: float = 0.2,
) -> np.ndarray:
    """Auxiliary cost function used by compute_times_linalg.

    Parameters
    ----------
    lat : np.ndarray
        Latitudes coordinates
    lon : np.ndarray
        Longitude coordinates
    ts : np.ndarray
        Timestamps
    angle : np.ndarray
        Angle of the ship
    distance : np.ndarray
        Distance between the two points
    vel_ship : float
        Ship velocity in m/s.
    ocean_data : Ocean
        Ocean object.
    clip_p : float
        Percentage to set the minimum of the velocity over ground, by default 0.2

    Returns
    -------
    np.ndarray
        Costs per segment
    """
    current_v, current_u = ocean_data.get_currents(lat, lon, ts)
    cur_mod, cur_ang = module_angle_from_components(current_v, current_u)

    # Angle of the current projected on the line of movement over ground
    cur_ang_proj = cur_ang - angle
    # This angle is with respect to X-axis, however the rest of the functions
    # refer to Y-axis. To compensate, in the next function we will assume (u, v)
    # instead of the usual (v, u)

    wave_height, wave_angle = ocean_data.get_waves(lat, lon, ts)

    # Computes wave incidence angle
    wia = np.mod(np.abs(angle - wave_angle), 360)
    wave_incidence_angle = np.minimum(wia, 360 - wia)
    # Computes beaufort and speed loss percentage
    beaufort_levels = beaufort_scale(
        wave_height=wave_height, asfloat=True, beaufort_max=8
    )
    speed_loss = _speed_loss_involuntary(
        beaufort=beaufort_levels,
        wave_incidence_angle=wave_incidence_angle,
        vel_ship=vel_ship,
    )

    # Apply speed loss
    vel_ship = vel_ship * (100 - speed_loss) / 100

    # u is the components perpendicular to the ship direction over ground
    # v is the components parallel to the ship direction over ground
    cur_u_proj, cur_v_proj = components_from_module_angle(cur_mod, cur_ang_proj)
    # Clip the perpendicular velocity to avoid negative values
    # We assume the vessel will increase its power to compensate
    vel_v2 = np.clip(vel_ship**2 - cur_v_proj**2, a_min=0, a_max=None)
    vel_ground = cur_u_proj + np.sqrt(vel_v2)
    # When wave heights are too high, the velocity over ground can turn negative
    # which is physically not possible. To avoid this, we clip the values to a
    # minimum of 20% of the vel_ship
    vel_ground = np.clip(vel_ground, a_min=clip_p * vel_ship, a_max=None)
    return distance / vel_ground


def compute_times_linalg(
    lat_start: np.array | float,
    lon_start: np.array | float,
    lat_end: np.array | float,
    lon_end: np.array | float,
    timestamps: np.array | float,
    vel_ship: float,
    ocean_data: Ocean,
) -> np.array:
    """Compute travel times for segments using the linearized cost model.

    Parameters
    ----------
    lat_start : np.array
        Array containing the latitude of the starting points.
    lon_start : np.array
        Array containing the longitude of the starting points.
    lat_end : np.array
        Array containing the latitude of the ending points.
    lon_end : np.array
        Array containing the longitude of the ending points.
    timestamps : np.array
        Array containing the timestamps.
    vel_ship : float
        Ship velocity in m/s.
    ocean_data : Ocean
        Ocean object.

    Returns
    -------
    np.array
        Array containing the time in seconds for the given lat, lon coordinates.
    """
    lat_start = np.atleast_1d(lat_start)
    lon_start = np.atleast_1d(lon_start)
    lat_end = np.atleast_1d(lat_end)
    lon_end = np.atleast_1d(lon_end)
    timestamps = np.atleast_1d(timestamps)

    # Compute the distance between the two points
    distance = distance_from_points(lat_start, lon_start, lat_end, lon_end)
    # Compute the distance between the two points
    angle = angle_from_points(lat_start, lon_start, lat_end, lon_end)

    # Compute the time
    l1 = _cost_function_linalg(
        lat_start,
        lon_start,
        timestamps,
        angle=angle,
        distance=distance,
        vel_ship=vel_ship,
        ocean_data=ocean_data,
    )
    l2 = _cost_function_linalg(
        lat_end,
        lon_end,
        timestamps,
        angle=angle,
        distance=distance,
        vel_ship=vel_ship,
        ocean_data=ocean_data,
    )
    # Takes the average of the cost from the start points and end points
    ld = (l1 + l2) / 2
    return ld


def _cost_function_integral(
    lat: np.ndarray,
    lon: np.ndarray,
    dist_lat: np.ndarray,
    dist_lon: np.ndarray,
    ts: np.ndarray,
    vel_ship: float,
    ocean_data: Ocean,
) -> np.ndarray:
    """Integral-based cost function used by compute_times_integral.

    Parameters
    ----------
    lat : np.ndarray
        latitudes coordinates
    lon : np.ndarray
        longitude coordinates
    dist_lat : np.ndarray
        latitude component of ship's velocity over ground
    dist_lon : np.ndarray
        longitude component of ship's velocity over ground

    Returns
    -------
    np.ndarray
        costs per segment
    """
    v_cur, u_cur = ocean_data.get_currents(lat, lon, ts)

    # alpha gives a ratio of the ship's velocity with respect
    # to the currents
    alpha = vel_ship**2 - (u_cur**2 + v_cur**2)
    cost = np.sqrt(
        1 / alpha * (dist_lon**2 + dist_lat**2)
        + 1 / (alpha**2) * (u_cur * dist_lon + v_cur * dist_lat) ** 2
    ) - 1 / alpha * (u_cur * dist_lon + v_cur * dist_lat)
    # TODO: Include the effect of waves
    return cost


def compute_times_integral(
    lat_start: np.array | float,
    lon_start: np.array | float,
    lat_end: np.array | float,
    lon_end: np.array | float,
    timestamps: np.array | float,
    vel_ship: float,
    ocean_data: Ocean,
) -> np.array:
    """Compute times using a numerical integral-based cost function.

    Reference: https://arxiv.org/abs/2109.05559

    Parameters
    ----------
    lat_start : np.array
        Array containing the latitude of the starting points.
    lon_start : np.array
        Array containing the longitude of the starting points.
    lat_end : np.array
        Array containing the latitude of the ending points.
    lon_end : np.array
        Array containing the longitude of the ending points.
    timestamps : np.array
        Array containing the timestamps.
    vel_ship : float
        Ship velocity in m/s.
    ocean_data : Ocean
        Ocean object.

    Returns
    -------
    np.array
        Array containing the time in seconds for the given lat, lon coordinates.
    """
    lat_start = np.atleast_1d(lat_start)
    lon_start = np.atleast_1d(lon_start)
    lat_end = np.atleast_1d(lat_end)
    lon_end = np.atleast_1d(lon_end)
    timestamps = np.atleast_1d(timestamps)

    dist = distance_from_points(lat_start, lon_start, lat_end, lon_end)

    ang = angle_from_points(lat_start, lon_start, lat_end, lon_end)

    dist_lat, dist_lon = components_from_module_angle(dist, ang)

    l1 = _cost_function_integral(
        lat_start,
        lon_start,
        dist_lat,
        dist_lon,
        timestamps,
        vel_ship=vel_ship,
        ocean_data=ocean_data,
    )
    l2 = _cost_function_integral(
        lat_end,
        lon_end,
        dist_lat,
        dist_lon,
        timestamps,
        vel_ship=vel_ship,
        ocean_data=ocean_data,
    )
    # Takes the average of the cost from the start points and end points
    ld = (l1 + l2) / 2
    return ld


def velocity_over_ground(
    vel_ship: float,
    lat_start: np.array | float,
    lon_start: np.array | float,
    lat_end: np.array | float,
    lon_end: np.array | float,
    timestamps: np.array | float,
    ocean_data: Ocean,
):
    """Return the vessel's velocity over ground for a segment."""
    current_v_proj, current_u_proj = compute_currents_projection(
        lat_start, lon_start, lat_end, lon_end, timestamps, ocean_data
    )
    return current_u_proj + np.sqrt(vel_ship**2 - current_v_proj**2)


def compute_times_given_coordinates_and_start(
    latitudes: np.array,
    longitudes: np.array,
    ts_start: np.array,
    vel_ship: float,
    ocean_data: Ocean,
    land_penalization: float = 1e6,
) -> np.array:
    """
    Compute the time series for the given lat, lon coordinates and timestamps.

    Parameters
    ----------
    latitudes : np.array
        Array containing the latitude of the route points.
        It shape can be (N_routes, N_points) or (N_points,)
    longitudes : np.array
        Array containing the longitude of the starting points.
        It shape can be (N_routes, N_points) or (N_points,)
    ts_start : np.array
        Starting timestamp.
        It shape can be (N_routes,) or (1,)
    vel_ship : float
        Ship velocity in m/s.
    ocean_data : Ocean
        Ocean object.
    land_penalization : float
        Time to be assigned when land is reached, by default 1e6.

    Returns
    -------
    np.array
        Array containing the time in seconds for the given lat, lon coordinates.
    """
    latitudes = np.atleast_2d(latitudes)
    longitudes = np.atleast_2d(longitudes)
    ts_start = np.atleast_1d(ts_start)

    time_now = ts_start
    ts: list[np.datetime64] = [time_now]

    for i in range(latitudes.shape[1] - 1):
        lat_start = latitudes[:, i]
        lon_start = longitudes[:, i]
        lat_end = latitudes[:, i + 1]
        lon_end = longitudes[:, i + 1]

        time = compute_times_linalg(
            lat_start,
            lon_start,
            lat_end,
            lon_end,
            time_now,
            vel_ship,
            ocean_data,
        )

        # When NaN or negative is returned, it means that the
        # currents are higher than the ship velocity
        # This can happen when the ship is too slow
        time = np.nan_to_num(time, nan=land_penalization)
        time[time.astype(float) < 0] = land_penalization

        time = time + ocean_data.get_land(lat_start, lon_start) * land_penalization

        time_now = time_now + np.array(time, dtype="timedelta64[s]")

        ts.append(time_now)

    return np.array(ts).swapaxes(0, 1)


def route_segments_distance(
    lat: np.array,
    lon: np.array,
    ocean_data: Ocean = None,
    land_penalization: float = 1e9,
) -> float:
    """Compute distances for each segment of multiple routes.

    Parameters
    ----------
    lat : np.array
        Latitudes of the route.
    lon : np.array
        Longitudes of the route
    ocean_data : Ocean
        Ocean object containing the ocean data.
    land_penalization : float
        Value to be added to the distance when the route touches land, by default 1e9.

    Returns
    -------
    float
        Distance of the route.
    """
    num_routes = lat.shape[0]
    lat_start = lat[:, :-1].flatten()
    lon_start = lon[:, :-1].flatten()
    lat_end = lat[:, 1:].flatten()
    lon_end = lon[:, 1:].flatten()
    dist = np.reshape(
        distance_from_points(
            lat_start=lat_start,
            lon_start=lon_start,
            lat_end=lat_end,
            lon_end=lon_end,
            ocean_data=ocean_data,
            land_penalization=land_penalization,
        ),
        (num_routes, -1),
    )

    return dist


def route_distance(
    lat: np.array,
    lon: np.array,
    ocean_data: Ocean = None,
    land_penalization: float = 1e9,
) -> float:
    """Compute total route distance from segment distances.

    Parameters
    ----------
    lat : np.array
        Latitudes of the route.
    lon : np.array
        Longitudes of the route
    ocean_data : Ocean
        Ocean object containing the ocean data.
    land_penalization : float
        Value to be added to the distance when the route touches land, by default 1e9.

    Returns
    -------
    float
        Distance of the route.
    """
    dist = np.sum(
        route_segments_distance(
            lat=lat,
            lon=lon,
            ocean_data=ocean_data,
            land_penalization=land_penalization,
        ),
        axis=1,
    )

    return dist


def even_reparametrization(
    curve: np.array,
    cost: callable,
    n_points: int = None,
    cost_per_segment: float = None,
    n_iter: int = 5,
    verbose: bool = True,
) -> np.array:
    """Reparameterize a polyline so cost per segment is approximately equal.

    Parameters
    ----------
    curve :  np.ndarray
        A polyline with shape P x 2
    cost : callable
        a function taking a polyline (shape P x 2) and returning a
        vector of costs (shape P-1)
    n_points : int
        Number of desired points for the new polyline. Default is None. Pass either this
          or `cost_per_segment`
    cost_per_segment : float
        The target cost of each segment. If None (default), `n_points` points will be
        generated
    n_iter : int, optional
        Number of iterations, by default 5
    verbose : bool, optional
        Print the standard deviation of the cost at each iteration

    Returns
    -------
        A new polyline approximating the original,
        and an array with the cost of each segment
    """

    def resample_trajectory(t, y):
        """Evaluate points on a trajectory parameterized as a polyline.

        Parameters
        ----------
        t: np.ndarray
            Evaluation points (vector of shape K), all between 0 and 1
        y: np.ndarray
            Segment endpoints, with shape P x 2

        Returns
        -------
            Natch of resampled y (matrix of shape K x 2)
        """
        assert t.ndim == 1
        assert y.ndim == 2

        def do_slice(t, y):
            n_points = y.shape[0]
            prevt = np.linspace(0, 1, n_points)
            newt = scipy.interpolate.interp1d(
                prevt, y, assume_sorted=True, fill_value="extrapolate"
            )(t)
            return newt

        return np.concatenate(
            [do_slice(t, y[:, sl])[:, None] for sl in range(y.shape[1])], axis=1
        )

    def even_resample(t: np.array, y: np.array, n_points: int, cost_per_segment: float):
        """Resample t so that y-values are evenly spaced.

        Parameters
        ----------
        (kept same as outer function)
        """
        assert t.ndim == 1
        assert y.ndim == 1
        assert t.shape == y.shape

        if n_points is None:
            n_points = int(np.floor((y[-1] - y[0]) / cost_per_segment)) + 2
            newy = np.concatenate(
                [
                    np.linspace(
                        y[0], y[0] + (n_points - 2) * cost_per_segment, (n_points - 1)
                    ),
                    [y[-1]],
                ]
            )
        else:
            newy = np.linspace(y[0], y[-1], n_points)
        newt = scipy.interpolate.interp1d(
            y, t, assume_sorted=True, fill_value="extrapolate"
        )(newy)
        return newt

    assert curve.ndim == 2

    newt = np.linspace(0, 1, curve.shape[0]).T
    newcurve = curve
    for i in range(n_iter):
        current_cost = cost(newcurve)
        if verbose:
            print(f"Iteration {i} | std = {np.std(current_cost)}")
        cumulative_cost = np.concatenate([np.zeros(1), np.cumsum(current_cost)])
        # Get values of t that even out the costs
        newt = even_resample(newt, cumulative_cost, n_points, cost_per_segment)
        # Update the curve with the new t
        newcurve = resample_trajectory(t=newt, y=curve)

    return newcurve, cost(newcurve)


def split_route_segments(
    lat: np.ndarray, lon: np.ndarray, threshold: float = 10000
) -> tuple[np.ndarray, np.ndarray]:
    """Split long route segments so each is shorter than threshold.

    Parameters
    ----------
    lat : np.ndarray
        Array of latitude values, in degrees.
    lon : np.ndarray
        Array of longitude values, in degrees.
    threshold : float, optional
        Maximum allowed segment length in meters (default 10000).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Arrays with latitudes and longitudes of the split route.
    """

    def recursion(
        a_lat: float, a_lon: float, b_lat: float, b_lon: float
    ) -> list[tuple[float, float]]:
        """Recursively split a segment until it is shorter than threshold.

        Returns a list of (lat, lon) pairs including endpoints.
        """
        dist = distance_from_points(a_lat, a_lon, b_lat, b_lon, land_penalization=0)
        if dist <= threshold:
            return [(a_lat, a_lon), (b_lat, b_lon)]
        # Convert to radians
        a_latr, a_lonr, b_latr, b_lonr = np.deg2rad([a_lat, a_lon, b_lat, b_lon])
        # split the segment in two (accounting for the Earth curvature)
        bx = np.cos(b_latr) * np.cos(b_lonr - a_lonr)
        by = np.cos(b_latr) * np.sin(b_lonr - a_lonr)
        mid_lat = np.arctan2(
            np.sin(a_latr) + np.sin(b_latr),
            np.sqrt((np.cos(a_latr) + bx) ** 2 + by**2),
        )
        mid_lon = a_lonr + np.arctan2(by, np.cos(a_latr) + bx)
        # Convert back to degrees
        mid_lat, mid_lon = np.rad2deg(mid_lat), np.rad2deg(mid_lon)
        return (
            recursion(a_lat, a_lon, mid_lat, mid_lon)
            + recursion(mid_lat, mid_lon, b_lat, b_lon)[1:]
        )

    result_lat = [lat[0]]
    result_lon = [lon[0]]
    # iterate over the pairs of points
    for i in range(1, len(lat)):
        result = recursion(lat[i - 1], lon[i - 1], lat[i], lon[i])[1:]
        result_lat.extend([point[0] for point in result])
        result_lon.extend([point[1] for point in result])
    # return the result as numpy arrays
    return np.array(result_lat), np.array(result_lon)
