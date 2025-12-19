import jax.numpy as jnp


def beaufort_scale(
    wind_speed: jnp.array = None,
    wave_height: jnp.array = None,
    beaufort_max: float = 12,
    asfloat: bool = False,
) -> jnp.ndarray:
    """Compute Beaufort scale [0, 12] from wind speed in m/s.

    One of `wind_speed`, or `wave_height` must be given.

    If wave_height is given, we compute approximate wind speed from wave height,
    then compute Beaufort scale.

    Derived from NOAA:
    https://www.vos.noaa.gov/MWL/201512/waveheight.shtml

    Parameters
    ----------
    wind_speed : jnp.ndarray, optional
        Wind speed in meters per second.
    wave_height : jnp.ndarray, optional
        Wave heights in meters.
    beaufort_max : float, optional
        maximum beaufort scale, by default 12
    asfloat : bool, optional
        if True, returns beaufort scale as a float, else return as integer
        by default False

    Returns
    -------
    jnp.ndarray
        beaufort scale
    """
    if (wind_speed is None) and (wave_height is None):
        raise ValueError("No data provided, can not compute beaufort.")
    elif (wave_height is not None) and (wind_speed is None):
        wind_speed = 3.2 * jnp.abs(wave_height)

    bn = jnp.clip(jnp.power(jnp.array(wind_speed) / 0.836, 2 / 3), 0, beaufort_max)
    if not asfloat:
        bn = jnp.round(bn).astype(int)
    return bn


def _coefficient_direction_reduction(
    wave_incidence_angle: jnp.ndarray, beaufort: jnp.ndarray
) -> jnp.ndarray:
    """Compute the direction reduction coefficient due to waves.

    Interpolated from Ship Resistance and Propulsion by Anthony F. Molland,
    and Townsin-Kwon's paper on Speed loss due to added resistance in wind and waves.

    Parameters
    ----------
    wave_incidence_angle : jnp.ndarray
        Relative angle between ship and incoming waves in degrees.
    beaufort : jnp.ndarray
        Sliding Beaufort level, converted from wave height

    Returns
    -------
    jnp.ndarray
        Direction reduction coefficient, C_beta or mu
    """
    rad = jnp.radians(wave_incidence_angle)
    a = 6 * jnp.sin(rad / 2) ** (2 / 3) + 2
    r = 1.205 * rad
    b = 0.06 * (1 + (jnp.sin(r) - jnp.cos(r))) / (1 + jnp.sqrt(2))
    c = 2 - 1.6 * jnp.sin(rad / 2)
    return (c - b * (beaufort - a) ** 2) / 2


def _coefficient_speed_reduction(
    beaufort: jnp.ndarray,
    displacement: float,
) -> jnp.ndarray:
    """Compute speed reduction coefficient.

    This equation gives a ratio of the ship's velocity with respect to only the
    beaufort levels.

    Equation 3.65 from Ship Resistance and Propulsion by Anthony F. Molland.

    Parameters
    ----------
    beaufort : jnp.ndarray
        Sliding Beaufort level
    displacement : float
        Ship displacement in m^3

    Returns
    -------
    jnp.ndarray
        speed reduction coefficient, C_u or alpha
    """
    return (0.7 * beaufort + beaufort ** (6.5)) / (22 * displacement ** (2 / 3))


def _correction_factor(froude_number: jnp.ndarray, coef_block: float) -> jnp.ndarray:
    """Taken from Table 3.12 of Ship Resistance and Propulsion by Anthony F. Molland.

    The range of block coefficient is from 0.55 - 0.85.

    Parameters
    ----------
    froude_number : jnp.ndarray
        froude number as list of float
    coef_block : float
        block coefficient as list of float

    Returns
    -------
    jnp.ndarray
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


def speed_loss_involuntary(
    wave_height: jnp.ndarray,
    wave_incidence_angle: jnp.ndarray,
    vel_ship: float,
    coef_block: float = 0.6,
    length: float = 220,
    displacement: float = 36500,
) -> jnp.ndarray:
    """Compute speed loss as a percentage between 0% to 100%.

    The speed loss must be applied to the original vessel speed over water to get
    the effective speed over water.

    Only valid for large ships (displacement > 100,000 m3).
    Typical reductions should be between 0 to 30.


    Parameters
    ----------
    beaufort : jnp.ndarray
        Sliding scale of Beaufort levels
    wave_incidence_angle : jnp.ndarray
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
    jnp.ndarray
        Adjusted ship speed over water after involuntary speed loss due to waves.
    """
    beaufort = beaufort_scale(wave_height=wave_height, asfloat=True)
    c_beta = _coefficient_direction_reduction(wave_incidence_angle, beaufort)
    c_u = _coefficient_speed_reduction(beaufort, displacement)

    # Computes froude number, given the ship's velocity and length,
    # g = 9.81 m/s^2, the gravitational acceleration
    froude_number = vel_ship / (length * 9.81) ** 0.5
    alpha = _correction_factor(froude_number, coef_block)
    speed_loss = c_beta * c_u * alpha
    # Apply speed loss
    return vel_ship * (100 - speed_loss) / 100
