import jax.numpy as jnp


def _coefficient_direction_reduction(
    wave_incidence_angle: jnp.ndarray, beaufort: jnp.ndarray
) -> jnp.ndarray:
    """Compute the direction reduction coefficient due to waves.

    Interpolated from Ship Resistance and Propulsion by Anthony F. Molland,
    and Townsin-Kwon's paper on Speed loss due to added resistance in wind and waves.

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
    rad = jnp.radians(wave_incidence_angle)
    a = 6 * jnp.sin(rad / 2) ** (2 / 3) + 2
    r = 1.205 * rad
    b = 0.06 * (1 + (jnp.sin(r) - jnp.cos(r))) / (1 + jnp.sqrt(2))
    c = 2 - 1.6 * jnp.sin(rad / 2)
    return (c - b * (beaufort - a) ** 2) / 2
