from collections.abc import Callable

import jax.numpy as jnp
from jax import jit


def time_variant(
    func: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]
    ],
) -> Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]:
    """Mark a vector field as time variant.

    Parameters
    ----------
    func : Callable
        Vector field function.

    Returns
    -------
    Callable
        Vector field function with time variant attribute.
    """
    func.is_time_variant = True  # type: ignore[attr-defined]

    return func


def time_invariant(
    func: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]
    ],
) -> Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]:
    """Mark a vector field as time invariant.

    Parameters
    ----------
    func : Callable
        Vector field function.

    Returns
    -------
    Callable
        Vector field function with time invariant attribute.
    """
    func.is_time_variant = False  # type: ignore[attr-defined]

    return func


@jit  # type: ignore[misc]
@time_invariant
def vectorfield_circular(
    x: jnp.ndarray,
    y: jnp.ndarray,
    t: jnp.ndarray,
    intensity: float = -0.9,
    centre: tuple[float, float] = (0, 0),
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Vector field with a circular pattern.

    Source: Techy 2011
    https://doi.org/10.1007/s11370-011-0092-9
    """
    x0, y0 = centre
    u = -intensity * (y - y0)
    v = intensity * (x - x0)
    return u, v


@jit  # type: ignore[misc]
@time_variant
def vectorfield_doublegyre(
    x: jnp.ndarray,
    y: jnp.ndarray,
    t: jnp.ndarray,
    amp: float = 0.1,
    eps: float = 0.25,
    w: float = 1,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Vector field with a double gyre pattern.

    Source: Shadden 2005
    https://doi.org/10.1016/j.physd.2005.10.007
    Gunnarson 2021 (use case)
    https://doi.org/10.1038/s41467-021-27015-y

    Parameters
    ----------
    amp : float
        Amplitude of the gyre, by default 0.1
    eps: float
        Eccentricity of the gyre, by default 0.25. For eps = 0 the system can be thought
        of as a time-independent 2-D Hamiltonian system.
    w : float
        Frequency of the gyre, by default 1
    """
    a = eps * jnp.sin(w * t)
    b = 1 - 2 * eps * jnp.sin(w * t)
    f = a * jnp.power(x, 2) + b * x
    dfdx = 2 * a * x + b
    u = -jnp.pi * amp * jnp.sin(jnp.pi * f) * jnp.cos(jnp.pi * y)
    v = jnp.pi * amp * jnp.cos(jnp.pi * f) * jnp.sin(jnp.pi * y) * dfdx
    return u, v


def _Ru(x: jnp.ndarray, y: jnp.ndarray, a: float, b: float) -> jnp.ndarray:
    return 1 / (3 * ((x - a) ** 2 + (y - b) ** 2) + 1) * -(y - b)


def _Rv(x: jnp.ndarray, y: jnp.ndarray, a: float, b: float) -> jnp.ndarray:
    return 1 / (3 * ((x - a) ** 2 + (y - b) ** 2) + 1) * (x - a)


@jit  # type: ignore[misc]
@time_invariant
def vectorfield_fourvortices(
    x: jnp.ndarray,
    y: jnp.ndarray,
    t: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Vector field with four vortices.

    Source: Ferraro 2021
    https://doi.org/10.1016/j.ifacol.2021.11.097
    """
    u = 1.7 * (-_Ru(x, y, 2, 2) - _Ru(x, y, 4, 4) - _Ru(x, y, 2, 5) + _Ru(x, y, 5, 1))
    v = 1.7 * (-_Rv(x, y, 2, 2) - _Rv(x, y, 4, 4) - _Rv(x, y, 2, 5) + _Rv(x, y, 5, 1))
    return u, v


@jit  # type: ignore[misc]
@time_invariant
def vectorfield_swirlys(
    x: jnp.ndarray,
    y: jnp.ndarray,
    t: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Vector field with periodic behaviour.

    Source: Ferraro 2021
    https://doi.org/10.1016/j.ifacol.2021.11.097
    """
    u = jnp.cos(2 * x - y - 6)
    v = 2 / 3 * jnp.sin(y) + x - 3
    return u, v


@jit  # type: ignore[misc]
@time_variant
def vectorfield_techy(
    x: jnp.ndarray, y: jnp.ndarray, t: jnp.ndarray, sink: float = -0.3
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Vector field with time dependency.

    Time-varying flow that includes a sink < 0, and a vortex that changes vorticity
    linearly during the maneuver.

    Source:
    "Optimal navigation in planar time-varying flow: Zermelo's problem revisited"
    Techy 2011, Fig 12a-c
    "Visir-1.b: ocean surface gravity waves and currents for
    energy-efficient navigation" Mannarini 2019, Fig 2b
    """
    vortex = t - 0.5
    u = sink * x - vortex * y
    v = vortex * x + sink * y
    return u, v


@jit  # type: ignore[misc]
@time_invariant
def vectorfield_zero(
    x: jnp.ndarray, y: jnp.ndarray, t: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """No currents."""
    return jnp.zeros_like(x), jnp.zeros_like(y)
