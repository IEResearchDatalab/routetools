import jax.numpy as jnp
import pytest

from routetools.cmaes import control_to_curve, curve_to_control, optimize
from routetools.land import Land
from routetools.vectorfield import (
    vectorfield_fourvortices,
    vectorfield_swirlys,
    vectorfield_techy,
)


@pytest.mark.parametrize(
    "vectorfield, src, dst, expected",
    [
        (
            vectorfield_fourvortices,
            jnp.array([0, 0]),
            jnp.array([6, 2]),
            10.0,
        ),
        (
            vectorfield_techy,
            jnp.array([jnp.cos(jnp.pi / 6), jnp.sin(jnp.pi / 6)]),
            jnp.array([0, 1]),
            1.04,
        ),
    ],
)
def test_cmaes_constant_speed(
    vectorfield: callable,
    src: jnp.array,
    dst: jnp.array,
    expected: float,
    L: int = 64,
):
    curve, dict_cmaes = optimize(
        vectorfield,
        src=src,
        dst=dst,
        travel_stw=1,
        L=L,
        popsize=10,
        sigma0=1,
        seed=1,
    )
    assert isinstance(curve, jnp.ndarray)
    assert curve.shape[0] == L
    assert curve.shape[1] == 2
    cost = dict_cmaes["cost"]
    assert isinstance(cost, float)
    assert cost <= expected, f"cost: {cost} > expected: {expected}"


@pytest.mark.parametrize(
    "vectorfield, src, dst, expected",
    [
        (
            vectorfield_swirlys,
            jnp.array([0, 0]),
            jnp.array([6, 5]),
            6.0,
        ),
    ],
)
def test_cmaes_constant_time(
    vectorfield: callable,
    src: jnp.array,
    dst: jnp.array,
    expected: float,
    L: int = 64,
):
    curve, dict_cmaes = optimize(
        vectorfield,
        src=src,
        dst=dst,
        travel_time=30,
        L=L,
        popsize=1000,
        sigma0=2,
        seed=1,
    )
    assert isinstance(curve, jnp.ndarray)
    assert curve.shape[0] == L
    assert curve.shape[1] == 2
    cost = dict_cmaes["cost"]
    assert isinstance(cost, float)
    assert cost <= expected, f"cost: {cost} > expected: {expected}"


@pytest.mark.parametrize(
    "vectorfield, src, dst",
    [
        (
            vectorfield_fourvortices,
            jnp.array([0, 0]),
            jnp.array([6, 2]),
        ),
        (
            vectorfield_techy,
            jnp.array([jnp.cos(jnp.pi / 6), jnp.sin(jnp.pi / 6)]),
            jnp.array([0, 1]),
        ),
    ],
)
def test_cmaes_constant_speed_with_land(
    vectorfield: callable,
    src: jnp.array,
    dst: jnp.array,
):
    xlim = sorted((src[0], dst[0]))
    ylim = sorted((src[1], dst[1]))
    land = Land(xlim, ylim, random_seed=1, resolution=10)

    curve, dict_cmaes = optimize(
        vectorfield,
        src=src,
        dst=dst,
        land=land,
        penalty=0.1,
        travel_stw=1,
        popsize=10,
        sigma0=1,
        seed=1,
    )
    assert isinstance(curve, jnp.ndarray)
    assert curve.shape[1] == 2
    assert isinstance(dict_cmaes["cost"], float)


@pytest.mark.parametrize(
    "vectorfield, src, dst, expected, K, L, num_pieces",
    [
        (
            vectorfield_fourvortices,
            jnp.array([0, 0]),
            jnp.array([6, 2]),
            10.0,
            13,
            61,
            4,
        ),
        (
            vectorfield_techy,
            jnp.array([jnp.cos(jnp.pi / 6), jnp.sin(jnp.pi / 6)]),
            jnp.array([0, 1]),
            1.04,
            7,
            61,
            2,
        ),
    ],
)
def test_cmaes_constant_speed_piecewise(
    vectorfield: callable,
    src: jnp.array,
    dst: jnp.array,
    expected: float,
    K: int,
    L: int,
    num_pieces: int,
):
    curve, dict_cmaes = optimize(
        vectorfield,
        src=src,
        dst=dst,
        travel_stw=1,
        K=K,
        L=L,
        num_pieces=num_pieces,
        popsize=100,
        sigma0=5,
        tolfun=0.1,
        seed=1,
    )
    cost = dict_cmaes["cost"]
    assert isinstance(curve, jnp.ndarray)
    assert curve.shape[0] == L
    assert curve.shape[1] == 2
    assert isinstance(cost, float)
    assert cost <= expected, f"cost: {cost} > expected: {expected}"


def test_curve_to_control_onepiece(L: int = 64, K: int = 8):
    """Test the curve_to_control function."""
    t = jnp.linspace(0, 1, L)
    curve = jnp.stack((t, t**2), axis=1)  # A simple quadratic curve
    control_points = curve_to_control(curve, K=K, num_pieces=1)
    assert control_points.shape == (
        2 * K - 4,
    ), f"Expected shape (2*{K - 2},), got {control_points.shape}"

    # Test if we can reconstruct the curve from control points
    reconstructed_curve = control_to_curve(
        control_points,
        src=curve[0],
        dst=curve[-1],
        L=L,
        num_pieces=1,
    )
    assert reconstructed_curve.shape == (
        L,
        2,
    ), f"Expected shape ({L}, 2), got {reconstructed_curve.shape}"
    # The reconstructed curve should be close to the original curve
    assert jnp.allclose(
        reconstructed_curve, curve, atol=1e-1
    ), "Reconstructed curve does not match original curve"


def test_curve_to_control_piecewise(L: int = 127, K: int = 10, num_pieces: int = 3):
    """Test the curve_to_control function."""
    t = jnp.linspace(0, 1, L)
    curve = jnp.stack((t, t**2), axis=1)  # A simple quadratic curve
    control_points = curve_to_control(curve, K=K, num_pieces=num_pieces)
    assert control_points.shape == (
        2 * K - 4,
    ), f"Expected shape (2*{K - 2},), got {control_points.shape}"

    # Test if we can reconstruct the curve from control points
    reconstructed_curve = control_to_curve(
        control_points,
        src=curve[0],
        dst=curve[-1],
        L=L,
        num_pieces=num_pieces,
    )
    assert reconstructed_curve.shape == (
        L,
        2,
    ), f"Expected shape ({L}, 2), got {reconstructed_curve.shape}"
    # The reconstructed curve should be close to the original curve
    assert jnp.allclose(
        reconstructed_curve, curve, atol=1e-1
    ), "Reconstructed curve does not match original curve"
