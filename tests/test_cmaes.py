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


def test_optimize_with_bounds():
    """CMA-ES respects geographic bounds on control points."""
    src = jnp.array([0.0, 0.0])
    dst = jnp.array([6.0, 2.0])
    L = 32
    K = 6  # 4 free interior control points → 8 parameters

    # Tight bounds: lon in [0, 6], lat in [-1, 3]
    n_free = (K - 2) * 2
    lower = [0.0 if i % 2 == 0 else -1.0 for i in range(n_free)]
    upper = [6.0 if i % 2 == 0 else 3.0 for i in range(n_free)]

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        curve, info = optimize(
            vectorfield_fourvortices,
            src=src,
            dst=dst,
            travel_stw=1,
            K=K,
            L=L,
            popsize=10,
            sigma0=1,
            seed=1,
            maxfevals=500,
            verbose=False,
            bounds=[lower, upper],
        )

    # All waypoints must stay within the prescribed corridor
    assert curve[:, 0].min() >= -1.0, "Longitude below lower bound"
    assert curve[:, 0].max() <= 7.0, "Longitude above upper bound"
    assert curve[:, 1].min() >= -2.0, "Latitude below lower bound"
    assert curve[:, 1].max() <= 4.0, "Latitude above upper bound"


@pytest.mark.parametrize("penalty_type", ["hard", "smooth"])
def test_optimize_weather_penalty_type(penalty_type):
    """``optimize`` accepts both ``weather_penalty_type`` values."""
    src = jnp.array([0.0, 0.0])
    dst = jnp.array([6.0, 2.0])

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        curve, info = optimize(
            vectorfield_fourvortices,
            src=src,
            dst=dst,
            travel_stw=1,
            K=6,
            L=32,
            popsize=10,
            sigma0=1,
            seed=1,
            maxfevals=200,
            verbose=False,
            weather_penalty_type=penalty_type,
        )

    assert curve.shape == (32, 2)
    assert info["cost"] > 0


def test_cmaes_snapshot_callback_receives_iteration_population():
    """CMA-ES snapshot callback should receive per-iteration route batches."""
    src = jnp.array([0.0, 0.0])
    dst = jnp.array([6.0, 2.0])
    snapshots: list[dict[str, object]] = []

    def callback(snapshot):
        snapshots.append(
            {
                "iteration": snapshot["iteration"],
                "population_shape": tuple(snapshot["population_curves"].shape),
                "cost_shape": tuple(snapshot["population_costs"].shape),
                "generation_best_index": snapshot["generation_best_index"],
                "generation_best_shape": tuple(snapshot["generation_best_curve"].shape),
                "best_shape": tuple(snapshot["best_curve"].shape),
                "best_cost": float(snapshot["best_cost"]),
            }
        )

    optimize(
        vectorfield_fourvortices,
        src=src,
        dst=dst,
        travel_stw=1,
        L=20,
        popsize=4,
        sigma0=1,
        maxfevals=12,
        seed=1,
        verbose=False,
        snapshot_callback=callback,
    )

    assert snapshots
    assert snapshots[0]["iteration"] == 1
    assert all(item["population_shape"] == (4, 20, 2) for item in snapshots)
    assert all(item["cost_shape"] == (4,) for item in snapshots)
    assert all(item["generation_best_shape"] == (20, 2) for item in snapshots)
    assert all(item["best_shape"] == (20, 2) for item in snapshots)
    assert all(isinstance(item["generation_best_index"], int) for item in snapshots)
    assert all(item["best_cost"] >= 0 for item in snapshots)
