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


# ---------------------------------------------------------------------------
# dt_eval_minutes  (Δt₂ decoupling)
# ---------------------------------------------------------------------------


def test_dt_eval_minutes_output_shape():
    """Output curve uses L (Δt₁), not L_eval, when dt_eval_minutes is set."""
    L = 32
    travel_time = 10.0  # hours
    dt_eval = 5.0  # minutes → L_eval = 10*60/5 + 1 = 121
    curve, info = optimize(
        vectorfield_swirlys,
        src=jnp.array([0.0, 0.0]),
        dst=jnp.array([6.0, 5.0]),
        travel_time=travel_time,
        L=L,
        dt_eval_minutes=dt_eval,
        popsize=10,
        sigma0=1,
        seed=42,
    )
    assert curve.shape == (L, 2), f"Expected output shape ({L}, 2), got {curve.shape}"


def test_dt_eval_minutes_zero_is_backward_compatible():
    """dt_eval_minutes=0 behaves identically to omitting it."""
    kwargs = dict(
        vectorfield=vectorfield_swirlys,
        src=jnp.array([0.0, 0.0]),
        dst=jnp.array([6.0, 5.0]),
        travel_time=30,
        L=64,
        popsize=10,
        sigma0=1,
        seed=1,
    )
    _, info_default = optimize(**kwargs)
    _, info_zero = optimize(**kwargs, dt_eval_minutes=0.0)
    assert info_default["cost"] == info_zero["cost"]


def test_dt_eval_minutes_finer_grid_does_not_degrade():
    """Using a finer eval grid should not produce worse results."""
    kwargs = dict(
        vectorfield=vectorfield_swirlys,
        src=jnp.array([0.0, 0.0]),
        dst=jnp.array([6.0, 5.0]),
        travel_time=30,
        L=64,
        popsize=200,
        sigma0=2,
        seed=1,
    )
    _, info_coarse = optimize(**kwargs, dt_eval_minutes=0.0)
    _, info_fine = optimize(**kwargs, dt_eval_minutes=5.0)
    # Fine grid may produce a different cost but shouldn't be wildly worse
    assert info_fine["cost"] < info_coarse["cost"] * 2, (
        f"Fine grid cost {info_fine['cost']} much worse than "
        f"coarse cost {info_coarse['cost']}"
    )


# ---------------------------------------------------------------------------
# Weather penalty time-awareness regression tests
# ---------------------------------------------------------------------------


def _windfield_time_dependent(lon, lat, t):
    """Wind field: calm at t=0, stormy (30 m/s) for t >= 5."""
    tws = jnp.where(t >= 5.0, 30.0, 5.0)
    u10 = tws * jnp.ones_like(lon)
    v10 = jnp.zeros_like(lon)
    return u10, v10


def _wavefield_time_dependent(lon, lat, t):
    """Wave field: calm at t=0, rough (12 m) for t >= 5."""
    hs = jnp.where(t >= 5.0, 12.0, 1.0)
    mwd = jnp.zeros_like(lon)
    return hs, mwd


def test_wind_penalty_smooth_uses_travel_time():
    """wind_penalty_smooth must evaluate weather at actual voyage timestamps.

    Regression: when travel_time was not forwarded the penalty evaluated
    everything at t=0 (calm) and returned ≈0 even though the real voyage
    would encounter 30 m/s winds.
    """
    from routetools.weather import wind_penalty_smooth

    # Straight route, 10 points
    L = 10
    lons = jnp.linspace(0.0, 6.0, L)
    lats = jnp.linspace(0.0, 5.0, L)
    curve = jnp.stack([lons, lats], axis=-1)[jnp.newaxis, :, :]  # (1, L, 2)

    # With travel_time=10 h, segments span t ∈ [0, 10]: most are at t>=5 → stormy
    penalty_with_time = wind_penalty_smooth(
        curve,
        windfield=_windfield_time_dependent,
        tws_limit=20.0,
        weight=100.0,
        travel_time=10.0 * 3600,  # seconds
        spherical_correction=False,
    )

    # Without travel_time → t=0 everywhere → calm (5 m/s < 20 limit) → penalty ≈ 0
    penalty_no_time = wind_penalty_smooth(
        curve,
        windfield=_windfield_time_dependent,
        tws_limit=20.0,
        weight=100.0,
        spherical_correction=False,
    )

    assert float(penalty_no_time[0]) == 0.0, "At t=0 wind is calm, penalty must be 0"
    assert (
        float(penalty_with_time[0]) > 0.0
    ), "With travel_time the route hits stormy conditions, penalty must be > 0"


def test_wave_penalty_smooth_uses_travel_time():
    """wave_penalty_smooth must evaluate weather at actual voyage timestamps."""
    from routetools.weather import wave_penalty_smooth

    L = 10
    lons = jnp.linspace(0.0, 6.0, L)
    lats = jnp.linspace(0.0, 5.0, L)
    curve = jnp.stack([lons, lats], axis=-1)[jnp.newaxis, :, :]

    penalty_with_time = wave_penalty_smooth(
        curve,
        wavefield=_wavefield_time_dependent,
        hs_limit=7.0,
        weight=100.0,
        travel_time=10.0 * 3600,
        spherical_correction=False,
    )

    penalty_no_time = wave_penalty_smooth(
        curve,
        wavefield=_wavefield_time_dependent,
        hs_limit=7.0,
        weight=100.0,
        spherical_correction=False,
    )

    assert float(penalty_no_time[0]) == 0.0, "At t=0 seas are calm, penalty must be 0"
    assert (
        float(penalty_with_time[0]) > 0.0
    ), "With travel_time the route hits rough seas, penalty must be > 0"


def test_weather_penalty_smooth_uses_travel_time():
    """weather_penalty_smooth (combined) must forward travel_time."""
    from routetools.weather import weather_penalty_smooth

    L = 10
    lons = jnp.linspace(0.0, 6.0, L)
    lats = jnp.linspace(0.0, 5.0, L)
    curve = jnp.stack([lons, lats], axis=-1)[jnp.newaxis, :, :]

    penalty_with_time = weather_penalty_smooth(
        curve,
        windfield=_windfield_time_dependent,
        wavefield=_wavefield_time_dependent,
        tws_limit=20.0,
        hs_limit=7.0,
        penalty=100.0,
        travel_time=10.0 * 3600,
        spherical_correction=False,
    )

    penalty_no_time = weather_penalty_smooth(
        curve,
        windfield=_windfield_time_dependent,
        wavefield=_wavefield_time_dependent,
        tws_limit=20.0,
        hs_limit=7.0,
        penalty=100.0,
        spherical_correction=False,
    )

    assert float(penalty_no_time[0]) == 0.0
    assert float(penalty_with_time[0]) > 0.0


def test_cmaes_penalty_forwards_time_params():
    """CMA-ES loop must forward travel_time and time_offset to penalties.

    Uses a time-dependent wind field where storms only occur at t >= 5 h.
    With travel_time=10 h, the penalty should be non-zero and influence
    the optimizer to find a different route than without the penalty.
    """
    src = jnp.array([0.0, 0.0])
    dst = jnp.array([6.0, 5.0])

    common = dict(
        vectorfield=vectorfield_swirlys,
        src=src,
        dst=dst,
        travel_time=30.0,
        L=16,
        popsize=10,
        sigma0=1,
        seed=42,
        verbose=False,
    )

    _, info_no_penalty = optimize(
        **common,
        windfield=_windfield_time_dependent,
        wind_penalty_weight=0.0,
    )

    _, info_with_penalty = optimize(
        **common,
        windfield=_windfield_time_dependent,
        wind_penalty_weight=100.0,
    )

    # The penalized run must have a higher total cost (energy + penalty)
    assert info_with_penalty["cost"] > info_no_penalty["cost"], (
        f"Penalized cost ({info_with_penalty['cost']:.4f}) should exceed "
        f"unpenalized cost ({info_no_penalty['cost']:.4f}) when storms are present"
    )
