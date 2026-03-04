"""Tests for routetools.weather — weather constraint penalties."""

import jax.numpy as jnp
import pytest

from routetools.weather import (
    DEFAULT_HS_LIMIT,
    DEFAULT_TWS_LIMIT,
    RouteWeatherStats,
    evaluate_weather,
    weather_penalty,
    weather_penalty_smooth,
)


# ---------------------------------------------------------------------------
# Synthetic field helpers
# ---------------------------------------------------------------------------
def _constant_windfield(u: float, v: float):
    """Return a windfield closure producing constant (u, v)."""

    def _field(lon, lat, t):
        return jnp.full_like(lon, u), jnp.full_like(lon, v)

    return _field


def _constant_wavefield(hs: float, mwd: float = 0.0):
    """Return a wavefield closure producing constant (hs, mwd)."""

    def _field(lon, lat, t):
        return jnp.full_like(lon, hs), jnp.full_like(lon, mwd)

    return _field


def _make_curve(n_routes: int = 1, n_points: int = 10) -> jnp.ndarray:
    """Create a batch of straight-line curves from (0,0) to (1,0)."""
    src = jnp.array([0.0, 0.0])
    dst = jnp.array([1.0, 0.0])
    single = jnp.linspace(src, dst, n_points)  # (L, 2)
    return jnp.tile(single[jnp.newaxis, :, :], (n_routes, 1, 1))  # (B, L, 2)


# ---------------------------------------------------------------------------
# Tests for default constants
# ---------------------------------------------------------------------------
class TestDefaults:
    def test_tws_limit(self):
        assert DEFAULT_TWS_LIMIT == 20.0

    def test_hs_limit(self):
        assert DEFAULT_HS_LIMIT == 7.0


# ---------------------------------------------------------------------------
# Tests for RouteWeatherStats
# ---------------------------------------------------------------------------
class TestRouteWeatherStats:
    def test_frozen(self):
        stats = RouteWeatherStats(
            max_tws=jnp.array([10.0]),
            max_hs=jnp.array([3.0]),
            tws_exceeded=jnp.array([False]),
            hs_exceeded=jnp.array([False]),
        )
        with pytest.raises(AttributeError):
            stats.max_tws = jnp.array([99.0])

    def test_fields(self):
        stats = RouteWeatherStats(
            max_tws=jnp.array([10.0]),
            max_hs=jnp.array([3.0]),
            tws_exceeded=jnp.array([False]),
            hs_exceeded=jnp.array([False]),
        )
        assert stats.max_tws.shape == (1,)
        assert stats.max_hs.shape == (1,)
        assert stats.tws_exceeded.shape == (1,)
        assert stats.hs_exceeded.shape == (1,)


# ---------------------------------------------------------------------------
# Tests for evaluate_weather
# ---------------------------------------------------------------------------
class TestEvaluateWeather:
    """Test evaluate_weather with synthetic fields."""

    def test_no_fields_returns_zeros(self):
        curve = _make_curve()
        stats = evaluate_weather(curve)
        assert jnp.allclose(stats.max_tws, 0.0)
        assert jnp.allclose(stats.max_hs, 0.0)
        assert not stats.tws_exceeded.any()
        assert not stats.hs_exceeded.any()

    def test_within_limits(self):
        curve = _make_curve()
        wf = _constant_windfield(10.0, 0.0)  # TWS = 10 < 20
        wvf = _constant_wavefield(3.0)  # Hs = 3 < 7
        stats = evaluate_weather(curve, windfield=wf, wavefield=wvf)
        assert jnp.allclose(stats.max_tws, 10.0, atol=1e-5)
        assert jnp.allclose(stats.max_hs, 3.0, atol=1e-5)
        assert not stats.tws_exceeded.any()
        assert not stats.hs_exceeded.any()

    def test_tws_exceeded(self):
        curve = _make_curve()
        wf = _constant_windfield(15.0, 15.0)  # TWS ≈ 21.2 > 20
        stats = evaluate_weather(curve, windfield=wf)
        assert stats.tws_exceeded.all()

    def test_hs_exceeded(self):
        curve = _make_curve()
        wvf = _constant_wavefield(8.0)  # Hs = 8 > 7
        stats = evaluate_weather(curve, wavefield=wvf)
        assert stats.hs_exceeded.all()

    def test_batch_dimension(self):
        curve = _make_curve(n_routes=5)
        wf = _constant_windfield(10.0, 0.0)
        stats = evaluate_weather(curve, windfield=wf)
        assert stats.max_tws.shape == (5,)
        assert stats.tws_exceeded.shape == (5,)

    def test_custom_limits(self):
        curve = _make_curve()
        wf = _constant_windfield(10.0, 0.0)  # TWS = 10
        stats = evaluate_weather(curve, windfield=wf, tws_limit=5.0)
        assert stats.tws_exceeded.all()  # 10 > 5

    def test_wind_only(self):
        curve = _make_curve()
        wf = _constant_windfield(25.0, 0.0)
        stats = evaluate_weather(curve, windfield=wf)
        assert jnp.allclose(stats.max_tws, 25.0, atol=1e-5)
        assert jnp.allclose(stats.max_hs, 0.0)

    def test_wave_only(self):
        curve = _make_curve()
        wvf = _constant_wavefield(5.0)
        stats = evaluate_weather(curve, wavefield=wvf)
        assert jnp.allclose(stats.max_tws, 0.0)
        assert jnp.allclose(stats.max_hs, 5.0, atol=1e-5)

    def test_tws_from_vector_components(self):
        """TWS = sqrt(u² + v²) for diagonal wind."""
        curve = _make_curve()
        wf = _constant_windfield(12.0, 16.0)  # TWS = 20 exactly
        stats = evaluate_weather(curve, windfield=wf)
        assert jnp.allclose(stats.max_tws, 20.0, atol=1e-5)
        # Exactly at limit — not exceeded
        assert not stats.tws_exceeded.any()


# ---------------------------------------------------------------------------
# Tests for weather_penalty (hard step)
# ---------------------------------------------------------------------------
class TestWeatherPenalty:
    """Test the hard (step) weather penalty."""

    def test_zero_when_within_limits(self):
        curve = _make_curve()
        wf = _constant_windfield(10.0, 0.0)
        wvf = _constant_wavefield(3.0)
        pen = weather_penalty(curve, windfield=wf, wavefield=wvf)
        assert jnp.allclose(pen, 0.0)

    def test_nonzero_when_tws_exceeded(self):
        curve = _make_curve(n_points=10)  # 9 segments
        wf = _constant_windfield(25.0, 0.0)
        pen = weather_penalty(curve, windfield=wf, penalty=10.0)
        # All 9 segments violate → 9 * 10 = 90
        assert jnp.allclose(pen, 90.0)

    def test_nonzero_when_hs_exceeded(self):
        curve = _make_curve(n_points=10)
        wvf = _constant_wavefield(10.0)
        pen = weather_penalty(curve, wavefield=wvf, penalty=5.0)
        assert jnp.allclose(pen, 45.0)  # 9 * 5

    def test_combined_violations(self):
        curve = _make_curve(n_points=10)
        wf = _constant_windfield(25.0, 0.0)
        wvf = _constant_wavefield(10.0)
        pen = weather_penalty(curve, windfield=wf, wavefield=wvf, penalty=1.0)
        # 9 TWS violations + 9 Hs violations = 18
        assert jnp.allclose(pen, 18.0)

    def test_no_fields_returns_zero(self):
        curve = _make_curve()
        pen = weather_penalty(curve)
        assert jnp.allclose(pen, 0.0)

    def test_batch(self):
        curve = _make_curve(n_routes=3)
        wf = _constant_windfield(25.0, 0.0)
        pen = weather_penalty(curve, windfield=wf, penalty=1.0)
        assert pen.shape == (3,)
        assert jnp.all(pen > 0)

    def test_custom_limits(self):
        curve = _make_curve(n_points=10)
        wf = _constant_windfield(10.0, 0.0)  # TWS = 10
        pen = weather_penalty(curve, windfield=wf, tws_limit=5.0, penalty=1.0)
        assert jnp.allclose(pen, 9.0)  # 9 segments × penalty 1

    def test_zero_penalty_weight(self):
        curve = _make_curve()
        wf = _constant_windfield(25.0, 0.0)
        pen = weather_penalty(curve, windfield=wf, penalty=0.0)
        assert jnp.allclose(pen, 0.0)


# ---------------------------------------------------------------------------
# Tests for weather_penalty_smooth
# ---------------------------------------------------------------------------
class TestWeatherPenaltySmooth:
    """Test the smooth (differentiable) weather penalty."""

    def test_zero_when_within_limits(self):
        curve = _make_curve()
        wf = _constant_windfield(10.0, 0.0)
        wvf = _constant_wavefield(3.0)
        pen = weather_penalty_smooth(curve, windfield=wf, wavefield=wvf)
        assert jnp.allclose(pen, 0.0)

    def test_positive_when_exceeded(self):
        curve = _make_curve()
        wf = _constant_windfield(25.0, 0.0)  # TWS = 25 > 20
        pen = weather_penalty_smooth(curve, windfield=wf)
        assert jnp.all(pen > 0)

    def test_increases_with_excess(self):
        curve = _make_curve()
        wf_21 = _constant_windfield(21.0, 0.0)  # excess = 1
        wf_25 = _constant_windfield(25.0, 0.0)  # excess = 5
        pen_21 = weather_penalty_smooth(curve, windfield=wf_21)
        pen_25 = weather_penalty_smooth(curve, windfield=wf_25)
        assert jnp.all(pen_25 > pen_21)

    def test_quadratic_growth(self):
        """Penalty grows as (excess)² — doubling excess quadruples penalty."""
        curve = _make_curve()
        wf_22 = _constant_windfield(22.0, 0.0)  # excess = 2
        wf_24 = _constant_windfield(24.0, 0.0)  # excess = 4
        pen_22 = weather_penalty_smooth(
            curve, windfield=wf_22, penalty=1.0, sharpness=1.0
        )
        pen_24 = weather_penalty_smooth(
            curve, windfield=wf_24, penalty=1.0, sharpness=1.0
        )
        ratio = pen_24 / pen_22
        expected_ratio = (4.0**2) / (2.0**2)  # 16/4 = 4
        assert jnp.allclose(ratio, expected_ratio, atol=1e-4)

    def test_sharpness_scales_penalty(self):
        curve = _make_curve()
        wf = _constant_windfield(25.0, 0.0)
        pen_s1 = weather_penalty_smooth(curve, windfield=wf, sharpness=1.0)
        pen_s5 = weather_penalty_smooth(curve, windfield=wf, sharpness=5.0)
        assert jnp.allclose(pen_s5, pen_s1 * 5.0, atol=1e-4)

    def test_no_fields_returns_zero(self):
        curve = _make_curve()
        pen = weather_penalty_smooth(curve)
        assert jnp.allclose(pen, 0.0)

    def test_batch(self):
        curve = _make_curve(n_routes=4)
        wvf = _constant_wavefield(10.0)
        pen = weather_penalty_smooth(curve, wavefield=wvf)
        assert pen.shape == (4,)
        assert jnp.all(pen > 0)

    def test_at_limit_is_zero(self):
        """Exactly at the limit → zero penalty (excess = 0)."""
        curve = _make_curve()
        wf = _constant_windfield(20.0, 0.0)  # TWS = 20, limit = 20
        pen = weather_penalty_smooth(curve, windfield=wf)
        assert jnp.allclose(pen, 0.0)
