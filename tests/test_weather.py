"""Tests for routetools.weather — weather constraint penalties."""

import jax.numpy as jnp
import pytest

from routetools.weather import (
    DEFAULT_HS_LIMIT,
    DEFAULT_TWS_LIMIT,
    RouteWeatherStats,
    evaluate_weather,
    wave_penalty_smooth,
    weather_penalty,
    weather_penalty_smooth,
    wind_penalty_smooth,
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


class TestTimeVariation:
    """Verify that weather is evaluated at the correct elapsed time."""

    @staticmethod
    def _time_varying_windfield():
        """Wind field that varies only in time (not space).

        TWS profile (piecewise-constant in time):
          t < 1 day   → TWS = 10  (below 20 m/s limit)
          1 ≤ t < 2   → TWS = 30  (above limit)
          t ≥ 2 days  → TWS = 10  (below limit)

        Implementation: u = TWS, v = 0 so TWS = |u|.
        """
        day = 86400.0  # seconds

        def _field(lon, lat, t):
            # t is elapsed time in seconds, shape (B, S)
            high = (t >= 1 * day) & (t < 2 * day)
            u = jnp.where(high, 30.0, 10.0)
            v = jnp.zeros_like(u)
            return u, v

        return _field

    def test_all_time_zero_misses_violation(self):
        """Without travel info, all segments query t=0 → no violation."""
        # 4 points → 3 segments, each ~1 day apart at constant speed
        curve = _make_curve(n_points=4)
        wf = self._time_varying_windfield()
        # No travel info → t=0 for all segments → TWS=10 → no penalty
        pen = weather_penalty(curve, windfield=wf, penalty=1.0)
        assert jnp.allclose(pen, 0.0)

    def test_with_travel_time_catches_violation(self):
        """With travel_time, middle segment is at day 1 → violation."""
        # 4 equally spaced points → 3 equal segments
        curve = _make_curve(n_points=4)
        wf = self._time_varying_windfield()
        day = 86400.0
        total_time = 3.0 * day  # 3 days total trip
        # Segment midpoints at t = 0.5, 1.5, 2.5 days
        # Only t=1.5 days (segment 2) → TWS=30 → 1 violation
        pen = weather_penalty(
            curve,
            windfield=wf,
            penalty=1.0,
            travel_time=total_time,
            spherical_correction=False,
        )
        assert jnp.allclose(pen, 1.0)

    def test_with_travel_stw_catches_violation(self):
        """With travel_stw, middle segment triggers violation."""
        # 4 points from (0,0) to (3,0) → 3 segments of length 1°
        src = jnp.array([0.0, 0.0])
        dst = jnp.array([3.0, 0.0])
        curve = jnp.linspace(src, dst, 4)[jnp.newaxis]  # (1, 4, 2)
        wf = self._time_varying_windfield()
        day = 86400.0
        # speed such that each segment of 1° takes exactly 1 day
        # segment length = 1° (unitless, no spherical correction)
        stw = 1.0 / day  # degrees per second
        # Midpoints at t = 0.5, 1.5, 2.5 days → middle violates
        pen = weather_penalty(
            curve,
            windfield=wf,
            penalty=1.0,
            travel_stw=stw,
            spherical_correction=False,
        )
        assert jnp.allclose(pen, 1.0)

    def test_smooth_penalty_time_variation(self):
        """Smooth penalty is non-zero only from the violating segment."""
        curve = _make_curve(n_points=4)
        wf = self._time_varying_windfield()
        day = 86400.0
        total_time = 3.0 * day
        # Without time info → all at t=0 → TWS=10 → pen=0
        pen_no_time = weather_penalty_smooth(
            curve,
            windfield=wf,
            penalty=1.0,
            sharpness=1.0,
        )
        # With time info → middle segment at t=1.5d → TWS=30 → pen>0
        pen_with_time = weather_penalty_smooth(
            curve,
            windfield=wf,
            penalty=1.0,
            sharpness=1.0,
            travel_time=total_time,
            spherical_correction=False,
        )
        assert jnp.allclose(pen_no_time, 0.0)
        assert pen_with_time > 0
        # Expected: excess = 30 - 20 = 10, one segment → 10² × 1 × 1 = 100
        assert jnp.allclose(pen_with_time, 100.0, atol=1e-4)

    def test_evaluate_weather_time_variation(self):
        """evaluate_weather reports correct max TWS with time info."""
        curve = _make_curve(n_points=4)
        wf = self._time_varying_windfield()
        day = 86400.0
        total_time = 3.0 * day
        stats_no_time = evaluate_weather(curve, windfield=wf)
        stats_with_time = evaluate_weather(
            curve,
            windfield=wf,
            travel_time=total_time,
            spherical_correction=False,
        )
        # Without time: all at t=0 → TWS=10 → not exceeded
        assert jnp.allclose(stats_no_time.max_tws, 10.0, atol=1e-5)
        assert not stats_no_time.tws_exceeded[0]
        # With time: max TWS = 30 → exceeded
        assert jnp.allclose(stats_with_time.max_tws, 30.0, atol=1e-5)
        assert stats_with_time.tws_exceeded[0]


class TestEdgeCases:
    """Edge cases for curve shapes."""

    def test_single_point_evaluate(self):
        """Curve with a single point (no segments) returns zeros."""
        curve = jnp.array([[[0.0, 0.0]]])  # (1, 1, 2)
        wf = _constant_windfield(25.0, 0.0)
        stats = evaluate_weather(curve, windfield=wf)
        assert stats.max_tws.shape == (1,)
        assert jnp.allclose(stats.max_tws, 0.0)
        assert not stats.tws_exceeded[0]

    def test_single_point_penalty(self):
        """weather_penalty returns zero for single-point curve."""
        curve = jnp.array([[[0.0, 0.0]]])
        wf = _constant_windfield(25.0, 0.0)
        pen = weather_penalty(curve, windfield=wf)
        assert jnp.allclose(pen, 0.0)

    def test_single_point_smooth(self):
        """weather_penalty_smooth returns zero for single-point curve."""
        curve = jnp.array([[[0.0, 0.0]]])
        wf = _constant_windfield(25.0, 0.0)
        pen = weather_penalty_smooth(curve, windfield=wf)
        assert jnp.allclose(pen, 0.0)


# ---------------------------------------------------------------------------
# Tests for split penalties (wind_penalty_smooth, wave_penalty_smooth)
# ---------------------------------------------------------------------------
class TestWindPenaltySmooth:
    """Test the split wind-only smooth penalty."""

    def test_zero_within_limits(self):
        curve = _make_curve()
        wf = _constant_windfield(10.0, 0.0)
        pen = wind_penalty_smooth(curve, windfield=wf)
        assert jnp.allclose(pen, 0.0)

    def test_nonzero_when_exceeded(self):
        curve = _make_curve(n_points=10)
        wf = _constant_windfield(25.0, 0.0)  # TWS=25 > 20
        pen = wind_penalty_smooth(curve, windfield=wf, weight=1.0)
        assert pen.item() > 0.0

    def test_weight_scaling(self):
        curve = _make_curve(n_points=10)
        wf = _constant_windfield(25.0, 0.0)
        p1 = wind_penalty_smooth(curve, windfield=wf, weight=1.0)
        p5 = wind_penalty_smooth(curve, windfield=wf, weight=5.0)
        assert jnp.allclose(p5, p1 * 5.0, atol=1e-5)

    def test_batch(self):
        curve = _make_curve(n_routes=3, n_points=10)
        wf = _constant_windfield(25.0, 0.0)
        pen = wind_penalty_smooth(curve, windfield=wf, weight=1.0)
        assert pen.shape == (3,)
        assert jnp.all(pen > 0)


class TestWavePenaltySmooth:
    """Test the split wave-only smooth penalty."""

    def test_zero_within_limits(self):
        curve = _make_curve()
        wvf = _constant_wavefield(3.0)
        pen = wave_penalty_smooth(curve, wavefield=wvf)
        assert jnp.allclose(pen, 0.0)

    def test_nonzero_when_exceeded(self):
        curve = _make_curve(n_points=10)
        wvf = _constant_wavefield(10.0)  # Hs=10 > 7
        pen = wave_penalty_smooth(curve, wavefield=wvf, weight=1.0)
        assert pen.item() > 0.0

    def test_weight_scaling(self):
        curve = _make_curve(n_points=10)
        wvf = _constant_wavefield(10.0)
        p1 = wave_penalty_smooth(curve, wavefield=wvf, weight=1.0)
        p5 = wave_penalty_smooth(curve, wavefield=wvf, weight=5.0)
        assert jnp.allclose(p5, p1 * 5.0, atol=1e-5)

    def test_batch(self):
        curve = _make_curve(n_routes=3, n_points=10)
        wvf = _constant_wavefield(10.0)
        pen = wave_penalty_smooth(curve, wavefield=wvf, weight=1.0)
        assert pen.shape == (3,)
        assert jnp.all(pen > 0)

    def test_independent_from_wind(self):
        """Wave penalty should not depend on wind field presence."""
        curve = _make_curve(n_points=10)
        wvf = _constant_wavefield(10.0)
        pen = wave_penalty_smooth(curve, wavefield=wvf, weight=1.0)
        assert pen.item() > 0.0
