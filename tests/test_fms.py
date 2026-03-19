"""Tests for routetools.fms constraint handling."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from routetools.fms import _apply_curve_constraints, optimize_fms
from routetools.vectorfield import vectorfield_fourvortices


def _curve() -> jnp.ndarray:
    return jnp.array([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]], dtype=jnp.float32)


def _violating_windfield(lon, lat, t):
    return jnp.full_like(lon, 25.0), jnp.zeros_like(lon)


def _banded_windfield(lon, lat, t):
    tws = jnp.where(lat > 0.25, 25.0, 5.0)
    return tws, jnp.zeros_like(lon)


class _BandLand:
    def __call__(self, curve):
        x = curve[..., 0]
        y = curve[..., 1]
        return ((x > 0.75) & (x < 1.25) & (y > 0.25)).astype(jnp.int32)


class TestApplyCurveConstraints:
    def test_no_land_passes_curve_through(self):
        """Without land, the updated curve is returned unchanged."""
        curve_old = _curve()
        curve_new = curve_old + jnp.array([[[0.0, 0.0], [0.0, 1.0], [0.0, 0.0]]])

        constrained = _apply_curve_constraints(curve_new, curve_old)

        assert jnp.allclose(constrained, curve_new)

    def test_newly_invalid_weather_route_is_rolled_back(self):
        """A newly weather-invalid route should revert to the prior route."""
        curve_old = _curve()
        curve_new = curve_old + jnp.array([[[0.0, 0.0], [0.0, 1.0], [0.0, 0.0]]])

        constrained = _apply_curve_constraints(
            curve_new,
            curve_old,
            windfield=_banded_windfield,
            enforce_weather_limits=True,
            travel_time=2.0,
        )

        assert jnp.allclose(constrained, curve_old)

    def test_still_invalid_weather_route_keeps_new_value(self):
        """An already weather-invalid route should keep moving."""
        curve_old = _curve() + jnp.array([[[0.0, 0.0], [0.0, 1.0], [0.0, 0.0]]])
        curve_new = curve_old + jnp.array([[[0.0, 0.0], [0.0, -0.1], [0.0, 0.0]]])

        constrained = _apply_curve_constraints(
            curve_new,
            curve_old,
            windfield=_violating_windfield,
            enforce_weather_limits=True,
            travel_time=2.0,
        )

        assert jnp.allclose(constrained, curve_new)

    def test_newly_invalid_land_waypoint_is_rolled_back(self):
        """A valid waypoint that moves onto land should revert to the old value."""
        land = _BandLand()
        curve_old = _curve()
        curve_new = curve_old + jnp.array([[[0.0, 0.0], [0.0, 0.5], [0.0, 0.0]]])

        constrained = _apply_curve_constraints(
            curve_new,
            curve_old,
            land=land,
            penalty=1.0,
        )

        assert jnp.allclose(constrained, curve_old)

    def test_still_invalid_land_waypoint_keeps_new_value(self):
        """An already-invalid waypoint should keep moving even if still invalid."""
        land = _BandLand()
        curve_old = _curve() + jnp.array([[[0.0, 0.0], [0.0, 0.5], [0.0, 0.0]]])
        curve_new = curve_old + jnp.array([[[0.0, 0.0], [0.0, -0.1], [0.0, 0.0]]])

        constrained = _apply_curve_constraints(
            curve_new,
            curve_old,
            land=land,
            penalty=1.0,
        )

        assert jnp.allclose(constrained, curve_new)


class TestOptimizeFmsWeatherLimits:
    def test_fms_escapes_violating_initial_route(self):
        """When the initial route violates weather, FMS should still keep moving.

        Newly-invalid updates are rejected, but already-invalid routes are allowed
        to evolve so the solver can escape the violation.
        """
        src = jnp.array([0.0, 0.0])
        dst = jnp.array([6.0, 2.0])

        call_count = {"n": 0}

        def counting_violating_windfield(lon, lat, t):
            call_count["n"] += 1
            return jnp.full_like(lon, 25.0), jnp.zeros_like(lon)

        _, info = optimize_fms(
            vectorfield_fourvortices,
            src=src,
            dst=dst,
            num_curves=1,
            num_points=10,
            travel_time=5.0,
            maxfevals=20,
            patience=5,
            verbose=False,
            enforce_weather_limits=True,
            windfield=counting_violating_windfield,
        )

        assert info["niter"] > 0
        assert call_count["n"] > 0

    @pytest.mark.parametrize("enforce", [False, True])
    def test_fms_runs_without_weather_fields(self, enforce):
        """optimize_fms must not raise when enforce_weather_limits=True but no
        windfield/wavefield is given."""
        src = jnp.array([0.0, 0.0])
        dst = jnp.array([6.0, 2.0])

        _, info = optimize_fms(
            vectorfield_fourvortices,
            src=src,
            dst=dst,
            num_curves=1,
            num_points=10,
            travel_time=5.0,
            maxfevals=5,
            patience=3,
            verbose=False,
            enforce_weather_limits=enforce,
        )

        assert info["niter"] <= 5

    def test_custom_costfun_can_capture_fields_internally(self):
        """Custom FMS cost closures should work without vectorfield injection."""
        src = jnp.array([0.0, 0.0])
        dst = jnp.array([6.0, 2.0])

        def custom_cost(curve):
            return jnp.sum((curve[:, 1:, :] - curve[:, :-1, :]) ** 2, axis=(1, 2))

        _, info = optimize_fms(
            vectorfield_fourvortices,
            src=src,
            dst=dst,
            num_curves=1,
            num_points=10,
            travel_time=5.0,
            maxfevals=2,
            patience=2,
            verbose=False,
            costfun=custom_cost,
        )

        assert info["niter"] <= 2

    def test_custom_costfun_accepts_explicit_kwargs(self):
        """Custom FMS costs should receive forwarded explicit kwargs."""
        src = jnp.array([0.0, 0.0])
        dst = jnp.array([6.0, 2.0])
        seen: dict[str, float] = {}

        def custom_cost(*, curve, travel_time=None, scale=1.0, **kwargs):
            seen["scale"] = scale
            return scale * jnp.sum(
                (curve[:, 1:, :] - curve[:, :-1, :]) ** 2,
                axis=(1, 2),
            )

        _, info = optimize_fms(
            vectorfield_fourvortices,
            src=src,
            dst=dst,
            num_curves=1,
            num_points=10,
            travel_time=5.0,
            maxfevals=2,
            patience=2,
            verbose=False,
            costfun=custom_cost,
            costfun_kwargs={"scale": 2.5},
        )

        assert info["niter"] <= 2
        assert seen["scale"] == pytest.approx(2.5)

    def test_fms_accepts_initially_invalid_land_waypoint(self):
        """FMS should improve an initially invalid waypoint instead of raising."""
        land = _BandLand()
        curve_init = jnp.array(
            [[[0.0, 0.0], [1.0, 0.5], [2.0, 0.0]]],
            dtype=jnp.float32,
        )

        def zero_field(lon, lat, t):
            return jnp.zeros_like(lon), jnp.zeros_like(lat)

        curve_out, info = optimize_fms(
            zero_field,
            curve=curve_init,
            land=land,
            penalty=1.0,
            travel_time=2.0,
            damping=0.0,
            maxfevals=5,
            patience=5,
            verbose=False,
        )

        assert info["niter"] > 0
        assert float(curve_out[0, 1, 1]) < float(curve_init[0, 1, 1])


class TestFmsConvergence:
    def test_sinusoid_converges_to_straight_line(self):
        """FMS with zero wind should straighten a sinusoidal route to a straight line.

        The default physics cost ‖SOG‖² (wind = 0) is minimised by the
        shortest path, so the Euler-Lagrange equations drive every interior
        waypoint toward the straight line connecting the two endpoints.

        Convergence rate per iteration = cos(π/(L-1)).  For L=10, this is
        ~0.940, so after 100 Jacobi steps (damping=0) the initial amplitude
        of 0.5 is reduced to < 1e-3, well within the 0.05 tolerance.
        """
        n_pts = 10
        x = jnp.linspace(0.0, 6.0, n_pts)
        # Half-period sinusoid: y = 0 at both endpoints, peak amplitude 0.5
        y = 0.5 * jnp.sin(jnp.pi * x / 6.0)
        curve_init = jnp.stack([x, y], axis=-1)[None, ...]  # shape (1, n_pts, 2)

        def zero_field(lon, lat, t):
            return jnp.zeros_like(lon), jnp.zeros_like(lat)

        curve_out, _ = optimize_fms(
            zero_field,
            curve=curve_init,
            travel_time=6.0,
            damping=0.0,
            maxfevals=100,
            patience=20,
            verbose=False,
        )

        # All interior y-coordinates should be near 0 (straight line y = 0)
        interior_y = curve_out[0, 1:-1, 1]
        assert float(jnp.abs(interior_y).max()) < 0.05
