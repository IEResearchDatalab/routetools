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


class TestApplyCurveConstraints:
    def test_no_land_passes_curve_through(self):
        """Without land, the updated curve is returned unchanged."""
        curve_old = _curve()
        curve_new = curve_old + jnp.array([[[0.0, 0.0], [0.0, 1.0], [0.0, 0.0]]])

        constrained = _apply_curve_constraints(curve_new, curve_old)

        assert jnp.allclose(constrained, curve_new)

    def test_weather_violating_update_is_not_rolled_back(self):
        """Weather violations are no longer reverted here — they are handled via
        effective_cost in the main loop so FMS can escape a violating initial state."""
        curve_old = _curve()
        curve_new = curve_old + jnp.array([[[0.0, 0.0], [0.0, 1.0], [0.0, 0.0]]])

        constrained = _apply_curve_constraints(curve_new, curve_old)

        # Curve moves forward; no rollback for weather violations.
        assert jnp.allclose(constrained, curve_new)


class TestOptimizeFmsWeatherLimits:
    def test_fms_escapes_violating_initial_route(self):
        """When enforce_weather_limits=True and the initial route violates weather,
        FMS should still run to maxfevals rather than immediately stagnating."""
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

        # With always-violating weather, no feasible solution is found,
        # so early-stop must NOT trigger before maxfevals.
        assert info["niter"] == 20, (
            f"Expected 20 iters (maxfevals), got {info['niter']}. "
            "Early-stop fired before maxfevals — stagnation counter is wrong."
        )

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
