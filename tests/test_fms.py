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

    def test_eval_costfun_used_for_cost_best(self):
        """When eval_costfun is provided, cost_best reflects eval_costfun output."""
        src = jnp.array([0.0, 0.0])
        dst = jnp.array([6.0, 2.0])

        sentinel = 42.0

        def fixed_eval_costfun(**kwargs):
            """Always returns a fixed sentinel value regardless of the route."""
            curve = kwargs["curve"]
            return jnp.full((curve.shape[0],), sentinel)

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
            eval_costfun=fixed_eval_costfun,
        )

        # cost_best must come from eval_costfun (sentinel=42.0), not the
        # default physics cost (which would be an entirely different value).
        assert info["cost"][0] == pytest.approx(sentinel)

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
