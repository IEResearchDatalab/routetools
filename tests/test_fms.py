"""Tests for routetools.fms constraint handling."""

from __future__ import annotations

import jax.numpy as jnp

from routetools.fms import _apply_curve_constraints


def _curve() -> jnp.ndarray:
    return jnp.array([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]], dtype=jnp.float32)


def _safe_windfield(lon, lat, t):
    return jnp.full_like(lon, 10.0), jnp.zeros_like(lon)


def _violating_windfield(lon, lat, t):
    return jnp.full_like(lon, 25.0), jnp.zeros_like(lon)


def _safe_wavefield(lon, lat, t):
    return jnp.full_like(lon, 3.0), jnp.zeros_like(lon)


def _violating_wavefield(lon, lat, t):
    return jnp.full_like(lon, 8.0), jnp.zeros_like(lon)


class TestApplyCurveConstraints:
    def test_keeps_weather_safe_update(self):
        curve_old = _curve()
        curve_new = curve_old + jnp.array([[[0.0, 0.0], [0.0, 1.0], [0.0, 0.0]]])

        constrained = _apply_curve_constraints(
            curve_new,
            curve_old,
            windfield=_safe_windfield,
            wavefield=_safe_wavefield,
            enforce_weather_limits=True,
            travel_time=10.0,
            spherical_correction=True,
            time_offset=0.0,
        )

        assert jnp.allclose(constrained, curve_new)

    def test_reverts_route_when_wind_limit_is_exceeded(self):
        curve_old = _curve()
        curve_new = curve_old + jnp.array([[[0.0, 0.0], [0.0, 1.0], [0.0, 0.0]]])

        constrained = _apply_curve_constraints(
            curve_new,
            curve_old,
            windfield=_violating_windfield,
            enforce_weather_limits=True,
            travel_time=10.0,
            spherical_correction=True,
            time_offset=0.0,
        )

        assert jnp.allclose(constrained, curve_old)

    def test_reverts_route_when_wave_limit_is_exceeded(self):
        curve_old = _curve()
        curve_new = curve_old + jnp.array([[[0.0, 0.0], [0.0, 1.0], [0.0, 0.0]]])

        constrained = _apply_curve_constraints(
            curve_new,
            curve_old,
            wavefield=_violating_wavefield,
            enforce_weather_limits=True,
            travel_time=10.0,
            spherical_correction=True,
            time_offset=0.0,
        )

        assert jnp.allclose(constrained, curve_old)
