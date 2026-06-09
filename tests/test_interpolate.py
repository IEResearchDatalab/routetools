"""Tests for routetools.interpolate — cubic B-spline interpolation in JAX.

Verifies that the JAX-based cubic B-spline interpolation matches
``scipy.ndimage.map_coordinates`` with ``order=3, mode='nearest'`` to
high precision.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.ndimage import map_coordinates as scipy_map_coordinates

from routetools.interpolate import (
    map_coordinates_cubic,
    prefilter_cubic,
)


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random generator for reproducibility."""
    return np.random.default_rng(42)


# ---------------------------------------------------------------------------
# 1-D tests
# ---------------------------------------------------------------------------


class TestCubicInterp1D:
    """Cubic interpolation on 1-D arrays."""

    def test_at_grid_points(self, rng: np.random.Generator) -> None:
        """Values at integer coordinates should match the original data."""
        data = rng.standard_normal(20).astype(np.float32)
        coeffs, npad = prefilter_cubic(data)

        indices = np.arange(20, dtype=np.float32)
        coords = jnp.array([indices])
        result = map_coordinates_cubic(coeffs, coords, npad=npad)

        np.testing.assert_allclose(np.array(result), data, atol=1e-4, rtol=1e-4)

    def test_matches_scipy(self, rng: np.random.Generator) -> None:
        """Interpolated values match scipy order=3 at random query points."""
        data = rng.standard_normal(50).astype(np.float32)
        coeffs, npad = prefilter_cubic(data)

        query = rng.uniform(0, 49, size=100).astype(np.float32)
        coords_jax = jnp.array([query])
        result_jax = np.array(map_coordinates_cubic(coeffs, coords_jax, npad=npad))

        coords_scipy = np.array([query])
        result_scipy = scipy_map_coordinates(
            data, coords_scipy, order=3, mode="nearest"
        )

        np.testing.assert_allclose(result_jax, result_scipy, atol=5e-4, rtol=1e-3)


# ---------------------------------------------------------------------------
# 2-D tests
# ---------------------------------------------------------------------------


class TestCubicInterp2D:
    """Cubic interpolation on 2-D arrays."""

    def test_at_grid_points(self, rng: np.random.Generator) -> None:
        """Values at integer grid points should match original data."""
        data = rng.standard_normal((10, 12)).astype(np.float32)
        coeffs, npad = prefilter_cubic(data)

        rows, cols = np.meshgrid(np.arange(10), np.arange(12), indexing="ij")
        coords = jnp.array(
            [rows.ravel().astype(np.float32), cols.ravel().astype(np.float32)]
        )
        result = np.array(map_coordinates_cubic(coeffs, coords, npad=npad))

        np.testing.assert_allclose(result, data.ravel(), atol=1e-4, rtol=1e-4)

    def test_matches_scipy(self, rng: np.random.Generator) -> None:
        """Random 2-D queries match scipy order=3."""
        data = rng.standard_normal((15, 20)).astype(np.float32)
        coeffs, npad = prefilter_cubic(data)

        n_queries = 200
        q_row = rng.uniform(0, 14, size=n_queries).astype(np.float32)
        q_col = rng.uniform(0, 19, size=n_queries).astype(np.float32)

        coords_jax = jnp.array([q_row, q_col])
        result_jax = np.array(map_coordinates_cubic(coeffs, coords_jax, npad=npad))

        coords_scipy = np.array([q_row, q_col])
        result_scipy = scipy_map_coordinates(
            data, coords_scipy, order=3, mode="nearest"
        )

        np.testing.assert_allclose(result_jax, result_scipy, atol=5e-4, rtol=1e-3)


# ---------------------------------------------------------------------------
# 3-D tests (the ERA5 use case: time × lat × lon)
# ---------------------------------------------------------------------------


class TestCubicInterp3D:
    """Cubic interpolation on 3-D arrays (ERA5-like: T × lat × lon)."""

    def test_at_grid_points(self, rng: np.random.Generator) -> None:
        """Values at integer grid points match original data."""
        data = rng.standard_normal((8, 10, 12)).astype(np.float32)
        coeffs, npad = prefilter_cubic(data)

        tt, yy, xx = np.meshgrid(
            np.arange(8), np.arange(10), np.arange(12), indexing="ij"
        )
        coords = jnp.array(
            [
                tt.ravel().astype(np.float32),
                yy.ravel().astype(np.float32),
                xx.ravel().astype(np.float32),
            ]
        )
        result = np.array(map_coordinates_cubic(coeffs, coords, npad=npad))

        np.testing.assert_allclose(result, data.ravel(), atol=1e-4, rtol=1e-4)

    def test_matches_scipy(self, rng: np.random.Generator) -> None:
        """Random 3-D queries match scipy order=3."""
        data = rng.standard_normal((8, 10, 12)).astype(np.float32)
        coeffs, npad = prefilter_cubic(data)

        n_queries = 500
        q_t = rng.uniform(0, 7, size=n_queries).astype(np.float32)
        q_lat = rng.uniform(0, 9, size=n_queries).astype(np.float32)
        q_lon = rng.uniform(0, 11, size=n_queries).astype(np.float32)

        coords_jax = jnp.array([q_t, q_lat, q_lon])
        result_jax = np.array(map_coordinates_cubic(coeffs, coords_jax, npad=npad))

        coords_scipy = np.array([q_t, q_lat, q_lon])
        result_scipy = scipy_map_coordinates(
            data, coords_scipy, order=3, mode="nearest"
        )

        np.testing.assert_allclose(result_jax, result_scipy, atol=5e-4, rtol=1e-3)

    def test_near_boundary(self, rng: np.random.Generator) -> None:
        """Queries near array edges (clamped to nearest) match scipy."""
        data = rng.standard_normal((6, 8, 10)).astype(np.float32)
        coeffs, npad = prefilter_cubic(data)

        # Points very close to and beyond boundaries
        q_t = np.array([-0.5, 0.0, 0.1, 4.9, 5.0, 5.5], dtype=np.float32)
        q_lat = np.array([0.0, 0.5, 3.5, 7.0, 7.5, 8.0], dtype=np.float32)
        q_lon = np.array([0.0, 1.0, 5.0, 9.0, 9.5, 10.0], dtype=np.float32)

        coords_jax = jnp.array([q_t, q_lat, q_lon])
        result_jax = np.array(map_coordinates_cubic(coeffs, coords_jax, npad=npad))

        coords_scipy = np.array([q_t, q_lat, q_lon])
        result_scipy = scipy_map_coordinates(
            data, coords_scipy, order=3, mode="nearest"
        )

        np.testing.assert_allclose(result_jax, result_scipy, atol=5e-4, rtol=1e-3)


# ---------------------------------------------------------------------------
# JIT compatibility
# ---------------------------------------------------------------------------


class TestJITCompatibility:
    """Verify that map_coordinates_cubic works under jax.jit."""

    def test_jit_compiles(self, rng: np.random.Generator) -> None:
        """Function can be JIT-compiled and produces correct results."""
        data = rng.standard_normal((6, 8, 10)).astype(np.float32)
        coeffs, npad = prefilter_cubic(data)

        @jax.jit
        def interpolate(c, coords):
            return map_coordinates_cubic(c, coords, npad=npad)

        q_t = np.array([1.5, 3.2], dtype=np.float32)
        q_lat = np.array([2.7, 5.1], dtype=np.float32)
        q_lon = np.array([4.3, 7.8], dtype=np.float32)
        coords = jnp.array([q_t, q_lat, q_lon])

        result_jit = np.array(interpolate(coeffs, coords))

        result_scipy = scipy_map_coordinates(
            data, np.array([q_t, q_lat, q_lon]), order=3, mode="nearest"
        )

        np.testing.assert_allclose(result_jit, result_scipy, atol=5e-4, rtol=1e-3)

    def test_jit_matches_eager(self, rng: np.random.Generator) -> None:
        """JIT and eager modes produce identical results."""
        data = rng.standard_normal((6, 8, 10)).astype(np.float32)
        coeffs, npad = prefilter_cubic(data)

        q = jnp.array([[1.5, 3.2], [2.7, 5.1], [4.3, 7.8]])

        result_eager = map_coordinates_cubic(coeffs, q, npad=npad)

        @jax.jit
        def fn(c, coords):
            return map_coordinates_cubic(c, coords, npad=npad)

        result_jit = fn(coeffs, q)

        np.testing.assert_allclose(
            np.array(result_eager), np.array(result_jit), atol=1e-6
        )


# ---------------------------------------------------------------------------
# Prefilter sanity
# ---------------------------------------------------------------------------


class TestPrefilter:
    """Tests for the prefilter step."""

    def test_output_shape(self, rng: np.random.Generator) -> None:
        """Prefiltered output has padded shape."""
        data = rng.standard_normal((5, 7, 9)).astype(np.float32)
        coeffs, npad = prefilter_cubic(data)
        expected = tuple(s + 2 * npad for s in data.shape)
        assert coeffs.shape == expected

    def test_output_is_jax_array(self, rng: np.random.Generator) -> None:
        """Prefilter returns a JAX array."""
        data = rng.standard_normal((4, 4)).astype(np.float32)
        coeffs, _npad = prefilter_cubic(data)
        assert isinstance(coeffs, jnp.ndarray)

    def test_constant_data_unchanged(self) -> None:
        """Constant arrays should pass through prefilter unchanged."""
        data = np.full((5, 5, 5), 3.14, dtype=np.float32)
        coeffs, npad = prefilter_cubic(data)
        # All coefficients (including padding) should equal the constant
        np.testing.assert_allclose(np.array(coeffs), 3.14, atol=1e-5)
