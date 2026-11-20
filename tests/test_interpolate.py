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
    SegmentedCubicInterpolator,
    detect_cadence_segments,
    map_coordinates_cubic,
    map_coordinates_cubic_derivative,
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


VARIABLE_TIMESTAMPS = np.array(
    [0.0, 1.0, 2.0, 3.0, 6.0, 9.0, 15.0, 21.0, 27.0, 29.0, 31.0]
)
VARIABLE_CADENCES = np.array([1.0, 3.0, 6.0, 2.0])


def _raw_segment_value_and_derivative(
    data: np.ndarray,
    interpolator: SegmentedCubicInterpolator,
    segment_index: int,
    timestamp: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate one raw segment independently for transition assertions."""
    segment = interpolator.segments[segment_index]
    samples = data[segment.source_start : segment.source_end + 1]
    coefficients, npad = prefilter_cubic(samples)
    coordinate = np.asarray(
        [(timestamp - segment.start_time) / segment.cadence], dtype=np.float32
    )
    value = map_coordinates_cubic(coefficients, coordinate[None, :], npad=npad)
    derivative = (
        map_coordinates_cubic_derivative(coefficients, coordinate[None, :], npad=npad)
        / segment.cadence
    )
    return np.asarray(value)[0], np.asarray(derivative)[0]


class TestSegmentedCubicInterpolator:
    """Tests for piecewise-uniform physical timestamp interpolation."""

    def test_uniform_cadence_matches_existing_and_scipy(
        self, rng: np.random.Generator
    ) -> None:
        """A uniform timestamp axis preserves the original cubic result."""
        data = rng.standard_normal(20)
        cadence = 2.5
        timestamps = np.arange(data.size) * cadence
        query = rng.uniform(timestamps[0], timestamps[-1], size=40)

        interpolator = SegmentedCubicInterpolator(timestamps, data)
        actual = np.asarray(interpolator(query))
        coefficients, npad = prefilter_cubic(data)
        expected_coordinates = np.asarray([query / cadence])
        existing = map_coordinates_cubic(coefficients, expected_coordinates, npad=npad)
        scipy_result = scipy_map_coordinates(
            data, expected_coordinates, order=3, mode="nearest"
        )

        np.testing.assert_allclose(actual, existing, atol=5e-6, rtol=1e-6)
        np.testing.assert_allclose(actual, scipy_result, atol=5e-4, rtol=1e-3)

    def test_multiple_cadences_and_newer_segment_ownership(self) -> None:
        """Every detected run owns one side of its transition."""
        data = np.sin(VARIABLE_TIMESTAMPS) + VARIABLE_TIMESTAMPS / 20.0
        interpolator = SegmentedCubicInterpolator(VARIABLE_TIMESTAMPS, data)

        np.testing.assert_allclose(
            [segment.cadence for segment in interpolator.segments],
            VARIABLE_CADENCES,
        )
        query = np.array([2.9, 3.0, 3.1, 8.9, 9.0, 9.1, 26.9, 27.0, 27.1])
        np.testing.assert_array_equal(
            interpolator.owning_segment(query), [0, 1, 1, 1, 2, 2, 2, 3, 3]
        )

    def test_value_and_derivative_are_c1_at_every_transition(
        self, rng: np.random.Generator
    ) -> None:
        """New corrected segments match the preceding value and time derivative."""
        data = rng.standard_normal(VARIABLE_TIMESTAMPS.size)
        interpolator = SegmentedCubicInterpolator(VARIABLE_TIMESTAMPS, data)

        for segment_index in range(1, len(interpolator.segments)):
            transition = interpolator.segments[segment_index].start_time
            previous_value, previous_derivative = _raw_segment_value_and_derivative(
                data, interpolator, segment_index - 1, transition
            )
            if segment_index > 1:
                previous = interpolator.segments[segment_index - 1]
                s = (transition - previous.start_time) / previous.cadence
                dy = np.asarray(interpolator._correction_values[segment_index - 1])
                dd = np.asarray(interpolator._correction_derivatives[segment_index - 1])
                if 0.0 <= s <= 1.0:
                    previous_value += (2.0 * s**3 - 3.0 * s**2 + 1.0) * dy + (
                        s**3 - 2.0 * s**2 + s
                    ) * previous.cadence * dd
                    previous_derivative += (
                        6.0 * s**2 - 6.0 * s
                    ) * dy / previous.cadence + (3.0 * s**2 - 4.0 * s + 1.0) * dd

            np.testing.assert_allclose(
                interpolator(transition), previous_value, atol=2e-6, rtol=2e-6
            )
            np.testing.assert_allclose(
                interpolator.derivative(transition),
                previous_derivative,
                atol=2e-6,
                rtol=2e-6,
            )

    def test_corrections_rejoin_raw_new_segments(self) -> None:
        """Correction value and derivative vanish after one new cadence."""
        data = np.cos(VARIABLE_TIMESTAMPS) + 0.07 * VARIABLE_TIMESTAMPS
        interpolator = SegmentedCubicInterpolator(VARIABLE_TIMESTAMPS, data)

        for segment_index in range(1, len(interpolator.segments)):
            segment = interpolator.segments[segment_index]
            timestamp = segment.start_time + segment.cadence
            raw_value, raw_derivative = _raw_segment_value_and_derivative(
                data, interpolator, segment_index, timestamp
            )
            np.testing.assert_allclose(
                interpolator(timestamp), raw_value, atol=2e-6, rtol=2e-6
            )
            np.testing.assert_allclose(
                interpolator.derivative(timestamp),
                raw_derivative,
                atol=2e-6,
                rtol=2e-6,
            )

    def test_third_transition_uses_corrected_preceding_metadata(self) -> None:
        """Stored incoming corrections are computed from the preceding chain."""
        data = np.arange(VARIABLE_TIMESTAMPS.size, dtype=float) ** 2
        interpolator = SegmentedCubicInterpolator(VARIABLE_TIMESTAMPS, data)
        segment_index = 2
        transition = interpolator.segments[segment_index].start_time
        previous_value, previous_derivative = _raw_segment_value_and_derivative(
            data, interpolator, segment_index - 1, transition
        )
        previous = interpolator.segments[segment_index - 1]
        s = (transition - previous.start_time) / previous.cadence
        dy = np.asarray(interpolator._correction_values[segment_index - 1])
        dd = np.asarray(interpolator._correction_derivatives[segment_index - 1])
        if 0.0 <= s <= 1.0:
            previous_value += (2.0 * s**3 - 3.0 * s**2 + 1.0) * dy + (
                s**3 - 2.0 * s**2 + s
            ) * previous.cadence * dd
            previous_derivative += (6.0 * s**2 - 6.0 * s) * dy / previous.cadence + (
                3.0 * s**2 - 4.0 * s + 1.0
            ) * dd
        _, raw_new_derivative = _raw_segment_value_and_derivative(
            data, interpolator, segment_index, transition
        )
        new_source_start = interpolator.segments[segment_index].source_start

        np.testing.assert_allclose(
            interpolator._correction_values[segment_index],
            previous_value - float(data[new_source_start]),
        )
        np.testing.assert_allclose(
            interpolator._correction_derivatives[segment_index],
            previous_derivative - raw_new_derivative,
            atol=2e-6,
        )

    @pytest.mark.parametrize(
        ("timestamps", "message"),
        [
            ([0.0, 1.0, 1.0, 2.0], "strictly increasing"),
            ([0.0, 2.0, 1.0, 3.0], "strictly increasing"),
            ([0.0, 1.0, 2.0, 5.0], "too short"),
            ([0.0, 1.0, 2.0, 5.0, 6.0, 7.0], "too short"),
            ([0.0, 1.0], "at least two intervals"),
        ],
    )
    def test_invalid_timestamps_raise_clear_errors(
        self, timestamps: list[float], message: str
    ) -> None:
        """Invalid cadence and ordering inputs fail during construction."""
        with pytest.raises(ValueError, match=message):
            SegmentedCubicInterpolator(timestamps, np.arange(len(timestamps)))

    def test_rejects_data_length_mismatch(self) -> None:
        """Source length mismatches are explicit."""
        with pytest.raises(ValueError, match="incompatible"):
            SegmentedCubicInterpolator(VARIABLE_TIMESTAMPS, np.arange(3))

    def test_cadence_detection_tolerance_and_datetime_input(self) -> None:
        """Tolerance is applied in seconds to NumPy timestamp vectors."""
        times = np.datetime64("2024-01-01") + np.arange(7) * np.timedelta64(1, "h")
        times = times.astype("datetime64[s]")
        times = times + np.array([0, 0, 0, 0, 0, 0, 0], dtype="timedelta64[s]")
        segments = detect_cadence_segments(times)
        assert len(segments) == 1
        assert segments[0].cadence == 3600.0

    def test_large_absolute_timestamps_are_normalized_before_jax(self) -> None:
        """Unix-scale eager queries retain minute-scale sample precision."""
        origin = 1_700_000_000.0
        offsets = np.array([0.0, 60.0, 120.0, 180.0])
        timestamps = origin + offsets
        data = np.array([0.0, 60.0, 120.0, 180.0])
        interpolator = SegmentedCubicInterpolator(timestamps, data)
        query_offsets = np.array([0.0, 61.0, 121.0])
        coefficients, npad = prefilter_cubic(data)
        expected = map_coordinates_cubic(
            coefficients, jnp.asarray([query_offsets / 60.0]), npad=npad
        )

        np.testing.assert_allclose(
            interpolator(origin + query_offsets),
            expected,
            atol=2e-5,
        )
        np.testing.assert_array_equal(interpolator.owning_segment(timestamps), 0)
        assert not np.isclose(
            np.asarray(interpolator(origin + 60.0)),
            np.asarray(interpolator(origin + 120.0)),
            rtol=0.0,
            atol=1.0,
        )

        variable = origin + VARIABLE_TIMESTAMPS
        variable_data = np.arange(variable.size, dtype=float)
        segmented = SegmentedCubicInterpolator(variable, variable_data)
        transitions = variable[[3, 5, 8]]
        np.testing.assert_allclose(
            segmented(transitions), variable_data[[3, 5, 8]], atol=2e-6
        )
        np.testing.assert_array_equal(segmented.owning_segment(transitions), [1, 2, 3])

    def test_datetime_queries_subtract_integer_nanoseconds_first(self) -> None:
        """Fine datetime transitions are not moved by epoch cancellation."""
        offsets = np.asarray(VARIABLE_TIMESTAMPS, dtype="timedelta64[us]")
        timestamps = np.datetime64("2024-01-01") + offsets
        data = np.arange(timestamps.size, dtype=float)
        interpolator = SegmentedCubicInterpolator(timestamps, data)

        transitions = timestamps[[3, 5, 8]]
        np.testing.assert_array_equal(
            interpolator.owning_segment(transitions), [1, 2, 3]
        )
        np.testing.assert_allclose(
            interpolator(transitions), data[[3, 5, 8]], atol=2e-6
        )

    def test_jit_gradient_and_analytic_derivative(
        self, rng: np.random.Generator
    ) -> None:
        """Numeric value and time-gradient paths are JAX traceable."""
        data = rng.standard_normal(VARIABLE_TIMESTAMPS.size)
        interpolator = SegmentedCubicInterpolator(VARIABLE_TIMESTAMPS, data)
        query = jnp.array([2.75, 3.25, 8.5, 9.5, 26.5, 27.5])
        eager = interpolator(query)

        jitted = jax.jit(interpolator.evaluate_relative)(query)
        np.testing.assert_allclose(jitted, eager, atol=1e-6)

        timestamp = jnp.array(3.4)
        gradient = jax.grad(interpolator.evaluate_relative)(timestamp)
        np.testing.assert_allclose(
            gradient,
            interpolator.derivative_relative(timestamp),
            atol=2e-6,
            rtol=2e-6,
        )

        outside = jnp.array([-1.0, 32.0])
        np.testing.assert_array_equal(interpolator.derivative_relative(outside), 0.0)
        np.testing.assert_allclose(
            jax.vmap(jax.grad(interpolator.evaluate_relative))(outside),
            interpolator.derivative_relative(outside),
            atol=1e-6,
        )

    def test_spatial_dimensions_and_batched_queries(
        self, rng: np.random.Generator
    ) -> None:
        """Spatial grids retain their shapes for mixed-segment batches."""
        data = rng.standard_normal((*VARIABLE_TIMESTAMPS.shape, 4, 5))
        interpolator = SegmentedCubicInterpolator(VARIABLE_TIMESTAMPS, data)
        query = jnp.array([[1.2, 3.4, 9.5], [15.5, 27.5, 30.0]])
        latitude = jnp.array([[0.2, 1.7, 2.4], [0.8, 2.1, 3.0]])
        longitude = jnp.array([[1.1, 2.6, 4.2], [3.4, 1.0, 0.5]])
        result = interpolator(query, jnp.stack([latitude, longitude]))

        assert result.shape == query.shape
        scalar = np.asarray(interpolator(3.4, jnp.array([1.7, 2.6])))
        np.testing.assert_allclose(result[0, 1], scalar, atol=5e-6)

    @pytest.mark.parametrize("spatial", [False, True])
    def test_reproduces_every_source_timestamp(
        self, rng: np.random.Generator, spatial: bool
    ) -> None:
        """Every original timestamp reproduces its full source sample."""
        spatial_shape = (3, 4) if spatial else ()
        data = rng.standard_normal((VARIABLE_TIMESTAMPS.size, *spatial_shape))
        interpolator = SegmentedCubicInterpolator(VARIABLE_TIMESTAMPS, data)
        if spatial:
            actual = interpolator(
                VARIABLE_TIMESTAMPS[:, None, None],
                jnp.stack(
                    [
                        np.broadcast_to(np.arange(3)[:, None], (1, 3, 4)),
                        np.broadcast_to(np.arange(4)[None, :], (1, 3, 4)),
                    ]
                ),
            )
        else:
            actual = interpolator(VARIABLE_TIMESTAMPS)

        np.testing.assert_allclose(actual, data, atol=5e-4, rtol=1e-3)
