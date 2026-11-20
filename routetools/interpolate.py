"""Cubic B-spline interpolation for JAX (JIT-compatible).

This module provides a drop-in replacement for
``jax.scipy.ndimage.map_coordinates`` with ``order=3`` (cubic), which JAX
does not natively support.

The approach mirrors how SciPy implements cubic spline interpolation:

1. **Prefilter** the data once (outside JIT) using
   ``scipy.ndimage.spline_filter`` to convert raw samples into B-spline
   coefficients.  The data is padded by 12 elements per side (matching
   SciPy's internal ``_prepad_for_spline_filter`` for ``mode='nearest'``)
   so that boundary coefficients are computed identically to
   ``scipy.ndimage.map_coordinates``.
2. **Evaluate** the cubic B-spline at arbitrary fractional coordinates
   using pure JAX arithmetic — fully JIT-traceable and differentiable.
   Coordinates are shifted internally to account for the padding.

Usage
-----
>>> import numpy as np
>>> from routetools.interpolate import prefilter_cubic, map_coordinates_cubic
>>> data = np.random.randn(10, 20, 30).astype(np.float32)
>>> coeffs, npad = prefilter_cubic(data)    # once, outside JIT
>>> coords = jnp.array([[1.5, 2.3], [4.7, 8.1], [10.2, 25.6]])
>>> values = map_coordinates_cubic(coeffs, coords, npad=npad)  # inside JIT
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
from scipy.ndimage import spline_filter

# SciPy pads by 12 for mode='nearest' before prefiltering.
_NPAD = 12
DEFAULT_CADENCE_RTOL = 1e-9
DEFAULT_CADENCE_ATOL = 1e-9


@dataclass(frozen=True)
class CadenceSegment:
    """Metadata for one uniform-cadence interpolation segment.

    Timestamps are stored as seconds relative to the first input timestamp.
    Source ranges are inclusive, and adjacent segments share the transition
    sample.  The final sample of each non-final segment is therefore a
    one-sample right halo for its raw spline.
    """

    start_time: float
    end_time: float
    cadence: float
    source_start: int
    source_end: int
    time_halo_before: int
    time_halo_after: int
    prefilter_padding: int = _NPAD


def prefilter_cubic(data: np.ndarray) -> tuple[jnp.ndarray, int]:
    """Convert raw samples to cubic B-spline coefficients.

    The data is edge-padded by 12 elements per side before filtering,
    replicating the internal behaviour of ``scipy.ndimage.map_coordinates``
    for ``mode='nearest'``.

    Parameters
    ----------
    data : np.ndarray
        Input array of any dimensionality (float32 or float64).

    Returns
    -------
    coeffs : jnp.ndarray
        Padded B-spline coefficient array, ready to be passed to
        :func:`map_coordinates_cubic`.  Shape is ``data.shape + 2*npad``
        per axis.
    npad : int
        Amount of padding applied per side.  Must be forwarded to
        :func:`map_coordinates_cubic`.
    """
    data_np = np.asarray(data, dtype=np.float64)
    padded = np.pad(data_np, _NPAD, mode="edge")
    coeffs = spline_filter(padded, order=3, output=np.float64, mode="nearest")
    return jnp.array(coeffs, dtype=jnp.float32), _NPAD


def _cubic_bspline_weights(
    t: jnp.ndarray,
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:
    """Compute cubic B-spline basis weights for fractional position *t*.

    Parameters
    ----------
    t : jnp.ndarray
        Fractional part in ``[0, 1)``, arbitrary shape.

    Returns
    -------
    w_m1, w_0, w_1, w_2
        Weights for indices ``floor - 1, floor, floor + 1, floor + 2``.
    """
    t2 = t * t
    t3 = t2 * t
    one_minus_t = 1.0 - t

    w_m1 = one_minus_t * one_minus_t * one_minus_t / 6.0
    w_0 = (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0
    w_1 = (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0
    w_2 = t3 / 6.0

    return w_m1, w_0, w_1, w_2


def _cubic_bspline_weight_derivatives(
    t: jnp.ndarray,
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:
    """Compute derivatives of the cubic B-spline weights with respect to *t*."""
    t2 = t * t
    dw_m1 = -((1.0 - t) ** 2) / 2.0
    dw_0 = (9.0 * t2 - 12.0 * t) / 6.0
    dw_1 = (-9.0 * t2 + 6.0 * t + 3.0) / 6.0
    dw_2 = t2 / 2.0
    return dw_m1, dw_0, dw_1, dw_2


def _coordinates_and_weights(
    coeffs: jnp.ndarray,
    coordinates: jnp.ndarray,
    npad: int,
    derivative_axis: int | None,
) -> tuple[list[list[jnp.ndarray]], list[list[jnp.ndarray]]]:
    """Return clipped stencil indices and basis weights for every axis."""
    shape = jnp.array(coeffs.shape)
    all_weights: list[list[jnp.ndarray]] = []
    all_indices: list[list[jnp.ndarray]] = []
    for dimension in range(coeffs.ndim):
        coordinate = jnp.clip(
            coordinates[dimension] + npad, 0.0, shape[dimension] - 1.0
        )
        floor = jnp.floor(coordinate).astype(jnp.int32)
        fraction = coordinate - floor
        weights = _cubic_bspline_weights(fraction)
        if derivative_axis == dimension:
            weights = _cubic_bspline_weight_derivatives(fraction)
        indices = (
            jnp.clip(floor - 1, 0, shape[dimension] - 1),
            jnp.clip(floor, 0, shape[dimension] - 1),
            jnp.clip(floor + 1, 0, shape[dimension] - 1),
            jnp.clip(floor + 2, 0, shape[dimension] - 1),
        )
        all_weights.append(list(weights))
        all_indices.append(list(indices))
    return all_weights, all_indices


def map_coordinates_cubic(
    coeffs: jnp.ndarray,
    coordinates: jnp.ndarray,
    *,
    npad: int = _NPAD,
) -> jnp.ndarray:
    """Evaluate a pre-filtered cubic B-spline at fractional coordinates.

    This function is fully JIT-traceable and produces results that match
    ``scipy.ndimage.map_coordinates(data, coords, order=3, mode='nearest')``
    when *coeffs* was produced by :func:`prefilter_cubic`.

    Parameters
    ----------
    coeffs : jnp.ndarray
        Pre-filtered, **padded** B-spline coefficient array as returned by
        :func:`prefilter_cubic`.
    coordinates : jnp.ndarray
        Query coordinates of shape ``(ndim, N)`` where ``ndim`` matches the
        number of dimensions of the **original** (unpadded) data and ``N``
        is the number of query points.  Coordinates are in original
        grid-index space (i.e. fractional indices into the *unpadded* data).
    npad : int, keyword-only
        Padding applied per side.  Must match the second element of the
        tuple returned by :func:`prefilter_cubic`.

    Returns
    -------
    jnp.ndarray
        Interpolated values, shape ``(N,)``.
    """
    all_weights, all_indices = _coordinates_and_weights(
        coeffs, coordinates, npad, derivative_axis=None
    )
    return _evaluate_stencil(coeffs, all_weights, all_indices)


def map_coordinates_cubic_derivative(
    coeffs: jnp.ndarray,
    coordinates: jnp.ndarray,
    *,
    axis: int = 0,
    npad: int = _NPAD,
) -> jnp.ndarray:
    """Evaluate an analytic cubic B-spline derivative.

    The result is differentiated with respect to the requested original
    grid coordinate.  To obtain a physical-time derivative, divide by the
    physical time interval represented by one coordinate unit.

    Parameters
    ----------
    coeffs : jnp.ndarray
        Padded coefficients returned by :func:`prefilter_cubic`.
    coordinates : jnp.ndarray
        Query coordinates with shape ``(ndim, N)``.
    axis : int, optional
        Coordinate axis to differentiate.  Default is 0.
    npad : int, optional
        Prefilter padding.

    Returns
    -------
    jnp.ndarray
        Derivative values with shape ``(N,)``.
    """
    if axis < 0 or axis >= coeffs.ndim:
        raise ValueError(f"axis {axis} is outside the {coeffs.ndim}-D input")
    all_weights, all_indices = _coordinates_and_weights(
        coeffs, coordinates, npad, derivative_axis=axis
    )
    result = _evaluate_stencil(coeffs, all_weights, all_indices)
    original_size = coeffs.shape[axis] - 2 * npad
    in_domain = (coordinates[axis] >= 0.0) & (coordinates[axis] <= original_size - 1.0)
    return jnp.where(in_domain, result, jnp.zeros_like(result))


def _evaluate_stencil(
    coeffs: jnp.ndarray,
    all_weights: list[list[jnp.ndarray]],
    all_indices: list[list[jnp.ndarray]],
) -> jnp.ndarray:
    """Evaluate all Cartesian products in a cubic interpolation stencil."""
    result = jnp.zeros_like(all_weights[0][0])
    for stencil in product(range(4), repeat=coeffs.ndim):
        weight = jnp.ones_like(result)
        indices = []
        for dimension, offset in enumerate(stencil):
            weight = weight * all_weights[dimension][offset]
            indices.append(all_indices[dimension][offset])
        result = result + weight * coeffs[tuple(indices)]
    return result


def _timestamps_to_relative_seconds(
    timestamps: Sequence[object] | np.ndarray,
) -> tuple[np.ndarray, float | int, bool]:
    """Validate timestamp dtype and convert it to seconds relative to item zero."""
    values = np.asarray(timestamps)
    if values.ndim != 1:
        raise ValueError("timestamps must be a one-dimensional vector")
    if values.size < 2:
        raise ValueError("at least two timestamps are required")

    is_datetime = np.issubdtype(values.dtype, np.datetime64)
    is_timedelta = np.issubdtype(values.dtype, np.timedelta64)
    origin: float | int
    if is_datetime:
        nanoseconds = values.astype("datetime64[ns]").astype(np.int64)
        if np.any(np.isnat(values)):
            raise ValueError("timestamps must not contain NaT")
        origin = int(nanoseconds[0])
        seconds = (nanoseconds - nanoseconds[0]) / 1e9
        return seconds.astype(np.float64), origin, True
    if is_timedelta:
        nanoseconds = values.astype("timedelta64[ns]").astype(np.int64)
        if np.any(np.isnat(values)):
            raise ValueError("timestamps must not contain NaT")
        origin = int(nanoseconds[0])
        seconds = (nanoseconds - nanoseconds[0]) / 1e9
        return seconds.astype(np.float64), origin, True

    try:
        numeric = values.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "timestamps must be numeric seconds or a NumPy datetime/timedelta array"
        ) from exc
    if not np.all(np.isfinite(numeric)):
        raise ValueError("timestamps must contain only finite values")
    origin = float(numeric[0])
    return numeric - origin, origin, False


def detect_cadence_segments(
    timestamps: Sequence[object] | np.ndarray,
    *,
    rtol: float = DEFAULT_CADENCE_RTOL,
    atol: float = DEFAULT_CADENCE_ATOL,
) -> tuple[CadenceSegment, ...]:
    """Detect contiguous uniform-cadence runs in a timestamp vector.

    Adjacent segments share the transition timestamp in their source ranges.
    Cadence intervals are uniform according to :func:`numpy.isclose` with the
    supplied tolerances.  Every segment must contain at least two intervals
    (three source samples) so that a short cadence run cannot be mistaken for
    an isolated timestamp.

    Parameters
    ----------
    timestamps : sequence
        Strictly increasing numeric seconds or NumPy datetime/timedelta values.
    rtol, atol : float, optional
        Relative and absolute cadence tolerance in seconds.

    Returns
    -------
    tuple[CadenceSegment, ...]
        Segment metadata in chronological order.
    """
    seconds, _origin, _is_datetime = _timestamps_to_relative_seconds(timestamps)
    intervals = np.diff(seconds)
    nonpositive = np.flatnonzero(intervals <= 0.0)
    if nonpositive.size:
        index = int(nonpositive[0]) + 1
        raise ValueError(
            "timestamps must be strictly increasing; "
            f"duplicate or decreasing value at index {index}"
        )

    runs: list[tuple[int, int, float]] = []
    run_start = 0
    reference = float(intervals[0])
    for index in range(1, intervals.size):
        if not np.isclose(intervals[index], reference, rtol=rtol, atol=atol):
            cadence = float(np.median(intervals[run_start:index]))
            runs.append((run_start, index, cadence))
            run_start = index
            reference = float(intervals[index])
    runs.append(
        (
            run_start,
            intervals.size,
            float(np.median(intervals[run_start : intervals.size])),
        )
    )

    if len(runs) > 1 and any(end - start < 2 for start, end, _ in runs):
        short_run = next(run for run in runs if run[1] - run[0] < 2)
        raise ValueError(
            "each cadence segment must contain at least two intervals "
            f"(three samples); interval run {short_run[0]}:{short_run[1]} is "
            "too short"
        )
    if len(runs) == 1 and intervals.size < 2:
        raise ValueError(
            "a uniform interpolation segment must contain at least two intervals "
            "(three samples)"
        )

    segments = []
    for segment_index, (interval_start, interval_end, cadence) in enumerate(runs):
        source_start = 0 if segment_index == 0 else interval_start
        source_end = interval_end
        segments.append(
            CadenceSegment(
                start_time=float(seconds[source_start]),
                end_time=float(seconds[source_end]),
                cadence=cadence,
                source_start=source_start,
                source_end=source_end,
                time_halo_before=0,
                time_halo_after=int(segment_index < len(runs) - 1),
            )
        )
    return tuple(segments)


def _evaluate_coefficient_batch(
    coefficients: jnp.ndarray,
    coordinates: jnp.ndarray,
    segment_ids: jnp.ndarray,
    coefficient_shapes: jnp.ndarray,
    *,
    npad: int,
    derivative_axis: int | None = None,
) -> jnp.ndarray:
    """Evaluate one selected coefficient array per query point."""
    selected_shapes = coefficient_shapes[segment_ids]
    all_weights: list[list[jnp.ndarray]] = []
    all_indices: list[list[jnp.ndarray]] = []
    for dimension in range(coordinates.shape[0]):
        padded_coordinate = coordinates[dimension] + npad
        upper = selected_shapes[:, dimension] - 1
        padded_coordinate = jnp.clip(padded_coordinate, 0.0, upper)
        floor = jnp.floor(padded_coordinate).astype(jnp.int32)
        fraction = padded_coordinate - floor
        weights = _cubic_bspline_weights(fraction)
        if derivative_axis == dimension:
            weights = _cubic_bspline_weight_derivatives(fraction)
        all_weights.append(list(weights))
        all_indices.append(
            [
                jnp.clip(floor - 1, 0, upper),
                jnp.clip(floor, 0, upper),
                jnp.clip(floor + 1, 0, upper),
                jnp.clip(floor + 2, 0, upper),
            ]
        )

    result = jnp.zeros(coordinates.shape[1], dtype=coefficients.dtype)
    for stencil in product(range(4), repeat=coordinates.shape[0]):
        weight = jnp.ones_like(result)
        indices = [segment_ids]
        for dimension, offset in enumerate(stencil):
            weight = weight * all_weights[dimension][offset]
            indices.append(all_indices[dimension][offset])
        result = result + weight * coefficients[tuple(indices)]
    return result


def _contract_time_coefficient(
    coefficients: jnp.ndarray,
    coordinate: float,
    *,
    derivative: bool,
    npad: int,
) -> jnp.ndarray:
    """Contract the time axis of one padded spline coefficient array."""
    padded_coordinate = jnp.clip(
        jnp.asarray(coordinate + npad), 0.0, coefficients.shape[0] - 1.0
    )
    floor = jnp.floor(padded_coordinate).astype(jnp.int32)
    fraction = padded_coordinate - floor
    weights = (
        _cubic_bspline_weight_derivatives(fraction)
        if derivative
        else _cubic_bspline_weights(fraction)
    )
    indices = [
        jnp.clip(floor + offset, 0, coefficients.shape[0] - 1)
        for offset in (-1, 0, 1, 2)
    ]
    result = jnp.zeros_like(coefficients[0])
    for weight, index in zip(weights, indices, strict=True):
        result = result + weight * coefficients[index]
    return result


class SegmentedCubicInterpolator:
    """Cubic B-spline interpolation over piecewise-uniform timestamps.

    Construction, cadence detection, prefiltering, and transition correction
    coefficients are host-side.  Numeric query evaluation is JAX-compatible and
    can be JIT-compiled or differentiated with respect to query coordinates.
    Exact transition timestamps are owned by the newer segment.

    Python and NumPy numeric queries are absolute seconds and are normalized
    on the host. JAX array queries are already seconds relative to the first
    training timestamp; use :meth:`evaluate_relative` to state that contract
    explicitly.
    """

    def __init__(
        self,
        timestamps: Sequence[object] | np.ndarray,
        data: np.ndarray | jnp.ndarray,
        *,
        cadence_rtol: float = DEFAULT_CADENCE_RTOL,
        cadence_atol: float = DEFAULT_CADENCE_ATOL,
    ) -> None:
        relative_times, time_origin, datetime_input = _timestamps_to_relative_seconds(
            timestamps
        )
        self.segments = detect_cadence_segments(
            timestamps, rtol=cadence_rtol, atol=cadence_atol
        )
        self._time_origin = time_origin
        self._datetime_input = datetime_input
        self._cadence_rtol = cadence_rtol
        self._cadence_atol = cadence_atol

        data_array = np.asarray(data)
        if data_array.ndim < 1:
            raise ValueError("data must have at least a leading time dimension")
        if data_array.shape[0] != relative_times.size:
            raise ValueError(
                "timestamps and data are incompatible: "
                f"{relative_times.size} timestamps for {data_array.shape[0]} samples"
            )
        if not np.issubdtype(data_array.dtype, np.number):
            raise TypeError("data must contain numeric values")

        raw_coefficients: list[jnp.ndarray] = []
        for segment in self.segments:
            samples = data_array[segment.source_start : segment.source_end + 1]
            coefficients, npad = prefilter_cubic(samples)
            raw_coefficients.append(coefficients)
        self.npad = npad
        max_time_length = max(coefficient.shape[0] for coefficient in raw_coefficients)
        self._coefficient_shapes = jnp.asarray(
            [coefficient.shape for coefficient in raw_coefficients], dtype=jnp.int32
        )
        self._raw_coefficients = jnp.stack(
            [
                jnp.pad(
                    coefficient,
                    ((0, max_time_length - coefficient.shape[0]),)
                    + ((0, 0),) * (data_array.ndim - 1),
                    mode="edge",
                )
                for coefficient in raw_coefficients
            ]
        )

        zero_spatial = jnp.zeros_like(raw_coefficients[0][0])
        correction_values = [zero_spatial]
        correction_derivatives = [zero_spatial]
        for index in range(1, len(self.segments)):
            previous = self.segments[index - 1]
            current = self.segments[index]
            transition = current.start_time

            previous_coordinate = (transition - previous.start_time) / previous.cadence
            previous_value = _contract_time_coefficient(
                raw_coefficients[index - 1],
                previous_coordinate,
                derivative=False,
                npad=self.npad,
            )
            previous_derivative = (
                _contract_time_coefficient(
                    raw_coefficients[index - 1],
                    previous_coordinate,
                    derivative=True,
                    npad=self.npad,
                )
                / previous.cadence
            )
            if index > 1:
                previous_s = previous_coordinate
                if 0.0 <= previous_s <= 1.0:
                    dy = correction_values[index - 1]
                    dd = correction_derivatives[index - 1]
                    previous_value = previous_value + (
                        (2.0 * previous_s**3 - 3.0 * previous_s**2 + 1.0) * dy
                        + (previous_s**3 - 2.0 * previous_s**2 + previous_s)
                        * previous.cadence
                        * dd
                    )
                    previous_derivative = previous_derivative + (
                        (6.0 * previous_s**2 - 6.0 * previous_s) * dy / previous.cadence
                        + (3.0 * previous_s**2 - 4.0 * previous_s + 1.0) * dd
                    )

            current_value = _contract_time_coefficient(
                raw_coefficients[index],
                0.0,
                derivative=False,
                npad=self.npad,
            )
            current_derivative = (
                _contract_time_coefficient(
                    raw_coefficients[index],
                    0.0,
                    derivative=True,
                    npad=self.npad,
                )
                / current.cadence
            )
            correction_values.append(previous_value - current_value)
            correction_derivatives.append(previous_derivative - current_derivative)

        self._correction_values = jnp.stack(correction_values)
        self._correction_derivatives = jnp.stack(correction_derivatives)
        spatial_shape = self._correction_values.shape[1:]
        self._correction_shapes = jnp.broadcast_to(
            jnp.asarray(spatial_shape, dtype=jnp.int32),
            (len(self.segments), len(spatial_shape)),
        )
        dtype = self._raw_coefficients.dtype
        self._segment_starts = jnp.asarray(
            [segment.start_time for segment in self.segments], dtype=dtype
        )
        self._segment_cadences = jnp.asarray(
            [segment.cadence for segment in self.segments], dtype=dtype
        )
        self._segment_coordinate_ends = jnp.asarray(
            [
                (segment.end_time - segment.start_time) / segment.cadence
                for segment in self.segments
            ],
            dtype=dtype,
        )
        self._domain_start = dtype.type(self.segments[0].start_time)
        self._domain_end = dtype.type(self.segments[-1].end_time)

    @property
    def ndim(self) -> int:
        """Number of dimensions: time followed by all spatial dimensions."""
        return int(self._coefficient_shapes.shape[1])

    @property
    def spatial_ndim(self) -> int:
        """Number of spatial dimensions after time."""
        return self.ndim - 1

    def _query_times(self, timestamps: object) -> jnp.ndarray:
        """Normalize host timestamps, or accept already-relative JAX times."""
        if not isinstance(timestamps, jax.Array | jax.core.Tracer):
            values = np.asarray(timestamps)
            if np.issubdtype(values.dtype, np.datetime64) or np.issubdtype(
                values.dtype, np.timedelta64
            ):
                if not self._datetime_input:
                    raise TypeError(
                        "datetime-like queries require datetime-like training input"
                    )
                if np.any(np.isnat(values)):
                    raise ValueError("query timestamps must not contain NaT")
                query_nanoseconds = values.astype("datetime64[ns]").astype(np.int64)
                values = (query_nanoseconds - int(self._time_origin)) / 1e9
                return jnp.asarray(values)
            try:
                values = values.astype(np.float64)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "query timestamps must be numeric or datetime64"
                ) from exc
            return jnp.asarray(values - self._time_origin)
        return jnp.asarray(timestamps)

    def _owning_relative_segment(self, timestamps: jnp.ndarray) -> jnp.ndarray:
        """Dispatch timestamps already normalized relative to the input origin."""
        return jnp.clip(
            jnp.searchsorted(self._segment_starts, timestamps, side="right") - 1,
            0,
            len(self.segments) - 1,
        )

    def owning_segment(self, timestamps: object) -> jnp.ndarray:
        """Return the owning segment index for each timestamp.

        Host inputs are absolute; JAX inputs are relative to the first training
        timestamp. Exact transitions use the newer segment (``side='right'``).
        """
        return self._owning_relative_segment(self._query_times(timestamps))

    def owning_segment_relative(self, timestamps: jnp.ndarray) -> jnp.ndarray:
        """Return ownership for seconds relative to the first training timestamp."""
        return self._owning_relative_segment(jnp.asarray(timestamps))

    def _broadcast_relative_query(
        self,
        timestamps: object,
        spatial_coordinates: jnp.ndarray | Sequence[jnp.ndarray] | None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, tuple[int, ...]]:
        query = jnp.asarray(timestamps)
        if self.spatial_ndim == 0:
            if spatial_coordinates is not None:
                raise ValueError(
                    "one-dimensional data does not accept spatial coordinates"
                )
            broadcast_query = jnp.asarray(query)
            return broadcast_query.ravel(), broadcast_query, broadcast_query.shape

        if spatial_coordinates is None:
            raise ValueError(
                f"{self.spatial_ndim} spatial coordinate array(s) are required"
            )
        if isinstance(spatial_coordinates, tuple | list):
            spatial = jnp.stack(
                [jnp.asarray(coordinate) for coordinate in spatial_coordinates]
            )
        else:
            spatial = jnp.asarray(spatial_coordinates)
            if spatial.ndim == 1 and self.spatial_ndim == 1:
                spatial = spatial[None, :]
        if spatial.ndim == 0 or spatial.shape[0] != self.spatial_ndim:
            raise ValueError(
                "spatial_coordinates must have leading shape "
                f"({self.spatial_ndim}, ...)"
            )
        broadcast_coordinates = jnp.broadcast_arrays(query, *spatial)
        shape = broadcast_coordinates[0].shape
        return (
            jnp.stack([jnp.ravel(coordinate) for coordinate in broadcast_coordinates]),
            broadcast_coordinates[0],
            shape,
        )

    def evaluate_relative(
        self,
        relative_timestamps: jnp.ndarray,
        spatial_coordinates: jnp.ndarray | Sequence[jnp.ndarray] | None = None,
        *,
        derivative: bool = False,
    ) -> jnp.ndarray:
        """Interpolate using seconds relative to the first training timestamp.

        This is the JIT-compatible entry point. Spatial coordinates are original
        grid-index coordinates, matching :func:`map_coordinates_cubic`.
        """
        coordinates, _time_query, output_shape = self._broadcast_relative_query(
            relative_timestamps, spatial_coordinates
        )
        if self.spatial_ndim == 0:
            flat_times = coordinates
            coordinates = flat_times[None, :]
        else:
            flat_times = coordinates[0]
            spatial = coordinates[1:]

        segment_ids = self._owning_relative_segment(flat_times)
        selected_starts = self._segment_starts[segment_ids]
        selected_cadences = self._segment_cadences[segment_ids]
        local_times = (flat_times - selected_starts) / selected_cadences
        local_times = jnp.clip(
            local_times, 0.0, self._segment_coordinate_ends[segment_ids]
        )
        raw_coordinates = (
            local_times[None, :]
            if self.spatial_ndim == 0
            else jnp.vstack([local_times, spatial])
        )
        raw_values = _evaluate_coefficient_batch(
            self._raw_coefficients,
            raw_coordinates,
            segment_ids,
            self._coefficient_shapes,
            npad=self.npad,
            derivative_axis=0 if derivative else None,
        )
        if derivative:
            raw_values = raw_values / selected_cadences
            in_domain = (flat_times >= self._domain_start) & (
                flat_times <= self._domain_end
            )
            raw_values = jnp.where(in_domain, raw_values, jnp.zeros_like(raw_values))

        if self.spatial_ndim:
            correction_values = _evaluate_coefficient_batch(
                self._correction_values,
                spatial,
                segment_ids,
                self._correction_shapes,
                npad=self.npad,
            )
            correction_derivatives = _evaluate_coefficient_batch(
                self._correction_derivatives,
                spatial,
                segment_ids,
                self._correction_shapes,
                npad=self.npad,
            )
        else:
            correction_values = self._correction_values[segment_ids]
            correction_derivatives = self._correction_derivatives[segment_ids]

        s = local_times
        active = (s >= 0.0) & (s <= 1.0)
        if derivative:
            correction = (
                6.0 * s**2 - 6.0 * s
            ) * correction_values / selected_cadences + (
                3.0 * s**2 - 4.0 * s + 1.0
            ) * correction_derivatives
        else:
            correction = (2.0 * s**3 - 3.0 * s**2 + 1.0) * correction_values + (
                s**3 - 2.0 * s**2 + s
            ) * selected_cadences * correction_derivatives
        result = raw_values + jnp.where(active, correction, jnp.zeros_like(correction))
        return result.reshape(output_shape)

    def evaluate(
        self,
        timestamps: object,
        spatial_coordinates: jnp.ndarray | Sequence[jnp.ndarray] | None = None,
        *,
        derivative: bool = False,
    ) -> jnp.ndarray:
        """Interpolate host absolute timestamps or JAX relative timestamps.

        Python/NumPy numeric values are normalized in host float64 and datetime
        values are normalized from integer nanoseconds. JAX inputs are already
        relative seconds and should use :meth:`evaluate_relative` explicitly.
        """
        return self.evaluate_relative(
            self._query_times(timestamps), spatial_coordinates, derivative=derivative
        )

    def __call__(
        self,
        timestamps: object,
        spatial_coordinates: jnp.ndarray | Sequence[jnp.ndarray] | None = None,
    ) -> jnp.ndarray:
        """Evaluate interpolated values."""
        return self.evaluate(timestamps, spatial_coordinates)

    def derivative(
        self,
        timestamps: object,
        spatial_coordinates: jnp.ndarray | Sequence[jnp.ndarray] | None = None,
    ) -> jnp.ndarray:
        """Evaluate the derivative with respect to physical timestamp seconds."""
        return self.evaluate(timestamps, spatial_coordinates, derivative=True)

    def derivative_relative(
        self,
        relative_timestamps: jnp.ndarray,
        spatial_coordinates: jnp.ndarray | Sequence[jnp.ndarray] | None = None,
    ) -> jnp.ndarray:
        """Differentiate using relative physical timestamp seconds."""
        return self.evaluate_relative(
            relative_timestamps, spatial_coordinates, derivative=True
        )
