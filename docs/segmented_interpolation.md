# Piecewise-cadence weather interpolation

`SegmentedCubicInterpolator(timestamps, data)` accepts a strictly increasing
timestamp vector and an array whose first dimension is time. Numeric timestamps
are interpreted in seconds; NumPy `datetime64` values are converted to seconds
relative to the first timestamp. Remaining dimensions are spatial grid
dimensions and retain the coordinate convention of `map_coordinates_cubic`:
one fractional grid-index array per spatial dimension.

```python
interpolator = SegmentedCubicInterpolator(timestamps, data)
values = interpolator(query_timestamps, spatial_coordinates)
time_derivatives = interpolator.derivative(query_timestamps, spatial_coordinates)

# JIT-safe inputs are seconds relative to the first training timestamp.
relative_values = interpolator.evaluate_relative(relative_query_timestamps)
```

`owning_segment` exposes dispatch metadata. Exact transition timestamps belong
to the newer (right-hand) segment, so dispatch is deterministic. Each data
point is evaluated from only that segment's raw spline. Adjacent segments
share the transition source sample; this appears as a one-sample right halo in
the earlier segment and in `CadenceSegment`. Every segment is independently
SciPy-style edge-padded and cubic-prefiltered.

## Cadence detection

Consecutive timestamp differences must be strictly positive and match a run's
cadence within `rtol=1e-9` and `atol=1e-9` seconds by default. Runs may have any
number of cadence changes, but every run needs at least two intervals (three
source samples). Single-interval runs are rejected because the cadence cannot
be distinguished reliably from a timestamp anomaly. This policy makes
overlapping correction intervals impossible: every middle run lasts at least
two of its own cadences, longer than its one-cadence correction. Duplicate,
decreasing, non-finite, length-mismatched, and out-of-domain numeric timestamp
types fail during construction with `TypeError` or `ValueError`.

## Chaining and derivatives

At a transition, the incoming segment receives the specified cubic Hermite
correction over its first complete cadence interval. Construction evaluates
the previous corrected spline at the transition and stores correction
coefficient fields, so evaluation does not consult a neighboring segment and
is not query-order dependent. `derivative` is analytic with respect to physical
timestamp seconds and includes both the spline coordinate scaling and the
Hermite correction derivative.

## JAX boundary

Timestamp validation, cadence detection, SciPy prefiltering, and transition
coefficient calculation run on the host during object construction. Numeric
query evaluation uses static segment metadata and JAX operations and supports
`jit`, automatic differentiation, vectorized timestamps, and spatial batches.
Eager Python/NumPy numeric queries are absolute seconds and are normalized in
host float64 before conversion to JAX. JAX arrays, including traced values, are
seconds relative to the first training timestamp. An absolute Unix timestamp
already downcast to traced `float32` cannot recover its lost precision. Use
`evaluate_relative` and `derivative_relative` to state the JIT contract
explicitly. Datetime conversion remains host-side before `jit`. Queries outside
the source time domain retain nearest-edge spline behavior.

The ERA5 loader converts its hour-based dataset coordinates to seconds before
constructing the segmented interpolators. Existing `prefilter_cubic` and
`map_coordinates_cubic` APIs remain unchanged.
