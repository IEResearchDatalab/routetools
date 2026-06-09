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

import jax.numpy as jnp
import numpy as np
from scipy.ndimage import spline_filter

# SciPy pads by 12 for mode='nearest' before prefiltering.
_NPAD = 12


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
    ndim = coeffs.ndim
    shape = jnp.array(coeffs.shape)

    # Shift coordinates into padded space and clamp.
    floors = []
    fracs = []
    for d in range(ndim):
        c = coordinates[d] + npad  # shift to padded space
        c = jnp.clip(c, 0.0, shape[d] - 1.0)
        fl = jnp.floor(c).astype(jnp.int32)
        fr = c - fl
        floors.append(fl)
        fracs.append(fr)

    # Compute per-axis weights: 4 weights per axis
    all_weights = []
    all_indices = []
    for d in range(ndim):
        w_m1, w_0, w_1, w_2 = _cubic_bspline_weights(fracs[d])
        weights_d = [w_m1, w_0, w_1, w_2]
        indices_d = [
            jnp.clip(floors[d] - 1, 0, shape[d] - 1),
            jnp.clip(floors[d], 0, shape[d] - 1),
            jnp.clip(floors[d] + 1, 0, shape[d] - 1),
            jnp.clip(floors[d] + 2, 0, shape[d] - 1),
        ]
        all_weights.append(weights_d)
        all_indices.append(indices_d)

    # Iterate over the 4^ndim stencil points
    result = jnp.zeros_like(coordinates[0])

    def _recurse(dim: int, weight_acc: jnp.ndarray, idx_acc: list[jnp.ndarray]):
        nonlocal result
        if dim == ndim:
            vals = coeffs[tuple(idx_acc)]
            result = result + weight_acc * vals
            return
        for k in range(4):
            new_weight = weight_acc * all_weights[dim][k]
            _recurse(dim + 1, new_weight, [*idx_acc, all_indices[dim][k]])

    _recurse(0, jnp.ones_like(coordinates[0]), [])

    return result
