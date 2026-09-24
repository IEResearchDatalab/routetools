import json

import jax.numpy as jnp
import numpy as np
import pytest

from revision.task8_local_optimality import (
    OPTIMIZED_CASES,
    AuditSettings,
    _build_segment_derivative_function,
    _validate_experiment_manifest,
    assemble_interior_blocks,
    factor_spd_block_tridiagonal,
    scale_blocks_to_meters,
    solve_block_ldlt,
)


def _dense_block_matrix(diagonal: np.ndarray, upper: np.ndarray) -> np.ndarray:
    n_blocks = len(diagonal)
    dense = np.zeros((2 * n_blocks, 2 * n_blocks), dtype=np.float64)
    for index, block in enumerate(diagonal):
        block_slice = slice(2 * index, 2 * index + 2)
        dense[block_slice, block_slice] = block
        if index < len(upper):
            next_slice = slice(2 * index + 2, 2 * index + 4)
            dense[block_slice, next_slice] = upper[index]
            dense[next_slice, block_slice] = upper[index].T
    return dense


def test_assemble_interior_blocks_matches_dense_segment_sum():
    rng = np.random.default_rng(42)
    segment_gradients = rng.normal(size=(4, 4))
    segment_hessians = rng.normal(size=(4, 4, 4))
    segment_hessians = segment_hessians + np.swapaxes(segment_hessians, 1, 2)

    gradient, diagonal, upper = assemble_interior_blocks(
        segment_gradients,
        segment_hessians,
    )

    dense_gradient = np.zeros((5, 2))
    dense_hessian = np.zeros((10, 10))
    for index in range(4):
        dense_gradient[index : index + 2] += segment_gradients[index].reshape(2, 2)
        indices = np.arange(2 * index, 2 * index + 4)
        dense_hessian[np.ix_(indices, indices)] += segment_hessians[index]

    np.testing.assert_allclose(gradient, dense_gradient[1:-1])
    np.testing.assert_allclose(
        _dense_block_matrix(diagonal, upper),
        dense_hessian[2:-2, 2:-2],
    )


def test_block_ldlt_matches_dense_solve_after_physical_scaling():
    rng = np.random.default_rng(8)
    matrix = rng.normal(size=(8, 8))
    dense_degrees = matrix.T @ matrix + np.eye(8)
    diagonal = np.stack(
        [dense_degrees[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] for i in range(4)]
    )
    upper = np.stack(
        [
            dense_degrees[2 * i : 2 * i + 2, 2 * i + 2 : 2 * i + 4]
            for i in range(3)
        ]
    )
    # A dense random SPD matrix is not generally block tridiagonal. Build the
    # reference from the retained blocks, then add diagonal dominance.
    diagonal += 20.0 * np.eye(2)[None, ...]
    gradient = rng.normal(size=(4, 2))

    gradient_m, diagonal_m, upper_m = scale_blocks_to_meters(
        gradient,
        diagonal,
        upper,
        np.array([30.0, 40.0, 50.0, 55.0]),
    )
    dense_m = _dense_block_matrix(diagonal_m, upper_m)
    factorization = factor_spd_block_tridiagonal(
        diagonal_m,
        upper_m,
        relative_tolerance=1.0e-12,
    )

    assert factorization.is_positive_definite
    actual = solve_block_ldlt(factorization, -gradient_m)
    expected = np.linalg.solve(dense_m, -gradient_m.reshape(-1)).reshape(-1, 2)
    np.testing.assert_allclose(actual, expected, rtol=1.0e-10, atol=1.0e-8)


def test_block_ldlt_rejects_indefinite_hessian():
    diagonal = np.array([np.eye(2), np.diag([1.0, -0.1])])
    upper = np.zeros((1, 2, 2))
    factorization = factor_spd_block_tridiagonal(
        diagonal,
        upper,
        relative_tolerance=1.0e-12,
    )

    assert not factorization.is_positive_definite
    with pytest.raises(np.linalg.LinAlgError):
        solve_block_ldlt(factorization, np.ones((2, 2)))


@pytest.mark.parametrize("bad_value", [0.0, np.nan])
def test_block_ldlt_records_singular_or_nonfinite_hessian(bad_value):
    diagonal = np.array([np.diag([1.0, bad_value]), np.eye(2)])
    upper = np.zeros((1, 2, 2))

    factorization = factor_spd_block_tridiagonal(
        diagonal,
        upper,
        relative_tolerance=1.0e-12,
    )

    assert not factorization.is_positive_definite


def test_manifest_validation_binds_audit_to_final_objective(tmp_path):
    settings = AuditSettings()
    run = {
        "name": "both-corridors",
        "cases": list(OPTIMIZED_CASES),
        "weather_penalty_weight": settings.weather_penalty_weight,
        "wind_penalty_weight": settings.wind_penalty_weight,
        "wave_penalty_weight": settings.wave_penalty_weight,
        "land_distance_weight": settings.land_distance_weight,
        "land_distance_epsilon": settings.land_distance_epsilon,
        "distance_penalty_weight": settings.distance_penalty_weight,
    }
    manifest_path = tmp_path / "experiment_manifest.json"
    manifest_path.write_text(json.dumps({"runs": [run]}))

    actual_path, digest = _validate_experiment_manifest(tmp_path, settings)

    assert actual_path == manifest_path
    assert len(digest) == 64

    run["land_distance_weight"] = 50.0
    manifest_path.write_text(json.dumps({"runs": [run]}))
    with pytest.raises(ValueError, match="land_distance_weight=50"):
        _validate_experiment_manifest(tmp_path, settings)


def test_exact_segment_objective_derivatives_compile_and_are_finite():
    def windfield(lon, lat, time):
        del lat, time
        return jnp.zeros_like(lon) + 4.0, jnp.zeros_like(lon) + 1.0

    def wavefield(lon, lat, time):
        del lat, time
        return jnp.zeros_like(lon) + 1.5, jnp.zeros_like(lon) + 250.0

    settings = AuditSettings(land_distance_weight=0.0)
    derivatives = _build_segment_derivative_function(
        windfield=windfield,
        wavefield=wavefield,
        land=None,
        segment_hours=2.0,
        wps=False,
        settings=settings,
    )
    flat_segments = jnp.array(
        [
            [-20.0, 40.0, -19.8, 40.1],
            [-19.8, 40.1, -19.6, 40.2],
        ],
        dtype=jnp.float64,
    )

    values, gradients, hessians = derivatives(
        flat_segments,
        jnp.array([0.0, 2.0], dtype=jnp.float64),
    )

    assert values.shape == (2,)
    assert gradients.shape == (2, 4)
    assert hessians.shape == (2, 4, 4)
    assert np.all(np.isfinite(np.asarray(values)))
    assert np.all(np.isfinite(np.asarray(gradients)))
    assert np.all(np.isfinite(np.asarray(hessians)))
