import jax.numpy as jnp
import numpy as np

from routetools.cost import interpolate_to_constant_cost
from routetools.vectorfield import vectorfield_zero


def test_interpolate_to_constant_cost():
    src = jnp.array([0.0, 0.0])
    dst = jnp.array([10.0, 0.0])

    # Create a simple straight-line curve from src to dst
    curve = jnp.linspace(src, dst, 500)
    # Take 10 random points along the curve (including endpoints)
    indices = np.random.randint(1, curve.shape[0] - 1, size=8)
    indices = jnp.concatenate(
        [jnp.array([0]), jnp.sort(indices), jnp.array([curve.shape[0] - 1])]
    )
    curve = curve[indices]

    curve_interpolated = interpolate_to_constant_cost(
        curve=curve,
        vectorfield=vectorfield_zero,
        travel_stw=1.0,
        cost_per_segment=2.0,
    )

    # Ensure the first and last points are the same as the original curve
    assert jnp.allclose(curve_interpolated[0], curve[0]), "First point mismatch"
    assert jnp.allclose(curve_interpolated[-1], curve[-1]), "Last point mismatch"

    # Check that the distances between consecutive points are approximately equal
    distances = jnp.linalg.norm(
        curve_interpolated[1:] - curve_interpolated[:-1], axis=1
    )
    avg_distance = jnp.mean(distances)
    assert jnp.allclose(
        distances, avg_distance, atol=1e-2
    ), f"distances: {distances}, avg_distance: {avg_distance}"
