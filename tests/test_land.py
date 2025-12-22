import jax.numpy as jnp
import pytest

from routetools.land import Land


def test_generate_land_array():
    xlim = [-5, 5]
    land = Land(xlim, xlim, random_seed=1, resolution=10)
    assert land.shape == (100, 100)
    assert land.array.max() == 1
    assert land.array.min() == 0


@pytest.mark.parametrize("water_level", [0, 1])
def test_water_level(water_level: float):
    xlim = [-5, 5]
    land = Land(xlim, xlim, water_level=water_level, random_seed=1)
    assert land.array.mean() == (1 - water_level)


def test_land_inbounds():
    xlim = [-5, 5]
    x = jnp.linspace(-5, 5, 100)
    # First generate the array
    land = Land(
        xlim, xlim, water_level=0.5, random_seed=1, resolution=10, interpolate=0
    )
    # Prepare a curve of (X, X) coordinates
    curve = jnp.stack([x, x], axis=-1)
    out = land(curve)
    expected = jnp.diag(land.array)
    # This curve should return the diagonal of the land array
    assert jnp.allclose(out, expected)


def test_land_outbounds():
    xlim = [-5, 5]
    # First generate the array
    land = Land(
        xlim, xlim, water_level=0.5, random_seed=1, resolution=10, interpolate=0
    )
    # A point outside the limits should return the closest
    out = land(jnp.array([[-6], [-5]]))
    expected = land.array[0, 0]
    assert jnp.allclose(out, expected)
    # Same in both bounds
    out = land(jnp.array([[6], [5]]))
    expected = land.array[-1, -1]
    assert jnp.allclose(out, expected)


def test_distance_to_land():
    xlim = [-5, 5]
    land = Land(
        xlim,
        xlim,
        water_level=0.5,
        random_seed=1,
        resolution=10,
        interpolate=0,
        outbounds_is_land=True,
    )
    # First turn all into water
    land._array = land._array.at[:, :].set(0)
    # Add a land patch in the center
    land._array = land._array.at[45:55, 45:55].set(1)
    # Get distance to land for a set of points
    curve = jnp.array(
        [
            [-4, -4],
            [0, 0],
            [4, 4],
            [-6, 0],  # Out of bounds
            [0, 6],  # Out of bounds
        ]
    )
    dists = land.distance_to_land(curve)
    assert dists.shape == (curve.shape[0],)
    assert jnp.all(dists >= 0)
    # We know the expected distances
    expected_dists = jnp.array(
        [
            5,  # From (-4,-4) to land patch
            0,  # From (0,0) to land patch
            5,  # From (4,4) to land patch
            0,  # When out of bounds, distance is 0
            0,  # When out of bounds, distance is 0
        ]
    )
    assert jnp.allclose(dists, expected_dists, atol=1e3)
