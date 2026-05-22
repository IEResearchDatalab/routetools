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


class TestDistancePenalty:
    """Tests for the EDT-based distance_penalty method."""

    @staticmethod
    def _make_land() -> Land:
        """Create a land with a known land patch in the centre."""
        xlim = [-5, 5]
        land = Land(
            xlim,
            xlim,
            water_level=0.5,
            random_seed=1,
            resolution=10,
            interpolate=0,
        )
        # All water, then add a land patch
        land._array = land._array.at[:, :].set(0)
        land._array = land._array.at[45:55, 45:55].set(1)
        # Recompute EDT after modifying the land array
        import numpy as np
        from scipy.ndimage import distance_transform_edt

        binary_land = np.asarray(land._array > land.water_level)
        land._edt = jnp.asarray(distance_transform_edt(~binary_land), dtype=jnp.float32)
        return land

    def test_penalty_on_land_higher(self):
        """Points on land should get a higher penalty than points in water."""
        land = self._make_land()
        # Curve through land (centre)
        on_land = jnp.array([[[0.0, 0.0], [0.0, 0.0]]])
        # Curve far from land (corner)
        far_water = jnp.array([[[-4.0, -4.0], [-4.0, -4.0]]])
        p_land = land.distance_penalty(on_land, weight=1.0)
        p_water = land.distance_penalty(far_water, weight=1.0)
        assert p_land.item() > p_water.item()

    def test_weight_scaling(self):
        """Penalty should scale linearly with weight."""
        land = self._make_land()
        curve = jnp.array([[[-3.0, -3.0], [-2.0, -2.0]]])
        p1 = land.distance_penalty(curve, weight=1.0)
        p5 = land.distance_penalty(curve, weight=5.0)
        assert p5.item() == pytest.approx(p1.item() * 5.0, rel=1e-5)

    def test_non_negative(self):
        """Penalty should always be non-negative."""
        land = self._make_land()
        curve = jnp.array([[[-4.5, -4.5], [4.5, 4.5]]])
        p = land.distance_penalty(curve, weight=1.0)
        assert p.item() >= 0.0
