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
    """Tests for Land.distance_penalty (EDT-based smooth penalty)."""

    @staticmethod
    def _land_with_strip():
        """Create a land with a vertical strip of land at x=0."""
        xlim = (-5, 5)
        ylim = (-5, 5)
        land = Land(
            xlim, ylim, water_level=0.5, random_seed=1, resolution=10, interpolate=0
        )
        # Overwrite: all water, then a vertical land strip at columns 48-52
        land._array = land._array.at[:, :].set(0)
        land._array = land._array.at[48:53, :].set(1)
        # Recompute EDT
        import numpy as np
        from scipy.ndimage import distance_transform_edt

        binary_land = np.asarray(land._array > land.water_level)
        land._edt = jnp.array(distance_transform_edt(~binary_land))
        return land

    def test_on_land_gives_large_penalty(self):
        """Points directly on land should produce high penalty."""
        land = self._land_with_strip()
        # Route through the land strip at x≈0
        curve = jnp.array([[[0.0, -2.0], [0.0, 0.0], [0.0, 2.0]]])
        pen = land.distance_penalty(curve, weight=1.0, epsilon=1.0)
        assert pen.shape == (1,)
        assert pen[0] > 1.0  # EDT ≈ 0 → penalty ≈ 1/eps per point

    def test_far_from_land_gives_small_penalty(self):
        """Points far from land should have very small penalty."""
        land = self._land_with_strip()
        # Route far away at x=-4
        curve = jnp.array([[[-4.0, -2.0], [-4.0, 0.0], [-4.0, 2.0]]])
        pen = land.distance_penalty(curve, weight=1.0, epsilon=1.0)
        # EDT values large → 1/(edt+1) is small
        assert pen[0] < 1.0

    def test_closer_route_penalized_more(self):
        """Route closer to land should get a higher penalty."""
        land = self._land_with_strip()
        # Near route (x ≈ -1) vs far route (x ≈ -4)
        near = jnp.array([[[-1.0, -2.0], [-1.0, 0.0], [-1.0, 2.0]]])
        far = jnp.array([[[-4.0, -2.0], [-4.0, 0.0], [-4.0, 2.0]]])
        pen_near = land.distance_penalty(near, weight=1.0, epsilon=1.0)
        pen_far = land.distance_penalty(far, weight=1.0, epsilon=1.0)
        assert pen_near[0] > pen_far[0]

    def test_weight_scales_penalty(self):
        """Doubling weight should double penalty."""
        land = self._land_with_strip()
        curve = jnp.array([[[-2.0, 0.0], [-1.0, 0.0], [0.0, 0.0]]])
        pen_w1 = land.distance_penalty(curve, weight=1.0, epsilon=1.0)
        pen_w2 = land.distance_penalty(curve, weight=2.0, epsilon=1.0)
        assert jnp.allclose(pen_w2, 2.0 * pen_w1)

    def test_batch_dimension(self):
        """distance_penalty handles batched curves correctly."""
        land = self._land_with_strip()
        curve = jnp.array(
            [
                [[-4.0, 0.0], [-3.0, 0.0], [-2.0, 0.0]],
                [[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]],
            ]
        )
        pen = land.distance_penalty(curve, weight=1.0, epsilon=1.0)
        assert pen.shape == (2,)
        # Second route is closer to / on land → higher penalty
        assert pen[1] > pen[0]
