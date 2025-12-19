import jax.numpy as jnp

from routetools.cost import cost_function
from routetools.vectorfield import vectorfield_zero


def test_cost_function_constant_speed_time_variant():
    """Test cost function with constant speed and time-variant vector field."""
    # Define a simple curve (lat, lon) in degrees
    # from (0, 0) to (20, 20) with 1000 points
    num_points = 1000
    lats = jnp.linspace(0, 20, num_points)
    lons = jnp.linspace(0, 20, num_points)
    curve = jnp.stack([lats, lons], axis=1)  # shape (num_points, 2)
    curve = curve[jnp.newaxis, ...]  # shape (1, num_points, 2)

    # Define a constant speed over water in meters per second
    speed_over_water = 10.0  # m/s

    # Compute the cost
    cost = cost_function(
        vectorfield=vectorfield_zero,
        curve=curve,
        travel_stw=speed_over_water,
        spherical_correction=True,
    )

    # Distance from (0, 0) to (20, 20) along the curve
    distance_km = 3112.4
    # Expected travel time in seconds
    expected_travel_time = (distance_km * 1000) / speed_over_water

    # Assert that the computed cost is close to the expected travel time
    assert jnp.isclose(
        cost, expected_travel_time, rtol=1e-3
    ), f"Cost {cost} does not match expected {expected_travel_time}"
