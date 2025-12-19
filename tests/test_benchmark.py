import jax.numpy as jnp
from jax import jit

from routetools.cost import cost_function
from routetools.vectorfield import vectorfield_zero


@jit  # type: ignore[misc]
def wavefield_constant(
    x: jnp.ndarray,
    y: jnp.ndarray,
    t: jnp.ndarray,
    height: float = 1.0,
    angle: float = 0.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    # return constant wave height of 1 meter and zero direction
    return height * jnp.ones_like(x), angle * jnp.ones_like(y)


def test_cost_function_constant_speed_time_variant():
    """Test cost function with constant speed and time-variant vector field."""
    # Define a simple curve (lon, lat) in degrees
    # from (0, 0) to (20, 20) with 1000 points
    num_points = 1000
    lats = jnp.linspace(0, 20, num_points)
    lons = jnp.linspace(0, 20, num_points)
    curve = jnp.stack([lons, lats], axis=1)  # shape (num_points, 2)
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


def test_low_wavefield():
    """Test cost function with constant speed and time-variant vector field,
    and wavefield."""
    # Define a simple curve (lon, lat) in degrees
    # from (0, 0) to (20, 20) with 1000 points
    num_points = 1000
    lats = jnp.linspace(0, 20, num_points)
    lons = jnp.linspace(0, 20, num_points)
    curve = jnp.stack([lons, lats], axis=1)  # shape (num_points, 2)
    curve = curve[jnp.newaxis, ...]  # shape (1, num_points, 2)

    # Define a constant speed over water in meters per second
    speed_over_water = 10.0  # m/s

    # Compute the cost
    cost = cost_function(
        vectorfield=vectorfield_zero,
        curve=curve,
        wavefield=wavefield_constant,
        travel_stw=speed_over_water,
        spherical_correction=True,
    )

    # Distance from (0, 0) to (20, 20) along the curve
    distance_km = 3112.4
    # Expected travel time in seconds
    expected_travel_time = (distance_km * 1000) / speed_over_water

    # For waves of 1 meter height, the effect on travel time should be negligible
    assert jnp.isclose(
        cost, expected_travel_time, rtol=1e-3
    ), f"Cost {cost} does not match expected {expected_travel_time}"


def test_medium_wavefield():
    """Test cost function with constant speed and time-variant vector field,
    and wavefield."""
    # Define a simple curve (lon, lat) in degrees
    # from (0, 0) to (20, 20) with 1000 points
    num_points = 1000
    lats = jnp.linspace(0, 20, num_points)
    lons = jnp.linspace(0, 20, num_points)
    curve = jnp.stack([lons, lats], axis=1)  # shape (num_points, 2)
    curve = curve[jnp.newaxis, ...]  # shape (1, num_points, 2)

    # Define a constant speed over water in meters per second
    speed_over_water = 10.0  # m/s

    # Define a wavefield function that returns 5 meter height
    def wavefield_medium(
        x: jnp.ndarray, y: jnp.ndarray, t: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        return wavefield_constant(x, y, t, height=5.0)

    # Compute the cost
    cost = cost_function(
        vectorfield=vectorfield_zero,
        curve=curve,
        wavefield=wavefield_medium,
        travel_stw=speed_over_water,
        spherical_correction=True,
    )

    # Distance from (0, 0) to (20, 20) along the curve
    distance_km = 3112.4
    # Expected travel time in seconds
    expected_travel_time = (distance_km * 1000) / speed_over_water

    # For waves of 5 meter height, the effect on travel time should be more noticeable
    assert (
        cost > expected_travel_time
    ), f"Cost {cost} should be greater than expected {expected_travel_time}"


def test_wave_directions():
    """Test cost function with wave directions affecting the cost."""
    # Define a simple curve (lon, lat) in degrees
    # from (0, 0) to (20, 0) with 1000 points
    # The ship angle is 0 degrees with respect to true north
    num_points = 1000
    lats = jnp.zeros(num_points)
    lons = jnp.linspace(0, 20, num_points)
    curve = jnp.stack([lons, lats], axis=1)  # shape (num_points, 2)
    curve = curve[jnp.newaxis, ...]  # shape (1, num_points, 2)

    # Define a constant speed over water in meters per second
    speed_over_water = 10.0  # m/s

    # Define a wavefield function that returns 3 meter height and direction
    def wavefield_parallel(
        x: jnp.ndarray, y: jnp.ndarray, t: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        return wavefield_constant(x, y, t, height=5.0, angle=0.0)

    # Compute the cost
    cost_parallel = cost_function(
        vectorfield=vectorfield_zero,
        curve=curve,
        wavefield=wavefield_parallel,
        travel_stw=speed_over_water,
        spherical_correction=True,
    )

    # Define a wavefield function that returns 3 meter height and 90 degree direction
    def wavefield_orthogonal(
        x: jnp.ndarray, y: jnp.ndarray, t: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        return wavefield_constant(x, y, t, height=5.0, angle=90.0)

    # Compute the cost
    cost_orthogonal = cost_function(
        vectorfield=vectorfield_zero,
        curve=curve,
        wavefield=wavefield_orthogonal,
        travel_stw=speed_over_water,
        spherical_correction=True,
    )

    # We expect that sailing parallel to the waves results in lower cost
    # than sailing orthogonal to the waves
    assert cost_parallel < cost_orthogonal, (
        f"Cost with parallel waves {cost_parallel} should be less than "
        f"cost with orthogonal waves {cost_orthogonal}"
    )

    # Now test with diagonal wave direction (45 degrees)
    def wavefield_diagonal(
        x: jnp.ndarray, y: jnp.ndarray, t: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        return wavefield_constant(x, y, t, height=5.0, angle=45.0)

    cost_diagonal = cost_function(
        vectorfield=vectorfield_zero,
        curve=curve,
        wavefield=wavefield_diagonal,
        travel_stw=speed_over_water,
        spherical_correction=True,
    )

    # We expect that sailing diagonal to the waves results in cost
    # between parallel and orthogonal
    assert cost_parallel < cost_diagonal < cost_orthogonal, (
        f"Cost with diagonal waves {cost_diagonal} should be between "
        f"cost with parallel waves {cost_parallel} and "
        f"cost with orthogonal waves {cost_orthogonal}"
    )

    # Now test going against the waves (180 degrees)
    def wavefield_against(
        x: jnp.ndarray, y: jnp.ndarray, t: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        return wavefield_constant(x, y, t, height=5.0, angle=180.0)

    cost_against = cost_function(
        vectorfield=vectorfield_zero,
        curve=curve,
        wavefield=wavefield_against,
        travel_stw=speed_over_water,
        spherical_correction=True,
    )

    # We expect that sailing against the waves results in the highest cost
    assert cost_against > cost_orthogonal, (
        f"Cost with against waves {cost_against} should be greater than "
        f"cost with orthogonal waves {cost_orthogonal}"
    )
