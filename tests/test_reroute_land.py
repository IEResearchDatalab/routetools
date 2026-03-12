"""Tests for reroute_around_land using a Land object and A* corrections."""

import jax.numpy as jnp
import numpy as np
import pytest

from routetools.land import Land, reroute_around_land


def _make_land(strips: list[tuple[float, float, float, float]]) -> Land:
    """Create deterministic Land with rectangular strip obstacles."""
    xlim = (0.0, 6.0)
    ylim = (-2.0, 2.0)
    resolution = (20, 20)

    nx = int(np.ceil(xlim[1] - xlim[0])) * resolution[0]
    ny = int(np.ceil(ylim[1] - ylim[0])) * resolution[1]
    x = np.linspace(*xlim, nx)
    y = np.linspace(*ylim, ny)

    array = np.zeros((nx, ny), dtype=float)
    for xmin, xmax, ymin, ymax in strips:
        x_mask = (x >= xmin) & (x <= xmax)
        y_mask = (y >= ymin) & (y <= ymax)
        array[np.ix_(x_mask, y_mask)] = 1.0

    return Land(
        xlim=xlim,
        ylim=ylim,
        water_level=0.5,
        resolution=resolution,
        interpolate=0,
        outbounds_is_land=True,
        land_array=jnp.array(array),
        penalize_segments=False,
        map_mode="nearest",
        map_order=0,
    )


def _segment_crosses_land(a: np.ndarray, b: np.ndarray, land: Land) -> bool:
    """Dense segment check against Land object."""
    samples = np.linspace(a, b, 200)
    is_land = np.asarray(land(samples), dtype=bool).reshape(-1)
    return bool(is_land[1:-1].any())


def _route_has_land_crossing(route: np.ndarray, land: Land) -> bool:
    """Return True if any route segment crosses land."""
    return any(
        _segment_crosses_land(route[i], route[i + 1], land)
        for i in range(len(route) - 1)
    )


def _route_has_land_crossing_except_last(route: np.ndarray, land: Land) -> bool:
    """Return True if any segment except the last one crosses land."""
    return any(
        _segment_crosses_land(route[i], route[i + 1], land)
        for i in range(len(route) - 2)
    )


@pytest.fixture()
def vertical_strip_land() -> Land:
    """Single obstacle crossing the route center line."""
    return _make_land([(2.0, 3.0, -1.0, 1.0)])


@pytest.fixture()
def route_no_crossing() -> np.ndarray:
    """Route entirely in water, far from obstacles."""
    return np.array([[0.0, -1.8], [1.0, -1.8], [2.0, -1.8], [3.0, -1.8], [4.0, -1.8]])


@pytest.fixture()
def route_crossing() -> np.ndarray:
    """Route with two interior points inside the strip obstacle."""
    return np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.2, 0.0],
            [2.8, 0.0],
            [4.0, 0.0],
            [5.0, 0.0],
        ]
    )


def test_no_crossing_returns_same(route_no_crossing, vertical_strip_land):
    """Route with no land crossings must remain unchanged."""
    result = reroute_around_land(route_no_crossing, vertical_strip_land)
    np.testing.assert_array_equal(result, route_no_crossing)


def test_output_same_shape(route_crossing, vertical_strip_land):
    """Output shape must match input route shape."""
    result = reroute_around_land(route_crossing, vertical_strip_land)
    assert result.shape == route_crossing.shape


def test_only_crossing_middle_is_replaced(route_crossing, vertical_strip_land):
    """Non-crossing anchor points stay untouched; only middle points are replaced."""
    result = reroute_around_land(route_crossing, vertical_strip_land)
    np.testing.assert_allclose(result[0], route_crossing[0])
    np.testing.assert_allclose(result[1], route_crossing[1])
    np.testing.assert_allclose(result[4], route_crossing[4])
    np.testing.assert_allclose(result[5], route_crossing[5])
    assert not np.allclose(result[2], route_crossing[2])
    assert not np.allclose(result[3], route_crossing[3])


def test_astar_correction_avoids_land(route_crossing, vertical_strip_land):
    """A* replacement yields a route with no segment crossing land."""
    result = reroute_around_land(
        route_crossing,
        vertical_strip_land,
        astar_resolution_scale=3,
    )
    assert not _route_has_land_crossing(result, vertical_strip_land)


def test_multiple_crossing_runs_are_replaced():
    """Two disjoint crossing runs are both corrected."""
    land = _make_land([(1.0, 2.0, -1.0, 1.0), (4.0, 5.0, -1.0, 1.0)])
    route = np.array(
        [
            [0.0, 0.0],
            [0.6, 0.0],
            [1.1, 0.0],
            [1.4, 0.0],
            [1.7, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [3.6, 0.0],
            [4.1, 0.0],
            [4.4, 0.0],
            [4.7, 0.0],
            [5.0, 0.0],
            [6.0, 0.0],
        ]
    )
    result = reroute_around_land(route, land, astar_resolution_scale=3)
    assert result.shape == route.shape
    assert not _route_has_land_crossing(result, land)


def test_accepts_jax_route(route_crossing, vertical_strip_land):
    """Function should accept JAX arrays as route input."""
    jax_route = jnp.array(route_crossing)
    result = reroute_around_land(jax_route, vertical_strip_land)
    assert result.shape == route_crossing.shape


def test_invalid_land_type_raises(route_crossing):
    """A Land object is required by the new API."""
    with pytest.raises(TypeError, match="land must be an instance"):
        reroute_around_land(route_crossing, land=np.zeros((2, 2)))  # type: ignore[arg-type]


def test_all_water_land_is_noop(route_crossing):
    """If land map is fully water, route is unchanged."""
    land = _make_land([])
    result = reroute_around_land(route_crossing, land)
    np.testing.assert_array_equal(result, route_crossing)


def test_endpoint_on_land_still_reroutes_middle_points():
    """When destination is on land, interior points are still rerouted."""
    land = _make_land([(4.5, 6.0, -0.5, 0.5)])
    route = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
            [5.0, 0.0],
            [6.0, 0.0],
        ]
    )

    assert bool(np.asarray(land(route[-1]), dtype=bool).item())

    result = reroute_around_land(route, land, astar_resolution_scale=3)

    np.testing.assert_allclose(result[0], route[0])
    np.testing.assert_allclose(result[-1], route[-1])
    assert not np.allclose(result[5], route[5])
    assert not _route_has_land_crossing_except_last(result, land)
