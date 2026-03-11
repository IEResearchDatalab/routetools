"""Tests for reroute_around_land in routetools.wrr_bench.polygons."""

import numpy as np
import pytest
from shapely.geometry import LineString, MultiPolygon, Polygon

from routetools.wrr_bench.polygons import reroute_around_land

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def vertical_strip() -> MultiPolygon:
    """A vertical rectangular land strip centred at lon=2.5, spanning lat ±1."""
    return MultiPolygon([Polygon([(2.0, -1.0), (3.0, -1.0), (3.0, 1.0), (2.0, 1.0)])])


@pytest.fixture()
def route_no_crossing() -> np.ndarray:
    """Five waypoints that stay well clear of the vertical strip."""
    return np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.4, 0.0], [1.6, 0.0]])


@pytest.fixture()
def route_crossing(vertical_strip) -> np.ndarray:
    """Route that travels east at lat=0 directly through the vertical strip."""
    return np.array([[0.0, 0.0], [1.0, 0.0], [2.5, 0.0], [4.0, 0.0], [5.0, 0.0]])


# ---------------------------------------------------------------------------
# Basic contract tests
# ---------------------------------------------------------------------------


def test_no_crossing_returns_same(route_no_crossing, vertical_strip):
    """Route with no crossings must be returned unchanged."""
    result = reroute_around_land(route_no_crossing, vertical_strip, dilation=0.05)
    np.testing.assert_array_equal(result, route_no_crossing)


def test_output_same_shape(route_crossing, vertical_strip):
    """Output must have the same shape as the input."""
    result = reroute_around_land(route_crossing, vertical_strip)
    assert result.shape == route_crossing.shape


def test_anchor_points_preserved(route_crossing, vertical_strip):
    """Anchor points (last water before / first water after crossing) are kept."""
    dilation = 0.05  # small enough that route[0] and route[4] are safe anchors
    result = reroute_around_land(route_crossing, vertical_strip, dilation=dilation)
    # route[0] and route[4] are well outside the strip + dilation → unchanged
    np.testing.assert_array_almost_equal(result[0], route_crossing[0])
    np.testing.assert_array_almost_equal(result[-1], route_crossing[-1])


def test_bypass_avoids_land(route_crossing, vertical_strip):
    """After rerouting, no route segment should cross land interior."""
    result = reroute_around_land(route_crossing, vertical_strip, dilation=0.0)
    for i in range(len(result) - 1):
        seg = LineString([tuple(result[i]), tuple(result[i + 1])])
        assert not seg.crosses(
            vertical_strip
        ), f"Segment {i}→{i + 1} still crosses land after rerouting"
        assert not seg.within(
            vertical_strip
        ), f"Segment {i}→{i + 1} is still fully within land after rerouting"


def test_bypass_avoids_dilated_land(route_crossing, vertical_strip):
    """Bypass segments should not cross the dilated land interior."""
    dilation = 0.1
    result = reroute_around_land(route_crossing, vertical_strip, dilation=dilation)
    dilated = vertical_strip.buffer(dilation, join_style=2)
    for i in range(len(result) - 1):
        seg = LineString([tuple(result[i]), tuple(result[i + 1])])
        assert not seg.crosses(
            dilated
        ), f"Segment {i}→{i + 1} still crosses dilated land after rerouting"
        assert not seg.within(
            dilated
        ), f"Segment {i}→{i + 1} is still fully inside dilated land after rerouting"


def test_single_polygon_input(route_crossing):
    """Works with a plain Polygon (not wrapped in MultiPolygon)."""
    land = Polygon([(2.0, -1.0), (3.0, -1.0), (3.0, 1.0), (2.0, 1.0)])
    result = reroute_around_land(route_crossing, land)
    assert result.shape == route_crossing.shape


def test_numpy_accepts_jax_array(route_crossing, vertical_strip):
    """Function accepts JAX arrays (converted internally via np.asarray)."""
    pytest.importorskip("jax")
    import jax.numpy as jnp

    jax_route = jnp.array(route_crossing)
    result = reroute_around_land(jax_route, vertical_strip)
    assert result.shape == route_crossing.shape


def test_no_land_polygon_is_noop():
    """An empty MultiPolygon (no land) returns the route unchanged."""
    land = MultiPolygon()
    route = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    result = reroute_around_land(route, land)
    np.testing.assert_array_equal(result, route)


# ---------------------------------------------------------------------------
# Edge-case tests
# ---------------------------------------------------------------------------


def test_crossing_run_touching_start_is_handled():
    """A crossing run that starts at segment 0 is still rerouted safely."""
    land = MultiPolygon([Polygon([(0.0, -1.0), (1.5, -1.0), (1.5, 1.0), (0.0, 1.0)])])
    route = np.array([[0.5, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    result = reroute_around_land(route, land, dilation=0.0)
    assert result.shape == route.shape


def test_multiple_crossing_segments_all_replaced():
    """Two disjoint land strips both crossed by the route are both rerouted."""
    land = MultiPolygon(
        [
            Polygon([(1.0, -1.0), (2.0, -1.0), (2.0, 1.0), (1.0, 1.0)]),
            Polygon([(4.0, -1.0), (5.0, -1.0), (5.0, 1.0), (4.0, 1.0)]),
        ]
    )
    route = np.array(
        [
            [0.0, 0.0],
            [1.5, 0.0],  # inside first strip
            [3.0, 0.0],
            [4.5, 0.0],  # inside second strip
            [6.0, 0.0],
        ]
    )
    result = reroute_around_land(route, land, dilation=0.0)
    assert result.shape == route.shape
    # Verify neither land strip is crossed in the result
    for strip in land.geoms:
        for i in range(len(result) - 1):
            seg = LineString([tuple(result[i]), tuple(result[i + 1])])
            assert not seg.crosses(
                strip
            ), f"Segment {i}→{i + 1} still crosses land strip after rerouting"
            assert not seg.within(
                strip
            ), f"Segment {i}→{i + 1} is still fully within land strip after rerouting"
