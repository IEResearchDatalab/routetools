import numpy as np
from shapely.geometry import LineString, box
from shapely.prepared import prep

from revision.task7_landfree_export import repair_route


def test_repair_route_removes_segment_polygon_crossing():
    land = box(0.45, -0.05, 0.55, 0.05)
    prepared_land = prep(land)
    route = np.array(
        [
            [-1.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ]
    )

    repaired, changes = repair_route(
        route,
        prepared_land,
        max_nudge_deg=0.5,
        radii=50,
        angles=180,
    )

    assert changes
    assert not prepared_land.intersects(LineString(repaired))
    assert np.array_equal(repaired[0], route[0])
    assert np.array_equal(repaired[-1], route[-1])


def test_repair_route_leaves_clear_route_unchanged():
    prepared_land = prep(box(10.0, 10.0, 11.0, 11.0))
    route = np.array([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])

    repaired, changes = repair_route(route, prepared_land)

    assert changes == []
    np.testing.assert_array_equal(repaired, route)
