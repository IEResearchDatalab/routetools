import numpy as np
from shapely.geometry import box
from shapely.prepared import prep

from revision.task7_land_verification import (
    _route_crosses_land,
    _unwrap_route_longitudes,
)


def test_unwrap_route_longitudes_keeps_antimeridian_segment_local():
    longitude = _unwrap_route_longitudes(np.array([179.0, -179.0]))

    np.testing.assert_allclose(longitude, np.array([179.0, 181.0]))

    # A raw planar line from +179 to -179 would cross this polygon at 0 deg.
    land_near_greenwich = box(-1.0, -1.0, 1.0, 1.0)
    crosses, extent = _route_crosses_land(
        longitude,
        np.zeros_like(longitude),
        land_near_greenwich,
        prep(land_near_greenwich),
    )

    assert not crosses
    assert extent == 0.0


def test_unwrap_route_longitudes_preserves_ordinary_route():
    longitude = _unwrap_route_longitudes(np.array([-70.0, -40.0, -10.0]))

    np.testing.assert_allclose(longitude, np.array([-70.0, -40.0, -10.0]))
