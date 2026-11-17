"""Verify that the source audit handles different longitude origins."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

scripts = Path(__file__).parents[1] / "scripts/data"
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location(
    "coordinate_audit", scripts / "audit_forecast_coordinates.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize(
    "native,target,expected",
    [
        ([-180.0, -179.75, -179.5, -179.25], [180.25, 180.5], [1, 2]),
        ([179.5, 179.75, 180.0, 180.25], [-180.0, -179.75], [2, 3]),
    ],
)
def test_longitude_origin_is_matched_geographically(native, target, expected):
    xx, yy = np.meshgrid(native, [60.0, 59.75])
    rows, cols = module.coordinate_indices(yy, xx, np.array([59.75]), np.array(target))
    np.testing.assert_array_equal(rows, [1])
    np.testing.assert_array_equal(cols, expected)
