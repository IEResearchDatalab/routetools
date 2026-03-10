"""Tests for scripts/validate_routes.py land-intersection validation."""

from __future__ import annotations

import numpy as np
import pytest

from shapely.geometry import Polygon
from shapely.ops import unary_union


# Import the functions under test
import importlib.util, sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "validate_routes",
    Path(__file__).resolve().parent.parent / "scripts" / "validate_routes.py",
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["validate_routes"] = _mod
_spec.loader.exec_module(_mod)

validate_track = _mod.validate_track
_interpolate_segment = _mod._interpolate_segment


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def square_land():
    """A simple square land polygon: lon ∈ [1, 3], lat ∈ [1, 3]."""
    return Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])


# ---------------------------------------------------------------------------
# Tests for _interpolate_segment
# ---------------------------------------------------------------------------
class TestInterpolateSegment:
    def test_endpoints_included(self):
        p1 = np.array([0.0, 0.0])
        p2 = np.array([10.0, 0.0])
        pts = _interpolate_segment(p1, p2, density=5)
        assert pts.shape == (6, 2)
        np.testing.assert_allclose(pts[0], p1)
        np.testing.assert_allclose(pts[-1], p2)

    def test_density_one(self):
        p1 = np.array([0.0, 0.0])
        p2 = np.array([4.0, 0.0])
        pts = _interpolate_segment(p1, p2, density=1)
        assert pts.shape == (2, 2)


# ---------------------------------------------------------------------------
# Tests for validate_track
# ---------------------------------------------------------------------------
class TestValidateTrack:
    def test_no_land_crossing(self, square_land):
        """Route passes entirely outside the land polygon."""
        waypoints = np.array([
            [0.0, 0.0],
            [0.0, 5.0],
            [5.0, 5.0],
        ])
        violations = validate_track(waypoints, square_land, density=20)
        assert violations == []

    def test_clear_land_crossing(self, square_land):
        """Route goes straight through the land polygon."""
        waypoints = np.array([
            [0.0, 2.0],
            [5.0, 2.0],
        ])
        violations = validate_track(waypoints, square_land, density=50)
        assert len(violations) == 1
        assert violations[0]["segment"] == 0
        assert violations[0]["land_fraction"] > 0

    def test_multiple_segments_one_violation(self, square_land):
        """Only the segment crossing land is flagged."""
        waypoints = np.array([
            [-1.0, 2.0],  # before land
            [0.0, 2.0],   # still before land
            [5.0, 2.0],   # crosses land
            [6.0, 2.0],   # after land
        ])
        violations = validate_track(waypoints, square_land, density=50)
        # Only segment 1→2 (from [0,2] to [5,2]) crosses land
        assert len(violations) == 1
        assert violations[0]["from_idx"] == 1

    def test_boundary_touching(self, square_land):
        """Segment touching the land boundary should be detected."""
        waypoints = np.array([
            [1.0, 0.0],
            [1.0, 2.0],  # runs along the left edge of the square
        ])
        violations = validate_track(waypoints, square_land, density=50)
        # Endpoint at (1, 2) is on the boundary — covers() should catch it
        assert len(violations) >= 1

    def test_entirely_on_land(self, square_land):
        """Segment entirely inside land polygon."""
        waypoints = np.array([
            [1.5, 1.5],
            [2.5, 2.5],
        ])
        violations = validate_track(waypoints, square_land, density=20)
        assert len(violations) == 1
        assert violations[0]["land_fraction"] == 1.0

    def test_single_point(self, square_land):
        """Single waypoint — no segments to check."""
        waypoints = np.array([[2.0, 2.0]])
        violations = validate_track(waypoints, square_land, density=10)
        assert violations == []

    def test_returns_correct_keys(self, square_land):
        """Violation dict has all expected keys."""
        waypoints = np.array([[0.0, 2.0], [5.0, 2.0]])
        violations = validate_track(waypoints, square_land, density=10)
        assert len(violations) == 1
        v = violations[0]
        expected_keys = {
            "segment", "from_idx", "to_idx",
            "from_coord", "to_coord",
            "land_points", "total_points", "land_fraction",
        }
        assert set(v.keys()) == expected_keys
