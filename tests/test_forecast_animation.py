"""Checks that the animation changes forecasts rather than the target weather."""

import importlib.util
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

script_dir = Path(__file__).parents[1] / "scripts/data"
sys.path.insert(0, str(script_dir))
spec = importlib.util.spec_from_file_location(
    "forecast_animation", script_dir / "animate_forecast_revisions.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_all_frames_verify_the_same_instant():
    target = datetime(2025, 1, 30)
    frames = module.schedule(target)
    assert len(frames) == 31
    assert frames[0] == (datetime(2025, 1, 15), 360)
    assert frames[-1] == (target, 0)
    assert all(issue + timedelta(hours=lead) == target for issue, lead in frames)
    assert all(
        frames[i + 1][0] - frames[i][0] == timedelta(hours=12) for i in range(30)
    )
    with pytest.raises(ValueError):
        module.schedule(datetime(2025, 1, 30, 6))


def test_wind_arrows_follow_wave_ocean_mask():
    hs = np.array([[np.nan, 3.0]])
    ds = xr.Dataset(
        {
            module.HS: (("latitude", "longitude"), hs),
            module.U: (("latitude", "longitude"), [[5.0, 6.0]]),
            module.V: (("latitude", "longitude"), [[7.0, 8.0]]),
        }
    )
    wave, u, v = module.arrays(ds)
    assert np.isnan(u[0, 0]) and np.isnan(v[0, 0])
    assert u[0, 1] == 6 and v[0, 1] == 8
    np.testing.assert_array_equal(wave, hs)


def test_error_uses_paired_finite_cells():
    forecast = np.array([[np.nan, 3.0], [9.0, 2.0]])
    reference = np.array([[1.0, 1.0], [np.nan, 2.0]])
    result = module.rmse(forecast, reference, np.array([60.0, 30.0]))
    expected = np.sqrt(4 * 0.5 / (0.5 + np.cos(np.deg2rad(30))))
    assert result == pytest.approx(expected)
    with pytest.raises(ValueError):
        module.rmse(np.full((2, 2), np.nan), reference, np.array([60.0, 30.0]))


def test_wrong_forecast_issue_with_same_target_is_rejected(tmp_path, monkeypatch):
    """Receipt validity alone must not permit another issue under this frame label."""
    import json

    target = datetime(2025, 1, 30)
    issue = datetime(2025, 1, 15)
    wrong_request = dict(
        corridor="atlantic",
        initialization=datetime(2025, 1, 15, 12).isoformat(),
        lead=348,
        stream="enfo",
        ensemble=True,
        variables=list(module.acquisition.PARAMS["enfo"].values()),
    )
    path = tmp_path / "lead_360.nc"
    path.with_suffix(".json").write_text(json.dumps(dict(request=wrong_request)))
    # Emulate a valid receipt for the other issue, still predicting the same target.
    monkeypatch.setattr(
        module.acquisition,
        "completed",
        lambda path, expected: expected == wrong_request,
    )
    raw = xr.Dataset(
        coords={
            "number": [0, 8, 47],
            "step": [np.timedelta64(348, "h")],
            "valid_time": ("step", [np.datetime64(target)]),
        }
    )
    monkeypatch.setattr(
        module.xr, "open_dataset", lambda *args, **kwargs: raw.copy(deep=True)
    )
    with pytest.raises(ValueError, match="unverified forecast shard"):
        module.read_forecast(path, target, [0, 8, 47], issue, 360, "enfo")


def test_schedule_rejects_fractional_seconds():
    with pytest.raises(ValueError):
        module.schedule(datetime(2025, 1, 30, microsecond=1))


@pytest.mark.parametrize(
    "centres,expected",
    [
        ([-80.0, -79.75, -79.5], [-80.125, -79.875, -79.625, -79.375]),
        ([60.0, 59.75, 59.5], [60.125, 59.875, 59.625, 59.375]),
    ],
)
def test_cell_edges_preserve_source_centres_and_coordinate_order(centres, expected):
    centres = np.array(centres)
    original = centres.copy()
    edges = module.cell_edges(centres)
    np.testing.assert_array_equal(edges, expected)
    np.testing.assert_array_equal((edges[:-1] + edges[1:]) / 2, centres)
    np.testing.assert_array_equal(centres, original)
    with pytest.raises(ValueError, match="at least two"):
        module.cell_edges(np.array([60.0]))


def test_polygon_land_mask_preserves_raw_dataset():
    ds = xr.Dataset(
        {
            module.HS: (("latitude", "longitude"), [[2.0, 3.0, np.nan]]),
            module.U: (("latitude", "longitude"), [[5.0, 6.0, 7.0]]),
            module.V: (("latitude", "longitude"), [[8.0, 9.0, 10.0]]),
        },
        coords={"latitude": [50.0], "longitude": [-60.0, -59.75, -59.5]},
    )
    original = ds.copy(deep=True)
    land = np.array([[True, False, False]])
    land_original = land.copy()
    wave, u, v = module.arrays(ds, land)
    assert np.isnan(wave[0, 0]) and np.isnan(u[0, 0]) and np.isnan(v[0, 0])
    assert (wave[0, 1], u[0, 1], v[0, 1]) == (3.0, 6.0, 9.0)
    assert np.isnan(wave[0, 2]) and np.isnan(u[0, 2]) and np.isnan(v[0, 2])
    xr.testing.assert_identical(ds, original)
    np.testing.assert_array_equal(land, land_original)
    # Returned display buffers must not alias source data either.
    wave[0, 1] = 99.0
    u[0, 1] = 88.0
    v[0, 1] = 77.0
    xr.testing.assert_identical(ds, original)
