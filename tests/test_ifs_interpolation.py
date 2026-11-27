"""Tests for IFS ensemble forecast temporal interpolation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from routetools.ifs.interpolation import (
    interpolate_field_cubic,
    load_ensemble_waves,
    load_ensemble_wind,
    relative_rmse_percent,
    resample_wind_at_resolutions,
    temporal_stride_for_resolution,
)


@pytest.mark.parametrize(
    ("target_h", "expected_stride"),
    [(24.0, 8), (12.0, 4), (6.0, 2), (3.0, 1)],
)
def test_temporal_stride_for_resolution(
    target_h: float,
    expected_stride: int,
) -> None:
    """Resolution targets map to exact native 3-hour strides."""
    assert temporal_stride_for_resolution(3.0, target_h) == expected_stride


@pytest.mark.parametrize(
    ("source_h", "target_h"),
    [(0.0, 3.0), (3.0, 0.0), (3.0, 2.0), (3.0, 7.0)],
)
def test_temporal_stride_rejects_invalid_values(
    source_h: float,
    target_h: float,
) -> None:
    """Invalid or incompatible resolutions raise clear errors."""
    with pytest.raises(ValueError):
        temporal_stride_for_resolution(source_h, target_h)


def test_cubic_interpolation_reproduces_linear_field() -> None:
    """Cubic interpolation preserves a linear field at every target lead time."""
    source_hours = np.array([0.0, 12.0, 24.0])
    values = np.arange(18, dtype=np.float64).reshape(3, 2, 3)
    target_hours = np.arange(0.0, 25.0, 3.0)

    result = interpolate_field_cubic(values, source_hours, target_hours)

    expected = (
        values[0][None, :, :]
        + target_hours[:, None, None]
        * (values[1] - values[0])[None, :, :]
        / 12.0
    )
    np.testing.assert_allclose(result, expected, atol=1e-10)


def test_cubic_interpolation_preserves_missing_spatial_cells() -> None:
    """NaNs in land or incomplete cells do not break interpolation elsewhere."""
    source_hours = np.array([0.0, 6.0, 12.0, 18.0, 24.0])
    values = np.array(
        [
            [[0.0, np.nan, 0.0]],
            [[6.0, np.nan, np.nan]],
            [[12.0, np.nan, 12.0]],
            [[18.0, np.nan, 18.0]],
            [[24.0, np.nan, 24.0]],
        ]
    )
    target_hours = np.arange(0.0, 25.0, 3.0)

    result = interpolate_field_cubic(values, source_hours, target_hours)

    np.testing.assert_allclose(
        result[:, 0, 0],
        target_hours,
        atol=1e-10,
    )
    assert np.isnan(result[:, 0, 1]).all()
    np.testing.assert_allclose(
        result[:, 0, 2],
        target_hours,
        atol=1e-10,
    )


def test_resample_contains_native_and_coarse_fields() -> None:
    """The resampler includes native 3-hour and three coarser resolutions."""
    hours = np.arange(0.0, 27.0, 3.0)
    u = np.broadcast_to(hours[:, None, None], (len(hours), 2, 2)).copy()
    v = np.zeros_like(u)

    fields = resample_wind_at_resolutions(hours, u, v)

    assert set(fields) == {24.0, 12.0, 6.0, 3.0}
    np.testing.assert_array_equal(fields[3.0][0], u)
    for resolution_h in (24.0, 12.0, 6.0):
        np.testing.assert_allclose(fields[resolution_h][0], u, atol=1e-10)


def test_relative_rmse_is_normalized_by_reference_mean_speed() -> None:
    """Relative RMSE reports error as a percentage of reference mean speed."""
    result = relative_rmse_percent(
        np.array([2.0, 4.0]),
        np.array([10.0, 20.0]),
    )

    np.testing.assert_allclose(result, [20.0, 20.0])


def test_relative_rmse_rejects_zero_reference_speed() -> None:
    """Relative RMSE fails clearly when the normalization is undefined."""
    with pytest.raises(ValueError, match="greater than zero"):
        relative_rmse_percent(np.array([1.0]), np.array([0.0]))


def test_load_ensemble_wind_averages_members(tmp_path: Path) -> None:
    """Loader returns the ensemble mean over a cropped, time-bounded grid."""
    path = tmp_path / "ensemble.nc"
    step = np.arange(0, 27, 3).astype("timedelta64[h]")
    ds = xr.Dataset(
        {
            "10m_u_component_of_wind": (
                ("number", "step", "latitude", "longitude"),
                np.arange(2 * len(step) * 3 * 3, dtype=np.float32).reshape(
                    2, len(step), 3, 3
                ),
            ),
            "10m_v_component_of_wind": (
                ("number", "step", "latitude", "longitude"),
                np.ones((2, len(step), 3, 3), dtype=np.float32),
            ),
        },
        coords={
            "number": [1, 2],
            "step": step,
            "latitude": [60.0, 50.0, 40.0],
            "longitude": [-10.0, 0.0, 10.0],
        },
    )
    ds.to_netcdf(path, engine="scipy")

    hours, lats, lons, u, v = load_ensemble_wind(
        path,
        lon_bounds=(-10.0, 10.0),
        lat_bounds=(40.0, 60.0),
        max_hours=24.0,
    )

    assert hours.tolist() == [float(value) for value in range(0, 27, 3) if value <= 24]
    assert lats.tolist() == [60.0, 50.0, 40.0]
    assert lons.tolist() == [-10.0, 0.0, 10.0]
    assert u.shape == (9, 3, 3)
    np.testing.assert_allclose(v, 1.0)


def test_load_ensemble_waves_supports_mean_and_selected_member(
    tmp_path: Path,
) -> None:
    """Wave loader averages members by default or selects the requested one."""
    path = tmp_path / "waves.nc"
    step = np.arange(0, 27, 3).astype("timedelta64[h]")
    heights = np.stack(
        [
            np.full((len(step), 2, 2), 1.0, dtype=np.float32),
            np.full((len(step), 2, 2), 3.0, dtype=np.float32),
        ]
    )
    ds = xr.Dataset(
        {
            "significant_height_of_combined_wind_waves_and_swell": (
                ("number", "step", "latitude", "longitude"),
                heights,
            ),
        },
        coords={
            "number": [1, 2],
            "step": step,
            "latitude": [50.0, 40.0],
            "longitude": [-10.0, 0.0],
        },
    )
    ds.to_netcdf(path, engine="scipy")

    hours, _lats, _lons, mean_height = load_ensemble_waves(path, max_hours=24.0)
    _hours, _lats, _lons, member_height = load_ensemble_waves(
        path,
        ensemble_member=2,
        max_hours=24.0,
    )

    assert hours.shape == (9,)
    np.testing.assert_allclose(mean_height, 2.0)
    np.testing.assert_allclose(member_height, 3.0)
