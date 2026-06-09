"""Tests for the ERA5 data loading and field construction.

These tests create small synthetic NetCDF files that mimic the ERA5 format
and verify that the loaders produce JAX-compatible field closures with
correct interpolation behaviour.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr

# ---------------------------------------------------------------------------
# Fixtures: small synthetic ERA5-like NetCDF files
# ---------------------------------------------------------------------------


def _make_wind_nc(path: Path) -> None:
    """Create a small synthetic ERA5 wind NetCDF file."""
    # 4 time steps (6-hourly over one day), 5 lats, 6 lons
    times = np.array(
        [
            "2024-01-15T00:00",
            "2024-01-15T06:00",
            "2024-01-15T12:00",
            "2024-01-15T18:00",
        ],
        dtype="datetime64[ns]",
    )
    lats = np.array([30.0, 35.0, 40.0, 45.0, 50.0])
    lons = np.array([-70.0, -60.0, -50.0, -40.0, -30.0, -20.0])

    # Simple pattern: u10 = lon/10 + t*0.1, v10 = lat/10 + t*0.1 (varies with time)
    u10 = np.zeros((4, 5, 6), dtype=np.float32)
    v10 = np.zeros((4, 5, 6), dtype=np.float32)
    for t in range(4):
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                u10[t, i, j] = lon / 10.0 + t * 0.1
                v10[t, i, j] = lat / 10.0 + t * 0.1

    ds = xr.Dataset(
        {
            "u10": (["time", "latitude", "longitude"], u10),
            "v10": (["time", "latitude", "longitude"], v10),
        },
        coords={
            "time": times,
            "latitude": lats,
            "longitude": lons,
        },
    )
    ds.to_netcdf(path, engine="scipy")


def _make_wave_nc(path: Path) -> None:
    """Create a small synthetic ERA5 wave NetCDF file."""
    times = np.array(
        [
            "2024-01-15T00:00",
            "2024-01-15T06:00",
            "2024-01-15T12:00",
            "2024-01-15T18:00",
        ],
        dtype="datetime64[ns]",
    )
    lats = np.array([30.0, 35.0, 40.0, 45.0, 50.0])
    lons = np.array([-70.0, -60.0, -50.0, -40.0, -30.0, -20.0])

    swh = np.ones((4, 5, 6), dtype=np.float32) * 2.0  # 2m everywhere
    mwd = np.ones((4, 5, 6), dtype=np.float32) * 180.0  # from South

    ds = xr.Dataset(
        {
            "swh": (["time", "latitude", "longitude"], swh),
            "mwd": (["time", "latitude", "longitude"], mwd),
        },
        coords={
            "time": times,
            "latitude": lats,
            "longitude": lons,
        },
    )
    ds.to_netcdf(path, engine="scipy")


def _make_descending_lat_nc(path: Path) -> None:
    """Create a wind NetCDF with descending latitudes (ERA5 default order)."""
    times = np.array(
        ["2024-01-15T00:00", "2024-01-15T06:00"],
        dtype="datetime64[ns]",
    )
    lats = np.array([50.0, 45.0, 40.0, 35.0, 30.0])  # descending!
    lons = np.array([-70.0, -60.0, -50.0, -40.0])

    u10 = np.zeros((2, 5, 4), dtype=np.float32)
    v10 = np.zeros((2, 5, 4), dtype=np.float32)
    for t in range(2):
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                u10[t, i, j] = lon / 10.0
                v10[t, i, j] = lat / 10.0

    ds = xr.Dataset(
        {
            "u10": (["time", "latitude", "longitude"], u10),
            "v10": (["time", "latitude", "longitude"], v10),
        },
        coords={
            "time": times,
            "latitude": lats,
            "longitude": lons,
        },
    )
    ds.to_netcdf(path, engine="scipy")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLoadERA5Windfield:
    """Tests for load_era5_windfield / load_era5_vectorfield."""

    def test_basic_load_and_shape(self) -> None:
        """Windfield closure returns arrays of the correct shape."""
        from routetools.era5.loader import load_era5_windfield

        with tempfile.TemporaryDirectory() as tmpdir:
            nc_path = Path(tmpdir) / "wind.nc"
            _make_wind_nc(nc_path)

            wf = load_era5_windfield(nc_path)

            # Query at a single point
            lon = jnp.array([[-50.0]])
            lat = jnp.array([[40.0]])
            t = jnp.array([0.0])

            u, v = wf(lon, lat, t)
            assert u.shape == lon.shape
            assert v.shape == lon.shape

    def test_time_variant_attribute(self) -> None:
        """Windfield should be marked as time-variant."""
        from routetools.era5.loader import load_era5_windfield

        with tempfile.TemporaryDirectory() as tmpdir:
            nc_path = Path(tmpdir) / "wind.nc"
            _make_wind_nc(nc_path)
            wf = load_era5_windfield(nc_path)
            assert hasattr(wf, "is_time_variant")
            assert wf.is_time_variant is True

    def test_interpolation_values(self) -> None:
        """Values at grid points should match the synthetic data."""
        from routetools.era5.loader import load_era5_windfield

        with tempfile.TemporaryDirectory() as tmpdir:
            nc_path = Path(tmpdir) / "wind.nc"
            _make_wind_nc(nc_path)
            wf = load_era5_windfield(nc_path)

            # At t=0, grid point (lon=-50, lat=40): u10 = -50/10 = -5.0
            lon = jnp.array([-50.0])
            lat = jnp.array([40.0])
            t = jnp.array([0.0])

            u, v = wf(lon, lat, t)
            np.testing.assert_allclose(float(u[0]), -5.0, atol=0.1)
            np.testing.assert_allclose(float(v[0]), 4.0, atol=0.1)

    def test_batch_2d_input(self) -> None:
        """Windfield handles 2D batched inputs (B, L-1)."""
        from routetools.era5.loader import load_era5_windfield

        with tempfile.TemporaryDirectory() as tmpdir:
            nc_path = Path(tmpdir) / "wind.nc"
            _make_wind_nc(nc_path)
            wf = load_era5_windfield(nc_path)

            # Batch of 3 paths, each with 4 segments
            lon = jnp.ones((3, 4)) * -50.0
            lat = jnp.ones((3, 4)) * 40.0
            t = jnp.array([0.0, 0.0, 0.0])

            u, v = wf(lon, lat, t)
            assert u.shape == (3, 4)
            assert v.shape == (3, 4)

    def test_departure_time_offset(self) -> None:
        """When departure_time is set, t=0 maps to that time."""
        from routetools.era5.loader import load_era5_windfield

        with tempfile.TemporaryDirectory() as tmpdir:
            nc_path = Path(tmpdir) / "wind.nc"
            _make_wind_nc(nc_path)

            # Departure at 12:00 (third time step, index 2)
            wf = load_era5_windfield(nc_path, departure_time="2024-01-15T12:00")

            lon = jnp.array([-50.0])
            lat = jnp.array([40.0])
            t = jnp.array([0.0])  # Should map to 12:00 UTC

            u, v = wf(lon, lat, t)
            # At t=12h (index 2): u10 = -50/10 + 2*0.1 = -4.8
            np.testing.assert_allclose(float(u[0]), -4.8, atol=0.15)

    def test_descending_latitude(self) -> None:
        """Loader handles ERA5 files with descending latitudes."""
        from routetools.era5.loader import load_era5_windfield

        with tempfile.TemporaryDirectory() as tmpdir:
            nc_path = Path(tmpdir) / "wind_desc.nc"
            _make_descending_lat_nc(nc_path)
            wf = load_era5_windfield(nc_path)

            lon = jnp.array([-50.0])
            lat = jnp.array([40.0])
            t = jnp.array([0.0])

            u, v = wf(lon, lat, t)
            np.testing.assert_allclose(float(u[0]), -5.0, atol=0.1)
            np.testing.assert_allclose(float(v[0]), 4.0, atol=0.1)

    def test_file_not_found(self) -> None:
        """Raises FileNotFoundError for missing files."""
        from routetools.era5.loader import load_era5_windfield

        with pytest.raises(FileNotFoundError):
            load_era5_windfield("/nonexistent/path.nc")


class TestLoadERA5Wavefield:
    """Tests for load_era5_wavefield."""

    def test_basic_load(self) -> None:
        """Wavefield closure returns correct values."""
        from routetools.era5.loader import load_era5_wavefield

        with tempfile.TemporaryDirectory() as tmpdir:
            nc_path = Path(tmpdir) / "waves.nc"
            _make_wave_nc(nc_path)
            wf = load_era5_wavefield(nc_path)

            lon = jnp.array([-50.0])
            lat = jnp.array([40.0])
            t = jnp.array([0.0])

            hs, mwd = wf(lon, lat, t)
            np.testing.assert_allclose(float(hs[0]), 2.0, atol=0.1)
            np.testing.assert_allclose(float(mwd[0]), 180.0, atol=0.1)

    def test_wavefield_not_time_variant(self) -> None:
        """Wavefield should NOT have is_time_variant=True."""
        from routetools.era5.loader import load_era5_wavefield

        with tempfile.TemporaryDirectory() as tmpdir:
            nc_path = Path(tmpdir) / "waves.nc"
            _make_wave_nc(nc_path)
            wf = load_era5_wavefield(nc_path)
            # wavefield closures don't get the time_variant decorator
            assert not getattr(wf, "is_time_variant", False)


class TestLoadERA5Vectorfield:
    """Tests for load_era5_vectorfield (alias of windfield)."""

    def test_alias_works(self) -> None:
        """load_era5_vectorfield returns same result as load_era5_windfield."""
        from routetools.era5.loader import (
            load_era5_vectorfield,
            load_era5_windfield,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            nc_path = Path(tmpdir) / "wind.nc"
            _make_wind_nc(nc_path)

            vf = load_era5_vectorfield(nc_path)
            wf = load_era5_windfield(nc_path)

            lon = jnp.array([-50.0])
            lat = jnp.array([40.0])
            t = jnp.array([0.0])

            u1, v1 = vf(lon, lat, t)
            u2, v2 = wf(lon, lat, t)
            np.testing.assert_allclose(u1, u2)
            np.testing.assert_allclose(v1, v2)


class TestDownloadCDS:
    """Tests for the CDS download module (structure only, no actual API calls)."""

    def test_corridors_defined(self) -> None:
        """Verify corridor bounding boxes are defined."""
        from routetools.era5.download_cds import CORRIDORS

        assert "atlantic" in CORRIDORS
        assert "pacific" in CORRIDORS
        for name, bbox in CORRIDORS.items():
            assert len(bbox) == 4, f"Corridor {name} should have 4 bounds"

    def test_variables_defined(self) -> None:
        """Verify ERA5 variable names are defined."""
        from routetools.era5.download_cds import WAVE_VARIABLES, WIND_VARIABLES

        assert len(WIND_VARIABLES) == 2
        assert len(WAVE_VARIABLES) == 2


class TestDownloadGCS:
    """Tests for the GCS download module (structure only, no actual downloads)."""

    def test_corridors_defined(self) -> None:
        """Verify corridor bounding boxes are defined."""
        from routetools.era5.download_gcs import CORRIDORS

        assert "atlantic" in CORRIDORS
        assert "pacific" in CORRIDORS

    def test_gcs_path_exists(self) -> None:
        """Verify GCS bucket path is configured."""
        from routetools.era5.download_gcs import GCS_ERA5_SINGLE_LEVEL

        assert "gcp-public-data" in GCS_ERA5_SINGLE_LEVEL


class TestLoadERA5LandMask:
    """Tests for load_era5_land_mask — derives Land from wave NaN patterns."""

    def test_basic_land_mask(self, tmp_path: Path) -> None:
        """Wave NaN cells become land, valid cells become water."""
        from routetools.era5.loader import load_era5_land_mask

        # Create a synthetic wave NC with known NaN pattern
        times = np.array(["2024-01-15T00:00"], dtype="datetime64[ns]")
        lats = np.array([30.0, 35.0, 40.0, 45.0, 50.0])
        lons = np.array([-70.0, -60.0, -50.0, -40.0, -30.0, -20.0])
        swh = np.ones((1, 5, 6), dtype=np.float32) * 2.0
        mwd = np.ones((1, 5, 6), dtype=np.float32) * 180.0

        # Mark some cells as land (NaN)
        swh[0, 0, 0] = np.nan  # (lat=30, lon=-70) → land
        swh[0, 2, 3] = np.nan  # (lat=40, lon=-40) → land
        mwd[0, 0, 0] = np.nan
        mwd[0, 2, 3] = np.nan

        ds = xr.Dataset(
            {
                "swh": (["time", "latitude", "longitude"], swh),
                "mwd": (["time", "latitude", "longitude"], mwd),
            },
            coords={"time": times, "latitude": lats, "longitude": lons},
        )
        nc_path = tmp_path / "wave_land.nc"
        ds.to_netcdf(nc_path, engine="scipy")

        land = load_era5_land_mask(nc_path)

        # Land object grid should cover the file extent
        assert land.xmin == -70.0
        assert land.xmax == -20.0
        assert land.ymin == 30.0
        assert land.ymax == 50.0

        # Internal array shape should be (lon, lat) = (6, 5)
        assert land._array.shape == (6, 5)

        # Check NaN → land (value > 0.5) and valid → water (value < 0.5)
        arr = np.array(land._array)
        assert arr[0, 0] > 0.5, "NaN cell should be land"
        assert arr[3, 2] > 0.5, "NaN cell should be land"
        assert arr[1, 1] < 0.5, "Valid cell should be water"
        assert arr[5, 4] < 0.5, "Valid cell should be water"

    def test_missing_variable_raises(self, tmp_path: Path) -> None:
        """KeyError when no wave-height variable is found."""
        from routetools.era5.loader import load_era5_land_mask

        times = np.array(["2024-01-15T00:00"], dtype="datetime64[ns]")
        lats = np.array([30.0, 35.0])
        lons = np.array([-70.0, -60.0])
        ds = xr.Dataset(
            {
                "temperature": (
                    ["time", "latitude", "longitude"],
                    np.ones((1, 2, 2), dtype=np.float32),
                )
            },
            coords={"time": times, "latitude": lats, "longitude": lons},
        )
        nc_path = tmp_path / "no_wave.nc"
        ds.to_netcdf(nc_path, engine="scipy")

        with pytest.raises(KeyError, match="Cannot find wave height"):
            load_era5_land_mask(nc_path)


# ---------------------------------------------------------------------------
# Helpers for valid_time and multi-file tests
# ---------------------------------------------------------------------------


def _make_wind_nc_valid_time(path: Path) -> None:
    """Create a wind NetCDF using ``valid_time`` instead of ``time``."""
    times = np.array(
        [
            "2024-01-15T00:00",
            "2024-01-15T06:00",
            "2024-01-15T12:00",
            "2024-01-15T18:00",
        ],
        dtype="datetime64[ns]",
    )
    lats = np.array([30.0, 35.0, 40.0, 45.0, 50.0])
    lons = np.array([-70.0, -60.0, -50.0, -40.0, -30.0, -20.0])

    u10 = np.zeros((4, 5, 6), dtype=np.float32)
    v10 = np.zeros((4, 5, 6), dtype=np.float32)
    for t in range(4):
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                u10[t, i, j] = lon / 10.0 + t * 0.1
                v10[t, i, j] = lat / 10.0 + t * 0.1

    ds = xr.Dataset(
        {
            "u10": (["valid_time", "latitude", "longitude"], u10),
            "v10": (["valid_time", "latitude", "longitude"], v10),
        },
        coords={
            "valid_time": times,
            "latitude": lats,
            "longitude": lons,
        },
    )
    ds.to_netcdf(path, engine="scipy")


def _make_wind_nc_part(path: Path, day: int) -> None:
    """Create a 1-day wind NetCDF for multi-file concat tests.

    Parameters
    ----------
    day : int
        Day of month (e.g. 15 or 16) — determines timestamps and values.
    """
    times = np.array(
        [
            f"2024-01-{day:02d}T00:00",
            f"2024-01-{day:02d}T06:00",
            f"2024-01-{day:02d}T12:00",
            f"2024-01-{day:02d}T18:00",
        ],
        dtype="datetime64[ns]",
    )
    lats = np.array([30.0, 35.0, 40.0, 45.0, 50.0])
    lons = np.array([-70.0, -60.0, -50.0, -40.0, -30.0, -20.0])

    # Use day offset so concatenated files have distinct values
    u10 = np.full((4, 5, 6), float(day), dtype=np.float32)
    v10 = np.full((4, 5, 6), float(day) + 0.5, dtype=np.float32)

    ds = xr.Dataset(
        {
            "u10": (["time", "latitude", "longitude"], u10),
            "v10": (["time", "latitude", "longitude"], v10),
        },
        coords={
            "time": times,
            "latitude": lats,
            "longitude": lons,
        },
    )
    ds.to_netcdf(path, engine="scipy")


class TestValidTimeDimension:
    """Tests for files using ``valid_time`` instead of ``time``."""

    def test_load_windfield_valid_time(self) -> None:
        """Windfield loads correctly from a file with ``valid_time`` dim."""
        from routetools.era5.loader import load_era5_windfield

        with tempfile.TemporaryDirectory() as tmpdir:
            nc_path = Path(tmpdir) / "wind_vt.nc"
            _make_wind_nc_valid_time(nc_path)

            wf = load_era5_windfield(nc_path)

            lon = jnp.array([-50.0])
            lat = jnp.array([40.0])
            t = jnp.array([0.0])

            u, v = wf(lon, lat, t)
            np.testing.assert_allclose(float(u[0]), -5.0, atol=0.1)
            np.testing.assert_allclose(float(v[0]), 4.0, atol=0.1)


class TestMultiFileConcat:
    """Tests for loading and concatenating multiple NetCDF files."""

    def test_concat_two_files(self) -> None:
        """Two single-day files concatenate into 8 time steps."""
        from routetools.era5.loader import _load_datasets

        with tempfile.TemporaryDirectory() as tmpdir:
            p1 = Path(tmpdir) / "day15.nc"
            p2 = Path(tmpdir) / "day16.nc"
            _make_wind_nc_part(p1, 15)
            _make_wind_nc_part(p2, 16)

            ds = _load_datasets([p1, p2])
            time_name = "valid_time" if "valid_time" in ds.dims else "time"
            assert ds.sizes[time_name] == 8

    def test_concat_windfield_values(self) -> None:
        """Multi-file windfield returns values from both files."""
        from routetools.era5.loader import load_era5_windfield

        with tempfile.TemporaryDirectory() as tmpdir:
            p1 = Path(tmpdir) / "day15.nc"
            p2 = Path(tmpdir) / "day16.nc"
            _make_wind_nc_part(p1, 15)
            _make_wind_nc_part(p2, 16)

            wf = load_era5_windfield([p1, p2])

            lon = jnp.array([-50.0])
            lat = jnp.array([40.0])

            # t=0h maps to day-15 00:00; u10 should be 15.0
            u0, _ = wf(lon, lat, jnp.array([0.0]))
            np.testing.assert_allclose(float(u0[0]), 15.0, atol=0.1)

            # t=24h maps to day-16 00:00; u10 should be 16.0
            u24, _ = wf(lon, lat, jnp.array([24.0]))
            np.testing.assert_allclose(float(u24[0]), 16.0, atol=0.1)

    def test_single_path_unchanged(self) -> None:
        """Passing a single Path still works (backward compat)."""
        from routetools.era5.loader import load_era5_windfield

        with tempfile.TemporaryDirectory() as tmpdir:
            nc_path = Path(tmpdir) / "wind.nc"
            _make_wind_nc(nc_path)

            wf = load_era5_windfield(nc_path)
            lon = jnp.array([-50.0])
            lat = jnp.array([40.0])
            t = jnp.array([0.0])
            u, v = wf(lon, lat, t)
            np.testing.assert_allclose(float(u[0]), -5.0, atol=0.1)

    def test_mixed_time_valid_time(self) -> None:
        """Files with ``time`` and ``valid_time`` dims concatenate."""
        from routetools.era5.loader import _load_datasets

        with tempfile.TemporaryDirectory() as tmpdir:
            # File 1: uses "time" dimension
            p1 = Path(tmpdir) / "day15.nc"
            _make_wind_nc_part(p1, 15)

            # File 2: uses "valid_time" dimension
            p2 = Path(tmpdir) / "day16.nc"
            _make_wind_nc_valid_time(p2)

            ds = _load_datasets([p1, p2])
            time_name = "valid_time" if "valid_time" in ds.dims else "time"
            assert ds.sizes[time_name] == 8


class TestVoyageCoverageWarning:
    """Tests for the voyage coverage warning in windfield/wavefield."""

    def test_warning_logged_when_insufficient(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A warning is logged when data doesn't cover the full voyage."""
        import logging

        from routetools.era5.loader import load_era5_windfield

        with tempfile.TemporaryDirectory() as tmpdir:
            nc_path = Path(tmpdir) / "wind.nc"
            _make_wind_nc(nc_path)  # covers 18h (4 steps × 6h)

            with caplog.at_level(logging.WARNING, logger="routetools.era5.loader"):
                load_era5_windfield(nc_path, voyage_hours=100.0)

            assert any(
                "covers" in r.message and "voyage" in r.message for r in caplog.records
            )

    def test_no_warning_when_sufficient(self, caplog: pytest.LogCaptureFixture) -> None:
        """No warning when data covers the voyage."""
        import logging

        from routetools.era5.loader import load_era5_windfield

        with tempfile.TemporaryDirectory() as tmpdir:
            nc_path = Path(tmpdir) / "wind.nc"
            _make_wind_nc(nc_path)  # covers 18h

            with caplog.at_level(logging.WARNING, logger="routetools.era5.loader"):
                load_era5_windfield(nc_path, voyage_hours=10.0)

            assert not any("covers" in r.message for r in caplog.records)
