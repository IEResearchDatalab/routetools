"""Tests for utility helpers in routetools.era5.loader."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from routetools.era5.loader import load_dataset_epoch


def _write_dataset(path: Path, time_coord: str) -> None:
    """Write a minimal ERA5-like dataset with a configurable time coordinate."""
    times = np.array(
        ["2024-01-01T12:00:00", "2024-01-01T13:00:00"], dtype="datetime64[ns]"
    )
    ds = xr.Dataset(
        {
            "u10": (
                (time_coord, "latitude", "longitude"),
                np.zeros((2, 1, 1), dtype=np.float32),
            )
        },
        coords={
            time_coord: times,
            "latitude": np.array([0.0], dtype=np.float64),
            "longitude": np.array([0.0], dtype=np.float64),
        },
    )
    ds.to_netcdf(path, engine="scipy")
    ds.close()


class TestLoadDatasetEpoch:
    def test_reads_time_coordinate(self, tmp_path: Path):
        nc = tmp_path / "wind_time.nc"
        _write_dataset(nc, "time")

        epoch = load_dataset_epoch(nc)
        assert epoch == datetime(2024, 1, 1, 12, 0, 0)

    def test_reads_valid_time_coordinate(self, tmp_path: Path):
        nc = tmp_path / "wind_valid_time.nc"
        _write_dataset(nc, "valid_time")

        epoch = load_dataset_epoch(nc)
        assert epoch == datetime(2024, 1, 1, 12, 0, 0)

    def test_raises_without_time_coord(self, tmp_path: Path):
        nc = tmp_path / "bad.nc"
        ds = xr.Dataset(
            {
                "u10": (("foo", "latitude", "longitude"), np.zeros((1, 1, 1))),
            },
            coords={
                "foo": np.array([0]),
                "latitude": np.array([0.0]),
                "longitude": np.array([0.0]),
            },
        )
        ds.to_netcdf(nc, engine="scipy")
        ds.close()

        with pytest.raises(KeyError, match="time"):
            load_dataset_epoch(nc)
