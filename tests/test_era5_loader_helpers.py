"""Tests for utility helpers in routetools.era5.loader."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from routetools.era5.loader import load_dataset_epoch, loadable_era5_paths


def _write_dataset(
    path: Path,
    time_coord: str,
    times: np.ndarray | None = None,
) -> None:
    """Write a minimal ERA5-like dataset with a configurable time coordinate."""
    times = (
        times
        if times is not None
        else np.array(
            ["2024-01-01T12:00:00", "2024-01-01T13:00:00"], dtype="datetime64[ns]"
        )
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

    def test_reads_earliest_time_from_multiple_files(self, tmp_path: Path):
        later = tmp_path / "wind_later.nc"
        earlier = tmp_path / "wind_earlier.nc"
        _write_dataset(
            later,
            "time",
            np.array(
                ["2024-01-02T12:00:00", "2024-01-02T13:00:00"], dtype="datetime64[ns]"
            ),
        )
        _write_dataset(
            earlier,
            "valid_time",
            np.array(
                ["2024-01-01T12:00:00", "2024-01-01T13:00:00"], dtype="datetime64[ns]"
            ),
        )

        epoch = load_dataset_epoch([later, earlier])
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


class TestLoadableEra5Paths:
    def test_returns_only_base_when_no_continuation_file(self, tmp_path: Path):
        base = tmp_path / "era5_wind_atlantic_2024.nc"
        base.touch()

        assert loadable_era5_paths(base) == [base]

    def test_returns_base_plus_exact_next_year_when_present(self, tmp_path: Path):
        base = tmp_path / "era5_wind_atlantic_2024.nc"
        next_year = tmp_path / "era5_wind_atlantic_2025.nc"
        base.touch()
        next_year.touch()

        assert loadable_era5_paths(base) == [base, next_year]

    def test_returns_base_plus_monthly_glob_files(self, tmp_path: Path):
        base = tmp_path / "era5_wind_pacific_2024.nc"
        jan = tmp_path / "era5_wind_pacific_2025_01.nc"
        feb = tmp_path / "era5_wind_pacific_2025_02.nc"
        base.touch()
        jan.touch()
        feb.touch()

        assert loadable_era5_paths(base) == [base, jan, feb]

    def test_returns_only_base_when_filename_does_not_match_pattern(
        self, tmp_path: Path
    ):
        base = tmp_path / "custom_weather_file.nc"
        base.touch()

        assert loadable_era5_paths(base) == [base]
