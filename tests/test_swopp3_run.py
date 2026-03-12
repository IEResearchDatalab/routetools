"""Tests for the SWOPP3 CLI input-validation helpers."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import xarray as xr


def _load_swopp3_run_module():
    """Load the CLI module directly from scripts/swopp3_run.py."""
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "swopp3_run.py"
    spec = importlib.util.spec_from_file_location("swopp3_run", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_swopp3_run = _load_swopp3_run_module()
_validate_required_data_paths = _swopp3_run._validate_required_data_paths
_load_corridor_land_mask = _swopp3_run._load_corridor_land_mask


def _write_grid_dataset(
    path: Path,
    *,
    lon_name: str = "longitude",
    lat_name: str = "latitude",
) -> None:
    """Write a minimal dataset with lon/lat coordinates for land-mask loading."""
    ds = xr.Dataset(
        {
            "dummy": (
                (lat_name, lon_name),
                np.zeros((2, 3), dtype=np.float32),
            )
        },
        coords={
            lon_name: np.array([-20.0, -10.0, 0.0], dtype=np.float64),
            lat_name: np.array([35.0, 45.0], dtype=np.float64),
        },
    )
    ds.to_netcdf(path, engine="scipy")
    ds.close()


def test_validate_required_data_paths_reports_missing_files(tmp_path: Path):
    """Validation error should explain which datasets are missing and why."""
    wind_path = tmp_path / "era5_wind_atlantic_2024.nc"
    wave_path = tmp_path / "era5_waves_atlantic_2024.nc"

    with pytest.raises(
        FileNotFoundError,
        match="SWOPP3 input validation failed",
    ) as exc_info:
        _validate_required_data_paths(
            ["AGC_WPS"],
            {"atlantic": wind_path},
            {"atlantic": wave_path},
        )

    message = str(exc_info.value)
    assert str(wind_path) in message
    assert str(wave_path) in message
    assert "there is no fallback to GC or no-weather mode" in message
    assert "uv run scripts/download_era5.py" in message


def test_validate_required_data_paths_accepts_existing_files(tmp_path: Path):
    """Validation should pass when the required files already exist."""
    wind_path = tmp_path / "era5_wind_atlantic_2024.nc"
    wave_path = tmp_path / "era5_waves_atlantic_2024.nc"
    wind_path.touch()
    wave_path.touch()

    _validate_required_data_paths(
        ["AGC_WPS"],
        {"atlantic": wind_path},
        {"atlantic": wave_path},
    )


def test_load_corridor_land_mask_passes_resolution_and_avoidance_scale(
    tmp_path: Path,
    monkeypatch,
):
    """Corridor land-mask loading should forward the configured resolution knobs."""
    nc_path = tmp_path / "corridor.nc"
    _write_grid_dataset(nc_path)

    captured: dict[str, object] = {}
    sentinel = object()

    def fake_load_natural_earth_land_mask(
        lon_range,
        lat_range,
        resolution=0.01,
        ne_resolution="10m",
        interpolate=50,
        avoidance_resolution_scale=2,
    ):
        captured.update(
            {
                "lon_range": lon_range,
                "lat_range": lat_range,
                "resolution": resolution,
                "ne_resolution": ne_resolution,
                "interpolate": interpolate,
                "avoidance_resolution_scale": avoidance_resolution_scale,
            }
        )
        return sentinel

    monkeypatch.setattr(
        _swopp3_run,
        "load_natural_earth_land_mask",
        fake_load_natural_earth_land_mask,
    )

    result = _load_corridor_land_mask(
        nc_path,
        land_resolution=0.05,
        land_avoidance_resolution_scale=1,
    )

    assert result is sentinel
    assert captured == {
        "lon_range": (-20.0, 0.0),
        "lat_range": (35.0, 45.0),
        "resolution": 0.05,
        "ne_resolution": "10m",
        "interpolate": 50,
        "avoidance_resolution_scale": 1,
    }
