"""Unit tests for forecast-aware experiment download helpers."""

from datetime import date
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import numpy as np
import xarray as xr


def _load_script(name: str, relative_path: str) -> ModuleType:
    """Load a repository script without requiring ``scripts`` as a package."""
    path = Path(__file__).parents[1] / relative_path
    spec = spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FORECAST = _load_script(
    "download_ecmwf_forecasts", "scripts/data/download_ecmwf_forecasts.py"
)
ERA5 = _load_script("download_era5_window", "scripts/data/download_era5_window.py")


def test_native_forecast_steps() -> None:
    """The deterministic IFS horizon has the expected native 85 steps."""
    assert len(FORECAST.FORECAST_STEPS) == 85
    assert FORECAST.FORECAST_STEPS[:3] == [0, 3, 6]
    assert FORECAST.FORECAST_STEPS[48:51] == [144, 150, 156]
    assert FORECAST.FORECAST_STEPS[-1] == 360


def test_inclusive_dates() -> None:
    """Run-date generation includes both requested endpoints."""
    assert FORECAST._dates(date(2025, 7, 15), date(2025, 7, 17)) == [
        date(2025, 7, 15),
        date(2025, 7, 16),
        date(2025, 7, 17),
    ]


def test_crop_atlantic_matches_bers_grid() -> None:
    """Atlantic cropping converts 0–360 longitude to the BERS grid."""
    ds = xr.Dataset(
        coords={
            "latitude": np.arange(90.0, -90.25, -0.25, dtype=np.float32),
            "longitude": np.arange(0.0, 360.0, 0.25, dtype=np.float32),
        }
    )
    cropped = FORECAST._crop_corridor(ds, "atlantic")

    np.testing.assert_array_equal(
        cropped.latitude.values,
        np.arange(60.0, 24.75, -0.25, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        cropped.longitude.values,
        np.arange(-80.0, 10.25, 0.25, dtype=np.float32),
    )


def test_era5_window_naming_and_months(tmp_path) -> None:
    """ERA5 exact-window files are deterministic and span both months."""
    start = date(2025, 7, 15)
    end = date(2025, 8, 8)
    assert ERA5._months_between(start, end) == [7, 8]
    assert ERA5._output_path(tmp_path, "wind", "atlantic", start, end).name == (
        "era5_wind_atlantic_20250715_20250808.nc"
    )
