"""Tests for the SWOPP3 CLI input-validation helpers."""

import importlib.util
from pathlib import Path

import pytest


def _load_swopp3_run_module():
    """Load the CLI module directly from scripts/swopp3_run.py."""
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "swopp3_run.py"
    spec = importlib.util.spec_from_file_location("swopp3_run", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_validate_required_data_paths = _load_swopp3_run_module()._validate_required_data_paths


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
