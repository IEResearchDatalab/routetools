"""Tests for the SWOPP3 CLI input-validation helpers."""

import importlib.util
import json
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


_swopp3_run = _load_swopp3_run_module()
_loadable_era5_paths = _swopp3_run._loadable_era5_paths
_load_experiment_profile = _swopp3_run._load_experiment_profile
_resolve_case_ids = _swopp3_run._resolve_case_ids
_resolve_config_value_path = _swopp3_run._resolve_config_value_path
_validate_required_data_paths = _swopp3_run._validate_required_data_paths
_write_experiment_manifest = _swopp3_run._write_experiment_manifest


def _write_config(tmp_path: Path, content: str) -> Path:
    """Create a temporary SWOPP3 experiment config file."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "experiments.toml"
    config_path.write_text(content)
    return config_path


def test_loadable_era5_paths_adds_next_year_continuation(tmp_path: Path):
    """Runner should pick up next-year continuation files automatically."""
    base = tmp_path / "era5_wind_pacific_2024.nc"
    jan = tmp_path / "era5_wind_pacific_2025_01.nc"
    feb = tmp_path / "era5_wind_pacific_2025_02.nc"
    base.touch()
    jan.touch()
    feb.touch()

    assert _loadable_era5_paths(base) == [base, jan, feb]


def test_resolve_config_value_path_anchors_relative_paths_to_config_dir(tmp_path: Path):
    """Relative config paths should resolve from the TOML directory."""
    config_path = tmp_path / "configs" / "experiments.toml"
    config_path.parent.mkdir()

    resolved = _resolve_config_value_path(config_path, "../data/example.nc")

    assert resolved == (tmp_path / "data" / "example.nc").resolve()


def test_load_experiment_profile_merges_defaults_and_resolves_paths(tmp_path: Path):
    """Profile loading should merge defaults with per-run overrides."""
    config_path = _write_config(
        tmp_path,
        """
[swopp3.experiments.demo]
description = "Demo profile"
source_script = "scripts/swopp3_slurm.sh"
output_dir = "../output/demo"

[swopp3.experiments.demo.defaults]
wind_path_atlantic = "../data/default_wind.nc"
wave_path_atlantic = "../data/default_wave.nc"
n_points = 178
submission = 2

[[swopp3.experiments.demo.runs]]
name = "atlantic"
cases = ["AO_WPS"]

[[swopp3.experiments.demo.runs]]
name = "override"
cases = "PO_WPS"
wind_path_atlantic = "../data/override_wind.nc"
""".strip(),
    )

    profile = _load_experiment_profile(config_path, "demo")

    assert profile["name"] == "demo"
    assert profile["output_dir"] == (tmp_path / "output" / "demo").resolve()
    assert len(profile["runs"]) == 2
    assert profile["runs"][0]["n_points"] == 178
    assert profile["runs"][0]["submission"] == 2
    assert (
        profile["runs"][0]["wind_path_atlantic"]
        == (tmp_path / "data" / "default_wind.nc").resolve()
    )
    assert (
        profile["runs"][1]["wind_path_atlantic"]
        == (tmp_path / "data" / "override_wind.nc").resolve()
    )
    assert profile["runs"][1]["cases"] == ["PO_WPS"]


def test_load_experiment_profile_raises_for_unknown_name(tmp_path: Path):
    """Unknown experiment names should list the available profiles."""
    config_path = _write_config(
        tmp_path,
        """
[swopp3.experiments.demo]
[[swopp3.experiments.demo.runs]]
name = "run"
cases = ["AO_WPS"]
""".strip(),
    )

    with pytest.raises(KeyError, match="Unknown SWOPP3 experiment 'missing'"):
        _load_experiment_profile(config_path, "missing")


def test_load_experiment_profile_raises_for_empty_runs(tmp_path: Path):
    """Profiles without runs should fail with a clear error."""
    config_path = _write_config(
        tmp_path,
        """
[swopp3.experiments.demo]
description = "Demo profile"
""".strip(),
    )

    with pytest.raises(ValueError, match="does not define any runs"):
        _load_experiment_profile(config_path, "demo")


def test_write_experiment_manifest_serializes_resolved_paths(tmp_path: Path):
    """Manifest writing should preserve resolved paths as JSON strings."""
    config_path = tmp_path / "configs" / "experiments.toml"
    config_path.parent.mkdir()
    output_dir = tmp_path / "output" / "demo"
    profile = {
        "name": "demo",
        "description": "Demo profile",
        "source_script": "scripts/swopp3_slurm.sh",
        "output_dir": output_dir,
        "runs": [
            {
                "name": "atlantic",
                "cases": ["AO_WPS"],
                "wind_path_atlantic": tmp_path / "data" / "wind.nc",
            }
        ],
    }

    manifest_path = _write_experiment_manifest(
        config_path=config_path,
        profile=profile,
    )

    manifest = json.loads(manifest_path.read_text())
    assert manifest["experiment"] == "demo"
    assert manifest["config_path"] == str(config_path)
    assert manifest["output_dir"] == str(output_dir)
    assert manifest["runs"][0]["wind_path_atlantic"] == str(
        tmp_path / "data" / "wind.nc"
    )


def test_resolve_case_ids_filters_selected_strategy():
    """Strategy filtering should keep only matching case IDs."""
    case_ids = _resolve_case_ids(None, "optimised")

    assert case_ids
    assert all(case_id.endswith(("WPS", "noWPS")) for case_id in case_ids)
    assert all(
        case_id in {"AO_WPS", "AO_noWPS", "PO_WPS", "PO_noWPS"} for case_id in case_ids
    )


def test_resolve_case_ids_raises_for_empty_strategy_match():
    """Unknown strategy filters should fail with a clear error."""
    with pytest.raises(ValueError, match="No cases match strategy 'missing'"):
        _resolve_case_ids(None, "missing")


def test_shared_cli_paths_override_default_corridor_paths(monkeypatch):
    """Shared CLI paths should override the built-in corridor defaults."""

    class StopCli(Exception):
        pass

    captured: dict[str, dict[str, Path]] = {}

    def fake_validate(case_ids, corridor_wind, corridor_wave):
        captured["wind"] = corridor_wind.copy()
        captured["wave"] = corridor_wave.copy()
        raise StopCli()

    monkeypatch.setattr(_swopp3_run, "_validate_required_data_paths", fake_validate)

    with pytest.raises(StopCli):
        _swopp3_run.main(
            cases=["AGC_WPS", "PGC_WPS"],
            strategy=None,
            wind_path=Path("shared_wind.nc"),
            wave_path=Path("shared_wave.nc"),
            wind_path_atlantic=Path("data/era5/era5_wind_atlantic_2024.nc"),
            wave_path_atlantic=Path("data/era5/era5_waves_atlantic_2024.nc"),
            wind_path_pacific=Path("data/era5/era5_wind_pacific_2024.nc"),
            wave_path_pacific=Path("data/era5/era5_waves_pacific_2024.nc"),
            output_dir=Path("output/swopp3"),
            submission=1,
            n_points=100,
            max_departures=1,
            quiet=True,
        )

    assert captured["wind"] == {
        "atlantic": Path("shared_wind.nc"),
        "pacific": Path("shared_wind.nc"),
    }
    assert captured["wave"] == {
        "atlantic": Path("shared_wave.nc"),
        "pacific": Path("shared_wave.nc"),
    }


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
