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


_swopp3_run = _load_swopp3_run_module()
_loadable_era5_paths = _swopp3_run._loadable_era5_paths
_load_experiment_profile = _swopp3_run._load_experiment_profile
_validate_required_data_paths = _swopp3_run._validate_required_data_paths


def test_loadable_era5_paths_adds_next_year_continuation(tmp_path: Path):
    """Runner should pick up next-year continuation files automatically."""
    base = tmp_path / "era5_wind_pacific_2024.nc"
    jan = tmp_path / "era5_wind_pacific_2025_01.nc"
    feb = tmp_path / "era5_wind_pacific_2025_02.nc"
    base.touch()
    jan.touch()
    feb.touch()

    assert _loadable_era5_paths(base) == [base, jan, feb]


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


def test_load_experiment_profile_merges_defaults_and_resolves_paths(tmp_path: Path):
    """Experiment profiles should inherit defaults and resolve config paths."""
    config_path = tmp_path / "experiments.toml"
    config_path.write_text(
        """
[swopp3.experiments.demo]
description = "demo profile"
source_script = "scripts/demo.sh"
output_dir = "output/demo"

[swopp3.experiments.demo.defaults]
strategy = "optimised"
n_points = 178
wind_path_atlantic = "data/wind_atlantic.nc"

[[swopp3.experiments.demo.runs]]
name = "atlantic"
cases = ["AO_WPS", "AO_noWPS"]
wave_path_atlantic = "data/wave_atlantic.nc"
""".strip()
        + "\n"
    )

    profile = _load_experiment_profile(config_path, "demo")

    assert profile["name"] == "demo"
    assert profile["source_script"] == "scripts/demo.sh"
    assert profile["output_dir"] == (tmp_path / "output" / "demo").resolve()
    assert profile["runs"] == [
        {
            "name": "atlantic",
            "cases": ["AO_WPS", "AO_noWPS"],
            "strategy": "optimised",
            "n_points": 178,
            "wind_path_atlantic": (tmp_path / "data" / "wind_atlantic.nc").resolve(),
            "wave_path_atlantic": (tmp_path / "data" / "wave_atlantic.nc").resolve(),
        }
    ]


def test_load_experiment_profile_rejects_unknown_name(tmp_path: Path):
    """Unknown experiment keys should produce a useful error message."""
    config_path = tmp_path / "experiments.toml"
    config_path.write_text("[swopp3.experiments.demo]\n")

    with pytest.raises(KeyError, match="Unknown SWOPP3 experiment 'missing'"):
        _load_experiment_profile(config_path, "missing")


def test_main_runs_experiment_profile_and_writes_manifest(
    monkeypatch,
    tmp_path: Path,
):
    """Experiment-mode CLI should execute each resolved run and emit a manifest."""
    config_path = tmp_path / "experiments.toml"
    output_dir = tmp_path / "output" / "demo"
    config_path.write_text(
        f"""
[swopp3.experiments.demo]
description = "demo profile"
source_script = "scripts/demo.sh"
output_dir = "{output_dir}"

[swopp3.experiments.demo.defaults]
dt_eval_minutes = 30
distance_penalty_weight = 10

[[swopp3.experiments.demo.runs]]
name = "atlantic"
cases = ["AO_WPS"]
wind_path_atlantic = "/tmp/atlantic_wind.nc"
wave_path_atlantic = "/tmp/atlantic_wave.nc"
n_points = 178

[[swopp3.experiments.demo.runs]]
name = "pacific"
cases = ["PO_WPS"]
wind_path_pacific = "/tmp/pacific_wind.nc"
wave_path_pacific = "/tmp/pacific_wave.nc"
n_points = 293
""".strip()
        + "\n"
    )

    captured: list[dict[str, object]] = []

    def fake_run_swopp3_configuration(**kwargs):
        captured.append(kwargs)

    monkeypatch.setattr(
        _swopp3_run,
        "_run_swopp3_configuration",
        fake_run_swopp3_configuration,
    )

    _swopp3_run.main(
        experiment="demo",
        config_path=config_path,
        quiet=True,
    )

    assert len(captured) == 2
    assert captured[0]["cases"] == ["AO_WPS"]
    assert captured[0]["n_points"] == 178
    assert captured[0]["output_dir"] == output_dir
    assert captured[1]["cases"] == ["PO_WPS"]
    assert captured[1]["n_points"] == 293
    assert captured[1]["output_dir"] == output_dir

    manifest = output_dir / "experiment_manifest.json"
    assert manifest.exists()
    manifest_text = manifest.read_text()
    assert '"experiment": "demo"' in manifest_text
    assert '"source_script": "scripts/demo.sh"' in manifest_text
