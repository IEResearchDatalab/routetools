"""Tests for SWOPP3 analysis script helpers."""

import importlib.util
import sys
from pathlib import Path


def _load_swopp3_analysis_module():
    """Load the analysis script directly from scripts/swopp3_analysis.py."""
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "swopp3_analysis.py"
    spec = importlib.util.spec_from_file_location("swopp3_analysis", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_swopp3_analysis = _load_swopp3_analysis_module()
AnalysisPaths = _swopp3_analysis.AnalysisPaths
_configured_output_dirs = _swopp3_analysis._configured_output_dirs
_experiment_folder = _swopp3_analysis._experiment_folder


def test_configured_output_dirs_reads_swopp3_profiles(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[swopp3.experiments.no_penalty]
output_dir = "output/swopp3_no_penalty"

[swopp3.experiments.split_penalty]
output_dir = "output/swopp3_split_penalty"
""".strip()
    )

    output_dirs = _configured_output_dirs(config_path)

    assert output_dirs == {
        "no_penalty": "swopp3_no_penalty",
        "split_penalty": "swopp3_split_penalty",
    }


def test_experiment_folder_prefers_existing_legacy_output_when_config_target_missing(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[swopp3.experiments.split_penalty]
output_dir = "output/swopp3_split_penalty"
""".strip()
    )
    (tmp_path / "output" / "swopp3_penalty").mkdir(parents=True)

    paths = AnalysisPaths(
        output_dir=tmp_path / "output",
        figs_dir=tmp_path / "analysis",
        config_path=config_path,
    )

    assert _experiment_folder("penalty", paths) == "swopp3_penalty"


def test_experiment_folder_uses_configured_folder_when_present(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[swopp3.experiments.no_penalty]
output_dir = "output/swopp3_no_penalty"
""".strip()
    )
    (tmp_path / "output" / "swopp3_no_penalty").mkdir(parents=True)

    paths = AnalysisPaths(
        output_dir=tmp_path / "output",
        figs_dir=tmp_path / "analysis",
        config_path=config_path,
    )

    assert _experiment_folder("no_penalty", paths) == "swopp3_no_penalty"
