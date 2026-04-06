"""Shared configuration helpers for SWOPP3 analysis.

This module provides the stable, importable API used by both
``scripts/swopp3_analysis.py`` and its test suite.  Keeping these helpers
in a proper library module avoids the fragile ``importlib`` pattern that
would otherwise be needed to test them.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from functools import cache
from pathlib import Path


@dataclass(frozen=True)
class AnalysisPaths:
    """Filesystem locations used by the SWOPP3 analysis script."""

    output_dir: Path
    figs_dir: Path
    config_path: Path


# ---------------------------------------------------------------------------
# Experiment registry — all known experiments across all profiles
# ---------------------------------------------------------------------------
EXPERIMENTS_REGISTRY: dict[str, dict] = {
    # ── No-penalty profile (four-experiment) ────────────────────────────
    "no_penalty": {
        "folder": "swopp3_no_penalty",
        "label": "CMA-ES",
        "short": "No Penalty",
        "color": "#F23333",  # IE law red — unconstrained
        "color_light": "#FF9B9B",
        "hatch": "",
        "order": 1,
    },
    "no_penalty_fms": {
        "folder": "swopp3_no_penalty_fms",
        "label": "CMA-ES + FMS",
        "short": "No Penalty + FMS",
        "color": "#007A3D",  # emerald green — high contrast with red
        "color_light": "#5CC28A",
        "hatch": "///",
        "order": 2,
    },
    "penalty": {
        "folder": "swopp3_penalty",
        "label": "CMA-ES + Penalty",
        "short": "Penalty",
        "color": "#000066",  # IE primary ocean-blue — constrained
        "color_light": "#6080CC",
        "hatch": "",
        "order": 3,
    },
    "penalty_fms": {
        "folder": "swopp3_penalty_fms",
        "label": "CMA-ES + Penalty + FMS",
        "short": "Penalty + FMS",
        "color": "#E09400",  # amber — high contrast with dark navy
        "color_light": "#FFCC66",
        "hatch": "///",
        "order": 4,
    },
    # ── Sweep-combined profile (two-experiment) ──────────────────────────
    "sweep_combined": {
        "folder": "sweep_combined",
        "label": "CMA-ES",
        "short": "Sweep Combined",
        "color": "#F23333",  # IE law red — unconstrained
        "color_light": "#FF9B9B",
        "hatch": "",
        "order": 1,
    },
    "sweep_combined_fms": {
        "folder": "sweep_combined_fms",
        "label": "CMA-ES + FMS",
        "short": "Sweep Combined + FMS",
        "color": "#007A3D",  # emerald green — high contrast with red
        "color_light": "#5CC28A",
        "hatch": "///",
        "order": 2,
    },
}


@cache
def _configured_output_dirs(config_path: Path) -> dict[str, str]:
    """Return output-folder names declared in the SWOPP3 config file."""
    if not config_path.exists():
        return {}

    with config_path.open("rb") as handle:
        config = tomllib.load(handle)

    experiments = config.get("swopp3", {}).get("experiments", {})
    output_dirs: dict[str, str] = {}
    for experiment_name, experiment_config in experiments.items():
        output_dir = experiment_config.get("output_dir")
        if isinstance(output_dir, str) and output_dir:
            output_dirs[experiment_name] = Path(output_dir).name
    return output_dirs


def _experiment_folder(exp_key: str, paths: AnalysisPaths) -> str:
    """Return the folder name for one analysis experiment.

    Prefer config-driven folder names when the merged SWOPP3 experiment config
    defines a matching output directory. Keep the legacy folder names as a
    fallback so older result folders remain readable.
    """
    metadata = EXPERIMENTS_REGISTRY[exp_key]
    configured_dirs = _configured_output_dirs(paths.config_path)
    candidates: list[str] = []

    config_experiment = metadata.get("config_experiment")
    if isinstance(config_experiment, str):
        configured = configured_dirs.get(config_experiment)
        if configured is not None:
            candidates.append(configured)

    config_parent = metadata.get("config_parent")
    if isinstance(config_parent, str):
        configured_parent = configured_dirs.get(config_parent)
        if configured_parent is not None:
            candidates.append(f"{configured_parent}_fms")

    legacy_folder = str(metadata["folder"])
    candidates.append(legacy_folder)

    for candidate in candidates:
        if (paths.output_dir / candidate).exists():
            return candidate
    return candidates[0]
