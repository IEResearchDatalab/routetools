"""Tests for the daily adaptive-routing experiment helpers."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np


def load_module():
    """Load the experiment script as a testable module."""
    path = (
        Path(__file__).parents[1]
        / "scripts"
        / "experiments"
        / "adaptive_forecast_routing.py"
    )
    spec = spec_from_file_location("adaptive_forecast_routing", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sample_parameter_interval_advances_by_elapsed_time() -> None:
    """A 24/96 time fraction must end one quarter along the active plan."""
    module = load_module()
    curve = np.column_stack([np.linspace(0.0, 8.0, 9), np.zeros(9)])
    segment = module.sample_parameter_interval(curve, 0.0, 24.0 / 96.0, 7)
    np.testing.assert_allclose(segment[0], [0.0, 0.0])
    np.testing.assert_allclose(segment[-1], [2.0, 0.0])


def test_remaining_suffix_starts_at_sailed_endpoint() -> None:
    """The next plan initializer must preserve exact position continuity."""
    module = load_module()
    curve = np.column_stack([np.linspace(0.0, 10.0, 11), np.linspace(2.0, 4.0, 11)])
    fraction = 24.0 / 120.0
    sailed = module.sample_parameter_interval(curve, 0.0, fraction, 25)
    suffix = module.sample_parameter_interval(curve, fraction, 1.0, 120)
    np.testing.assert_allclose(sailed[-1], suffix[0])
    np.testing.assert_allclose(suffix[-1], curve[-1])
