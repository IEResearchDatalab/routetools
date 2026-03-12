"""Tests for the land-avoidance demo CLI."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import typer


def _load_demo_module():
    """Load the demo module directly from scripts/demo_land_avoidance.py."""
    module_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "demo_land_avoidance.py"
    )
    spec = importlib.util.spec_from_file_location("demo_land_avoidance", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_demo = _load_demo_module()


def test_main_rejects_unknown_src_port():
    """CLI validation should raise BadParameter for an unknown source port."""
    with pytest.raises(typer.BadParameter, match="Unknown src_port"):
        _demo.main(src_port="NOPE")


def test_main_rejects_unknown_dst_port():
    """CLI validation should raise BadParameter for an unknown destination port."""
    with pytest.raises(typer.BadParameter, match="Unknown dst_port"):
        _demo.main(dst_port="NOPE")
