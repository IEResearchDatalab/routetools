"""Regression tests for the CodaBench scorer's ERA5 re-evaluation."""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")

_SCORER_PATH = (
    Path(__file__).resolve().parents[1] / "codabench" / "scoring_program" / "scoring.py"
)
_DEPARTURE = datetime(2024, 1, 1, 12, 0, 0)


@pytest.fixture(scope="module")
def scoring(tmp_path_factory):
    """Import the scorer, which reads ``sys.argv`` at module level."""
    tmp = tmp_path_factory.mktemp("scorer")
    argv = sys.argv
    sys.argv = ["scoring.py", str(tmp / "input"), str(tmp / "output")]
    try:
        spec = importlib.util.spec_from_file_location("codabench_scoring", _SCORER_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.argv = argv
    return module


def _write_atlantic_era5(ref_dir: Path, mwd: np.ndarray) -> None:
    """Write calm-wind, constant-Hs Atlantic files with the given MWD field."""
    lat = np.arange(44.0, 39.75, -0.25)
    lon = np.arange(-10.0, -1.75, 0.25)
    hours = np.arange(0, 48, dtype=np.float64)
    shape = (len(hours), len(lat), len(lon))
    fields = {
        "wind": {"u10": np.zeros(shape), "v10": np.zeros(shape)},
        "waves": {"swh": np.full(shape, 3.0), "mwd": np.broadcast_to(mwd, shape)},
    }
    for kind, variables in fields.items():
        with netCDF4.Dataset(ref_dir / f"era5_{kind}_atlantic_2024.nc", "w") as ds:
            ds.createDimension("valid_time", len(hours))
            ds.createDimension("latitude", len(lat))
            ds.createDimension("longitude", len(lon))
            t = ds.createVariable("valid_time", "f8", ("valid_time",))
            t.units = "hours since 2024-01-01 00:00:00"
            t[:] = hours
            ds.createVariable("latitude", "f8", ("latitude",))[:] = lat
            ds.createVariable("longitude", "f8", ("longitude",))[:] = lon
            for name, values in variables.items():
                dims = ("valid_time", "latitude", "longitude")
                ds.createVariable(name, "f4", dims)[:] = values


def _southbound_energy(scoring, ref_dir: Path) -> float:
    evaluate = scoring.try_load_era5_scorer(ref_dir)
    waypoints = [
        (_DEPARTURE, 43.0, -6.125),
        (_DEPARTURE + timedelta(hours=6), 42.0, -6.125),
    ]
    result = evaluate("AO_noWPS", waypoints, str(_DEPARTURE))
    return result["energy_mwh"]


def test_mean_wave_direction_is_interpolated_as_an_angle(scoring, tmp_path):
    """Waves from 359°/1° must stay northerly, not average to southerly 180°."""
    lon_count = len(np.arange(-10.0, -1.75, 0.25))
    alternating = np.where(np.arange(lon_count) % 2 == 0, 359.0, 1.0)
    northerly = np.zeros(lon_count)

    (tmp_path / "alternating").mkdir()
    (tmp_path / "northerly").mkdir()
    _write_atlantic_era5(tmp_path / "alternating", alternating)
    _write_atlantic_era5(tmp_path / "northerly", northerly)

    energy_alternating = _southbound_energy(scoring, tmp_path / "alternating")
    energy_northerly = _southbound_energy(scoring, tmp_path / "northerly")

    assert energy_alternating == pytest.approx(energy_northerly, rel=1e-3)
