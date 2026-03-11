"""Tests for routetools.swopp3_runner — case runner and energy evaluation."""

from __future__ import annotations

import csv
from datetime import UTC, datetime, timedelta
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from routetools.swopp3 import SWOPP3_CASES, great_circle_route
from routetools.swopp3_runner import (
    DepartureResult,
    evaluate_energy,
    run_case,
    run_gc_departure,
    run_optimised_departure,
    segment_bearings_deg,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_DEP = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
_N = 50  # waypoints for tests


def _atlantic_gc(n: int = _N) -> jnp.ndarray:
    """Great circle route Santander → New York."""
    src = jnp.array([-4.0, 43.6])
    dst = jnp.array([-73.80, 40.53])
    return great_circle_route(src, dst, n_points=n)


def _zero_windfield(lon, lat, t):
    """Wind field that returns zero everywhere."""
    return jnp.zeros_like(lon), jnp.zeros_like(lon)


def _constant_windfield(tws: float = 10.0, direction_deg: float = 270.0):
    """Wind field with constant speed and direction (FROM)."""
    dir_rad = np.radians(direction_deg)
    # Wind FROM direction_deg → components point opposite
    u10 = -tws * np.sin(dir_rad)
    v10 = -tws * np.cos(dir_rad)

    def _field(lon, lat, t):
        return jnp.full_like(lon, u10), jnp.full_like(lon, v10)

    return _field


def _constant_wavefield(hs: float = 2.0, mwd: float = 270.0):
    """Wave field with constant hs and direction."""

    def _field(lon, lat, t):
        return jnp.full_like(lon, hs), jnp.full_like(lon, mwd)

    return _field


# ---------------------------------------------------------------------------
# segment_bearings_deg
# ---------------------------------------------------------------------------
class TestSegmentBearings:
    def test_eastward(self):
        """Eastward route → bearing ≈ 90°."""
        curve = jnp.array([[0.0, 0.0], [1.0, 0.0]])
        b = segment_bearings_deg(curve)
        assert len(b) == 1
        assert abs(b[0] - 90.0) < 1.0

    def test_northward(self):
        """Northward route → bearing ≈ 0° (or 360°)."""
        curve = jnp.array([[0.0, 0.0], [0.0, 1.0]])
        b = segment_bearings_deg(curve)
        assert abs(b[0] % 360.0) < 1.0

    def test_westward(self):
        curve = jnp.array([[1.0, 0.0], [0.0, 0.0]])
        b = segment_bearings_deg(curve)
        assert abs(b[0] - 270.0) < 1.0

    def test_multiple_segments(self):
        curve = jnp.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        b = segment_bearings_deg(curve)
        assert len(b) == 2
        assert abs(b[0] - 90.0) < 1.0  # east
        assert abs(b[1] % 360.0) < 1.0  # north


# ---------------------------------------------------------------------------
# evaluate_energy
# ---------------------------------------------------------------------------
class TestEvaluateEnergy:
    def test_zero_wind_returns_hull_drag_only(self):
        """With zero wind and waves, energy should be pure hull drag."""
        curve = _atlantic_gc()
        energy, max_tws, max_hs = evaluate_energy(
            curve,
            _DEP,
            354.0,
            wps=False,
            windfield=_zero_windfield,
        )
        assert energy > 0, "Hull drag should produce positive energy"
        assert max_tws == pytest.approx(0.0, abs=1e-3)
        assert max_hs == 0.0

    def test_no_fields_no_energy(self):
        """With no fields at all (None), everything is zero wind/wave."""
        curve = _atlantic_gc()
        energy, max_tws, max_hs = evaluate_energy(
            curve,
            _DEP,
            354.0,
            wps=False,
        )
        assert energy > 0  # hull drag only

    def test_wps_reduces_energy(self):
        """WPS (wingsails) should never increase energy."""
        curve = _atlantic_gc()
        wf = _constant_windfield(tws=15.0, direction_deg=180.0)

        e_no_wps, _, _ = evaluate_energy(
            curve,
            _DEP,
            354.0,
            wps=False,
            windfield=wf,
        )
        e_wps, _, _ = evaluate_energy(
            curve,
            _DEP,
            354.0,
            wps=True,
            windfield=wf,
        )
        assert (
            e_wps <= e_no_wps + 1e-6
        ), f"WPS energy {e_wps} should be ≤ noWPS energy {e_no_wps}"

    def test_with_wavefield(self):
        """Wavefield should increase energy (added resistance)."""
        curve = _atlantic_gc()

        e_calm, _, _ = evaluate_energy(
            curve,
            _DEP,
            354.0,
            wps=False,
        )
        wv = _constant_wavefield(hs=4.0, mwd=270.0)
        e_waves, _, max_hs = evaluate_energy(
            curve,
            _DEP,
            354.0,
            wps=False,
            wavefield=wv,
        )
        assert e_waves > e_calm, "Waves should add resistance"
        assert max_hs == pytest.approx(4.0, abs=0.1)

    def test_max_tws_correct(self):
        curve = _atlantic_gc(20)
        wf = _constant_windfield(tws=12.5, direction_deg=0.0)
        _, max_tws, _ = evaluate_energy(
            curve,
            _DEP,
            354.0,
            wps=False,
            windfield=wf,
        )
        assert max_tws == pytest.approx(12.5, abs=0.5)

    def test_curve_with_single_point_raises(self):
        """Routes with fewer than two points are invalid for segment-based energy."""
        curve = jnp.array([[0.0, 0.0]])
        with pytest.raises(ValueError, match="at least 2 points"):
            evaluate_energy(curve, _DEP, 354.0, wps=False)


# ---------------------------------------------------------------------------
# run_gc_departure
# ---------------------------------------------------------------------------
class TestRunGCDeparture:
    def test_returns_result(self):
        result = run_gc_departure("AGC_WPS", _DEP, n_points=20)
        assert isinstance(result, DepartureResult)
        assert result.departure == _DEP
        assert result.curve.shape == (20, 2)
        assert result.distance_nm > 0
        assert result.energy_mwh > 0

    def test_gc_distance_reasonable(self):
        """Atlantic GC distance should be roughly 2800-3000 nm."""
        result = run_gc_departure("AGC_noWPS", _DEP, n_points=200)
        assert 2700 < result.distance_nm < 3200

    def test_pacific_distance(self):
        result = run_gc_departure("PGC_WPS", _DEP, n_points=200)
        assert 4500 < result.distance_nm < 5000


# ---------------------------------------------------------------------------
# run_optimised_departure  (without vectorfield → GC fallback)
# ---------------------------------------------------------------------------
class TestRunOptimisedDeparture:
    def test_fallback_without_vectorfield(self):
        """Without a vectorfield, should fall back to GC route."""
        result = run_optimised_departure("AO_WPS", _DEP, n_points=20)
        assert isinstance(result, DepartureResult)
        assert result.curve.shape == (20, 2)
        assert result.energy_mwh > 0

    def test_vectorfield_defaults_to_windfield_for_rise_cost(self, monkeypatch):
        """When windfield is missing, vectorfield should feed RISE energy cost."""
        captured: dict[str, object] = {}

        def fake_cost_function_rise(
            *,
            windfield,
            curve,
            travel_time,
            wavefield,
            wps,
            time_offset,
        ):
            captured["windfield"] = windfield
            return jnp.zeros(curve.shape[0], dtype=jnp.float32)

        def fake_optimize(*, vectorfield, src, dst, land=None, **kwargs):
            # Force one evaluation of the injected cost closure.
            _ = kwargs["cost_fn"](jnp.zeros((1, kwargs["L"], 2), dtype=jnp.float32))
            return great_circle_route(src, dst, n_points=kwargs["L"]), {"cost": 0.0}

        monkeypatch.setattr(
            "routetools.cost.cost_function_rise",
            fake_cost_function_rise,
        )
        monkeypatch.setattr("routetools.cmaes.optimize", fake_optimize)

        with pytest.warns(
            UserWarning,
            match="defaulting windfield to vectorfield",
        ):
            result = run_optimised_departure(
                "AO_WPS",
                _DEP,
                vectorfield=_zero_windfield,
                windfield=None,
                n_points=20,
            )

        assert isinstance(result, DepartureResult)
        assert captured["windfield"] is _zero_windfield


# ---------------------------------------------------------------------------
# run_case
# ---------------------------------------------------------------------------
class TestRunCase:
    def test_gc_case_no_output(self):
        """Run a GC case with 2 departures, no output dir."""
        deps = [_DEP, _DEP + timedelta(days=1)]
        results = run_case(
            "AGC_noWPS",
            deps,
            n_points=20,
            verbose=False,
        )
        assert len(results) == 2
        assert all(isinstance(r, DepartureResult) for r in results)

    def test_gc_case_with_output(self, tmp_path: Path):
        """Run a GC case and write output files."""
        deps = [_DEP, _DEP + timedelta(days=1)]
        run_case(
            "AGC_noWPS",
            deps,
            output_dir=tmp_path,
            submission=1,
            n_points=20,
            verbose=False,
        )
        # File A should exist
        fa = tmp_path / "IEUniversity-1-AGC_noWPS.csv"
        assert fa.exists(), f"File A not found: {fa}"
        with fa.open() as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2

        # File B should exist for each departure
        fb_dir = tmp_path / "tracks"
        assert fb_dir.exists()
        fb_files = list(fb_dir.glob("*.csv"))
        assert len(fb_files) == 2

    def test_optimised_fallback_case(self, tmp_path: Path):
        """Optimised case without vectorfield → GC fallback, with output."""
        deps = [_DEP]
        results = run_case(
            "AO_WPS",
            deps,
            output_dir=tmp_path,
            submission=1,
            n_points=20,
            verbose=False,
        )
        assert len(results) == 1
        fa = tmp_path / "IEUniversity-1-AO_WPS.csv"
        assert fa.exists()

    def test_all_cases_runnable(self):
        """Smoke test: every case can run with 1 departure."""
        for case_id in SWOPP3_CASES:
            results = run_case(
                case_id,
                [_DEP],
                n_points=10,
                verbose=False,
            )
            assert len(results) == 1
            assert results[0].energy_mwh > 0
