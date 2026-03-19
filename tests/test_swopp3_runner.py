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
# run_optimised_departure
# ---------------------------------------------------------------------------
class TestRunOptimisedDeparture:
    def test_requires_vectorfield(self):
        """Optimised departures should fail fast when vectorfield is missing."""
        with pytest.raises(ValueError, match="requires a vectorfield"):
            run_optimised_departure("AO_WPS", _DEP, n_points=20)

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

        def fake_weather_penalty_smooth(
            curve,
            *,
            windfield,
            wavefield,
            travel_time,
            spherical_correction,
            time_offset,
        ):
            captured["penalty_windfield"] = windfield
            return jnp.zeros(curve.shape[0], dtype=jnp.float32)

        def fake_optimize(*, vectorfield, src, dst, land=None, **kwargs):
            # Force one evaluation of the injected cost closure.
            _ = kwargs["cost_fn"](jnp.zeros((1, kwargs["L"], 2), dtype=jnp.float32))
            return great_circle_route(src, dst, n_points=kwargs["L"]), {"cost": 0.0}

        monkeypatch.setattr(
            "routetools.swopp3_runner.cost_function_rise",
            fake_cost_function_rise,
        )
        monkeypatch.setattr(
            "routetools.swopp3_runner.weather_penalty_smooth",
            fake_weather_penalty_smooth,
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
                verbosity=0,
            )

        assert isinstance(result, DepartureResult)
        assert captured["windfield"] is _zero_windfield
        assert captured["penalty_windfield"] is _zero_windfield

    def test_fms_refines_curve_before_energy_evaluation(self, monkeypatch):
        """FMS output should be the route that gets evaluated and returned."""
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
            captured["fms_cost_travel_time"] = travel_time
            captured.setdefault("time_offsets", []).append(time_offset)
            return jnp.zeros(curve.shape[0], dtype=jnp.float32)

        def fake_weather_penalty_smooth(
            curve,
            *,
            windfield,
            wavefield,
            travel_time,
            spherical_correction,
            time_offset,
        ):
            captured.setdefault("penalty_time_offsets", []).append(time_offset)
            return jnp.zeros(curve.shape[0], dtype=jnp.float32)

        def fake_optimize(*, vectorfield, src, dst, land=None, **kwargs):
            captured["calls"] = ["cmaes"]
            curve = great_circle_route(src, dst, n_points=kwargs["L"])
            captured["cmaes_curve"] = curve
            return curve, {"cost": 0.0}

        def fake_optimize_fms(
            *,
            vectorfield,
            curve,
            land=None,
            wavefield=None,
            travel_time=None,
            costfun,
            time_offset=None,
            **kwargs,
        ):
            cast_calls = captured.setdefault("calls", [])
            cast_calls.append("fms")
            captured["fms_time_offset"] = time_offset
            _ = costfun(
                curve=curve[None, ...],
                travel_time=travel_time,
                time_offset=time_offset,
            )
            refined_curve = curve + jnp.array([0.1, 0.0])
            captured["refined_curve"] = refined_curve
            return refined_curve[None, ...], {"cost": [0.0]}

        def fake_evaluate_energy(
            curve,
            departure,
            passage_hours,
            wps,
            windfield=None,
            wavefield=None,
            departure_offset_h=0.0,
        ):
            if "refined_curve" in captured and jnp.allclose(
                curve, captured["refined_curve"]
            ):
                captured["evaluated_curve"] = curve
                return 11.0, 3.0, 4.0
            captured["evaluated_curve"] = curve
            return 12.0, 3.0, 4.0

        monkeypatch.setattr(
            "routetools.swopp3_runner.cost_function_rise",
            fake_cost_function_rise,
        )
        monkeypatch.setattr(
            "routetools.swopp3_runner.weather_penalty_smooth",
            fake_weather_penalty_smooth,
        )
        monkeypatch.setattr("routetools.cmaes.optimize", fake_optimize)
        monkeypatch.setattr("routetools.fms.optimize_fms", fake_optimize_fms)
        monkeypatch.setattr(
            "routetools.swopp3_runner.evaluate_energy",
            fake_evaluate_energy,
        )

        result = run_optimised_departure(
            "AO_WPS",
            _DEP,
            vectorfield=_zero_windfield,
            windfield=_zero_windfield,
            n_points=20,
            departure_offset_h=12.0,
        )

        assert isinstance(result, DepartureResult)
        assert captured["calls"] == ["cmaes", "fms"]
        assert jnp.allclose(result.curve, captured["refined_curve"])
        assert jnp.allclose(captured["evaluated_curve"], captured["refined_curve"])
        assert captured["fms_cost_travel_time"] == pytest.approx(354.0)
        assert captured["fms_time_offset"] == pytest.approx(12.0)
        assert captured["time_offsets"][-1] == pytest.approx(12.0)

    def test_fms_route_is_rejected_when_weather_limit_is_exceeded(self, monkeypatch):
        """FMS should be discarded when the refined route violates weather limits."""
        captured: dict[str, object] = {}

        def fake_optimize(*, vectorfield, src, dst, land=None, **kwargs):
            curve = great_circle_route(src, dst, n_points=kwargs["L"])
            captured["cmaes_curve"] = curve
            return curve, {"cost": 0.0}

        def fake_optimize_fms(
            *,
            vectorfield,
            curve,
            land=None,
            wavefield=None,
            travel_time=None,
            costfun,
            time_offset=None,
            **kwargs,
        ):
            refined_curve = curve + jnp.array([0.2, 0.0])
            captured["refined_curve"] = refined_curve
            return refined_curve[None, ...], {"cost": [0.0]}

        def fake_evaluate_energy(
            curve,
            departure,
            passage_hours,
            wps,
            windfield=None,
            wavefield=None,
            departure_offset_h=0.0,
        ):
            if "refined_curve" in captured and jnp.allclose(
                curve, captured["refined_curve"]
            ):
                return 10.0, 19.0, 8.0
            return 12.0, 18.0, 6.0

        monkeypatch.setattr("routetools.cmaes.optimize", fake_optimize)
        monkeypatch.setattr("routetools.fms.optimize_fms", fake_optimize_fms)
        monkeypatch.setattr(
            "routetools.swopp3_runner.evaluate_energy",
            fake_evaluate_energy,
        )

        result = run_optimised_departure(
            "AO_WPS",
            _DEP,
            vectorfield=_zero_windfield,
            windfield=_zero_windfield,
            n_points=20,
            verbosity=0,
        )

        assert isinstance(result, DepartureResult)
        assert jnp.allclose(result.curve, captured["cmaes_curve"])
        assert not jnp.allclose(result.curve, captured["refined_curve"])
        assert result.energy_mwh == pytest.approx(12.0)

    def test_cmaes_uses_penalized_cost_fms_uses_pure_energy_cost(self, monkeypatch):
        """CMA-ES uses penalized RISE cost; FMS uses pure energy only."""
        captured: dict[str, object] = {}

        def fake_optimize(*, vectorfield, src, dst, land=None, **kwargs):
            curve = great_circle_route(src, dst, n_points=kwargs["L"])
            captured["cmaes_curve"] = curve
            captured["cmaes_cost"] = kwargs["cost_fn"](
                jnp.zeros((1, kwargs["L"], 2), dtype=jnp.float32)
            )
            return curve, {"cost": 0.0}

        def fake_cost_function_rise(
            *,
            windfield,
            curve,
            travel_time,
            wavefield,
            wps,
            time_offset,
        ):
            captured.setdefault("rise_calls", []).append(
                {
                    "travel_time": travel_time,
                    "time_offset": time_offset,
                    "curve_shape": tuple(curve.shape),
                }
            )
            return jnp.full(curve.shape[0], 10.0)

        def fake_weather_penalty_smooth(
            curve,
            *,
            windfield,
            wavefield,
            travel_time,
            spherical_correction,
            time_offset,
        ):
            captured.setdefault("penalty_calls", []).append(
                {
                    "travel_time": travel_time,
                    "time_offset": time_offset,
                    "curve_shape": tuple(curve.shape),
                    "spherical_correction": spherical_correction,
                }
            )
            return jnp.full(curve.shape[0], 3.0)

        def fake_optimize_fms(
            *,
            vectorfield,
            curve,
            windfield=None,
            land=None,
            wavefield=None,
            travel_time=None,
            costfun,
            time_offset=None,
            enforce_weather_limits=False,
            tws_limit=20.0,
            hs_limit=7.0,
            spherical_correction=False,
            **kwargs,
        ):
            captured["windfield"] = windfield
            captured["enforce_weather_limits"] = enforce_weather_limits
            captured["tws_limit"] = tws_limit
            captured["hs_limit"] = hs_limit
            captured["travel_time"] = travel_time
            captured["time_offset"] = time_offset
            captured["spherical_correction"] = spherical_correction
            captured["enforce_weather_limits"] = enforce_weather_limits
            captured["fms_cost"] = costfun(
                curve=curve[None, ...],
                travel_time=travel_time,
                time_offset=time_offset,
            )
            return curve[None, ...], {"cost": [10.0]}

        def fake_evaluate_energy(
            curve,
            departure,
            passage_hours,
            wps,
            windfield=None,
            wavefield=None,
            departure_offset_h=0.0,
        ):
            return 10.0, 18.0, 6.0

        monkeypatch.setattr(
            "routetools.swopp3_runner.cost_function_rise",
            fake_cost_function_rise,
        )
        monkeypatch.setattr(
            "routetools.swopp3_runner.weather_penalty_smooth",
            fake_weather_penalty_smooth,
        )
        monkeypatch.setattr("routetools.cmaes.optimize", fake_optimize)
        monkeypatch.setattr("routetools.fms.optimize_fms", fake_optimize_fms)
        monkeypatch.setattr(
            "routetools.swopp3_runner.evaluate_energy",
            fake_evaluate_energy,
        )

        result = run_optimised_departure(
            "AO_WPS",
            _DEP,
            vectorfield=_zero_windfield,
            windfield=_zero_windfield,
            n_points=20,
            tws_limit=19.0,
            hs_limit=6.5,
            verbosity=0,
        )

        assert isinstance(result, DepartureResult)
        assert jnp.allclose(result.curve, captured["cmaes_curve"])
        assert captured["windfield"] is _zero_windfield
        assert jnp.allclose(captured["cmaes_cost"], jnp.array([13.0]))
        # FMS uses pure energy only (no weather_penalty_smooth), so its cost
        # matches cost_function_rise alone (10.0), not the penalized value (13.0).
        assert jnp.allclose(captured["fms_cost"], jnp.array([10.0]))
        assert captured["enforce_weather_limits"] is True
        assert captured["tws_limit"] == pytest.approx(19.0)
        assert captured["hs_limit"] == pytest.approx(6.5)
        assert captured["travel_time"] == pytest.approx(354.0)
        assert captured["time_offset"] == pytest.approx(0.0)
        assert captured["spherical_correction"] is True
        assert captured["rise_calls"]
        assert captured["penalty_calls"]
        assert captured["penalty_calls"][-1]["spherical_correction"] is True

    def test_feasible_fms_beats_infeasible_cmaes(self, monkeypatch):
        """A feasible FMS route should beat an infeasible CMA-ES route."""
        captured: dict[str, object] = {}

        def fake_optimize(*, vectorfield, src, dst, land=None, **kwargs):
            curve = great_circle_route(src, dst, n_points=kwargs["L"])
            captured["cmaes_curve"] = curve
            return curve, {"cost": 0.0}

        def fake_optimize_fms(
            *,
            vectorfield,
            curve,
            land=None,
            wavefield=None,
            travel_time=None,
            costfun,
            time_offset=None,
            **kwargs,
        ):
            refined_curve = curve + jnp.array([0.2, 0.0])
            captured["refined_curve"] = refined_curve
            return refined_curve[None, ...], {"cost": [0.0]}

        def fake_evaluate_energy(
            curve,
            departure,
            passage_hours,
            wps,
            windfield=None,
            wavefield=None,
            departure_offset_h=0.0,
        ):
            if "refined_curve" in captured and jnp.allclose(
                curve, captured["refined_curve"]
            ):
                return 13.0, 18.0, 6.0
            return 12.0, 18.0, 8.0

        monkeypatch.setattr("routetools.cmaes.optimize", fake_optimize)
        monkeypatch.setattr("routetools.fms.optimize_fms", fake_optimize_fms)
        monkeypatch.setattr(
            "routetools.swopp3_runner.evaluate_energy",
            fake_evaluate_energy,
        )

        result = run_optimised_departure(
            "AO_WPS",
            _DEP,
            vectorfield=_zero_windfield,
            windfield=_zero_windfield,
            n_points=20,
            verbosity=0,
        )

        assert isinstance(result, DepartureResult)
        assert jnp.allclose(result.curve, captured["refined_curve"])
        assert result.energy_mwh == pytest.approx(13.0)
        assert result.max_hs_m == pytest.approx(6.0)


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

    def test_incremental_output_replaces_stale_summary_and_appends_rows(
        self,
        tmp_path: Path,
        monkeypatch,
    ):
        """Incremental output should rewrite File A and append one row per result."""
        deps = [_DEP, _DEP + timedelta(days=1)]
        curves_by_departure = {
            deps[0]: jnp.array(
                [[-4.0, 43.6], [-38.9, 42.0], [-73.8, 40.53]],
                dtype=jnp.float32,
            ),
            deps[1]: jnp.array(
                [[-4.0, 43.6], [-39.5, 41.5], [-73.8, 40.53]],
                dtype=jnp.float32,
            ),
        }
        summary_path = tmp_path / "IEUniversity-1-AGC_noWPS.csv"
        summary_path.write_text("stale_header\nstale_row\n")

        def fake_run_gc_departure(
            case_id,
            departure,
            windfield=None,
            wavefield=None,
            departure_offset_h=0.0,
            n_points=100,
        ):
            curve = curves_by_departure[departure]
            day_index = (departure - deps[0]).days
            return DepartureResult(
                departure=departure,
                curve=curve,
                energy_mwh=100.0 + day_index,
                max_tws_mps=10.0 + day_index,
                max_hs_m=2.0 + day_index,
                distance_nm=3000.0 + day_index,
                comp_time_s=1.0 + day_index,
            )

        monkeypatch.setattr(
            "routetools.swopp3_runner.run_gc_departure",
            fake_run_gc_departure,
        )

        run_case(
            "AGC_noWPS",
            deps,
            output_dir=tmp_path,
            submission=1,
            n_points=3,
            verbose=False,
        )

        summary_lines = summary_path.read_text().splitlines()
        assert summary_lines == [
            (
                "departure_time_utc,arrival_time_utc,energy_cons_mwh,"
                "max_wind_mps,max_hs_m,sailed_distance_nm,details_filename"
            ),
            summary_lines[1],
            summary_lines[2],
        ]
        assert all("stale" not in line for line in summary_lines)

        with summary_path.open() as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 2
        assert [row["departure_time_utc"] for row in rows] == [
            dep.strftime("%Y-%m-%d %H:%M:%S") for dep in deps
        ]
        assert len({row["details_filename"] for row in rows}) == 2

        for dep, row in zip(deps, rows, strict=False):
            track_path = tmp_path / "tracks" / row["details_filename"]
            assert track_path.exists(), f"Track CSV not found: {track_path}"
            with track_path.open() as f:
                track_rows = list(csv.DictReader(f))
            assert len(track_rows) == curves_by_departure[dep].shape[0]
            assert track_rows[0]["time_utc"] == dep.strftime("%Y-%m-%d %H:%M:%S")

    def test_optimised_case_with_vectorfield(self, tmp_path: Path):
        """Optimised case writes output when the required vectorfield is provided."""
        import warnings

        deps = [_DEP]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            results = run_case(
                "AO_WPS",
                deps,
                vectorfield=_zero_windfield,
                windfield=_zero_windfield,
                output_dir=tmp_path,
                submission=1,
                n_points=20,
                verbose=False,
            )
        assert len(results) == 1
        fa = tmp_path / "IEUniversity-1-AO_WPS.csv"
        assert fa.exists()

    def test_optimised_case_requires_vectorfield(self):
        """Optimised cases should fail fast instead of silently degrading to GC."""
        with pytest.raises(ValueError, match="requires a vectorfield"):
            run_case(
                "AO_WPS",
                [_DEP],
                n_points=20,
                verbose=False,
            )

    def test_all_cases_runnable(self):
        """Smoke test: every case can run with 1 departure."""
        for case_id in SWOPP3_CASES:
            kwargs = {}
            if SWOPP3_CASES[case_id]["strategy"] == "optimised":
                kwargs["vectorfield"] = _zero_windfield
                kwargs["windfield"] = _zero_windfield
            results = run_case(
                case_id,
                [_DEP],
                n_points=10,
                verbose=False,
                **kwargs,
            )
            assert len(results) == 1
            assert results[0].energy_mwh > 0
