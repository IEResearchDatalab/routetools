"""Focused tests for the Task 0 reconciliation script (small tmp fixtures only)."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "revision2"
    / ("task0_config_reconciliation.py")
)
spec = importlib.util.spec_from_file_location("task0", SCRIPT)
task0 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(task0)

HEADER = "departure_time_utc,energy_cons_mwh,max_wind_mps,max_hs_m\n"


def _dates(n):
    return [f"2024-01-01 {i // 60:02d}:{i % 60:02d}:00" for i in range(n)]


def _write(path, energies, dates=None, n=3):
    dates = dates or _dates(len(energies))
    lines = [
        f"{d},{e},{20 + i * 0.1},{7 + i * 0.01}"
        for i, (d, e) in enumerate(zip(dates, energies, strict=True))
    ]
    path.write_text(HEADER + "\n".join(lines) + "\n", encoding="utf-8")


def test_read_summary_parses_and_keys_by_date(tmp_path):
    f = tmp_path / "a.csv"
    _write(f, [1.0, 2.0, 3.0])
    data = task0.read_summary(f, expected=3)
    assert data[_dates(3)[1]]["energy_cons_mwh"] == 2.0


def test_read_summary_inventory_failures(tmp_path):
    f = tmp_path / "a.csv"
    with pytest.raises(FileNotFoundError):
        task0.read_summary(f, expected=3)
    _write(f, [1.0, 2.0])
    with pytest.raises(ValueError, match="expected 3"):
        task0.read_summary(f, expected=3)
    _write(f, [1.0, 2.0, 3.0], dates=[_dates(1)[0]] * 3)
    with pytest.raises(ValueError, match="duplicate"):
        task0.read_summary(f, expected=3)
    f.write_text("departure_time_utc,energy_cons_mwh\nx,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="lacks columns"):
        task0.read_summary(f, expected=1)
    _write(f, [1.0], dates=["not-a-date"])
    with pytest.raises(ValueError, match="Invalid row"):
        task0.read_summary(f, expected=1)


def test_pairing_is_by_timestamp_not_order():
    d = _dates(2)
    base = {d[0]: {"energy_cons_mwh": 100.0}, d[1]: {"energy_cons_mwh": 200.0}}
    cand = {d[1]: {"energy_cons_mwh": 100.0}, d[0]: {"energy_cons_mwh": 90.0}}
    assert task0.paired_improvement(base, cand) == pytest.approx((10 + 50) / 2)
    other = {"2025-01-01 00:00:00": {"energy_cons_mwh": 1.0}, d[0]: cand[d[0]]}
    with pytest.raises(ValueError, match="different departure dates"):
        task0.paired_improvement(base, other)


def test_rounding_comparison():
    assert task0.matches(310.346441, 310.3, 1)
    assert not task0.matches(310.36, 310.3, 1)
    assert task0.matches(27.534, 27.53, 2)
    row = task0.make_row("r", "s", "c", "p", 5.97, 5.9741, 2, "direct")
    assert row["match"] == "yes"
    assert task0.make_row("r", "s", "c", "p", 20, 19.9, 1, "x")["match"] == "no"
    assert task0.make_row("r", "s", "c", "p", None, "x", None, "x")["match"] == "n/a"
    unknown = task0.make_row("r", "s", "c", "p", 1, None, None, "unverified")
    assert unknown["match"] == "unverified"


def _build_roots(tmp_path, bers_gc_energy=None, shift_date=False):
    cm, be = tmp_path / "cm", tmp_path / "be"
    cm.mkdir()
    be.mkdir()
    for case, detail in task0.CASES.items():
        gc = detail["gc"]
        _write(cm / f"IEUniversity-1-{gc}.csv", [10.0, 10.0, 10.0])
        _write(be / f"IEUniversity-1-{gc}.csv", bers_gc_energy or [10.0] * 3)
        _write(cm / f"IEUniversity-1-{case}.csv", [8.0, 8.0, 8.0])
        dates = _dates(3)
        if shift_date:
            dates[2] = "2030-01-01 00:00:00"
        _write(be / f"IEUniversity-1-{case}.csv", [6.0] * 3, dates=dates)
    return cm, be


def test_load_summaries_and_gc_consistency(tmp_path, monkeypatch):
    monkeypatch.setattr(task0, "N_DEPARTURES", 3)
    monkeypatch.setattr(
        task0, "read_summary", lambda p, e=3, f=task0.read_summary: f(p, 3)
    )
    cm, be = _build_roots(tmp_path)
    out = task0.load_summaries(cm, be)
    assert set(out) == set(task0.CASES)
    assert task0.mean_energy(out["AO_WPS"]["BERS"]) == 6.0
    (be / "IEUniversity-1-AGC_WPS.csv").unlink()
    with pytest.raises(FileNotFoundError):
        task0.load_summaries(cm, be)


def test_load_summaries_rejects_inconsistent_data(tmp_path, monkeypatch):
    monkeypatch.setattr(
        task0, "read_summary", lambda p, e=3, f=task0.read_summary: f(p, 3)
    )
    cm, be = _build_roots(tmp_path, bers_gc_energy=[10.0, 10.0, 11.0])
    with pytest.raises(ValueError, match="GC data differs"):
        task0.load_summaries(cm, be)


def test_load_summaries_rejects_mismatched_dates(tmp_path, monkeypatch):
    monkeypatch.setattr(
        task0, "read_summary", lambda p, e=3, f=task0.read_summary: f(p, 3)
    )
    (tmp_path / "z").mkdir()
    cm, be = _build_roots(tmp_path / "z", shift_date=True)
    with pytest.raises(ValueError, match="Departure dates"):
        task0.load_summaries(cm, be)


def test_non_366_inventory_fails_with_default_bound(tmp_path):
    f = tmp_path / "a.csv"
    _write(f, [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="expected 366"):
        task0.read_summary(f)


def test_legacy_versus_manifest_configuration_handling(tmp_path):
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    assert task0.root_provenance_basis(legacy) == "legacy"
    run_root = tmp_path / "run"
    (run_root / "bers").mkdir(parents=True)
    (run_root / "bers" / "experiment_manifest.json").write_text("{}")
    assert task0.root_provenance_basis(run_root / "bers") == "manifest"

    rows = task0.legacy_config_rows()
    assert {r["role"] for r in rows} == {task0.ROLE_PRIMARY}
    assert {r["provenance"] for r in rows} == {
        "direct",
        "reconstructed",
        "unverified",
    }
    # Reconstructed FMS launcher settings are never labelled as direct.
    for r in rows:
        if r["provenance"] != "unverified" and r["parameter"].startswith(
            ("FMS", "Wind threshold", "Wave threshold")
        ):
            assert r["provenance"] == "reconstructed"
    thresholds = [r for r in rows if "threshold used by FMS" in r["parameter"]]
    assert thresholds
    assert all(r["match"] == "no" for r in thresholds)
    assert not any(
        r["match"] == "no" for r in rows if "threshold" not in r["parameter"]
    )


def _later_run(tmp_path, pacific_popsize=200):
    import json

    keys = {
        "dt_eval_minutes": 30,
        "wind_penalty_weight": 50,
        "wave_penalty_weight": 50,
        "distance_penalty_weight": 0,
        "land_distance_weight": 100,
        "land_distance_epsilon": 1,
        "land_crossing_penalty": 1e6,
        "cmaes_k": 10,
        "sigma0": 0.1,
        "popsize": 200,
        "maxfevals": 25000,
        "fms_patience": 50,
        "fms_damping": 0.9,
        "fms_maxfevals": 5000,
    }
    (tmp_path / "bers").mkdir(parents=True)
    runs = [
        {"name": "atlantic", "n_points": 178, **keys},
        {"name": "pacific", "n_points": 293, **keys, "popsize": pacific_popsize},
    ]
    (tmp_path / "bers" / "experiment_manifest.json").write_text(
        json.dumps({"experiment": "e", "runs": runs}), encoding="utf-8"
    )
    toml = "[swopp3.experiments.e.defaults]\n" + "\n".join(
        f"{k} = {v!r}" for k, v in keys.items()
    )
    (tmp_path / "resolved_config.toml").write_text(toml, encoding="utf-8")
    return tmp_path


def test_manifest_rows_are_direct_except_unrecorded_thresholds(tmp_path):
    rows = task0.manifest_config_rows(_later_run(tmp_path))
    assert {r["role"] for r in rows} == {task0.ROLE_LATER}
    for r in rows:
        expected = "reconstructed" if "threshold" in r["parameter"] else "direct"
        assert r["provenance"] == expected
    popsize = [r for r in rows if r["parameter"] == "CMA-ES population P"]
    assert all(r["match"] == "no" for r in popsize)


def test_manifest_rejects_divergent_second_run(tmp_path):
    run = _later_run(tmp_path, pacific_popsize=400)
    with pytest.raises(ValueError, match="run pacific"):
        task0.manifest_config_rows(run)
