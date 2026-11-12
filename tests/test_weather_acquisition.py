"""Regression checks against silent incomplete weather ingestion."""

import importlib.util
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

pytestmark = pytest.mark.filterwarnings(
    "ignore:numpy.ndarray size changed:RuntimeWarning"
)

spec = importlib.util.spec_from_file_location(
    "acquire_weather", Path(__file__).parents[1] / "scripts/data/acquire_weather.py"
)
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def rows():
    return [
        dict(
            param=p,
            number=str(m),
            type="cf" if m == 0 else "pf",
            date="20250115",
            time="0000",
            step="0",
            stream="waef",
            _offset=10 * i,
            _length=10,
        )
        for i, (p, m) in enumerate((p, m) for p in M.PARAMS["waef"] for m in range(51))
    ]


def test_missing_member_and_duplicates_fail_before_download():
    rs = rows()
    assert len(M.select_rows(rs, "waef", datetime(2025, 1, 15), 0)) == 153
    with pytest.raises(ValueError, match="incomplete"):
        M.select_rows(rs[:-1], "waef", datetime(2025, 1, 15), 0)
    with pytest.raises(ValueError, match="incomplete"):
        M.select_rows(rs + [rs[0]], "waef", datetime(2025, 1, 15), 0)
    with pytest.raises(ValueError, match="does not match"):
        M.select_rows(rs, "waef", datetime(2025, 1, 16), 0)


def test_short_cycle_cannot_request_15_days():
    assert len(M.steps_for(0)) == 85
    assert M.steps_for(12)[-1] == 360
    assert len(M.steps_for(6)) == 49
    assert M.steps_for(18)[-1] == 144


def test_ranges_do_not_include_unselected_messages():
    assert M.byte_ranges(
        [
            dict(_offset=0, _length=10),
            dict(_offset=10, _length=10),
            dict(_offset=30, _length=10),
        ]
    ) == [(0, 19), (30, 39)]


def make_dataset():
    lat, lon = M.grid("atlantic")
    name = M.PARAMS["wave"]["swh"]
    expected = dict(
        corridor="atlantic",
        initialization="2025-01-15T00:00:00",
        lead=0,
        stream="wave",
        ensemble=False,
        variables=[name],
    )
    ds = xr.Dataset(
        {
            name: (
                ("step", "latitude", "longitude"),
                np.ones((1, len(lat), len(lon)), dtype="float32"),
                {"units": "m"},
            )
        },
        coords=dict(
            latitude=lat,
            longitude=lon,
            step=np.array([0], dtype="timedelta64[h]"),
            time=np.datetime64("2025-01-15"),
            valid_time=("step", np.array(["2025-01-15"], dtype="datetime64[ns]")),
        ),
    )
    return ds, expected


def test_wrong_lead_and_all_missing_are_rejected():
    ds, expected = make_dataset()
    M.validate(ds, expected)
    with pytest.raises(AssertionError):
        M.validate(
            ds.assign_coords(step=np.array([3], dtype="timedelta64[h]")), expected
        )
    bad = ds.copy(deep=True)
    bad[expected["variables"][0]].values[:] = np.nan
    with pytest.raises(ValueError, match="all-missing"):
        M.validate(bad, expected)


def test_resume_requires_receipt_exact_request_and_hash(tmp_path):
    ds, expected = make_dataset()
    target = tmp_path / "lead_000.nc"
    ds.to_netcdf(target)
    assert not M.completed(target, expected)
    M.atomic_json(
        target.with_suffix(".json"),
        dict(status="complete", request=expected, sha256=M.digest(target)),
    )
    assert M.completed(target, expected)
    changed = dict(expected, lead=3)
    assert not M.completed(target, changed)
    with target.open("ab") as f:
        f.write(b"corrupted")
    assert not M.completed(target, expected)


def test_pacific_crop_preserves_dateline_grid():
    lat = np.arange(90.0, -90.25, -0.25)
    lon = np.arange(-180.0, 180.0, 0.25)
    ds = xr.Dataset(coords=dict(latitude=lat, longitude=lon))
    out = M.crop(ds, "pacific")
    assert out.sizes == {"latitude": 161, "longitude": 481}
    np.testing.assert_array_equal(out.longitude, np.arange(120, 240.25, 0.25))


def test_direction_packing_tolerance_preserves_small_overshoots():
    ds, expected = make_dataset()
    old = expected["variables"][0]
    name = M.PARAMS["wave"]["mwd"]
    ds = ds.rename({old: name})
    ds[name].attrs["units"] = "Degree true"
    expected["variables"] = [name]
    ds[name].values[:] = 360.004
    M.validate(ds, expected)
    assert float(ds[name].max()) > 360
    ds[name].values[:] = 360.1
    with pytest.raises(ValueError, match="packing tolerance"):
        M.validate(ds, expected)


def test_units_and_final_era5_cutoff_are_enforced():
    ds, expected = make_dataset()
    ds[expected["variables"][0]].attrs["units"] = "feet"
    with pytest.raises(ValueError, match="unexpected units"):
        M.validate(ds, expected)
    from datetime import date

    ds.attrs["valid_time_stop"] = "2025-11-30"
    M.require_final_era5(ds, date(2025, 1, 15))
    with pytest.raises(ValueError, match="final ERA5"):
        M.require_final_era5(ds, date(2025, 12, 1))


def test_raw_grib_time_cannot_be_overwritten_into_request():
    ds, _ = make_dataset()
    M.validate_source_coordinates(ds, datetime(2025, 1, 15), 0)
    with pytest.raises(AssertionError):
        M.validate_source_coordinates(
            ds.assign_coords(time=np.datetime64("2025-01-16")), datetime(2025, 1, 15), 0
        )


def test_decode_rejects_wrong_grib_time_before_relabeling(monkeypatch):
    ds, _ = make_dataset()
    name = M.PARAMS["wave"]["swh"]
    raw = ds.rename({name: "swh"}).isel(step=0)
    raw = raw.assign_coords(time=np.datetime64("2025-01-14"))
    monkeypatch.setattr(
        M.xr, "open_dataset", lambda *args, **kwargs: raw.copy(deep=True)
    )
    with pytest.raises(AssertionError):
        M.decode(Path("unused.grib2"), "wave", "atlantic", datetime(2025, 1, 15), 0)
