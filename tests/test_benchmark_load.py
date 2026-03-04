import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from routetools._ports import DICT_INSTANCES
from routetools.benchmark import LandBenchmark, load_benchmark_instance
from routetools.wrr_bench.ocean import data_zero


def assert_basic_output(out):
    # Check presence of keys
    for key in (
        "lat_start",
        "lon_start",
        "lat_end",
        "lon_end",
        "src",
        "dst",
        "data",
        "vectorfield",
        "wavefield",
        "land",
    ):
        assert key in out

    # src/dst shapes
    assert hasattr(out["src"], "shape") and out["src"].shape == (2,)
    assert hasattr(out["dst"], "shape") and out["dst"].shape == (2,)

    # data and callables
    assert callable(out["vectorfield"]) and callable(out["wavefield"])
    assert isinstance(out["land"], LandBenchmark)


def test_load_benchmark_instance_basic(tmp_path):
    """Basic smoke test using the first configured instance.

    Create minimal NetCDF files under a temporary `data_path` and call the
    loader with that `data_path` so no monkeypatching is required.
    """
    data_dir = Path(tmp_path) / "data"
    (data_dir / "currents").mkdir(parents=True)
    (data_dir / "waves").mkdir(parents=True)
    # Create a minimal geojson land file that the loader can read
    land_geo = {
        "type": "GeometryCollection",
        "geometries": [
            {
                "type": "Polygon",
                "coordinates": [
                    [[-180, -90], [180, -90], [180, 90], [-180, 90], [-180, -90]]
                ],
            }
        ],
    }
    with open(data_dir / "earth-seas-2km5-valid.geo.json", "w") as f:
        json.dump(land_geo, f)

    instance_name = next(iter(DICT_INSTANCES.keys()))
    date_start = DICT_INSTANCES[instance_name].get("date_start", "2023-01-01")
    date_base = date_start.split("T")[0]
    for day in range(15):
        d = np.datetime64(date_base) + np.timedelta64(day, "D")
        datestr = str(d)
        ds_c = data_zero(bounding_box=None, data_vars=("vo", "uo"))
        ds_w = data_zero(bounding_box=None, data_vars=("height", "direction"))
        # Reduce to a single timestamp per file (use the file date) so that
        # concatenated datasets have evenly spaced times
        ds_c = ds_c.isel(time=0).expand_dims(time=[np.datetime64(datestr)])
        ds_w = ds_w.isel(time=0).expand_dims(time=[np.datetime64(datestr)])
        ds_c.to_netcdf(data_dir / "currents" / f"{datestr}.nc", engine="scipy")
        ds_w.to_netcdf(data_dir / "waves" / f"{datestr}.nc", engine="scipy")

    # Suppress netCDF4 C-extension ABI RuntimeWarning (may be raised on import)
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="numpy.ndarray size changed, may indicate binary incompatibility",
            category=RuntimeWarning,
        )
        out = load_benchmark_instance(
            instance_name, date_start=date_base, data_path=str(data_dir), route_days=15
        )
    assert_basic_output(out)


@pytest.mark.parametrize("instance_name", list(DICT_INSTANCES.keys()))
def test_load_benchmark_for_all_instances(tmp_path, instance_name):
    """Run load_benchmark_instance for every configured instance (smoke test)."""
    data_dir = Path(tmp_path) / "data"
    (data_dir / "currents").mkdir(parents=True)
    (data_dir / "waves").mkdir(parents=True)
    # Create a minimal geojson land file that the loader can read
    land_geo = {
        "type": "GeometryCollection",
        "geometries": [
            {
                "type": "Polygon",
                "coordinates": [
                    [[-180, -90], [180, -90], [180, 90], [-180, 90], [-180, -90]]
                ],
            }
        ],
    }
    with open(data_dir / "earth-seas-2km5-valid.geo.json", "w") as f:
        json.dump(land_geo, f)

    date_start = DICT_INSTANCES[instance_name].get("date_start", "2023-01-01")
    date_base = date_start.split("T")[0]
    for day in range(15):
        d = np.datetime64(date_base) + np.timedelta64(day, "D")
        datestr = str(d)
        ds_c = data_zero(bounding_box=None, data_vars=("vo", "uo"))
        ds_w = data_zero(bounding_box=None, data_vars=("height", "direction"))
        # Reduce to a single timestamp per file (use the file date) so that
        # concatenated datasets have evenly spaced times
        ds_c = ds_c.isel(time=0).expand_dims(time=[np.datetime64(datestr)])
        ds_w = ds_w.isel(time=0).expand_dims(time=[np.datetime64(datestr)])
        ds_c.to_netcdf(data_dir / "currents" / f"{datestr}.nc", engine="scipy")
        ds_w.to_netcdf(data_dir / "waves" / f"{datestr}.nc", engine="scipy")

    # Suppress netCDF4 C-extension ABI RuntimeWarning (may be raised on import)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="numpy.ndarray size changed, may indicate binary incompatibility",
            category=RuntimeWarning,
        )
        out = load_benchmark_instance(
            instance_name, date_start=date_base, data_path=str(data_dir), route_days=15
        )
    # Basic smoke checks
    assert_basic_output(out)
