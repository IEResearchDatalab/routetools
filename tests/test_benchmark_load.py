import pytest

from routetools._ports import DICT_INSTANCES
from routetools.benchmark import LandBenchmark, load_benchmark_instance


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


def test_load_benchmark_instance_basic(monkeypatch):
    """Basic smoke test using the first configured instance.

    Monkeypatch `load_files` to avoid reading netCDF files and let `Ocean`
    construct synthetic zero datasets.
    """
    monkeypatch.setattr("routetools.wrr_bench.load.load_files", lambda *a, **k: {})
    instance_name = next(iter(DICT_INSTANCES.keys()))
    out = load_benchmark_instance(instance_name)
    assert_basic_output(out)


@pytest.mark.parametrize("instance_name", list(DICT_INSTANCES.keys()))
def test_load_benchmark_for_all_instances(monkeypatch, instance_name):
    """Run load_benchmark_instance for every configured instance (smoke test)."""
    monkeypatch.setattr("routetools.wrr_bench.load.load_files", lambda *a, **k: {})
    out = load_benchmark_instance(instance_name)
    # Basic smoke checks
    assert_basic_output(out)
