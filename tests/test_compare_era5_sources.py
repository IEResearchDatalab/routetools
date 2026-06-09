"""Tests for scripts/compare_era5_sources.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import routetools.era5.download_gcs as download_gcs


def _load_compare_module():
    """Load the comparison script directly from scripts/compare_era5_sources.py."""
    module_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "compare_era5_sources.py"
    )
    spec = importlib.util.spec_from_file_location("compare_era5_sources", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_compare_era5_sources = _load_compare_module()


def test_main_downloads_hourly_gcs_by_default(tmp_path: Path, monkeypatch):
    """The GCS comparison helper should use hourly downloads by default."""
    cds_dir = tmp_path / "cds"
    gcs_dir = tmp_path / "gcs"
    cds_dir.mkdir()

    for field in ("wind", "waves"):
        (cds_dir / f"era5_{field}_atlantic_2024.nc").touch()

    calls: list[tuple[str, int, list[int]]] = []

    def _fake_download(field: str):
        def _inner(*, output_dir, corridor, year, months, time_step):
            calls.append((field, time_step, months))
            output_path = download_gcs._output_filename(
                Path(output_dir), field, corridor, year, months
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.touch()
            return output_path

        return _inner

    monkeypatch.setattr(
        download_gcs,
        "download_era5_wind_gcs",
        _fake_download("wind"),
    )
    monkeypatch.setattr(
        download_gcs,
        "download_era5_waves_gcs",
        _fake_download("waves"),
    )
    monkeypatch.setattr(
        _compare_era5_sources,
        "compare_datasets",
        lambda gcs_path, cds_path, field: [{"match": True}],
    )
    monkeypatch.setattr(_compare_era5_sources, "print_results", lambda results: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_era5_sources.py",
            "--cds-dir",
            str(cds_dir),
            "--gcs-dir",
            str(gcs_dir),
        ],
    )

    _compare_era5_sources.main()

    assert calls == [("wind", 1, [1]), ("waves", 1, [1])]
