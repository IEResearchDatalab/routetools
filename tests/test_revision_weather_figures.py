import csv
from datetime import datetime

import numpy as np
import pytest

from revision.task4_weather_violations import (
    _segment_midpoint_stats,
    _weighted_density_curve,
    _weighted_histogram,
    plot_weather_distributions,
)
from routetools.violations import CorridorWeatherResources


def test_weighted_histogram_uses_segment_duration():
    percentages = _weighted_histogram(
        values=np.array([0.5, 1.5]),
        durations_h=np.array([1.0, 3.0]),
        bins=np.array([0.0, 1.0, 2.0]),
    )

    np.testing.assert_allclose(percentages, [25.0, 75.0])


def test_weighted_density_curve_is_smooth_and_integrates_to_100():
    x, density = _weighted_density_curve(
        values=np.array([0.5, 1.5]),
        durations_h=np.array([1.0, 3.0]),
        bins=np.linspace(0.0, 2.0, 81),
        bandwidth=0.1,
    )

    assert len(x) == 80
    assert np.count_nonzero(density > 0) > 20
    assert np.trapezoid(density, x) == pytest.approx(100.0)


def test_segment_midpoints_normalise_wrapped_pacific_longitudes():
    sampled_longitudes: list[np.ndarray] = []

    def field(lon, lat, time):
        sampled_longitudes.append(np.asarray(lon))
        return np.asarray(lon), np.zeros_like(np.asarray(lon))

    field.longitude_bounds = (100.0, 250.0)
    resources = CorridorWeatherResources(
        dataset_epoch=datetime(2024, 1, 1),
        windfield=field,
        wavefield=field,
    )
    curve = np.array([[179.0, 40.0], [-179.0, 41.0], [-170.0, 42.0]])
    times = [
        datetime(2024, 1, 1, 0, 0),
        datetime(2024, 1, 1, 1, 0),
        datetime(2024, 1, 1, 2, 0),
    ]

    tws, hs, durations = _segment_midpoint_stats(curve, times, resources)

    np.testing.assert_allclose(sampled_longitudes[0], [180.0, 185.5])
    np.testing.assert_allclose(tws, [180.0, 185.5])
    np.testing.assert_allclose(hs, [180.0, 185.5])
    np.testing.assert_allclose(durations, [1.0, 1.0])


def test_weather_distribution_figure_exports_exact_plotted_curves(tmp_path):
    rows = []
    for corridor in ("atlantic", "pacific"):
        for wps in ("noWPS", "WPS"):
            for strategy in ("GC", "CMA-ES", "BERS"):
                rows.extend(
                    [
                        {
                            "corridor": corridor,
                            "strategy": strategy,
                            "wps": wps,
                            "dt_hours": 1.0,
                            "tws_mps": 18.0,
                            "hs_m": 5.0,
                        },
                        {
                            "corridor": corridor,
                            "strategy": strategy,
                            "wps": wps,
                            "dt_hours": 3.0,
                            "tws_mps": 21.0,
                            "hs_m": 8.0,
                        },
                    ]
                )
    pdf_path = tmp_path / "weather.pdf"
    csv_path = tmp_path / "weather_curves.csv"

    plot_weather_distributions(rows, pdf_path, csv_path)

    assert pdf_path.stat().st_size > 0
    with csv_path.open(newline="") as handle:
        plotted = list(csv.DictReader(handle))
    assert plotted
    grouped: dict[tuple[str, str, str, str], list[tuple[float, float]]] = {}
    for row in plotted:
        key = (row["corridor"], row["variable"], row["strategy"], row["wps"])
        grouped.setdefault(key, []).append(
            (
                float(row["curve_x"]),
                float(row["exposure_density_pct_per_unit"]),
            )
        )
    assert len(grouped) == 24
    for points in grouped.values():
        points.sort()
        x, density = np.asarray(points).T
        assert np.trapezoid(density, x) == pytest.approx(100.0)
    assert {float(row["above_threshold_pct"]) for row in plotted} == {75.0}
