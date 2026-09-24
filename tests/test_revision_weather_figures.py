import csv

import numpy as np

from revision.task4_weather_violations import (
    _empirical_exceedance_curve,
    _weighted_histogram,
    plot_violation_exceedance_curves,
    plot_weather_distributions,
)


def test_weighted_histogram_uses_segment_duration():
    percentages = _weighted_histogram(
        values=np.array([0.5, 1.5]),
        durations_h=np.array([1.0, 3.0]),
        bins=np.array([0.0, 1.0, 2.0]),
    )

    np.testing.assert_allclose(percentages, [25.0, 75.0])


def test_empirical_exceedance_curve_uses_all_departures_as_denominator():
    x_values, y_values = _empirical_exceedance_curve(
        np.array([0.0, 0.0, 0.5, 1.0])
    )

    np.testing.assert_allclose(x_values, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(y_values, [50.0, 25.0, 0.0])


def test_weather_distribution_figure_exports_exact_plotted_bins(tmp_path):
    rows = []
    for corridor in ("atlantic", "pacific"):
        for strategy, wps in (("GC", "noWPS"), ("BERS", "noWPS"), ("BERS", "WPS")):
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
    csv_path = tmp_path / "weather_bins.csv"

    plot_weather_distributions(rows, pdf_path, csv_path)

    assert pdf_path.stat().st_size > 0
    with csv_path.open(newline="") as handle:
        plotted = list(csv.DictReader(handle))
    assert plotted
    grouped: dict[tuple[str, str, str, str], float] = {}
    for row in plotted:
        key = (row["corridor"], row["variable"], row["strategy"], row["wps"])
        grouped[key] = grouped.get(key, 0.0) + float(row["exposure_pct"])
    assert len(grouped) == 12
    np.testing.assert_allclose(list(grouped.values()), 100.0)
    assert {float(row["above_threshold_pct"]) for row in plotted} == {75.0}


def test_violation_curves_export_all_three_strategies(tmp_path):
    rows = []
    for corridor in ("atlantic", "pacific"):
        for wps in ("noWPS", "WPS"):
            for strategy, excesses in (
                ("GC", [0.0, 1.0, 2.0, 3.0]),
                ("CMA-ES", [0.0, 0.0, 0.5, 1.0]),
                ("BERS", [0.0, 0.0, 0.0, 0.1]),
            ):
                rows.extend(
                    {
                        "corridor": corridor,
                        "wps": wps,
                        "strategy": strategy,
                        "wind_max_exceedance": excess,
                        "wave_max_exceedance": excess / 2.0,
                    }
                    for excess in excesses
                )
    pdf_path = tmp_path / "violation_curves.pdf"
    csv_path = tmp_path / "violation_curve_points.csv"

    plot_violation_exceedance_curves(rows, pdf_path, csv_path)

    assert pdf_path.stat().st_size > 0
    with csv_path.open(newline="") as handle:
        plotted = list(csv.DictReader(handle))
    assert {row["strategy"] for row in plotted} == {"GC", "CMA-ES", "BERS"}
    assert {
        (row["corridor"], row["wps"], row["hazard"])
        for row in plotted
    } == {
        (corridor, wps, hazard)
        for corridor in ("atlantic", "pacific")
        for wps in ("noWPS", "WPS")
        for hazard in ("wind", "wave")
    }
