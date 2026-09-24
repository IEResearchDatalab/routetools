import csv

import numpy as np

from revision.task4_weather_violations import (
    _weighted_histogram,
    plot_weather_distributions,
)


def test_weighted_histogram_uses_segment_duration():
    percentages = _weighted_histogram(
        values=np.array([0.5, 1.5]),
        durations_h=np.array([1.0, 3.0]),
        bins=np.array([0.0, 1.0, 2.0]),
    )

    np.testing.assert_allclose(percentages, [25.0, 75.0])


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
