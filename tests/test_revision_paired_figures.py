import csv

import numpy as np

from revision.task3_paired_improvements import (
    CONFIGURATIONS,
    plot_paired_improvements,
)


def test_paired_improvement_histograms_export_exact_bins(tmp_path):
    rows = []
    for configuration in CONFIGURATIONS:
        rows.extend(
            {
                "configuration": configuration,
                "improvement_vs_gc_pct": value,
            }
            for value in np.linspace(-10.0, 40.0, 61)
        )
    pdf_path = tmp_path / "paired.pdf"
    bins_path = tmp_path / "paired_bins.csv"

    plot_paired_improvements(rows, pdf_path, bins_path)

    assert pdf_path.stat().st_size > 0
    with bins_path.open(newline="") as handle:
        plotted = list(csv.DictReader(handle))
    assert len(plotted) == len(CONFIGURATIONS) * 30
    for configuration in CONFIGURATIONS:
        percentages = [
            float(row["departures_pct"])
            for row in plotted
            if row["configuration"] == configuration
        ]
        np.testing.assert_allclose(sum(percentages), 100.0)
