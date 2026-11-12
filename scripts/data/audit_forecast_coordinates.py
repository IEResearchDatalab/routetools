#!/usr/bin/env python
"""Compare saved Atlantic fields with original message coordinates and values."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import acquire_weather as acquisition
import numpy as np
import requests
import xarray as xr


def coordinate_indices(source_lat, source_lon, latitude, longitude):
    """Match actual coordinates without assuming that longitude starts at zero."""
    rows = {float(v): i for i, v in enumerate(source_lat[:, 0])}
    columns = {float(v % 360): i for i, v in enumerate(source_lon[0, :])}
    if len(rows) != source_lat.shape[0] or len(columns) != source_lon.shape[1]:
        raise ValueError("nonunique source coordinates")
    jj = np.array([rows[float(v)] for v in latitude])
    ii = np.array([columns[float(v % 360)] for v in longitude])
    shape = (len(latitude), len(longitude))
    np.testing.assert_array_equal(
        source_lat[np.ix_(jj, ii)], np.broadcast_to(latitude[:, None], shape)
    )
    np.testing.assert_array_equal(
        source_lon[np.ix_(jj, ii)] % 360,
        np.broadcast_to((longitude % 360)[None, :], shape),
    )
    return jj, ii


def main():
    """Audit first/last control issues and the fixed January ERA5 target."""
    import eccodes

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("/home/SHARED/data/weather-routing")
    )
    args = parser.parse_args()
    root = args.root
    output = root / "diagnostics/fixed_target_20250130T0000Z"
    checks = []
    variables = list(acquisition.PARAMS["oper"].values()) + [
        acquisition.PARAMS["wave"]["swh"]
    ]
    for run, lead in [("20250115T0000Z", 360), ("20250130T0000Z", 0)]:
        for stream, param, name in [
            ("waef", "swh", variables[2]),
            ("enfo", "10u", variables[0]),
            ("enfo", "10v", variables[1]),
        ]:
            path = (
                root / "ifs/v1/atlantic/2025/01" / run / stream / f"lead_{lead:03d}.nc"
            )
            receipt = json.loads(path.with_suffix(".json").read_text())
            expected = dict(
                corridor="atlantic",
                initialization=datetime.strptime(run, "%Y%m%dT%H%MZ").isoformat(),
                lead=lead,
                stream=stream,
                ensemble=True,
                variables=list(acquisition.PARAMS[stream].values()),
            )
            if not acquisition.completed(path, expected):
                raise ValueError(f"unverified shard: {path}")
            provenance = receipt["provenance"]
            row = next(
                r
                for r in provenance["selected_index_rows"]
                if r["param"] == param and r["type"] == "cf"
            )
            lo = row["_offset"]
            hi = lo + row["_length"] - 1
            response = requests.get(
                provenance["grib_url"],
                headers={"Range": f"bytes={lo}-{hi}"},
                timeout=60,
            )
            response.raise_for_status()
            if response.status_code != 206 or len(response.content) != hi - lo + 1:
                raise ValueError("incorrect original-message byte range")
            message = eccodes.codes_new_from_message(response.content)
            try:
                for key in [
                    "iScansNegatively",
                    "jScansPositively",
                    "jPointsAreConsecutive",
                ]:
                    if eccodes.codes_get(message, key) != 0:
                        raise ValueError(f"unsupported scan flag {key}")
                shape = (
                    eccodes.codes_get(message, "Nj"),
                    eccodes.codes_get(message, "Ni"),
                )
                source_lat = eccodes.codes_get_array(message, "latitudes").reshape(
                    shape
                )
                source_lon = eccodes.codes_get_array(message, "longitudes").reshape(
                    shape
                )
                values = eccodes.codes_get_array(message, "values").reshape(shape)
                if eccodes.codes_get(message, "bitmapPresent"):
                    values = np.where(
                        eccodes.codes_get_array(message, "bitmap").reshape(shape),
                        values,
                        np.nan,
                    )
                with xr.open_dataset(path) as saved:
                    latitude = saved.latitude.values
                    longitude = saved.longitude.values
                    jj, ii = coordinate_indices(
                        source_lat, source_lon, latitude, longitude
                    )
                    original = values[np.ix_(jj, ii)]
                    cropped = saved[name].sel(number=0).isel(step=0).values
                    np.testing.assert_allclose(
                        cropped, original, atol=1e-6, rtol=1e-7, equal_nan=True
                    )
                    checks.append(
                        dict(
                            product="IFS",
                            run=run,
                            lead=lead,
                            variable=name,
                            grid_cells=int(original.size),
                            maximum_absolute_difference=float(
                                np.nanmax(np.abs(original - cropped))
                            ),
                            missing_masks_equal=bool(
                                np.array_equal(np.isnan(original), np.isnan(cropped))
                            ),
                            native_first_longitude=float(source_lon[0, 0]),
                            grib_url=provenance["grib_url"],
                            source_etag=response.headers.get("ETag"),
                            source_generation=response.headers.get("x-goog-generation"),
                            byte_range=[lo, hi],
                            message_sha256=hashlib.sha256(response.content).hexdigest(),
                        )
                    )
            finally:
                eccodes.codes_release(message)
    from routetools.era5.download_gcs import _normalize_time_dim, _open_era5_zarr

    with _open_era5_zarr(variables) as raw:
        selected = _normalize_time_dim(raw).sel(
            valid_time="2025-01-30T00:00",
            latitude=slice(float(latitude[0]), float(latitude[-1])),
        )
        # Two contiguous longitude slices avoid expensive global fancy indexing.
        original = xr.concat(
            [
                selected.sel(longitude=slice(280, 359.75)),
                selected.sel(longitude=slice(0, 10)),
            ],
            dim="longitude",
        ).load()
        for name in variables:
            field = "waves" if name == variables[2] else "wind"
            reference_path = root / (f"era5/v1/atlantic/2025/01/20250130_{field}.nc")
            reference_expected = dict(
                corridor="atlantic",
                day="2025-01-30",
                variables=list(
                    acquisition.PARAMS["wave" if field == "waves" else "oper"].values()
                ),
            )
            if not acquisition.completed(reference_path, reference_expected):
                raise ValueError(f"unverified ERA5 shard: {reference_path}")
            with xr.open_dataset(reference_path) as saved:
                np.testing.assert_array_equal(saved.latitude, latitude)
                np.testing.assert_array_equal(saved.longitude, longitude)
                np.testing.assert_array_equal(original.latitude, latitude)
                np.testing.assert_array_equal(original.longitude % 360, longitude % 360)
                cropped = saved[name].sel(valid_time="2025-01-30T00:00").values
                np.testing.assert_allclose(
                    cropped, original[name].values, atol=0, rtol=0, equal_nan=True
                )
                checks.append(
                    dict(
                        product="ERA5",
                        variable=name,
                        grid_cells=int(cropped.size),
                        maximum_absolute_difference=0,
                        missing_masks_equal=True,
                    )
                )
    report = dict(
        status="passed",
        checks=checks,
        scope=(
            "All Atlantic grid cells for first/last control wind and "
            "wave height, and target ERA5; not certification of every "
            "date/member."
        ),
        coordinate_shift_applied=False,
        audited_at=acquisition.utcnow(),
        script_sha256=acquisition.digest(Path(__file__)),
    )
    acquisition.atomic_json(output / "whole_grid_coordinate_audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
