#!/usr/bin/env python
"""Acquire production wind/wave shards, with provenance and verified resume.

No pilot-specific paths or partial-member files: every forecast shard contains
one complete lead, with either a deterministic field or all 51 ensemble members.
The CLI supports 2025 cycle 49r1 and its December 2024/January 2026 boundaries.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import logging
import os
import subprocess
import tempfile
import time
from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import requests
import xarray as xr

SCRIPT_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
LOG = logging.getLogger("weather_acquisition")
BASE = "https://storage.googleapis.com/ecmwf-open-data"
ARCO = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
SCHEMA = "weather-routing-wind-waves-v1"
PARAMS = {
    "oper": {"10u": "10m_u_component_of_wind", "10v": "10m_v_component_of_wind"},
    "wave": {
        "swh": "significant_height_of_combined_wind_waves_and_swell",
        "mwd": "mean_wave_direction",
        "mwp": "mean_wave_period",
    },
}
PARAMS["enfo"] = PARAMS["oper"]
PARAMS["waef"] = PARAMS["wave"]
PARAMS["scda"] = PARAMS["oper"]
PARAMS["scwv"] = PARAMS["wave"]
SHORT = {"10u": "u10", "10v": "v10", "swh": "swh", "mwd": "mwd", "mwp": "mwp"}
BOXES = {"atlantic": (25.0, 60.0, -80.0, 10.0), "pacific": (15.0, 55.0, 120.0, 240.0)}


def utcnow():
    """Return an aware UTC acquisition timestamp."""
    return datetime.now(UTC).isoformat()


def digest(path):
    """Hash a file incrementally without loading it into memory."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    """Publish a JSON receipt through an atomic rename."""
    path = Path(path)
    part = path.with_name(path.name + f".{os.getpid()}.part")
    try:
        part.write_text(json.dumps(value, indent=2, default=str) + "\n")
        part.replace(path)
    finally:
        part.unlink(missing_ok=True)


@contextmanager
def locked(target):
    """Serialize writers across users; the advisory lock is released on crash."""
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = target.with_suffix(target.suffix + ".lock")
    with lock.open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield


def steps_for(hour):
    """Return the disseminated lead schedule for a forecast cycle."""
    return list(range(0, 145, 3)) + (
        list(range(150, 361, 6)) if hour in (0, 12) else []
    )


def grid(corridor):
    """Return the exact BERS latitude and longitude coordinates."""
    south, north, west, east = BOXES[corridor]
    return np.arange(north, south - 0.125, -0.25), np.arange(west, east + 0.125, 0.25)


def crop(ds, corridor):
    """Select the corridor grid without interpolating values."""
    lat, lon = grid(corridor)
    # Relabel coordinates, never interpolate values or average wave directions.
    if BOXES[corridor][2] < 0:
        ds = ds.assign_coords(longitude=(ds.longitude + 180) % 360 - 180)
    else:
        ds = ds.assign_coords(longitude=ds.longitude % 360)
    ds = ds.sortby("longitude").sortby("latitude", ascending=False)
    return ds.sel(latitude=lat, longitude=lon)


def validate(ds, expected):
    """Reject incomplete or mismatched data before publication or resume."""
    lat, lon = grid(expected["corridor"])
    np.testing.assert_array_equal(ds.latitude, lat)
    np.testing.assert_array_equal(ds.longitude, lon)
    if set(ds.data_vars) != set(expected["variables"]):
        raise ValueError("variable set differs from request")
    if "lead" in expected:
        np.testing.assert_array_equal(
            ds.step, np.array([expected["lead"]], dtype="timedelta64[h]")
        )
        init = np.datetime64(expected["initialization"])
        if ds.time.values != init:
            raise ValueError("wrong initialization")
        np.testing.assert_array_equal(ds.valid_time.values, init + ds.step.values)
        if expected["ensemble"]:
            np.testing.assert_array_equal(ds.number, np.arange(51))
        elif "number" in ds.dims:
            raise ValueError("unexpected member dimension")
    else:
        t = np.datetime64(expected["day"])
        np.testing.assert_array_equal(
            ds.valid_time, t + np.arange(24).astype("timedelta64[h]")
        )
    for name in expected["variables"]:
        v = ds[name]
        if v.dtype != np.float32:
            raise ValueError(f"{name}: expected float32")
        if not v.attrs.get("units"):
            raise ValueError(f"{name}: missing units")
        units = v.attrs["units"]
        allowed = {
            "10m_u_component_of_wind": {"m s**-1", "m s-1", "m/s"},
            "10m_v_component_of_wind": {"m s**-1", "m s-1", "m/s"},
            PARAMS["wave"]["swh"]: {"m"},
            PARAMS["wave"]["mwp"]: {"s"},
            PARAMS["wave"]["mwd"]: {"Degree true", "degrees", "degree", "degrees true"},
        }
        if units not in allowed[name]:
            raise ValueError(f"{name}: unexpected units {units}")
        if "lead" in expected:
            dims = (
                ("number", "step", "latitude", "longitude")
                if expected["ensemble"]
                else ("step", "latitude", "longitude")
            )
        else:
            dims = ("valid_time", "latitude", "longitude")
        if v.dims != dims:
            raise ValueError(f"{name}: dimensions {v.dims} != {dims}")
        a = v.values
        if np.isinf(a).any():
            raise ValueError(f"{name}: infinite values")
        spatial_axes = (-2, -1)
        if not np.all(np.isfinite(a).any(axis=spatial_axes)):
            raise ValueError(f"{name}: all-missing member/time slice")
        if name in (PARAMS["wave"]["swh"], PARAMS["wave"]["mwp"]) and np.nanmin(a) < 0:
            raise ValueError(f"{name}: negative height or period")
        # Archived packing can overshoot 360 by a few thousandths of a degree.
        # Preserve the source values; only derived circular calculations wrap.
        if name == PARAMS["wave"]["mwd"] and (
            np.nanmin(a) < -0.01 or np.nanmax(a) > 360.01
        ):
            raise ValueError("wave directions outside packing tolerance [-0.01,360.01]")


def completed(target, expected):
    """Require a receipt and full content audit before skipping a file."""
    receipt = target.with_suffix(".json")
    if not target.exists() or not receipt.exists():
        return False
    try:
        meta = json.loads(receipt.read_text())
        if meta.get("request") != expected or meta.get("status") != "complete":
            return False
        if meta["sha256"] != digest(target):
            return False
        with xr.open_dataset(target) as ds:
            validate(ds, expected)
        return True
    except (ValueError, OSError, KeyError, AssertionError):
        return False


def publish(ds, target, expected, provenance, started):
    """Validate and atomically publish a NetCDF shard and its receipt."""
    validate(ds, expected)
    ds.attrs.update(
        schema=SCHEMA,
        corridor=expected["corridor"],
        historical_available_at=(
            "unknown; do not equate initialization with availability"
        ),
    )
    if "lead" in expected:
        ds.attrs.update(
            model="IFS",
            cycle="49r1",
            stream=expected["stream"],
            source="ECMWF Open Data, Google Cloud mirror",
            license="CC BY 4.0; credit ECMWF",
            member_definition="0 control; 1..50 perturbed"
            if expected["ensemble"]
            else "deterministic/control",
        )
    else:
        ds.attrs.update(
            source="ERA5 via Google ARCO-ERA5",
            license="Copernicus licence; credit ECMWF, Copernicus and Hersbach "
            "et al. 2020",
        )
    part = target.with_name(target.name + f".{os.getpid()}.part")
    try:
        enc = {
            v: {"dtype": "float32", "zlib": True, "complevel": 4, "shuffle": True}
            for v in ds.data_vars
        }
        ds.to_netcdf(part, engine="netcdf4", format="NETCDF4", encoding=enc)
        with xr.open_dataset(part) as check:
            validate(check, expected)
        part.replace(target)
        try:
            revision = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip()
        except subprocess.CalledProcessError:
            revision = "unknown"
        meta = dict(
            status="complete",
            schema=SCHEMA,
            request=expected,
            bytes=target.stat().st_size,
            sha256=digest(target),
            acquired_at=utcnow(),
            elapsed_seconds=time.monotonic() - started,
            code_revision=revision,
            script_sha256=SCRIPT_SHA256,
            script_hash_capture="loaded startup snapshot",
            provenance=provenance,
            quality={
                v: dict(
                    minimum=float(np.nanmin(ds[v].values)),
                    maximum=float(np.nanmax(ds[v].values)),
                    finite_count=int(np.isfinite(ds[v].values).sum()),
                    total_count=int(ds[v].size),
                )
                for v in ds.data_vars
            },
        )
        atomic_json(target.with_suffix(".json"), meta)
        LOG.info(
            "COMPLETE %s %.2f MB %.1f s",
            target,
            meta["bytes"] / 1e6,
            meta["elapsed_seconds"],
        )
        return meta
    finally:
        part.unlink(missing_ok=True)


def get_index(url):
    """Read an inventory with bounded retries and version evidence."""
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=(15, 120))
            r.raise_for_status()
            rows = [json.loads(x) for x in r.text.splitlines() if x.strip()]
            return rows, dict(
                index_url=url,
                index_sha256=hashlib.sha256(r.content).hexdigest(),
                index_etag=r.headers.get("ETag"),
                index_generation=r.headers.get("x-goog-generation"),
            )
        except (requests.RequestException, ValueError):
            if attempt == 3:
                raise
            time.sleep(2**attempt)


def select_rows(rows, stream, run, lead):
    """Require exactly the requested parameters and member set."""
    ensemble = stream in ("enfo", "waef")
    selected = [
        r
        for r in rows
        if r.get("param") in PARAMS[stream]
        and r.get("type") in (("cf", "pf") if ensemble else ("fc",))
    ]
    keys = []
    for r in selected:
        if (
            r["date"] != run.strftime("%Y%m%d")
            or int(r["time"]) != run.hour * 100
            or int(r["step"]) != lead
            or r["stream"] != stream
        ):
            raise ValueError("source index does not match requested run/lead/stream")
        member = int(r.get("number", 0))
        if r["type"] == "cf" and member != 0:
            raise ValueError("control member must be zero")
        if r["type"] == "pf" and member not in range(1, 51):
            raise ValueError("unexpected perturbed member")
        keys.append((r["param"], member))
    wanted = {(p, m) for p in PARAMS[stream] for m in (range(51) if ensemble else [0])}
    if set(keys) != wanted or len(keys) != len(wanted):
        raise ValueError(
            f"incomplete/duplicate source inventory: {len(keys)} vs {len(wanted)}"
        )
    return sorted(selected, key=lambda r: r["_offset"])


def byte_ranges(rows):
    """Coalesce adjacent wanted messages without downloading unrelated fields."""
    ranges = []
    for r in rows:
        start, stop = int(r["_offset"]), int(r["_offset"]) + int(r["_length"]) - 1
        if ranges and start == ranges[-1][1] + 1:
            ranges[-1] = (ranges[-1][0], stop)
        else:
            ranges.append((start, stop))
    return ranges


def download_ranges(url, rows, destination, workers):
    """Fetch indexed ranges and verify response sizes and versions."""
    ranges = byte_ranges(rows)

    def fetch(pair):
        lo, hi = pair
        for attempt in range(4):
            try:
                r = requests.get(
                    url, headers={"Range": f"bytes={lo}-{hi}"}, timeout=(15, 120)
                )
                r.raise_for_status()
                if r.status_code != 206 or len(r.content) != hi - lo + 1:
                    raise ValueError(
                        "server ignored byte range or returned wrong length"
                    )
                if (
                    r.headers.get("Content-Range", "").split("/")[0]
                    != f"bytes {lo}-{hi}"
                ):
                    raise ValueError("wrong Content-Range")
                return (
                    r.content,
                    r.headers.get("ETag"),
                    r.headers.get("x-goog-generation"),
                )
            except (requests.RequestException, ValueError):
                if attempt == 3:
                    raise
                time.sleep(2**attempt)

    versions = set()
    with (
        concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool,
        destination.open("wb") as f,
    ):
        # At most one lead's selected compressed global fields is buffered.
        for data, etag, generation in pool.map(fetch, ranges):
            versions.add((etag, generation))
            f.write(data)
    if len(versions) != 1:
        raise ValueError("source object version changed during range retrieval")
    etag, generation = versions.pop()
    return dict(
        grib_url=url,
        source_etag=etag,
        source_generation=generation,
        selected_bytes=destination.stat().st_size,
        request_ranges=ranges,
        selected_grib_sha256=digest(destination),
        selected_index_rows=rows,
    )


def decode(path, stream, corridor, run, lead):
    """Check decoded GRIB metadata before shaping a cropped shard."""
    kinds = ("cf", "pf") if stream in ("enfo", "waef") else ("fc",)
    pieces = []
    for kind in kinds:
        with xr.open_dataset(
            path,
            engine="cfgrib",
            backend_kwargs={"indexpath": "", "filter_by_keys": {"dataType": kind}},
        ) as raw:
            validate_source_coordinates(raw, run, lead)
            if (
                kind == "cf"
                and "number" in raw.coords
                and np.any(raw.number.values != 0)
            ):
                raise ValueError("decoded control member is not zero")
            ds = crop(raw, corridor).load()
            ds = ds[list(SHORT[p] for p in PARAMS[stream])]
            if kind == "cf":
                if "number" in ds.coords and "number" not in ds.dims:
                    ds = ds.drop_vars("number")
                ds = ds.expand_dims(number=[0])
            if "step" not in ds.dims:
                ds = ds.expand_dims(step=[np.timedelta64(lead, "h")])
            pieces.append(ds)
    ds = xr.concat(pieces, "number").sortby("number") if len(pieces) == 2 else pieces[0]
    ds = ds.rename({SHORT[p]: n for p, n in PARAMS[stream].items()})
    # Scalar valid_time from cfgrib becomes a step coordinate for multifile reading.
    ds = ds.assign_coords(
        time=np.datetime64(run.isoformat()),
        valid_time=(
            "step",
            [np.datetime64(run.isoformat()) + np.timedelta64(lead, "h")],
        ),
    )
    dims = (
        ("number", "step", "latitude", "longitude")
        if len(pieces) == 2
        else ("step", "latitude", "longitude")
    )
    return ds.transpose(*dims)


def validate_source_coordinates(raw, run, lead):
    """Check GRIB-derived coordinates before reshaping scalar coordinates."""
    init = np.datetime64(run.isoformat())
    step = np.timedelta64(lead, "h")
    np.testing.assert_array_equal(np.atleast_1d(raw.time.values), np.array([init]))
    np.testing.assert_array_equal(np.atleast_1d(raw.step.values), np.array([step]))
    np.testing.assert_array_equal(
        np.atleast_1d(raw.valid_time.values), np.array([init + step])
    )


def ifs(args):
    """Acquire requested forecast shards or verify existing receipts."""
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    if start < date(2024, 12, 1) or end > date(2026, 1, 31):
        raise ValueError("supported: 2025 cycle49r1 plus Dec2024/Jan2026 boundaries")
    for day_offset in range((end - start).days + 1):
        day = start + timedelta(days=day_offset)
        for hour in args.hours:
            run = datetime.combine(day, datetime.min.time()).replace(hour=hour)
            leads = args.leads if args.leads is not None else steps_for(hour)
            if any(lead not in steps_for(hour) for lead in leads):
                raise ValueError("unsupported lead for this forecast cycle")
            for stream in args.streams:
                base_stream = (
                    {"oper": "scda", "wave": "scwv"}.get(stream, stream)
                    if hour in (6, 18)
                    else stream
                )
                folder = (
                    args.root
                    / "ifs/v1"
                    / args.corridor
                    / f"{day:%Y/%m}/{run:%Y%m%dT%H%MZ}"
                    / stream
                )
                for lead in leads:
                    target = folder / f"lead_{lead:03d}.nc"
                    expected = dict(
                        corridor=args.corridor,
                        initialization=run.isoformat(),
                        lead=lead,
                        stream=base_stream,
                        ensemble=stream in ("enfo", "waef"),
                        variables=list(PARAMS[stream].values()),
                    )
                    with locked(target):
                        if completed(target, expected):
                            LOG.info("VERIFIED SKIP %s", target)
                            continue
                        started = time.monotonic()
                        # Validate requested shards before reusing old runs.
                        legacy = (
                            args.root
                            / f"ecmwf-open-data/{day.year}/{args.corridor}"
                            / base_stream
                            / f"{day:%Y%m%d}_{hour:02d}z.nc"
                        )
                        if stream in ("oper", "wave") and legacy.exists():
                            with xr.open_dataset(legacy) as raw:
                                ds = raw.sel(step=[np.timedelta64(lead, "h")])[
                                    expected["variables"]
                                ].load()
                            provenance = dict(
                                method="validated local reuse",
                                source_path=str(legacy),
                                source_sha256=digest(legacy),
                                network_bytes=0,
                                source_attributes=dict(ds.attrs),
                            )
                            publish(ds, target, expected, provenance, started)
                            continue
                        typ = "ef" if expected["ensemble"] else "fc"
                        url = (
                            f"{BASE}/{day:%Y%m%d}/{hour:02d}z/ifs/0p25/"
                            f"{base_stream}/{run:%Y%m%d%H%M%S}-"
                            f"{lead}h-{base_stream}-{typ}"
                        )
                        rows, idx = get_index(url + ".index")
                        selected = select_rows(rows, base_stream, run, lead)
                        with tempfile.TemporaryDirectory(
                            prefix="grib_", dir=folder
                        ) as tmp:
                            rawpath = Path(tmp) / "selected.grib2"
                            provenance = download_ranges(
                                url + ".grib2", selected, rawpath, args.workers
                            )
                            provenance.update(
                                idx,
                                method="indexed HTTPS byte ranges",
                                retrieved_at=utcnow(),
                            )
                            ds = decode(rawpath, stream, args.corridor, run, lead)
                            publish(ds, target, expected, provenance, started)


def era5(args):
    """Acquire daily final ERA5 shards using validated local sources first."""
    from routetools.era5.download_gcs import _normalize_time_dim, _open_era5_zarr

    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    for i in range((end - start).days + 1):
        day = start + timedelta(days=i)
        for field, stream in (("wind", "oper"), ("waves", "wave")):
            target = (
                args.root
                / f"era5/v1/{args.corridor}/{day:%Y/%m}/{day:%Y%m%d}_{field}.nc"
            )
            expected = dict(
                corridor=args.corridor,
                day=day.isoformat(),
                variables=list(PARAMS[stream].values()),
            )
            with locked(target):
                if completed(target, expected):
                    LOG.info("VERIFIED SKIP %s", target)
                    continue
                started = time.monotonic()
                source = None
                for candidate in args.local_era5:
                    with xr.open_dataset(candidate) as raw:
                        if (
                            set(expected["variables"]).issubset(raw.data_vars)
                            and raw.valid_time.values[0]
                            <= np.datetime64(day.isoformat())
                            and raw.valid_time.values[-1]
                            >= np.datetime64(day.isoformat()) + np.timedelta64(23, "h")
                        ):
                            require_final_era5(raw, day)
                            ds = crop(
                                raw.sel(valid_time=str(day))[expected["variables"]],
                                args.corridor,
                            ).load()
                            source = Path(candidate)
                            break
                if source:
                    provenance = dict(
                        method="validated local reuse",
                        source_path=str(source),
                        source_sha256=digest(source),
                        network_bytes=0,
                        source_attributes=dict(ds.attrs),
                        original_distribution=ARCO,
                    )
                else:
                    with _open_era5_zarr(expected["variables"]) as raw:
                        require_final_era5(raw, day)
                        ds = crop(
                            _normalize_time_dim(raw).sel(valid_time=str(day)),
                            args.corridor,
                        ).load()
                        provenance = dict(
                            method="anonymous ARCO Zarr read",
                            source_uri=ARCO,
                            source_attributes=dict(raw.attrs),
                            network_bytes="not instrumented",
                        )
                # Do not carry a legacy experiment label into canonical data.
                ds.attrs = {
                    k: v
                    for k, v in ds.attrs.items()
                    if k not in ("experiment", "window_start", "window_end")
                }
                publish(ds, target, expected, provenance, started)


def require_final_era5(ds, day):
    """ARCO exposes a final ERA5 cutoff separately from preliminary ERA5T."""
    cutoff = ds.attrs.get("valid_time_stop")
    if cutoff is None or np.datetime64(str(cutoff)[:10]) < np.datetime64(
        day.isoformat()
    ):
        raise ValueError(
            f"cannot establish final ERA5 coverage for {day}; final cutoff={cutoff}"
        )


def main():
    """Run the command with the declared schema and validation gates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("product", choices=("ifs", "era5"))
    parser.add_argument(
        "--root", type=Path, default=Path("/home/SHARED/data/weather-routing")
    )
    parser.add_argument("--corridor", choices=BOXES, default="atlantic")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument(
        "--hours", type=int, nargs="+", choices=(0, 6, 12, 18), default=[0, 12]
    )
    parser.add_argument(
        "--streams",
        nargs="+",
        choices=("oper", "wave", "enfo", "waef"),
        default=["oper", "wave", "enfo", "waef"],
    )
    parser.add_argument(
        "--leads",
        nargs="+",
        type=int,
        help="explicit whole-lead shards; all members always required",
    )
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=4)
    parser.add_argument("--local-era5", nargs="*", default=[])
    args = parser.parse_args()
    if date.fromisoformat(args.end) < date.fromisoformat(args.start):
        parser.error("end precedes start")
    if args.leads is not None and sorted(set(args.leads)) != args.leads:
        parser.error("leads must be unique and increasing")
    os.umask(0o002)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    catalog = args.root / args.product / "v1" / "dataset.json"
    description = dict(
        schema=SCHEMA,
        format="NetCDF4, lossless deflate level 4, float32",
        product=args.product,
        boxes=BOXES,
        grid_degrees=0.25,
        granularity="one forecast initialization/stream/lead, all 51 members "
        "for ensembles"
        if args.product == "ifs"
        else "one UTC day/field group",
        variables={
            k: v for k, v in PARAMS.items() if k in ("oper", "wave", "enfo", "waef")
        },
        source=BASE if args.product == "ifs" else ARCO,
        direction_policy="preserve source values; allow 0.01 degree packing "
        "tolerance; wrap only circular derived calculations",
    )
    with locked(catalog):
        if catalog.exists() and json.loads(catalog.read_text()) != json.loads(
            json.dumps(description)
        ):
            raise ValueError("dataset schema differs from catalog; use a new version")
        atomic_json(catalog, description)
    (ifs if args.product == "ifs" else era5)(args)


if __name__ == "__main__":
    main()
