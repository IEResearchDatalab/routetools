"""Task 1: performance-model domain audit and clipped-input sensitivity.

Re-evaluates the saved GC, CMA-ES and BERS routes of the manuscript-number roots
with the exact File A evaluator conventions
(:func:`routetools.cost.evaluate_route_energy`): saved waypoints, uniform
passage schedule, segment midpoints, ERA5 closures from
:mod:`routetools.era5.loader`, and the analytical model in
:mod:`routetools.performance`.  Every segment input is recorded, compared with
the model's domain, and re-evaluated with TWS, Hs and speed clipped at the final
power-evaluation step only.  The clipped numbers are a sensitivity analysis, not
a corrected physical model.

Stages
------
``smoke``    one departure per strategy and case, reproduction check only.
``extract``  resumable, chunked segment extraction (one subprocess per corridor).
``analyze``  reproduction gate, domain audit, clipped-input tables and report.
``all``      extract (isolated per corridor) followed by analyze.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import os
import subprocess
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from routetools.performance import (
    A_W,
    K_A,
    K_H,
    K_S,
    K_W,
    SAIL_DEAD_ZONE_DEG,
    SAIL_QUADRATIC_CORRECTION,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "revision2"
WORK = OUT / "task1_work"
DEFAULT_CMAES_ROOT = ROOT / "output" / "sweep_combined"
DEFAULT_BERS_ROOT = ROOT / "output" / "sweep_combined_fms_strict"
ERA5_DIR = Path("/codabench/data")
ERA5_DEFAULTS = {
    "wind_atlantic": ERA5_DIR / "era5_wind_atlantic_2024.nc",
    "wave_atlantic": ERA5_DIR / "era5_waves_atlantic_2024.nc",
    "wind_pacific": ERA5_DIR / "era5_wind_pacific_2024.nc",
    "wave_pacific": ERA5_DIR / "era5_waves_pacific_2024.nc",
}
ERA5_SHA_CACHE = WORK / "era5_sha256_cache.json"
TEAM_PREFIX = "IEUniversity-1"
N_DEPARTURES = 366
DTFMT = "%Y-%m-%d %H:%M:%S"

DOMAIN: dict[str, tuple[float, float]] = {
    "tws": (0.0, 30.0),
    "hs": (0.0, 10.0),
    "speed": (0.0, 14.5),
    "twa": (0.0, 180.0),
    "mwa": (0.0, 180.0),
}
VARIABLES = tuple(DOMAIN)
RAW_COLUMN = {
    "tws": "tws_mps",
    "hs": "hs_m",
    "speed": "speed_mps",
    "twa": "twa_sym_deg",
    "mwa": "mwa_sym_deg",
}
STRATEGIES = ("GC", "CMA-ES", "BERS")
# case id -> (corridor label, GC case id)
CASES = {
    "AO_noWPS": ("Atlantic", "AGC_noWPS"),
    "AO_WPS": ("Atlantic", "AGC_WPS"),
    "PO_noWPS": ("Pacific", "PGC_noWPS"),
    "PO_WPS": ("Pacific", "PGC_WPS"),
}
# Documented analytical-model accuracy against the compiled model (kW per segment).
MODEL_MAX_ABS_ERROR_KW = 0.05
COMPONENTS = ("hull", "wind", "wave", "sail", "clamp", "total")
LAST_2024_WEATHER_TIME = pd.Timestamp("2024-12-31 23:00:00")


class ReproductionGateError(RuntimeError):
    """Raised when segment energies do not reproduce the File A energies."""


# ---------------------------------------------------------------------------
# Pure model helpers
# ---------------------------------------------------------------------------
def fold_twa(twa_deg: np.ndarray) -> np.ndarray:
    """Fold a true wind angle onto [0, 180] (the model is symmetric in sign)."""
    wrapped = np.mod(np.asarray(twa_deg, dtype=np.float64), 360.0)
    return np.where(wrapped > 180.0, 360.0 - wrapped, wrapped)


def center_mwa(mwa_deg: np.ndarray) -> np.ndarray:
    """Centre a wave angle on [-180, 180) exactly as the performance model does."""
    return np.mod(np.asarray(mwa_deg, dtype=np.float64) + 180.0, 360.0) - 180.0


def power_components(
    tws: np.ndarray,
    twa: np.ndarray,
    hs: np.ndarray,
    mwa: np.ndarray,
    speed: np.ndarray,
    *,
    wps: bool,
) -> dict[str, np.ndarray]:
    """Return unclamped power components (kW) following ``predict_power_batch``.

    ``raw = hull + wind + wave - sail`` and ``power = max(raw, 0)``.
    """
    tws, twa, hs, mwa, v = (
        np.asarray(x, dtype=np.float64) for x in (tws, twa, hs, mwa, speed)
    )
    twa_rad = np.radians(twa)
    mwa_rad = np.radians(center_mwa(mwa))
    ux = tws * np.cos(twa_rad) + v
    uy = tws * np.sin(twa_rad)
    vr2 = ux**2 + uy**2
    vr = np.sqrt(vr2)
    hull = K_H * v**3
    wind = K_A * v * (vr * ux - v**2)
    wave = A_W * hs**2 * v**1.5 * np.exp(-K_W * np.abs(mwa_rad) ** 3)
    if wps:
        awa_deg = np.degrees(np.arctan2(np.abs(uy), ux))
        alpha = np.radians(np.maximum(awa_deg - SAIL_DEAD_ZONE_DEG, 0.0))
        sin_a = np.sin(alpha)
        c_awa = K_S * sin_a * (1.0 + SAIL_QUADRATIC_CORRECTION * sin_a**2)
        sail = c_awa * vr2 * v
    else:
        sail = np.zeros_like(hull)
    raw = hull + wind + wave - sail
    return {
        "hull": hull,
        "wind": wind,
        "wave": wave,
        "sail": sail,
        "raw": raw,
        "power": np.maximum(raw, 0.0),
    }


def exceedance(values: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    """Distance outside the closed interval ``bounds`` (zero inside it)."""
    lo, hi = bounds
    values = np.asarray(values, dtype=np.float64)
    return np.maximum(values - hi, 0.0) + np.maximum(lo - values, 0.0)


def domain_flags(
    tws: np.ndarray,
    twa_sym: np.ndarray,
    hs: np.ndarray,
    mwa_sym: np.ndarray,
    speed: np.ndarray,
) -> dict[str, np.ndarray]:
    """Boolean out-of-domain flags; the boundary itself is inside the domain."""
    arrays = {"tws": tws, "hs": hs, "speed": speed, "twa": twa_sym, "mwa": mwa_sym}
    flags = {}
    for name, values in arrays.items():
        values = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Non-finite {name} input to the domain audit.")
        lo, hi = DOMAIN[name]
        flags[name] = (values < lo) | (values > hi)
    flags["any"] = np.logical_or.reduce([flags[name] for name in VARIABLES])
    return flags


def clip_inputs(
    tws: np.ndarray, hs: np.ndarray, speed: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clip TWS, Hs and speed to the performance-model domain (angles untouched)."""
    return (
        np.clip(tws, *DOMAIN["tws"]),
        np.clip(hs, *DOMAIN["hs"]),
        np.clip(speed, *DOMAIN["speed"]),
    )


def weighted_share(mask: np.ndarray, weights: np.ndarray) -> float:
    """Share of the total weight carried by ``mask`` (time- or energy-weighted)."""
    total = float(np.sum(weights))
    return float(np.sum(np.asarray(weights)[np.asarray(mask, dtype=bool)]) / total)


# ---------------------------------------------------------------------------
# Segment extraction (mirrors routetools.cost.evaluate_route_energy)
# ---------------------------------------------------------------------------
def segment_inputs(
    curve: Any,
    passage_hours: float,
    departure_offset_h: float,
    windfield: Callable,
    wavefield: Callable,
) -> dict[str, np.ndarray]:
    """Return per-segment performance-model inputs for one route."""
    import jax.numpy as jnp

    from routetools.cost import haversine_distance_from_curve, segment_bearings_deg

    n_points = curve.shape[0]
    if n_points < 2:
        raise ValueError(f"curve must have at least 2 points, got {n_points}")
    n_seg = n_points - 1
    mid_lon = np.asarray((curve[:-1, 0] + curve[1:, 0]) / 2, dtype=np.float64)
    mid_lat = np.asarray((curve[:-1, 1] + curve[1:, 1]) / 2, dtype=np.float64)
    t_hours = departure_offset_h + (np.arange(n_seg) + 0.5) / n_seg * passage_hours
    bearing = np.asarray(segment_bearings_deg(curve), dtype=np.float64)
    seg_dist_m = np.asarray(haversine_distance_from_curve(curve), dtype=np.float64)
    dt_h = passage_hours / n_seg
    speed = seg_dist_m / (dt_h * 3600.0)

    args = (jnp.array(mid_lon), jnp.array(mid_lat), jnp.array(t_hours))
    u10, v10 = (np.asarray(x, dtype=np.float64) for x in windfield(*args))
    tws = np.sqrt(u10**2 + v10**2)
    wind_from = (180.0 + np.degrees(np.arctan2(u10, v10))) % 360.0
    twa = (wind_from - bearing) % 360.0
    hs, mwd = (np.asarray(x, dtype=np.float64) for x in wavefield(*args))
    mwa = (mwd - bearing) % 360.0
    return {
        "mid_lon": mid_lon,
        "mid_lat": mid_lat,
        "t_hours": t_hours,
        "bearing": bearing,
        "dt_h": np.full(n_seg, dt_h),
        "tws": tws,
        "twa": twa,
        "hs": hs,
        "mwa": mwa,
        "speed": speed,
    }


def segment_frame(
    inputs: Mapping[str, np.ndarray],
    *,
    strategy: str,
    case: str,
    wps: bool,
    departure: datetime,
    route_file: str,
) -> pd.DataFrame:
    """Build the segment table for one route (unclipped and clipped)."""
    n = len(inputs["tws"])
    duration_h = np.asarray(inputs["dt_h"], dtype=np.float64)
    midpoint_h = (np.arange(n) + 0.5) * duration_h
    dep_naive = pd.Timestamp(departure.replace(tzinfo=None))
    mid_time = (dep_naive + pd.to_timedelta(midpoint_h, unit="h")).strftime(DTFMT)
    twa_sym = fold_twa(inputs["twa"])
    mwa_sym = np.abs(center_mwa(inputs["mwa"]))
    flags = domain_flags(inputs["tws"], twa_sym, inputs["hs"], mwa_sym, inputs["speed"])
    comp = power_components(
        inputs["tws"],
        inputs["twa"],
        inputs["hs"],
        inputs["mwa"],
        inputs["speed"],
        wps=wps,
    )
    tws_c, hs_c, v_c = clip_inputs(inputs["tws"], inputs["hs"], inputs["speed"])
    comp_c = power_components(tws_c, inputs["twa"], hs_c, inputs["mwa"], v_c, wps=wps)
    frame = {
        "strategy": strategy,
        "case": case,
        "wps": int(wps),
        "departure_utc": departure.strftime(DTFMT),
        "route_file": route_file,
        "segment_index": np.arange(n),
        "n_segments": n,
        "midpoint_time_utc": mid_time,
        "midpoint_offset_h": inputs["t_hours"],
        "duration_h": duration_h,
        "mid_lon_deg": inputs["mid_lon"],
        "mid_lat_deg": inputs["mid_lat"],
        "bearing_deg": inputs["bearing"],
        "tws_mps": inputs["tws"],
        "twa_deg": inputs["twa"],
        "twa_sym_deg": twa_sym,
        "hs_m": inputs["hs"],
        "mwa_deg": inputs["mwa"],
        "mwa_sym_deg": mwa_sym,
        "speed_mps": inputs["speed"],
    }
    for name in (*VARIABLES, "any"):
        frame[f"{name}_out"] = flags[name].astype(np.int8)
    for key in ("hull", "wind", "wave", "sail", "raw", "power"):
        frame[f"p_{key}_kw"] = comp[key]
    frame["energy_mwh"] = comp["power"] * duration_h / 1000.0
    frame["tws_clip_mps"] = tws_c
    frame["hs_clip_m"] = hs_c
    frame["speed_clip_mps"] = v_c
    for key in ("hull", "wind", "wave", "sail", "raw", "power"):
        frame[f"c_p_{key}_kw"] = comp_c[key]
    frame["c_energy_mwh"] = comp_c["power"] * duration_h / 1000.0
    return pd.DataFrame(frame)


# ---------------------------------------------------------------------------
# Inventory, pairing and checkpointed extraction
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Unit:
    """One strategy/case route set with its summary and track location."""

    strategy: str
    case: str
    root: Path
    file_case: str

    @property
    def summary_path(self) -> Path:
        """Return the File A summary path."""
        return self.root / f"{TEAM_PREFIX}-{self.file_case}.csv"

    @property
    def key(self) -> str:
        """Return a filesystem-safe unit key."""
        return f"{self.strategy}__{self.case}"


def build_units(cmaes_root: Path, bers_root: Path) -> list[Unit]:
    """Return the 12 strategy/case units; GC comes from the CMA-ES root."""
    units = []
    for case, (_corridor, gc_case) in CASES.items():
        units.append(Unit("GC", case, cmaes_root, gc_case))
        units.append(Unit("CMA-ES", case, cmaes_root, case))
        units.append(Unit("BERS", case, bers_root, case))
    return units


def verify_gc_identical(cmaes_root: Path, bers_root: Path) -> None:
    """Require byte-identical GC summaries in both roots (Task 0 finding)."""
    for _case, (_corridor, gc_case) in CASES.items():
        name = f"{TEAM_PREFIX}-{gc_case}.csv"
        if sha256_file(cmaes_root / name) != sha256_file(bers_root / name):
            raise ValueError(f"GC summary {name} differs between the two roots.")


def corridor_of(case: str) -> str:
    """Return the lower-case corridor key (atlantic/pacific) of a case."""
    return CASES[case][0].lower()


def expected_departures() -> list[datetime]:
    """Return the 366 daily 12:00 UTC departures of 2024."""
    start = datetime(2024, 1, 1, 12, tzinfo=UTC)
    return [start + timedelta(days=d) for d in range(N_DEPARTURES)]


def read_file_a(path: Path, expected: Sequence[datetime] | None = None) -> pd.DataFrame:
    """Read a File A summary keyed by departure, failing loudly on any anomaly."""
    if not path.is_file():
        raise FileNotFoundError(f"Required route summary is missing: {path}")
    frame = pd.read_csv(path, dtype={"details_filename": str})
    needed = {"departure_time_utc", "energy_cons_mwh", "details_filename"}
    if needed - set(frame.columns):
        raise ValueError(f"{path} lacks columns {sorted(needed - set(frame.columns))}.")
    expected = list(expected_departures() if expected is None else expected)
    if frame["departure_time_utc"].duplicated().any():
        raise ValueError(f"{path} has duplicate departure dates.")
    if len(frame) != len(expected):
        raise ValueError(f"{path} has {len(frame)} rows; expected {len(expected)}.")
    want = {d.strftime(DTFMT) for d in expected}
    if set(frame["departure_time_utc"]) != want:
        raise ValueError(f"{path} departures differ from the expected schedule.")
    frame = frame.sort_values("departure_time_utc").reset_index(drop=True)
    frame["departure"] = [
        datetime.strptime(s, DTFMT).replace(tzinfo=UTC)
        for s in frame["departure_time_utc"]
    ]
    return frame


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def chunk_slices(n: int, size: int) -> list[tuple[int, int]]:
    """Return half-open ``(start, stop)`` slices covering ``range(n)``."""
    if size <= 0:
        raise ValueError("chunk size must be positive")
    return [(i, min(i + size, n)) for i in range(0, n, size)]


def _write_part(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False, compression={"method": "gzip", "mtime": 0})
    os.replace(tmp, path)


class CheckpointMismatchError(RuntimeError):
    """A checkpoint sidecar disagrees with the current inputs, code or options."""


SAMPLING_FILE_A = "file_a_scoring_uniform_waypoint_midpoints"
CODE_FILES = (
    "revision2/task1_out_of_domain_audit.py",
    "routetools/performance.py",
    "routetools/cost.py",
    "routetools/violations.py",
    "routetools/era5/loader.py",
    "routetools/swopp3.py",
)


def code_fingerprint(root: Path = ROOT) -> dict[str, Any]:
    """Return per-file and combined SHA-256 of the code defining the extraction."""
    files = {name: sha256_file(root / name) for name in CODE_FILES}
    combined = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {"files": files, "combined_sha256": combined}


def era5_file_info(path: Path, sha_cache: Path | None = None) -> dict[str, Any]:
    """Describe one ERA5 NetCDF file (size, time axis, dims, variables, SHA-256)."""
    from routetools.era5.loader import _load_dataset

    if not path.is_file():
        raise FileNotFoundError(f"ERA5 file is missing: {path}")
    stat = path.stat()
    key = f"{path}|{stat.st_size}|{stat.st_mtime_ns}"
    cache: dict[str, str] = {}
    if sha_cache is not None and sha_cache.is_file():
        cache = json.loads(sha_cache.read_text(encoding="utf-8"))
    if key not in cache:
        cache[key] = sha256_file(path)
        if sha_cache is not None:
            sha_cache.parent.mkdir(parents=True, exist_ok=True)
            sha_cache.write_text(json.dumps(cache, indent=1), encoding="utf-8")
    with _load_dataset(path) as ds:
        name = "valid_time" if "valid_time" in ds.coords else "time"
        times = pd.DatetimeIndex(ds[name].values)
        steps = sorted({float(x) for x in np.diff(times.asi8) / 3.6e12})
        return {
            "path": str(path),
            "bytes": stat.st_size,
            "mtime_utc": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
            "sha256": cache[key],
            "time_name": name,
            "time_start": times[0].isoformat(),
            "time_end": times[-1].isoformat(),
            "n_times": len(times),
            "step_hours": steps,
            "dims": {k: int(v) for k, v in ds.sizes.items()},
            "variables": sorted(ds.data_vars),
        }


def resolve_era5_inputs(
    era5: Mapping[str, Path], sha_cache: Path | None = None
) -> dict[str, dict[str, Any]]:
    """Resolve each base ERA5 file with the loader's next-year continuations."""
    from routetools.era5.loader import loadable_era5_paths

    out = {}
    for key, base in era5.items():
        out[key] = {
            "base": str(base),
            "resolved": [
                era5_file_info(p, sha_cache) for p in loadable_era5_paths(base)
            ],
        }
    return out


def check_era5_continuity(
    info: Mapping[str, Mapping[str, Any]], last_needed: datetime | None = None
) -> None:
    """Require hourly, gap-free, ordered time axes across resolved files."""
    for key, entry in info.items():
        files = entry["resolved"]
        for f in files:
            if f["step_hours"] != [1.0]:
                raise ValueError(
                    f"{f['path']} is not strictly hourly: {f['step_hours']}."
                )
        for prev, nxt in itertools.pairwise(files):
            gap = pd.Timestamp(nxt["time_start"]) - pd.Timestamp(prev["time_end"])
            if gap != pd.Timedelta(hours=1):
                raise ValueError(
                    f"{key}: {nxt['path']} starts {gap} after {prev['path']} ends."
                )
        if last_needed is not None:
            end = pd.Timestamp(files[-1]["time_end"])
            if end < pd.Timestamp(last_needed.replace(tzinfo=None)):
                raise ValueError(
                    f"{key}: ERA5 ends {end}, before the last needed {last_needed}."
                )


def era5_fingerprint(info: Mapping[str, Mapping[str, Any]]) -> str:
    """Return one stable SHA-256 over all resolved ERA5 files (paths and content)."""
    items = {
        key: [(Path(f["path"]).name, f["bytes"], f["sha256"]) for f in e["resolved"]]
        for key, e in info.items()
    }
    return hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest()


def select_rows(
    summary: pd.DataFrame,
    max_departures: int | None = None,
    departures: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return the requested subset of departures, failing on unknown dates."""
    if departures is not None:
        unknown = set(departures) - set(summary["departure_time_utc"])
        if unknown:
            raise ValueError(f"Unknown departures requested: {sorted(unknown)}")
        summary = summary[summary["departure_time_utc"].isin(departures)]
    if max_departures is not None:
        summary = summary.iloc[:max_departures]
    return summary.reset_index(drop=True)


def track_fingerprint(unit: Unit, rows: pd.DataFrame) -> str:
    """Hash the contents of the track files of a chunk."""
    digest = hashlib.sha256()
    for name in rows["details_filename"]:
        digest.update(name.encode())
        digest.update(sha256_file(unit.root / "tracks" / name).encode())
    return digest.hexdigest()


def _filename_fingerprint(unit: Unit, rows: pd.DataFrame) -> str:
    return hashlib.sha256("|".join(rows["details_filename"]).encode()).hexdigest()


def part_expectation(
    unit: Unit,
    rows: pd.DataFrame,
    start: int,
    stop: int,
    context: Mapping[str, Any],
    track_sha: str,
) -> dict[str, Any]:
    """Return the sidecar fields a valid checkpoint part must carry."""
    return {
        "unit": unit.key,
        "strategy": unit.strategy,
        "case": unit.case,
        "route_root": str(unit.root),
        "start": start,
        "stop": stop,
        "n_routes": stop - start,
        "departures_sha256": hashlib.sha256(
            "|".join(rows["departure_time_utc"]).encode()
        ).hexdigest(),
        "first_departure": rows["departure_time_utc"].iloc[0],
        "last_departure": rows["departure_time_utc"].iloc[-1],
        "summary_sha256": sha256_file(unit.summary_path),
        "track_sha256": track_sha,
        **context,
    }


def part_state(part: Path, meta: Path, expected: Mapping[str, Any]) -> str:
    """Classify a checkpoint: ``missing``, ``corrupt``, ``stale`` or ``valid``."""
    if not (part.is_file() and meta.is_file()):
        return "missing"
    try:
        info = json.loads(meta.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "corrupt"
    if info.get("part_sha256") != sha256_file(part):
        return "corrupt"
    if any(info.get(k) != v for k, v in expected.items()):
        return "stale"
    return "valid"


def iter_expected_parts(
    units: Sequence[Unit],
    *,
    chunk_size: int,
    max_departures: int | None = None,
    departures: Sequence[str] | None = None,
    context: Mapping[str, Any] | None = None,
    track_fp: Callable[[Unit, pd.DataFrame], str] = _filename_fingerprint,
) -> Iterator[tuple[Unit, str, pd.DataFrame, dict[str, Any]]]:
    """Yield ``(unit, stem, rows, expected_sidecar)`` for every expected chunk."""
    ctx = {
        **(context or {}),
        "chunk_size": chunk_size,
        "max_departures": max_departures,
        "departure_selection": list(departures) if departures is not None else None,
    }
    for unit in units:
        summary = select_rows(
            read_file_a(unit.summary_path), max_departures, departures
        )
        for start, stop in chunk_slices(len(summary), chunk_size):
            rows = summary.iloc[start:stop]
            stem = f"{unit.key}__{start:03d}-{stop:03d}"
            yield (
                unit,
                stem,
                rows,
                part_expectation(unit, rows, start, stop, ctx, track_fp(unit, rows)),
            )


def verify_parts(parts_dir: Path, **kwargs: Any) -> int:
    """Require exactly the expected, matching parts; return their number."""
    expected_stems = set()
    for _unit, stem, _rows, expected in iter_expected_parts(**kwargs):
        expected_stems.add(stem)
        state = part_state(
            parts_dir / f"{stem}.csv.gz", parts_dir / f"{stem}.json", expected
        )
        if state != "valid":
            raise CheckpointMismatchError(f"Checkpoint {stem} is {state}.")
    extra = {p.name.removesuffix(".csv.gz") for p in parts_dir.glob("*.csv.gz")}
    if extra - expected_stems:
        raise CheckpointMismatchError(
            f"Unexpected checkpoint parts: {sorted(extra - expected_stems)[:3]}"
        )
    return len(expected_stems)


def extract_units(
    units: Sequence[Unit],
    parts_dir: Path,
    compute: Callable[[Unit, pd.DataFrame], pd.DataFrame],
    *,
    chunk_size: int = 61,
    max_departures: int | None = None,
    departures: Sequence[str] | None = None,
    context: Mapping[str, Any] | None = None,
    track_fp: Callable[[Unit, pd.DataFrame], str] = _filename_fingerprint,
    log: Callable[[str], None] = print,
) -> dict[str, int]:
    """Compute missing chunks only; reject checkpoints bound to other inputs."""
    parts_dir.mkdir(parents=True, exist_ok=True)
    counts = {"computed": 0, "reused": 0}
    expected_parts = list(
        iter_expected_parts(
            units,
            chunk_size=chunk_size,
            max_departures=max_departures,
            departures=departures,
            context=context,
            track_fp=track_fp,
        )
    )
    prefixes = tuple(f"{u.key}__" for u in units)
    present = {
        p.name.removesuffix(".csv.gz")
        for p in parts_dir.glob("*.csv.gz")
        if p.name.startswith(prefixes)
    }
    unexpected = present - {stem for _u, stem, _r, _e in expected_parts}
    if unexpected:
        raise CheckpointMismatchError(
            f"Parts {sorted(unexpected)[:3]} do not belong to these options "
            "(different chunking, selection or fingerprints); use a fresh "
            "--parts-dir."
        )
    for unit, stem, rows, expected in expected_parts:
        part = parts_dir / f"{stem}.csv.gz"
        meta = parts_dir / f"{stem}.json"
        state = part_state(part, meta, expected)
        if state == "valid":
            counts["reused"] += 1
            continue
        if state == "stale":
            raise CheckpointMismatchError(
                f"Checkpoint {stem} was made for different route, weather, code "
                "or option fingerprints; use a fresh --parts-dir."
            )
        frame = compute(unit, rows)
        _write_part(frame, part)
        meta.write_text(
            json.dumps(
                {
                    **expected,
                    "n_rows": len(frame),
                    "part_sha256": sha256_file(part),
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        counts["computed"] += 1
        log(f"  computed {stem} ({len(frame)} segments)")
    return counts


def read_track_times(path: Path) -> list[datetime]:
    """Read the ``time_utc`` column of a track as naive datetimes."""
    with path.open(newline="") as handle:
        return [datetime.strptime(r["time_utc"], DTFMT) for r in csv.DictReader(handle)]


def check_track_schedule(
    times: Sequence[datetime], departure: datetime, passage_hours: float
) -> None:
    """Require the uniform passage schedule assumed by the File A evaluator."""
    dep = departure.replace(tzinfo=None)
    n_seg = len(times) - 1
    if n_seg < 1 or times[0] != dep:
        raise ValueError(f"Track starts at {times[0] if times else None}, not {dep}.")
    expected_end = dep + timedelta(hours=passage_hours)
    if abs((times[-1] - expected_end).total_seconds()) > 1.0:
        raise ValueError(f"Track ends at {times[-1]}, expected {expected_end}.")
    step = passage_hours * 3600.0 / n_seg
    for k, stamp in enumerate(times):
        if abs((stamp - dep).total_seconds() - k * step) > 1.0:
            raise ValueError(f"Non-uniform track schedule at waypoint {k}.")


SAMPLING_OPT30 = "optimization_grid_proxy_30min_linear_interpolation_of_saved_waypoints"
OPT_GRID_MINUTES = 30.0


def optimization_grid_segments(passage_hours: float) -> int:
    """Return the CMA-ES objective's segment count (``num_pieces=1`` default)."""
    from routetools.cmaes import _compute_l_eval

    l_eval = _compute_l_eval(
        L=0, dt_eval_minutes=OPT_GRID_MINUTES, travel_time=passage_hours, num_pieces=1
    )
    return l_eval - 1


def resample_uniform(curve: Any, n_seg: int) -> Any:
    """Linearly resample a uniformly timed polyline to ``n_seg`` equal-time steps."""
    import jax.numpy as jnp

    arr = np.asarray(curve, dtype=np.float64)
    pos = np.linspace(0.0, len(arr) - 1, n_seg + 1)
    lon = np.interp(pos, np.arange(len(arr)), arr[:, 0])
    lat = np.interp(pos, np.arange(len(arr)), arr[:, 1])
    return jnp.asarray(np.stack([lon, lat], axis=1), dtype=jnp.float32)


def make_route_computer(
    resources: Mapping[str, Any], sampling: str = "file_a"
) -> Callable:
    """Return a chunk computer reading real tracks and the ERA5 closures."""
    from routetools.swopp3 import SWOPP3_CASES
    from routetools.violations import (
        departure_offset_hours,
        normalise_route_longitudes,
        read_track_curve,
        weather_longitude_bounds,
    )

    def compute(unit: Unit, rows: pd.DataFrame) -> pd.DataFrame:
        spec = SWOPP3_CASES[unit.case]
        res = resources[str(spec["route"])]
        bounds = weather_longitude_bounds(res)
        frames = []
        for _, row in rows.iterrows():
            track = unit.root / "tracks" / row["details_filename"]
            curve = normalise_route_longitudes(read_track_curve(track), bounds)
            passage = float(spec["passage_hours"])
            check_track_schedule(read_track_times(track), row["departure"], passage)
            if sampling == "opt30":
                curve = resample_uniform(curve, optimization_grid_segments(passage))
            offset = departure_offset_hours(row["departure"], res.dataset_epoch)
            inputs = segment_inputs(
                curve, passage, offset, res.windfield, res.wavefield
            )
            frames.append(
                segment_frame(
                    inputs,
                    strategy=unit.strategy,
                    case=unit.case,
                    wps=bool(spec["wps"]),
                    departure=row["departure"],
                    route_file=row["details_filename"],
                )
            )
        return pd.concat(frames, ignore_index=True)

    return compute


def load_parts(parts_dir: Path) -> pd.DataFrame:
    """Concatenate all checkpoint parts in deterministic order."""
    files = sorted(parts_dir.glob("*.csv.gz"))
    if not files:
        raise FileNotFoundError(f"No checkpoint parts found in {parts_dir}.")
    for f in files:
        side = f.with_name(f.name.removesuffix(".csv.gz") + ".json")
        if not side.is_file() or (
            json.loads(side.read_text(encoding="utf-8")).get("part_sha256")
            != sha256_file(f)
        ):
            raise ValueError(f"Checkpoint part {f.name} fails its sidecar checksum.")
    return pd.concat((pd.read_csv(f) for f in files), ignore_index=True)


def combine_parts(parts_dir: Path, out_path: Path) -> dict[str, Any]:
    """Stream the parts into a single gzip CSV; return row count, size, checksum."""
    files = sorted(parts_dir.glob("*.csv.gz"))
    rows = 0
    tmp = out_path.with_name(out_path.name + ".tmp")
    with tmp.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
        for i, part in enumerate(files):
            with gzip.open(part, "rb") as stream:
                header = stream.readline()
                if i == 0:
                    gz.write(header)
                for line in stream:
                    gz.write(line)
                    rows += 1
    os.replace(tmp, out_path)
    return {
        "rows": rows,
        "bytes": out_path.stat().st_size,
        "sha256": sha256_file(out_path),
        "parts": len(files),
    }


# ---------------------------------------------------------------------------
# Reproduction gate
# ---------------------------------------------------------------------------
def reproduction_table(segments: pd.DataFrame, file_a: pd.DataFrame) -> pd.DataFrame:
    """Compare a common-weather re-score with legacy File A for every route."""
    from routetools.swopp3 import SWOPP3_CASES

    keys = ["strategy", "case", "departure_utc", "route_file"]
    model = (
        segments.assign(
            _midpoint=pd.to_datetime(segments["midpoint_time_utc"], errors="raise")
        )
        .groupby(keys)
        .agg(
            common_weather_mwh=("energy_mwh", "sum"),
            final_required_evaluation_time_utc=("_midpoint", "max"),
        )
        .reset_index()
    )
    model["requires_2025_weather"] = (
        model["final_required_evaluation_time_utc"] > LAST_2024_WEATHER_TIME
    )
    model["final_required_evaluation_time_utc"] = model[
        "final_required_evaluation_time_utc"
    ].dt.strftime(DTFMT)
    merged = model.merge(
        file_a, on=["strategy", "case", "departure_utc"], how="outer", indicator=True
    )
    if (merged["_merge"] == "left_only").any():
        raise ValueError("Segment routes without a File A row.")
    present = set(zip(model["strategy"], model["case"], strict=True))
    missing = {(s, c) for s in STRATEGIES for c in CASES} - present
    if missing:
        raise ValueError(f"Missing strategy/case combinations: {sorted(missing)}")
    right_only = merged[merged["_merge"] == "right_only"]
    if len(right_only):
        raise ValueError(
            f"{len(right_only)} File A routes have no extracted segments, e.g. "
            f"{right_only[['strategy', 'case', 'departure_utc']].iloc[0].tolist()}."
        )
    merged = merged[merged["_merge"] == "both"].drop(columns="_merge")
    if len(merged) != len(model):
        raise ValueError("Route pairing lost rows.")
    if (merged["route_file"] != merged["details_filename"]).any():
        raise ValueError("Route file does not match the File A details filename.")
    merged = merged.rename(columns={"energy_cons_mwh": "legacy_file_a_mwh"})
    merged["weather_consistency_delta_mwh"] = (
        merged["common_weather_mwh"] - merged["legacy_file_a_mwh"]
    )
    merged["abs_diff_mwh"] = merged["weather_consistency_delta_mwh"].abs()
    merged["rel_diff"] = merged["abs_diff_mwh"] / merged["legacy_file_a_mwh"].abs()
    hours = merged["case"].map(lambda c: float(SWOPP3_CASES[c]["passage_hours"]))
    merged["tolerance_mwh"] = MODEL_MAX_ABS_ERROR_KW * hours / 1000.0
    merged["within_tolerance"] = merged["abs_diff_mwh"] <= merged["tolerance_mwh"]
    expected_mismatch = (
        merged["strategy"].eq("BERS")
        & merged["requires_2025_weather"]
        & ~merged["within_tolerance"]
    )
    unexpected_failure = ~merged["within_tolerance"] & ~expected_mismatch
    merged["gate_classification"] = np.select(
        [unexpected_failure, expected_mismatch],
        ["unexpected_failure", "expected_bers_weather_mismatch"],
        default="reproduced_legacy_file_a",
    )
    cols = [
        "strategy", "case", "departure_utc", "route_file",
        "final_required_evaluation_time_utc", "requires_2025_weather",
        "legacy_file_a_mwh", "common_weather_mwh",
        "weather_consistency_delta_mwh", "abs_diff_mwh", "rel_diff",
        "tolerance_mwh", "within_tolerance", "gate_classification",
    ]  # fmt: skip
    return merged[cols].reset_index(drop=True)


def failure_diagnostics(table: pd.DataFrame) -> str:
    """Describe where gate failures occur (descriptive only; nothing is tuned)."""
    if "requires_2025_weather" in table:
        beyond = table["requires_2025_weather"].astype(bool)
    else:
        from routetools.swopp3 import SWOPP3_CASES

        dep = pd.to_datetime(table["departure_utc"])
        hours = table["case"].map(lambda c: float(SWOPP3_CASES[c]["passage_hours"]))
        beyond = dep + pd.to_timedelta(hours, unit="h") >= pd.Timestamp("2025-01-01")
    bad = ~table["within_tolerance"]
    lines = [
        "Diagnostics (descriptive; no remedy applied):",
        "",
        "| strategy | case | failing routes | first failing departure | "
        "routes requiring 2025 weather | failing among those |",
        "|---|---|---|---|---|---|",
    ]
    for (st, case), g in table.groupby(["strategy", "case"], sort=False):
        b = bad[g.index]
        first = g.loc[b, "departure_utc"].min() if b.any() else "-"
        lines.append(
            f"| {st} | {case} | {int(b.sum())} | {first} | "
            f"{int(beyond[g.index].sum())} | {int((b & beyond[g.index]).sum())} |"
        )
    lines += [
        "",
        f"- Failing routes not requiring 2025 weather: {int((bad & ~beyond).sum())}.",
        f"- Largest discrepancy among routes not requiring 2025 weather: "
        f"{table.loc[~beyond, 'abs_diff_mwh'].max():.3e} MWh.",
        "- With the continuous 2024 plus January-2025 ERA5 inputs, preserved "
        "GC/CMA-ES File A values reproduce, whereas affected BERS File A values "
        "match the separately preserved 2024-only/clamped evaluation.",
    ]
    return "\n".join(lines)


def reproduction_stats(table: pd.DataFrame) -> dict[str, Any]:
    """Summarise the reproduction discrepancy."""
    worst_abs = table.loc[table["abs_diff_mwh"].idxmax()]
    worst_rel = table.loc[table["rel_diff"].idxmax()]
    return {
        "n_routes": len(table),
        "max_abs_mwh": float(table["abs_diff_mwh"].max()),
        "mean_abs_mwh": float(table["abs_diff_mwh"].mean()),
        "max_rel": float(table["rel_diff"].max()),
        "max_abs_route": f"{worst_abs['strategy']} {worst_abs['case']} "
        f"{worst_abs['departure_utc']}",
        "max_rel_route": f"{worst_rel['strategy']} {worst_rel['case']} "
        f"{worst_rel['departure_utc']}",
        "n_outside_tolerance": int((~table["within_tolerance"]).sum()),
        "n_expected_bers_weather_mismatch": int(
            table.get("gate_classification", pd.Series(dtype=str))
            .eq("expected_bers_weather_mismatch")
            .sum()
        ),
        "n_unexpected_failure": int(
            table.get("gate_classification", pd.Series(dtype=str))
            .eq("unexpected_failure")
            .sum()
        ),
    }


def check_reproduction_gate(table: pd.DataFrame) -> dict[str, Any]:
    """Allow only the pre-specified late-BERS legacy-weather mismatches."""
    stats = reproduction_stats(table)
    if "gate_classification" not in table:
        failures = ~table["within_tolerance"]
    else:
        failures = table["gate_classification"].eq("unexpected_failure")
    if failures.any():
        worst = table.loc[failures].sort_values("abs_diff_mwh").iloc[-1]
        raise ReproductionGateError(
            f"{int(failures.sum())} routes violate the conditional gate; maximum "
            f"unexpected |diff| = {worst['abs_diff_mwh']:.6f} MWh at "
            f"{worst['strategy']} {worst['case']} {worst['departure_utc']}."
        )
    return stats


def common_horizon_departures(table: pd.DataFrame) -> dict[str, set[str]]:
    """Departures for which no strategy requires post-2024 weather."""
    required = table.groupby(["case", "departure_utc"])["requires_2025_weather"].max()
    return {
        case: set(
            required.xs(case, level="case")
            .loc[lambda values: ~values.astype(bool)]
            .index
        )
        for case in CASES
    }


def route_energy_series(table: pd.DataFrame, column: str) -> pd.Series:
    """Return one route-energy variant indexed for Table 11/12 aggregation."""
    return table.set_index(["case", "strategy", "departure_utc"])[column].sort_index()


def weather_consistency_tables(table: pd.DataFrame) -> pd.DataFrame:
    """Tables 11/12 under legacy, common-weather, and common-horizon estimands."""
    legacy = pd.DataFrame(
        table_statistics(route_energy_series(table, "legacy_file_a_mwh"))
    ).rename(columns={"value": "legacy_file_a"})
    common = pd.DataFrame(
        table_statistics(route_energy_series(table, "common_weather_mwh"))
    ).rename(columns={"value": "common_weather_rescore"})
    horizon = pd.DataFrame(
        table_statistics(
            route_energy_series(table, "common_weather_mwh"),
            common_horizon_departures(table),
        )
    ).rename(columns={"value": "common_horizon_subset"})
    keys = ["table", "case", "subject", "statistic"]
    out = legacy.merge(common, on=keys, validate="1:1").merge(
        horizon, on=keys, validate="1:1"
    )
    out["common_minus_legacy"] = out["common_weather_rescore"] - out["legacy_file_a"]
    return out


def file_a_frame(
    units: Sequence[Unit],
    max_departures: int | None = None,
    departures: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return File A energies for all units in a single frame."""
    frames = []
    for unit in units:
        frame = select_rows(read_file_a(unit.summary_path), max_departures, departures)
        frames.append(
            frame.assign(strategy=unit.strategy, case=unit.case)[
                [
                    "strategy",
                    "case",
                    "departure_time_utc",
                    "energy_cons_mwh",
                    "details_filename",
                ]
            ].rename(columns={"departure_time_utc": "departure_utc"})
        )
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Domain audit
# ---------------------------------------------------------------------------
def domain_summary(seg: pd.DataFrame) -> pd.DataFrame:
    """Per strategy/case/variable domain statistics (any-variable included)."""
    rows = []
    for (strategy, case), grp in seg.groupby(["strategy", "case"], sort=False):
        dur = grp["duration_h"].to_numpy()
        energy = grp["energy_mwh"].to_numpy()
        n_dep = grp["route_file"].nunique()
        for name in (*VARIABLES, "any"):
            mask = grp[f"{name}_out"].to_numpy().astype(bool)
            if name == "any":
                raw = exc = None
            else:
                values = grp[RAW_COLUMN[name]].to_numpy()
                exc_all = exceedance(values, DOMAIN[name])
                raw, exc = values, exc_all
            dep_out = grp.loc[mask, "route_file"].nunique()
            rows.append(
                {
                    "strategy": strategy,
                    "case": case,
                    "corridor": CASES[case][0],
                    "variable": name,
                    "n_segments": len(grp),
                    "n_out": int(mask.sum()),
                    "pct_segments_out": 100.0 * mask.mean(),
                    "n_departures": n_dep,
                    "n_departures_out": int(dep_out),
                    "pct_departures_out": 100.0 * dep_out / n_dep,
                    "max_raw": float(raw.max()) if raw is not None else np.nan,
                    "max_exceedance": float(exc.max()) if exc is not None else np.nan,
                    "mean_raw_violating": float(raw[mask].mean())
                    if raw is not None and mask.any()
                    else np.nan,
                    "mean_exceedance_violating": float(exc[mask].mean())
                    if exc is not None and mask.any()
                    else np.nan,
                    "hours_out": float(dur[mask].sum()),
                    "pct_time_out": 100.0 * weighted_share(mask, dur),
                    "energy_out_mwh": float(energy[mask].sum()),
                    "energy_total_mwh": float(energy.sum()),
                    "pct_energy_out": 100.0 * weighted_share(mask, energy),
                }
            )
    return pd.DataFrame(rows)


def departure_energy(seg: pd.DataFrame, column: str) -> pd.DataFrame:
    """Per-departure voyage energy (MWh) indexed by (case, strategy, departure)."""
    return seg.groupby(["case", "strategy", "departure_utc"])[column].sum().sort_index()


def clean_departures(seg: pd.DataFrame) -> dict[str, set[str]]:
    """Departures per case for which no strategy has an out-of-domain segment."""
    bad = seg.groupby(["case", "strategy", "departure_utc"])["any_out"].max()
    out = {}
    for case in CASES:
        sub = bad.xs(case, level="case")
        all_dep = set(sub.index.get_level_values("departure_utc"))
        flagged = set(sub[sub > 0].index.get_level_values("departure_utc"))
        out[case] = all_dep - flagged
    return out


def paired_improvement_pct(base: pd.Series, cand: pd.Series) -> pd.Series:
    """Per-departure percent improvement, paired by departure timestamp."""
    if not base.index.equals(cand.index):
        raise ValueError("Cannot pair routes with different departure dates.")
    return (base - cand) / base * 100.0


def _series(
    energy: pd.Series, case: str, strategy: str, keep: set[str] | None
) -> pd.Series:
    s = energy.xs((case, strategy), level=("case", "strategy"))
    return s if keep is None else s[s.index.isin(keep)]


COMPARISONS = (
    ("BERS", "GC"),
    ("BERS", "CMA-ES"),
    ("CMA-ES", "GC"),
)


def table_statistics(
    energy: pd.Series, keep: Mapping[str, set[str]] | None = None
) -> list[dict[str, Any]]:
    """Table 11 and Table 12 statistics for one energy variant."""
    rows: list[dict[str, Any]] = []
    for case, (corridor, _gc) in CASES.items():
        ks = None if keep is None else keep[case]
        nowps_case = f"{'A' if corridor == 'Atlantic' else 'P'}O_noWPS"
        ref_nowps = _series(energy, nowps_case, "GC", None if keep is None else ks)
        for strategy in STRATEGIES:
            s = _series(energy, case, strategy, ks)
            gc = _series(energy, case, "GC", ks)
            nowps = ref_nowps.reindex(s.index)
            base = {"table": "Table 11", "case": case, "subject": strategy}
            rows.append({**base, "statistic": "mean_mwh", "value": s.mean()})
            rows.append({**base, "statistic": "std_mwh", "value": s.std(ddof=1)})
            rows.append(
                {
                    **base,
                    "statistic": "saving_vs_gc_same_mode_pct",
                    "value": 100.0 * (1.0 - s.mean() / gc.mean()),
                }
            )
            rows.append(
                {
                    **base,
                    "statistic": "saving_vs_gc_nowps_pct",
                    "value": 100.0 * (1.0 - s.mean() / nowps.mean()),
                }
            )
            rows.append({**base, "statistic": "n_departures", "value": len(s)})
        for cand, ref in COMPARISONS:
            imp = paired_improvement_pct(
                _series(energy, case, ref, ks), _series(energy, case, cand, ks)
            )
            base = {"table": "Table 12", "case": case, "subject": f"{cand} vs {ref}"}
            stats = {
                "mean_pct": imp.mean(),
                "std_pct": imp.std(ddof=1),
                "median_pct": imp.median(),
                "p5_pct": imp.quantile(0.05),
                "p95_pct": imp.quantile(0.95),
                "min_pct": imp.min(),
                "max_pct": imp.max(),
                "n_departures": len(imp),
            }
            rows.extend({**base, "statistic": k, "value": v} for k, v in stats.items())
    return rows


def headline_ranges(rows: pd.DataFrame) -> pd.DataFrame:
    """Min/max over the four cases of the headline Table 11/12 statistics."""
    picks = (
        ("Table 11", "saving_vs_gc_same_mode_pct", ("BERS", "CMA-ES")),
        ("Table 11", "saving_vs_gc_nowps_pct", ("BERS", "CMA-ES")),
        ("Table 12", "mean_pct", tuple(f"{c} vs {r}" for c, r in COMPARISONS)),
    )
    out = []
    for table, stat, subjects in picks:
        for subject in subjects:
            sel = rows[
                (rows["table"] == table)
                & (rows["statistic"] == stat)
                & (rows["subject"] == subject)
            ]
            if sel.empty:
                continue
            for agg in ("min", "max"):
                out.append(
                    {
                        "table": "Headline",
                        "case": "all four cases",
                        "subject": f"{subject} [{table}: {stat}]",
                        "statistic": agg,
                        "unclipped": getattr(sel["unclipped"], agg)(),
                        "clipped": getattr(sel["clipped"], agg)(),
                    }
                )
    return pd.DataFrame(out)


def intersect_departure_sets(
    left: Mapping[str, set[str]], right: Mapping[str, set[str]]
) -> dict[str, set[str]]:
    """Intersect two per-case departure selections."""
    return {case: left[case] & right[case] for case in CASES}


def clipped_vs_unclipped(
    seg: pd.DataFrame, common_horizon: Mapping[str, set[str]] | None = None
) -> pd.DataFrame:
    """Side-by-side Tables 11/12 for all pre-specified analysis populations."""
    clean = clean_departures(seg)
    samples: list[tuple[str, Mapping[str, set[str]] | None]] = [
        ("all departures", None),
        ("no-out-of-domain subset", clean),
    ]
    if common_horizon is not None:
        samples.insert(1, ("common-horizon subset", common_horizon))
        samples.append(
            (
                "common-horizon no-out-of-domain subset",
                intersect_departure_sets(clean, common_horizon),
            )
        )
    frames = []
    for sample, keep_arg in samples:
        un = pd.DataFrame(
            table_statistics(departure_energy(seg, "energy_mwh"), keep_arg)
        )
        cl = pd.DataFrame(
            table_statistics(departure_energy(seg, "c_energy_mwh"), keep_arg)
        )
        merged = un.merge(
            cl,
            on=["table", "case", "subject", "statistic"],
            suffixes=("_unclipped", "_clipped"),
        ).rename(columns={"value_unclipped": "unclipped", "value_clipped": "clipped"})
        merged.insert(0, "sample", sample)
        frames.append(merged)
        if sample in {"all departures", "common-horizon subset"}:
            frames.append(headline_ranges(merged).assign(sample=sample))
    out = pd.concat(frames, ignore_index=True)
    out["delta_clipped_minus_unclipped"] = out["clipped"] - out["unclipped"]
    return out[
        [
            "sample", "table", "case", "subject", "statistic", "unclipped",
            "clipped", "delta_clipped_minus_unclipped",
        ]
    ]  # fmt: skip


def component_energies(seg: pd.DataFrame, clipped: bool) -> pd.DataFrame:
    """Per-segment component energies (MWh); sail is the positive thrust energy."""
    prefix = "c_" if clipped else ""
    dur = seg["duration_h"] / 1000.0
    energy = pd.DataFrame(
        {
            "hull": seg[f"{prefix}p_hull_kw"] * dur,
            "wind": seg[f"{prefix}p_wind_kw"] * dur,
            "wave": seg[f"{prefix}p_wave_kw"] * dur,
            "sail": seg[f"{prefix}p_sail_kw"] * dur,
            "clamp": (seg[f"{prefix}p_power_kw"] - seg[f"{prefix}p_raw_kw"]) * dur,
            "total": seg[f"{prefix}p_power_kw"] * dur,
        }
    )
    return energy


def component_deltas(seg: pd.DataFrame) -> pd.DataFrame:
    """Signed clipped-minus-unclipped component changes per departure."""
    keys = seg[["case", "strategy", "departure_utc"]]
    un = pd.concat([keys, component_energies(seg, False)], axis=1)
    cl = pd.concat([keys, component_energies(seg, True)], axis=1)
    group = ["case", "strategy", "departure_utc"]
    un_dep = un.groupby(group).sum()
    cl_dep = cl.groupby(group).sum()
    rows = []
    for (case, strategy), _ in un_dep.groupby(level=["case", "strategy"]):
        u = un_dep.xs((case, strategy), level=("case", "strategy"))
        c = cl_dep.xs((case, strategy), level=("case", "strategy"))
        for comp in COMPONENTS:
            delta = c[comp] - u[comp]
            rows.append(
                {
                    "case": case,
                    "strategy": strategy,
                    "component": comp,
                    "mean_unclipped_mwh": u[comp].mean(),
                    "mean_clipped_mwh": c[comp].mean(),
                    "mean_delta_mwh": delta.mean(),
                    "mean_delta_pct_of_total_unclipped": 100.0
                    * delta.mean()
                    / u["total"].mean(),
                    "min_delta_mwh": delta.min(),
                    "max_delta_mwh": delta.max(),
                    "n_departures_increased": int((delta > 1e-12).sum()),
                    "n_departures_decreased": int((delta < -1e-12).sum()),
                    "n_departures": len(delta),
                }
            )
    return pd.DataFrame(rows)


def select_departures(seg: pd.DataFrame, keep: Mapping[str, set[str]]) -> pd.DataFrame:
    """Select segment rows using a per-case departure mapping."""
    mask = np.zeros(len(seg), dtype=bool)
    for case, departures in keep.items():
        mask |= seg["case"].eq(case) & seg["departure_utc"].isin(departures)
    return seg.loc[mask].copy()


def sampled_domain_summary(
    seg: pd.DataFrame, common_horizon: Mapping[str, set[str]]
) -> pd.DataFrame:
    """Domain summaries for all routes and the uncontaminated horizon subset."""
    all_rows = domain_summary(seg)
    all_rows.insert(0, "sample", "all departures")
    horizon = domain_summary(select_departures(seg, common_horizon))
    horizon.insert(0, "sample", "common-horizon subset")
    return pd.concat([all_rows, horizon], ignore_index=True)


def sampled_component_deltas(
    seg: pd.DataFrame, common_horizon: Mapping[str, set[str]]
) -> pd.DataFrame:
    """Component sensitivities for all routes and the common-horizon subset."""
    all_rows = component_deltas(seg)
    all_rows.insert(0, "sample", "all departures")
    horizon = component_deltas(select_departures(seg, common_horizon))
    horizon.insert(0, "sample", "common-horizon subset")
    return pd.concat([all_rows, horizon], ignore_index=True)


# ---------------------------------------------------------------------------
# Provenance and output writers
# ---------------------------------------------------------------------------
def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def collect_metadata(
    cmaes_root: Path, bers_root: Path, era5_info: Mapping[str, Any]
) -> dict[str, Any]:
    """Gather code, input-root and model provenance."""
    dirty = _git("status", "--porcelain", "--", "routetools", "revision2/*.py", "tests")
    return {
        "analysis_git_commit": _git("rev-parse", "HEAD"),
        "analysis_tree_clean": not dirty,
        "cmaes_gc_root": str(cmaes_root),
        "bers_gc_root": str(bers_root),
        "era5_files": era5_info,
        "code_fingerprint": code_fingerprint(),
        "model_implementation": {
            "performance_py_blob": _git("hash-object", "routetools/performance.py"),
            "cost_py_blob": _git("hash-object", "routetools/cost.py"),
            "evaluator": "routetools.cost.evaluate_route_energy conventions",
            "domain": {k: list(v) for k, v in DOMAIN.items()},
            "jax_precision": "float32 fields/curves (evaluator default); float64 power",
        },
    }


def md_table(frame: pd.DataFrame, floats: int = 3) -> str:
    """Render a DataFrame as a GitHub markdown table."""

    def fmt(value: Any) -> str:
        if isinstance(value, float | np.floating):
            return "" if np.isnan(value) else f"{value:.{floats}f}"
        return str(value)

    head = "| " + " | ".join(map(str, frame.columns)) + " |"
    sep = "|" + "|".join("---" for _ in frame.columns) + "|"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.to_numpy()]
    return "\n".join([head, sep, *body])


def write_companion(
    csv_path: Path,
    description: str,
    columns: Mapping[str, str],
    meta: Mapping[str, Any],
) -> Path:
    """Write the schema/provenance companion of a small output file."""
    path = csv_path.with_suffix(".schema.md")
    (csv_path.parent / "task1_provenance.json").write_text(
        json.dumps(meta, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        f"# {csv_path.name}",
        "",
        description,
        "",
        "## Columns",
        "",
        "| Column | Meaning |",
        "|---|---|",
        *[f"| `{k}` | {v} |" for k, v in columns.items()],
        "",
        "## Provenance",
        "",
        f"- Analysis code commit: `{meta['analysis_git_commit']}` "
        f"(source tree clean: {meta['analysis_tree_clean']})",
        "- Full ERA5 file records (size, time range, dims, variables, SHA-256) "
        "and code fingerprints: `task1_provenance.json`.",
        f"- CMA-ES and GC route root: `{meta['cmaes_gc_root']}`",
        f"- BERS and GC route root: `{meta['bers_gc_root']}`",
        "- ERA5 inputs (resolved, with SHA-256): "
        + ", ".join(
            f"`{f['path']}` (`{f['sha256'][:16]}`)"
            for v in meta["era5_files"].values()
            for f in v["resolved"]
        ),
        "- Model: `routetools/performance.py` "
        f"(blob `{meta['model_implementation']['performance_py_blob']}`) with the "
        "File A evaluator conventions of `routetools.cost.evaluate_route_energy` "
        f"(blob `{meta['model_implementation']['cost_py_blob']}`).",
        f"- Domain: {meta['model_implementation']['domain']}",
        "- Sensitivity analysis only: clipping is not a corrected physical model.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def summary_latex(summary: pd.DataFrame) -> str:
    """Compact LaTeX table of out-of-domain shares per case and strategy."""
    if "sample" in summary:
        summary = summary[summary["sample"] == "all departures"]
    piv = summary.pivot_table(
        index=["case", "strategy"],
        columns="variable",
        values=[
            "pct_segments_out",
            "pct_departures_out",
            "pct_time_out",
            "pct_energy_out",
        ],
    )
    lines = [
        "\\begin{tabular}{llrrrrrrr}",
        "\\hline",
        "Case & Strategy & TWS\\,(\\%) & Hs\\,(\\%) & Speed\\,(\\%) & Any seg.\\,(\\%) "
        "& Dep.\\,(\\%) & Time\\,(\\%) & Energy\\,(\\%) \\\\",
        "\\hline",
    ]
    order = [(c, s) for c in CASES for s in STRATEGIES]
    for case, strategy in order:
        r = piv.loc[(case, strategy)]
        lines.append(
            f"{case.replace('_', '\\_')} & {strategy} & "
            f"{r[('pct_segments_out', 'tws')]:.2f} & "
            f"{r[('pct_segments_out', 'hs')]:.2f} & "
            f"{r[('pct_segments_out', 'speed')]:.2f} & "
            f"{r[('pct_segments_out', 'any')]:.2f} & "
            f"{r[('pct_departures_out', 'any')]:.1f} & "
            f"{r[('pct_time_out', 'any')]:.2f} & "
            f"{r[('pct_energy_out', 'any')]:.2f} \\\\"
        )
    lines += ["\\hline", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


SEGMENT_COLUMN_DOCS = {
    "strategy": "GC, CMA-ES or BERS.",
    "case": "AO_noWPS, AO_WPS, PO_noWPS or PO_WPS (GC rows use the same label).",
    "wps": "1 when wingsails are deployed.",
    "departure_utc": "Departure timestamp (UTC), the pairing key.",
    "route_file": "File B track filename (details_filename).",
    "segment_index": "0-based segment index along the route.",
    "n_segments": "Number of segments of the route.",
    "midpoint_time_utc": "Segment midpoint time (UTC).",
    "midpoint_offset_h": "Midpoint time in hours since the ERA5 dataset epoch.",
    "duration_h": "Segment duration (h) = passage hours / n_segments.",
    "mid_lon_deg": "Midpoint longitude in the ERA5 grid frame (deg, unwrapped).",
    "mid_lat_deg": "Midpoint latitude (deg).",
    "bearing_deg": "Segment true bearing (deg).",
    "tws_mps": "True wind speed at the midpoint (m/s).",
    "twa_deg": "True wind angle as supplied to the model, in [0, 360) (deg).",
    "twa_sym_deg": "TWA folded to [0, 180] by the model's sign symmetry (deg).",
    "hs_m": "Significant wave height (m).",
    "mwa_deg": "Mean wave angle as supplied to the model, in [0, 360) (deg).",
    "mwa_sym_deg": "|MWA| after the model's centring to [-180, 180) (deg).",
    "speed_mps": "Speed through water = segment distance / duration (m/s).",
    "tws_out, hs_out, speed_out, twa_out, mwa_out, any_out": (
        "1 when the variable lies outside its closed domain interval."
    ),
    "p_hull_kw, p_wind_kw, p_wave_kw, p_sail_kw": (
        "Unclipped power components (kW); sail thrust is a positive quantity."
    ),
    "p_raw_kw": "hull + wind + wave - sail before the non-negativity clamp (kW).",
    "power_kw": "Unclipped total power = max(p_raw_kw, 0) (kW), as in File A.",
    "energy_mwh": "Unclipped segment energy = power_kw * duration_h / 1000 (MWh).",
    "tws_clip_mps, hs_clip_m, speed_clip_mps": "Inputs after clipping to the domain.",
    "c_*": "Same power components/total/energy evaluated with the clipped inputs.",
}


def write_segment_doc(
    path: Path, info: Mapping[str, Any], meta: Mapping[str, Any], gz_name: str
) -> None:
    """Describe the large segment artifact (schema, provenance, checksum)."""
    lines = [
        "# task1_segment_inputs",
        "",
        f"Per-segment performance-model inputs for GC, CMA-ES and BERS routes of the "
        f"four cases and 366 departures. File: `{gz_name}` (gzip CSV; Parquet "
        "support is unavailable in this environment). It is **not tracked in Git**; "
        "it stays on Vangelis under `revision2/`.",
        "",
        f"- Rows (segments): {info['rows']}",
        f"- Size: {info['bytes']} bytes",
        f"- SHA-256: `{info['sha256']}`",
        f"- Checkpoint parts combined: {info['parts']}",
        "",
        "## Columns",
        "",
        "| Column | Meaning |",
        "|---|---|",
        *[f"| `{k}` | {v} |" for k, v in SEGMENT_COLUMN_DOCS.items()],
        "",
        "Units: speeds m/s, heights m, angles deg, power kW, energy MWh, time h.",
        "",
        "## Provenance",
        "",
        f"- Analysis code commit: `{meta['analysis_git_commit']}` "
        f"(source tree clean: {meta['analysis_tree_clean']})",
        f"- CMA-ES and GC root: `{meta['cmaes_gc_root']}`",
        f"- BERS and GC root: `{meta['bers_gc_root']}`",
        "- ERA5: "
        + ", ".join(
            f"`{f['path']}`" for v in meta["era5_files"].values() for f in v["resolved"]
        ),
        f"- Model: {meta['model_implementation']}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def render_report(
    meta: Mapping[str, Any],
    repro: Mapping[str, Any],
    gate_error: str | None,
    summary: pd.DataFrame | None = None,
    cvu: pd.DataFrame | None = None,
    deltas: pd.DataFrame | None = None,
    smoke: Mapping[str, Any] | None = None,
    segment_info: Mapping[str, Any] | None = None,
    grid_text: str | None = None,
    audit_trail: str | None = None,
    table_comparison: pd.DataFrame | None = None,
) -> str:
    """Render the Task 1 report."""
    lines = [
        "# Task 1: out-of-domain audit and clipped-input sensitivity",
        "",
        f"- Analysis code commit: `{meta['analysis_git_commit']}` "
        f"(source tree clean: {meta['analysis_tree_clean']})",
        f"- CMA-ES and GC root: `{meta['cmaes_gc_root']}`",
        f"- BERS and GC root: `{meta['bers_gc_root']}`",
        "- Domain: TWS [0, 30] m/s, Hs [0, 10] m, speed [0, 14.5] m/s, symmetric "
        "TWA/MWA [0, 180] deg. Reporting thresholds stay at 20 m/s and 7 m; the "
        "19.9/6.9 FMS values are an activation buffer, not a domain.",
        "- This is a sensitivity analysis, not a corrected physical model; "
        "clipping is not presented as realistic.",
        "- The common-weather calculation is a post hoc re-score of preserved "
        "routes, not an optimizer rerun. It makes evaluation weather consistent "
        "but does not give affected BERS optimizations the continuation weather "
        "they apparently lacked during route generation.",
        "- Evaluation grid: File A energies (and therefore Tables 11 and 12, the "
        "reproduction gate, the domain audit and the clipped sensitivity) use the "
        "saved uniformly timed waypoints with segment midpoints. The 30-minute "
        "grid belongs only to the CMA-ES objective (see the optimization-grid "
        "audit); the two samplings are never pooled.",
        "- Resolved ERA5 files (hourly, 2024 plus the January-2025 continuation "
        "found by `routetools.era5.loader.loadable_era5_paths`):",
        *[
            f"  - `{f['path']}`: {f['n_times']} steps, {f['time_start']} to "
            f"{f['time_end']}, {f['bytes']} bytes, dims {f['dims']}, variables "
            f"{f['variables']}, SHA-256 `{f['sha256']}`"
            for e in meta["era5_files"].values()
            for f in e["resolved"]
        ],
        "- Full machine-readable provenance: `task1_provenance.json`.",
        "",
        "## 1. Reproduction gate",
        "",
    ]
    if smoke:
        lines += [
            f"Smoke test ({smoke['n_routes']} routes: six departures per strategy "
            "and case, one early-year control and five late-2024 departures "
            f"{', '.join(d[:10] for d in SMOKE_DEPARTURES[1:])}): "
            f"max |diff| = {smoke['max_abs_mwh']:.3e} MWh, "
            f"{smoke['n_outside_tolerance']} outside tolerance.",
            "",
        ]
    tol = (
        f"{MODEL_MAX_ABS_ERROR_KW} kW x passage hours / 1000 per route (documented "
        "maximum analytical-model error per segment, applied to every segment "
        "coherently; 0.0177 MWh Atlantic, 0.0292 MWh Pacific)"
    )
    lines += [
        f"Tolerance: {tol}.",
        "",
        f"- Routes compared: {repro['n_routes']}",
        f"- Maximum absolute discrepancy: {repro['max_abs_mwh']:.6e} MWh "
        f"({repro['max_abs_route']})",
        f"- Mean absolute discrepancy: {repro['mean_abs_mwh']:.6e} MWh",
        f"- Maximum relative discrepancy: {repro['max_rel']:.6e} "
        f"({repro['max_rel_route']})",
        f"- Routes outside tolerance: {repro['n_outside_tolerance']}",
        f"- Pre-specified expected late-BERS weather mismatches: "
        f"{repro['n_expected_bers_weather_mismatch']}",
        f"- Unexpected gate failures: {repro['n_unexpected_failure']}",
        "",
    ]
    if gate_error:
        lines += [
            "**Conditional reproduction gate FAILED.** Clipped savings were not "
            "produced.",
            "",
            gate_error,
            "",
            "No formula, interpolation or tolerance was tuned. Stop and decide "
            "how to proceed.",
        ]
        return "\n".join(lines) + "\n"
    lines += [
        "The conditional gate passed: GC/CMA-ES and all BERS routes that do not "
        "require 2025 weather reproduce File A. Late BERS differences are retained "
        "as the pre-specified evidence of the mixed-weather legacy evaluation.",
        "",
    ]
    if summary is None:
        lines += ["Gate-only evidence: the domain audit has not been run yet."]
        if audit_trail:
            lines += ["", audit_trail]
        return "\n".join(lines) + "\n"
    if segment_info:
        lines += [
            f"Segment artifact: {segment_info['rows']} rows, "
            f"{segment_info['bytes']} bytes, SHA-256 `{segment_info['sha256']}` "
            "(not tracked in Git).",
            "",
        ]
    assert summary is not None and cvu is not None and deltas is not None
    assert table_comparison is not None
    key_stats = table_comparison[
        table_comparison["statistic"].isin(
            ["mean_mwh", "saving_vs_gc_same_mode_pct", "mean_pct", "n_departures"]
        )
    ]
    lines += [
        "## 2. Weather-consistency correction and manuscript tables",
        "",
        "`legacy_file_a` preserves the manuscript inputs; `common_weather_rescore` "
        "uses one continuous weather treatment for every saved route; "
        "`common_horizon_subset` excludes every departure whose evaluation uses "
        "post-2024 weather for any strategy.",
        "",
        md_table(key_stats, 4),
        "",
        "The all-departure common-weather comparison is evaluatively consistent, "
        "but affected BERS routes were optimized under the legacy clamped weather. "
        "The common-horizon subset is the comparison free of both defects without "
        "an optimizer rerun.",
        "",
    ]
    anyrows = summary[summary["variable"] == "any"].drop(
        columns=["variable", "corridor"]
    )
    lines += [
        "## 3. Domain audit (any variable)",
        "",
        md_table(
            anyrows[
                [
                    "strategy",
                    "case",
                    "n_segments",
                    "n_out",
                    "pct_segments_out",
                    "pct_departures_out",
                    "hours_out",
                    "pct_time_out",
                    "energy_out_mwh",
                    "pct_energy_out",
                ]
            ]
        ),
        "",
        "Per-variable rows (including angles, which cannot leave [0, 180] after "
        "folding) are in `task1_out_of_domain_summary.csv`.",
        "",
        "## 4. Clipped-input sensitivity",
        "",
        "### Table 11 (all departures)",
        "",
    ]
    t11 = cvu[(cvu["sample"] == "all departures") & (cvu["table"] == "Table 11")]
    t11 = t11[t11["statistic"] != "n_departures"]
    lines += [
        md_table(t11.drop(columns=["sample", "table"]), 4),
        "",
        "### Table 12",
        "",
    ]
    t12 = cvu[(cvu["sample"] == "all departures") & (cvu["table"] == "Table 12")]
    t12 = t12[t12["statistic"].isin(["mean_pct", "median_pct", "std_pct"])]
    lines += [
        md_table(t12.drop(columns=["sample", "table"]), 4),
        "",
        "### Headline ranges",
        "",
    ]
    lines += [
        md_table(cvu[cvu["table"] == "Headline"].drop(columns=["sample", "table"]), 4),
        "",
    ]
    sub = cvu[
        (cvu["sample"] != "all departures") & (cvu["statistic"] == "n_departures")
    ]
    lines += [
        "### Restricted-subset sample sizes",
        "",
        "Retained departures for the common-horizon, no-out-of-domain, and "
        "intersection analyses:",
        "",
        md_table(
            sub[["sample", "case", "subject", "unclipped"]].drop_duplicates(
                ["sample", "case"]
            ),
            0,
        ),
        "",
        "## 5. Component deltas (clipped minus unclipped, MWh per voyage)",
        "",
        "`total = hull + wind + wave - sail + clamp`; `clamp` is the "
        "non-negativity residual. Signs are reported as computed.",
        "",
        md_table(
            deltas[
                [
                    "sample",
                    "case",
                    "strategy",
                    "component",
                    "mean_delta_mwh",
                    "mean_delta_pct_of_total_unclipped",
                    "min_delta_mwh",
                    "max_delta_mwh",
                ]
            ],
            5,
        ),
        "",
        *([grid_text, ""] if grid_text else []),
        "## 6. Limitations and manuscript implications",
        "",
        "- The domain flags use the evaluator's midpoint convention; sub-segment "
        "excursions are not seen.",
        "- Clipping changes only TWS, Hs and speed at the final evaluation; "
        "routes, angles, durations and optimiser outputs are untouched.",
        "- Legacy File A cross-strategy comparisons are inconsistent for routes "
        "requiring 2025 weather. They should not remain as unqualified evidence.",
        "- Common-weather re-scoring supports a consistent deployment-style "
        "comparison; the common-horizon subset is the clean optimizer comparison "
        "available without rerunning BERS.",
    ]
    if audit_trail:
        lines += ["", audit_trail]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, float_format="%.9g")


SMOKE_DEPARTURES = (
    "2024-01-01 12:00:00",
    "2024-12-08 12:00:00",
    "2024-12-17 12:00:00",
    "2024-12-25 12:00:00",
    "2024-12-26 12:00:00",
    "2024-12-31 12:00:00",
)
LAST_DEPARTURE = datetime(2024, 12, 31, 12, tzinfo=UTC)


def _era5(args: argparse.Namespace) -> dict[str, Path]:
    return {k: getattr(args, k) for k in ERA5_DEFAULTS}


def _resolve_era5(args: argparse.Namespace) -> dict[str, Any]:
    """Resolve and validate all ERA5 files (hourly, contiguous, long enough)."""
    from routetools.swopp3 import SWOPP3_CASES

    info = resolve_era5_inputs(_era5(args), ERA5_SHA_CACHE)
    longest = max(float(c["passage_hours"]) for c in SWOPP3_CASES.values())
    check_era5_continuity(info, LAST_DEPARTURE + timedelta(hours=longest))
    return info


def _context(args: argparse.Namespace, info: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "code_sha256": code_fingerprint()["combined_sha256"],
        "era5_sha256": era5_fingerprint(info),
        "sampling": SAMPLING_OPT30 if args.sampling == "opt30" else SAMPLING_FILE_A,
    }


def _departure_list(args: argparse.Namespace) -> list[str] | None:
    return args.departures.split(",") if args.departures else None


def _expected_kwargs(args: argparse.Namespace, units: Sequence[Unit], ctx: Mapping):
    return {
        "units": units,
        "chunk_size": args.chunk_size,
        "max_departures": args.max_departures,
        "departures": _departure_list(args),
        "context": ctx,
        "track_fp": track_fingerprint,
    }


def run_extract(args: argparse.Namespace) -> None:
    """Extract segments for the selected corridor(s) into the parts directory."""
    from routetools.violations import load_default_weather_resources

    verify_gc_identical(args.cmaes_root, args.bers_root)
    info = _resolve_era5(args)
    units = [
        u for u in build_units(args.cmaes_root, args.bers_root)
        if args.corridor == "all" or corridor_of(u.case) == args.corridor
    ]  # fmt: skip
    resources = load_default_weather_resources(
        wind_path_atlantic=args.wind_atlantic,
        wave_path_atlantic=args.wave_atlantic,
        wind_path_pacific=args.wind_pacific,
        wave_path_pacific=args.wave_pacific,
    )
    counts = extract_units(
        units,
        args.parts_dir,
        make_route_computer(resources, args.sampling),
        chunk_size=args.chunk_size,
        max_departures=args.max_departures,
        departures=_departure_list(args),
        context=_context(args, info),
        track_fp=track_fingerprint,
    )
    print(f"extract[{args.corridor}]: {counts}")


def _spawn_extract(args: argparse.Namespace, corridor: str) -> None:
    cmd = [
        sys.executable, str(Path(__file__).resolve()), "extract",
        "--corridor", corridor,
        "--parts-dir", str(args.parts_dir),
        "--cmaes-root", str(args.cmaes_root),
        "--bers-root", str(args.bers_root),
        "--chunk-size", str(args.chunk_size),
        "--sampling", args.sampling,
    ]  # fmt: skip
    for key, path in _era5(args).items():
        cmd += [f"--{key.replace('_', '-')}", str(path)]
    if args.max_departures is not None:
        cmd += ["--max-departures", str(args.max_departures)]
    if args.departures:
        cmd += ["--departures", args.departures]
    subprocess.run(cmd, check=True, cwd=ROOT)


REPRO_COLUMNS = {
    "final_required_evaluation_time_utc": "Latest segment midpoint queried (UTC).",
    "requires_2025_weather": "At least one segment midpoint is after the final "
    "2024 ERA5 timestamp (2024-12-31 23:00 UTC).",
    "legacy_file_a_mwh": "Preserved energy_cons_mwh from File A (MWh).",
    "common_weather_mwh": "Saved route re-scored with continuous 2024+Jan-2025 "
    "ERA5 (MWh).",
    "weather_consistency_delta_mwh": "common_weather - legacy_file_a (MWh).",
    "abs_diff_mwh": "Absolute weather-consistency delta (MWh).",
    "rel_diff": "abs_diff / legacy_file_a.",
    "tolerance_mwh": "0.05 kW x passage hours / 1000.",
    "within_tolerance": "abs_diff <= tolerance.",
    "gate_classification": "reproduced_legacy_file_a, the pre-specified expected "
    "late-BERS mismatch, or an unexpected failure.",
}


def run_smoke(args: argparse.Namespace) -> int:
    """Late-departure smoke test plus an early-year control: reproduction only."""
    args.parts_dir = args.smoke_dir
    args.max_departures = None
    args.sampling = "file_a"
    args.departures = ",".join(SMOKE_DEPARTURES)
    info = _resolve_era5(args)
    for corridor in ("atlantic", "pacific"):
        _spawn_extract(args, corridor)
    units = build_units(args.cmaes_root, args.bers_root)
    verify_parts(args.smoke_dir, **_expected_kwargs(args, units, _context(args, info)))
    table = reproduction_table(
        load_parts(args.smoke_dir),
        file_a_frame(units, departures=SMOKE_DEPARTURES),
    )
    meta = collect_metadata(args.cmaes_root, args.bers_root, info)
    path = OUT / "task1_smoke_reproduction.csv"
    _write_csv(table, path)
    write_companion(
        path,
        "Smoke test with hourly 2024 plus January-2025 ERA5: early control "
        f"({SMOKE_DEPARTURES[0]}) and late departures {SMOKE_DEPARTURES[1:]} for "
        "every strategy and case; summed analytical-model segment energy versus "
        "File A.",
        REPRO_COLUMNS,
        meta,
    )
    print(json.dumps(reproduction_stats(table), indent=2))
    check_reproduction_gate(table)
    return 0


def run_grid30(args: argparse.Namespace) -> int:
    """Optimization-grid audit: domain incidence on the 30-minute proxy grid."""
    args.sampling = "opt30"
    args.departures = None
    args.max_departures = None
    info = _resolve_era5(args)
    for corridor in ("atlantic", "pacific"):
        _spawn_extract(args, corridor)
    units = build_units(args.cmaes_root, args.bers_root)
    verify_parts(args.parts_dir, **_expected_kwargs(args, units, _context(args, info)))
    seg = load_parts(args.parts_dir)
    summary = domain_summary(seg)
    meta = collect_metadata(args.cmaes_root, args.bers_root, info)
    path = OUT / "task1_optimization_grid_audit.csv"
    _write_csv(summary, path)
    write_companion(
        path,
        "Optimization-grid audit (proxy): domain incidence on a 30-minute "
        "time grid obtained by linear interpolation of the saved waypoints. NOT an "
        "exact reconstruction of the CMA-ES objective, whose Bezier curve and "
        "control points were not preserved. Never pooled with the File A "
        "sampling; energy columns are proxy values on this grid.",
        {"(same columns as task1_out_of_domain_summary.csv)": "See that file."},
        meta,
    )
    manifest = OUT / "task1_optimization_grid_audit.manifest.json"
    manifest.write_text(
        json.dumps(grid_manifest(args, info), indent=1, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"grid30 complete; {len(seg)} proxy segments")
    return 0


def grid_manifest(args: argparse.Namespace, info: Mapping[str, Any]) -> dict[str, Any]:
    """Context binding the optimization-grid audit CSV to its inputs."""
    args.sampling = "opt30"
    return {
        **_context(args, info),
        "cmaes_gc_root": str(args.cmaes_root),
        "bers_gc_root": str(args.bers_root),
        "dt_eval_minutes": OPT_GRID_MINUTES,
    }


def check_grid_manifest(path: Path, expected: Mapping[str, Any]) -> None:
    """Reject an optimization-grid audit made with other inputs, code or sampling."""
    if not path.is_file():
        raise CheckpointMismatchError(f"Grid-audit manifest {path.name} is missing.")
    found = json.loads(path.read_text(encoding="utf-8"))
    if found != dict(expected):
        raise CheckpointMismatchError(
            f"{path.name} does not match the current inputs/code/sampling; rerun "
            "the grid30 stage."
        )


def grid_comparison(file_a: pd.DataFrame, opt: pd.DataFrame) -> pd.DataFrame:
    """Side-by-side incidence, each sampling with its own denominators."""
    keys = ["strategy", "case", "variable"]
    cols = ["n_segments", "pct_segments_out", "pct_time_out", "pct_departures_out"]
    merged = file_a[[*keys, *cols]].merge(
        opt[[*keys, *cols]], on=keys, suffixes=("_file_a", "_opt30"), validate="1:1"
    )
    for c in cols[1:]:
        merged[f"{c}_diff_pp"] = merged[f"{c}_opt30"] - merged[f"{c}_file_a"]
    return merged


def audit_trail_text(before_path: Path, info: Mapping[str, Any], after: Mapping) -> str:
    """Summarise the 2024-only failure and the recovery with the resolved files."""
    lines = ["## Audit trail: 2024-only failure and recovery", ""]
    if before_path.is_file():
        before = reproduction_stats(pd.read_csv(before_path))
        lines += [
            "- The first run used only the 2024 ERA5 files. The loader then clamps "
            "weather after 2024-12-31 23:00. Result: "
            f"{before['n_outside_tolerance']} of {before['n_routes']} routes outside "
            f"tolerance, max |diff| {before['max_abs_mwh']:.3f} MWh "
            f"({before['max_abs_route']}), mean {before['mean_abs_mwh']:.3f} MWh "
            f"(evidence: `{before_path.name}`, commit f4f83f5). All failures were "
            "GC/CMA-ES routes ending in 2025.",
        ]
    lines += [
        "- The hourly January-2025 continuation files exist next to the 2024 "
        "files and are resolved by `routetools.era5.loader.loadable_era5_paths`. "
        "Files used:",
        *[
            f"  - `{f['path']}` (SHA-256 `{f['sha256'][:16]}...`)"
            for e in info.values()
            for f in e["resolved"]
        ],
        f"- After: {after['n_outside_tolerance']} of {after['n_routes']} routes "
        f"outside tolerance, max |diff| {after['max_abs_mwh']:.3e} MWh "
        f"({after['max_abs_route']}), mean {after['mean_abs_mwh']:.3e} MWh.",
        "- Formulas, interpolation and tolerance were not changed. The 6-hour "
        "backup files were not used.",
    ]
    return "\n".join(lines)


def grid_section(comp: pd.DataFrame) -> str:
    """Render the optimization-grid audit section of the report."""
    anyrows = comp[comp["variable"] == "any"]
    show = anyrows[
        [
            "strategy",
            "case",
            "n_segments_file_a",
            "pct_segments_out_file_a",
            "n_segments_opt30",
            "pct_segments_out_opt30",
            "pct_segments_out_diff_pp",
            "pct_departures_out_file_a",
            "pct_departures_out_opt30",
        ]
    ]
    worst = anyrows["pct_segments_out_diff_pp"].abs().max()
    return "\n".join(
        [
            "### Optimization-grid audit (30-minute proxy; separate denominators)",
            "",
            "The CMA-ES objective samples a Bezier curve on a 30-minute grid "
            "(`dt_eval_minutes`); the control points were not preserved, so that "
            "grid cannot be reconstructed exactly. The rows below use a proxy: the "
            "saved waypoints linearly interpolated onto the objective's segment "
            "count. File A sampling and proxy are never pooled; each column set "
            "has its own denominator. Energies are not compared across grids.",
            "",
            md_table(show, 2),
            "",
            f"Largest absolute difference in percentage of segments outside the "
            f"domain (any variable): {worst:.2f} percentage points. Per-variable "
            "rows: `task1_grid_comparison.csv`.",
        ]
    )


def run_analyze(args: argparse.Namespace) -> int:
    """Reproduction gate, domain audit, clipped tables and report."""
    verify_gc_identical(args.cmaes_root, args.bers_root)
    args.sampling = "file_a"
    args.departures = None
    args.max_departures = None
    info = _resolve_era5(args)
    units = build_units(args.cmaes_root, args.bers_root)
    verify_parts(args.parts_dir, **_expected_kwargs(args, units, _context(args, info)))
    meta = collect_metadata(args.cmaes_root, args.bers_root, info)
    seg = load_parts(args.parts_dir)
    table = reproduction_table(seg, file_a_frame(units))
    if len(table) != len(units) * N_DEPARTURES:
        raise ValueError(
            f"Expected {len(units) * N_DEPARTURES} routes, got {len(table)}."
        )
    repro_path = OUT / "task1_reproduction.csv"
    _write_csv(table, repro_path)
    write_companion(
        repro_path,
        "Reproduction gate with hourly 2024 plus January-2025 ERA5: per-route "
        "summed analytical-model segment energy versus File A.",
        REPRO_COLUMNS,
        meta,
    )
    correction_path = OUT / "task1_weather_consistency_correction.csv"
    _write_csv(table, correction_path)
    write_companion(
        correction_path,
        "Route-level legacy File A versus common-weather post hoc re-score. The "
        "classification implements the pre-specified conditional gate; no route "
        "or optimizer output is changed.",
        REPRO_COLUMNS,
        meta,
    )
    repro = reproduction_stats(table)
    smoke_path = OUT / "task1_smoke_reproduction.csv"
    smoke = (
        reproduction_stats(pd.read_csv(smoke_path)) if smoke_path.is_file() else None
    )
    trail = audit_trail_text(OUT / "task1_reproduction_2024_only.csv", info, repro)
    try:
        check_reproduction_gate(table)
    except ReproductionGateError as exc:
        (OUT / "task1_report.md").write_text(
            render_report(
                meta,
                repro,
                f"{exc}\n\n{failure_diagnostics(table)}",
                smoke=smoke,
                audit_trail=trail,
            ),
            encoding="utf-8",
        )
        print(f"GATE FAILED: {exc}")
        return 2

    if args.gate_only:
        (OUT / "task1_report.md").write_text(
            render_report(meta, repro, None, smoke=smoke, audit_trail=trail),
            encoding="utf-8",
        )
        print("gate passed; gate-only evidence written")
        return 0

    seg_info = combine_parts(args.parts_dir, args.segment_path)
    write_segment_doc(
        OUT / "task1_segment_inputs.md", seg_info, meta, args.segment_path.name
    )
    common_horizon = common_horizon_departures(table)
    table_comp = weather_consistency_tables(table)
    summary = sampled_domain_summary(seg, common_horizon)
    cvu = clipped_vs_unclipped(seg, common_horizon)
    deltas = sampled_component_deltas(seg, common_horizon)
    grid_text = None
    opt_path = OUT / "task1_optimization_grid_audit.csv"
    if opt_path.is_file():
        check_grid_manifest(
            OUT / "task1_optimization_grid_audit.manifest.json",
            grid_manifest(args, info),
        )
        args.sampling = "file_a"
        comp = grid_comparison(
            summary[summary["sample"] == "all departures"].drop(columns="sample"),
            pd.read_csv(opt_path),
        )
        comp_path = OUT / "task1_grid_comparison.csv"
        _write_csv(comp, comp_path)
        write_companion(
            comp_path,
            "Domain incidence on the File A sampling versus the 30-minute "
            "optimization-grid proxy, side by side. Each sampling keeps its own "
            "denominators; *_diff_pp = opt30 minus file_a in percentage points.",
            {"variable": "tws, hs, speed, twa, mwa or any."},
            meta,
        )
        grid_text = grid_section(comp)

    outputs = {
        "task1_out_of_domain_summary.csv": (
            summary,
            "Domain audit per strategy, case and variable (variable 'any' = union).",
            {
                "sample": "All departures or the common-horizon subset.",
                "variable": "tws, hs, speed, twa, mwa or any.",
                "n_segments / n_out / pct_segments_out": "Segment counts and share.",
                "n_departures_out / pct_departures_out": "Departures with >=1 "
                "out-of-domain segment.",
                "max_raw / max_exceedance": "Maximum raw value (angles after "
                "folding) and distance beyond the domain.",
                "mean_raw_violating / mean_exceedance_violating": "Means over "
                "violating segments.",
                "hours_out / pct_time_out": "Duration outside the domain "
                "(time-weighted share).",
                "energy_out_mwh / pct_energy_out": "Signed unclipped energy of "
                "segments flagged for the variable, and its share of voyage energy.",
            },
        ),
        "task1_clipped_vs_unclipped.csv": (
            cvu,
            "Tables 11 and 12 recomputed unclipped and clipped, plus headline "
            "ranges. Savings use ratio of means; Table 12 uses per-departure "
            "paired percentages (std with ddof=1).",
            {
                "sample": "All departures, common horizon, no-out-of-domain, or "
                "their intersection.",
                "table / case / subject / statistic": "Row identity.",
                "unclipped / clipped": "Statistic value for each variant.",
                "delta_clipped_minus_unclipped": "Signed difference.",
            },
        ),
        "task1_component_deltas.csv": (
            deltas,
            "Signed clipped-minus-unclipped changes per voyage by component. "
            "total = hull + wind + wave - sail + clamp.",
            {
                "sample": "All departures or the common-horizon subset.",
                "component": "hull, wind, wave, sail (thrust energy), clamp "
                "(non-negativity residual) or total.",
                "mean_*": "Means over departures (MWh).",
                "mean_delta_pct_of_total_unclipped": "Mean delta over mean "
                "unclipped total energy (%).",
                "n_departures_increased / decreased": "Departures with positive / "
                "negative change.",
            },
        ),
        "task1_table_11_12_weather_consistency.csv": (
            table_comp,
            "Tables 11 and 12 under preserved legacy File A, a common-weather "
            "post hoc re-score of all routes, and the uncontaminated common-horizon "
            "subset.",
            {
                "table / case / subject / statistic": "Row identity.",
                "legacy_file_a": "Statistic from preserved File A energies.",
                "common_weather_rescore": "Statistic after evaluating every saved "
                "route with continuous 2024+January-2025 weather.",
                "common_horizon_subset": "Statistic after additionally excluding "
                "departures that require post-2024 weather.",
                "common_minus_legacy": "Signed all-departure common-weather minus "
                "legacy statistic; units follow statistic.",
            },
        ),
    }
    for name, (frame, desc, cols) in outputs.items():
        _write_csv(frame, OUT / name)
        write_companion(OUT / name, desc, cols, meta)
    (OUT / "task1_out_of_domain_summary.tex").write_text(
        summary_latex(summary), encoding="utf-8"
    )
    (OUT / "task1_report.md").write_text(
        render_report(
            meta,
            repro,
            None,
            summary,
            cvu,
            deltas,
            smoke=smoke,
            segment_info=seg_info,
            grid_text=grid_text,
            audit_trail=trail,
            table_comparison=table_comp,
        ),
        encoding="utf-8",
    )
    print(f"analyze complete; {seg_info['rows']} segments, sha256 {seg_info['sha256']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "stage", choices=["smoke", "extract", "analyze", "grid30", "all"]
    )
    parser.add_argument("--cmaes-root", type=Path, default=DEFAULT_CMAES_ROOT)
    parser.add_argument("--bers-root", type=Path, default=DEFAULT_BERS_ROOT)
    for key, default in ERA5_DEFAULTS.items():
        parser.add_argument(f"--{key.replace('_', '-')}", type=Path, default=default)
    parser.add_argument("--parts-dir", type=Path, default=WORK / "full_weather")
    parser.add_argument("--smoke-dir", type=Path, default=WORK / "full_weather_smoke")
    parser.add_argument("--opt30-dir", type=Path, default=WORK / "full_weather_opt30")
    parser.add_argument("--sampling", choices=["file_a", "opt30"], default="file_a")
    parser.add_argument(
        "--departures", default=None, help="Comma-separated departure timestamps."
    )
    parser.add_argument(
        "--gate-only", action="store_true", help="Stop after the reproduction gate."
    )
    parser.add_argument(
        "--segment-path", type=Path, default=OUT / "task1_segment_inputs.csv.gz"
    )
    parser.add_argument(
        "--corridor", choices=["atlantic", "pacific", "all"], default="all"
    )
    parser.add_argument("--chunk-size", type=int, default=61)
    parser.add_argument("--max-departures", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested stage."""
    args = build_parser().parse_args(argv)
    if args.stage == "extract":
        run_extract(args)
        return 0
    if args.stage == "smoke":
        return run_smoke(args)
    if args.stage == "grid30":
        args.parts_dir = args.opt30_dir
        return run_grid30(args)
    if args.stage == "all":
        for corridor in ("atlantic", "pacific"):
            _spawn_extract(args, corridor)
    return run_analyze(args)


if __name__ == "__main__":
    sys.exit(main())
