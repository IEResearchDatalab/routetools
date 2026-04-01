"""Count Codabench-style route violations for SWOPP3 output folders.

This module reproduces the user-facing violation counting convention used for
local analysis of SWOPP3 outputs.

By default, great-circle (GC) cases are excluded from the report entirely.
Set ``include_gc=True`` to include them:

- wind violations: number of File A rows with ``max_wind_mps > 20``
- wave violations: number of File A rows with ``max_hs_m > 7``
- land violations: number of sampled File B waypoints on land, using the same
    waypoint subsampling rule as the Codabench scorer
    (``step = max(1, len(waypoints) // 50)``)

The resulting totals match the expected 2087 count for the
``output/cmaes_weather`` folder in this repository.

When ERA5 resources are supplied, the same pass can also accumulate the smooth
wind and wave penalties used by the optimisation code. That keeps the local
report aligned with both the threshold-based Codabench counts and the softer
objective-level penalty signal.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

import jax.numpy as jnp

from routetools.era5.loader import (
    load_dataset_epoch,
    load_era5_wavefield,
    load_era5_windfield,
)
from routetools.swopp3 import SWOPP3_CASES
from routetools.weather import (
    DEFAULT_HS_LIMIT,
    DEFAULT_TWS_LIMIT,
    wave_penalty_smooth,
    wind_penalty_smooth,
)

_DTFMT = "%Y-%m-%d %H:%M:%S"
_ERA5_FILE_RE = r"^(?P<prefix>era5_[^_]+_[^_]+_)(?P<year>\d{4})(?:_(?P<suffix>\d{2}(?:-\d{2})?))?\.nc$"  # noqa: E501

CASE_ORDER = [
    "AO_WPS",
    "AO_noWPS",
    "AGC_WPS",
    "AGC_noWPS",
    "PO_WPS",
    "PO_noWPS",
    "PGC_WPS",
    "PGC_noWPS",
]


class LandChecker(Protocol):
    """Callable protocol for land checks."""

    def __call__(self, lat: float, lon: float) -> bool:  # noqa: D102
        ...


@dataclass(frozen=True)
class ScenarioViolationCounts:
    """Violation counts for one SWOPP3 scenario.

    Parameters
    ----------
    folder : str
        Name of the analysed output folder.
    case_id : str
        SWOPP3 case identifier.
    wind_violations : int
        Number of File A rows exceeding the wind threshold.
    wave_violations : int
        Number of File A rows exceeding the wave threshold.
    land_violations : int
        Number of sampled File B waypoints on land.
    """

    folder: str
    case_id: str
    wind_violations: int
    wave_violations: int
    land_violations: int
    wind_penalty: float = 0.0
    wave_penalty: float = 0.0

    @property
    def total_violations(self) -> int:
        """Return the total violations for this scenario."""
        return self.wind_violations + self.wave_violations + self.land_violations

    @property
    def total_penalty(self) -> float:
        """Return the total weather penalty for this scenario."""
        return self.wind_penalty + self.wave_penalty


@dataclass(frozen=True)
class CorridorWeatherResources:
    """Loaded weather resources for one corridor."""

    dataset_epoch: datetime
    windfield: object
    wavefield: object


def is_gc_case(case_id: str) -> bool:
    """Return whether a SWOPP3 case is a great-circle case."""
    return str(SWOPP3_CASES[case_id]["strategy"]) == "gc"


def find_team_prefix(input_dir: Path) -> str:
    """Detect the team prefix from submitted File A CSVs.

    Parameters
    ----------
    input_dir : Path
        SWOPP3 output directory containing File A CSV files.

    Returns
    -------
    str
        Team prefix including the submission id, for example ``IEUniversity-1``.

    Raises
    ------
    FileNotFoundError
        If no SWOPP3 File A CSV files are found.
    """
    for path in sorted(input_dir.glob("*.csv")):
        for case_id in CASE_ORDER:
            suffix = f"-{case_id}.csv"
            if path.name.endswith(suffix):
                return path.name[: -len(suffix)]
    raise FileNotFoundError(f"No SWOPP3 File A CSV files found in {input_dir}")


def load_default_land_checker(shapefile_path: Path | None = None) -> LandChecker:
    """Load the Natural Earth 10m land checker used for local validation.

    Parameters
    ----------
    shapefile_path : Path, optional
        Explicit path to a Natural Earth land shapefile. When omitted, the
        default Cartopy-managed ``ne_10m_land.shp`` is used.

    Returns
    -------
    LandChecker
        Callable ``(lat, lon) -> bool``.
    """
    if shapefile_path is None:
        import cartopy.io.shapereader as shpreader

        shapefile_path = Path(
            shpreader.natural_earth(
                resolution="10m",
                category="physical",
                name="land",
            )
        )

    import shapefile as shp
    from shapely.geometry import Point, shape
    from shapely.ops import unary_union

    reader = shp.Reader(str(shapefile_path))
    land_union = unary_union([shape(record) for record in reader.shapes()])

    def is_on_land(lat: float, lon: float) -> bool:
        return bool(land_union.contains(Point(lon, lat)))

    return is_on_land


def _loadable_era5_paths(path: Path) -> list[Path]:
    """Return the base ERA5 file plus any next-year continuation files."""
    import re

    match = re.match(_ERA5_FILE_RE, path.name)
    if match is None:
        return [path]

    prefix = match.group("prefix")
    next_year = int(match.group("year")) + 1
    exact_next_year = path.with_name(f"{prefix}{next_year}.nc")
    if exact_next_year.exists():
        return [path, exact_next_year]

    continuation_paths = sorted(path.parent.glob(f"{prefix}{next_year}_*.nc"))
    return [path, *continuation_paths]


def load_default_weather_resources(
    *,
    wind_path_atlantic: Path = Path("data/era5/era5_wind_atlantic_2024.nc"),
    wave_path_atlantic: Path = Path("data/era5/era5_waves_atlantic_2024.nc"),
    wind_path_pacific: Path = Path("data/era5/era5_wind_pacific_2024.nc"),
    wave_path_pacific: Path = Path("data/era5/era5_waves_pacific_2024.nc"),
) -> dict[str, CorridorWeatherResources]:
    """Load default ERA5 weather resources for both SWOPP3 corridors.

    Each input may resolve to a single annual file or to an annual file plus a
    next-year continuation file when the dataset has been split on disk.
    """
    path_map = {
        "atlantic": (wind_path_atlantic, wave_path_atlantic),
        "pacific": (wind_path_pacific, wave_path_pacific),
    }
    resources: dict[str, CorridorWeatherResources] = {}
    for corridor, (wind_path, wave_path) in path_map.items():
        wind_paths = _loadable_era5_paths(wind_path)
        wave_paths = _loadable_era5_paths(wave_path)
        wind_target = wind_paths if len(wind_paths) > 1 else wind_paths[0]
        wave_target = wave_paths if len(wave_paths) > 1 else wave_paths[0]
        resources[corridor] = CorridorWeatherResources(
            dataset_epoch=load_dataset_epoch(wind_target),
            windfield=load_era5_windfield(wind_target),
            wavefield=load_era5_wavefield(wave_target),
        )
    return resources


def read_track_curve(track_path: Path) -> jnp.ndarray:
    """Read a track CSV into a ``(L, 2)`` ``(lon, lat)`` array."""
    lons: list[float] = []
    lats: list[float] = []
    with track_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            lats.append(float(row["lat_deg"]))
            lons.append(float(row["lon_deg"]))
    return jnp.stack(
        [jnp.asarray(lons, dtype=jnp.float32), jnp.asarray(lats, dtype=jnp.float32)],
        axis=1,
    )


def departure_offset_hours(departure: datetime, dataset_epoch: datetime) -> float:
    """Return departure offset in hours relative to the dataset epoch."""
    departure_naive = departure.replace(tzinfo=None) if departure.tzinfo else departure
    epoch_naive = (
        dataset_epoch.replace(tzinfo=None)
        if getattr(dataset_epoch, "tzinfo", None)
        else dataset_epoch
    )
    return (departure_naive - epoch_naive).total_seconds() / 3600.0


def count_land_violations(track_path: Path, land_checker: LandChecker) -> int:
    """Count sampled File B waypoints on land.

    Uses the same subsampling rule as the Codabench scorer:
    ``step = max(1, len(waypoints) // 50)`` and then checks ``waypoints[::step]``.

    Parameters
    ----------
    track_path : Path
        Track CSV path.
    land_checker : LandChecker
        Callable ``(lat, lon) -> bool``.

    Returns
    -------
    int
        Number of sampled waypoints on land.
    """
    with track_path.open(newline="") as handle:
        waypoints = list(csv.DictReader(handle))

    step = max(1, len(waypoints) // 50)
    violations = 0
    for waypoint in waypoints[::step]:
        lat = float(waypoint["lat_deg"])
        lon = float(waypoint["lon_deg"])
        violations += int(land_checker(lat, lon))
    return violations


def count_summary_weather_violations(
    summary_path: Path, case_id: str
) -> tuple[int, int]:
    """Count wind and wave threshold violations from one File A CSV.

    Parameters
    ----------
    summary_path : Path
        File A summary CSV.
    case_id : str
        SWOPP3 case identifier.

    Returns
    -------
    tuple[int, int]
        ``(wind_violations, wave_violations)``.
    """
    wind_violations = 0
    wave_violations = 0
    with summary_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            wind_violations += int(float(row["max_wind_mps"]) > DEFAULT_TWS_LIMIT)
            wave_violations += int(float(row["max_hs_m"]) > DEFAULT_HS_LIMIT)
    return wind_violations, wave_violations


def count_folder_violations(
    input_dir: Path,
    *,
    land_checker: LandChecker,
    weather_resources: dict[str, CorridorWeatherResources] | None = None,
    wind_penalty_weight: float = 1000.0,
    wave_penalty_weight: float = 1000.0,
    include_gc: bool = False,
) -> list[ScenarioViolationCounts]:
    """Count violations for every SWOPP3 scenario in an output folder.

    Parameters
    ----------
    input_dir : Path
        SWOPP3 output directory.
    land_checker : LandChecker
        Callable ``(lat, lon) -> bool``.
    weather_resources : dict[str, CorridorWeatherResources], optional
        Preloaded ERA5 resources keyed by corridor. When provided, smooth wind
        and wave penalties are accumulated alongside the threshold counts.
    include_gc : bool, optional
        When ``False`` (default), omit great-circle cases from the report.

    Returns
    -------
    list[ScenarioViolationCounts]
        One row per scenario in standard SWOPP3 order.
    """
    input_dir = Path(input_dir)
    team_prefix = find_team_prefix(input_dir)
    tracks_dir = input_dir / "tracks"
    rows: list[ScenarioViolationCounts] = []

    for case_id in CASE_ORDER:
        if not include_gc and is_gc_case(case_id):
            continue

        summary_path = input_dir / f"{team_prefix}-{case_id}.csv"
        if not summary_path.exists():
            continue

        case = SWOPP3_CASES[case_id]
        corridor = str(case["route"])
        passage_hours = float(case["passage_hours"])

        wind_violations, wave_violations = count_summary_weather_violations(
            summary_path,
            case_id,
        )

        land_violations = 0
        wind_penalty = 0.0
        wave_penalty = 0.0
        with summary_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                track_path = tracks_dir / row["details_filename"]
                land_violations += count_land_violations(track_path, land_checker)

                if weather_resources is not None:
                    # Reuse the same track sample for the soft penalties so the
                    # weather totals line up with the voyage-level threshold counts.
                    departure = datetime.strptime(row["departure_time_utc"], _DTFMT)
                    curve = read_track_curve(track_path)[None, ...]
                    resources = weather_resources[corridor]
                    time_offset = departure_offset_hours(
                        departure,
                        resources.dataset_epoch,
                    )
                    wind_penalty += float(
                        wind_penalty_smooth(
                            curve,
                            resources.windfield,
                            weight=wind_penalty_weight,
                            travel_time=passage_hours,
                            time_offset=time_offset,
                        )[0]
                    )
                    wave_penalty += float(
                        wave_penalty_smooth(
                            curve,
                            resources.wavefield,
                            weight=wave_penalty_weight,
                            travel_time=passage_hours,
                            time_offset=time_offset,
                        )[0]
                    )

        rows.append(
            ScenarioViolationCounts(
                folder=input_dir.name,
                case_id=case_id,
                wind_violations=wind_violations,
                wave_violations=wave_violations,
                land_violations=land_violations,
                wind_penalty=wind_penalty,
                wave_penalty=wave_penalty,
            )
        )

    return rows


def format_violation_table(rows: list[ScenarioViolationCounts]) -> str:
    """Format scenario counts and penalties as a plain-text table.

    Parameters
    ----------
    rows : list[ScenarioViolationCounts]
        Scenario rows to render.

    Returns
    -------
    str
        Human-readable table including a total row.
    """
    header = (
        f"{'Folder':<18} {'Case':<10} {'Wind':>6} {'Wave':>6} {'Land':>6} {'Total':>6} "
        f"{'Wind Pen':>12} {'Wave Pen':>12} {'Total Pen':>12}"
    )
    lines = [header, "-" * len(header)]

    total_wind = 0
    total_wave = 0
    total_land = 0
    total = 0
    total_wind_penalty = 0.0
    total_wave_penalty = 0.0
    total_penalty = 0.0
    for row in rows:
        lines.append(
            f"{row.folder:<18} {row.case_id:<10} {row.wind_violations:>6} "
            f"{row.wave_violations:>6} {row.land_violations:>6} "
            f"{row.total_violations:>6} {row.wind_penalty:>12.3f} "
            f"{row.wave_penalty:>12.3f} {row.total_penalty:>12.3f}"
        )
        total_wind += row.wind_violations
        total_wave += row.wave_violations
        total_land += row.land_violations
        total += row.total_violations
        total_wind_penalty += row.wind_penalty
        total_wave_penalty += row.wave_penalty
        total_penalty += row.total_penalty

    lines.append("-" * len(header))
    lines.append(
        f"{'TOTAL':<18} {'':<10} {total_wind:>6} {total_wave:>6} "
        f"{total_land:>6} {total:>6} {total_wind_penalty:>12.3f} "
        f"{total_wave_penalty:>12.3f} {total_penalty:>12.3f}"
    )
    return "\n".join(lines)


def format_grouped_violation_table(rows: list[ScenarioViolationCounts]) -> str:
    """Format rows grouped by case with folders stacked together."""
    folder_order = sorted({row.folder for row in rows})
    row_map = {(row.case_id, row.folder): row for row in rows}
    header = (
        f"{'Folder':<18} {'Case':<10} {'Wind':>6} {'Wave':>6} {'Land':>6} {'Total':>6} "
        f"{'Wind Pen':>12} {'Wave Pen':>12} {'Total Pen':>12}"
    )
    lines = [header, "-" * len(header)]

    for case_id in CASE_ORDER:
        printed = False
        for folder in folder_order:
            row = row_map.get((case_id, folder))
            if row is None:
                continue
            lines.append(
                f"{row.folder:<18} {row.case_id:<10} {row.wind_violations:>6} "
                f"{row.wave_violations:>6} {row.land_violations:>6} "
                f"{row.total_violations:>6} {row.wind_penalty:>12.3f} "
                f"{row.wave_penalty:>12.3f} {row.total_penalty:>12.3f}"
            )
            printed = True
        if printed:
            lines.append("-" * len(header))

    # Append one total row per folder so side-by-side experiment comparisons can
    # be copied directly into spreadsheets or review comments.
    totals_by_folder: dict[str, ScenarioViolationCounts] = {}
    for folder in folder_order:
        folder_rows = [row for row in rows if row.folder == folder]
        totals_by_folder[folder] = ScenarioViolationCounts(
            folder=folder,
            case_id="TOTAL",
            wind_violations=sum(row.wind_violations for row in folder_rows),
            wave_violations=sum(row.wave_violations for row in folder_rows),
            land_violations=sum(row.land_violations for row in folder_rows),
            wind_penalty=sum(row.wind_penalty for row in folder_rows),
            wave_penalty=sum(row.wave_penalty for row in folder_rows),
        )

    for folder in folder_order:
        row = totals_by_folder[folder]
        lines.append(
            f"{row.folder:<18} {row.case_id:<10} {row.wind_violations:>6} "
            f"{row.wave_violations:>6} {row.land_violations:>6} "
            f"{row.total_violations:>6} {row.wind_penalty:>12.3f} "
            f"{row.wave_penalty:>12.3f} {row.total_penalty:>12.3f}"
        )
    return "\n".join(lines)


def grouped_violation_rows(
    rows: list[ScenarioViolationCounts],
) -> list[dict[str, str | int | float]]:
    """Return grouped rows ready to be written as CSV.

    Parameters
    ----------
    rows : list[ScenarioViolationCounts]
        Scenario rows to export.

    Returns
    -------
    list[dict[str, str | int | float]]
        Export rows in grouped case order with per-folder totals appended.
    """
    folder_order = sorted({row.folder for row in rows})
    row_map = {(row.case_id, row.folder): row for row in rows}
    export_rows: list[dict[str, str | int | float]] = []

    for case_id in CASE_ORDER:
        for folder in folder_order:
            row = row_map.get((case_id, folder))
            if row is None:
                continue
            export_rows.append(
                {
                    "folder": row.folder,
                    "case_id": row.case_id,
                    "wind_violations": row.wind_violations,
                    "wave_violations": row.wave_violations,
                    "land_violations": row.land_violations,
                    "total_violations": row.total_violations,
                    "wind_penalty": row.wind_penalty,
                    "wave_penalty": row.wave_penalty,
                    "total_penalty": row.total_penalty,
                }
            )

    for folder in folder_order:
        folder_rows = [row for row in rows if row.folder == folder]
        total_row = ScenarioViolationCounts(
            folder=folder,
            case_id="TOTAL",
            wind_violations=sum(row.wind_violations for row in folder_rows),
            wave_violations=sum(row.wave_violations for row in folder_rows),
            land_violations=sum(row.land_violations for row in folder_rows),
            wind_penalty=sum(row.wind_penalty for row in folder_rows),
            wave_penalty=sum(row.wave_penalty for row in folder_rows),
        )
        export_rows.append(
            {
                "folder": total_row.folder,
                "case_id": total_row.case_id,
                "wind_violations": total_row.wind_violations,
                "wave_violations": total_row.wave_violations,
                "land_violations": total_row.land_violations,
                "total_violations": total_row.total_violations,
                "wind_penalty": total_row.wind_penalty,
                "wave_penalty": total_row.wave_penalty,
                "total_penalty": total_row.total_penalty,
            }
        )

    return export_rows


def write_grouped_violation_csv(
    rows: list[ScenarioViolationCounts],
    output_path: Path,
) -> Path:
    """Write grouped violation rows to CSV.

    Parameters
    ----------
    rows : list[ScenarioViolationCounts]
        Scenario rows to export.
    output_path : Path
        CSV destination.

    Returns
    -------
    Path
        Written CSV path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "folder",
        "case_id",
        "wind_violations",
        "wave_violations",
        "land_violations",
        "total_violations",
        "wind_penalty",
        "wave_penalty",
        "total_penalty",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(grouped_violation_rows(rows))
    return output_path
