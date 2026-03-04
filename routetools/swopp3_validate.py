"""SWOPP3 output validation — verify File A, File B and submission compliance.

Currently implemented checks:

- File A columns, formats, no NaN/missing values.
- File B columns and timestamp ordering.
- Naming convention:  ``IEUniversity-{sub}-{case}.csv``.
- Energy sanity: WPS ≤ noWPS, optimised ≤ GC (across case pairs).

Planned / not yet wired:

- File B timestamps consistent with File A arrival/departure.
- Route distance range checks.
- Fixed passage time enforcement.

Example
-------
>>> from routetools.swopp3_validate import validate_file_a, validate_file_b
>>> errors = validate_file_a("output/swopp3/IEUniversity-1-AGC_WPS.csv")
>>> assert not errors, errors
"""

from __future__ import annotations

import csv
import re
from datetime import datetime, timedelta
from pathlib import Path

__all__ = [
    "validate_file_a",
    "validate_file_b",
    "validate_case_pair_wps",
    "validate_case_pair_strategy",
    "validate_submission_dir",
    "ValidationError",
]

_DTFMT = "%Y-%m-%d %H:%M:%S"
_TEAM = "IEUniversity"

_FILE_A_COLUMNS = [
    "departure_time_utc",
    "arrival_time_utc",
    "energy_cons_mwh",
    "max_wind_mps",
    "max_hs_m",
    "sailed_distance_nm",
    "details_filename",
]

_FILE_B_COLUMNS = ["time_utc", "lat_deg", "lon_deg"]

# Expected passage hours by route tag
_PASSAGE_HOURS = {"A": 354, "P": 583}  # Atlantic / Pacific


class ValidationError:
    """A single validation issue."""

    def __init__(self, file: str, row: int | None, message: str):
        self.file = file
        self.row = row
        self.message = message

    def __repr__(self) -> str:
        loc = f"row {self.row}" if self.row is not None else "file"
        return f"ValidationError({self.file}, {loc}: {self.message})"


# ---------------------------------------------------------------------------
# File A validation
# ---------------------------------------------------------------------------
def validate_file_a(
    path: str | Path,
    expected_rows: int = 366,
) -> list[ValidationError]:
    """Validate a File A CSV.

    Parameters
    ----------
    path : str or Path
        Path to the File A CSV.
    expected_rows : int
        Expected number of data rows (default 366).

    Returns
    -------
    list[ValidationError]
        Empty list if valid.
    """
    path = Path(path)
    errors: list[ValidationError] = []
    fname = path.name

    # ---- Naming convention ----
    pattern = re.compile(rf"^{_TEAM}-\d+-\w+\.csv$")
    if not pattern.match(fname):
        errors.append(ValidationError(fname, None, f"Name '{fname}' doesn't match pattern"))

    if not path.exists():
        errors.append(ValidationError(fname, None, "File not found"))
        return errors

    with path.open() as f:
        reader = csv.DictReader(f)

        # ---- Columns ----
        if reader.fieldnames is None:
            errors.append(ValidationError(fname, None, "No header row"))
            return errors

        missing = set(_FILE_A_COLUMNS) - set(reader.fieldnames)
        extra = set(reader.fieldnames) - set(_FILE_A_COLUMNS)
        if missing:
            errors.append(ValidationError(fname, None, f"Missing columns: {missing}"))
        if extra:
            errors.append(ValidationError(fname, None, f"Extra columns: {extra}"))

        rows = list(reader)

    # ---- Row count ----
    if len(rows) != expected_rows:
        errors.append(
            ValidationError(fname, None, f"Expected {expected_rows} rows, got {len(rows)}")
        )

    for i, row in enumerate(rows, 1):
        # ---- No empty values ----
        for col in _FILE_A_COLUMNS:
            val = row.get(col, "")
            if not val.strip():
                errors.append(ValidationError(fname, i, f"Empty value in '{col}'"))

        # ---- Datetime format ----
        for col in ("departure_time_utc", "arrival_time_utc"):
            try:
                datetime.strptime(row[col], _DTFMT)
            except (ValueError, KeyError):
                errors.append(ValidationError(fname, i, f"Bad datetime in '{col}': {row.get(col)}"))

        # ---- Numeric fields ----
        for col in ("energy_cons_mwh", "max_wind_mps", "max_hs_m", "sailed_distance_nm"):
            try:
                v = float(row[col])
                if v != v:  # NaN check
                    errors.append(ValidationError(fname, i, f"NaN in '{col}'"))
                if v < 0:
                    errors.append(ValidationError(fname, i, f"Negative value in '{col}': {v}"))
            except (ValueError, KeyError):
                errors.append(ValidationError(fname, i, f"Non-numeric '{col}': {row.get(col)}"))

        # ---- Passage time vs arrival - departure ----
        try:
            dep = datetime.strptime(row["departure_time_utc"], _DTFMT)
            arr = datetime.strptime(row["arrival_time_utc"], _DTFMT)
            passage_h = (arr - dep).total_seconds() / 3600
            if passage_h <= 0:
                errors.append(ValidationError(fname, i, f"Arrival before departure"))
        except (ValueError, KeyError):
            pass  # already reported above

        # ---- details_filename ----
        details = row.get("details_filename", "")
        if details and not details.endswith(".csv"):
            errors.append(ValidationError(fname, i, f"details_filename not .csv: {details}"))

    return errors


# ---------------------------------------------------------------------------
# File B validation
# ---------------------------------------------------------------------------
def validate_file_b(
    path: str | Path,
    min_waypoints: int = 2,
) -> list[ValidationError]:
    """Validate a File B (track) CSV.

    Parameters
    ----------
    path : str or Path
        Path to the File B CSV.
    min_waypoints : int
        Minimum number of waypoints (default 2).

    Returns
    -------
    list[ValidationError]
        Empty list if valid.
    """
    path = Path(path)
    errors: list[ValidationError] = []
    fname = path.name

    if not path.exists():
        errors.append(ValidationError(fname, None, "File not found"))
        return errors

    with path.open() as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            errors.append(ValidationError(fname, None, "No header row"))
            return errors

        missing = set(_FILE_B_COLUMNS) - set(reader.fieldnames)
        if missing:
            errors.append(ValidationError(fname, None, f"Missing columns: {missing}"))

        rows = list(reader)

    if len(rows) < min_waypoints:
        errors.append(
            ValidationError(fname, None, f"Only {len(rows)} waypoints (min {min_waypoints})")
        )

    prev_time = None
    for i, row in enumerate(rows, 1):
        # ---- Datetime ----
        try:
            t = datetime.strptime(row["time_utc"], _DTFMT)
            if prev_time is not None and t <= prev_time:
                errors.append(ValidationError(fname, i, "Timestamps not strictly increasing"))
            prev_time = t
        except (ValueError, KeyError):
            errors.append(ValidationError(fname, i, f"Bad time_utc: {row.get('time_utc')}"))

        # ---- Coordinates ----
        for col, lo, hi in [("lat_deg", -90, 90), ("lon_deg", -360, 360)]:
            try:
                v = float(row[col])
                if v != v:
                    errors.append(ValidationError(fname, i, f"NaN in '{col}'"))
                elif v < lo or v > hi:
                    errors.append(ValidationError(fname, i, f"'{col}'={v} out of range [{lo},{hi}]"))
            except (ValueError, KeyError):
                errors.append(ValidationError(fname, i, f"Non-numeric '{col}': {row.get(col)}"))

    return errors


# ---------------------------------------------------------------------------
# Cross-case comparisons
# ---------------------------------------------------------------------------
def _load_energies(path: Path) -> list[float]:
    """Load energy_cons_mwh column from a File A CSV.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    KeyError
        If the ``energy_cons_mwh`` column is missing.
    ValueError
        If a cell value is not numeric.
    """
    with path.open() as f:
        reader = csv.DictReader(f)
        return [float(row["energy_cons_mwh"]) for row in reader]


def validate_case_pair_wps(
    wps_path: str | Path,
    nowps_path: str | Path,
) -> list[ValidationError]:
    """Check that WPS energy ≤ noWPS energy for each departure.

    Returns
    -------
    list[ValidationError]
        One error per departure where WPS > noWPS.
    """
    errors: list[ValidationError] = []
    try:
        wps_e = _load_energies(Path(wps_path))
        nowps_e = _load_energies(Path(nowps_path))
    except (FileNotFoundError, KeyError, ValueError) as exc:
        errors.append(ValidationError(str(wps_path), None, f"Cannot load energies: {exc}"))
        return errors
    n = min(len(wps_e), len(nowps_e))
    violations = 0
    for i in range(n):
        if wps_e[i] > nowps_e[i] + 1e-6:
            violations += 1
    if violations:
        errors.append(
            ValidationError(
                f"{Path(wps_path).name} vs {Path(nowps_path).name}",
                None,
                f"WPS energy > noWPS in {violations}/{n} departures",
            )
        )
    return errors


def validate_case_pair_strategy(
    opt_path: str | Path,
    gc_path: str | Path,
) -> list[ValidationError]:
    """Check that optimised energy ≤ GC energy for most departures.

    We allow up to 10% of departures where optimised is worse (stochastic
    optimisation may not always beat the baseline).

    Returns
    -------
    list[ValidationError]
        Error if too many departures show optimised > GC.
    """
    errors: list[ValidationError] = []
    try:
        opt_e = _load_energies(Path(opt_path))
        gc_e = _load_energies(Path(gc_path))
    except (FileNotFoundError, KeyError, ValueError) as exc:
        errors.append(ValidationError(str(opt_path), None, f"Cannot load energies: {exc}"))
        return errors
    n = min(len(opt_e), len(gc_e))
    worse = sum(1 for i in range(n) if opt_e[i] > gc_e[i] + 1e-6)
    pct = worse / n * 100 if n else 0
    if pct > 10:
        errors.append(
            ValidationError(
                f"{Path(opt_path).name} vs {Path(gc_path).name}",
                None,
                f"Optimised worse than GC in {worse}/{n} ({pct:.1f}%) departures",
            )
        )
    return errors


# ---------------------------------------------------------------------------
# Full submission directory validation
# ---------------------------------------------------------------------------
def validate_submission_dir(
    output_dir: str | Path,
    submission: int = 1,
    expected_departures: int = 366,
    verbose: bool = True,
) -> list[ValidationError]:
    """Validate all files in a submission directory.

    Parameters
    ----------
    output_dir : str or Path
        Directory containing File A CSVs and ``tracks/`` subdirectory.
    submission : int
        Submission number.
    expected_departures : int
        Expected departures per case.
    verbose : bool
        Print progress.

    Returns
    -------
    list[ValidationError]
        All errors found.
    """
    output_dir = Path(output_dir)
    errors: list[ValidationError] = []

    case_names = [
        "AO_WPS", "AO_noWPS", "AGC_WPS", "AGC_noWPS",
        "PO_WPS", "PO_noWPS", "PGC_WPS", "PGC_noWPS",
    ]

    # ---- File A ----
    for cname in case_names:
        fa = output_dir / f"{_TEAM}-{submission}-{cname}.csv"
        if verbose:
            print(f"Validating File A: {fa.name} ... ", end="")
        errs = validate_file_a(fa, expected_rows=expected_departures)
        errors.extend(errs)
        if verbose:
            print(f"{'FAIL' if errs else 'OK'} ({len(errs)} issues)")

        # ---- File B for each departure ----
        if fa.exists():
            with fa.open() as f:
                reader = csv.DictReader(f)
                for row_i, row in enumerate(reader, 1):
                    fb_name = row.get("details_filename", "")
                    if fb_name:
                        fb_path = output_dir / "tracks" / fb_name
                        fb_errs = validate_file_b(fb_path)
                        errors.extend(fb_errs)

    # ---- WPS vs noWPS comparisons ----
    pairs_wps = [
        ("AO_WPS", "AO_noWPS"),
        ("AGC_WPS", "AGC_noWPS"),
        ("PO_WPS", "PO_noWPS"),
        ("PGC_WPS", "PGC_noWPS"),
    ]
    for wps_case, nowps_case in pairs_wps:
        wps_fa = output_dir / f"{_TEAM}-{submission}-{wps_case}.csv"
        nowps_fa = output_dir / f"{_TEAM}-{submission}-{nowps_case}.csv"
        if wps_fa.exists() and nowps_fa.exists():
            if verbose:
                print(f"Comparing {wps_case} vs {nowps_case} ... ", end="")
            errs = validate_case_pair_wps(wps_fa, nowps_fa)
            errors.extend(errs)
            if verbose:
                print(f"{'FAIL' if errs else 'OK'}")

    # ---- Optimised vs GC comparisons ----
    pairs_strategy = [
        ("AO_WPS", "AGC_WPS"),
        ("AO_noWPS", "AGC_noWPS"),
        ("PO_WPS", "PGC_WPS"),
        ("PO_noWPS", "PGC_noWPS"),
    ]
    for opt_case, gc_case in pairs_strategy:
        opt_fa = output_dir / f"{_TEAM}-{submission}-{opt_case}.csv"
        gc_fa = output_dir / f"{_TEAM}-{submission}-{gc_case}.csv"
        if opt_fa.exists() and gc_fa.exists():
            if verbose:
                print(f"Comparing {opt_case} vs {gc_case} ... ", end="")
            errs = validate_case_pair_strategy(opt_fa, gc_fa)
            errors.extend(errs)
            if verbose:
                print(f"{'FAIL' if errs else 'OK'}")

    # ---- Summary ----
    if verbose:
        print(f"\n{'='*50}")
        if errors:
            print(f"VALIDATION FAILED: {len(errors)} issue(s)")
            for e in errors[:20]:
                print(f"  {e}")
            if len(errors) > 20:
                print(f"  ... and {len(errors) - 20} more")
        else:
            print("VALIDATION PASSED: all checks OK")

    return errors
