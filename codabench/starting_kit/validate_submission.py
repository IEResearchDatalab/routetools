#!/usr/bin/env python3
"""Validate a SWOPP3 submission directory or ZIP before uploading it."""

from __future__ import annotations

import argparse
import csv
import sys
import tempfile
import zipfile
from pathlib import Path

CASE_NAMES = (
    "AO_WPS",
    "AO_noWPS",
    "AGC_WPS",
    "AGC_noWPS",
    "PO_WPS",
    "PO_noWPS",
    "PGC_WPS",
    "PGC_noWPS",
)
FILE_A_COLUMNS = {
    "departure_time_utc",
    "arrival_time_utc",
    "energy_cons_mwh",
    "max_wind_mps",
    "max_hs_m",
    "sailed_distance_nm",
    "details_filename",
}
EXPECTED_DEPARTURES = 366


def find_team_prefix(submission_dir: Path) -> str | None:
    """Return the File-A filename prefix found at the submission root."""
    for file_path in submission_dir.glob("*.csv"):
        for case_name in CASE_NAMES:
            suffix = f"-{case_name}.csv"
            if file_path.name.endswith(suffix):
                return file_path.name.removesuffix(suffix)
    return None


def validate_submission(submission_dir: Path) -> list[str]:
    """Return structural errors for an unpacked submission directory."""
    errors: list[str] = []
    team_prefix = find_team_prefix(submission_dir)
    if team_prefix is None:
        nested_dirs = sorted(
            entry.name
            for entry in submission_dir.iterdir()
            if entry.is_dir() and any(entry.glob("*.csv"))
        )
        errors.append(
            "No File-A CSV was found at the submission root. "
            "Create the ZIP from inside your submission directory."
        )
        if nested_dirs:
            errors.append(
                "File-A CSVs were found inside nested director"
                f"y/directories: {nested_dirs}."
            )
        return errors

    tracks_dir = submission_dir / "tracks"
    if not tracks_dir.is_dir():
        errors.append("Missing required tracks/ directory at the submission root.")

    for case_name in CASE_NAMES:
        file_a_path = submission_dir / f"{team_prefix}-{case_name}.csv"
        if not file_a_path.is_file():
            errors.append(f"Missing File A: {file_a_path.name}")
            continue

        with file_a_path.open(newline="") as file_a:
            reader = csv.DictReader(file_a)
            fieldnames = set(reader.fieldnames or ())
            missing_columns = FILE_A_COLUMNS - fieldnames
            if missing_columns:
                errors.append(
                    f"{file_a_path.name}: missing columns {sorted(missing_columns)}"
                )
                continue
            rows = list(reader)

        if len(rows) != EXPECTED_DEPARTURES:
            errors.append(
                f"{file_a_path.name}: expected {EXPECTED_DEPARTURES} rows, "
                f"found {len(rows)}"
            )

        if not tracks_dir.is_dir():
            continue
        missing_tracks = 0
        invalid_track_names = 0
        for row in rows:
            track_name = row["details_filename"]
            if Path(track_name).name != track_name:
                invalid_track_names += 1
            elif not (tracks_dir / track_name).is_file():
                missing_tracks += 1
        if invalid_track_names:
            errors.append(
                f"{file_a_path.name}: {invalid_track_names} details_filename value(s) "
                "must be bare filenames stored in tracks/."
            )
        if missing_tracks:
            errors.append(
                f"{file_a_path.name}: {missing_tracks} referenced File-B track(s) "
                "are missing from tracks/."
            )

    return errors


def extract_archive(archive_path: Path, destination: Path) -> None:
    """Extract an archive after rejecting paths that escape *destination*."""
    destination_root = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            member_path = (destination / member.filename).resolve()
            if not member_path.is_relative_to(destination_root):
                raise ValueError(f"Archive member escapes destination: {member.filename}")
        archive.extractall(destination)


def main() -> int:
    """Validate the requested submission directory or ZIP archive."""
    parser = argparse.ArgumentParser(
        description="Validate a SWOPP3 submission before uploading it to CodaBench."
    )
    parser.add_argument("submission", type=Path, help="Submission directory or .zip file")
    args = parser.parse_args()

    submission_path = args.submission
    if submission_path.is_dir():
        errors = validate_submission(submission_path)
    elif submission_path.is_file() and zipfile.is_zipfile(submission_path):
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                extract_archive(submission_path, Path(temp_dir))
            except ValueError as error:
                errors = [str(error)]
            else:
                errors = validate_submission(Path(temp_dir))
    else:
        errors = [f"Not a submission directory or ZIP archive: {submission_path}"]

    if errors:
        print("Submission validation failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print("Submission structure is valid and ready to upload.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())