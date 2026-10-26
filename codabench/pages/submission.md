# Submission Format

## What to Submit

Upload a **single `.zip` file** containing:

```
submission.zip
├── TeamName-1-AO_WPS.csv         # File A (Atlantic Optimised, with WPS)
├── TeamName-1-AO_noWPS.csv       # File A (Atlantic Optimised, without WPS)
├── TeamName-1-AGC_WPS.csv        # File A (Atlantic Great Circle, with WPS)
├── TeamName-1-AGC_noWPS.csv      # File A (Atlantic Great Circle, without WPS)
├── TeamName-1-PO_WPS.csv         # File A (Pacific Optimised, with WPS)
├── TeamName-1-PO_noWPS.csv       # File A (Pacific Optimised, without WPS)
├── TeamName-1-PGC_WPS.csv        # File A (Pacific Great Circle, with WPS)
├── TeamName-1-PGC_noWPS.csv      # File A (Pacific Great Circle, without WPS)
└── tracks/
    ├── TeamName-1-AO_WPS-20240101.csv    # File B (waypoints)
    ├── TeamName-1-AO_WPS-20240102.csv
    ├── ...                               # 366 files per case × 8 cases
    └── TeamName-1-PGC_noWPS-20241231.csv
```

Replace `TeamName` with your team name and `1` with your submission number.

## Build and Check the ZIP

The File A CSVs and `tracks/` directory must be at the **root of the ZIP**.
Do not zip the enclosing submission directory.

```bash
python starting_kit.py --output-dir submission --team TeamName --submission 1
python validate_submission.py submission

(cd submission && zip -r ../TeamName-1.zip .)
python validate_submission.py TeamName-1.zip
```

Upload `TeamName-1.zip` only after the validator reports that its structure is
valid. The `validate_submission.py` tool is included with the starting kit and
checks the ZIP root, all File A CSVs, and the File B tracks they reference.

## File A — Energy Summary (one per case)

**Filename:** `TeamName-{submission}-{casename}.csv`

Each file has **366 rows** (one per departure) with these columns:

| Column               | Type     | Description                                     |
| -------------------- | -------- | ----------------------------------------------- |
| `departure_time_utc` | datetime | `YYYY-MM-DD HH:MM:SS` format                    |
| `arrival_time_utc`   | datetime | `YYYY-MM-DD HH:MM:SS` format                    |
| `energy_cons_mwh`    | float    | Total energy consumption in MWh                 |
| `max_wind_mps`       | float    | Maximum true wind speed encountered (m/s)       |
| `max_hs_m`           | float    | Maximum significant wave height encountered (m) |
| `sailed_distance_nm` | float    | Total sailed distance in nautical miles         |
| `details_filename`   | string   | Name of the corresponding File B CSV            |

**Example:**

```csv
departure_time_utc,arrival_time_utc,energy_cons_mwh,max_wind_mps,max_hs_m,sailed_distance_nm,details_filename
2024-01-01 12:00:00,2024-01-16 06:00:00,336.375913,17.3875,6.8464,2891.2207,TeamName-1-AO_WPS-20240101.csv
2024-01-02 12:00:00,2024-01-17 06:00:00,199.891800,17.3153,6.3659,2874.9013,TeamName-1-AO_WPS-20240102.csv
```

## File B — Track Waypoints (one per departure per case)

**Filename:** `TeamName-{submission}-{casename}-{YYYYMMDD}.csv`

| Column     | Type     | Description                                |
| ---------- | -------- | ------------------------------------------ |
| `time_utc` | datetime | `YYYY-MM-DD HH:MM:SS`, strictly increasing |
| `lat_deg`  | float    | Latitude in degrees [-90, 90]              |
| `lon_deg`  | float    | Longitude in degrees [-360, 360]           |

**Example:**

```csv
time_utc,lat_deg,lon_deg
2024-01-01 12:00:00,43.6000,-4.0000
2024-01-01 15:30:00,43.5700,-4.6400
...
2024-01-16 06:00:00,40.6000,-69.0000
```

### How the Scorer Reads a Track

- Consecutive waypoints are joined by the **shorter great-circle arc**, sailed at
  **constant speed** between the two timestamps. The scorer resamples every
  track along these arcs before evaluating energy, constraints and land.
- Place waypoints densely enough that these arcs stay at sea, especially near
  coasts and headlands.
- Longitudes may be given in [−180, 180] or [0, 360). Pacific tracks may cross
  the antimeridian (e.g. from 179.9 to −179.9); this is treated as a short step
  east, not a jump around the globe.

## Important Rules

### Passage Time

- **Fixed passage:** Arrival time must equal departure + passage hours (354 h for Atlantic, 583 h for Pacific), with a tolerance of ±1 hour.

### Departure Schedule

- **All 366 departures required:** One row per day from 2024-01-01 to 2024-12-31, departing at **12:00 UTC**.
- Departure timestamps must match the official schedule exactly.

### Start and End Points

Every track starts and ends at the fixed offshore points below, not at the port
berths:

| Cases     | Start (lat, lon)                 | End (lat, lon)                       |
| --------- | -------------------------------- | ------------------------------------ |
| `A*` (4)  | (43.60, −4.00) Santander approach | (40.60, −69.00) New York approach    |
| `P*` (4)  | (34.80, 140.00) Tokyo approach   | (34.40, −121.00) Los Angeles approach |

- **Position:** the first and last waypoints of File B must each be within
  **0.5°** of these points, measured as $\sqrt{\Delta\text{lat}^2 + \Delta\text{lon}^2}$
  in degrees (longitude difference wrapped to [−180°, 180°]). We recommend
  placing them exactly on the points.
- **Time:** the first waypoint timestamp must equal `departure_time_utc` and the
  last must equal `arrival_time_utc` from File A, to the second.

### Operational Constraints (Optimised Cases Only)

- **True wind speed:** `max_wind_mps` must not exceed **20 m/s** along the route.
- **Significant wave height:** `max_hs_m` must not exceed **7 m** along the route.
- Each File A row reporting a value above these limits counts as a validation error.
  The scorer also recomputes wind and waves from ERA5 along your track and reports
  any exceedances it finds as warnings.
- **Not enforced for GC cases** — the great circle path is fixed and may traverse severe weather.

### Land Avoidance (Optimised Cases Only)

- No part of the track may be on land. The scoring program resamples the track
  every 30 minutes along the great-circle legs and checks every resampled point
  against the Natural Earth 1:10m land polygons included in the data release.
- **Not enforced for GC cases** — the great circle path is fixed.

### Numeric Values

- No negative energy values.
- No NaN values in any numeric column.
- **Timestamps must be strictly increasing** in File B.

### Route Constraints

- **Great Circle cases** (`*GC_*`): The route must follow the great circle path. Only the speed profile may vary.
- **Optimised cases** (`*O_*`): Both route and speed may be optimised freely.

### WPS Consistency

- With wingsails (`*_WPS`) routes should consume **less or equal** energy compared to the same route without wingsails (`*_noWPS`). The system flags a warning if this is violated.
