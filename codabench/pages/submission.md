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
2024-01-01 12:00:00,2024-01-16 06:00:00,336.375913,17.3875,7.8464,3022.5591,TeamName-1-AO_WPS-20240101.csv
2024-01-02 12:00:00,2024-01-17 06:00:00,199.891800,17.3153,7.3659,3022.5591,TeamName-1-AO_WPS-20240102.csv
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
2024-01-01 12:00:00,43.4600,-3.8100
2024-01-01 15:32:00,43.5100,-5.2300
...
2024-01-16 06:00:00,40.5300,-73.8000
```

## Important Rules

### Passage Time

- **Fixed passage:** Arrival time must equal departure + passage hours (354 h for Atlantic, 583 h for Pacific), with a tolerance of ±1 hour.

### Departure Schedule

- **All 366 departures required:** One row per day from 2024-01-01 to 2024-12-31, departing at **12:00 UTC**.
- Departure timestamps must match the official schedule exactly.

### Endpoint Positions

- First waypoint in File B must be within **0.5°** of the expected source port.
- Last waypoint in File B must be within **0.5°** of the expected destination port.
- First/last waypoint timestamps must match the departure/arrival times from File A.

### Operational Constraints (Optimised Cases Only)

- **True wind speed:** `max_wind_mps` must be < **20 m/s** along the route.
- **Significant wave height:** `max_hs_m` must be < **7 m** along the route.
- Submissions exceeding these limits receive a validation error per violation.
- **Not enforced for GC cases** — the great circle path is fixed and may traverse severe weather.

### Land Avoidance (Optimised Cases Only)

- Waypoints in File B must not cross land. The scoring program checks a sample of waypoints against a Natural Earth shapefile.
- **Not enforced for GC cases** — the great circle path is fixed and may cross land (e.g. Newfoundland on the Atlantic route).

### Numeric Values

- No negative energy values.
- No NaN values in any numeric column.
- **Timestamps must be strictly increasing** in File B.

### Route Constraints

- **Great Circle cases** (`*GC_*`): The route must follow the great circle path. Only the speed profile may vary.
- **Optimised cases** (`*O_*`): Both route and speed may be optimised freely.

### WPS Consistency

- With wingsails (`*_WPS`) routes should consume **less or equal** energy compared to the same route without wingsails (`*_noWPS`). The system flags a warning if this is violated.
