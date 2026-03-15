# Evaluation

## Scoring

Submissions are scored on **total energy consumption (MWh)** re-evaluated by the scoring program using the official ERA5 data and the RISE performance model. The re-evaluated energy (not the self-reported value) determines the leaderboard ranking.

If ERA5 reference data is unavailable on the server, the scoring program falls back to self-reported energy values from the participant's CSVs.

### Metrics on the Leaderboard

| Metric | Description | Ranking |
|--------|-------------|---------|
| **Total Energy (MWh)** | Sum of energy across all 8 cases | Primary (ascending) |
| Per-case energy | Breakdown for each of the 8 cases | Secondary |
| Validation errors | Number of format/constraint violations | Lower is better |

### Validation Checks

The scoring program performs the following checks automatically:

#### File A Checks
1. **File presence:** All 8 File A CSVs present with the expected naming pattern.
2. **Column structure:** All required columns (`departure_time_utc`, `arrival_time_utc`, `energy_cons_mwh`, `max_wind_mps`, `max_hs_m`, `sailed_distance_nm`, `details_filename`) present.
3. **Row count:** Each File A has exactly 366 rows.
4. **Datetime format:** All timestamps match `YYYY-MM-DD HH:MM:SS`.
5. **Departure schedule:** Each departure matches the official schedule (2024-01-01 to 2024-12-31, noon UTC).
6. **Passage time:** Arrival − departure equals the expected passage time (354 h for Atlantic, 583 h for Pacific) within ±1 hour tolerance.
7. **Numeric values:** All energy, wind, wave, and distance values are positive and non-NaN.
8. **Operational constraint — wind:** `max_wind_mps` ≤ 20 m/s (optimised cases only).
9. **Operational constraint — waves:** `max_hs_m` ≤ 7 m (optimised cases only).

#### File B Checks
10. **File existence:** Every File B referenced in `details_filename` exists under `tracks/`.
11. **Column structure:** Required columns (`time_utc`, `lat_deg`, `lon_deg`) present.
12. **Minimum waypoints:** At least 2 waypoints per track.
13. **Timestamp ordering:** Waypoint timestamps are strictly increasing.
14. **Coordinate bounds:** Latitudes in [−90, 90], longitudes in [−360, 360].
15. **Start position:** First waypoint within 0.5° of the expected source port.
16. **End position:** Last waypoint within 0.5° of the expected destination port.
17. **Start time:** First waypoint timestamp matches the departure time from File A.
18. **End time:** Last waypoint timestamp matches the arrival time from File A.
19. **Land crossing:** Sampled waypoints checked against a Natural Earth land shapefile (optimised cases only; when available on the server).

#### Cross-Case Checks
20. **WPS consistency:** With-wingsail cases should have ≤ energy compared to without-wingsail cases (warning, not error).

Submissions with validation errors still receive a score, but the error count is displayed on the leaderboard.

> **Note on Great Circle cases:** Checks 8, 9, and 19 (operational constraints and land crossing) are skipped for GC cases (`AGC_*`, `PGC_*`). The great circle path is fixed and participants cannot modify it, so these constraints are not enforceable.

### Disqualification

Submissions may be disqualified if:
- Optimised routes clearly cross land without avoidance.
- Reported energy values are materially inconsistent with the RISE performance model re-evaluation.
- The fixed passage time constraint is violated by more than the tolerance.

## Fair Comparison Guarantee

All participants use:
- **The same ERA5 weather data** (0.25° grid, 6-hourly, 2024)
- **The same RISE performance model** (formulas provided in the **Data** tab)
- **The same route definitions** (ports, passage times, departure schedule)

The only variable is the optimization algorithm. This ensures that differences in the leaderboard reflect genuine algorithmic improvements.
