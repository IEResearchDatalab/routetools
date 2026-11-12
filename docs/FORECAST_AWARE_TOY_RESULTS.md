# Forecast-aware toy results

The first comparison run completed on 2026-11-10 using commit `ead1e4d` of
branch `codex/forecast-aware-toy`.

## Configuration

- Route/case: Atlantic Santander–New York, `AO_WPS` (354 hours).
- Departures: 2025-07-15 through 2025-07-24 at 00 UTC.
- Forecast: deterministic ECMWF IFS `oper` + `wave`, initialized at departure.
- Truth: hourly ERA5 reanalysis.
- Optimizer: seed 42, 120 route points, 10 Bézier control points, CMA-ES
  population 50 and 1,000 maximum evaluations, followed by FMS with patience
  15 and 500 maximum evaluations.
- Each forecast-optimized and ERA5-optimized route was scored on both weather
  products. The same optimizer settings and seed were used on both sides.

## Aggregate result

Across the ten departures, the forecast-optimized route consumed a mean
145.288 MWh when scored on ERA5. The ERA5-optimized perfect-information
reference consumed 130.723 MWh. The mean gap was therefore 14.566 MWh, or
11.641%; the median gap was 13.426 MWh, or 11.610%.

All ten gaps were positive. They ranged from 4.162% to 20.997%. Mean forecast
error on the forecast route itself was -1.395 MWh (-0.666%), although positive
and negative daily errors cancel in that mean. The two optimizations consumed
1,453.7 seconds of reported optimizer time in total.

| Departure | Forecast route on ERA5 (MWh) | ERA5 route on ERA5 (MWh) | Gap (MWh) | Gap (%) |
|---|---:|---:|---:|---:|
| 2025-07-15 | 175.555 | 146.891 | 28.664 | 19.514 |
| 2025-07-16 | 160.667 | 148.127 | 12.541 | 8.466 |
| 2025-07-17 | 162.910 | 154.320 | 8.590 | 5.566 |
| 2025-07-18 | 159.773 | 153.389 | 6.384 | 4.162 |
| 2025-07-19 | 158.882 | 141.616 | 17.267 | 12.193 |
| 2025-07-20 | 142.533 | 134.525 | 8.008 | 5.952 |
| 2025-07-21 | 139.119 | 124.808 | 14.311 | 11.466 |
| 2025-07-22 | 122.841 | 101.524 | 21.317 | 20.997 |
| 2025-07-23 | 122.529 | 105.318 | 17.211 | 16.342 |
| 2025-07-24 | 108.074 | 96.707 | 11.366 | 11.753 |

## Shared artifacts

The canonical results are under:

```text
/home/SHARED/data/weather-routing/results/toy_20250715_20250724/AO_WPS/
├── summary.csv
├── YYYYMMDD_00z.json
└── YYYYMMDD_00z_routes.npz
```

There are ten JSON metric files and ten compressed paired-route files. The run
log is
`/home/SHARED/data/weather-routing/logs/toy_20250715_20250724/experiment.log`.

This is a toy result, not a performance claim: it covers one corridor, one
case, ten consecutive July departures, one optimizer seed, and a limited
optimization budget. “Perfect information” here means the route found by the
same finite-budget optimizer using ERA5, not a proven global optimum.
