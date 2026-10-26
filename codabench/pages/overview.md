# Weather Routing Benchmark 001

## Overview

The **Weather Routing Benchmark** is an ongoing, open benchmark for weather-routing optimisation. Each benchmark number fixes one complete set of evaluation conditions, so results stay comparable over time.

**Benchmark 001** evaluates optimisers on real ERA5 weather data using the RISE performance model for an 88 m cargo ship with optional wingsails. Its conditions are frozen: the RISE wingsail performance model, a fixed hourly ERA5 release, and deterministic route definitions. Future benchmark numbers may define different conditions, such as probabilistic weather or a different performance model.

### The Challenge

Participants must find minimum-energy routes across **two ocean corridors**:

| Route              | From      | To          | Passage Time | GC Distance |
| ------------------ | --------- | ----------- | ------------ | ----------- |
| **Trans-Atlantic** | Santander | New York    | 354 hours    | 2,833 nm    |
| **Trans-Pacific**  | Tokyo     | Los Angeles | 583 hours    | 4,663 nm    |

For each route, there are **4 cases** combining route optimisation strategy and wingsail configuration:

| Case        | Strategy                      | Wingsails (WPS) |
| ----------- | ----------------------------- | --------------- |
| `*O_WPS`    | **Optimised** route and speed | Yes             |
| `*O_noWPS`  | **Optimised** route and speed | No              |
| `*GC_WPS`   | **Great Circle**, fixed speed | Yes             |
| `*GC_noWPS` | **Great Circle**, fixed speed | No              |

This gives **8 cases total** × **366 daily departures** (every day of 2024, noon UTC) = **2,928 route evaluations per submission**.

### Start and End Points

Routes start and end at fixed **offshore points**, not at the port berths. Each
point lies at sea, off the approaches to the named port, so harbour pilotage and
coastal legs are excluded from the benchmark.

| Point                      | Latitude | Longitude | Decimal (lat, lon) |
| -------------------------- | -------- | --------- | ------------------ |
| Santander approach (start) | 43.60° N | 4.00° W   | (43.60, −4.00)     |
| New York approach (end)    | 40.60° N | 69.00° W  | (40.60, −69.00)    |
| Tokyo approach (start)     | 34.80° N | 140.00° E | (34.80, 140.00)    |
| Los Angeles approach (end) | 34.40° N | 121.00° W | (34.40, −121.00)   |

> The New York point lies about 220 nm east of New York harbour, south of Nantucket. Use these exact coordinates, not the harbour position: the first and last waypoints of every track must be within 0.5° of them (see **Submission Format**).

### Operational Constraints

**Optimised cases** (`*O_*`) must respect these weather safety limits along the entire path:

| Constraint                  | Limit            | Description                         |
| --------------------------- | ---------------- | ----------------------------------- |
| **Significant wave height** | Hs ≤ **7 m**     | Maximum wave height along the route |
| **True wind speed**         | TWS ≤ **20 m/s** | Maximum wind speed along the route  |
| **Land avoidance**          | No crossings     | No part of the route may be on land |

> **Note on Great Circle cases:** GC cases follow a fixed great-circle path that participants cannot modify; only the speed along it may vary. Because the path is fixed and may traverse severe weather, **operational and land-crossing checks are not enforced** for GC cases.

### What Makes This Different

**All participants use the same weather data and the same performance model.** This ensures results are directly comparable — the only differentiator is the optimization algorithm.

### Ranking

Submissions are ranked by **total energy consumption (MWh)** summed across all 8 cases and 366 departures. Lower is better.

### Getting Started

You can choose between two approaches to obtain the ERA5 data:

**Option A — Download the frozen release** (recommended):

1. Download the pre-built ERA5 `.nc` files from Zenodo ([doi:10.5281/zenodo.23470870](https://doi.org/10.5281/zenodo.23470870)), and verify them against the published `SHA256SUMS`
2. Implement the RISE performance model (formulas provided in the **Data** tab)
3. Submit a zip file with your CSV results

**Option B — Download via CDS API**:

1. Register at the Copernicus Climate Data Store and download ERA5 data with the `cdsapi` package
2. Implement the RISE performance model (formulas provided in the **Data** tab)
3. Submit a zip file with your CSV results

See the **Data & Performance Model** tab for details on both options.
