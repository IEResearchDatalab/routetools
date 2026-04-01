# SWOPP3 Weather Routing Benchmark

## Overview

The **SWOPP3 Weather Routing Benchmark** evaluates weather routing optimizers on real ERA5 weather data using the RISE performance model for an 88 m cargo ship with optional wingsails.

### The Challenge

Participants must find minimum-energy routes across **two ocean corridors**:

| Route              | From              | To                  | Passage Time | GC Distance |
| ------------------ | ----------------- | ------------------- | ------------ | ----------- |
| **Trans-Atlantic** | Santander (ESSDR) | New York (USNYC)    | 354 hours    | 2,833 nm    |
| **Trans-Pacific**  | Tokyo (JPTYO)     | Los Angeles (USLAX) | 583 hours    | 4,663 nm    |

For each route, there are **4 cases** combining route optimisation strategy and wingsail configuration:

| Case        | Strategy                      | Wingsails (WPS) |
| ----------- | ----------------------------- | --------------- |
| `*O_WPS`    | **Optimised** route and speed | Yes             |
| `*O_noWPS`  | **Optimised** route and speed | No              |
| `*GC_WPS`   | **Great Circle**, fixed speed | Yes             |
| `*GC_noWPS` | **Great Circle**, fixed speed | No              |

This gives **8 cases total** × **366 daily departures** (every day of 2024, noon UTC) = **2,928 route evaluations per submission**.

### Port Coordinates

| Port Code | City             | Latitude | Longitude |
| --------- | ---------------- | -------- | --------- |
| ESSDR     | Santander, Spain | 43.60° N | 4.00° W   |
| USNYC     | New York, USA    | 40.53° N | 73.80° W  |
| JPTYO     | Tokyo, Japan     | 34.80° N | 140.00° E |
| USLAX     | Los Angeles, USA | 34.40° N | 121.00° W |

### Operational Constraints

**Optimised cases** (`*O_*`) must respect these weather safety limits along the entire path:

| Constraint                  | Limit            | Description                         |
| --------------------------- | ---------------- | ----------------------------------- |
| **Significant wave height** | Hs < **7 m**     | Maximum wave height along the route |
| **True wind speed**         | TWS < **20 m/s** | Maximum wind speed along the route  |
| **Land avoidance**          | No crossings     | Route waypoints must not cross land |

> **Note on Great Circle cases:** GC cases follow a fixed geodesic path that participants cannot modify. Because the great circle may cross land (e.g. the Atlantic route clips Newfoundland) and traverse severe weather, **operational and land-crossing checks are not enforced** for GC cases.

### What Makes This Different

Unlike the original SWOPP3 competition, **all participants use the same weather data and the same performance model**. This ensures results are directly comparable — the only differentiator is the optimization algorithm.

### Ranking

Submissions are ranked by **total energy consumption (MWh)** summed across all 8 cases and 366 departures. Lower is better.

### Getting Started

You can choose between two approaches to obtain the ERA5 data:

**Option A — Download from CodaBench** (recommended):

1. Download the pre-built ERA5 `.nc` files from the **Files** tab of this competition
2. Implement the RISE performance model (formulas provided in the **Data** tab)
3. Submit a zip file with your CSV results

**Option B — Download via CDS API**:

1. Register at the Copernicus Climate Data Store and download ERA5 data with the `cdsapi` package
2. Implement the RISE performance model (formulas provided in the **Data** tab)
3. Submit a zip file with your CSV results

See the **Data & Performance Model** tab for details on both options.
