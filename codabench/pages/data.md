# Data & Performance Model

All participants use the **same ERA5 weather data** and the **same RISE performance model**. You can choose between two approaches to obtain the data.

---

## Option A — Download ERA5 from CodaBench (Recommended)

The ERA5 NetCDF files are available for direct download from the **Files** tab of this competition. You need 2024 data plus January 2025 (late-December departures extend into January 2025):

| File | Size (approx.) | Contents |
|------|----------------|----------|
| `era5_wind_atlantic_2024.nc` | ~570 MB | 10 m wind (u10, v10) — Atlantic corridor |
| `era5_waves_atlantic_2024.nc` | ~570 MB | Wave height (swh) and direction (mwd) — Atlantic corridor |
| `era5_wind_pacific_2024.nc` | ~865 MB | 10 m wind (u10, v10) — Pacific corridor |
| `era5_waves_pacific_2024.nc` | ~865 MB | Wave height (swh) and direction (mwd) — Pacific corridor |
| `era5_wind_atlantic_2025_01.nc` | ~49 MB | 10 m wind — January 2025, Atlantic |
| `era5_waves_atlantic_2025_01.nc` | ~49 MB | Waves — January 2025, Atlantic |
| `era5_wind_pacific_2025_01.nc` | ~74 MB | 10 m wind — January 2025, Pacific |
| `era5_waves_pacific_2025_01.nc` | ~74 MB | Waves — January 2025, Pacific |

These are the exact same files produced by the `routetools` downloader. The NetCDF variables are:
- **Wind:** `u10` (eastward), `v10` (northward) in m/s
- **Waves:** `swh` (significant wave height in m), `mwd` (mean wave direction in degrees)
- **Grid:** 0.25° × 0.25°, 6-hourly time steps (00:00, 06:00, 12:00, 18:00 UTC)

With these files in hand, implement the RISE performance model from the formulas below.

> **Tip:** For higher temporal resolution, you can download hourly ERA5 data yourself via the CDS API (Option B) or by using the `routetools` downloader with `--time-step 1`.

---

## Option B — Download ERA5 via CDS API

If you prefer to download the data yourself, register for a free account at [CDS](https://cds.climate.copernicus.eu/) and install the `cdsapi` package:

```bash
pip install cdsapi
```

Download wind and waves for each corridor. You need **both 2024 and 2025** (late-December departures extend into January 2025):

```python
import cdsapi

client = cdsapi.Client()

MONTHS = [f"{m:02d}" for m in range(1, 13)]
DAYS = [f"{d:02d}" for d in range(1, 32)]
TIMES = ["00:00", "06:00", "12:00", "18:00"]

CORRIDORS = {
    "atlantic": [60, -80, 25, 10],     # [N, W, S, E]
    "pacific":  [55, 120, 15, 240],    # Uses 0-360° longitude
}

WIND_VARS = ["10m_u_component_of_wind", "10m_v_component_of_wind"]
WAVE_VARS = [
    "significant_height_of_combined_wind_waves_and_swell",
    "mean_wave_direction",
]

for year in ["2024", "2025"]:
    for corridor, area in CORRIDORS.items():
        for var_type, variables in [("wind", WIND_VARS), ("waves", WAVE_VARS)]:
            client.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": "reanalysis",
                    "variable": variables,
                    "year": year,
                    "month": MONTHS,
                    "day": DAYS,
                    "time": TIMES,
                    "area": area,
                    "grid": [0.25, 0.25],
                    "data_format": "netcdf",
                },
                f"era5_{var_type}_{corridor}_{year}.nc",
            )
```

The resulting NetCDF files contain variables named `u10`, `v10` (wind) and `swh`, `mwd` (waves), on a 0.25° grid at 6-hourly intervals.

> **Note:** To download hourly data instead, change `TIMES` to `[f"{h:02d}:00" for h in range(24)]`.

### RISE Performance Model — Full Specification

The RISE model computes instantaneous **power in kW** for an 88 m cargo ship given:

- $v$ — Ship speed through water (m/s)
- TWS — True wind speed (m/s)
- TWA — True wind angle (degrees, 0° = headwind)
- SWH — Significant wave height (m)
- MWA — Mean wave angle relative to heading (degrees)

#### Constants

| Symbol | Value | Exact Fraction |
|--------|-------|---------------|
| $K_H$ | 4.2876… | 969 / 226 |
| $K_A$ | 0.153125 | 49 / 320 |
| $A_W$ | 11.1395 | — |
| $K_W$ | 0.28935… | 125 / 432 |
| $K_S$ | 0.85903125 | 27489 / 32000 |
| Dead zone | 10° | — |

#### Power Components

**Hull resistance:**

$$P_{\text{hull}} = K_H \cdot v^3$$

**Apparent wind:**

$$u_x = \text{TWS} \cdot \cos(\text{TWA}) + v$$
$$u_y = \text{TWS} \cdot \sin(\text{TWA})$$
$$V_R = \sqrt{u_x^2 + u_y^2}$$

(TWA in radians for $\cos$/$\sin$ calls.)

**Aerodynamic drag:**

$$P_{\text{wind}} = K_A \cdot v \cdot (V_R \cdot u_x - v^2)$$

**Wave added resistance:**

$$P_{\text{wave}} = A_W \cdot \text{SWH}^2 \cdot v^{3/2} \cdot \exp\!\left(-K_W \cdot |\theta_{\text{MWA}}|^3\right)$$

where $\theta_{\text{MWA}}$ is the mean wave angle in **radians**.

**Wingsail thrust (WPS only):**

$$\text{AWA} = \text{atan2}(|u_y|,\, u_x) \quad \text{(in degrees)}$$

If $\text{AWA} < 10°$, sail contribution is zero. Otherwise:

$$\alpha = (\text{AWA} - 10°) \cdot \pi / 180$$
$$P_{\text{sail}} = K_S \cdot \sin(\alpha) \cdot \left(1 + \frac{3}{20}\sin^2(\alpha)\right) \cdot V_R^2 \cdot v$$

#### Total Power and Energy

**Without wingsails:**

$$P = \max\!\left(0,\; P_{\text{hull}} + P_{\text{wind}} + P_{\text{wave}}\right)$$

**With wingsails (WPS):**

$$P = \max\!\left(0,\; P_{\text{hull}} + P_{\text{wind}} + P_{\text{wave}} - P_{\text{sail}}\right)$$

**Energy integration** — sum over all waypoint segments:

$$E_{\text{MWh}} = \frac{1}{1000} \sum_{i=1}^{n} P_i \cdot \Delta t_{h,i}$$

where $\Delta t_{h,i}$ is the duration of segment $i$ in hours, and $P_i$ is the power in kW at the midpoint of each segment.

### Route and Departure Definitions

| Case | Source (lat, lon) | Destination (lat, lon) | Passage (h) | WPS |
|------|------------------|----------------------|-------------|-----|
| AO_WPS | (43.60, −4.00) | (40.53, −73.80) | 354 | Yes |
| AO_noWPS | (43.60, −4.00) | (40.53, −73.80) | 354 | No |
| AGC_WPS | (43.60, −4.00) | (40.53, −73.80) | 354 | Yes |
| AGC_noWPS | (43.60, −4.00) | (40.53, −73.80) | 354 | No |
| PO_WPS | (34.80, 140.00) | (34.40, −121.00) | 583 | Yes |
| PO_noWPS | (34.80, 140.00) | (34.40, −121.00) | 583 | No |
| PGC_WPS | (34.80, 140.00) | (34.40, −121.00) | 583 | Yes |
| PGC_noWPS | (34.80, 140.00) | (34.40, −121.00) | 583 | No |

**Departures:** 366 departures — every day of 2024 at **12:00 UTC** (from 2024-01-01 to 2024-12-31).

**Great Circle cases** (`*GC_*`): The route must follow the great circle path between source and destination. Only the speed profile along that path may vary.

**Optimised cases** (`*O_*`): Both route geometry and speed profile may be freely optimised.
