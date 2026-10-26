# Data & Performance Model

Benchmark 001 uses a fixed hourly ERA5 weather-data release and the RISE
performance model. All participants and the scoring worker must use this exact
release; its SHA-256 checksums are published with the data files.

---

## Option A — Download the frozen release (Recommended)

The release is hosted on Zenodo: at 19.57 GB it exceeds the 15 GB CodaBench
storage quota, so it cannot be attached to this benchmark directly.

**Download:** [doi:10.5281/zenodo.23470870](https://doi.org/10.5281/zenodo.23470870)

You need 2024 data plus January 2025 (late-December departures extend into January 2025):

| File                             | Size (approx.) | Contents                                                  |
| -------------------------------- | -------------- | --------------------------------------------------------- |
| `era5_wind_atlantic_2024.nc`     | ~3.58 GB       | 10 m wind (u, v) — Atlantic corridor                      |
| `era5_waves_atlantic_2024.nc`    | ~3.58 GB       | Wave height and direction — Atlantic corridor             |
| `era5_wind_pacific_2024.nc`      | ~5.44 GB       | 10 m wind (u, v) — Pacific corridor                       |
| `era5_waves_pacific_2024.nc`     | ~5.44 GB       | Wave height and direction — Pacific corridor              |
| `era5_wind_atlantic_2025_01.nc`  | ~303 MB        | 10 m wind — January 2025, Atlantic                        |
| `era5_waves_atlantic_2025_01.nc` | ~303 MB        | Waves — January 2025, Atlantic                            |
| `era5_wind_pacific_2025_01.nc`   | ~461 MB        | 10 m wind — January 2025, Pacific                         |
| `era5_waves_pacific_2025_01.nc`  | ~461 MB        | Waves — January 2025, Pacific                             |

The release also contains the Natural Earth 1:10m land polygons
(`ne_10m_land.{shp,shx,dbf,prj}`) the scorer uses for land checks, and
`SHA256SUMS`.

The NetCDF files are spatial and temporal subsets of ERA5, taken from the
analysis-ready ARCO-ERA5 archive. Variables:

- **Wind:** `10m_u_component_of_wind` (eastward), `10m_v_component_of_wind` (northward), in m/s
- **Waves:** `significant_height_of_combined_wind_waves_and_swell` (m), `mean_wave_direction` (degrees, direction waves come from, clockwise from north)
- **Coordinates:** `valid_time`, `latitude` (descending), `longitude`. Pacific files use 0–360° longitude (120° to 240°).
- **Grid:** 0.25° × 0.25°, hourly time steps (00:00 through 23:00 UTC)
- **Corridors:** Atlantic 25°N–60°N, 80°W–10°E; Pacific 15°N–55°N, 120°E–120°W

The eight NetCDF files total 19.57 GB uncompressed. Verify each downloaded file
against the published `SHA256SUMS` before using it — this is the only way to
confirm you hold the same release the scorer uses. With these files in hand,
implement the RISE performance model from the formulas below.

---

## Option B — Download ERA5 via CDS API

If you prefer to download the data yourself, register for a free account at [CDS](https://cds.climate.copernicus.eu/) and install the `cdsapi` package:

```bash
pip install cdsapi
```

Download wind and waves for each corridor. You need all of **2024** plus **January 2025** (late-December departures extend into January 2025):

```python
import cdsapi

client = cdsapi.Client()

DAYS = [f"{d:02d}" for d in range(1, 32)]
TIMES = [f"{hour:02d}:00" for hour in range(24)]
PERIODS = {
    "2024": ("2024", [f"{m:02d}" for m in range(1, 13)]),
    "2025_01": ("2025", ["01"]),
}

CORRIDORS = {
    "atlantic": [60, -80, 25, 10],     # [N, W, S, E]
    "pacific":  [55, 120, 15, 240],    # Uses 0-360° longitude
}

WIND_VARS = ["10m_u_component_of_wind", "10m_v_component_of_wind"]
WAVE_VARS = [
    "significant_height_of_combined_wind_waves_and_swell",
    "mean_wave_direction",
]

for label, (year, months) in PERIODS.items():
    for corridor, area in CORRIDORS.items():
        for var_type, variables in [("wind", WIND_VARS), ("waves", WAVE_VARS)]:
            client.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": "reanalysis",
                    "variable": variables,
                    "year": year,
                    "month": months,
                    "day": DAYS,
                    "time": TIMES,
                    "area": area,
                    "grid": [0.25, 0.25],
                    "data_format": "netcdf",
                },
                f"era5_{var_type}_{corridor}_{label}.nc",
            )
```

The resulting NetCDF files name the variables `u10`, `v10` (wind) and `swh`,
`mwd` (waves). CDS regrids the 0.5° native wave fields to 0.25° on request.
A CDS export is **not** byte-identical to the Benchmark 001 release and will
not match `SHA256SUMS`, so small differences against the scorer are expected.
The scorer always uses the Zenodo release; use Option A when you need exact
agreement.

### RISE Performance Model — Full Specification

The RISE model computes instantaneous **power in kW** for an 88 m cargo ship given:

- $v$ — Ship speed through water (m/s)
- TWS — True wind speed (m/s)
- TWA — True wind angle (degrees): direction the wind comes **from**, minus the ship heading. 0° = headwind
- SWH — Significant wave height (m)
- MWA — Mean wave angle (degrees): `mean_wave_direction` minus the ship heading, wrapped to [−180°, 180°]. 0° = head seas

#### Constants

| Symbol    | Value      | Exact Fraction |
| --------- | ---------- | -------------- |
| $K_H$     | 4.2876…    | 969 / 226      |
| $K_A$     | 0.153125   | 49 / 320       |
| $A_W$     | 11.1395    | —              |
| $K_W$     | 0.28935…   | 125 / 432      |
| $K_S$     | 0.85903125 | 27489 / 32000  |
| Dead zone | 10°        | —              |

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

where $\theta_{\text{MWA}}$ is the mean wave angle in **radians**, in [−π, π].

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

**Energy integration** — the scorer evaluates each track as follows:

1. Resample the track to points every **15 minutes** along the great-circle legs
   between your waypoints (your first and last waypoints are kept exactly).
2. Interpolate wind and waves at every point, tricubically in latitude,
   longitude and time.
3. For each sub-segment $i$, take the speed $v_i$ as its great-circle length
   (spherical Earth, $R = 6371$ km) divided by its duration, and the heading as
   the initial bearing of the sub-segment.
4. Evaluate the power at both ends of the sub-segment with that speed and
   heading, $P_i^{\text{start}}$ and $P_i^{\text{end}}$, and apply the
   trapezoidal rule:

$$E_{\text{MWh}} = \frac{1}{1000} \sum_{i=1}^{n} \frac{P_i^{\text{start}} + P_i^{\text{end}}}{2} \cdot \Delta t_{h,i}$$

where $\Delta t_{h,i}$ is the duration of sub-segment $i$ in hours. Ship speed
is taken over ground: ocean currents are not modelled in Benchmark 001.

### Route and Departure Definitions

| Case      | Source (lat, lon) | Destination (lat, lon) | Passage (h) | WPS |
| --------- | ----------------- | ---------------------- | ----------- | --- |
| AO_WPS    | (43.60, −4.00)    | (40.60, −69.00)        | 354         | Yes |
| AO_noWPS  | (43.60, −4.00)    | (40.60, −69.00)        | 354         | No  |
| AGC_WPS   | (43.60, −4.00)    | (40.60, −69.00)        | 354         | Yes |
| AGC_noWPS | (43.60, −4.00)    | (40.60, −69.00)        | 354         | No  |
| PO_WPS    | (34.80, 140.00)   | (34.40, −121.00)       | 583         | Yes |
| PO_noWPS  | (34.80, 140.00)   | (34.40, −121.00)       | 583         | No  |
| PGC_WPS   | (34.80, 140.00)   | (34.40, −121.00)       | 583         | Yes |
| PGC_noWPS | (34.80, 140.00)   | (34.40, −121.00)       | 583         | No  |

**Departures:** 366 departures — every day of 2024 at **12:00 UTC** (from 2024-01-01 to 2024-12-31).

Source and destination are fixed offshore points off the named ports, not the
port berths; see **Submission Format** for the 0.5° endpoint tolerance.

**Great Circle cases** (`*GC_*`): The route must follow the great circle path between source and destination. Only the speed profile along that path may vary.

**Optimised cases** (`*O_*`): Both route geometry and speed profile may be freely optimised.
