# CMA-ES Diagnostic: Issues Found During SWOPP3 Integration

## Background

The CMA-ES optimizer in `routetools` was designed for **ocean-current** routing — the vectorfield represents water velocity, and the cost functions model a vessel moving through a medium. SWOPP3 requires **wind-assisted** routing with a fixed passage time and ERA5 weather data (wind + waves). Several issues emerged when adapting CMA-ES for this scenario.

---

## Issue 1: Time Offset — All Departures Sampled Weather at t=0

**Location:** `routetools/swopp3_runner.py` → `run_case()`, `routetools/cost.py` → `cost_function()`

**Problem:** `departure_offset_h` was hardcoded to `0.0` in `run_case()`. All 366 departures optimised against January 1 weather regardless of their actual departure date.

**Root cause:** `cost_function()` had no `time_offset` parameter. The time-variant cost functions computed segment timestamps starting from `t=0`.

**Fix (committed `73e8b05`):**

- Added `time_offset: float = 0.0` to `cost_function`, `cost_function_constant_cost_time_variant`, `_cma_evolution_strategy`, `optimize`, and `optimize_with_increasing_penalization`
- Made `time_offset` a **non-static** JAX argument (not in `static_argnames`) so changing it per-departure does not trigger JIT recompilation
- `run_case()` now computes offset from the dataset epoch:

```python
# routetools/swopp3_runner.py — run_case()
departure_offset_h = (dep_naive - epoch_naive).total_seconds() / 3600.0
```

**Status:** ✅ Fixed and verified — second departure takes 0.1s (JIT cache hit), different energy values confirm per-departure weather.

---

## Issue 2: Wind Treated as Ocean Current (Fundamental Model Mismatch)

**Location:** `routetools/cost.py` — all `cost_function_constant_cost_*` variants

**Problem:** The "constant cost" (fixed travel time) cost functions compute:

$$\text{cost} = \sum_i \frac{1}{2} \|\mathbf{v}_{\text{SOG},i} - \mathbf{w}_i\|^2 \cdot \Delta t$$

where $\mathbf{w}$ is the vectorfield (wind in SWOPP3). This formula interprets $\mathbf{w}$ as **medium velocity** (ocean current) — the ship's effort is its speed _through the water_ relative to the current.

For ocean currents this is physically correct: a ship traveling with the current needs less thrust, against it needs more. But **wind does not transport the ship**. Wind acts on sails (WPS) and creates aerodynamic drag, but it does not change the ship's speed-over-ground like a current does.

**Consequence:** CMA-ES finds routes that "follow the wind" (minimising ‖SOG − wind‖²), producing detours 1.5–2× longer than GC. These routes have _lower CMA-ES cost_ but _higher actual energy_ in the RISE performance model because:

- Longer distance at fixed time → higher SOG → enormously more hydrodynamic drag (P ∝ v³)
- The "benefit" of aligning with wind is illusory — wind velocity (10 m/s) was being subtracted from ship displacement rate, but that's not how ship propulsion works

**Evidence (smoke test with correct units, 2 departures):**

| Case   | GC Distance | GC Energy  | Opt Distance | Opt Energy |
| ------ | ----------- | ---------- | ------------ | ---------- |
| AO_WPS | 3023 nm     | 185.65 MWh | 4557 nm      | 434 MWh    |
| PO_WPS | 4653 nm     | 124.49 MWh | 5702 nm      | 268 MWh    |

**Root cause:** The cost functions in `routetools/cost.py` were designed for the ocean-current benchmark problems (Zermelo-type navigation). The `constant_speed` variants use the correct current physics:

```python
# routetools/cost.py — cost_function_constant_speed_time_invariant()
# Time for a vessel with STW=v to traverse displacement d in current w:
#
# dt = sqrt(d² / (v² − w²) + (d·w)² / (v² − w²)²) − (d·w) / (v² − w²)
#
v2 = travel_stw_mod**2
w2 = uinterp**2 + vinterp**2
dw = dx * uinterp + dy * vinterp
dt = jnp.sqrt(d2 / (v2 - w2) + dw**2 / (v2 - w2) ** 2) - dw / (v2 - w2)
```

This is correct for currents but inapplicable to wind.

The `constant_cost` (fixed travel time) variants compute:

```python
# routetools/cost.py — cost_function_constant_cost_time_variant()
# STW cost = ‖SOG − current‖² / 2 · dt
dxdt = dx / dt_s
dydt = dy / dt_s
cost = ((dxdt - uinterp) ** 2 + (dydt - vinterp) ** 2) / 2
```

This is kinetic energy of the ship's velocity _relative to the medium_ (water + current). Only makes physical sense when `w` is an ocean current (medium velocity), not wind.

**What would be needed:** A JAX-compatible cost function that models the effect of wind and waves on ship energy — essentially a differentiable or at least JAX-traced version of the RISE performance model (`predict_power_batch`). The current RISE model uses NumPy lookup tables and is not JIT-compatible.

**Status:** ❌ Unfixed. **This is the core blocker for meaningful optimisation.**

---

## Issue 3: Unit Mismatch — SOG in m/h vs Wind in m/s

**Location:** `routetools/cost.py` → `cost_function_constant_cost_time_variant()`

**Problem:** When `spherical_correction=True`, haversine gives distances in **meters**. With `travel_time` in **hours**, `dt` is in hours:

```python
dt = travel_time / n_seg          # hours
dxdt = dx / dt                    # meters / hours ≈ 11,000 m/h per segment
```

But wind from ERA5 is in **m/s** ≈ 10 m/s. The cost ‖SOG − wind‖² was dominated by SOG (~11,000 m/h vs ~10 m/s), making wind completely negligible.

**Fix (committed `6406e20`):**

```python
# routetools/cost.py — cost_function_constant_cost_time_variant()
# Convert dt from hours to seconds so SOG is in m/s
dt_s = dt * 3600.0
dxdt = dx / dt_s
dydt = dy / dt_s
cost = ((dxdt - uinterp) ** 2 + (dydt - vinterp) ** 2) / 2
return cost * dt_s
```

**Note:** The time-invariant variant (`cost_function_constant_cost_time_invariant`) has the same pattern — `dxdt = dx / dt` without unit conversion. But it doesn't use `spherical_correction`, so the units depend on the coordinate system. This should be reviewed.

**Status:** ✅ Fixed for the time-variant variant. Time-invariant variant needs review.

---

## Issue 4: Pacific Longitude Wrapping (Antimeridian)

**Location:** `routetools/swopp3.py` → `case_endpoints()`, `great_circle_route()`, `routetools/cmaes.py` → endpoint validation

**Problem:** Pacific route goes Tokyo (140°E) → Los Angeles (−121° = 239°E), crossing 180°. `great_circle_route()` correctly uses SLERP + longitude unwrapping, producing continuous coordinates (140° → 239°). But `case_endpoints()` returns `dst = [-121, 34.4]`.

CMA-ES validates endpoints:

```python
# routetools/cmaes.py — optimize()
if not jnp.allclose(curve0[-1, :], dst):
    raise ValueError(
        "The ending point of curve0 does not match dst. "
        f"curve0[-1,:]={curve0[-1, :]}, dst={dst}"
    )
# → ValueError: curve0[-1,:]=[238.99998  34.4], dst=[-121.    34.4]
```

**Fix (committed `6406e20`):** Extract the unwrapped endpoints from the GC curve and pass those to CMA-ES:

```python
# routetools/swopp3_runner.py — run_optimised_departure()
gc_init = great_circle_route(src, dst, n_points=n_points)
# great_circle_route may unwrap longitude through the antimeridian
# (e.g. -121° becomes 239°).  Use the unwrapped endpoints so the
# CMA-ES endpoint check passes and the Bézier curve stays in a
# consistent longitude range.
src_opt = jnp.array([gc_init[0, 0], gc_init[0, 1]])
dst_opt = jnp.array([gc_init[-1, 0], gc_init[-1, 1]])
```

**Status:** ✅ Fixed.

---

## Issue 5: CMA-ES Sigma Scaling for Large Coordinate Spans

**Location:** `routetools/cmaes.py` → `optimize()`

**Problem:** `sigma0` is scaled by the Euclidean distance between endpoints:

```python
# routetools/cmaes.py — optimize()
# Initial standard deviation to sample new solutions
# One sigma is half the distance between src and dst
sigma0 = float(jnp.linalg.norm(dst - src) * sigma0 / 2)
```

For Pacific routes (140° to 239°), `norm ≈ 102`, giving effective `sigma0 ≈ 51°` with default `sigma0=1`. This causes extreme search space exploration — the optimizer samples control points 50° away from the initial route.

**Workaround (committed `6406e20`):** Pass `sigma0=0.1` from `run_optimised_departure()`:

```python
# routetools/swopp3_runner.py — run_optimised_departure()
defaults = dict(
    L=n_points,
    curve0=gc_init,
    travel_time=travel_time,
    spherical_correction=True,
    time_offset=departure_offset_h,
    sigma0=0.1,
    verbose=False,
)
```

This gives effective sigma ≈ 5° for Pacific, ≈ 3° for Atlantic.

**Deeper issue:** The sigma scaling assumes coordinates are in a "reasonable" range. For global routes in degrees, the norm can be 100+, making the default too large. Consider scaling by route length in the cost-function's own units, or using normalised coordinates.

**Status:** ⚠️ Workaround in place. May need revisiting.

---

## Issue 6: Waves Not Incorporated in Fixed-Time Cost Functions

**Location:** `routetools/cost.py`

**Status of wave integration across cost functions:**

| Cost Function                   | Waves? | Note                                      |
| ------------------------------- | ------ | ----------------------------------------- |
| `constant_speed_time_invariant` | ✅ Yes | `wave_adjusted_speed()` reduces STW       |
| `constant_speed_time_variant`   | ✅ Yes | Same wave model, with `lax.scan` for time |
| `constant_cost_time_invariant`  | ❌ No  | Has `# TODO: Implement wavefield effects` |
| `constant_cost_time_variant`    | ❌ No  | New function, no wave integration         |

The wave model in the constant-speed variants uses Townsin-Kwon involuntary speed loss (via `wave_adjusted_speed` in `routetools/_cost/waves.py`). This reduces the ship's effective STW based on Beaufort scale and wave incidence angle:

```python
# routetools/_cost/waves.py — wave_adjusted_speed()
wia = jnp.mod(jnp.abs(angle - wave_angle), 360)
wave_incidence_angle = jnp.minimum(wia, 360 - wia)
beaufort = beaufort_scale(wave_height=wave_height, asfloat=True, ...)
speed_loss = speed_loss_involuntary(
    beaufort=beaufort,
    wave_incidence_angle=wave_incidence_angle,
    vel_ship=vel_ship,
    ...
)
return jnp.asarray(vel_ship) * (100 - speed_loss) / 100
```

For the constant-cost (fixed-time) variants, wave effects would need to be incorporated differently — perhaps as an additive resistance term rather than a speed reduction.

**Status:** ❌ Known limitation. Marked with TODO in code.

---

## Issue 7: Weather Penalty Uses t=0 for Time-Variant Fields

**Location:** `routetools/weather.py` → `weather_penalty()`

**Problem:** The `weather_penalty()` function always queries fields at `t=0`, ignoring the departure time:

```python
# routetools/weather.py — weather_penalty()
mid_lon = (curve[:, :-1, 0] + curve[:, 1:, 0]) / 2
mid_lat = (curve[:, :-1, 1] + curve[:, 1:, 1]) / 2
t_zeros = jnp.zeros_like(mid_lon)   # ← always t=0

if windfield is not None:
    u10, v10 = windfield(mid_lon, mid_lat, t_zeros)
    tws = jnp.sqrt(u10**2 + v10**2)
    violations = violations + jnp.sum(tws > tws_limit, axis=1)

if wavefield is not None:
    hs, _ = wavefield(mid_lon, mid_lat, t_zeros)
    violations = violations + jnp.sum(hs > hs_limit, axis=1)
```

When `weather_penalty_weight > 0`, the penalty would be based on January 1 weather regardless of actual departure date.

**Status:** ⚠️ Not critical since `weather_penalty_weight` defaults to `0.0`, but would need fixing if weather penalties are enabled.

---

## Summary

### What Works

| Component                   | Status                                                   |
| --------------------------- | -------------------------------------------------------- |
| GC cases (all 8)            | ✅ Great-circle + RISE energy evaluation. Valid outputs. |
| Per-departure time offset   | ✅ Each departure samples correct weather.               |
| ERA5 data loading           | ✅ Per-corridor caching, correct longitude handling.     |
| Unit conversion (m/h → m/s) | ✅ `dt_s = dt * 3600` in time-variant cost.              |
| Pacific longitude wrapping  | ✅ Unwrapped endpoints passed to CMA-ES.                 |

### What Doesn't

| Component                       | Status                                                      |
| ------------------------------- | ----------------------------------------------------------- |
| CMA-ES + wind field             | ❌ Wind treated as current → routes diverge → worse energy. |
| CMA-ES + wave field in cost     | ❌ Not implemented for fixed-time variants.                 |
| Weather penalty time awareness  | ⚠️ Uses t=0 (OK since weight=0 by default).                 |
| Sigma scaling for global routes | ⚠️ Workaround (`sigma0=0.1`), needs proper fix.             |

### Core Blocker

The fundamental issue is that **the CMA-ES cost function models a vessel navigating through an ocean current** (Zermelo-type problem), but SWOPP3 requires optimising a **wind-assisted ship** where wind affects propulsion aerodynamics, not the medium itself. A JAX-compatible energy proxy (differentiable approximation of the RISE performance model) is needed to make CMA-ES optimisation meaningful.
