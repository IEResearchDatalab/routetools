# Adaptive weather-routing audit

## Decision

The existing adaptive animation is a storyboard, not an experiment. It must not
be reported or scored as an adaptive-routing result. A defensible adaptive run
must load a newly initialized deterministic IFS wind and wave forecast at every
decision time, freeze the sailed prefix, optimize only the remaining suffix,
and score the realized sailed segments in ERA5 at their actual timestamps.

## Forecast policy

For the current idealized pilot, use the deterministic ECMWF IFS `oper` and
`wave` 00 UTC runs. For a voyage departing at 00 UTC on 15 July 2025, the
decision sequence is:

- 15 July 00 UTC: optimize the full 354 h voyage with the 15 July run.
- 16--28 July 00 UTC: optimize the remaining voyage with that day's run, sail
  the next 24 h, and freeze it.
- 29 July 00 UTC: optimize and sail the final 18 h.

Each run covers lead 0--360 h, so every remaining-horizon problem fits inside a
single forecast. This convention assumes that the run initialized at 00 UTC is
available at the 00 UTC decision time. That is an analysis-time idealization,
not a latency-realistic operational protocol. A genuinely operational study
must define publication latency and use the latest run actually available.

## Verified data inventory on vangelis

Canonical root: `/home/SHARED/data/weather-routing`

- IFS deterministic forecasts currently present: 15--24 July 2025, separate
  `oper` wind and `wave` NetCDF files at 00 UTC.
- Each checked forecast has 85 native valid times from lead 0 to 360 h and a
  0.25-degree Atlantic grid.
- ERA5 truth proxy: hourly wind and wave files covering 15 July--8 August 2025.
- A complete 15 July adaptive voyage therefore requires five additional IFS
  forecast pairs: 25--29 July.
- No ensemble forecast archive is present for this pilot.

## Required algorithm

Let total passage time be `T = 354 h`, decision times be `t_d = 24d h`, and
`F_d` be the forecast initialized for decision day `d`.

At decision `d`:

1. The vessel state is the exact endpoint of the already sailed prefix.
2. The remaining time is `T_d = T - t_d`.
3. Load `F_d` and optimize a route from the current position to USNYS with
   fixed duration `T_d`.
4. Advance the vessel by elapsed time, not by distance or waypoint count:
   take the portion at normalized route time `min(24, T_d) / T_d`.
5. Append that portion permanently to the sailed track.
6. Score the realized segment in ERA5 for its true start time and duration.

The adaptive realized energy is the sum of segment energies:

`E_R(adaptive) = sum_d E_R(segment_d; t_d, min(24, T_d))`.

This segment sum avoids a timing error that would arise if a concatenated track
with unequal daily point counts were scored as though all of its segments had
equal duration.

The final comparison is made entirely under ERA5:

`saving(route) = E_R(Great Circle) - E_R(route)`

for the Great Circle, a priori, adaptive, and ERA5-informed routes.

## Blocking code findings

1. `run_optimised_departure` obtains source, destination, and nominal duration
   from the fixed SWOPP3 case. It cannot currently optimize from the vessel's
   daily position without refactoring.
2. Although a shorter `travel_time` can be passed into the optimizer, the
   runner evaluates the CMA-ES and FMS routes using the case's full 354 h. That
   would make adaptive suffix costs wrong.
3. CMA-ES fixes only the two route endpoints. FMS updates every interior
   waypoint. Neither method currently locks the incoming tangent at a daily
   splice, so the real implementation guarantees position continuity (C0) but
   not tangent continuity (C1).
4. The presentation storyboard constructs daily suffixes by blending the 22
   July a priori route with the ERA5-informed route and adding a sinusoidal
   perturbation. It therefore leaks hindsight geometry and is unsuitable for
   quantitative use, even though it is visibly labelled as a concept.
5. The stored July pilot uses one seed and a reduced budget: 120 output points,
   10 free Bezier controls, population 50, 1,000 CMA-ES evaluations, and 500
   FMS evaluations. The so-called perfect route is therefore an
   ERA5-informed finite-budget benchmark, not a proven global optimum.

## Experimental cautions

- ERA5 is a reanalysis truth proxy, not direct observation.
- Ten consecutive departures have strongly overlapping weather windows and
  are not ten independent samples.
- July is a stable-season pilot; January forecasts must be acquired before the
  winter experiment can be run.
- Optimizer noise must be separated from forecast-information effects using
  repeated seeds or a convergence check.
- The daily replan should initially be reported as C0-continuous unless an
  explicit tangent constraint is implemented and tested. Post-hoc smoothing
  is not acceptable because it changes the optimized route and its cost.

## Acceptance tests before publishing an adaptive result

- Every day uses a different forecast initialization and records its file hash.
- The first point of each new plan equals the final point of the sailed prefix.
- The final point is the SWOPP3 endpoint USNYS, not USNYC.
- Remaining time decreases exactly as 354, 330, ..., 18 h.
- The segment taken from each plan corresponds to 24 h, except the final 18 h.
- ERA5 segment timestamps cover the voyage continuously with no gaps or
  overlaps.
- Summed segment duration is exactly 354 h.
- Saved plans reproduce the animated vessel track exactly.
- Great Circle, a priori, adaptive, and ERA5-informed energies use the same
  performance model, endpoints, passage time, and ERA5 fields.
- Repeated-seed sensitivity is reported, or the single-seed limitation is
  explicit.
