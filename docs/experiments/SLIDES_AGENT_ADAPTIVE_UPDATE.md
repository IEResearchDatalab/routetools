# Slides agent handoff: adaptive routing and Great Circle update

## Purpose

Update the weather-routing presentation with the new Great Circle baseline and
the audited adaptive-routing protocol. Keep the existing IE University visual
language: navy, cyan, coral, white space, large maps, large numbers, and very
little body text.

The adaptive experiment has not been run. Present it as the next experiment,
not as a measured result.

## Evidence status

### Completed and safe to present

- Ten July departures, 15 to 24 July 2025 at 00 UTC.
- AO_WPS from ESSDR to the SWOPP3 New York point USNYS.
- Fixed passage time: 354 h.
- A priori route optimized with the deterministic IFS forecast initialized at
  departure.
- ERA5-informed route optimized with reanalysis.
- Great Circle, a priori, and ERA5-informed routes all scored in ERA5.
- CMA-ES versus CMA-ES plus FMS comparison.

### Protocol only

- Daily adaptive routing.
- January comparison.
- Ensemble routing.

Do not show adaptive or January energy values. They do not exist yet.

## Non-negotiable corrections

1. Use **USNYS, 40.6 degrees N, 69.0 degrees W**. Do not use USNYC.
2. Call ERA5 the **reanalysis truth proxy**. It is not direct observation.
3. Use **perfect information** only as a short display label. Add the note
   “finite-budget ERA5 optimization” in the speaker notes or footer.
4. State that all route comparisons use the same 354 h passage and ERA5 cost
   model.
5. The adaptive experiment must use a new forecast initialization every day.
   It must not continue using later lead times from the departure forecast.
6. The stored pilot uses one seed and a reduced optimizer budget. Do not imply
   statistical independence or global optimality.
7. Ten consecutive departures have overlapping weather windows. Treat the
   experiment as a pilot, not a population estimate.

## Slide 1: Great Circle baseline

### Title

**Great Circle is the common ERA5 baseline**

### Visual

Use one large chart occupying at least two thirds of the slide.

Preferred chart: ten horizontal rows, one per departure. Place Great Circle at
zero and show the ERA5 energy saving of the a priori and perfect-information
routes. Use cyan for a priori and coral for perfect information. Highlight 23
July because the a priori route lost to the Great Circle.

Alternative: grouped dots for absolute ERA5 energy, connected within each date.
Avoid a dashboard or multiple small cards.

### Large numbers

- **163.2 MWh** mean Great Circle cost
- **10.1%** mean a priori saving
- **19.5%** mean perfect-information saving

### Small annotation

**A priori beat Great Circle on 9 of 10 departures**

### Exact July values

| Departure | Great Circle MWh | A priori saving MWh | Perfect-information saving MWh |
|---|---:|---:|---:|
| 15 Jul | 192.576 | 17.021 | 45.685 |
| 16 Jul | 193.820 | 33.153 | 45.693 |
| 17 Jul | 194.894 | 31.984 | 40.574 |
| 18 Jul | 189.325 | 29.552 | 35.936 |
| 19 Jul | 181.067 | 22.184 | 39.451 |
| 20 Jul | 167.024 | 24.491 | 32.498 |
| 21 Jul | 145.926 | 6.807 | 21.118 |
| 22 Jul | 130.169 | 7.328 | 28.645 |
| 23 Jul | 121.038 | -1.492 | 15.719 |
| 24 Jul | 116.141 | 8.067 | 19.434 |

Mean absolute saving:

- A priori: **17.9 MWh**
- Perfect information: **32.5 MWh**

Use the mean of the ten paired percentages for the percentage claims, not the
percentage calculated from the mean energies.

### Speaker note

“The Great Circle provides the same weather-blind reference for every method.
The forecast route usually improves it, but on 23 July the optimization loses
1.5 MWh under ERA5. Perfect information improves the Great Circle on all ten
departures.”

## Slide 2: Adaptive forecast policy

### Title

**Adaptive routing uses a new forecast every day**

### Main visual

Use a North Atlantic spherical map on the left and a horizontal day sequence
below it. The day sequence should show different forecast stamps:

`15 Jul 00Z`, `16 Jul 00Z`, `17 Jul 00Z`, `...`, `29 Jul 00Z`

The visual must make clear that the forecast itself changes at each decision.
Do not draw synthetic daily routes on this slide.

### Minimal copy

- Freeze the sailed path
- Load the new daily forecast
- Optimize the remaining voyage

### Remaining-time sequence

Use a small sequence or animated counter:

`354, 330, 306, 282, 258, 234, 210, 186, 162, 138, 114, 90, 66, 42, 18 h`

### Method formula

Use only if it remains readable:

\[
\Gamma_d^* = \arg\min_{\Gamma:\,\Gamma(0)=x_d,\,\Gamma(T_d)=\mathrm{USNYS}}
E_{F_d}(\Gamma), \qquad T_d=354-24d
\]

### Visible footnote

**Hindcast convention: the 00Z run is treated as available at 00 UTC**

### Speaker note

“This is an idealized information experiment. A latency-realistic operational
study would use the latest run actually published at the decision time.”

## Slide 3: Realized adaptive route

### Title

**The realized route is assembled one day at a time**

### Animation specification

Build this animation only from the output of the real adaptive runner.

- Current daily plan: cyan.
- Permanently sailed track: navy.
- Previous plans: thin grey at low opacity.
- Vessel: coral point moving exactly along the active plan.
- Great Circle: dark grey dotted line.
- A priori route: cyan dashed line.
- Perfect-information route: coral thin line.
- Use the North Atlantic spherical projection.

At every day boundary:

1. The vessel reaches the exact 24 h position on the active plan.
2. The sailed prefix remains fixed.
3. The old unsailed suffix fades to grey.
4. The new optimized suffix starts at the vessel.

The final static frame should compare four routes:

- Great Circle
- A priori
- Adaptive realized
- Perfect information

Do not display an adaptive energy value until the daily experiment has run and
the assembled track has passed the timing checks.

### Energy definition for notes

\[
E_R(\mathrm{adaptive}) =
\sum_d E_R\!\left(\mathrm{segment}_d; t_d,\min(24,T_d)\right)
\]

The implementation must score the daily segments at their real ERA5
timestamps. It must not concatenate unequal daily point groups and then assign
equal time to every segment.

## Slide 4: Adaptive experiment status

This slide is optional. Use it only if the deck needs an explicit next-step
slide.

### Title

**The first adaptive voyage needs five more forecast runs**

### Visual

Show a simple date strip from 15 to 29 July.

- 15 to 24 July: available, navy.
- 25 to 29 July: missing, coral outline.
- ERA5 through 8 August: available, thin grey band below.

### Minimal copy

- IFS wind and wave forecasts available through 24 July
- Five forecast pairs needed for the 15 July voyage
- January forecasts and ensembles are not yet archived

### Speaker note

“The 15 July voyage is the fastest valid adaptive pilot because only five daily
forecast pairs are missing. A 22 July adaptive voyage would require forecasts
through 5 August.”

## Existing assets: usage rules

Folder:

`C:\Users\danip\OneDrive - IE University\WeatherRouting\paper_02_uncertainty\output\experiment_handoff`

Safe to use as completed evidence:

- `closest_18jul_route_map.png`
- `closest_18jul_route_reveal.gif`
- `largest_22jul_route_map.png`
- `largest_22jul_route_reveal.gif`
- `pilot_results_table.csv`
- `routes.json`

Do not use as measured adaptive evidence:

- `adaptive_daily_rerouting_concept.gif`
- `adaptive_rerouting_storyboard.png`
- `adaptive_three_route_comparison.png`

Those adaptive assets use synthetic route suffixes blended from a priori and
ERA5-informed geometry. They leak hindsight geometry and cannot support an
adaptive result. If the deck must retain one before the real run, label it
**PROTOCOL ILLUSTRATION. ROUTES ARE NOT OPTIMIZED.** Do not show an adaptive
energy or saving beside it.

The packaged CSV may predate the Great Circle backfill. Use the values in this
handoff or refresh the data from the canonical server results:

`/home/SHARED/data/weather-routing/results/toy_20250715_20250724/AO_WPS`

## Visual language

- A priori or current plan: cyan `#78D5EF`
- Adaptive realized route: navy `#000066`
- Perfect information or vessel emphasis: coral `#FF8066`
- Great Circle: dark grey `#4B5563`
- Previous plans: grey `#98A2B3` at 20 to 30 percent opacity
- Land: light grey
- Ocean: almost white

Keep maps large. Put metrics in one narrow area at the right or directly on the
chart. Avoid text over the route geometry.

## Terminology for the final deck

Use:

- **A priori route**
- **Adaptive route**
- **Perfect information**
- **Great Circle baseline**
- **ERA5 reanalysis truth proxy**
- **Deterministic IFS forecast**

Avoid:

- “Real weather”
- “Observed truth”
- “Globally optimal route”
- “Adaptive result” before the experiment has run
- “New York City” for the endpoint

## Source and audit references

- Adaptive audit:
  `C:\Users\danip\Documents\GitHub\routetools\docs\experiments\adaptive_routing_audit.md`
- Existing slide guide:
  `C:\Users\danip\OneDrive - IE University\WeatherRouting\paper_02_uncertainty\output\experiment_handoff\SLIDE_HANDOFF_GUIDE.md`
- Forecast data:
  `/home/SHARED/data/weather-routing/ecmwf-open-data/2025/toy_20250715_20250724/atlantic`
- ERA5 data:
  `/home/SHARED/data/weather-routing/era5/2025/toy_20250715_20250724/atlantic`

## Final quality check

Before delivery, verify that:

- Every map terminates at USNYS.
- Great Circle, a priori, and perfect-information energy values use ERA5.
- The Great Circle comparison includes the 23 July negative a priori saving.
- No adaptive number appears anywhere.
- Any adaptive animation follows the active plan exactly.
- The deck states that daily replanning uses a newly initialized forecast.
- The 00Z availability assumption appears in the notes or footer.
- The maps use a spherical projection.
