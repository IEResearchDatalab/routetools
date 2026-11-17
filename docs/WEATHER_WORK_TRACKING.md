# Weather data work tracking

This records the final status after the initial acquisition and animation work
on 12 November 2026. Earlier reports describe the initial pilot snapshot and
their statements about unpushed work are historical.

## Branches and pull requests

- [PR87](https://github.com/IEResearchDatalab/routetools/pull/87): existing
  `codex/forecast-aware-toy` foundation against `main`.
- [PR88](https://github.com/IEResearchDatalab/routetools/pull/88):
  `codex/weather-data-january`, stacked on the foundation branch.
- Acquisition implementation: `c3c1f81`.
- Animation, timeline and independent source audit: `50d2360`.

Both branches are pushed. The PRs are drafts; no merge was performed. Code,
tests and documentation are in Git. Shared data and media remain outside Git.

## Final validation

The frozen code passed 37 targeted tests and all commit checks. The source
audit compared 458,109 values over all Atlantic cells for the first/last
control u/v/SWH and target ERA5: wind and ERA5 exact, maximum wave-height
difference 2.2e-7 from float32 rounding. This does not certify every date/member.

The final GIF has 31 distinct forecast states, 1400×900 pixels and a duration
of 27.2 seconds. The decoded ERA5 map crop is pixel-identical in every frame.
Every initialization plus lead equals the fixed target; source-code hashes
match the frozen scripts. First/final PNGs were inspected, including the
timeline positions, Canadian coastline, legends and constant colour scale.

The animation sources include all 51 members for 31 initializations of the
fixed target 30 January 2025 00 UTC. This adds 60 shards to the original pilot,
bringing this January collection to 800 NC shards. No annual job or new route
optimization was launched.

Artifacts, manifests and output checks:
`/home/SHARED/data/weather-routing/diagnostics/fixed_target_20250130T0000Z/`.
Copies are in the OneDrive Paper Number2 output folder. Geographic land and
missing wave cells are rendered separately; source weather coordinates were
not shifted.
