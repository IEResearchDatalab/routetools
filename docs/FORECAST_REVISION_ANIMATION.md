# Fixed target wind and wave animation

The target is 30 January 2025 at 00 UTC. Every frame forecasts that exact instant;
only initialization changes, from 15 January 00 UTC (+360 h) through 30 January
00 UTC (+0 h), in 12 hour steps. This shows forecast revisions, not weather
advancing through time. The target reuses the first pilot's 15 day horizon;
it was not selected to maximize error or demonstrate monotonic improvement.

## Four panels

- Top left: perturbed member 8.
- Top right: perturbed member 47.
- Bottom left: control 0 (the 2025 deterministic-equivalent forecast).
- Bottom right: final ERA5 reference for the target, unchanged throughout.

Members 8/47 were sampled uniformly without replacement with NumPy seed 20250130.
Member numbers are local to each initialization, not persistent scenarios
across issues. All 51 members are retained in each production shard, although
only three are displayed. Wind/wave initialization, lead and member IDs must
match exactly. An exact scheduled request and decoded metadata check rejects
a shard from another initialization even if it has the same target time.

Wave colour is significant wave height in metres, with a fixed 0-to-global-max
scale across every panel/frame. White arrows show 10 m wind u/v, pointing toward
motion; arrow scale is fixed and the key is 10 m/s. Coastlines/land use cached
Natural Earth 1:10 million geometry, matching the earlier SWOPP land treatment.
The timeline shows the current initialization between 15 and 30 January.
No boats, invented intermediate forecasts, nearest-time matching or temporal
interpolation are used. The ERA5 panel uses
the existing final reference shards. Initialization precedes historical
publication; the animation is not an availability/latency simulation.

Weather coordinates are not shifted. Cell edges are explicitly halfway between
source grid centres. Detailed geographic land is independent of missing wave
values: pale blue means ocean without wave data, and grey means land. Marine
display arrays and descriptive wave RMSE exclude geographic land without
altering source NetCDF files. Scores use common finite ocean cells across the
three forecasts and ERA5 at each issue; counts are recorded and may change
between issues.

The GIF uses one global colour palette without dithering, avoiding changes in
the static reference from per-frame colour quantization. There are 31 weather
states, displayed at 0.8 s each, with extra final-frame hold. The full-colour
first/middle/final PNGs are also saved. Source paths/hashes, RNG seed, scale and
source-code hash are recorded in the accompanying manifest. The error CSV uses
the same marine score scope described above.

## Coordinate audit

`scripts/data/audit_forecast_coordinates.py` compares all 50,901 Atlantic grid
cells with original GRIB geography/values for the first and last control u/v
and wave-height fields, and with original ARCO ERA5 for the target. That is
458,109 source-value checks. Saved and source coordinate arrays, exact requests
and missingness are checked independently of the plot. Maximum IFS difference
was 2.2e-7 from float32 rounding; wind and ERA5 matched exactly. Native longitude
origins differ, so matching uses actual coordinate labels, not a zero-origin
array-index assumption. This bounded audit does not certify every member/date.

The older SWOPP fixes inspected were `21117f03` (detailed geographic land rather
than wave NaNs) and `766f7d46` (Pacific longitude normalization). The rendering
uses detailed polygons for both fill and coastline and does not equate the
coarse weather-grid mask with geography.

## Reproduce on vangelis

```bash
cd /home/SHARED/routetools-weather-data
export PYTHONPATH="$PWD"

# Run separately; the two jobs can run together (four HTTP workers each).
.venv/bin/python scripts/data/animate_forecast_revisions.py --fetch enfo --workers 4
.venv/bin/python scripts/data/animate_forecast_revisions.py --fetch waef --workers 4

# After both jobs finish, validate all source shards and render.
.venv/bin/python scripts/data/animate_forecast_revisions.py

# Optional independent source-coordinate/value audit.
.venv/bin/python scripts/data/audit_forecast_coordinates.py
```

Acquisitions call the existing production downloader and resume its verified
shards, under:

```
/home/SHARED/data/weather-routing/ifs/v1/atlantic/2025/01/
```

Outputs:

```
/home/SHARED/data/weather-routing/diagnostics/fixed_target_20250130T0000Z/
```

The target, output directory and sampling seed are CLI options. A new target
also needs canonical ERA5 wind/wave files for that exact UTC day; absence is an
error. The 15 day window requires 00/12 UTC target alignment. Scientific/code review
is recorded in `docs/FORECAST_ANIMATION_REVIEW.md`. ERA5 is a retrospective
reference, not exact observed truth, and zero-lead forecasts need not equal it.
