# Forecast revision animation review

Verdict: approved for the animation milestone. No unresolved critical or major code findings. Final full-animation render and output checks remain required before delivery.

## Scientific meaning

The target is fixed at 30 January 2025 00 UTC. The 31 scheduled forecast initializations advance in 12-hour steps from 15 January 00 UTC (+360 hours) through the target initialization (+0 hours). The three forecast panels show perturbed members 8 and 47 selected reproducibly with seed 20250130, and control member 0. The fourth panel holds the same ERA5 wind/wave reference fixed. Member IDs are explicitly local to an initialization, not continuous scenarios across cycles. The endpoint is not represented as observed truth, and no monotonic improvement is assumed.

## Resolved review findings

The first reader verified each shard against its own receipt and fixed target but did not check the scheduled initialization/lead/stream. This could allow valid files from another initialization with the same target to be displayed under the wrong frame label. The final reader constructs the expected request from the scheduled frame; request-aware checksum/content verification checks exact initialization, lead, stream, members, variables and grid before selecting display members.

Wave RMSE comparisons use the same finite ocean mask across both perturbed members, control and ERA5 and record the eligible cell count. Display fields retain their actual source masks. Forecast captions now say initialization and distinguish it from publication. Fractional-second targets are rejected. The white-on-white wind key was changed to a dark arrow.

## Checks

- One common wave colour scale is computed over all rendered forecast panels/frames and ERA5.
- Wind and waves use the same member and initialization, with matching coordinates.
- Wind u/v arrows point toward motion, use a fixed scale and are hidden on each panel's missing-wave mask.
- ERA5 is loaded once and is not updated by the animation callback.
- Frames use discrete archived issues; no weather-time interpolation or vessel model is introduced.
- Source records include forecast and reference hashes, seed and display scale.
- The initial preview was visually inspected: panel layout, coastlines, labels and wind/wave overlays are legible. The final full-animation outputs must still be inspected after acquisition finishes.

Added an end-to-end regression rejecting a valid other-initialization receipt for the same target, plus fractional-second target rejection. Verification:

```sh
PYTHONPATH=. .venv/bin/python -m pytest tests/test_forecast_animation.py tests/test_weather_acquisition.py -q
```

Result: 15 passed in 0.30 seconds. Ruff checks for the modified animation test file pass. Acquisition implementation was not changed for this milestone review.

## Limits

The figure illustrates forecast revisions for one January target. It does not establish ensemble calibration, seasonal skill or rerouting benefits. Actual historical publication times remain unknown. ERA5 is a retrospective reference with related model/observation ancestry. Separate scalar wind components and wave directions remain available in canonical data, although this visualization displays wave height and wind vectors only.

## Timeline and detailed coastline followup

Approved after the followup review: no unresolved critical or major findings.

The added timeline advances its initialization marker by the exact elapsed hours `360 - lead`, while the target remains fixed at 30 January 2025 00 UTC. Detailed Natural Earth land polygons now supply both fill and boundary lines, with the same geometry used to classify actual source-grid centres. Explicit midpoint-derived cell edges preserve descending latitude and ascending longitude; no weather-coordinate shift or interpolation is applied. Missing wave data over ocean has a separate pale-blue background rather than appearing as land. Natural Earth `10m` is correctly identified as map scale 1:10 million, not ten-metre spatial resolution.

The geographic mask is now explicitly part of the marine display and descriptive score scope. Wave RMSE uses the common finite samples from all three displayed forecasts and ERA5, excluding the same Natural Earth land mask, and per-initialization counts are saved. This changes eligibility relative to the earlier raw-wave-mask-only scores, and is declared in the manifest. Canonical NetCDF contents and coordinates remain untouched.

Added tests cover exact cell edges in both coordinate orders, preserved cell centres, invalid singleton coordinates, land/ocean display masking and non-aliasing/nonmutation of the input dataset and mask. The final combined animation/acquisition suite passes: 18 tests, 1.37 seconds. Ruff checks for the changed test file pass.

Reviewed the updated first-frame image: timeline, detailed Canadian islands/coastlines, distinct land/no-wave-ocean fills and wind scale are visible. The legend was subsequently moved down to avoid longitude tick overlap; final rendered output still needs the normal visual check. The common GIF palette is defined once for all frames, supporting a stable appearance of the fixed ERA5 panel.

The saved source-check JSON reports 30 Canadian-region samples per field: forecast u/v differences zero; significant wave height maximum difference about 2.2e-7 (float32 rounding), with 17 matching missing values. The independent ARCO comparison reports exact matches for the sampled ERA5 u/v/height and 14 matching missing wave values. These are bounded samples supporting the absence of a source-coordinate shift at the checked locations, not a domain-wide certification.

## Versioned whole-grid coordinate audit

Approved the additional read-only audit helper `scripts/data/audit_forecast_coordinates.py` and its pure coordinate-index tests. No unresolved critical or major findings.

The helper maps decoded native GRIB latitude/longitude arrays explicitly, including different longitude origins, rather than assuming an index offset. It checks the selected two-dimensional coordinate arrays, missing-data bitmaps and saved values. Exact expected initialization/lead/stream requests are verified for forecast files. The ERA5 path now also verifies its expected receipt and explicitly compares both saved and original selected coordinates, closing a review finding where unchanged ERA5 values could previously conceal shifted saved coordinate labels. Only the report is written; canonical data is not modified. Native ecCodes imports remain lazy for the pure tests.

The stored audit reports all 50,901 Atlantic cells for each of the first/last control u/v/height fields and target ERA5 u/v/height. Forecast wind and target ERA5 comparisons are exact; wave-height differences are below 2.2e-7, consistent with float32 conversion, and missing masks agree. This is a whole-grid check for those specific fields/times, not certification of every ensemble member or date. Final report provenance includes audit time and the audit script hash.

The combined targeted suite was independently run with the coordinate-audit, animation, acquisition and older download tests: 24 passed in 2.10 seconds. The final audit should be rerun from the frozen helper so its saved provenance identifies the version committed for this milestone.
