# Review 2 verification report

This branch contains one auditable chain of Review 2 evidence. It preserves the
completed routes and does not rerun CMA-ES or FMS.

## Task 0: run identity and configuration

- Manuscript-number roots: `output/sweep_combined` for GC/CMA-ES and
  `output/sweep_combined_fms_strict` for BERS; 366 paired departures per case.
- The FMS launcher used 19.9 m/s and 6.9 m activation buffers rather than the
  20 m/s and 7 m reporting thresholds.

## Task 1: reproduction, common weather, and input range

- The conditional gate covers 4,392 routes and 1,029,924 saved-waypoint
  segments, with zero unexpected failures.
- Seventy-eight late BERS records expose a historical end-of-2024 weather
  continuation mismatch. All routes were re-scored with common ERA5 2024 plus
  January-2025 weather; route geometry was not re-optimized.
- The largest headline change is Pacific/WPS BERS versus CMA-ES paired energy
  saving: 30.579% legacy versus 27.832% common weather.
- No CMA-ES or BERS segment is outside the documented model-input range.
  Great-circle incidence is 0.0278% Atlantic and 0.1020% Pacific.
- Clipping changes any same-mode mean saving by at most 0.037 percentage
  points. This is a sensitivity bound, not a physical correction.

## Task 2: stage-specific objectives

- `J_CMA`, normalized FMS update action `J_FMS`, and the best-iterate tracker
  are different functionals. FMS has no coastline objective and uses land only
  through rollback.
- Mean BERS-minus-CMA `J_FMS` changes are -12.593, -21.160, -0.607, and
  -17.842. It decreases on 356, 360, 352, and 350 of 366 departures.
- Six BERS routes activate the common raster-mask land count post hoc; this is
  distinct from the Natural Earth polygon audit.

## Task 3: Figure 9

- White bars are median voyage energies.
- Annotations are savings of marginal medians against the same-WPS GC median,
  not median paired savings. Table 11 uses reductions of means; Table 12 uses
  means of paired per-departure percentage reductions.

## Task 4: route-wise certificates

- All 20 travel-time routes certify only in a 198-variable transverse family.
  Maximum offset is 0.00291145 and maximum relative gap is 3.763e-5.
- Swirlys seeds 1, 3, and 4 certify after Newton post-processing in the full
  396-dimensional space; maximum per-waypoint displacement is 0.547930494 and
  maximum relative gap is 3.563e-4. Seeds 0 and 2 remain inconclusive.

The scripts regenerate the machine-readable evidence and reports under
`revision2/`. Those generated artifacts are intentionally ignored; the compact
checked-in status is `SUMMARY2.md`.

## Follow-up verification and propagation (C1--C6)

- C1 regenerates the seasonal and weather-exposure evidence and all requested
  PDFs from the common-weather artifacts. The preserved Review 1 exposure CSV
  does not reproduce the manuscript's BERS incidence rows, so the generated C1
  report records that provenance discrepancy instead of treating it as
  resolved.
- C2 decomposes every paired final-minus-CMA functional change and identifies
  the saved routes that activate the raster land-replacement rule. The
  Pacific/no-WPS sign is explained by those activations.
- C3 is an executable, read-only manuscript audit. Its generated report lists
  every checked table, prose, and regenerated-PDF annotation, including every
  mismatch at manuscript display precision.
- C4 maps each Section 2.4 equation to its implementing code and records the
  saved-route maximum-energy check and the candidate-history limitation.
- C5 provides the affected-record table and the reproduction statistics for
  every remaining case/strategy group.
- C6 reconciles the manuscript-number roots with both configuration sources.
  The optimizer settings match those roots, but the activation buffers mean
  Section 4.1 is not an exact description of the executed FMS configuration.

The follow-up pipeline generates a manifest mapping each relevant table and
figure to its producer, evidence root, output, and commit. `SUMMARY2.md` is the
compact checked-in C1--C6 handoff.
