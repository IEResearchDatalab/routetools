# BERS final corrected 2024 run

This note records the pre-specified configuration for the final real-ocean
rerun and links each methodological claim to a code/configuration artifact.

## Launch

From the repository root on the Ubuntu server:

```bash
bash scripts/run_bers_revision_2024_timefix.sh
```

The launcher creates a dated run root under `output/`, records the Git commit,
working-tree diff, resolved TOML configuration, ERA5 input metadata, launcher
snapshot, process ID, and run log, and writes `COMPLETED` or `FAILED` at the
run root.

## Frozen method settings

The `bers_revision_final` profile in `config.toml` is the source of truth.

- CMA-ES: `K=10`, `sigma0=0.1`, population 200, 25,000 evaluations.
- FMS: patience 50, damping 0.9, 5,000 evaluations.
- Energy quadrature: 30-minute evaluation spacing.
- Weather: separate smooth TWS and Hs penalty weights of 50; the combined
  legacy weather penalty is zero to prevent double counting.
- Land crossing: finite CMA-ES mask-intersection penalty of `1e6`; FMS rolls
  back updates that increase sampled land violations.
- Coast clearance: shared CMA-ES/FMS inverse-distance weight 100 and epsilon
  1.0. The older generic distance term is zero to prevent double counting.
- Weather thresholds remain soft objective penalties. FMS does not enforce
  TWS/Hs as hard feasibility constraints.
- Weather data are loaded in one-month departure batches with one additional
  month of voyage coverage, then released before the next batch.

The inverse-distance coast term has no literal clearance cutoff. Increasing
its weight from 50 to 100 increases near-coast repulsion but does not prove a
minimum coastline distance. Exact Natural Earth polygon intersection must be
checked after the rerun with `revision/task7_land_verification.py`.

## Stage outputs

One execution writes two complete, paired analysis roots:

- `bers/`: the post-FMS route for every optimised case, plus the GC cases.
- `cmaes/`: the pre-FMS CMA-ES route for every optimised case, plus identical
  GC cases.

The final BERS route is always the FMS stage. It is not replaced by CMA-ES
when raw energy increases or a soft weather threshold is exceeded. This makes
negative as well as positive paired FMS changes observable and keeps the
reported BERS definition consistent with the two-stage method.

## Required post-run checks

1. Confirm the run-root `COMPLETED` marker and absence of `FAILED`.
2. Confirm all eight File A summaries contain 366 departures and their File B
   track files exist in both `bers/` and `cmaes/`.
3. Recompute the paired CMA-ES-to-BERS statistics from the two stage roots.
4. Recompute weather-threshold frequency, magnitude, and duration statistics
   from `bers/`.
5. Run exact geometric land verification over every optimised `bers/` track.
6. Regenerate manuscript tables/figures and update every numerical statement
   in the response letter from these final outputs.
