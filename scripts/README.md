# Script guide

This directory contains runnable research workflows built on the installable
`routetools` package. Run commands from the repository root so that relative
paths resolve consistently. Install the locked Python 3.12 environment first:

```bash
uv sync
uv run python scripts/<category>/<script>.py --help
```

CPU workflows work on all supported platforms. GPU and SLURM launchers assume a
Linux host and site-specific scheduler, storage, and CUDA settings; review those
settings before submitting them. Input datasets belong under `data/`, and all
generated tables, figures, routes, checkpoints, manifests, and logs belong
under the ignored `output/` tree. The source tree should remain unchanged after
a run.

## Categories

- `benchmarks/` compares algorithms, datasets, or performance models.
- `data/` downloads and cross-checks ERA5 inputs.
- `demos/` creates small visual demonstrations of library behavior.
- `environment/` configures an interactive execution environment.
- `experiments/` runs parameter searches and diagnostic sweeps.
- `paper/` reproduces paper results, review analyses, and formal certificates.
  The synthetic paper workflow keeps its pinned `config_land.toml` and
  `config_noland.toml` beside the scripts that consume those experiment
  settings.
- `swopp3/` runs and analyzes the SWOPP3 weather-routing pipeline; `hpc/`
  contains cluster launchers.
- `validation/` independently checks scorers and submitted routes.

The catalog below is alphabetical by path. Each entry states the principal
action and its expected output. Modules marked **support module** are imported
by a neighboring entry point and are not normally run directly.

## Script catalog

| Script | What it does and produces |
| --- | --- |
| `benchmarks/era5_benchmark.py` | Optimizes and refines a New York–Hamburg route with ERA5 data and writes `output/era5_benchmark_usnyc_deham.jpg`. |
| `benchmarks/parametric_benchmark.py` | Compares the open analytical vessel model with the optional SWOPP3 reference and prints component and error statistics. |
| `benchmarks/results_on_literature_vector_fields.py` | Runs the literature vector-field benchmark suite and writes its aggregate benchmark results under `output/`. |
| `data/compare_era5_sources.py` | Compares GCS and CDS ERA5 subsets and prints variable-wise numerical differences, optionally caching GCS files under `data/era5_gcs/`. |
| `data/download_era5.py` | Downloads Atlantic and/or Pacific wind and wave NetCDF inputs from GCS or CDS into `data/era5/`. |
| `demos/demo_land_avoidance.py` | Compares great-circle and land-avoiding routes between configurable ports and writes `output/demo_land_avoidance.png`. |
| `demos/draw_fms.py` | Animates unconstrained FMS route refinement and writes `output/fms.gif`. |
| `demos/draw_fms_land.py` | Animates FMS refinement with land avoidance and writes `output/fms.gif`. |
| `demos/draw_fms_weather.py` | Animates weather-aware FMS refinement on a synthetic field and writes `output/fms_weather.gif` by default. |
| `demos/swopp_demo.py` | Runs the compact SWOPP routing demonstration and renders its route comparison. |
| `environment/activate_gpu.sh` | Exports GPU-first JAX environment variables in the current shell and produces no files. |
| `experiments/gradual_land_penalty.py` | Sweeps the synthetic land-penalty strength and writes per-step images plus `output/results_penalty.csv`. |
| `experiments/parameter_search.py` | Evaluates CMA-ES parameter combinations and writes resumable JSON records and `output/results.csv`. |
| `experiments/parameter_search.sh` | Runs `parameter_search.py` in fixed-size batches and produces the same search artifacts. |
| `experiments/plot_comptime_vs_popsize.py` | Plots computation time against CMA-ES population size and writes `output/comptime_vs_popsize_<label>.png`. |
| `experiments/single_run.py` | Runs one configured route optimization and writes `output/single_run.png`. |
| `experiments/slurm_sweep_atlantic.sh` | Submits the Atlantic generalization sweep and writes scheduler logs plus `output/sweep_atlantic.csv`. |
| `experiments/slurm_sweep_stage_a.sh` | Submits the Pacific catastrophic-detour sweep and writes scheduler logs plus `output/sweep_stage_a.csv`. |
| `experiments/slurm_sweep_stage_b.sh` | Submits the Pacific GC-stagnation sweep and writes scheduler logs plus `output/sweep_stage_b.csv`. |
| `experiments/sweep_atlantic.py` | Tests Pacific-tuned parameters on Atlantic departures and writes `output/sweep_atlantic.csv` by default. |
| `experiments/sweep_stage_a.py` | Searches penalty and exploration settings for severe Pacific detours and writes `output/sweep_stage_a.csv`. |
| `experiments/sweep_stage_b.py` | Searches exploration and convergence settings for GC-stuck Pacific cases and writes `output/sweep_stage_b.csv`. |
| `experiments/sweep_ww_analysis.py` | Re-evaluates wind–wave sweep tracks on a common temporal grid and writes `output/sweep_ww_summary.csv`. |
| `paper/real_world/clean.py` | Removes invalid rows from legacy real-world benchmark intermediates and writes cleaned CSV data under `output/json_benchmark/`. |
| `paper/real_world/figures.py` | Builds the legacy real-world dataframe and paper figures under `output/`. |
| `paper/real_world/results.py` | Executes the legacy real-world benchmark cases and writes per-run JSON files under `output/json_benchmark/`. |
| `paper/real_world/tables.py` | Converts real-world aggregate results into paper-ready LaTeX tables under `output/`. |
| `paper/reproducibility/synthetic_local_optimality/action.py` | **Support module:** evaluates the discrete action, exact gradient, and Hessian in floating-point or interval arithmetic. |
| `paper/reproducibility/synthetic_local_optimality/certify.py` | Certifies strict local minimality for a reference route and writes JSON certificates under `certificates/`. |
| `paper/reproducibility/synthetic_local_optimality/certify_bers_routes.py` | Certifies all 25 exported BERS routes and writes route arrays, per-route certificates, and `summary.json`. |
| `paper/reproducibility/synthetic_local_optimality/certify_full_hp.py` | Produces a high-precision full-space certificate and saves its JSON certificate and polished center. |
| `paper/reproducibility/synthetic_local_optimality/certify_route.py` | Certifies one supplied route and writes a sibling `_certificate.json` file. |
| `paper/reproducibility/synthetic_local_optimality/family.py` | **Support module:** defines transverse route deformations used by the travel-time certificates. |
| `paper/reproducibility/synthetic_local_optimality/find_routes.py` | Finds stationary reference routes with trust-region Newton refinement and writes `routes/<problem>.npy`. |
| `paper/reproducibility/synthetic_local_optimality/full_space_probe.py` | Probes full-space gradients and Hessians for diagnostic comparison and prints the result. |
| `paper/reproducibility/synthetic_local_optimality/global_search.py` | **Support module:** reimplements the CMA-ES Bézier global-search stage used by the certificate workflow. |
| `paper/reproducibility/synthetic_local_optimality/hp.py` | **Support module:** provides the 150-bit interval gradient and norm bounds used for polishing and certification. |
| `paper/reproducibility/synthetic_local_optimality/jets.py` | **Support module:** implements rigorous interval arithmetic and second-order automatic differentiation. |
| `paper/reproducibility/synthetic_local_optimality/polish_hp.py` | Polishes a reference route with a high-precision gradient and overwrites the corresponding route artifact. |
| `paper/reproducibility/synthetic_local_optimality/problems.py` | **Support module:** defines the five synthetic certificate problems independently of the library implementation. |
| `paper/reproducibility/synthetic_local_optimality/reference_numpy.py` | **Support module:** provides an independent NumPy objective used to cross-check the interval implementation. |
| `paper/reproducibility/synthetic_local_optimality/refine.py` | Refines a CMA-ES curve to a full-space stationary route and writes its NumPy route and JSON summary. |
| `paper/reproducibility/synthetic_local_optimality/refine_family.py` | Refines a route in the transverse family and writes `routes/<problem>_family.npz`. |
| `paper/reproducibility/synthetic_local_optimality/run_all.sh` | Runs the complete synthetic certificate pipeline and populates `routes/` and `certificates/`. |
| `paper/reproducibility/synthetic_local_optimality/run_stage1.py` | Runs certificate-stage CMA-ES for named single-basin problems and writes initial route artifacts. |
| `paper/reproducibility/synthetic_local_optimality/run_stage1_multi.py` | Runs multiple CMA-ES starts for the multi-basin problems and keeps the selected route artifacts. |
| `paper/reproducibility/synthetic_local_optimality/selftest.py` | Cross-checks interval and reference objective derivatives and prints pass/fail diagnostics. |
| `paper/revision1/run_revision_local_optimality.sh` | Runs Revision 1 Tasks 9–11 and writes provenance, checkpoints, summaries, and a completion marker below `output/revision_local_optimality/`. |
| `paper/revision1/run_revision_postprocessing.sh` | Runs the six final post-processing audits against a completed experiment and writes a provenance-rich analysis directory. |
| `paper/revision1/task1_speed_distribution.py` | Computes real-ocean segment-speed distributions and writes route-level CSV, summary CSV/LaTeX, and a PDF plot. |
| `paper/revision1/task10_synthetic_local_optimality.py` | Re-runs 25 synthetic BERS cases and writes route-wise derivative audits and polished-route evidence. |
| `paper/revision1/task11_real_ocean_newton_polish.py` | Applies diagnostic coupled-Newton polishing to saved ocean routes and writes per-route and summary CSV/LaTeX outputs. |
| `paper/revision1/task2_synthetic_dispersion.py` | Runs five seeds on five synthetic fields and writes run-level and dispersion summary CSV/LaTeX outputs. |
| `paper/revision1/task3_paired_improvements.py` | Computes paired BERS gains over GC and CMA-ES and writes raw metrics, summary CSV/LaTeX, histogram bins, and PDF plots. |
| `paper/revision1/task4_route_provenance.py` | Compares exported routes with GC and CMA-ES precursors and optionally writes a route-provenance CSV. |
| `paper/revision1/task4_weather_violations.py` | Audits wind and wave exceedances and writes route/segment CSVs, summary CSV/LaTeX, curve data, and a PDF. |
| `paper/revision1/task5_env_penalty_sensitivity.py` | Re-runs FMS on stress cases across weather weights and writes auditable routes, CSV/LaTeX summaries, and a PDF. |
| `paper/revision1/task6_lambda_land_sensitivity.py` | Sweeps land-penalty strength on synthetic scenarios and writes run-level and aggregate CSV/LaTeX results. |
| `paper/revision1/task7_land_verification.py` | Checks route segments against exact land polygons and writes a route CSV and verification report. |
| `paper/revision1/task7_landfree_export.py` | Creates a minimally repaired polygon-safe route export and writes a coordinate-change manifest after verification. |
| `paper/revision1/task8_local_optimality.py` | Audits stationarity, curvature, smoothness, and feasibility per route and writes checkpointed CSV/LaTeX/report artifacts. |
| `paper/revision1/task9_envelope_hessian.py` | Evaluates vessel-model curvature over its validated domain and writes grid evidence and summary outputs. |
| `paper/revision2/c3_consistency_check.py` | Checks manuscript values against Revision 2 evidence and writes `c3_report.md`. |
| `paper/revision2/followup_verification.py` | Generates C1, C2, and C4–C6 follow-up tables, figures, reports, summary, and manifest under `output/paper/revision2/`. |
| `paper/revision2/task0_config_reconciliation.py` | Reconciles manuscript configurations with preserved runs and writes a CSV plus `task0_report.md`. |
| `paper/revision2/task1_out_of_domain_audit.py` | Re-evaluates saved routes for model-domain and clipping sensitivity and writes compressed segment evidence, tables, provenance, and a report. |
| `paper/revision2/task2_common_functionals.py` | Evaluates CMA-ES and FMS objectives on common sampling and writes decomposition, paired summary, provenance, LaTeX, and report files. |
| `paper/revision2/task3_figure9_reconciliation.py` | Recomputes Figure 9 statistics and writes a reconciliation CSV and report. |
| `paper/revision2/task4_certificate_routes.py` | Validates the preserved 25-route certificate batch and writes exported routes, checksums, a reviewer CSV, and a report. |
| `paper/run_bers_revision_2024.sh` | Launches or resumes the complete 2024 BERS experiment and records manifests, logs, provenance, and completion status under `output/<run>/`. |
| `paper/run_results.sh` | Runs the legacy real-world results, figures, and tables sequence and appends progress to its configured log. |
| `paper/synthetic/figures.py` | Reads synthetic result CSVs and writes paper-ready synthetic benchmark figures under `output/`. |
| `paper/synthetic/results_land_avoidance.py` | Runs synthetic land-avoidance experiments and writes their result tables and figures under `output/`. |
| `paper/synthetic/results.py` | Executes the synthetic benchmark matrix using its colocated land/no-land config and writes result CSVs under `output/`. |
| `paper/synthetic/tables.py` | Converts synthetic result CSVs into paper-ready LaTeX tables under `output/`. |
| `swopp3/hpc/swopp3_0526_slurm.sh` | Runs the dated two-stage SWOPP3/FMS pipeline in one SLURM job and writes run outputs and scheduler logs. |
| `swopp3/hpc/swopp3_slurm_atlantic_k10.sh` | Runs the high-budget Atlantic K=10 experiment and writes SWOPP3 outputs plus scheduler logs. |
| `swopp3/hpc/swopp3_slurm_atlantic.sh` | Runs the hourly Atlantic GPU workload and writes corridor outputs plus scheduler logs. |
| `swopp3/hpc/swopp3_slurm_gpu.sh` | Runs the complete GPU SWOPP3 workload and writes route outputs plus scheduler logs. |
| `swopp3/hpc/swopp3_slurm_hard_penalty.sh` | Runs hard weather-penalty variants and writes each experiment directory plus scheduler logs. |
| `swopp3/hpc/swopp3_slurm_k15_popsize400.sh` | Runs a four-case K=15/population-400 SLURM array and writes per-case outputs and logs. |
| `swopp3/hpc/swopp3_slurm_k15_sweep.sh` | Sweeps K=15 normalized penalties and writes experiment outputs and array logs. |
| `swopp3/hpc/swopp3_slurm_max_penalty_sweep.sh` | Sweeps max-based weather penalties in a SLURM array and writes one output directory per configuration. |
| `swopp3/hpc/swopp3_slurm_no_penalty.sh` | Runs the no-weather-penalty baseline and writes its SWOPP3 outputs and logs. |
| `swopp3/hpc/swopp3_slurm_npoints.sh` | Re-evaluates selected routes at several reporting resolutions and writes comparison artifacts and logs. |
| `swopp3/hpc/swopp3_slurm_optimal_params.sh` | Runs the sweep-selected CMA-ES parameters and writes full SWOPP3 outputs and logs. |
| `swopp3/hpc/swopp3_slurm_pacific_k15_p400.sh` | Runs the high-budget Pacific K=15 experiment and writes SWOPP3 outputs plus scheduler logs. |
| `swopp3/hpc/swopp3_slurm_pacific.sh` | Runs the hourly Pacific GPU workload and writes corridor outputs plus scheduler logs. |
| `swopp3/hpc/swopp3_slurm_penalty_sweep.sh` | Runs normalized weather-penalty configurations in a SLURM array and writes per-configuration outputs. |
| `swopp3/hpc/swopp3_slurm_split_penalty.sh` | Runs separate wind and wave penalties and writes full SWOPP3 outputs and logs. |
| `swopp3/hpc/swopp3_slurm_stage.sh` | Stages the repository, environment, and ERA5 data onto node-local scratch and writes staging logs. |
| `swopp3/hpc/swopp3_slurm_wind_wave_sweep.sh` | Sweeps the two-dimensional wind/wave penalty grid and writes one output directory per array task. |
| `swopp3/hpc/swopp3_slurm.sh` | Runs the complete CPU SWOPP3 workload and writes route outputs plus scheduler logs. |
| `swopp3/hpc/swopp3_submit.sh` | Submits staging followed by parallel Atlantic and Pacific jobs and prints the dependency job IDs. |
| `swopp3/hpc/vangelis_pacific_noconstraint.sh` | Runs the Pacific no-constraint GPU experiment on the documented standalone host and writes its output directory and log. |
| `swopp3/plot_noconstraint_comparison.py` | Compares constrained and unconstrained Pacific outputs and writes four figures below the unconstrained run. |
| `swopp3/run_fms_sweep_combined_strict.sh` | Resumably applies strict FMS settings to `output/sweep_combined` and writes `output/sweep_combined_fms_strict`. |
| `swopp3/run_fms_sweep_combined.sh` | Resumably applies FMS to `output/sweep_combined` and writes `output/sweep_combined_fms`. |
| `swopp3/swopp3_analysis.py` | Generates the full comparative figure set and `table01_summary.csv` under `output/analysis/` by default. |
| `swopp3/swopp3_animate_single_departure.py` | Animates one optimized departure and writes a GIF under `output/figures/` by default. |
| `swopp3/swopp3_apply_fms_to_scored_submissions.py` | Extracts scored submissions, applies FMS, and writes refined submission folders and before/after comparison tables. |
| `swopp3/swopp3_apply_fms.py` | Refines every non-GC route in an existing output and writes a resumable sibling `_fms` directory. |
| `swopp3/swopp3_npoints_comparison.py` | Resamples selected optimized tracks and writes energy comparisons for multiple waypoint counts. |
| `swopp3/swopp3_plot_routes.py` | Draws publication-quality Atlantic and Pacific route maps into `<input-dir>/figures/`. |
| `swopp3/swopp3_run.py` | Runs configured SWOPP3 cases and writes File A summaries, track CSVs, logs, and an experiment manifest. |
| `swopp3/swopp3_submission_compare.py` | Validates submission folders and writes participant/month comparisons, line charts, tables, and an optional animation. |
| `validation/compare_scorers.py` | Evaluates one route with the JAX and independent NumPy scorers and prints intermediate differences. |
| `validation/validate_routes.py` | Checks one track or a directory against exact land polygons and prints per-file and aggregate validation results. |

## Reproduction entry points

- Paper workflow: follow [`../docs/REPRODUCING_BERS_PAPER.md`](../docs/REPRODUCING_BERS_PAPER.md).
- Paper-specific synthetic configurations are documented beside their runner
  in [`paper/synthetic/README.md`](paper/synthetic/README.md).
- Full real-ocean experiment: `bash scripts/paper/run_bers_revision_2024.sh <run-name>`.
- Revision 1 post-processing: `bash scripts/paper/revision1/run_revision_postprocessing.sh <run-root>`.
- Synthetic local-optimality certificates: change into
  `scripts/paper/reproducibility/synthetic_local_optimality/` and run
  `bash run_all.sh`.
- SWOPP3 local pipeline: run `uv run scripts/data/download_era5.py`, then
  `uv run scripts/swopp3/swopp3_run.py`.
