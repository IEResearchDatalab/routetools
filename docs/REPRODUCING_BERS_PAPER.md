# Reproducing the BERS paper (Ocean Engineering, OE-D-26-09120)

This guide lists everything needed to reproduce the results of _BERS: A
Continuous Hybrid Evolutionary–Variational Algorithm for Maritime Weather
Routing with Just-in-Time Arrival_: the code, the vessel performance model,
the data, and the script that produces each table and figure.

> **Status:** draft for the revision. Items marked **TODO** are still being
> completed (owner: Francisco Suárez); the rest is in the repository today.

## 1. Environment

```bash
git clone https://github.com/Weather-Routing-Research/routetools.git
cd routetools
uv sync            # Python 3.12, JAX 0.8.1 (see pyproject.toml / uv.lock)
```

The synthetic experiments and the local-optimality checks (Tasks 9–10) run on
a laptop CPU. The real-ocean runs were made on a Linux server with SLURM (the scripts
request 64 CPU cores per job).

## 2. Components

| Component                             | Where                                                   |
| ------------------------------------- | ------------------------------------------------------- |
| CMA-ES stage (Bézier routes)          | `routetools/cmaes.py`                                   |
| FMS refinement                        | `routetools/fms.py`                                     |
| Land mask, penalty and rollback       | `routetools/land.py`, `routetools/fms.py`               |
| Weather penalties                     | `routetools/weather.py`                                 |
| Vessel performance model (Appendix D) | `routetools/performance.py`, `docs/parametric_model.md` |
| Energy integration                    | `routetools/cost.py` (`cost_function_rise`)             |
| SWOPP3 corridors and passage times    | `routetools/swopp3.py`                                  |

The performance model is an open analytical reconstruction of the SWOPP3
evaluator. Its agreement with the original compiled evaluator is tested in
`tests/test_parametric_model.py`.

## 3. Data

The real-ocean experiments use ERA5 reanalysis for 2024 (10 m wind,
significant wave height, mean wave direction) over the two SWOPP3 corridors.
The files (about 20 GB) are not redistributed; download them with:

```bash
uv run scripts/download_era5.py              # Google Cloud ERA5 archive, no key
uv run scripts/download_era5.py --backend cds  # or the Copernicus CDS API
```

The files are written to `data/era5/` with the names the run scripts expect.
This is the same procedure used by the SWOPP3 benchmark participants.
**TODO:** link to the public benchmark page.

## 4. Results

### Synthetic fields (Section 3)

| Paper item                           | Command                                                                                        |
| ------------------------------------ | ---------------------------------------------------------------------------------------------- |
| Tables 3, 6; Figures 2–4             | `uv run scripts/synthetic/results.py` then `uv run scripts/synthetic/figures.py` / `tables.py` |
| Table 5 (seed dispersion)            | `uv run python revision/task2_synthetic_dispersion.py`                                         |
| Land experiments, Figures 5–6        | `uv run scripts/results_land_avoidance.py`                                                     |
| Table 8 (λ_land sensitivity)         | `uv run python revision/task6_lambda_land_sensitivity.py`                                      |
| Local-optimality audit (Section 3.3) | `uv run python revision/task10_synthetic_local_optimality.py`                                  |

### Real-ocean corridors (Section 4)

Route generation (server):

```bash
sbatch scripts/swopp3_slurm_atlantic_k10.sh        # CMA-ES, Atlantic
sbatch scripts/swopp3_slurm_pacific_k15_p400.sh    # CMA-ES, Pacific
bash   scripts/run_fms_sweep_combined_strict.sh    # FMS refinement, both
```

Post-processing (no route is re-optimized):

| Paper item                                | Command                                                                     |
| ----------------------------------------- | --------------------------------------------------------------------------- |
| Tables 12, 15; Figures 1, 7–9, 11         | `uv run scripts/realworld/figures.py`, `uv run scripts/realworld/tables.py` |
| Table 10, Figure 7 (segment speeds)       | `revision/task1_speed_distribution.py`                                      |
| Table 13, Figure 10 (paired improvements) | `revision/task3_paired_improvements.py`                                     |
| Table 16, Figure 12 (weather exposure)    | `revision/task4_weather_violations.py`                                      |
| Table 11 (exact land audit)               | `revision/task7_land_verification.py`                                       |
| Table 14 (route-wise derivative audit)    | `revision/task8_local_optimality.py`                                        |
| Operating-envelope Hessian (Appendix E)   | `revision/task9_envelope_hessian.py`                                        |
| Distance to a certified local minimum     | `revision/task11_real_ocean_newton_polish.py`                               |

`revision/run_revision_postprocessing.sh` and
`revision/run_revision_local_optimality.sh` run these steps in order and
record provenance (git commit, host, settings).

**TODO:** publish the final route files (CSV tracks for all 366 departures ×
4 configurations) with a DOI (Zenodo) so that the post-processing can be run
without regenerating the routes.

## 5. What depends on external data

- Exact bit-for-bit agreement with the original compiled SWOPP3 evaluator
  requires that binary. The open model agrees with it to 0.103 kW maximum
  absolute error (0.031% relative).
- The other benchmark participants' scores are held by the SWOPP3 organizers.
