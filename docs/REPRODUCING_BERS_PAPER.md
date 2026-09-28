# Reproducing the BERS paper (Ocean Engineering, OE-D-26-09120)

This repository contains the complete open-source workflow needed to regenerate
the results of _BERS: A Continuous Hybrid Evolutionary–Variational Algorithm
for Maritime Weather Routing with Just-in-Time Arrival_. It includes the
synthetic benchmarks, CMA-ES and FMS optimization tools, analytical
vessel-consumption models, experiment configurations, validation and
interval-certificate scripts, data-download instructions, and the scripts that
produce every table and figure. No proprietary software or data are required.

## 1. Environment

```bash
git clone https://github.com/IEResearchDatalab/routetools.git
cd routetools
uv sync            # Python 3.12, JAX 0.8.1 (see pyproject.toml / uv.lock)
```

The synthetic experiments and the local-optimality checks run on a CPU. The
full real-ocean sweep requires the 2024 ERA5 files and was run on a Linux
server with an NVIDIA RTX A6000 GPU.

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
The ERA5 files (about 20 GB) are not redistributed because they are available
from public archives; download them with:

```bash
uv run scripts/download_era5.py              # Google Cloud ERA5 archive, no key
uv run scripts/download_era5.py --backend cds  # or the Copernicus CDS API
```

The files are written to `data/era5/` with the names the run scripts expect.
Natural Earth 1:10m coastline data are also open and are downloaded and cached
automatically by Cartopy when the land-mask or exact-geometry checks first run.

## 4. Results

### Synthetic fields (Section 3)

| Paper item                                        | Command                                                                                        |
| ------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| Tables 3, 6; Figures 2–4                          | `uv run scripts/synthetic/results.py` then `uv run scripts/synthetic/figures.py` / `tables.py` |
| Table 5 (seed dispersion)                         | `uv run python revision/task2_synthetic_dispersion.py`                                         |
| Land experiments, Figures 5–6                     | `uv run scripts/results_land_avoidance.py`                                                     |
| Table 8 (λ_land sensitivity)                      | `uv run python revision/task6_lambda_land_sensitivity.py`                                      |
| Export BERS routes; numerical audit               | `uv run python revision/task10_synthetic_local_optimality.py`                                  |
| Computer-assisted proof (Section 3.3, Appendix E) | see `supplementary/synthetic_local_optimality_proof/certificate_report.md`                     |

### Real-ocean corridors (Section 4)

Route generation (server, complete 2024 sweep):

```bash
bash scripts/run_bers_revision_2024_timefix.sh RUN_NAME
```

The launcher records the Git commit, resolved configuration, environment,
input-file metadata and run log below `output/RUN_NAME/`. It can resume an
interrupted run with `--resume RUN_NAME`.

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

`revision/run_revision_postprocessing.sh` and
`revision/run_revision_local_optimality.sh` run these steps in order and
record provenance (git commit, host, settings).

The post-processing scripts take the generated route directories as explicit
arguments. The final route files may additionally be deposited with a DOI so
the tables and figures can be regenerated without repeating the optimization.

## 5. Historical benchmark reference

The documented reproduction workflow uses the open analytical performance
model in `routetools/performance.py`; the historical compiled SWOPP3 evaluator
is not required. Randomized validation against that evaluator found agreement
to 0.103 kW maximum absolute error (0.031% relative), so it serves only as an
independent validation reference. Other participants' scores are held by the
SWOPP3 organizers and appear in the paper only as external benchmark context,
not as outputs that must be regenerated.
