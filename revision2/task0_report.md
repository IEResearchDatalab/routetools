# Task 0: Configuration reconciliation

- Analysis code commit: `da2c53655f1ec4e5597fbafe20342a0e704a8ba9` (tree clean: true)
- CMA-ES and GC root: `/home/SHARED/routetools/output/sweep_combined`
- BERS and GC root: `/home/SHARED/routetools/output/sweep_combined_fms_strict`
- Later corrected run: `/home/SHARED/routetools/output/bers_revision_2024_final_20260923`

## 1. Numerical identification of the manuscript roots

All eight summary CSVs per root were read; each has exactly 366 unique departure timestamps, identical across GC, CMA-ES and BERS, and the GC summaries of the two roots are identical in energy and weather maxima. 20 of 20 comparisons with the exact author-supplied means (6 dp) and paired improvements (2 dp) agree. Against the manuscript tables: Table 11 12/12, Table 12 8/8, Table 14 6/6 agree after rounding. Table 14 is computed against the manuscript thresholds 20 m/s and 7 m.

### Identification

| Scope    | Measure                                       | Manuscript |      Value | Match |
| -------- | --------------------------------------------- | ---------: | ---------: | :---: |
| AO_noWPS | GC mean energy (MWh, 6 dp)                    | 310.346441 | 310.346441 |  yes  |
| AO_noWPS | CMA-ES mean energy (MWh, 6 dp)                | 220.241450 | 220.241450 |  yes  |
| AO_noWPS | BERS mean energy (MWh, 6 dp)                  | 206.159138 | 206.159138 |  yes  |
| AO_WPS   | GC mean energy (MWh, 6 dp)                    | 253.329296 | 253.329296 |  yes  |
| AO_WPS   | CMA-ES mean energy (MWh, 6 dp)                | 159.936026 | 159.936026 |  yes  |
| AO_WPS   | BERS mean energy (MWh, 6 dp)                  | 139.489065 | 139.489065 |  yes  |
| PO_noWPS | GC mean energy (MWh, 6 dp)                    | 251.725395 | 251.725395 |  yes  |
| PO_noWPS | CMA-ES mean energy (MWh, 6 dp)                | 200.205757 | 200.205757 |  yes  |
| PO_noWPS | BERS mean energy (MWh, 6 dp)                  | 194.758570 | 194.758570 |  yes  |
| PO_WPS   | GC mean energy (MWh, 6 dp)                    | 157.995371 | 157.995371 |  yes  |
| PO_WPS   | CMA-ES mean energy (MWh, 6 dp)                |  87.265266 |  87.265266 |  yes  |
| PO_WPS   | BERS mean energy (MWh, 6 dp)                  |  64.135670 |  64.135670 |  yes  |
| AO_noWPS | BERS over GC paired improvement (%, 2 dp)     |  27.530000 |  27.531483 |  yes  |
| AO_WPS   | BERS over GC paired improvement (%, 2 dp)     |  38.900000 |  38.902003 |  yes  |
| PO_noWPS | BERS over GC paired improvement (%, 2 dp)     |  17.520000 |  17.516463 |  yes  |
| PO_WPS   | BERS over GC paired improvement (%, 2 dp)     |  55.170000 |  55.171888 |  yes  |
| AO_noWPS | BERS over CMA-ES paired improvement (%, 2 dp) |   5.970000 |   5.967404 |  yes  |
| AO_WPS   | BERS over CMA-ES paired improvement (%, 2 dp) |  12.490000 |  12.494529 |  yes  |
| PO_noWPS | BERS over CMA-ES paired improvement (%, 2 dp) |   2.620000 |   2.617238 |  yes  |
| PO_WPS   | BERS over CMA-ES paired improvement (%, 2 dp) |  30.580000 |  30.579347 |  yes  |

### Table 11 (MWh)

| Scope    | Measure                  | Manuscript |   Value | Match |
| -------- | ------------------------ | ---------: | ------: | :---: |
| AO_noWPS | GC mean energy (MWh)     | 310.300000 | 310.346 |  yes  |
| AO_noWPS | CMA-ES mean energy (MWh) | 220.200000 | 220.241 |  yes  |
| AO_noWPS | BERS mean energy (MWh)   | 206.200000 | 206.159 |  yes  |
| AO_WPS   | GC mean energy (MWh)     | 253.300000 | 253.329 |  yes  |
| AO_WPS   | CMA-ES mean energy (MWh) | 159.900000 | 159.936 |  yes  |
| AO_WPS   | BERS mean energy (MWh)   | 139.500000 | 139.489 |  yes  |
| PO_noWPS | GC mean energy (MWh)     | 251.700000 | 251.725 |  yes  |
| PO_noWPS | CMA-ES mean energy (MWh) | 200.200000 | 200.206 |  yes  |
| PO_noWPS | BERS mean energy (MWh)   | 194.800000 | 194.759 |  yes  |
| PO_WPS   | GC mean energy (MWh)     | 158.000000 | 157.995 |  yes  |
| PO_WPS   | CMA-ES mean energy (MWh) |  87.300000 | 87.2653 |  yes  |
| PO_WPS   | BERS mean energy (MWh)   |  64.100000 | 64.1357 |  yes  |

### Table 12 (%)

| Scope    | Measure                                 | Manuscript |   Value | Match |
| -------- | --------------------------------------- | ---------: | ------: | :---: |
| AO_noWPS | BERS over GC paired improvement (%)     |  27.500000 | 27.5315 |  yes  |
| AO_noWPS | BERS over CMA-ES paired improvement (%) |   6.000000 |  5.9674 |  yes  |
| AO_WPS   | BERS over GC paired improvement (%)     |  38.900000 |  38.902 |  yes  |
| AO_WPS   | BERS over CMA-ES paired improvement (%) |  12.500000 | 12.4945 |  yes  |
| PO_noWPS | BERS over GC paired improvement (%)     |  17.500000 | 17.5165 |  yes  |
| PO_noWPS | BERS over CMA-ES paired improvement (%) |   2.600000 | 2.61724 |  yes  |
| PO_WPS   | BERS over GC paired improvement (%)     |  55.200000 | 55.1719 |  yes  |
| PO_WPS   | BERS over CMA-ES paired improvement (%) |  30.600000 | 30.5793 |  yes  |

### Table 14 (exceedance over 20 m/s and 7 m)

| Scope       | Measure                 | Manuscript |  Value | Match |
| ----------- | ----------------------- | ---------: | -----: | :---: |
| GC Atlantic | Maximum wind exceedance |   6.570000 | 6.5714 |  yes  |
| GC Atlantic | Maximum wave exceedance |   4.990000 |  4.989 |  yes  |
| GC Pacific  | Maximum wind exceedance |  10.640000 | 10.635 |  yes  |
| GC Pacific  | Maximum wave exceedance |   6.170000 | 6.1749 |  yes  |
| BERS All    | Maximum wind exceedance |   0.320000 |  0.315 |  yes  |
| BERS All    | Maximum wave exceedance |   2.160000 |  2.156 |  yes  |

## 2. Reconstructed configuration evidence

Basis: `direct` = literal in preserved source committed before the output (or an embedded manifest); `reconstructed` = inferred from material committed after the output or from code defaults whose use is not recorded; `unverified` = no preserved evidence. The legacy roots predate embedded manifests. The strict launcher and the task-8 manifest are reconstructions, not contemporaneous manifests. Atlantic and Pacific rows are identical for the FMS settings.

| Scope    | Measure                                           | Manuscript |                                     Value | Match | Basis                                                                                                                                                 |
| -------- | ------------------------------------------------- | ---------: | ----------------------------------------: | :---: | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| Atlantic | CMA-ES K                                          |         10 |                                        10 |  yes  | direct: scripts/swopp3_slurm_atlantic_k10.sh @ 4742156e                                                                                               |
| Atlantic | CMA-ES population P                               |        400 |                                       400 |  yes  | direct: scripts/swopp3_slurm_atlantic_k10.sh @ 4742156e                                                                                               |
| Atlantic | CMA-ES sigma0                                     |   0.300000 |                                       0.3 |  yes  | direct: scripts/swopp3_slurm_atlantic_k10.sh @ 4742156e                                                                                               |
| Atlantic | CMA-ES maximum evaluations                        |      50000 |                                     50000 |  yes  | direct: scripts/swopp3_slurm_atlantic_k10.sh @ 4742156e                                                                                               |
| Atlantic | CMA-ES wind weight                                |         50 |                                        50 |  yes  | direct: scripts/swopp3_slurm_atlantic_k10.sh @ 4742156e                                                                                               |
| Atlantic | CMA-ES wave weight                                |         50 |                                        50 |  yes  | direct: scripts/swopp3_slurm_atlantic_k10.sh @ 4742156e                                                                                               |
| Atlantic | Distance-to-land weight (distance_penalty_weight) |         10 |                                        10 |  yes  | direct: scripts/swopp3_slurm_atlantic_k10.sh @ 4742156e                                                                                               |
| Atlantic | Waypoints                                         |        178 |                                       178 |  yes  | direct: scripts/swopp3_slurm_atlantic_k10.sh @ 4742156e                                                                                               |
| Atlantic | Evaluation grid (minutes)                         |         30 |                                        30 |  yes  | direct: scripts/swopp3_slurm_atlantic_k10.sh @ 4742156e                                                                                               |
| Atlantic | Sampled-land penalty                              |       1000 |                                      1000 |  yes  | direct: routetools/swopp3_runner.py defaults @ 4742156e (identical @ 66192104)                                                                        |
| Pacific  | CMA-ES K                                          |         15 |                                        15 |  yes  | direct: scripts/swopp3_slurm_pacific_k15_p400.sh @ 40c91c02                                                                                           |
| Pacific  | CMA-ES population P                               |        400 |                                       400 |  yes  | direct: scripts/swopp3_slurm_pacific_k15_p400.sh @ 40c91c02                                                                                           |
| Pacific  | CMA-ES sigma0                                     |   0.300000 |                                       0.3 |  yes  | direct: scripts/swopp3_slurm_pacific_k15_p400.sh @ 40c91c02                                                                                           |
| Pacific  | CMA-ES maximum evaluations                        |      50000 |                                     50000 |  yes  | direct: scripts/swopp3_slurm_pacific_k15_p400.sh @ 40c91c02                                                                                           |
| Pacific  | CMA-ES wind weight                                |         50 |                                        50 |  yes  | direct: scripts/swopp3_slurm_pacific_k15_p400.sh @ 40c91c02                                                                                           |
| Pacific  | CMA-ES wave weight                                |        200 |                                       200 |  yes  | direct: scripts/swopp3_slurm_pacific_k15_p400.sh @ 40c91c02                                                                                           |
| Pacific  | Distance-to-land weight (distance_penalty_weight) |         10 |                                        10 |  yes  | direct: scripts/swopp3_slurm_pacific_k15_p400.sh @ 40c91c02                                                                                           |
| Pacific  | Waypoints                                         |        293 |                                       293 |  yes  | direct: scripts/swopp3_slurm_pacific_k15_p400.sh @ 40c91c02                                                                                           |
| Pacific  | Evaluation grid (minutes)                         |         30 |                                        30 |  yes  | direct: scripts/swopp3_slurm_pacific_k15_p400.sh @ 40c91c02                                                                                           |
| Pacific  | Sampled-land penalty                              |       1000 |                                      1000 |  yes  | direct: routetools/swopp3_runner.py defaults @ 4742156e (identical @ 66192104)                                                                        |
| Atlantic | FMS patience                                      |        200 |                                       200 |  yes  | reconstructed: scripts/swopp3_apply_fms.py @ 66192104 (unchanged since d2bcc51f)                                                                      |
| Atlantic | FMS damping                                       |   0.950000 |                                      0.95 |  yes  | reconstructed: scripts/swopp3_apply_fms.py @ 66192104 (unchanged since d2bcc51f)                                                                      |
| Atlantic | FMS maximum iterations (code name: fms_maxfevals) |      10000 |                                     10000 |  yes  | reconstructed: scripts/swopp3_apply_fms.py @ 66192104 (unchanged since d2bcc51f)                                                                      |
| Atlantic | FMS wind weight                                   |         50 |                                        50 |  yes  | reconstructed: scripts/run_fms_sweep_combined_strict.sh @ 66192104 (committed 2026-05-22; outputs dated 2026-04-06/07)                                |
| Atlantic | FMS wave weight                                   |         50 |                                        50 |  yes  | reconstructed: scripts/run_fms_sweep_combined_strict.sh @ 66192104 (committed 2026-05-22; outputs dated 2026-04-06/07)                                |
| Atlantic | Wind threshold used by FMS (m/s)                  |         20 |                                      19.9 |  no   | reconstructed: scripts/run_fms_sweep_combined_strict.sh @ 66192104 (committed 2026-05-22; outputs dated 2026-04-06/07)                                |
| Atlantic | Wave threshold used by FMS (m)                    |          7 |                                       6.9 |  no   | reconstructed: scripts/run_fms_sweep_combined_strict.sh @ 66192104 (committed 2026-05-22; outputs dated 2026-04-06/07)                                |
| Atlantic | FMS land handling                                 |        n/a | sampled-land rollback to the CMA-ES curve |  n/a  | reconstructed: scripts/swopp3_apply_fms.py @ 66192104 (unchanged since d2bcc51f)                                                                      |
| Atlantic | Coastline-distance term in FMS objective          |        n/a |                                    absent |  n/a  | reconstructed: scripts/swopp3_apply_fms.py @ 66192104 (unchanged since d2bcc51f); revision/task8_sweep_combined_fms_strict_manifest.json (2026-09-24) |
| Pacific  | FMS patience                                      |        200 |                                       200 |  yes  | reconstructed: scripts/swopp3_apply_fms.py @ 66192104 (unchanged since d2bcc51f)                                                                      |
| Pacific  | FMS damping                                       |   0.950000 |                                      0.95 |  yes  | reconstructed: scripts/swopp3_apply_fms.py @ 66192104 (unchanged since d2bcc51f)                                                                      |
| Pacific  | FMS maximum iterations (code name: fms_maxfevals) |      10000 |                                     10000 |  yes  | reconstructed: scripts/swopp3_apply_fms.py @ 66192104 (unchanged since d2bcc51f)                                                                      |
| Pacific  | FMS wind weight                                   |         50 |                                        50 |  yes  | reconstructed: scripts/run_fms_sweep_combined_strict.sh @ 66192104 (committed 2026-05-22; outputs dated 2026-04-06/07)                                |
| Pacific  | FMS wave weight                                   |         50 |                                        50 |  yes  | reconstructed: scripts/run_fms_sweep_combined_strict.sh @ 66192104 (committed 2026-05-22; outputs dated 2026-04-06/07)                                |
| Pacific  | Wind threshold used by FMS (m/s)                  |         20 |                                      19.9 |  no   | reconstructed: scripts/run_fms_sweep_combined_strict.sh @ 66192104 (committed 2026-05-22; outputs dated 2026-04-06/07)                                |
| Pacific  | Wave threshold used by FMS (m)                    |          7 |                                       6.9 |  no   | reconstructed: scripts/run_fms_sweep_combined_strict.sh @ 66192104 (committed 2026-05-22; outputs dated 2026-04-06/07)                                |
| Pacific  | FMS land handling                                 |        n/a | sampled-land rollback to the CMA-ES curve |  n/a  | reconstructed: scripts/swopp3_apply_fms.py @ 66192104 (unchanged since d2bcc51f)                                                                      |
| Pacific  | Coastline-distance term in FMS objective          |        n/a |                                    absent |  n/a  | reconstructed: scripts/swopp3_apply_fms.py @ 66192104 (unchanged since d2bcc51f); revision/task8_sweep_combined_fms_strict_manifest.json (2026-09-24) |

## 3. Unresolved provenance limitations

- Exact checked-out/dirty source state of the CMA-ES runs: No embedded manifest, git_commit.txt or resolved config in the root.
- Exact checked-out/dirty source state of the strict FMS run: Outputs predate the launcher; no reflog entry for the run was found.
- Combination of the K10 and K15 launcher outputs into sweep_combined: The combination step is not documented in the repository.
- FMS invoked with the code defaults (no explicit --fms-\* flags): Only the later launcher shows an invocation.
- FMS bound counts sweeps (manuscript) or evaluations (code name): The code exposes fms_maxfevals; the manuscript says sweeps.
- The strict FMS output directory is dated April 2026 while the launcher was committed on 2026-05-22 (`66192104`); no repository source file relevant to FMS or CMA-ES changed between the outputs and that commit, but the checked-out and dirty state at run time cannot be proven.
- No exact run commit is asserted.

## 4. Later corrected-run comparison (separate)

`output/bers_revision_2024_final_20260923` is complete and has strong embedded provenance (resolved config, manifest, git commit). It does not reproduce Tables 11 or 12, so it was not the source of the manuscript numbers and is not substituted here.

Configuration: 18 of 34 claims differ.

| Scope    | Measure                                | Manuscript | Value | Match |
| -------- | -------------------------------------- | ---------: | ----: | :---: |
| Atlantic | CMA-ES population P                    |        400 |   200 |  no   |
| Atlantic | CMA-ES sigma0                          |   0.300000 |   0.1 |  no   |
| Atlantic | CMA-ES maximum evaluations             |      50000 | 25000 |  no   |
| Atlantic | Sampled-land penalty                   |       1000 | 1e+06 |  no   |
| Atlantic | Distance-to-land weight                |         10 |   100 |  no   |
| Atlantic | FMS damping                            |   0.950000 |   0.9 |  no   |
| Atlantic | FMS patience                           |        200 |    50 |  no   |
| Atlantic | FMS maximum iterations (fms_maxfevals) |      10000 |  5000 |  no   |
| Pacific  | CMA-ES K                               |         15 |    10 |  no   |
| Pacific  | CMA-ES population P                    |        400 |   200 |  no   |
| Pacific  | CMA-ES sigma0                          |   0.300000 |   0.1 |  no   |
| Pacific  | CMA-ES maximum evaluations             |      50000 | 25000 |  no   |
| Pacific  | Sampled-land penalty                   |       1000 | 1e+06 |  no   |
| Pacific  | Distance-to-land weight                |         10 |   100 |  no   |
| Pacific  | CMA-ES wave weight                     |        200 |    50 |  no   |
| Pacific  | FMS damping                            |   0.950000 |   0.9 |  no   |
| Pacific  | FMS patience                           |        200 |    50 |  no   |
| Pacific  | FMS maximum iterations (fms_maxfevals) |      10000 |  5000 |  no   |

Tables: 23 of 26 comparisons differ.

| Scope    | Measure                                 | Manuscript |   Value | Match |
| -------- | --------------------------------------- | ---------: | ------: | :---: |
| AO_noWPS | GC mean energy (MWh)                    | 310.300000 | 305.478 |  no   |
| AO_noWPS | CMA-ES mean energy (MWh)                | 220.200000 | 257.572 |  no   |
| AO_noWPS | BERS mean energy (MWh)                  | 206.200000 | 236.649 |  no   |
| AO_noWPS | BERS over GC paired improvement (%)     |  27.500000 | 15.4148 |  no   |
| AO_noWPS | BERS over CMA-ES paired improvement (%) |   6.000000 | 8.04128 |  no   |
| AO_WPS   | GC mean energy (MWh)                    | 253.300000 | 248.735 |  no   |
| AO_WPS   | CMA-ES mean energy (MWh)                | 159.900000 | 194.027 |  no   |
| AO_WPS   | BERS mean energy (MWh)                  | 139.500000 | 168.727 |  no   |
| AO_WPS   | BERS over GC paired improvement (%)     |  38.900000 | 24.1173 |  no   |
| AO_WPS   | BERS over CMA-ES paired improvement (%) |  12.500000 | 13.1297 |  no   |
| PO_noWPS | GC mean energy (MWh)                    | 251.700000 | 250.621 |  no   |
| PO_noWPS | CMA-ES mean energy (MWh)                | 200.200000 | 224.626 |  no   |
| PO_noWPS | BERS mean energy (MWh)                  | 194.800000 | 212.009 |  no   |
| PO_noWPS | BERS over GC paired improvement (%)     |  17.500000 |  9.3374 |  no   |
| PO_noWPS | BERS over CMA-ES paired improvement (%) |   2.600000 | 5.55209 |  no   |
| PO_WPS   | GC mean energy (MWh)                    | 158.000000 | 158.146 |  no   |
| PO_WPS   | CMA-ES mean energy (MWh)                |  87.300000 | 115.354 |  no   |
| PO_WPS   | BERS mean energy (MWh)                  |  64.100000 | 89.2237 |  no   |
| PO_WPS   | BERS over GC paired improvement (%)     |  55.200000 | 36.2831 |  no   |
| PO_WPS   | BERS over CMA-ES paired improvement (%) |  30.600000 |  25.814 |  no   |

## 5. Decisions required from the authors

1. Confirm that the manuscript describes the legacy roots, and decide whether the later corrected run is to be reported separately.
2. The strict FMS used 19.9 m/s and 6.9 m while the manuscript states 20 m/s and 7 m: amend the text, or rerun, as a deliberate decision.
3. Decide whether the FMS bound is described as sweeps or as evaluations (the code name is `fms_maxfevals`).
4. Document how the K=10 and K=15 launcher outputs were combined into `sweep_combined`, or state the limitation.
5. Decide whether to accept the reconstructed provenance or to re-execute under an embedded manifest.

No optimizer was run, no route or run output was modified, and the manuscript was not edited.
