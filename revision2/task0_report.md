# Task 0: Configuration reconciliation

- Run tag: `bers_revision_2024_final_20260923`
- Run Git commit: `8671607609694d208a4c635b11b41edf87d8ae78`
- Analysis Git commit: `fa451908fd0c4254b7fe0d9ff25c6dd21c4c7f34`
- Input run root: `/home/SHARED/routetools/output/bers_revision_2024_final_20260923`
- BERS root: `/home/SHARED/routetools/output/bers_revision_2024_final_20260923/bers`
- CMA-ES root: `/home/SHARED/routetools/output/bers_revision_2024_final_20260923/cmaes`
- Resolved config and manifest: `/home/SHARED/routetools/output/bers_revision_2024_final_20260923/resolved_config.toml`, `/home/SHARED/routetools/output/bers_revision_2024_final_20260923/bers/experiment_manifest.json`
- Completion marker: present; 366 unique departures per case and strategy.
- GC summaries in the BERS and CMA-ES roots were identical for energy and weather maxima.
- Run identity is unverified: this authorized candidate does not reproduce the manuscript's Table 11 or Table 12 values.
- Expected config note path: `coding_agent/bers_revision_reference/final_run_configuration.md` (absent).
- Specified run-output parent: `coding_agent/bers_revision_outputs/` (absent).
- The completed local run above was used only after author confirmation.

## What was done

Compared the run manifest and resolved TOML with the reference settings. Recomputed Table 11 means, Table 12 paired mean improvements, and Table 14 maximum exceedances from the per-departure summary CSVs. No optimizer was run and no route was changed.

## Configuration mismatches

| Corridor | Parameter                  | Manuscript |            Run |
| -------- | -------------------------- | ---------: | -------------: |
| Atlantic | CMA-ES population P        |        400 |            200 |
| Atlantic | CMA-ES sigma0              |   0.300000 |       0.100000 |
| Atlantic | CMA-ES maximum evaluations |      50000 |          25000 |
| Atlantic | Sampled-land penalty       |       1000 | 1000000.000000 |
| Atlantic | Distance-to-land weight    |         10 |     100.000000 |
| Atlantic | FMS damping                |   0.950000 |       0.900000 |
| Atlantic | FMS patience               |        200 |             50 |
| Atlantic | FMS maximum sweeps         |      10000 |           5000 |
| Pacific  | CMA-ES K                   |         15 |             10 |
| Pacific  | CMA-ES population P        |        400 |            200 |
| Pacific  | CMA-ES sigma0              |   0.300000 |       0.100000 |
| Pacific  | CMA-ES maximum evaluations |      50000 |          25000 |
| Pacific  | Sampled-land penalty       |       1000 | 1000000.000000 |
| Pacific  | Distance-to-land weight    |         10 |     100.000000 |
| Pacific  | CMA-ES wave weight         |        200 |      50.000000 |
| Pacific  | FMS damping                |   0.950000 |       0.900000 |
| Pacific  | FMS patience               |        200 |             50 |
| Pacific  | FMS maximum sweeps         |      10000 |           5000 |

## Table 11 mean energies (MWh)

| Scope    | Measure                         | Manuscript | Recomputed | Match |
| -------- | ------------------------------- | ---------: | ---------: | :---: |
| AO_noWPS | GC mean voyage energy (MWh)     | 310.300000 |    305.478 |  no   |
| AO_noWPS | CMA-ES mean voyage energy (MWh) | 220.200000 |    257.572 |  no   |
| AO_noWPS | BERS mean voyage energy (MWh)   | 206.200000 |    236.649 |  no   |
| AO_WPS   | GC mean voyage energy (MWh)     | 253.300000 |    248.735 |  no   |
| AO_WPS   | CMA-ES mean voyage energy (MWh) | 159.900000 |    194.027 |  no   |
| AO_WPS   | BERS mean voyage energy (MWh)   | 139.500000 |    168.727 |  no   |
| PO_noWPS | GC mean voyage energy (MWh)     | 251.700000 |    250.621 |  no   |
| PO_noWPS | CMA-ES mean voyage energy (MWh) | 200.200000 |    224.626 |  no   |
| PO_noWPS | BERS mean voyage energy (MWh)   | 194.800000 |    212.009 |  no   |
| PO_WPS   | GC mean voyage energy (MWh)     | 158.000000 |    158.146 |  no   |
| PO_WPS   | CMA-ES mean voyage energy (MWh) |  87.300000 |    115.354 |  no   |
| PO_WPS   | BERS mean voyage energy (MWh)   |  64.100000 |     89.224 |  no   |

## Table 12 paired improvements (%)

| Scope    | Measure                                      | Manuscript | Recomputed | Match |
| -------- | -------------------------------------------- | ---------: | ---------: | :---: |
| AO_noWPS | BERS over GC mean paired improvement (%)     |  27.500000 |     15.415 |  no   |
| AO_noWPS | BERS over CMA-ES mean paired improvement (%) |   6.000000 |      8.041 |  no   |
| AO_WPS   | BERS over GC mean paired improvement (%)     |  38.900000 |     24.117 |  no   |
| AO_WPS   | BERS over CMA-ES mean paired improvement (%) |  12.500000 |     13.130 |  no   |
| PO_noWPS | BERS over GC mean paired improvement (%)     |  17.500000 |      9.337 |  no   |
| PO_noWPS | BERS over CMA-ES mean paired improvement (%) |   2.600000 |      5.552 |  no   |
| PO_WPS   | BERS over GC mean paired improvement (%)     |  55.200000 |     36.283 |  no   |
| PO_WPS   | BERS over CMA-ES mean paired improvement (%) |  30.600000 |     25.814 |  no   |

## Table 14 maximum exceedances

| Scope       | Measure                 | Manuscript | Recomputed | Match |
| ----------- | ----------------------- | ---------: | ---------: | :---: |
| GC Atlantic | Maximum wind exceedance |   6.570000 |      6.571 |  yes  |
| GC Atlantic | Maximum wave exceedance |   4.990000 |      4.989 |  yes  |
| GC Pacific  | Maximum wind exceedance |  10.640000 |     10.634 |  no   |
| GC Pacific  | Maximum wave exceedance |   6.170000 |      6.175 |  yes  |
| BERS All    | Maximum wind exceedance |   0.320000 |      0.284 |  no   |
| BERS All    | Maximum wave exceedance |   2.160000 |      0.488 |  no   |

## CSV columns

`section`, `scope`, `parameter`, `manuscript_value`, `actual_value`, `match`, and `notes`. The CSV begins with a metadata comment recording both commits, the run tag, and all input run roots.

## Decisions needed

- 18 configuration mismatches; 12 Table 11, 8 Table 12, and 3 Table 14 values also fail rounding agreement. The manuscript run identity/numbers must be resolved before proceeding to Tasks 1–4; no alternate data were substituted.
- No corrections or manuscript edits were made.
