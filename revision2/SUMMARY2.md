# Revision 2 follow-up summary

Analysis commit: `541fb3213cc0aa7ac67ed59b55fbcb57bb687b83`.

| Task | Status  | Most important finding                                                                                                                                                                    |
| ---- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| C6   | done    | Section 4.1 does not exactly describe the run: its optimizer settings match the manuscript-number roots, but the FMS launcher activated at 19.9 m/s and 6.9 m rather than 20 m/s and 7 m. |
| C4   | done    | The equations match the implementation, and all exported-route energies are below 1000 MWh; transient CMA-ES candidates were not saved.                                                   |
| C1   | partial | Common-weather tables and figures were regenerated, but the preserved Review 1 exposure CSV is not the source of the manuscript's BERS incidence rows.                                    |
| C5   | done    | Exactly 78 late-December BERS records require correction; all other records reproduce within tolerance.                                                                                   |
| C2   | done    | The positive Pacific/no-WPS mean J_CMA change is driven by the raster-mask land-replacement activations.                                                                                  |
| C3   | done    | The executable audit checked 360 manuscript/PDF values: 275 match and 85 mismatch, concentrated in the unpropagated seasonal and exposure results plus two segment-speed rounding cells.  |
