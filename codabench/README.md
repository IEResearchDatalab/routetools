# SWOPP3 Weather Routing Benchmark — CodaBench Setup

This directory contains everything needed to host the SWOPP3 Weather Routing
Benchmark on [CodaBench](https://www.codabench.org/).

## Directory Structure

```
codabench/
├── README.md                 ← You are here
├── competition.yaml          ← Competition configuration (title, phases, leaderboard)
├── scoring_program/
│   ├── scoring.py            ← Evaluates submissions (validation + scoring)
│   └── metadata.yaml         ← CodaBench scoring program metadata
├── pages/
│   ├── overview.md           ← Competition description
│   ├── data.md               ← Data download & API instructions
│   ├── submission.md         ← Submission format specification
│   ├── evaluation.md         ← How submissions are scored
│   └── terms.md              ← Terms and conditions
├── starting_kit/
│   └── starting_kit.py       ← Great-circle baseline (example submission)
└── logo.png                  ← Competition logo
```

## Step-by-Step Setup

### 1. Create the Competition Bundle

CodaBench requires a `.zip` bundle containing the competition definition.
Build it from this directory:

```bash
cd codabench
bash build_bundle.sh
```

This creates `scoring_program.zip`, `starting_kit.zip`, `reference_data.zip`,
and a combined `competition_bundle.zip`. See `build_bundle.sh` for details.

### 2. Create the Competition on CodaBench

1. Go to **[codabench.org](https://www.codabench.org/)** and sign in.
2. Click **Benchmarks/Competitions** → **Create**.
3. Fill in the **Details** tab:
   - **Title:** `SWOPP3 Weather Routing Benchmark`
   - **Logo:** Upload a logo (PNG)
   - **Description:** Copy from `pages/overview.md` or write a summary
   - **Competition Docker Image:** Use `python:3.11-slim`
   - **Competition Type:** Competition
4. **Pages** tab:
   - Add pages from the `pages/` directory (Overview, Data, Submission, Evaluation, Terms)
5. **Phases** tab:
   - Create one phase ("Main Phase")
   - Upload `scoring_program.zip` as the Scoring Program
   - Upload `reference_data.zip` as the Reference Data
   - Set start/end dates, max submissions per day (3), total max (100)
6. **Leaderboard** tab:
   - Add columns matching the keys in `competition.yaml` → `leaderboard`
   - Primary ranking column: `total_energy_mwh` (ascending)
7. **Publish** the competition.

### 3. Reference Data

The `reference_data/` directory must contain the 6-hourly ERA5 NetCDF files
(~3.1 GB total) and the Natural Earth land shapefile before building the
bundle. See `build_bundle.sh` for the full list of required files.

The scoring program is **self-contained** — it uses only `numpy`, `netCDF4`,
`pyshp`, `shapely`, and `matplotlib` (listed in `scoring_program/requirements.txt`).
No `routetools` or JAX installation is needed on the CodaBench worker.
The Docker image `fjsuarez/swopp3-scorer:latest` has all dependencies pre-installed.

### 4. Starting Kit

Upload `starting_kit.zip` to CodaBench so participants can download a working
baseline. The `starting_kit.py` script generates a valid submission using
great-circle routes.

## Scoring

The scoring program validates submission format and **re-evaluates every
route** using the RISE performance model with the official ERA5 data from
`reference_data/`. This guarantees all energy values are computed with the
same model and weather data.

If the ERA5 files are not present in `reference_data/`, the scorer falls back
to self-reported energy values from the participants' CSVs.

## Testing Locally

Test the scoring program locally before deploying:

```bash
# Create a mock submission using the starting kit
cd starting_kit
python starting_kit.py --output-dir /tmp/test_submission

# Simulate CodaBench's invocation
mkdir -p /tmp/codabench_input/res /tmp/codabench_input/ref /tmp/codabench_output
cp -r /tmp/test_submission/* /tmp/codabench_input/res/
cp reference_data/config.json /tmp/codabench_input/ref/

cd ../scoring_program
python scoring.py /tmp/codabench_input /tmp/codabench_output

# Check results
cat /tmp/codabench_output/scores.json
cat /tmp/codabench_output/scoring_log.txt
```

## Key CodaBench Concepts

| Concept             | Meaning in SWOPP3                                      |
| ------------------- | ------------------------------------------------------ |
| **Phase**           | Single evaluation phase (all 366 departures × 8 cases) |
| **Scoring Program** | `scoring.py` — validates and scores submissions        |
| **Reference Data**  | ERA5 NetCDF files + Natural Earth shapefile (~3.1 GB)  |
| **Input Data**      | The submission zip uploaded by participants            |
| **Starting Kit**    | `starting_kit.py` — great-circle baseline code         |
| **Leaderboard**     | Ranked by total energy (MWh), lower = better           |
