"""
FMS Route Optimization Script - HPC Version.

Optimized for HPC environments with sufficient memory to load full ERA5 datasets.
Processes maritime routes across four cases (AO_WPS, AO_noWPS, PO_WPS, PO_noWPS),
applies FMS optimization, and saves results for comparison.

Key differences from standard version:
- Loads full ERA5 datasets once per corridor (no rolling windows)
- Includes resumability for interrupted jobs
"""

import gc
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from routetools.cost import cost_function_rise_penalized
from routetools.era5.loader import (
    load_era5_wavefield,
    load_era5_windfield,
    load_natural_earth_land_mask,
)
from routetools.fms import optimize_fms


# Configuration
CASES = ["AO_WPS", "AO_noWPS", "PO_WPS", "PO_noWPS"]
NUM_POINTS = 200
CORRIDORS = {
    "atlantic": [60, -80, 25, 10],  # [N, W, S, E]
    "pacific": [55, 120, 15, 240],  # Uses 0-360° longitude
}
PENALTY = 10.0  # Penalty weight for waves in the cost function
DAMPING = 0.5  # Damping factor for FMS optimization
TWS_LIMIT = 20.0  # True wind speed limit (m/s)
HS_LIMIT = 7.0  # Significant wave height limit (m)

# Paths
INPUT_FOLDER = Path("output/mc_boatface/resampled")
OUTPUT_FOLDER = Path("output/mc_boatface")
DATA_FOLDER = Path("data/era5")

# ERA5 data paths per corridor
ERA5_PATHS = {
    "atlantic": {
        "wind": DATA_FOLDER / "era5_wind_atlantic_2024.nc",
        "waves": DATA_FOLDER / "era5_waves_atlantic_2024.nc",
    },
    "pacific": {
        "wind": DATA_FOLDER / "era5_wind_pacific_2024.nc",
        "waves": DATA_FOLDER / "era5_waves_pacific_2024.nc",
    },
}

# Output folders
TRACKS_FOLDER = OUTPUT_FOLDER / f"routes_{NUM_POINTS}"
FMS_FOLDER = OUTPUT_FOLDER / f"routes_{NUM_POINTS}_fms"
ENERGY_CSV = OUTPUT_FOLDER / "energy_comparison.csv"


def setup_output_folders():
    """Create output folder structure."""
    TRACKS_FOLDER.mkdir(parents=True, exist_ok=True)
    FMS_FOLDER.mkdir(parents=True, exist_ok=True)
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


def clear_jax_memory():
    """Clear JAX memory and run garbage collection."""
    jax.clear_caches()
    gc.collect()


def output_exists(case, track_name):
    """Check if output files already exist for this track."""
    case_tracks_folder = TRACKS_FOLDER / case
    case_fms_folder = FMS_FOLDER / case

    tracks_csv = case_tracks_folder / f"{track_name}.csv"
    fms_csv = case_fms_folder / f"{track_name}.csv"

    return tracks_csv.exists() and fms_csv.exists()


def load_and_resample_track(track_file, num_points=200):
    """
    Load a track CSV and resample to a fixed number of points.

    Args:
        track_file: Path to the track CSV file
        num_points: Number of points to resample to

    Returns:
        df_resampled: Resampled dataframe
        time_start: Start timestamp
        time_end: End timestamp
    """
    df = pd.read_csv(track_file, parse_dates=["timestamp"])

    time_start = df["timestamp"].min()
    time_end = df["timestamp"].max()

    # Interpolate to target number of points
    ts = pd.date_range(start=time_start, end=time_end, periods=num_points)

    df_resampled = (
        df.set_index("timestamp")[["lon", "lat"]]
        .reindex(ts)
        .interpolate(method="time")
        .reset_index()
    )
    df_resampled.rename(columns={"index": "timestamp"}, inplace=True)

    return df_resampled, time_start, time_end


def load_environmental_data_full(corridor="atlantic"):
    """
    Load ERA5 wind, wave, and land mask data for entire year.

    Args:
        corridor: Corridor name ("atlantic" or "pacific")

    Returns:
        windfield: Wind field data
        wavefield: Wave field data
        land: Land mask
    """
    wind_path = ERA5_PATHS[corridor]["wind"]
    waves_path = ERA5_PATHS[corridor]["waves"]

    print(f"    Loading full {corridor} ERA5 dataset...")
    print(f"      Wind: {wind_path}")
    print(f"      Waves: {waves_path}")

    # Load entire year at once - no time windowing
    windfield = load_era5_windfield(path=wind_path)
    wavefield = load_era5_wavefield(path=waves_path)

    corridor_bounds = CORRIDORS[corridor]
    land = load_natural_earth_land_mask(
        lon_range=(corridor_bounds[1], corridor_bounds[3]),
        lat_range=(corridor_bounds[2], corridor_bounds[0]),
    )

    print(f"    {corridor.capitalize()} dataset loaded successfully")
    return windfield, wavefield, land


def apply_fms_optimization(curve, passage_hours, windfield, wavefield, land, wps=True):
    """
    Apply FMS optimization to a route.

    Args:
        curve: L x 2 array of (lon, lat) coordinates
        passage_hours: Total passage time in hours
        windfield: Wind field data
        wavefield: Wave field data
        land: Land mask
        wps: Whether to use WPS

    Returns:
        curve_fms: Optimized curve (converted to numpy array)
    """
    curve_fms, _ = optimize_fms(
        vectorfield=None,
        curve=curve,
        land=land,
        windfield=windfield,
        wavefield=wavefield,
        penalty=PENALTY,
        damping=DAMPING,
        travel_time=passage_hours,
        spherical_correction=True,
        costfun=cost_function_rise_penalized,
        costfun_kwargs={
            "windfield": windfield,
            "wavefield": wavefield,
            "wps": wps,
            "wave_penalty_weight": PENALTY,
            "wind_penalty_weight": PENALTY,
            "tws_limit": TWS_LIMIT,
            "hs_limit": HS_LIMIT,
        },
        verbose=True,
    )

    # Convert to numpy immediately to free JAX memory
    return np.array(curve_fms)


def calculate_energy(curve, passage_hours, windfield, wavefield, wps=True):
    """
    Calculate energy consumption for a route.

    Args:
        curve: L x 2 array of (lon, lat) coordinates (numpy or JAX)
        passage_hours: Total passage time in hours
        windfield: Wind field data
        wavefield: Wave field data
        wps: Whether to use WPS

    Returns:
        energy: Total energy consumption (as Python float)
    """
    energy = cost_function_rise_penalized(
        curve=curve,
        travel_time=passage_hours,
        windfield=windfield,
        wavefield=wavefield,
        wps=wps,
    )

    # Use .item() to extract scalar value from array
    return float(np.asarray(energy).item())


def build_route_dataframe(curve, timestamps, windfield, wavefield):
    """
    Build a dataframe with route coordinates and environmental data.

    Args:
        curve: Optimized curve (batch_size x L x 2) - numpy or JAX array
        timestamps: Timestamp array
        windfield: Wind field data
        wavefield: Wave field data

    Returns:
        df: Route dataframe with all numpy arrays
    """
    # Ensure curve is numpy array
    curve_np = np.array(curve)
    lon, lat = curve_np[0, :, 0], curve_np[0, :, 1]

    # Convert timestamps to JAX array
    ts_jax = jnp.array(
        pd.to_datetime(timestamps).values.astype("datetime64[s]").astype(int)
    )

    # Sample environmental data
    wind_u, wind_v = windfield(lon, lat, ts_jax)
    wave_h, wave_a = wavefield(lon, lat, ts_jax)

    # Convert all JAX arrays to numpy before creating DataFrame
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "lon": np.array(lon),
            "lat": np.array(lat),
            "wind_u": np.array(wind_u),
            "wind_v": np.array(wind_v),
            "wave_h": np.array(wave_h),
            "wave_a": np.array(wave_a),
        }
    )

    return df


def save_energy_comparison(case, track_name, energy_original, energy_fms):
    """
    Save or append energy comparison results to CSV.

    Args:
        case: Case name (e.g., "AO_WPS")
        track_name: Track filename
        energy_original: Original route energy (as Python float)
        energy_fms: FMS optimized route energy (as Python float)
    """
    result = pd.DataFrame(
        {
            "case": [case],
            "track": [track_name],
            "energy_original": [float(energy_original)],
            "energy_fms": [float(energy_fms)],
            "energy_reduction": [float(energy_original - energy_fms)],
            "energy_reduction_pct": [
                float((energy_original - energy_fms) / energy_original * 100)
            ],
        }
    )

    # Append to CSV or create new
    if ENERGY_CSV.exists():
        result.to_csv(ENERGY_CSV, mode="a", header=False, index=False)
    else:
        result.to_csv(ENERGY_CSV, mode="w", header=True, index=False)


def process_route(track_file, case, windfield, wavefield, land):
    """
    Process a single route: resample, optimize with FMS, and save results.

    Args:
        track_file: Path to the track CSV file
        case: Case name (e.g., "AO_WPS")
        windfield: Wind field data (already loaded)
        wavefield: Wave field data (already loaded)
        land: Land mask (already loaded)
    """
    track_name = track_file.stem
    wps = "WPS" in case

    # Check if already processed
    if output_exists(case, track_name):
        print(f"    ✓ Skipping {track_name} (already processed)")
        return

    print(f"    Processing: {track_name}")

    # Load and resample track
    df_resampled, time_start, time_end = load_and_resample_track(
        track_file, num_points=NUM_POINTS
    )

    # Convert to JAX array
    curve = jnp.array(df_resampled[["lon", "lat"]].values)
    passage_hours = (
        pd.to_datetime(time_end) - pd.to_datetime(time_start)
    ).total_seconds() / 3600

    # Calculate original energy
    energy_original = calculate_energy(
        curve[jnp.newaxis, :, :], passage_hours, windfield, wavefield, wps=wps
    )

    # Apply FMS optimization
    curve_fms = apply_fms_optimization(
        curve, passage_hours, windfield, wavefield, land, wps=wps
    )

    # Calculate FMS energy
    energy_fms = calculate_energy(
        curve_fms, passage_hours, windfield, wavefield, wps=wps
    )

    # Build dataframes
    df_original = build_route_dataframe(
        curve[jnp.newaxis, :, :], df_resampled["timestamp"], windfield, wavefield
    )

    df_fms = build_route_dataframe(
        curve_fms, df_resampled["timestamp"], windfield, wavefield
    )

    # Save results
    case_tracks_folder = TRACKS_FOLDER / case
    case_fms_folder = FMS_FOLDER / case
    case_tracks_folder.mkdir(parents=True, exist_ok=True)
    case_fms_folder.mkdir(parents=True, exist_ok=True)

    df_original.to_csv(case_tracks_folder / f"{track_name}.csv", index=False)
    df_fms.to_csv(case_fms_folder / f"{track_name}.csv", index=False)

    # Save energy comparison
    save_energy_comparison(case, track_name, energy_original, energy_fms)

    # Clean up large objects
    del curve, curve_fms, df_original, df_fms
    clear_jax_memory()

    print(f"      Energy (original): {energy_original:.2f}")
    print(f"      Energy (FMS):      {energy_fms:.2f}")
    print(
        f"      Reduction:         {energy_original - energy_fms:.2f} "
        f"({(energy_original - energy_fms) / energy_original * 100:.1f}%)"
    )


def process_case(case):
    """
    Process all routes for a single case.

    Args:
        case: Case name (e.g., "AO_WPS")
    """
    print(f"\n{'='*60}")
    print(f"Processing case: {case}")
    print(f"{'='*60}")

    case_folder = INPUT_FOLDER / case

    if not case_folder.exists():
        print(f"  WARNING: Subfolder {case_folder} does not exist, skipping")
        return

    # Get all CSV files in the case folder
    track_files = sorted(case_folder.glob("*.csv"))

    if not track_files:
        print(f"  WARNING: No CSV files found in {case_folder}, skipping")
        return

    print(f"  Found {len(track_files)} routes")

    # Determine corridor based on case (AO = Atlantic Ocean, PO = Pacific Ocean)
    corridor = "atlantic" if case.startswith("AO") else "pacific"

    # Load environmental data ONCE for the entire case
    try:
        windfield, wavefield, land = load_environmental_data_full(corridor=corridor)
    except Exception as e:
        print(f"  ERROR loading {corridor} data: {e}")
        return

    # Process each route
    processed = 0
    skipped = 0
    errors = 0

    for track_file in track_files:
        try:
            if output_exists(case, track_file.stem):
                skipped += 1
            else:
                processed += 1
            process_route(track_file, case, windfield, wavefield, land)
        except Exception as e:
            errors += 1
            print(f"    ERROR processing {track_file.name}: {e}")
            continue

    # Clean up environmental data
    del windfield, wavefield, land
    clear_jax_memory()

    print(f"\n  Case {case} complete:")
    print(f"    Processed: {processed}")
    print(f"    Skipped:   {skipped}")
    print(f"    Errors:    {errors}")


def main():
    """Main processing loop."""
    print("=" * 60)
    print("FMS Route Optimization Script - HPC Version")
    print("=" * 60)
    print(f"JAX devices: {jax.devices()}")
    print(f"Input:  {INPUT_FOLDER}")
    print(f"Output: {OUTPUT_FOLDER}")
    print("=" * 60)

    # Validate input folder
    if not INPUT_FOLDER.exists():
        raise FileNotFoundError(f"Input folder {INPUT_FOLDER} does not exist")

    # Setup output folders
    setup_output_folders()

    # Process each case
    start_time = datetime.now()

    for case in CASES:
        try:
            process_case(case)
        except Exception as e:
            print(f"\nFATAL ERROR processing case {case}: {e}")
            continue

    end_time = datetime.now()
    elapsed = end_time - start_time

    print("\n" + "=" * 60)
    print("Processing complete!")
    print(f"Total time: {elapsed}")
    print(f"Results saved to: {OUTPUT_FOLDER}")
    print(f"Energy comparison: {ENERGY_CSV}")
    print("=" * 60)

    # Print summary statistics
    if ENERGY_CSV.exists():
        df = pd.read_csv(ENERGY_CSV)
        print(f"\nProcessed {len(df)} routes total")
        print(f"Average energy reduction: {df['energy_reduction_pct'].mean():.1f}%")
        print(f"Total energy saved: {df['energy_reduction'].sum():.2f} MWh")


if __name__ == "__main__":
    main()
