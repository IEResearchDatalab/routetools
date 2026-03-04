#!/usr/bin/env python3
"""Standalone benchmark: routetools.performance vs SWOPP3 reference.

Runs comprehensive comparisons and prints detailed statistics, error
distributions, and per-component breakdowns.  Not a pytest file — run
directly:

    python scripts/parametric_benchmark.py

Requires: swopp3_performance_model, numpy, routetools
"""

from __future__ import annotations

import sys
import time

import numpy as np

try:
    from swopp3_performance_model import predict_no_wps, predict_with_wps
except ImportError:
    print("ERROR: swopp3_performance_model wheel is not installed.", file=sys.stderr)
    sys.exit(1)

from routetools.performance import (
    K_A,
    K_H,
    K_W,
    A_W,
    predict_power_no_wps as parametric_no_wps,
    predict_power_with_wps as parametric_with_wps,
)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------
def error_stats(errs: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(errs)),
        "std": float(np.std(errs)),
        "p50": float(np.median(errs)),
        "p95": float(np.percentile(errs, 95)),
        "p99": float(np.percentile(errs, 99)),
        "max": float(np.max(errs)),
    }


def print_stats(label: str, stats: dict[str, float]) -> None:
    print(f"  {label}:")
    print(
        f"    mean={stats['mean']:.6f}  std={stats['std']:.6f}  "
        f"p50={stats['p50']:.6f}  p95={stats['p95']:.6f}  "
        f"p99={stats['p99']:.6f}  max={stats['max']:.6f}"
    )


# -----------------------------------------------------------------------
# Benchmark routines
# -----------------------------------------------------------------------
def benchmark_no_wps(n: int = 10_000, seed: int = 42) -> None:
    print(f"\n{'=' * 72}")
    print(f"  predict_no_wps  —  {n:,} random samples  (seed={seed})")
    print(f"{'=' * 72}")

    rng = np.random.default_rng(seed)
    tws = rng.uniform(0, 30, n)
    twa = rng.uniform(0, 180, n)
    swh = rng.uniform(0, 10, n)
    mwa = rng.uniform(0, 180, n)
    v = rng.uniform(0, 14.5, n)

    refs = np.empty(n)
    pars = np.empty(n)

    t0 = time.perf_counter()
    for i in range(n):
        refs[i] = predict_no_wps(tws[i], twa[i], swh[i], mwa[i], v[i])
    t_ref = time.perf_counter() - t0

    t0 = time.perf_counter()
    for i in range(n):
        pars[i] = parametric_no_wps(tws[i], twa[i], swh[i], mwa[i], v[i])
    t_par = time.perf_counter() - t0

    abs_errs = np.abs(pars - refs)
    # Relative error (only where ref > 1 kW to avoid div-by-zero noise)
    mask = refs > 1.0
    rel_errs = np.full(n, np.nan)
    rel_errs[mask] = abs_errs[mask] / refs[mask] * 100  # percent

    print(f"\n  Timing:  reference {t_ref:.3f}s ({t_ref/n*1e6:.1f} µs/call)  |  "
          f"parametric {t_par:.3f}s ({t_par/n*1e6:.1f} µs/call)")
    print(f"\n  Absolute error (kW):")
    print_stats("all", error_stats(abs_errs))
    if mask.sum() > 0:
        print(f"\n  Relative error (%, where ref > 1 kW, n={mask.sum():,}):")
        print_stats("filtered", error_stats(rel_errs[mask]))

    # Worst cases
    worst_idx = np.argsort(abs_errs)[-5:][::-1]
    print(f"\n  Top-5 worst absolute errors:")
    print(f"    {'TWS':>6} {'TWA':>6} {'SWH':>6} {'MWA':>6} {'V':>6}  "
          f"{'Ref':>10} {'Par':>10} {'Err':>10}")
    for idx in worst_idx:
        print(
            f"    {tws[idx]:6.2f} {twa[idx]:6.1f} {swh[idx]:6.2f} "
            f"{mwa[idx]:6.1f} {v[idx]:6.2f}  "
            f"{refs[idx]:10.4f} {pars[idx]:10.4f} {abs_errs[idx]:10.4f}"
        )

    n_exact = np.sum(abs_errs < 0.001)
    n_good = np.sum(abs_errs < 0.01)
    n_ok = np.sum(abs_errs < 0.1)
    print(f"\n  Error distribution:")
    print(f"    < 0.001 kW : {n_exact:6,} / {n:,}  ({n_exact/n*100:.1f}%)")
    print(f"    < 0.01  kW : {n_good:6,} / {n:,}  ({n_good/n*100:.1f}%)")
    print(f"    < 0.1   kW : {n_ok:6,} / {n:,}  ({n_ok/n*100:.1f}%)")
    print(f"    ≥ 0.1   kW : {n - n_ok:6,} / {n:,}  ({(n - n_ok)/n*100:.1f}%)")


def benchmark_with_wps(
    n: int = 10_000,
    seed: int = 99,
) -> None:
    print(f"\n{'=' * 72}")
    print(f"  predict_with_wps  —  {n:,} random samples  (seed={seed})")
    print(f"{'=' * 72}")

    rng = np.random.default_rng(seed)
    tws = rng.uniform(0, 30, n)
    twa = rng.uniform(0, 180, n)
    swh = rng.uniform(0, 10, n)
    mwa = rng.uniform(0, 180, n)
    v = rng.uniform(0, 14.5, n)

    refs = np.empty(n)
    pars = np.empty(n)

    t0 = time.perf_counter()
    for i in range(n):
        refs[i] = predict_with_wps(tws[i], twa[i], swh[i], mwa[i], v[i])
    t_ref = time.perf_counter() - t0

    t0 = time.perf_counter()
    for i in range(n):
        pars[i] = parametric_with_wps(
            tws[i], twa[i], swh[i], mwa[i], v[i],
        )
    t_par = time.perf_counter() - t0

    abs_errs = np.abs(pars - refs)
    mask = refs > 1.0
    rel_errs = np.full(n, np.nan)
    rel_errs[mask] = abs_errs[mask] / refs[mask] * 100

    print(f"\n  Timing:  reference {t_ref:.3f}s ({t_ref/n*1e6:.1f} µs/call)  |  "
          f"parametric {t_par:.3f}s ({t_par/n*1e6:.1f} µs/call)")
    print(f"\n  Absolute error (kW):")
    print_stats("all", error_stats(abs_errs))
    if mask.sum() > 0:
        print(f"\n  Relative error (%, where ref > 1 kW, n={mask.sum():,}):")
        print_stats("filtered", error_stats(rel_errs[mask]))

    worst_idx = np.argsort(abs_errs)[-5:][::-1]
    print(f"\n  Top-5 worst absolute errors:")
    print(f"    {'TWS':>6} {'TWA':>6} {'SWH':>6} {'MWA':>6} {'V':>6}  "
          f"{'Ref':>10} {'Par':>10} {'Err':>10}")
    for idx in worst_idx:
        print(
            f"    {tws[idx]:6.2f} {twa[idx]:6.1f} {swh[idx]:6.2f} "
            f"{mwa[idx]:6.1f} {v[idx]:6.2f}  "
            f"{refs[idx]:10.4f} {pars[idx]:10.4f} {abs_errs[idx]:10.4f}"
        )

    n_exact = np.sum(abs_errs < 0.01)
    n_good = np.sum(abs_errs < 0.1)
    n_ok = np.sum(abs_errs < 1.0)
    print(f"\n  Error distribution:")
    print(f"    < 0.01 kW : {n_exact:6,} / {n:,}  ({n_exact/n*100:.1f}%)")
    print(f"    < 0.1  kW : {n_good:6,} / {n:,}  ({n_good/n*100:.1f}%)")
    print(f"    < 1.0  kW : {n_ok:6,} / {n:,}  ({n_ok/n*100:.1f}%)")
    print(f"    ≥ 1.0  kW : {n - n_ok:6,} / {n:,}  ({(n - n_ok)/n*100:.1f}%)")


def benchmark_component_breakdown() -> None:
    """Show per-component accuracy vs reference decomposition."""
    print(f"\n{'=' * 72}")
    print(f"  Component breakdown (hull / wind / wave)")
    print(f"{'=' * 72}")

    speeds = [2, 4, 6, 8, 10, 12, 14]

    # Hull
    print(f"\n  Hull: P_hull = K_h · v³   (K_h = {K_H:.6f})")
    print(f"    {'v':>5}  {'Ref':>10}  {'Par':>10}  {'Err':>10}  {'Ratio':>10}")
    for v in speeds:
        ref = predict_no_wps(0, 0, 0, 0, v)
        par = K_H * v**3
        print(f"    {v:5.1f}  {ref:10.4f}  {par:10.4f}  {abs(par - ref):10.6f}  "
              f"{ref / v**3 if v > 0 else 0:10.6f}")

    # Wind (TWA sweep at fixed tws=15, v=8)
    print(f"\n  Wind: K_a · v · (VR·ux − v²)   (K_a = {K_A:.6f})")
    print(f"    {'TWA':>5}  {'TWS':>5}  {'V':>5}  {'Ref_wind':>10}  {'Par_wind':>10}  {'Err':>10}")
    tws_test, v_test = 15.0, 8.0
    p_hull = predict_no_wps(0, 0, 0, 0, v_test)
    for twa in [0, 30, 60, 90, 120, 150, 180]:
        twa_rad = np.radians(twa)
        ux = tws_test * np.cos(twa_rad) + v_test
        uy = tws_test * np.sin(twa_rad)
        vr = np.sqrt(ux**2 + uy**2)
        total_ref = predict_no_wps(tws_test, twa, 0, 0, v_test)
        # If the model clamps total power at zero (strong tailwind),
        # the decomposition total = hull + wind is invalid.
        if total_ref > 0.0:
            ref_wind = total_ref - p_hull
        else:
            ref_wind = float("nan")
        par_wind = K_A * v_test * (vr * ux - v_test**2)
        print(f"    {twa:5}  {tws_test:5.0f}  {v_test:5.0f}  {ref_wind:10.4f}  "
              f"{par_wind:10.4f}  {abs(par_wind - ref_wind):10.6f}")

    # Wave (MWA sweep at fixed swh=3, v=8)
    print(f"\n  Wave: A_w·swh²·v^1.5·exp(−k_w·|mwa|³)")
    print(f"    A_w = {A_W:.4f},  k_w = {K_W:.5f}")
    print(f"    {'MWA':>5}  {'SWH':>5}  {'V':>5}  {'Ref_wave':>10}  {'Par_wave':>10}  {'Err':>10}")
    swh_test, v_test2 = 3.0, 8.0
    p_hull2 = predict_no_wps(0, 0, 0, 0, v_test2)
    for mwa in [0, 20, 40, 60, 80, 100, 120, 140, 160, 180]:
        mwa_rad = np.radians(mwa)
        ref_wave = predict_no_wps(0, 0, swh_test, mwa, v_test2) - p_hull2
        par_wave = A_W * swh_test**2 * v_test2**1.5 * np.exp(-K_W * abs(mwa_rad)**3)
        print(f"    {mwa:5}  {swh_test:5.0f}  {v_test2:5.0f}  {ref_wave:10.4f}  "
              f"{par_wave:10.4f}  {abs(par_wave - ref_wave):10.6f}")


def benchmark_sail_savings() -> None:
    """Show sail power savings across conditions."""
    print(f"\n{'=' * 72}")
    print(f"  Sail power savings P_sail = P_no_wps − P_with_wps")
    print(f"{'=' * 72}")

    print(f"\n  P_sail at v=8 m/s, varying TWS and TWA:")
    print(f"    {'TWS':>5}  " + "  ".join(f"TWA={t:>3}°" for t in range(0, 181, 30)))
    for tws in [0, 5, 10, 15, 20, 25, 30]:
        row = f"    {tws:5}"
        for twa in range(0, 181, 30):
            p_no = predict_no_wps(tws, twa, 0, 0, 8)
            p_wp = predict_with_wps(tws, twa, 0, 0, 8)
            p_sail = p_no - p_wp
            row += f"  {p_sail:8.1f}"
        print(row)

    print(f"\n  Savings % at v=8 m/s (where P_no_wps > 0):")
    print(f"    {'TWS':>5}  " + "  ".join(f"TWA={t:>3}°" for t in range(0, 181, 30)))
    for tws in [5, 10, 15, 20, 25, 30]:
        row = f"    {tws:5}"
        for twa in range(0, 181, 30):
            p_no = predict_no_wps(tws, twa, 0, 0, 8)
            p_wp = predict_with_wps(tws, twa, 0, 0, 8)
            if p_no > 0:
                pct = (p_no - p_wp) / p_no * 100
                row += f"  {pct:7.1f}%"
            else:
                row += f"       n/a"
        print(row)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main() -> None:
    print("=" * 72)
    print("  PARAMETRIC MODEL vs SWOPP3 REFERENCE  —  BENCHMARK")
    print("  (Fully closed-form — no lookup tables)")
    print("=" * 72)

    # 1. Component breakdown
    benchmark_component_breakdown()

    # 2. predict_no_wps random
    benchmark_no_wps(n=10_000, seed=42)

    # 3. Sail savings overview
    benchmark_sail_savings()

    # 4. predict_with_wps random (fully closed-form)
    benchmark_with_wps(n=10_000, seed=99)

    print(f"\n{'=' * 72}")
    print("  BENCHMARK COMPLETE")
    print(f"{'=' * 72}\n")


if __name__ == "__main__":
    main()
