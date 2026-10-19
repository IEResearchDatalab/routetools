# Computer-assisted proof of local optimality — synthetic benchmarks: results

Companion to `SPEC_synthetic_local_optimality.md`. The interval implementation reproduces the discrete objectives in `routetools` (commit `8462156`), with L = 200 waypoints. The archived route-wise results in `bers_routes/` were produced from the 25 routes exported by `revision/task10_synthetic_local_optimality.py` at commit `4ba7618`.

## 1. Result

At benchmark level, every synthetic problem has at least one **certified strict local minimizer** of the discrete action. For the four travel-time problems, the reference certificate agrees with the BERS value to the reported precision. For Swirlys, an independently found certified minimizer is about 5% below the reported BERS value; this rules out global optimality but does not rule out a different local minimum near a BERS route. Section 5 gives the result tied to the actual 25 exported BERS routes: 23 are certified near strict local minima.

| Benchmark | Objective | Variations | Dim. | Certified J\* (enclosure width < 1e-12) | BERS (Table 5, mean) | ‖∇J(X̃)‖ ≤ | Ball radius R | min eig ∇²J on ball ≥ | Safety margin mR/(2‖∇J‖) |
|---|---|---|---|---|---|---|---|---|---|
| Circular | time | full waypoint space | 396 | 1.974958838640 | 1.9750 | 1.2e-38 | 4.4e-16 | 9.4e-7 | 1.7e16 |
| Circular | time | transverse | 198 | 1.974959641010 | 1.9750 | 1.9e-17 | 5.2e-15 | 1.8e-2 | 2.5 |
| Four Vortices | time | transverse | 198 | 8.949543208712 | 8.9495 | 1.4e-17 | 6.4e-15 | 1.1e-2 | 2.5 |
| Double Gyre | time (time-dependent) | transverse | 198 | 0.988842908792 | 0.9888 | 9.5e-17 | 1.9e-14 | 2.5e-2 | 2.5 |
| Techy | time (time-dependent) | full waypoint space | 396 | 1.031713402953 | 1.0317 | 3.4e-36 | 4.4e-16 | 7.9e-5 | 5.2e15 |
| Techy | time (time-dependent) | transverse | 198 | 1.031765869173 | 1.0317 | 4.6e-17 | 1.2e-14 | 2.0e-2 | 2.5 |
| Swirlys | fixed-time energy | full waypoint space | 396 | **1.867882848240** | **1.9702** | 6.4e-14 | 1.7e-10 | 1.9e-3 | 2.5 |

A margin above 1 means the certificate holds. Each certificate runs in about 0.2–6 s.

## 2. What exactly is proved

**Lemma (strong convexity on a ball).** If ∇²J ⪰ mI with m > 0 on the closed ball B(X̃, R), and ‖∇J(X̃)‖ < mR/2, then J has a unique minimizer X\* in B. It is interior, ∇J(X\*) = 0 and ∇²J(X\*) ⪰ mI. Hence X\* is a strict local minimizer, and J(X\*) ∈ [J(X̃) − ‖∇J(X̃)‖R, J(X̃)].

**Proposition (proved for each row of the table).** The discrete action J of the benchmark has a strict local minimizer X\* at distance at most R from the archived reference route X̃, with the Hessian bound and cost enclosure listed.

Two classes of variations are used:

- **Full waypoint space** (396 variables): every interior waypoint may move freely in the plane.
- **Transverse deformations** (198 variables): x_n = x̄_n + s_n ν_n, where x̄_n are fixed base waypoints and ν_n the unit normals of the route. This is local optimality of the **route shape**: no small deformation of the route decreases the travel time.

### Why the transverse class for the travel-time objective

The continuous travel time is invariant under reparametrization, so its second variation vanishes along the route. The discrete action inherits this only approximately: sliding waypoints along the route changes the cost only through the midpoint-rule quadrature error. Two situations then arise:

- **Circular and Techy.** The tangential curvature is tiny but positive, and a strict local minimum exists in the full space as well; both classes are certified. The smallest full-space eigenvalues (9e-7 and 8e-5) sit exactly in the tangential directions.
- **Four Vortices and Double Gyre.** At the certified optimal shape, the full-space Hessian has 64 and 55 negative eigenvalues respectively, all tangential. The cost can be lowered by about 5e-6 (relative) by redistributing waypoints along the route until consecutive waypoints merge, i.e. by exploiting the quadrature rule. So in the full waypoint space these routes are not local minima, and no strict local minimum exists nearby. This is a property of the discretised travel-time functional, not of the route geometry. The meaningful statement, which is certified, is local optimality of the shape.

For the fixed-time energy objective (Swirlys) the schedule is fixed and there is no such invariance, so the full-space certificate is the natural one.

## 3. How the rigour is obtained

- **Objective.** The segment costs and fields reproduce `routetools/cost.py` and `vectorfield.py`: midpoint field evaluation, sequential time propagation for time-dependent fields, and the fixed-time cost ½|d/h − w|²h. They were cross-checked against an independent plain-numpy transcription to 1e-16. For travel time the algebraically identical form Δt = |d|² / (√((d·w)² + (S² − |w|²)|d|²) + d·w) is used, and S² − |w|² > 0 is verified on every box (the smallest value is at least 0.19).
- **Gradient at the centre.** Rigorous 150-bit interval arithmetic (mpmath.iv), with dual numbers per segment and an adjoint of the time recursion t_{n+1} = t_n + Δt_n.
- **Hessian over the box ‖X − X̃‖∞ ≤ R**, which contains the ball. Interval second-order automatic differentiation in binary64 with outward rounding by one ulp after every +, −, ×, ÷, √ (IEEE-754, round-to-nearest). sin, cos and π are enclosed with mpmath.iv. For time-dependent fields the dense Hessian is propagated through the time recursion.
- **Eigenvalue bound.** The interval Hessian is split as centre ± radius. A floating-point Cholesky factorisation of (centre − sI) gives the bound, with a priori matrix-product rounding bounds (Higham, *Accuracy and Stability of Numerical Algorithms*, Thm 3.5). The result is m ≥ s − ‖residual‖₂ − ‖rounding‖₂ − ‖radius‖₂, with ‖·‖₂ bounded by the ∞-norm of symmetric non-negative matrices.
- **High-precision centres.** For the two full-space travel-time certificates the tangential eigenvalue is so small that the ball radius must be about 4e-16. The centre is therefore kept in 150-bit precision, polished by Newton steps to a gradient of 1e-36.
- **Self-test.** Interval enclosures of J, ∇J and ∇²J were checked against point evaluations at random points in boxes of radius 1e-6 to 1e-5 for all benchmarks: zero violations.
- **Constants.** π, 2/3, 1.7, 0.9, cos(π/6), etc. are treated as exact reals, enclosed in intervals. The code uses their binary64 values; the difference is about 1e-16 relative and far inside every margin, but formally the theorem is about the exact-constant problem.
- **Double Gyre amplitude.** A = 0.1 is the code default and is stated in Appendix A of the revised paper.

## 4. Additional Swirlys minimum

An independent search found a certified minimizer with J\* = 1.867883, whereas Table 5 reports 1.9702 ± 0.0010 for BERS (min 1.9690, max 1.9711 across seeds). Starting from a separate CMA-ES route (cost 2.68), trust-region Newton converges to the lower certified minimizer.

This establishes that the reported BERS routes are not globally optimal. It does **not** establish that they are not locally optimal, because Swirlys has multiple basins. The route-wise calculation in Section 5 resolves the local question directly: seeds 1, 3, and 4 lie near the certified local minimum at J\* = 1.968389, while seeds 0 and 2 reach near-singular stationary points for which the required interval Hessian bound is inconclusive. The revised paper therefore retains the reported BERS values, claims local certification only for the three successful Swirlys routes, and mentions the lower independent minimum solely to delimit the claim as local rather than global.

## 5. Certificates for the 25 exported BERS routes

The route-wise run is complete. The archive `bers_routes/` contains the exact
200-by-2 route array and JSON certificate for each of the five seeds of each
benchmark, plus `summary.json`. All 20 travel-time routes are certified in the
198-dimensional transverse family. Swirlys seeds 1, 3 and 4 are certified in
the full 396-dimensional waypoint space at J* = 1.9683892432. Swirlys seeds 0
and 2 reach stationary points with a smallest point-Hessian eigenvalue of
approximately 4.35e-10, but the interval lower bound over the required ball is
negative, so those two routes are not certified. Thus 23 of the 25 exported
BERS routes lie next to certified strict local minimizers.

Across the travel-time cases, the maximum normal offset is 0.00292 field units
and the largest relative cost gap is below 4.0e-5. For the three certified
Swirlys cases, the relative cost gap is below 3.6e-4. These are the values used
in the revised manuscript.

To repeat the batch from a Task 10 output directory:

```bash
cd supplementary/synthetic_local_optimality_proof
uv run --with mpmath==1.3.0 --with scipy \
  python certify_bers_routes.py /path/to/task10_synthetic_local_opt/curves
```

## 6. Claim used in the revised paper

For each of the 25 exported BERS routes, a Newton iteration first locates a nearby stationary point. The proof encloses ∇J at that point with 150-bit interval arithmetic and ∇²J over a ball with interval second-order automatic differentiation. A positive interval lower bound on the Hessian together with ‖∇J‖ < mR/2 certifies a unique strict local minimizer in the ball. This succeeds for all 20 travel-time routes with respect to transverse deformations and for three of five Swirlys routes in the full waypoint space. The two remaining Swirlys routes are reported as uncertified, not as failures of local minimality.

This replaces the Legendre-type argument in the earlier draft. The travel-time Lagrangian is 1-homogeneous in velocity, so its velocity Hessian is singular and cannot by itself establish strict local minimality.

## 7. Files

- `SPEC_synthetic_local_optimality.md`: specification.
- `certificates/*.json`: one certificate per row of the table, plus `summary.json`.
- `bers_routes/`: the 25 exported BERS routes, their route-wise certificates,
  and a combined `summary.json`.
- `routes/`: reference routes. `<name>.npy` is the full-space route; `<name>_family.npz` holds the transverse base, normals and s; `<name>_full_hpcentre.npy` holds the 150-bit centres as decimal strings; `<name>_cmaes.npy` is the stage-1 route.
- Code:
  - `jets.py`: interval arithmetic and second-order AD.
  - `problems.py`, `action.py`, `family.py`: objectives.
  - `hp.py`: 150-bit gradient.
  - `certify.py`, `certify_full_hp.py`: certificates.
  - `certify_route.py`: for BERS routes.
  - `certify_bers_routes.py`: batch driver for the 25 Task 10 routes.
  - `global_search.py`, `refine.py`, `refine_family.py`, `polish_hp.py`: route finding.
  - `selftest.py`, `reference_numpy.py`: checks.
- Reproduce with `bash run_all.sh`.
