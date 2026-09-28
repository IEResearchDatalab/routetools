# Computer-assisted proof of local optimality — synthetic benchmarks

Specification of the claim, the method, and the tasks. Scope: the five synthetic vector-field benchmarks of Section 3 (Circular, Four Vortices, Double Gyre, Techy, Swirlys). Real-ocean routes are out of scope here.

## 1. Why a new argument is needed

The current manuscript (§2.6, §3.4) checks a pointwise condition — positive-definiteness of the continuous velocity-Hessian along the route — and invokes Theorems 10/14 of Ferraro–Martín de Diego–Sato. That is not sufficient for a proof, for two reasons:

1. A positive velocity-Hessian is the Legendre condition. A local minimum additionally requires absence of conjugate points (Jacobi condition). At the discrete level both conditions together are exactly positive-definiteness of the **complete** Hessian of the discrete action with respect to all free waypoints. This must be checked route by route; it cannot be inferred from segment-wise quantities.
2. For the four travel-time benchmarks the segment cost Δt is 1-homogeneous in the displacement, so its Hessian with respect to velocity is singular along the direction of motion. The sentence in §3.4 "their Zermelo–Randers velocity-Hessians are positive definite" is incorrect as written (the positive-definite object is the Hessian of ½F², i.e. the Randers metric tensor).

Theorem 14 is also asymptotic (valid for an unspecified sufficiently small step); it never certifies the actual discretization used.

## 2. The discrete problems (exactly as implemented)

Reference implementation: `routetools/cost.py`, `routetools/vectorfield.py`, repository `IEResearchDatalab/routetools`, commit `8462156` (23 Sep 2026). Config: `config_noland.toml`.

- Route: L = 200 waypoints x_1,…,x_200 ∈ R², endpoints fixed (x_1 = source, x_200 = destination), 198 free waypoints, i.e. **396 free variables** X.
- Segment n (n = 1…199) has displacement d_n = x_{n+1} − x_n and midpoint m_n = (x_n + x_{n+1})/2. The field is evaluated at the midpoint.
- **Travel-time objective, constant speed S = 1** (Circular, Four Vortices, Double Gyre, Techy):
  Δt_n = [ √((d_n·w)² + (S² − |w|²)|d_n|²) − d_n·w ] / (S² − |w|²), with w = w(m_n, t_n),
  J(X) = Σ_n Δt_n, t_1 = 0, t_{n+1} = t_n + Δt_n.
  For time-invariant fields (Circular, Four Vortices) t is irrelevant; for Double Gyre and Techy the times couple all segments, so the Hessian is dense.
- **Fixed-time energy objective** (Swirlys, T = 30): h = T/199,
  J(X) = Σ_n ½ |d_n/h − w(m_n)|² h.
- Fields and parameters: Circular ω = −0.9; Four Vortices s = 1.7 with R_{a,b} = (−(y−b), x−a)/(3((x−a)²+(y−b)²)+1); Double Gyre A = 0.1, ε = 0.25, ω = 1; Techy s = −0.3, vortex t − 0.5; Swirlys w = (cos(2x−y−6), (2/3) sin y + x − 3). Endpoints as in Table 3. Constants (π, 2/3, cos π/6) are treated as exact real numbers, enclosed in intervals.

## 3. The claim to be proved (per benchmark)

> **Proposition.** For each benchmark there is a point X\* ∈ R³⁹⁶ such that
> (i) ∇J(X\*) = 0;
> (ii) ∇²J(X\*) ⪰ m I with an explicit m > 0, so X\* is a strict local minimizer of the discrete action J;
> (iii) ‖X\* − X̃‖₂ ≤ R for an explicit (tiny) R, where X̃ is an archived reference route;
> (iv) J(X\*) lies in an explicit interval.

Classes of variations. For the fixed-time energy objective the claim is in the full 396-dimensional waypoint space. For the travel-time objective, whose continuous version is invariant under reparametrization, the claim is stated for **transverse deformations** x_n = x̄_n + s_n ν_n (198 variables: fixed base waypoints x̄_n, unit normals ν_n), i.e. local optimality of the route shape. It is stated additionally in the full space where that holds (it does for Circular and Techy; it does not for Four Vortices and Double Gyre, where waypoints can be redistributed along the route to exploit the quadrature — see `certificate_report.md` §2).

Link to BERS: the archived BERS route X_BERS for each benchmark is compared with X̃ (distance and cost gap). The paper can then state that the BERS route lies within δ of a certified strict local minimizer of the discrete problem whose cost agrees with the reported value.

What is **not** claimed: global optimality; optimality for the continuous problem; convergence of FMS from arbitrary starts; any statement about the real-ocean routes.

## 4. Method

**Lemma (strong convexity on a ball).** Let B be the closed Euclidean ball of radius R centred at X̃. If ∇²J(X) ⪰ m I for every X ∈ B, with m > 0, and ‖∇J(X̃)‖₂ < mR/2, then J has a unique minimizer X\* on B, it lies in the interior of B, ∇J(X\*) = 0 and ∇²J(X\*) ⪰ m I.
*Proof.* For X ∈ ∂B, Taylor with the curvature bound gives J(X) ≥ J(X̃) − ‖∇J(X̃)‖R + mR²/2 > J(X̃). A continuous function on the compact ball attains its minimum, which cannot be on ∂B; hence it is an interior critical point, and strong convexity on B makes it unique and strict. ∎

Steps:

1. **Reference route X̃.** Compute a stationary point of J to near machine precision (Newton with exact second derivatives, started from a route in the BERS basin; cost must reproduce Table 3/5 values).
2. **Rigorous gradient.** Enclose ∇J(X̃) with interval arithmetic; take an upper bound g of its Euclidean norm.
3. **Rigorous Hessian over the ball.** Enclose ∇²J(X) for all X in the box ‖X − X̃‖∞ ≤ R (which contains the ball) by interval second-order automatic differentiation; for time-dependent fields the dense Hessian is propagated through the time recursion t_{n+1} = t_n + Δt_n.
4. **Rigorous eigenvalue bound.** Write the interval Hessian as H_c ± Δ. Choose a shift s, compute a floating-point Cholesky factor of H_c − sI, and bound the residual E with standard floating-point error bounds (Higham's γ_n). Then m ≥ s − ‖E‖₂ − ‖Δ‖₂.
5. **Check** g < mR/2. If it holds, the Proposition is proved for that benchmark.

**Rigour assumptions** (to be stated in the paper): IEEE-754 binary64 arithmetic with round-to-nearest for +, −, ×, ÷, √ (outward rounding by one ulp after each operation); transcendental functions (sin, cos) and π enclosed with `mpmath.iv` interval arithmetic; matrix-product rounding bounded a priori (Higham, *Accuracy and Stability of Numerical Algorithms*, Thm 3.5). The code, reference routes and certificate logs are archived for independent re-checking.

## 5. Tasks

| # | Task | Owner |
|---|---|---|
| 1 | Implement J, its jets, and the interval certificate (steps 1–5) for all five benchmarks | Done; see `certificate_report.md` |
| 2 | Export the 25 actual BERS routes at σ0 = 2, K = 9 as `.npy` files | Done; see `bers_routes/` |
| 3 | Certify each exported route and report its offset or distance and cost gap | Done; 23/25 certified |
| 4 | Correct the manuscript's Legendre/Jacobi discussion and state the interval argument | Done |
| 5 | State A = 0.1 for Double Gyre in Appendix A | Done |

## 6. Possible outcomes and how to report them

- **Certified:** the benchmark gets the Proposition.
- **Hessian indefinite at the stationary point:** the route is a saddle of the discrete problem. Report honestly and investigate (e.g. waypoint-sliding modes for the travel-time objective, or a conjugate point).
- **Positive-definite but too ill-conditioned to certify in double precision:** refine X̃ in extended precision and repeat; if still inconclusive, report as numerical evidence only.
