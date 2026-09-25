"""Task 10 (R1-2) — route-wise local-optimality audit for the synthetic fields.

For the five literature vector fields the segment cost is an explicit smooth
function (Zermelo travel time under constant speed, or the quadratic
fixed-time fuel cost of Swirlys), so the complete discrete action of a route
can be differentiated exactly by automatic differentiation.

This script reruns BERS (CMA-ES followed by FMS) with the configuration of
Task 2 (sigma0 = 2.0, K = 9, P = 500, L = 200, five seeds) and, for every final
route, evaluates with both endpoints fixed:

1. the gradient of the discrete action with respect to the interior
   waypoints;
2. the complete (dense) discrete Hessian and its smallest eigenvalue;
3. the *reduced* gradient and Hessian obtained by restricting each interior
   waypoint to move along the local normal of the route.  For the travel-time
   objective, sliding a waypoint along the route leaves the action unchanged
   to first order (a discretisation gauge freedom), so the complete Hessian is
   only positive semi-definite in those directions; the reduced Hessian
   removes this degeneracy.  For the fixed-time fuel objective the segment
   times are fixed and the complete Hessian itself is tested;
4. the Newton correction implied by (reduced) gradient and Hessian, compared
   with a tolerance of ``--stationarity-relative`` times the median segment
   length;
5. the drift-to-speed ratio max ||w|| / S along the route (time objectives).

A route is reported as a numerical local minimum when the relevant Hessian is
positive definite and the Newton correction is below tolerance.  This is a
numerical certificate for the discretised problem, not an analytic proof.

Runs on CPU in a few minutes (Swirlys dominates)::

    uv run python revision/task10_synthetic_local_optimality.py

Outputs (under ``--output-dir``, default ``output/task10_synthetic_local_opt``)
------------------------------------------------------------------------------
- ``task10_routes.csv``: one row per (field, seed).
- ``task10_summary.csv`` and ``task10_summary.tex``: aggregate per field.
"""

from __future__ import annotations

import csv
import tomllib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import typer

from routetools.cmaes import optimize
from routetools.cost import (
    cost_function,
    cost_function_constant_speed_time_variant,
)
from routetools.fms import optimize_fms

REPO_ROOT = Path(__file__).resolve().parent.parent
FIELDS = ["circular", "fourvortices", "doublegyre", "techy", "swirlys"]
LABELS = {
    "circular": "Circular",
    "fourvortices": "Four Vortices",
    "doublegyre": "Double Gyre",
    "techy": "Techy",
    "swirlys": "Swirlys",
}
SEEDS = [0, 1, 2, 3, 4]
CMAES_KWARGS = dict(
    K=9,
    L=200,
    num_pieces=1,
    popsize=500,
    sigma0=2.0,
    tolfun=1e-3,
    damping=1.0,
    maxfevals=500_000,
)
FMS_KWARGS = dict(patience=50, damping=0.5, maxfevals=500_000)


def _field(name: str):
    module = __import__("routetools.vectorfield", fromlist=[f"vectorfield_{name}"])
    return getattr(module, f"vectorfield_{name}")


def _action_function(vectorfield, src, dst, travel_stw, travel_time):
    def action(interior_flat: jnp.ndarray) -> jnp.ndarray:
        interior = interior_flat.reshape((-1, 2))
        curve = jnp.vstack([src[None, :], interior, dst[None, :]])
        return cost_function(
            vectorfield,
            curve[None, ...],
            travel_stw=travel_stw,
            travel_time=travel_time,
            spherical_correction=False,
        )[0]

    return action


def _normals(curve: np.ndarray) -> np.ndarray:
    tangent = curve[2:] - curve[:-2]
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    return np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)


def _local_model(action, curve: np.ndarray, time_objective: bool):
    """Return gradient, Hessian and basis of the tested free-variable space."""
    z = jnp.asarray(curve[1:-1].reshape(-1))
    gradient = np.asarray(jax.grad(action)(z))
    hessian = np.asarray(jax.hessian(action)(z))
    hessian = 0.5 * (hessian + hessian.T)
    if time_objective:
        normals = _normals(curve)
        n_int = normals.shape[0]
        basis = np.zeros((2 * n_int, n_int))
        for i in range(n_int):
            basis[2 * i : 2 * i + 2, i] = normals[i]
    else:
        basis = np.eye(gradient.size)
    g_test = basis.T @ gradient
    h_test = basis.T @ hessian @ basis
    eig_test = np.linalg.eigvalsh(h_test)
    scale = max(float(np.max(np.abs(eig_test))), np.finfo(float).tiny)
    pd = bool(eig_test[0] > 1e-10 * scale)
    step = -np.linalg.solve(h_test, g_test) if pd else None
    return gradient, hessian, basis, g_test, eig_test, pd, step


def _max_displacement(basis: np.ndarray, step: np.ndarray) -> float:
    return float(np.max(np.linalg.norm((basis @ step).reshape(-1, 2), axis=1)))


def _drift_to_speed(vectorfield, curve: np.ndarray, travel_stw: float) -> float:
    mid = 0.5 * (curve[:-1] + curve[1:])
    if getattr(vectorfield, "is_time_variant", False):
        dt = np.asarray(
            cost_function_constant_speed_time_variant(
                vectorfield,
                jnp.asarray(curve[None]),
                travel_stw,
                spherical_correction=False,
            )
        )[0]
        t = np.concatenate([[0.0], np.cumsum(dt)[:-1]])
    else:
        t = np.zeros(len(mid))
    u, v = vectorfield(jnp.asarray(mid[:, 0]), jnp.asarray(mid[:, 1]), jnp.asarray(t))
    return float(np.max(np.hypot(np.asarray(u), np.asarray(v))) / travel_stw)


def audit_route(
    vf_name: str,
    vf_config: dict,
    curve: np.ndarray,
    stationarity_relative: float,
    polish_iterations: int = 50,
) -> dict[str, object]:
    """Audit one exported route, then polish it with safeguarded Newton steps.

    The polishing step is the coupled Newton iteration on the complete
    (reduced) discrete Hessian that FMS approximates with per-waypoint
    updates.  It is applied only while the tested Hessian is positive
    definite, with a backtracking line search on the discrete action, so it
    can only lower the cost.  It locates the nearby numerical local minimum
    and measures how far the exported BERS route is from it.
    """
    vectorfield = _field(vf_name)
    src = jnp.asarray(curve[0])
    dst = jnp.asarray(curve[-1])
    travel_stw = vf_config.get("travel_stw")
    travel_time = vf_config.get("travel_time")
    time_objective = travel_stw is not None
    action = _action_function(vectorfield, src, dst, travel_stw, travel_time)

    segment = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    tolerance = stationarity_relative * float(np.median(segment))

    value0 = float(action(jnp.asarray(curve[1:-1].reshape(-1))))
    gradient, hessian, basis, g_test, eig_test, pd, step = _local_model(
        action, curve, time_objective
    )
    eig_full = np.linalg.eigvalsh(hessian)
    correction0 = _max_displacement(basis, step) if pd else float("nan")
    predicted0 = float(-0.5 * g_test @ step) if pd else float("nan")

    # Safeguarded coupled Newton polishing.  Where the tested Hessian is not
    # positive definite, a Levenberg--Marquardt shift makes the step a descent
    # direction; the certificate is only issued at a point where the unshifted
    # Hessian is positive definite and the pure Newton correction is below
    # tolerance.
    polished = curve.copy()
    value = value0
    iterations = 0
    pd_p, step_p, basis_p, eig_p = pd, step, basis, eig_test
    g_p = g_test
    h_p = None
    correction = correction0
    while iterations < polish_iterations and not (pd_p and correction <= tolerance):
        if pd_p:
            direction = step_p
        else:
            _, hess_full, basis_p, g_p, eig_p, _, _ = _local_model(
                action, polished, time_objective
            )
            h_p = basis_p.T @ hess_full @ basis_p
            shift = 1.1 * max(-float(eig_p[0]), 0.0) + 1e-8 * float(
                np.max(np.abs(eig_p))
            )
            direction = -np.linalg.solve(h_p + shift * np.eye(h_p.shape[0]), g_p)
        alpha = 1.0
        accepted = False
        for _ in range(30):
            trial = polished.copy()
            trial[1:-1] += alpha * (basis_p @ direction).reshape(-1, 2)
            trial_value = float(action(jnp.asarray(trial[1:-1].reshape(-1))))
            if np.isfinite(trial_value) and trial_value <= value:
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            break
        polished, value = trial, trial_value
        iterations += 1
        _, _, basis_p, g_p, eig_p, pd_p, step_p = _local_model(
            action, polished, time_objective
        )
        correction = _max_displacement(basis_p, step_p) if pd_p else float("nan")

    stationary = bool(pd_p and correction <= tolerance)
    shift = float(np.max(np.linalg.norm(polished - curve, axis=1)))

    row = {
        "action": value0,
        "gradient_norm": float(np.linalg.norm(gradient)),
        "full_hessian_min_eig": float(eig_full[0]),
        "full_hessian_max_eig": float(eig_full[-1]),
        "test_hessian": "reduced-normal" if time_objective else "complete",
        "test_hessian_min_eig": float(eig_test[0]),
        "test_hessian_max_eig": float(eig_test[-1]),
        "test_hessian_pd": pd,
        "max_newton_correction": correction0,
        "predicted_newton_decrease": predicted0,
        "stationarity_tolerance": tolerance,
        "median_segment_length": float(np.median(segment)),
        "polish_iterations": iterations,
        "polished_action": value,
        "polished_relative_cost_change": (value - value0) / abs(value0),
        "polished_max_shift": shift,
        "polished_max_shift_over_segment": shift / float(np.median(segment)),
        "polished_hessian_min_eig": float(eig_p[0]),
        "polished_hessian_pd": bool(pd_p),
        "polished_max_newton_correction": correction,
        "polished_stationary": stationary,
        "numerical_local_minimum": bool(pd_p and stationary),
        "max_drift_to_speed": (
            _drift_to_speed(vectorfield, curve, travel_stw)
            if time_objective
            else float("nan")
        ),
    }
    return row


def main(
    output_dir: str = "output/task10_synthetic_local_opt",
    stationarity_relative: float = 1.0e-3,
    config_path: str = "config.toml",
    fields: str = ",".join(FIELDS),
    seeds: str = ",".join(str(s) for s in SEEDS),
) -> None:
    """Rerun the synthetic BERS routes and audit each one."""
    out = Path(output_dir)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    with cfg_path.open("rb") as fh:
        config = tomllib.load(fh)

    # 1) Optimise in the default (float32) precision used by Task 2.
    field_list = [f for f in fields.split(",") if f]
    seed_list = [int(s) for s in seeds.split(",") if s]
    curve_dir = out / "curves"
    curve_dir.mkdir(exist_ok=True)
    runs: list[tuple[str, int, np.ndarray, float]] = []
    for vf_name in field_list:
        vf_config = config["vectorfield"][vf_name]
        vectorfield = _field(vf_name)
        for seed in seed_list:
            cached = curve_dir / f"{vf_name}_seed{seed}.npz"
            if cached.exists():
                data = np.load(cached)
                runs.append((vf_name, seed, data["curve"], float(data["cost"])))
                print(f"loaded {vf_name} seed={seed}")
                continue
            curve_cmaes, _ = optimize(
                vectorfield,
                jnp.array(vf_config["src"]),
                jnp.array(vf_config["dst"]),
                travel_stw=vf_config.get("travel_stw"),
                travel_time=vf_config.get("travel_time"),
                seed=seed,
                verbose=False,
                **CMAES_KWARGS,
            )
            curve_bers, info = optimize_fms(
                vectorfield,
                curve=curve_cmaes,
                travel_stw=vf_config.get("travel_stw"),
                travel_time=vf_config.get("travel_time"),
                seed=seed,
                verbose=False,
                **FMS_KWARGS,
            )
            curve = np.asarray(curve_bers, dtype=np.float64)
            if curve.ndim == 3:
                curve = curve[0]
            cost = float(np.asarray(info["cost"]).reshape(-1)[0])
            np.savez(cached, curve=curve, cost=cost)
            runs.append((vf_name, seed, curve, cost))
            print(f"optimised {vf_name} seed={seed} cost={cost:.4f}")

    # 2) Audit the exported routes in float64.
    jax.config.update("jax_enable_x64", True)
    rows: list[dict[str, object]] = []
    for vf_name, seed, curve, cost in runs:
        vf_config = config["vectorfield"][vf_name]
        row = {"vectorfield": vf_name, "seed": seed, "bers_cost": cost}
        row.update(audit_route(vf_name, vf_config, curve, stationarity_relative))
        print(
            f"{vf_name:>13} seed={seed} PD={row['test_hessian_pd']} "
            f"corr={row['max_newton_correction']:.2e} "
            f"tol={row['stationarity_tolerance']:.2e}"
        )
        rows.append(row)

    with (out / "task10_routes.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for vf_name in field_list:
        group = [r for r in rows if r["vectorfield"] == vf_name]
        summary.append(
            {
                "vectorfield": vf_name,
                "n_routes": len(group),
                "hessian_pd": sum(bool(r["test_hessian_pd"]) for r in group),
                "polished_stationary": sum(
                    bool(r["polished_stationary"]) for r in group
                ),
                "numerical_local_minimum": sum(
                    bool(r["numerical_local_minimum"]) for r in group
                ),
                "max_newton_correction": max(
                    float(r["max_newton_correction"]) for r in group
                ),
                "stationarity_tolerance": min(
                    float(r["stationarity_tolerance"]) for r in group
                ),
                "max_polished_relative_cost_change": max(
                    abs(float(r["polished_relative_cost_change"])) for r in group
                ),
                "max_polished_shift_over_segment": max(
                    float(r["polished_max_shift_over_segment"]) for r in group
                ),
                "min_test_eigenvalue": min(
                    float(r["test_hessian_min_eig"]) for r in group
                ),
                "max_drift_to_speed": max(
                    float(r["max_drift_to_speed"]) for r in group
                ),
            }
        )
    with (out / "task10_summary.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    tex = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Field & Routes & Hessian PD & Stationary (polished) & Local minimum \\",
        r"\midrule",
    ]
    for s in summary:
        tex.append(
            f"{LABELS[s['vectorfield']]} & {s['n_routes']} & {s['hessian_pd']} & "
            f"{s['polished_stationary']} & {s['numerical_local_minimum']} \\\\"
        )
    tex += [r"\bottomrule", r"\end{tabular}"]
    (out / "task10_summary.tex").write_text("\n".join(tex) + "\n")
    print(f"Wrote results to {out}")


if __name__ == "__main__":
    typer.run(main)
