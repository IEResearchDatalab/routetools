"""Find stationary routes (in the basin of the reported BERS solutions) with a
trust-region Newton method using the exact Hessian, then polish with pure
Newton steps. Saves reference routes X~ to routes/<name>.npy (200 x 2)."""
import sys
import json
import numpy as np
from scipy.optimize import minimize
from problems import PROBLEMS
from action import evaluate, endpoints, NVAR, NFREE

BIG = 1e6


def make_fun(p):
    cache = {}

    def full(X):
        key = X.tobytes()
        if key not in cache:
            cache.clear()
            try:
                with np.errstate(all="raise"):
                    J, G, H, dm = evaluate(p, X)
                if (dm is not None and dm <= 0) or not np.isfinite(J):
                    raise FloatingPointError
            except (FloatingPointError, ZeroDivisionError):
                J, G, H = BIG, np.zeros(NVAR), np.eye(NVAR)
            cache[key] = (J, G, H)
        return cache[key]

    return (lambda X: full(X)[0]), (lambda X: full(X)[1]), (lambda X: full(X)[2])


def initial_curve(p, coeffs):
    """Straight line plus sine modes (normal and tangential), coeffs: (2, m)."""
    src, dst = endpoints(p, False)
    src, dst = np.array(src), np.array(dst)
    s = np.linspace(0, 1, 200)
    base = src[None, :] + (dst - src)[None, :] * s[:, None]
    tvec = dst - src
    nvec = np.array([-tvec[1], tvec[0]])
    for k in range(coeffs.shape[1]):
        mode = np.sin((k + 1) * np.pi * s)[:, None]
        base = base + coeffs[0, k] * mode * nvec[None, :] + coeffs[1, k] * 0.05 * mode * tvec[None, :]
    return base


def newton_polish(p, X, iters=30):
    best = None
    for _ in range(iters):
        J, G, H, dm = evaluate(p, X)
        gn = np.linalg.norm(G)
        if best is None or gn < best[1]:
            best = (X.copy(), gn, J)
        try:
            step = np.linalg.solve(H, G)
        except np.linalg.LinAlgError:
            break
        X = X - step
        if np.linalg.norm(step) < 1e-15:
            break
    return best


def solve(name, nstarts=12, seed=0, maxiter=400):
    p = PROBLEMS[name]
    f, g, h = make_fun(p)
    rng = np.random.default_rng(seed)
    results = []
    starts = [np.zeros((2, 4))] + [rng.normal(0, 0.35, (2, 4)) / (1 + np.arange(4)) for _ in range(nstarts - 1)]
    for k, c in enumerate(starts):
        X0 = initial_curve(p, c)[1:-1].ravel()
        if f(X0) >= BIG:
            continue
        r = minimize(f, X0, jac=g, hess=h, method="trust-exact",
                     options={"maxiter": maxiter, "gtol": 1e-10})
        Xb, gn, J = newton_polish(p, r.x)
        lam = np.linalg.eigvalsh(evaluate(p, Xb)[2])
        results.append((J, gn, lam[0], Xb))
        print(f"  {name} start {k:2d}: J={J:.10f} |g|={gn:.2e} lam_min={lam[0]:.3e} lam_max={lam[-1]:.3e}", flush=True)
    results.sort(key=lambda r: r[0])
    return p, results


if __name__ == "__main__":
    import os
    os.makedirs("routes", exist_ok=True)
    names = sys.argv[1:] or list(PROBLEMS)
    summary = {}
    for name in names:
        n = 6 if PROBLEMS[name].time_variant else 12
        p, res = solve(name, nstarts=n)
        J, gn, lam, X = res[0]
        src, dst = endpoints(p, False)
        curve = np.vstack([src, X.reshape(NFREE, 2), dst])
        np.save(f"routes/{name}.npy", curve)
        summary[name] = {"J": J, "grad_norm": gn, "lam_min": lam, "reported_BERS": p.ref_cost,
                         "all_minima_J": sorted({round(r[0], 8) for r in res})}
        print(name, summary[name], flush=True)
    with open("routes/summary_" + "_".join(names) + ".json", "w") as fh:
        json.dump(summary, fh, indent=1)
