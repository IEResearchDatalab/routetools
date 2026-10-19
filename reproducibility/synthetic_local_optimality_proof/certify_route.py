"""
Certify local optimality for a given route file (e.g. an exported BERS route).

  python certify_route.py <problem> <route.npy>

<route.npy>: array (200, 2) with the waypoints, endpoints included.

Travel-time problems (circular, fourvortices, doublegyre, techy):
  base = the given route, normals from central differences; Newton in the
  transverse coordinates s (float Hessian + 150-bit gradient) converges to the
  nearby transverse stationary point s*, which is then certified. Reported:
  max |s*| (how far the given route is from the certified minimizer, measured
  along the normals) and the cost gap J(route) - J(X*).
Fixed-time problem (swirlys): Newton in the full 396-dimensional space from the
  given route, then full certification. Reported: ||X* - route|| and cost gap.
"""
import sys
import json
import numpy as np
from mpmath import iv
from problems import PROBLEMS
from action import evaluate
from family import normals, eval_family
from hp import family_grad_hp, _grad_hp_points, norm_upper, mid_float
from certify import certify


def newton(grad_hp, hess, z, iters=8):
    best = None
    for _ in range(iters):
        J, G = grad_hp(z)
        g = norm_upper(G)
        if best is None or g < best[1]:
            best = (z.copy(), g)
        else:
            break
        z = z - np.linalg.solve(hess(z), mid_float(G))
    return best[0]


def float_newton(fun, z, iters=60):
    from scipy.optimize import minimize
    r = minimize(lambda q: fun(q)[0], z, jac=lambda q: fun(q)[1], hess=lambda q: fun(q)[2],
                 method="trust-exact", options={"maxiter": 3000, "gtol": 1e-11})
    z = r.x
    for _ in range(iters):
        J, G, H, _ = fun(z)
        step = np.linalg.solve(H, G)
        z = z - step
        if np.linalg.norm(step) < 1e-15 * max(1, np.linalg.norm(z)):
            break
    return z


def main(name, path):
    p = PROBLEMS[name]
    route = np.load(path).astype(float)
    assert route.shape == (200, 2)
    J0 = evaluate(p, route[1:-1].ravel(), hessian=False)[0]
    out = {"problem": name, "route_file": path, "J_route": J0}
    if name != "swirlys":
        xbar, nu = route[1:-1].copy(), normals(route)
        fun = lambda s: eval_family(p, xbar, nu, s)
        s = float_newton(fun, np.zeros(198))
        s = newton(lambda z: family_grad_hp(p, xbar, nu, z), lambda z: fun(z)[2], s)
        res = certify(name, "family", s, family_data=(xbar, nu))
        out.update(res)
        out["max_abs_transverse_offset_to_certified_min"] = float(np.max(np.abs(s)))
    else:
        X = route[1:-1].ravel().copy()
        fun = lambda z: evaluate(p, z)
        X = float_newton(fun, X)
        X = newton(lambda z: _grad_hp_points(p, [[iv.mpf(float(z[2 * i])), iv.mpf(float(z[2 * i + 1]))]
                                                 for i in range(z.size // 2)]),
                   lambda z: fun(z)[2], X)
        res = certify(name, "full", X)
        out.update(res)
        out["dist_route_to_certified_min"] = float(np.linalg.norm(X - route[1:-1].ravel()))
    out["cost_gap_route_minus_certified"] = J0 - out["J_star_enclosure"][1]
    for k, v in out.items():
        print(f"  {k}: {v}")
    return out


if __name__ == "__main__":
    o = main(sys.argv[1], sys.argv[2])
    json.dump(o, open(sys.argv[2].replace(".npy", "_certificate.json"), "w"), indent=1)
