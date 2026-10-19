"""
Full-waypoint-space certificate with a high-precision centre.

In the full space the travel-time Hessian has a very small (tangential)
eigenvalue, so the ball radius must be ~1e-16; that requires ||grad J(c)|| ~ 1e-22,
below what a binary64 centre can reach. The centre c is therefore kept in
150-bit precision (Newton steps: 150-bit interval gradient, float Hessian), the
gradient bound is computed in 150-bit interval arithmetic, and the Hessian is
enclosed over the binary64 box [down(c - R), up(c + R)] which contains B(c, R).
"""
import sys
import json
import time
import numpy as np
from mpmath import iv, mp
from jets import IA
from problems import PROBLEMS
from action import evaluate
from hp import _grad_hp_points, norm_upper, PREC
from certify import lambda_min_lower_bound

name = sys.argv[1]
p = PROBLEMS[name]
t0 = time.time()
mp.prec = PREC
iv.prec = PREC
X0 = np.load(f"routes/{name}.npy")[1:-1].ravel()
c = [mp.mpf(float(v)) for v in X0]


def grad_at(c):
    pts = [[iv.mpf(c[2 * i]), iv.mpf(c[2 * i + 1])] for i in range(len(c) // 2)]
    return _grad_hp_points(p, pts)


Hf = evaluate(p, X0)[2]
for it in range(5):
    J, G = grad_at(c)
    gup = norm_upper(G)
    print(f"  it {it}: ||grad|| <= {mp.nstr(gup, 5)}", flush=True)
    if gup < mp.mpf("1e-30"):
        break
    gmid = np.array([float((gi.a + gi.b) / 2) for gi in G])
    step = np.linalg.solve(Hf, gmid)
    c = [ci - mp.mpf(float(si)) for ci, si in zip(c, step)]

J, G = grad_at(c)
g_up = norm_upper(G)
cf = np.array([float(ci) for ci in c])
lam_f = np.linalg.eigvalsh(evaluate(p, cf)[2])[0]
ulp = np.max(np.spacing(np.abs(cf)))
R = max(10 * float(g_up) / lam_f, 4 * ulp)
iv.prec = PREC
lo = np.array([np.nextafter(float((iv.mpf(ci) - iv.mpf(R)).a), -np.inf) for ci in c])
hi = np.array([np.nextafter(float((iv.mpf(ci) + iv.mpf(R)).b), np.inf) for ci in c])
iv.prec = 80
_, _, Hb, den_b = evaluate(p, IA(lo, hi), interval=True, hessian=True)
m_lo, info = lambda_min_lower_bound(Hb.lo, Hb.hi)
iv.prec = PREC
ok, margin = False, None
if m_lo is not None and m_lo > 0:
    rhs = ((iv.mpf(m_lo) * iv.mpf(R)) / 2).a
    ok = bool(iv.mpf(g_up).b < rhs)
    margin = float(rhs) / float(iv.mpf(g_up).b)
res = {"problem": name, "mode": "full(hp-centre)", "n_vars": len(c), "certified": ok,
       "margin_mR2_over_g": margin,
       "J_star_enclosure": [float((J - iv.mpf(g_up) * iv.mpf(R)).a), float(J.b)],
       "grad_norm_upper": float(g_up), "R": R, "m_lower": None if m_lo is None else float(m_lo),
       "float_lam_min_center": float(lam_f), "den_min_lower_on_box": den_b,
       "seconds": round(time.time() - t0, 1), **{"eig_" + k: v for k, v in info.items()}}
for k, v in res.items():
    print(f"  {k}: {v}")
json.dump(res, open(f"certificates/{name}_full.json", "w"), indent=1)
np.save(f"routes/{name}_full_hpcentre.npy", np.array([mp.nstr(ci, 45) for ci in c]))
