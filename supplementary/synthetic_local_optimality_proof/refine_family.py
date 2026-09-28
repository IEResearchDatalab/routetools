"""Stationary point of the travel-time action within the transverse family
x_n = xbar_n + s_n nu_n (base = arc-length resampling of the stage-1 curve).
Saves routes/<name>_family.npz with xbar, nu, s (reference s~)."""
import sys
import json
import numpy as np
from scipy.optimize import minimize
from problems import PROBLEMS
from family import normals, eval_family

name = sys.argv[1]
p = PROBLEMS[name]
fine = np.load(f"routes/{name}_cmaes.npy")
seg = np.linalg.norm(np.diff(fine, axis=0), axis=1)
s_ = np.concatenate([[0], np.cumsum(seg)])
s_ /= s_[-1]
tgt = np.linspace(0, 1, fine.shape[0])
base = np.stack([np.interp(tgt, s_, fine[:, 0]), np.interp(tgt, s_, fine[:, 1])], 1)
base[0], base[-1] = fine[0], fine[-1]
xbar = base[1:-1].copy()
nu = normals(base)

cache = {}


def full(s):
    k = s.tobytes()
    if k not in cache:
        cache.clear()
        try:
            with np.errstate(all="raise"):
                J, G, H, dm = eval_family(p, xbar, nu, s)
            if (dm is not None and dm <= 0) or not np.isfinite(J):
                raise FloatingPointError
        except FloatingPointError:
            J, G, H = 1e6, np.zeros(198), np.eye(198)
        cache[k] = (J, G, H)
    return cache[k]


s0 = np.zeros(198)
print(f"{name}: base J={full(s0)[0]:.12f}", flush=True)
r = minimize(lambda s: full(s)[0], s0, jac=lambda s: full(s)[1], hess=lambda s: full(s)[2],
             method="trust-exact", options={"maxiter": 2000, "gtol": 1e-12})
s = r.x
best = None
for _ in range(40):
    J, G, H, _ = eval_family(p, xbar, nu, s)
    gn = np.linalg.norm(G)
    if best is None or gn < best[1]:
        best = (s.copy(), gn, J)
    step = np.linalg.solve(H, G)
    s = s - step
    if np.linalg.norm(step) < 1e-16:
        break
s, gn, J = best
H = eval_family(p, xbar, nu, s)[2]
lam = np.linalg.eigvalsh(H)
print(f"  trust-exact nit={r.nit} msg={r.message}")
print(f"  family stationary: J={J:.14f} |g_s|={gn:.3e} lam_min={lam[0]:.4e} lam_max={lam[-1]:.4e} "
      f"max|s|={np.abs(s).max():.3e} reported BERS={p.ref_cost}", flush=True)
np.savez(f"routes/{name}_family.npz", xbar=xbar, nu=nu, s=s, src=base[0], dst=base[-1])
json.dump({"J": J, "grad_norm": float(gn), "lam_min": float(lam[0]), "lam_max": float(lam[-1])},
          open(f"routes/{name}_family.json", "w"), indent=1)
