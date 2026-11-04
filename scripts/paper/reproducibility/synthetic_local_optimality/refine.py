"""Stage 2: from the CMA-ES curve, converge to a stationary point of the discrete
action with a trust-region Newton method (exact Hessian), then pure Newton
polishing. Saves routes/<name>.npy (the reference route X~) and a summary."""
import sys
import json
import numpy as np
from scipy.optimize import minimize
from problems import PROBLEMS
from action import evaluate, NFREE
from find_routes import make_fun, newton_polish

name = sys.argv[1]
maxiter = 3000
p = PROBLEMS[name]
curve0 = np.load(f"routes/{name}_cmaes.npy")
if "--arclength" in sys.argv:
    # resample the CMA-ES curve at equal arc length (same geometry, better spacing)
    fine = np.load(f"routes/{name}_cmaes.npy")
    seg = np.linalg.norm(np.diff(fine, axis=0), axis=1)
    s_ = np.concatenate([[0], np.cumsum(seg)]); s_ /= s_[-1]
    tgt = np.linspace(0, 1, fine.shape[0])
    curve0 = np.stack([np.interp(tgt, s_, fine[:, 0]), np.interp(tgt, s_, fine[:, 1])], 1)
X0 = curve0[1:-1].ravel()
f, g, h = make_fun(p)
print(f"{name}: start J={f(X0):.10f}", flush=True)
r = minimize(f, X0, jac=g, hess=h, method="trust-exact",
             options={"maxiter": maxiter, "gtol": 1e-11})
print(f"  trust-exact: J={r.fun:.12f} |g|={np.linalg.norm(r.jac):.2e} nit={r.nit} msg={r.message}", flush=True)
Xb, gn, J = newton_polish(p, r.x, iters=40)
H = evaluate(p, Xb)[2]
lam = np.linalg.eigvalsh(H)
print(f"  polished: J={J:.14f} |g|={gn:.3e} lam_min={lam[0]:.4e} lam_2={lam[1]:.4e} lam_max={lam[-1]:.4e} "
      f"reported BERS={p.ref_cost}", flush=True)
curve = np.vstack([curve0[0], Xb.reshape(NFREE, 2), curve0[-1]])
np.save(f"routes/{name}.npy", curve)
json.dump({"J": J, "grad_norm": float(gn), "lam_min": float(lam[0]), "lam_max": float(lam[-1]),
           "reported_BERS_mean": p.ref_cost, "nit": int(r.nit)},
          open(f"routes/{name}_refine.json", "w"), indent=1)
