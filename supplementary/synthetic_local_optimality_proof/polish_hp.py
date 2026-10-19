"""Polish the reference route with Newton steps driven by the 150-bit gradient
(float Hessian as the Newton matrix), down to the limit of binary64
representability of the waypoint coordinates."""
import sys
import time
import numpy as np
from mpmath import iv
from problems import PROBLEMS
from action import evaluate
from family import eval_family
from hp import family_grad_hp, _grad_hp_points, norm_upper, mid_float

name, mode = sys.argv[1], sys.argv[2]
p = PROBLEMS[name]
t0 = time.time()
if mode == "family":
    d = dict(np.load(f"routes/{name}_family.npz"))
    s = d["s"].copy()
    xbar, nu = d["xbar"], d["nu"]
    grad = lambda z: family_grad_hp(p, xbar, nu, z)
    hess = lambda z: eval_family(p, xbar, nu, z)[2]
else:
    curve = np.load(f"routes/{name}.npy")
    s = curve[1:-1].ravel().copy()
    grad = lambda z: _grad_hp_points(p, [[iv.mpf(float(z[2 * i])), iv.mpf(float(z[2 * i + 1]))]
                                          for i in range(z.size // 2)])
    hess = lambda z: evaluate(p, z)[2]

best = None
for it in range(6):
    J, G = grad(s)
    gup = norm_upper(G)
    print(f"  it {it}: ||grad||<= {float(gup):.3e}  J in [{float(J.a):.17f}, {float(J.b):.17f}]", flush=True)
    if best is None or gup < best[1]:
        best = (s.copy(), gup)
    else:
        break
    step = np.linalg.solve(hess(s), mid_float(G))
    s = s - step
s = best[0]
if mode == "family":
    d["s"] = s
    np.savez(f"routes/{name}_family.npz", **d)
else:
    curve[1:-1] = s.reshape(-1, 2)
    np.save(f"routes/{name}.npy", curve)
print(f"{name} {mode}: polished, ||grad|| <= {float(best[1]):.3e}  ({time.time()-t0:.0f}s)")
