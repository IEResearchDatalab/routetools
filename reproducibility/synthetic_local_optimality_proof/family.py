"""
Transverse-deformation family for the travel-time objective.

Travel time of a continuous route is invariant under reparametrization, so the
continuous second variation is degenerate along the route. The discrete action
inherits this approximately: sliding waypoints along the route only changes the
quadrature (midpoint-rule) error, and a free-waypoint optimizer can exploit it,
collapsing waypoints. Local optimality of the route *shape* is therefore stated
for transverse deformations:

    x_n(s) = xbar_n + s_n * nu_n,   n = 2..199,   s in R^198,

with base points xbar_n and unit normals nu_n fixed binary64 data.
J(s) = J(X(s)); grad_s = N^T grad_X, Hess_s = N^T Hess_X N (X is affine in s).
"""
import numpy as np
from jets import IA, _as_ia
from action import evaluate, NFREE


def normals(curve):
    t = curve[2:] - curve[:-2]
    t = t / np.linalg.norm(t, axis=1, keepdims=True)
    return np.stack([-t[:, 1], t[:, 0]], 1)  # (198, 2)


def X_of_s(xbar, nu, s, interval=False):
    """xbar, nu: (198,2) float arrays; s: (198,) float or IA."""
    if interval:
        xs = _as_ia(xbar[:, 0]) + s * _as_ia(nu[:, 0])
        ys = _as_ia(xbar[:, 1]) + s * _as_ia(nu[:, 1])
        lo = np.stack([xs.lo, ys.lo], 1).ravel()
        hi = np.stack([xs.hi, ys.hi], 1).ravel()
        return IA(lo, hi)
    return (xbar + s[:, None] * nu).ravel()


def eval_family(problem, xbar, nu, s, interval=False, hessian=True):
    X = X_of_s(xbar, nu, s, interval)
    J, G, H, dm = evaluate(problem, X, interval=interval, hessian=hessian)
    nx, ny = nu[:, 0], nu[:, 1]
    if interval:
        Gs = G[0::2] * _as_ia(nx) + G[1::2] * _as_ia(ny)
        Hs = None
        if hessian:
            ax, ay = _as_ia(nx), _as_ia(ny)
            AX = IA(ax.lo[:, None], ax.hi[:, None]); AY = IA(ay.lo[:, None], ay.hi[:, None])
            BX = IA(ax.lo[None, :], ax.hi[None, :]); BY = IA(ay.lo[None, :], ay.hi[None, :])
            Hs = (H[0::2, 0::2] * AX * BX + H[0::2, 1::2] * AX * BY
                  + H[1::2, 0::2] * AY * BX + H[1::2, 1::2] * AY * BY)
        return J, Gs, Hs, dm
    Gs = G[0::2] * nx + G[1::2] * ny
    Hs = None
    if hessian:
        Hs = (H[0::2, 0::2] * np.outer(nx, nx) + H[0::2, 1::2] * np.outer(nx, ny)
              + H[1::2, 0::2] * np.outer(ny, nx) + H[1::2, 1::2] * np.outer(ny, ny))
    return J, Gs, Hs, dm
