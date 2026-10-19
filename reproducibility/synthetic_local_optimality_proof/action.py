"""
Discrete action J(X) of a route, with exact gradient and Hessian with respect to
the 396 free waypoint coordinates X, in float64 or rigorous interval arithmetic.

Time-dependent travel-time problems couple all segments through the schedule
t_{n+1} = t_n + dt_n, so the Hessian is dense; it is propagated through that
recursion by the chain rule. Time-independent problems give a banded Hessian.
"""
import numpy as np
from jets import IA, Jet, _as_ia
from problems import K, L_POINTS, Consts

NFREE = L_POINTS - 2
NVAR = 2 * NFREE


def endpoints(problem, interval):
    c = Consts(interval)
    src = [c(s) for s in problem.src]
    dst = [c(s) for s in problem.dst]
    return src, dst


def _points(problem, X, interval):
    """Return per-point coordinate arrays px, py (length 200) and index map."""
    src, dst = endpoints(problem, interval)
    if interval:
        Xl, Xh = X.lo.reshape(NFREE, 2), X.hi.reshape(NFREE, 2)
        pxl = np.concatenate([[src[0].lo], Xl[:, 0], [dst[0].lo]])
        pxh = np.concatenate([[src[0].hi], Xh[:, 0], [dst[0].hi]])
        pyl = np.concatenate([[src[1].lo], Xl[:, 1], [dst[1].lo]])
        pyh = np.concatenate([[src[1].hi], Xh[:, 1], [dst[1].hi]])
        return IA(pxl, pxh), IA(pyl, pyh)
    Xr = np.asarray(X, dtype=float).reshape(NFREE, 2)
    px = np.concatenate([[src[0]], Xr[:, 0], [dst[0]]])
    py = np.concatenate([[src[1]], Xr[:, 1], [dst[1]]])
    return px, py


def _local_index(n):
    """Global variable indices for local vars (x0,y0,x1,y1) of segment n (-1 = fixed)."""
    out = []
    for p in (n, n + 1):
        for comp in (0, 1):
            out.append(2 * (p - 1) + comp if 1 <= p <= L_POINTS - 2 else -1)
    return out


def _seg_jets(problem, px, py, t, seg, interval):
    """Jets of phi for the segments in `seg` (array of indices)."""
    c = Consts(interval)
    if interval:
        vals = [t, px[seg], py[seg], px[seg + 1], py[seg + 1]]
    else:
        vals = [t, px[seg], py[seg], px[seg + 1], py[seg + 1]]
    jets = [Jet.variable(v, i, K, interval=interval) for i, v in enumerate(vals)]
    phi, den = problem.phi(*jets, c)
    return phi, den


def evaluate(problem, X, interval=False, hessian=True):
    """Return (J, grad, Hess, den_min).

    X: float array (396,) or IA of shape (396,).
    J: float or scalar IA; grad: (396,), Hess: (396, 396) (None if hessian=False).
    den_min: lower bound on S^2 - |w|^2 over all segments (None for energy cost).
    """
    px, py = _points(problem, X, interval)
    nseg = L_POINTS - 1
    zero = (lambda shp: IA.zeros(shp)) if interval else (lambda shp: np.zeros(shp))

    G = zero((NVAR,))
    H = zero((NVAR, NVAR)) if hessian else None
    den_min = np.inf

    if not problem.time_variant:
        seg = np.arange(nseg)
        t = _as_ia(np.zeros(nseg)) if interval else np.zeros(nseg)
        phi, den = _seg_jets(problem, px, py, t, seg, interval)
        if den is not None:
            den_min = float(np.min(den.v.lo)) if interval else float(np.min(den.v))
        # value
        if interval:
            J = _as_ia(0.0)
            for n in range(nseg):
                J = J + phi.v[n]
        else:
            J = float(np.sum(phi.v))
        for n in range(nseg):
            idx = _local_index(n)
            for a in range(4):
                ia = idx[a]
                if ia < 0:
                    continue
                G[ia] = G[ia] + phi.g[n, a + 1]
                if hessian:
                    for b in range(4):
                        ib = idx[b]
                        if ib < 0:
                            continue
                        H[ia, ib] = H[ia, ib] + phi.H[n, a + 1, b + 1]
        return J, G, H, (den_min if den is not None else None)

    # ---- time-dependent: sequential recursion over the schedule --------------
    T = _as_ia(np.zeros(1)) if interval else np.zeros(1)
    for n in range(nseg):
        seg = np.array([n])
        phi, den = _seg_jets(problem, px, py, T, seg, interval)
        if den is not None:
            dm = float(den.v.lo[0]) if interval else float(den.v[0])
            den_min = min(den_min, dm)
        idx = [i for i in _local_index(n)]
        loc = [a for a in range(4) if idx[a] >= 0]
        gidx = np.array([idx[a] for a in loc], dtype=int)

        ph_t = phi.g[0, 0]
        ph_z = phi.g[0, 1:5][np.array(loc, dtype=int)] if loc else None
        if hessian:
            ph_tt = phi.H[0, 0, 0]
            ph_tz = phi.H[0, 0, 1:5][np.array(loc, dtype=int)] if loc else None
            ph_zz = phi.H[0, 1:5, 1:5][np.ix_(loc, loc)] if loc else None

            # Hessian update:  H <- (1+phi_t) H + phi_tt G G^T + G c^T + c G^T + P phi_zz P^T
            Hn = H * (ph_t + 1.0)
            Gc = G if not interval else G
            outerGG = _outer(Gc, Gc, interval)
            Hn = Hn + outerGG * ph_tt
            if loc:
                col = _outer(G, ph_tz, interval)  # (NVAR, m)
                _add_cols(Hn, gidx, col, interval)
                _add_rows(Hn, gidx, _transpose(col, interval), interval)
                _add_block(Hn, gidx, ph_zz, interval)
            H = Hn
        # gradient update: G <- (1+phi_t) G + P phi_z
        Gn = G * (ph_t + 1.0)
        if loc:
            for j, gi in enumerate(gidx):
                Gn[gi] = Gn[gi] + ph_z[j]
        G = Gn
        T = T + phi.v
    J = T[0] if interval else float(T[0])
    return J, G, H, (den_min if np.isfinite(den_min) else None)


# ---- small helpers that work on both backends --------------------------------
def _outer(a, b, interval):
    if interval:
        return IA(a.lo[:, None], a.hi[:, None]) * IA(b.lo[None, :], b.hi[None, :])
    return np.outer(a, b)


def _transpose(M, interval):
    if interval:
        return IA(M.lo.T.copy(), M.hi.T.copy())
    return M.T


def _add_cols(H, gidx, col, interval):
    if interval:
        sub = H[:, gidx] + col
        H.lo[:, gidx] = sub.lo
        H.hi[:, gidx] = sub.hi
    else:
        H[:, gidx] += col


def _add_rows(H, gidx, row, interval):
    if interval:
        sub = H[gidx, :] + row
        H.lo[gidx, :] = sub.lo
        H.hi[gidx, :] = sub.hi
    else:
        H[gidx, :] += row


def _add_block(H, gidx, B, interval):
    ix = np.ix_(gidx, gidx)
    if interval:
        sub = IA(H.lo[ix], H.hi[ix]) + B
        H.lo[ix] = sub.lo
        H.hi[ix] = sub.hi
    else:
        H[ix] += B
