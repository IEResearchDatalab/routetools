"""
Computer-assisted proof of strict local minimality (see SPEC, Section 4).

Lemma. If Hess J(X) >= m I (m > 0) for every X in the closed ball B(X~, R) and
||grad J(X~)||_2 < m R / 2, then J has a unique minimizer X* in B, it is interior,
grad J(X*) = 0 and Hess J(X*) >= m I; hence X* is a strict local minimizer and
J(X*) in [J(X~) - ||grad J(X~)|| R, J(X~)].

Every quantity entering the check is computed rigorously:
  * grad J(X~) and Hess J over the box ||X - X~||_inf <= R (which contains the
    ball) by interval second-order AD (jets.py / action.py / family.py);
  * a lower bound on the smallest eigenvalue over the interval Hessian via a
    floating-point Cholesky factorization of (mid - s I) with a priori rounding
    bounds (Higham, Thm 3.5) and the radius matrix;
  * scalar bookkeeping in mpmath.iv.

Usage: python certify.py <problem> full|family
"""
import sys
import json
import time
import numpy as np
from mpmath import iv
from jets import IA
from problems import PROBLEMS
from action import evaluate, NFREE
from family import eval_family
from hp import _grad_hp_points, family_grad_hp, norm_upper

U = 2.0 ** -53  # unit roundoff, binary64


def ivf(x):
    return iv.mpf(float(x))


def up(x):
    return float(iv.mpf(x).b) if not isinstance(x, float) else x


def lambda_min_lower_bound(Hlo, Hhi, verbose=True):
    """Rigorous lower bound on min eigenvalue of every symmetric H with Hlo <= H <= Hhi."""
    n = Hlo.shape[0]
    lo = np.maximum(Hlo, Hlo.T)  # true Hessian is symmetric: intersect (i,j),(j,i)
    hi = np.minimum(Hhi, Hhi.T)
    if np.any(lo > hi):
        raise ArithmeticError("empty symmetric intersection")
    C = 0.5 * lo + 0.5 * hi
    C = 0.5 * (C + C.T)  # exactly symmetric float matrix (data)
    Delta = np.nextafter(np.maximum(hi - C, C - lo), np.inf)
    Delta = np.maximum(Delta, Delta.T)
    normDelta = ivf(np.max(Delta.sum(1))) * ivf(1 + 1e-12)  # ||D||_2 <= ||D||_inf

    lam_est = np.linalg.eigvalsh(C)[0]
    if lam_est <= 0:
        return None, {"lam_est_center": float(lam_est)}
    s = 0.5 * lam_est
    for _ in range(20):
        A = C - s * np.eye(n)
        try:
            Lc = np.linalg.cholesky(A)
            break
        except np.linalg.LinAlgError:
            s *= 0.5
    else:
        return None, {"lam_est_center": float(lam_est)}
    gam = n * U / (1 - n * U)
    P = Lc @ Lc.T
    Efl = A - P
    absLL = np.abs(Lc) @ np.abs(Lc).T
    Bm = np.abs(Efl) * (1 + 2 * U) + gam * absLL * 1.001
    normE = ivf(np.max(Bm.sum(1))) * ivf(1 + 1e-12)
    normD1 = ivf(U * np.max(np.abs(np.diag(A)))) * ivf(1 + 1e-12)
    m = ivf(s) - normE - normD1 - normDelta
    info = {"lam_est_center": float(lam_est), "shift_s": float(s), "normE": float(normE.b),
            "normD1": float(normD1.b), "normDelta": float(normDelta.b), "m_lower": float(m.a)}
    return m.a, info


def certify(name, mode, center, R=None, family_data=None):
    p = PROBLEMS[name]
    t0 = time.time()
    if mode == "full":
        def ev(Z, interval, hessian=True):
            return evaluate(p, Z, interval=interval, hessian=hessian)
    else:
        xbar, nu = family_data

        def ev(Z, interval, hessian=True):
            return eval_family(p, xbar, nu, Z, interval=interval, hessian=hessian)

    # 1. rigorous gradient and value at the centre (150-bit interval arithmetic)
    if mode == "full":
        Jhp, Ghp = _grad_hp_points(p, [[iv.mpf(float(center[2 * i])), iv.mpf(float(center[2 * i + 1]))]
                                       for i in range(center.size // 2)])
    else:
        Jhp, Ghp = family_grad_hp(p, family_data[0], family_data[1], center)
    g_up = norm_upper(Ghp)
    iv.prec = 80
    Jc = IA(np.nextafter(float(Jhp.a), -np.inf), np.nextafter(float(Jhp.b), np.inf))
    # float estimate of curvature to choose R
    Hf = ev(center, False)[2]
    lam_f = np.linalg.eigvalsh(Hf)[0]
    if R is None:
        R = float(10 * float(g_up) / lam_f) if lam_f > 0 else 1e-10
    # 2. rigorous Hessian over the box
    Zb = IA(np.nextafter(center - R, -np.inf), np.nextafter(center + R, np.inf))
    Jb, Gb, Hb, den_b = ev(Zb, True, hessian=True)
    # 3. rigorous eigenvalue lower bound
    m_lo, info = lambda_min_lower_bound(Hb.lo, Hb.hi)
    # 4. check the lemma
    ok = False
    margin = None
    if m_lo is not None and m_lo > 0:
        rhs = ((iv.mpf(m_lo) * iv.mpf(R)) / 2).a  # lower bound of m R / 2
        ok = bool(iv.mpf(g_up).b < rhs)
        margin = float(rhs) / float(iv.mpf(g_up).b)
    Jstar_lo = (iv.mpf(float(Jc.lo)) - iv.mpf(g_up) * iv.mpf(R)).a
    res = {
        "problem": name, "mode": mode, "n_vars": int(center.size),
        "certified": ok,
        "margin_mR2_over_g": (margin if (m_lo is not None and m_lo > 0) else None),
        "J_center": [float(Jc.lo), float(Jc.hi)],
        "J_star_enclosure": [float(Jstar_lo), float(Jc.hi)],
        "grad_norm_upper": float(g_up),
        "R": R,
        "m_lower": None if m_lo is None else float(m_lo),
        "float_lam_min_center": float(lam_f),
        "den_min_lower_on_box": den_b,
        "hessian_box_max_width": float(np.max(Hb.width())),
        "seconds": round(time.time() - t0, 1),
        **{"eig_" + k: v for k, v in info.items()},
    }
    return res


if __name__ == "__main__":
    name, mode = sys.argv[1], sys.argv[2]
    if mode == "full":
        curve = np.load(f"routes/{name}.npy")
        center = curve[1:-1].ravel()
        fam = None
    else:
        d = np.load(f"routes/{name}_family.npz")
        center, fam = d["s"], (d["xbar"], d["nu"])
    R = float(sys.argv[3]) if len(sys.argv) > 3 else None
    res = certify(name, mode, center, R=R, family_data=fam)
    for k, v in res.items():
        print(f"  {k}: {v}")
    import os
    os.makedirs("certificates", exist_ok=True)
    json.dump(res, open(f"certificates/{name}_{mode}.json", "w"), indent=1)
