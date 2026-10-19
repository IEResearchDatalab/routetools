"""
High-precision rigorous gradient of the discrete action (mpmath.iv, 150 bits):
first-order dual numbers per segment + adjoint of the time recursion.
Used (i) to polish the reference route to the limit of binary64 representability
and (ii) to obtain a tight rigorous upper bound on ||grad J(X~)||.
"""
import numpy as np
from mpmath import iv, mp
from problems import PROBLEMS, mp_interval, L_POINTS

PREC = 150


class D:
    """First-order dual number over mpmath iv intervals: value + gradient (k)."""
    __slots__ = ("v", "g")

    def __init__(self, v, g):
        self.v, self.g = v, g

    def _l(self, o):
        return o if isinstance(o, D) else D(iv.mpf(o) if not isinstance(o, iv.mpf) else o, [0] * len(self.g))

    def __add__(self, o):
        o = self._l(o)
        return D(self.v + o.v, [a + b for a, b in zip(self.g, o.g)])

    __radd__ = __add__

    def __neg__(self):
        return D(-self.v, [-a for a in self.g])

    def __sub__(self, o):
        return self + (-self._l(o))

    def __rsub__(self, o):
        return self._l(o) - self

    def __mul__(self, o):
        if not isinstance(o, D):
            c = o if isinstance(o, iv.mpf) else iv.mpf(o)
            return D(self.v * c, [a * c for a in self.g])
        return D(self.v * o.v, [a * o.v + self.v * b for a, b in zip(self.g, o.g)])

    __rmul__ = __mul__

    def sq(self):
        return self * self

    def recip(self):
        r = 1 / self.v
        return D(r, [-(r * r) * a for a in self.g])

    def sqrt(self):
        s = iv.sqrt(self.v)
        return D(s, [a / (2 * s) for a in self.g])

    def sin(self):
        return D(iv.sin(self.v), [iv.cos(self.v) * a for a in self.g])

    def cos(self):
        return D(iv.cos(self.v), [-iv.sin(self.v) * a for a in self.g])


class HPConsts:
    def __call__(self, s):
        return mp_interval(s)


def grad_hp(problem, X):
    """Rigorous 150-bit enclosure of (J, grad_X J) at the float point X (396,)."""
    iv.prec = PREC
    c = HPConsts()
    src = [mp_interval(s) for s in problem.src]
    dst = [mp_interval(s) for s in problem.dst]
    Xr = np.asarray(X, dtype=float).reshape(-1, 2)
    px = [src[0]] + [iv.mpf(float(v)) for v in Xr[:, 0]] + [dst[0]]
    py = [src[1]] + [iv.mpf(float(v)) for v in Xr[:, 1]] + [dst[1]]
    nseg = L_POINTS - 1
    t = iv.mpf(0)
    phis = []
    for n in range(nseg):
        e = lambda i: [1 if j == i else 0 for j in range(5)]
        args = [D(t, e(0)), D(px[n], e(1)), D(py[n], e(2)), D(px[n + 1], e(3)), D(py[n + 1], e(4))]
        phi, den = problem.phi(*args, c)
        if den is not None and not (den.v.a > 0):
            raise ArithmeticError("feasibility (S^2 - |w|^2 > 0) not verified")
        phis.append(phi)
        if problem.time_variant:
            t = t + phi.v
        else:
            t = t + phi.v  # accumulator of the cost (phi does not depend on t)
    J = t
    # adjoint of t_{n+1} = t_n + phi_n(t_n, z_n);  J = t_N
    G = [iv.mpf(0)] * (2 * (L_POINTS - 2))
    lam = iv.mpf(1)
    for n in range(nseg - 1, -1, -1):
        g = phis[n].g
        for a, p in enumerate((n, n + 1)):
            if 1 <= p <= L_POINTS - 2:
                G[2 * (p - 1)] += lam * g[1 + 2 * a]
                G[2 * (p - 1) + 1] += lam * g[2 + 2 * a]
        if problem.time_variant:
            lam = lam * (1 + g[0])
    return J, G


def family_grad_hp(problem, xbar, nu, s):
    """Rigorous enclosure of grad_s J at float s, with X = xbar + s*nu (enclosed)."""
    iv.prec = PREC
    # X(s) exactly: xbar + s*nu has an exact real value; compute it in iv and pass
    # the enclosure through (we rebuild the points with iv directly).
    X_iv = [[iv.mpf(float(xbar[i, k])) + iv.mpf(float(s[i])) * iv.mpf(float(nu[i, k]))
             for k in range(2)] for i in range(xbar.shape[0])]
    J, G = _grad_hp_points(problem, X_iv)
    Gs = [G[2 * i] * iv.mpf(float(nu[i, 0])) + G[2 * i + 1] * iv.mpf(float(nu[i, 1]))
          for i in range(xbar.shape[0])]
    return J, Gs


def _grad_hp_points(problem, X_iv):
    iv.prec = PREC
    c = HPConsts()
    src = [mp_interval(s) for s in problem.src]
    dst = [mp_interval(s) for s in problem.dst]
    px = [src[0]] + [p[0] for p in X_iv] + [dst[0]]
    py = [src[1]] + [p[1] for p in X_iv] + [dst[1]]
    nseg = L_POINTS - 1
    t = iv.mpf(0)
    phis = []
    for n in range(nseg):
        e = lambda i: [1 if j == i else 0 for j in range(5)]
        args = [D(t, e(0)), D(px[n], e(1)), D(py[n], e(2)), D(px[n + 1], e(3)), D(py[n + 1], e(4))]
        phi, den = problem.phi(*args, c)
        if den is not None and not (den.v.a > 0):
            raise ArithmeticError("feasibility not verified")
        phis.append(phi)
        t = t + phi.v
    G = [iv.mpf(0)] * (2 * (L_POINTS - 2))
    lam = iv.mpf(1)
    for n in range(nseg - 1, -1, -1):
        g = phis[n].g
        for a, p in enumerate((n, n + 1)):
            if 1 <= p <= L_POINTS - 2:
                G[2 * (p - 1)] += lam * g[1 + 2 * a]
                G[2 * (p - 1) + 1] += lam * g[2 + 2 * a]
        if problem.time_variant:
            lam = lam * (1 + g[0])
    return t, G


def norm_upper(G):
    iv.prec = PREC
    return iv.sqrt(iv.fsum([gi * gi for gi in G])).b


def mid_float(G):
    return np.array([float((gi.a + gi.b) / 2) for gi in G])
