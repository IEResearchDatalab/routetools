"""
The five synthetic benchmarks, written to reproduce routetools/cost.py and
routetools/vectorfield.py (commit 8462156) exactly, but as generic functions of
jets so that they can be evaluated in float64 or in rigorous interval arithmetic.

Local variables of a segment: (t, x0, y0, x1, y1)  ->  k = 5.
"""
import re
import numpy as np
from jets import IA, Jet

K = 5
L_POINTS = 200


class Consts:
    """Exact real constants, enclosed in intervals when interval=True."""

    def __init__(self, interval):
        self.interval = interval

    def __call__(self, s):
        if self.interval:
            return IA.const(mp_interval(s))
        return float(eval(s, {"pi": np.pi, "cos": np.cos, "sin": np.sin}))


def mp_interval(s):
    """Rigorous mpmath.iv enclosure of an exact real written as an expression
    in decimal literals, pi, cos, sin, + - * /."""
    from mpmath import iv
    expr = re.sub(r"(\d+(?:\.\d+)?)", r"M('\1')", s)
    return iv.mpf(eval(expr, {"pi": iv.pi, "cos": iv.cos, "sin": iv.sin, "M": iv.mpf}))


# ---------------------------------------------------------------- fields
def field_circular(x, y, t, c):
    # intensity = -0.9 : u = -intensity*y, v = intensity*x
    return y * c("0.9"), x * c("-0.9")


def _R(x, y, a, b, c):
    dx, dy = x - a, y - b
    den = (dx.sq() + dy.sq()) * 3.0 + 1.0
    inv = den.recip()
    return -(dy * inv), dx * inv


def field_fourvortices(x, y, t, c):
    s = c("1.7")
    r1u, r1v = _R(x, y, 2.0, 2.0, c)
    r2u, r2v = _R(x, y, 4.0, 4.0, c)
    r3u, r3v = _R(x, y, 2.0, 5.0, c)
    r4u, r4v = _R(x, y, 5.0, 1.0, c)
    u = (-r1u - r2u - r3u + r4u) * s
    v = (-r1v - r2v - r3v + r4v) * s
    return u, v


def field_doublegyre(x, y, t, c):
    amp, eps, w = c("0.1"), c("0.25"), 1.0
    pi = c("pi")
    st = (t * w).sin()
    a = st * eps
    b = 1.0 - st * (eps * 2.0)
    f = a * x.sq() + b * x
    dfdx = a * x * 2.0 + b
    pa = pi * amp
    u = -((f * pi).sin() * (y * pi).cos()) * pa
    v = (f * pi).cos() * (y * pi).sin() * dfdx * pa
    return u, v


def field_techy(x, y, t, c):
    sink = c("-0.3")
    vortex = t - 0.5
    u = x * sink - vortex * y
    v = vortex * x + y * sink
    return u, v


def field_swirlys(x, y, t, c):
    u = (x * 2.0 - y - 6.0).cos()
    v = y.sin() * c("2/3") + x - 3.0
    return u, v


# ---------------------------------------------------------------- segment costs
def seg_time(field, S=1.0):
    """Travel time of a segment at constant speed S through water (cost.py)."""

    def phi(t, x0, y0, x1, y1, c):
        dx, dy = x1 - x0, y1 - y0
        mx, my = (x0 + x1) * 0.5, (y0 + y1) * 0.5
        u, v = field(mx, my, t, c)
        w2 = u.sq() + v.sq()
        dw = dx * u + dy * v
        d2 = dx.sq() + dy.sq()
        den = (w2 * -1.0) + S * S  # S^2 - |w|^2 (must be > 0)
        root = (dw.sq() + den * d2).sqrt()
        # (root - dw)/den == d2/(root + dw): identical function, no cancellation
        dt = d2 * (root + dw).recip()
        return dt, den

    return phi


def seg_energy(field, h):
    """Fixed-time energy cost of a segment: 0.5*|d/h - w(mid)|^2 * h (cost.py)."""

    def phi(t, x0, y0, x1, y1, c):
        dx, dy = x1 - x0, y1 - y0
        mx, my = (x0 + x1) * 0.5, (y0 + y1) * 0.5
        u, v = field(mx, my, t, c)
        hh = c(h) if isinstance(h, str) else h
        inv_h = hh.recip() if isinstance(hh, IA) else 1.0 / hh
        ex = dx * inv_h - u
        ey = dy * inv_h - v
        cost = (ex.sq() + ey.sq()) * (hh * 0.5)
        return cost, None

    return phi


class Problem:
    def __init__(self, name, phi, src, dst, time_variant, ref_cost):
        self.name, self.phi = name, phi
        self.src, self.dst = src, dst  # decimal strings (exact reals)
        self.time_variant = time_variant
        self.ref_cost = ref_cost


PROBLEMS = {
    "circular": Problem("circular", seg_time(field_circular),
                        ("cos(pi/6)", "sin(pi/6)"), ("0", "1"), False, 1.9750),
    "fourvortices": Problem("fourvortices", seg_time(field_fourvortices),
                            ("0", "0"), ("6", "2"), False, 8.9495),
    "doublegyre": Problem("doublegyre", seg_time(field_doublegyre),
                          ("1.5", "0.5"), ("0.5", "0.5"), True, 0.9888),
    "techy": Problem("techy", seg_time(field_techy),
                     ("cos(pi/6)", "sin(pi/6)"), ("0", "1"), True, 1.0317),
    # h = T/(L-1) = 30/199 ; cost uses the field at t = 0 (time-invariant field)
    "swirlys": Problem("swirlys", seg_energy(field_swirlys, "30/199"),
                       ("0", "0"), ("6", "5"), False, 1.9702),
}
