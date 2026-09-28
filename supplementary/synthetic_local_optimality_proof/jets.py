"""
Rigorous interval arithmetic (IA) on numpy arrays + second-order forward-mode
automatic differentiation ("jets") that runs on either float64 or IA.

Rigour model for IA:
  * +, -, *, /, sqrt are IEEE-754 binary64 operations with round-to-nearest,
    so the exact result lies within 1/2 ulp of the computed one. After every
    operation the lower endpoint is moved one ulp down and the upper endpoint
    one ulp up (np.nextafter), which therefore encloses the exact result.
  * sin, cos, and constants such as pi, 2/3, 1.7 are enclosed with
    mpmath.iv (rigorous multiprecision interval arithmetic), then converted to
    binary64 with an extra outward ulp.
"""
import numpy as np
import mpmath
from mpmath import iv

iv.prec = 80
_NINF, _PINF = -np.inf, np.inf


def _dn(x):
    return np.nextafter(x, _NINF)


def _up(x):
    return np.nextafter(x, _PINF)


class IA:
    """Array of closed intervals [lo, hi] with outward rounding."""

    __array_priority__ = 1000

    def __init__(self, lo, hi=None):
        lo = np.asarray(lo, dtype=np.float64)
        hi = lo if hi is None else np.asarray(hi, dtype=np.float64)
        self.lo, self.hi = lo, hi

    # ---- construction helpers ------------------------------------------
    @staticmethod
    def const(s):
        """Enclosure of an exact real given as a decimal string or mpmath iv."""
        x = s if isinstance(s, iv.mpf) else iv.mpf(s)
        return IA(_dn(float(x.a)), _up(float(x.b)))

    @staticmethod
    def zeros(shape):
        return IA(np.zeros(shape), np.zeros(shape))

    @property
    def shape(self):
        return self.lo.shape

    def __getitem__(self, k):
        return IA(self.lo[k], self.hi[k])

    def __setitem__(self, k, v):
        v = _as_ia(v)
        self.lo[k] = v.lo
        self.hi[k] = v.hi

    def copy(self):
        return IA(self.lo.copy(), self.hi.copy())

    def mid(self):
        return 0.5 * self.lo + 0.5 * self.hi

    def mag(self):
        return np.maximum(np.abs(self.lo), np.abs(self.hi))

    def width(self):
        return self.hi - self.lo

    # ---- arithmetic ----------------------------------------------------
    def __add__(self, o):
        o = _as_ia(o)
        return IA(_dn(self.lo + o.lo), _up(self.hi + o.hi))

    __radd__ = __add__

    def __neg__(self):
        return IA(-self.hi, -self.lo)

    def __sub__(self, o):
        o = _as_ia(o)
        return IA(_dn(self.lo - o.hi), _up(self.hi - o.lo))

    def __rsub__(self, o):
        return _as_ia(o) - self

    def __mul__(self, o):
        o = _as_ia(o)
        p1, p2 = self.lo * o.lo, self.lo * o.hi
        p3, p4 = self.hi * o.lo, self.hi * o.hi
        lo = np.minimum(np.minimum(p1, p2), np.minimum(p3, p4))
        hi = np.maximum(np.maximum(p1, p2), np.maximum(p3, p4))
        return IA(_dn(lo), _up(hi))

    __rmul__ = __mul__

    def recip(self):
        if np.any((self.lo <= 0) & (self.hi >= 0)):
            raise ArithmeticError("interval reciprocal of an interval containing 0")
        return IA(_dn(1.0 / self.hi), _up(1.0 / self.lo))

    def __truediv__(self, o):
        return self * _as_ia(o).recip()

    def __rtruediv__(self, o):
        return _as_ia(o) * self.recip()

    def sqrt(self):
        if np.any(self.lo < 0):
            raise ArithmeticError("interval sqrt of a possibly negative interval")
        return IA(np.maximum(_dn(np.sqrt(self.lo)), 0.0), _up(np.sqrt(self.hi)))

    def _mp_unary(self, f):
        lo = np.empty_like(self.lo)
        hi = np.empty_like(self.hi)
        for idx in np.ndindex(self.lo.shape):
            r = f(iv.mpf([float(self.lo[idx]), float(self.hi[idx])]))
            lo[idx] = _dn(float(r.a))
            hi[idx] = _up(float(r.b))
        return IA(lo, hi)

    def sin(self):
        return self._mp_unary(iv.sin)

    def cos(self):
        return self._mp_unary(iv.cos)

    def __repr__(self):
        return f"IA(lo={self.lo}, hi={self.hi})"


def _as_ia(x):
    if isinstance(x, IA):
        return x
    x = np.asarray(x, dtype=np.float64)  # exactly representable binary64 values
    return IA(x, x)


# ---- backend-dispatching elementary functions -----------------------------
def b_sqrt(x):
    return x.sqrt() if isinstance(x, IA) else np.sqrt(x)


def b_sin(x):
    return x.sin() if isinstance(x, IA) else np.sin(x)


def b_cos(x):
    return x.cos() if isinstance(x, IA) else np.cos(x)


def b_recip(x):
    return x.recip() if isinstance(x, IA) else 1.0 / x


def expand(x, axes):
    """Insert singleton axes (for broadcasting) on either backend."""
    if isinstance(x, IA):
        return IA(np.expand_dims(x.lo, axes), np.expand_dims(x.hi, axes))
    return np.expand_dims(x, axes)


def transpose_last2(x):
    if isinstance(x, IA):
        return IA(np.swapaxes(x.lo, -1, -2), np.swapaxes(x.hi, -1, -2))
    return np.swapaxes(x, -1, -2)


# ---- second-order jets ----------------------------------------------------
class Jet:
    """Value, gradient and Hessian with respect to k local variables.

    v: shape (S,), g: shape (S, k), H: shape (S, k, k); entries are floats or IA.
    """

    def __init__(self, v, g, H):
        self.v, self.g, self.H = v, g, H

    @staticmethod
    def variable(values, i, k, interval=False):
        S = values.shape[0]
        g = np.zeros((S, k))
        g[:, i] = 1.0
        H = np.zeros((S, k, k))
        if interval:
            return Jet(values, _as_ia(g), _as_ia(H))
        return Jet(values, g, H)

    @staticmethod
    def constant(c, like):
        k = like.g.shape[-1]
        S = like.g.shape[0]
        if isinstance(like.v, IA):
            cv = c if isinstance(c, IA) else _as_ia(np.full(S, c))
            if cv.shape == ():
                cv = IA(np.full(S, cv.lo), np.full(S, cv.hi))
            return Jet(cv, IA.zeros((S, k)), IA.zeros((S, k, k)))
        return Jet(np.full(S, c, dtype=float), np.zeros((S, k)), np.zeros((S, k, k)))

    def _lift(self, o):
        return o if isinstance(o, Jet) else Jet.constant(o, self)

    def __add__(self, o):
        o = self._lift(o)
        return Jet(self.v + o.v, self.g + o.g, self.H + o.H)

    __radd__ = __add__

    def __neg__(self):
        return Jet(-self.v, -self.g, -self.H)

    def __sub__(self, o):
        o = self._lift(o)
        return Jet(self.v - o.v, self.g - o.g, self.H - o.H)

    def __rsub__(self, o):
        return self._lift(o) - self

    def __mul__(self, o):
        if not isinstance(o, Jet):
            # scalar constant (python float or scalar IA), no derivatives
            return Jet(self.v * o, self.g * o, self.H * o)
        a, b = self, o
        av, bv = expand(a.v, 1), expand(b.v, 1)
        avH, bvH = expand(a.v, (1, 2)), expand(b.v, (1, 2))
        v = a.v * b.v
        g = av * b.g + bv * a.g
        outer = expand(a.g, 2) * expand(b.g, 1)
        H = avH * b.H + bvH * a.H + outer + transpose_last2(outer)
        return Jet(v, g, H)

    __rmul__ = __mul__

    def _unary(self, f0, f1, f2):
        g = expand(f1, 1) * self.g
        H = expand(f2, (1, 2)) * (expand(self.g, 2) * expand(self.g, 1)) + expand(f1, (1, 2)) * self.H
        return Jet(f0, g, H)

    def sqrt(self):
        s = b_sqrt(self.v)
        r = b_recip(s)
        f1 = 0.5 * r
        f2 = -0.25 * (r * b_recip(self.v))
        return self._unary(s, f1, f2)

    def recip(self):
        r = b_recip(self.v)
        return self._unary(r, -(r * r), 2.0 * (r * r * r))

    def __truediv__(self, o):
        if isinstance(o, Jet):
            return self * o.recip()
        return self * b_recip(_as_ia(o) if isinstance(self.v, IA) else o)

    def sin(self):
        s, c = b_sin(self.v), b_cos(self.v)
        return self._unary(s, c, -s)

    def cos(self):
        s, c = b_sin(self.v), b_cos(self.v)
        return self._unary(c, -s, -c)

    def sq(self):
        return self * self
