"""Independent plain-numpy transcription of routetools/cost.py (value only),
used to cross-check the jet implementation."""
import numpy as np


def vf_circular(x, y, t, intensity=-0.9):
    return -intensity * y, intensity * x


def vf_doublegyre(x, y, t, amp=0.1, eps=0.25, w=1):
    a = eps * np.sin(w * t)
    b = 1 - 2 * eps * np.sin(w * t)
    f = a * x**2 + b * x
    dfdx = 2 * a * x + b
    u = -np.pi * amp * np.sin(np.pi * f) * np.cos(np.pi * y)
    v = np.pi * amp * np.cos(np.pi * f) * np.sin(np.pi * y) * dfdx
    return u, v


def _Ru(x, y, a, b):
    return 1 / (3 * ((x - a) ** 2 + (y - b) ** 2) + 1) * -(y - b)


def _Rv(x, y, a, b):
    return 1 / (3 * ((x - a) ** 2 + (y - b) ** 2) + 1) * (x - a)


def vf_fourvortices(x, y, t):
    u = 1.7 * (-_Ru(x, y, 2, 2) - _Ru(x, y, 4, 4) - _Ru(x, y, 2, 5) + _Ru(x, y, 5, 1))
    v = 1.7 * (-_Rv(x, y, 2, 2) - _Rv(x, y, 4, 4) - _Rv(x, y, 2, 5) + _Rv(x, y, 5, 1))
    return u, v


def vf_swirlys(x, y, t):
    return np.cos(2 * x - y - 6), 2 / 3 * np.sin(y) + x - 3


def vf_techy(x, y, t, sink=-0.3):
    vortex = t - 0.5
    return sink * x - vortex * y, vortex * x + sink * y


def cost_time_invariant(vf, curve, v2=1.0):
    mx = (curve[:-1, 0] + curve[1:, 0]) / 2
    my = (curve[:-1, 1] + curve[1:, 1]) / 2
    u, v = vf(mx, my, 0 * mx)
    dx, dy = np.diff(curve[:, 0]), np.diff(curve[:, 1])
    d2 = dx**2 + dy**2
    w2 = u**2 + v**2
    dw = dx * u + dy * v
    dt = np.sqrt(d2 / (v2 - w2) + dw**2 / (v2 - w2) ** 2) - dw / (v2 - w2)
    return dt.sum()


def cost_time_variant(vf, curve, v2=1.0):
    mx = (curve[:-1, 0] + curve[1:, 0]) / 2
    my = (curve[:-1, 1] + curve[1:, 1]) / 2
    dx, dy = np.diff(curve[:, 0]), np.diff(curve[:, 1])
    t = 0.0
    for i in range(len(dx)):
        u, v = vf(mx[i], my[i], t)
        w2 = u**2 + v**2
        dw = dx[i] * u + dy[i] * v
        d2 = dx[i] ** 2 + dy[i] ** 2
        dt = np.sqrt(d2 / (v2 - w2) + dw**2 / (v2 - w2) ** 2) - dw / (v2 - w2)
        t += dt
    return t


def cost_fixed_time(vf, curve, T=30.0):
    mx = (curve[:-1, 0] + curve[1:, 0]) / 2
    my = (curve[:-1, 1] + curve[1:, 1]) / 2
    u, v = vf(mx, my, np.array([0.0]))
    dx, dy = np.diff(curve[:, 0]), np.diff(curve[:, 1])
    dt = T / (curve.shape[0] - 1)
    return (((dx / dt - u) ** 2 + (dy / dt - v) ** 2) / 2 * dt).sum()


REF = {
    "circular": lambda c: cost_time_invariant(vf_circular, c),
    "fourvortices": lambda c: cost_time_invariant(vf_fourvortices, c),
    "doublegyre": lambda c: cost_time_variant(vf_doublegyre, c),
    "techy": lambda c: cost_time_variant(vf_techy, c),
    "swirlys": lambda c: cost_fixed_time(vf_swirlys, c),
}
