"""Stage 1 of BERS, re-implemented for the certificate: CMA-ES over Bezier
control points (degree K, endpoints fixed, L = 200 samples), batched numpy costs.
The CMA-ES here is a standard (mu/mu_w, lambda) implementation (Hansen 2016)."""
import numpy as np
from math import comb
from reference_numpy import (vf_circular, vf_fourvortices, vf_doublegyre,
                             vf_techy, vf_swirlys)

L = 200


def bernstein(K, L=L):
    r = np.linspace(0, 1, L)
    return np.stack([comb(K, k) * (1 - r) ** (K - k) * r ** k for k in range(K + 1)], 1)


def batch_cost_time(vf, curves, time_variant, v2=1.0):
    mx = 0.5 * (curves[:, :-1, 0] + curves[:, 1:, 0])
    my = 0.5 * (curves[:, :-1, 1] + curves[:, 1:, 1])
    dx = np.diff(curves[:, :, 0], axis=1)
    dy = np.diff(curves[:, :, 1], axis=1)
    d2 = dx ** 2 + dy ** 2
    if not time_variant:
        u, v = vf(mx, my, 0 * mx)
        w2 = u ** 2 + v ** 2
        dw = dx * u + dy * v
        den = v2 - w2
        with np.errstate(all="ignore"):
            dt = d2 / (np.sqrt(dw ** 2 + den * d2) + dw)
        dt = np.where(den <= 0, 1e10, dt)
        return np.nan_to_num(dt.sum(1), nan=1e10)
    t = np.zeros(curves.shape[0])
    for i in range(dx.shape[1]):
        u, v = vf(mx[:, i], my[:, i], t)
        w2 = u ** 2 + v ** 2
        dw = dx[:, i] * u + dy[:, i] * v
        den = v2 - w2
        with np.errstate(all="ignore"):
            dt = d2[:, i] / (np.sqrt(dw ** 2 + den * d2[:, i]) + dw)
        dt = np.where(den <= 0, 1e10, dt)
        t = t + dt
    return np.nan_to_num(t, nan=1e10)


def batch_cost_fixed(vf, curves, T=30.0):
    mx = 0.5 * (curves[:, :-1, 0] + curves[:, 1:, 0])
    my = 0.5 * (curves[:, :-1, 1] + curves[:, 1:, 1])
    u, v = vf(mx, my, 0 * mx)
    dx = np.diff(curves[:, :, 0], axis=1)
    dy = np.diff(curves[:, :, 1], axis=1)
    h = T / (curves.shape[1] - 1)
    return (((dx / h - u) ** 2 + (dy / h - v) ** 2) / 2 * h).sum(1)


SETUP = {
    "circular": (lambda c: batch_cost_time(vf_circular, c, False), (np.cos(np.pi / 6), 0.5), (0.0, 1.0)),
    "fourvortices": (lambda c: batch_cost_time(vf_fourvortices, c, False), (0.0, 0.0), (6.0, 2.0)),
    "doublegyre": (lambda c: batch_cost_time(vf_doublegyre, c, True), (1.5, 0.5), (0.5, 0.5)),
    "techy": (lambda c: batch_cost_time(vf_techy, c, True), (np.cos(np.pi / 6), 0.5), (0.0, 1.0)),
    "swirlys": (lambda c: batch_cost_fixed(vf_swirlys, c), (0.0, 0.0), (6.0, 5.0)),
}


def cmaes(f, x0, sigma0, lam=500, tolfun=1e-3, maxgen=2000, seed=0, patience=10):
    rng = np.random.default_rng(seed)
    n = x0.size
    mu = lam // 2
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    w /= w.sum()
    mueff = 1 / (w ** 2).sum()
    cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
    cs = (mueff + 2) / (n + mueff + 5)
    c1 = 2 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
    chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
    m, sigma = x0.copy(), sigma0
    C = np.eye(n)
    pc, ps = np.zeros(n), np.zeros(n)
    best = (np.inf, None)
    hist = []
    for gen in range(maxgen):
        D2, B = np.linalg.eigh(C)
        D = np.sqrt(np.maximum(D2, 1e-30))
        z = rng.standard_normal((lam, n))
        y = z * D @ B.T
        X = m + sigma * y
        fx = f(X)
        idx = np.argsort(fx)
        if fx[idx[0]] < best[0]:
            best = (fx[idx[0]], X[idx[0]].copy())
        hist.append(fx[idx[0]])
        yw = w @ y[idx[:mu]]
        m = m + sigma * yw
        Cinvsqrt = B @ np.diag(1 / D) @ B.T
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * Cinvsqrt @ yw
        hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) / chiN < 1.4 + 2 / (n + 1)
        pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * yw
        yk = y[idx[:mu]]
        C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) \
            + cmu * (yk.T * w) @ yk
        sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
        if gen > patience and abs(hist[-patience] - hist[-1]) < tolfun * 1e-3:
            break
    return best, gen + 1


def bers_stage1(name, K=9, sigma0=2.0, seed=0):
    """K = number of FREE control points (routetools convention): degree K+1.
    sigma0 is scaled by half the source-destination distance (routetools)."""
    f_cost, src, dst = SETUP[name]
    src, dst = np.array(src), np.array(dst)
    Bm = bernstein(K + 1)
    ctrl0 = np.linspace(src, dst, K + 2)[1:-1].ravel()
    sigma0 = sigma0 * np.linalg.norm(dst - src) / 2

    def f(P):
        ctrl = P.reshape(-1, K, 2)
        full = np.concatenate([np.broadcast_to(src, (P.shape[0], 1, 2)), ctrl,
                               np.broadcast_to(dst, (P.shape[0], 1, 2))], 1)
        curves = np.einsum("lk,bkd->bld", Bm, full)
        return f_cost(curves)

    (fbest, pbest), ngen = cmaes(f, ctrl0, sigma0, seed=seed)
    ctrl = pbest.reshape(K, 2)
    full = np.vstack([src, ctrl, dst])
    return Bm @ full, fbest, ngen
