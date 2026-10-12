"""T4b additions (review fixes): natural cubic spline basis, simulator with observed initial history, prediction-interval helpers.
Interpretation: associations only."""
from __future__ import annotations

import numpy as np
from numba import njit


def ns_basis(x, df):
    """Natural cubic spline basis (Hastie, Tibshirani, Friedman 5.2.1) with df columns (no intercept), columns standardised.
    Knots: min, max and df-1 interior quantiles of x."""
    x = np.asarray(x, dtype=float)
    K = df + 1
    kn = np.quantile(x, np.linspace(0, 1, K))
    kn = np.unique(kn)
    K = len(kn)
    if K < 3:
        return np.zeros((len(x), 0))
    def d(k):
        return (np.maximum(x - kn[k], 0) ** 3 - np.maximum(x - kn[K - 1], 0) ** 3) / (kn[K - 1] - kn[k])
    cols = [x - kn[0]]
    dK1 = d(K - 2)
    for k in range(K - 2):
        cols.append(d(k) - dK1)
    B = np.column_stack(cols)
    B = B / np.maximum(B.std(axis=0), 1e-12)
    return B


@njit(cache=True)
def _sim2(eta0, phi1, phiL, alpha, L, seed, y0, n0):
    np.random.seed(seed)
    T = len(eta0)
    y = np.zeros(T)
    for t in range(min(n0, T)):
        y[t] = y0[t]
    for t in range(n0, T):
        s = 0.0
        for j in range(2, L + 1):
            if t - j >= 0:
                s += y[t - j]
        if eta0[t] <= -30.0:
            y[t] = 0.0
            continue
        lag1 = np.log1p(y[t - 1]) if t >= 1 else 0.0
        eta = eta0[t] + phi1 * lag1 + phiL * np.log1p(s)
        if eta > 9.0:
            eta = 9.0
        mu = np.exp(eta)
        if alpha < 1e-6:
            y[t] = np.random.poisson(mu)
        else:
            lam = np.random.gamma(1.0 / alpha, alpha * mu)
            y[t] = np.random.poisson(lam)
    return y


def simulate_nb2(eta0, phi1, phiL, alpha, seed, y0=None, n0=0, L=24):
    y0 = np.zeros(1) if y0 is None else np.asarray(y0, dtype=float)
    return _sim2(np.asarray(eta0, dtype=float), float(phi1), float(phiL), float(alpha), int(L), int(seed), y0, int(n0))
