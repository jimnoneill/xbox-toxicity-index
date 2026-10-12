"""Tests for xbox.forecast_p2 (P2 forecasting). Run: pytest tests/test_p2.py"""
import numpy as np
import pandas as pd
from scipy.stats import nbinom

from xbox import forecast_p2 as F


def test_nb_logpmf_matches_scipy():
    y = np.arange(0, 30).astype(float)
    mu, alpha = 3.7, 0.8
    r = 1 / alpha
    ref = nbinom.logpmf(y, r, r / (r + mu))
    assert np.allclose(F.nb_logpmf(y, np.full_like(y, mu), alpha), ref)


def test_crps_matches_numeric():
    mu, alpha = np.array([2.5, 0.2]), np.array([0.5, 2.0])
    y = np.array([4.0, 0.0])
    got = F.nb_crps(y, mu, alpha)
    for i in range(2):
        r = 1 / alpha[i]
        k = np.arange(0, 3000)
        Fk = nbinom.cdf(k, r, r / (r + mu[i]))
        assert abs(got[i] - ((Fk - (k >= y[i])) ** 2).sum()) < 1e-6


def test_pit_uniform_under_truth():
    rng = np.random.default_rng(1)
    mu = rng.uniform(0.2, 5, 20000)
    alpha = 0.7
    r = 1 / alpha
    y = rng.negative_binomial(r, r / (r + mu)).astype(float)
    p = F.nb_pit(y, mu, np.full_like(mu, alpha), rng.random(len(mu)))
    h = np.histogram(p, bins=10, range=(0, 1))[0] / len(p)
    assert np.abs(h - 0.1).max() < 0.012


def test_fit_nb_recovers_coefficients():
    rng = np.random.default_rng(2)
    n = 60000
    X = np.column_stack([np.ones(n), rng.normal(size=n)])
    off = rng.normal(size=n) * 0.3
    mu = np.exp(off + X @ np.array([-1.0, 0.5]))
    alpha = 0.6
    y = rng.negative_binomial(1 / alpha, (1 / alpha) / (1 / alpha + mu)).astype(float)
    b, a = F.fit_nb(X, y, off, ridge=1e-6, iters=3)
    assert np.allclose(b, [-1.0, 0.5], atol=0.03) and abs(a - alpha) < 0.06


def test_fit_logit():
    rng = np.random.default_rng(3)
    X = np.column_stack([np.ones(40000), rng.normal(size=40000)])
    y = (rng.random(40000) < 1 / (1 + np.exp(-(X @ np.array([-2.0, 1.0]))))).astype(float)
    assert np.allclose(F.fit_logit(X, y, ridge=1e-8), [-2.0, 1.0], atol=0.06)


def test_et_offset_dst():
    u = pd.DatetimeIndex(["2022-03-13 06:59", "2022-03-13 07:00", "2022-11-06 05:59", "2022-11-06 06:00", "2022-01-15 12:00"])
    assert list(F.et_offset_hours(u)) == [-5, -4, -4, -5, -5]


def test_ew_level_is_causal():
    """Changing future data must not change the level at an earlier hour."""
    class Pn:
        pass
    p = Pn()
    rng = np.random.default_rng(4)
    p.Y = rng.poisson(0.3, (3, 2000)).astype(np.float32)
    p.Y0 = p.Y.copy()
    a = F.ew_levels(p, 100)
    p.Y0[:, 1500:] += 50
    p.Y[:, 1500:] += 50
    b = F.ew_levels(p, 100)
    assert np.allclose(a[:, :1500], b[:, :1500])


def test_hawkes_forecast_matches_direct_recursion():
    """Closed-form multi-step state recursion equals a brute-force expected-count recursion of the linear process."""
    th = F.HK_INIT.copy()
    zeta, a, b, g, rho, eta, ks, alpha, w = F.hk_unpack(th)
    rng = np.random.default_rng(5)
    T = 700

    class Pn:
        pass
    p = Pn()
    p.T = T
    p.Y0 = np.zeros((1, T))
    p.Y0[0, :400] = rng.poisson(0.5, 400)
    p.P = p.Y0 + rng.poisson(0.2, (1, T))
    p.C = rng.poisson(80, T).astype(float) + p.Y0[0]
    p.how = np.arange(T) % 168
    t = np.array([399])
    prof = np.ones((1, 168))
    cprof = np.ones(168)
    Ls = np.full((1, T), 0.1)
    Lc = np.full(T, 70.0)
    phi = 0.6
    H = 20
    lam = F.hk_forecast(th, p, np.array([0]), t, Ls, Ls, Lc, prof, cprof, phi, np.zeros(1), np.array([0.3]), H=H)
    z = F.cong_z(p.C[:400], p.Y0[0, :400], Lc[:400] * 0 + 70.0, Ls[0, :400], np.ones(400), np.ones(400))
    dev = z[-3:].mean()
    m = list(p.Y0[0, :400] + eta * (p.P[0, :400] - p.Y0[0, :400]))
    zz = list(z)
    out = []
    for k in range(H):
        n = len(m)
        own = sum(a[i] * sum(np.exp(-b[i] * (n - 1 - j)) * m[j] for j in range(max(0, n - 336), n)) for i in range(2))
        cg = sum(g[i] * sum(np.exp(-rho[i] * (len(zz) - 1 - j)) * zz[j] for j in range(max(0, len(zz) - 336), len(zz))) for i in range(2))
        l = max(0.1 * max(zeta + cg, 0.02) + own, 1e-8)
        out.append(l)
        m.append(l * (1 + eta * 0.3))
        zz.append(dev * phi ** (k + 1))
    assert np.allclose(lam[0], out, rtol=1e-6)


def test_hawkes_fit_runs_and_branching_bounded():
    rng = np.random.default_rng(6)

    class Pn:
        pass
    p = Pn()
    A, T = 4, 3000
    p.Y = rng.poisson(0.4, (A, T)).astype(np.float32)
    p.P = p.Y + rng.poisson(0.1, (A, T)).astype(np.float32)
    p.C = p.Y.sum(0) + rng.poisson(50, T)
    p.how = np.arange(T) % 168
    L = np.full((A, T), 0.4, np.float32)
    prof = np.ones((A, 168))
    Lc = np.full(T, 60.0)
    th, f = F.hk_fit(p, np.arange(A), 5, T, L, L, Lc, prof, np.ones(168), np.zeros(A), maxiter=8)
    assert np.isfinite(f) and F.hk_branching(th) < 1.2
