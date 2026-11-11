"""Hawkes and Poisson models whose background rate follows the activity of the rest of Congress.

    lambda(t) = L[m(t)] * P[k(t)] * exp(gamma * z(t)) + eta * beta * sum_{t_i < t} exp(-beta (t - t_i))

m(t) is the calendar month, k(t) one of 48 local hour-of-week bins, and z(t) is the standardized logarithm of one
plus the number of events started by all OTHER members in the same one-hour slot. gamma = 0 gives the exponential
Hawkes model of xbox.pointprocess_models (hawkes_exp); eta = 0 gives an inhomogeneous Poisson model with the shared
driver. gamma is estimated by profile likelihood (bounded scalar search around an EM fit of the other parameters).
Log-likelihood, AIC, BIC and the rescaled intervals follow the conventions of xbox.pointprocess_models (conditional
on the first event; k counts nm + 47 baseline parameters, the kernel parameters, and gamma).
"""
import numpy as np
from numba import njit
from scipy import optimize

from xbox import hawkes as hk
from xbox import pointprocess_models as pm

SLOT = 60.0
GAMMA_BOUNDS = (-1.0, 4.0)


@njit(cache=True)
def _estep_w(t, ev_m, ev_k, wev, L, P, eta, beta, cnt):
    n = t.shape[0]
    cnt[:, :] = 0.0
    R = 0.0
    Rp = 0.0
    ll = 0.0
    sum_q = 0.0
    sum_lag = 0.0
    prev = t[0]
    for i in range(n):
        if i > 0:
            d = t[i] - prev
            e = np.exp(-beta * d)
            Rp = e * (Rp + d * (1.0 + R))
            R = e * (1.0 + R)
        mu = L[ev_m[i]] * P[ev_k[i]] * wev[i]
        if mu < 1e-300:
            mu = 1e-300
        lam = mu + eta * beta * R
        ll += np.log(lam)
        pb = mu / lam
        cnt[ev_m[i], ev_k[i]] += pb
        sum_q += 1.0 - pb
        sum_lag += eta * beta * Rp / lam
        prev = t[i]
    return ll, sum_q, sum_lag


@njit(cache=True)
def _em_w(t, ev_m, ev_k, wev, occ, T, L, P, eta, beta, fix_eta, max_iter, tol):
    """EM of xbox.hawkes._em with event multipliers wev and occupancy occ already weighted by the driver."""
    M, K = occ.shape
    n = t.shape[0]
    cnt = np.zeros((M, K))
    ll_prev = -1e300
    ll = -1e300
    for it in range(max_iter):
        ll_ev, sum_q, sum_lag = _estep_w(t, ev_m, ev_k, wev, L, P, eta, beta, cnt)
        comp = 0.0
        for m in range(M):
            for k in range(K):
                comp += L[m] * P[k] * occ[m, k]
        G = hk._sumG(t, beta, T)
        ll = ll_ev - comp - eta * G
        if abs(ll - ll_prev) < tol * n:
            break
        ll_prev = ll
        if not fix_eta:
            eta = sum_q / G
            if eta > hk.ETA_MAX:
                eta = hk.ETA_MAX
            if eta < 0.0:
                eta = 0.0
            if sum_lag > 0.0 and sum_q > 0.0:
                beta = sum_q / sum_lag
                if beta > hk.BETA_MAX:
                    beta = hk.BETA_MAX
                if beta < hk.BETA_MIN:
                    beta = hk.BETA_MIN
        for _ in range(3):
            for m in range(M):
                num = 0.0
                den = 0.0
                for k in range(K):
                    num += cnt[m, k]
                    den += P[k] * occ[m, k]
                L[m] = num / den if den > 0 else 0.0
            for k in range(K):
                num = 0.0
                den = 0.0
                for m in range(M):
                    num += cnt[m, k]
                    den += L[m] * occ[m, k]
                P[k] = num / den if den > 0 else 0.0
        tot = 0.0
        wsum = 0.0
        for m in range(M):
            for k in range(K):
                tot += P[k] * occ[m, k]
                wsum += occ[m, k]
        s = tot / wsum if wsum > 0 else 1.0
        if s > 0:
            for k in range(K):
                P[k] /= s
            for m in range(M):
                L[m] *= s
    ll_ev, sum_q, sum_lag = _estep_w(t, ev_m, ev_k, wev, L, P, eta, beta, cnt)
    comp = 0.0
    for m in range(M):
        for k in range(K):
            comp += L[m] * P[k] * occ[m, k]
    ll = ll_ev - comp - eta * hk._sumG(t, beta, T)
    return eta, beta, ll


def standardize(x):
    x = np.asarray(x, dtype="float64")
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)


def _fit_gamma(data, z, gamma, poisson, init=None):
    d = data.hawkes_design()
    w = np.exp(gamma * z)
    occ = np.zeros((data.nm, 48))
    np.add.at(occ, (data.slot_month, data.slot_bin), SLOT * w)
    wev = w[data.slot]
    if init is None:
        cm = np.bincount(d["ev_m"], minlength=data.nm).astype(float)
        om = occ.sum(axis=1)
        L = np.where(om > 0, cm / np.maximum(om, 1e-12), 0.0)
        P = np.ones(48)
        eta0, beta0 = (0.0, 1.0 / 15.0) if poisson else (0.2, 1.0 / 15.0)
    else:
        L, P, eta0, beta0 = init["L"].copy(), init["P"].copy(), init["eta"], init["beta"]
    if poisson:
        eta0 = 0.0
    eta, beta, ll = _em_w(d["t"], d["ev_m"], d["ev_k"], wev, occ, d["T"], L, P, float(eta0), float(beta0), bool(poisson), 3000, 1e-9)
    return dict(L=L, P=P, eta=float(eta), beta=float(beta), ll=float(ll), gamma=float(gamma), w=w)


def fit_shared(data, z, poisson=False, fix_gamma=None):
    """Fit the model with the shared driver z (one value per slot). Returns a pointprocess_models.Res."""
    z = np.asarray(z, dtype="float64")
    assert len(z) == data.ns
    state = {"f": None}

    def nll(g):
        f = _fit_gamma(data, z, g, poisson, init=state["f"] if (state["f"] is not None and not poisson) else None)
        if state["f"] is None or f["ll"] > state["f"]["ll"]:
            state["f"] = f
        return -f["ll"]

    if fix_gamma is None:
        nll(0.0)
        o = optimize.minimize_scalar(nll, bounds=GAMMA_BOUNDS, method="bounded", options=dict(xatol=0.005))
        f = _fit_gamma(data, z, float(o.x), poisson, init=None if poisson else state["f"])
        if state["f"]["ll"] > f["ll"]:
            f = state["f"]
    else:
        f = _fit_gamma(data, z, float(fix_gamma), poisson)
    rate_slot = f["L"][data.slot_month] * f["P"][data.slot_bin] * f["w"]
    cum = np.concatenate([[0.0], np.cumsum(rate_slot * SLOT)])
    s = data.slot
    mcum = cum[s] + rate_slot[s] * (data.t - s * SLOT)
    inc = hk._increments(data.t, mcum, 0.0, f["eta"], f["beta"])
    first = np.log(rate_slot[s[0]]) - mcum[0]
    kpar = data.nm + 47 + (0 if poisson else 2) + (1 if fix_gamma is None else 0)
    res = pm.Res(ll=float(f["ll"] - first), k=kpar, u=-np.expm1(-inc),
                 params=dict(eta=f["eta"], inv_beta_min=1.0 / f["beta"], gamma=f["gamma"]),
                 state=dict(L=f["L"], P=f["P"], eta=f["eta"], beta=f["beta"], gamma=f["gamma"], rate_slot=rate_slot))
    return pm.finish(res, data.n)


def simulate_shared(data, res, seed):
    st = res["state"]
    return hk.simulate_hawkes(st["rate_slot"], st["eta"], st["beta"], seed)


def lag1(u):
    """Lag-1 rank autocorrelation of the rescaled intervals (0 under a correct model)."""
    from scipy.stats import spearmanr
    return float(spearmanr(u[:-1], u[1:])[0]) if len(u) > 3 else np.nan


def ks_location(u):
    """Where the largest gap between the empirical distribution of u and the uniform lies, and its sign.
    sign > 0: more small values of u than expected (intervals shorter than the model predicts)."""
    x = np.sort(u)
    n = len(x)
    i = np.arange(1, n + 1)
    up, dn = i / n - x, x - (i - 1) / n
    if up.max() >= dn.max():
        return float(x[up.argmax()]), 1
    return float(x[dn.argmax()]), -1
