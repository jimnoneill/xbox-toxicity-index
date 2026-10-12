"""Univariate exponential Hawkes process with a structured inhomogeneous baseline (spec C5).

    lambda(t) = mu(t) + eta * beta * sum_{t_i < t} exp(-beta (t - t_i)),   mu(t) = L[m(t)] * P[k(t)]

eta = alpha/beta is the branching ratio, 1/beta the kernel time scale. m(t) is the calendar month
and k(t) one of 48 local hour-of-week bins (24 hours x weekday/weekend). mu is piecewise constant on
one-hour slots. Fit by EM (Veen-Schoenberg / Lewis-Mohler): E-step gives branching probabilities via
the O(n) exponential-kernel recursion, M-step updates eta, beta and the baseline factors.
Time unit inside this module is MINUTES. beta is per minute.

Also: profile-likelihood CI for eta, time-rescaling KS check, Ogata-thinning simulator for validation.
"""
from __future__ import annotations

import numpy as np
from numba import njit

SLOT_MIN = 60.0
BETA_MIN = 1.0 / (7 * 24 * 60.0)   # kernel scale at most 7 days
BETA_MAX = 1.0 / 0.1               # kernel scale at least 6 s
ETA_MAX = 0.999


# ------------------------------------------------------------------ design
def make_design(t_sec, slot_month, slot_bin, slot0_sec, n_months, n_bins=48, jitter_seed=0, jitter=True):
    """Build the integer design for one entity.

    t_sec        sorted event times, UTC epoch seconds (int)
    slot_month   month index (0..n_months-1) of every one-hour slot starting at slot0_sec
    slot_bin     hour-of-week bin (0..n_bins-1) of every slot
    Events get uniform sub-second jitter (fixed seed) so that equal-second posts are not exact ties.
    Returns dict with t (minutes since slot0), ev_m, ev_k, occ (n_months x n_bins minutes), T.
    """
    rng = np.random.default_rng(jitter_seed)
    t = (np.asarray(t_sec, dtype="float64") - slot0_sec) / 60.0
    if jitter:
        t = t + rng.uniform(0, 1.0 / 60.0, size=len(t))
    t = np.sort(t)
    ns = len(slot_month)
    slot = np.minimum((t // SLOT_MIN).astype("int64"), ns - 1)
    occ = np.zeros((n_months, n_bins))
    np.add.at(occ, (slot_month, slot_bin), SLOT_MIN)
    return dict(t=t, ev_m=slot_month[slot].astype("int64"), ev_k=slot_bin[slot].astype("int64"),
                occ=occ, T=ns * SLOT_MIN, slot_month=slot_month.astype("int64"),
                slot_bin=slot_bin.astype("int64"))


def constant_baseline_design(d):
    """Same events but a single level and a single bin (what a constant-rate fit would use)."""
    occ = np.array([[d["occ"].sum()]])
    z = np.zeros_like(d["ev_m"])
    return dict(t=d["t"], ev_m=z, ev_k=z.copy(), occ=occ, T=d["T"],
                slot_month=np.zeros_like(d["slot_month"]), slot_bin=np.zeros_like(d["slot_bin"]))


# ------------------------------------------------------------------ EM
@njit(cache=True)
def _estep(t, ev_m, ev_k, L, P, eta, beta, cnt):
    """One E-step pass. Returns (loglik_events, sum_children, sum_lag, sum_G) and fills cnt (M x K)
    with expected background counts."""
    n = t.shape[0]
    cnt[:, :] = 0.0
    R = 0.0
    Rp = 0.0
    ll = 0.0
    sum_q = 0.0
    sum_lag = 0.0
    sum_G = 0.0
    T_dummy = 0.0
    prev = t[0]
    for i in range(n):
        if i > 0:
            d = t[i] - prev
            e = np.exp(-beta * d)
            Rp = e * (Rp + d * (1.0 + R))
            R = e * (1.0 + R)
        mu = L[ev_m[i]] * P[ev_k[i]]
        if mu < 1e-300:
            mu = 1e-300
        exc = eta * beta * R
        lam = mu + exc
        ll += np.log(lam)
        pb = mu / lam
        cnt[ev_m[i], ev_k[i]] += pb
        sum_q += 1.0 - pb
        sum_lag += eta * beta * Rp / lam
        prev = t[i]
    return ll, sum_q, sum_lag


@njit(cache=True)
def _sumG(t, beta, T):
    s = 0.0
    for i in range(t.shape[0]):
        s += 1.0 - np.exp(-beta * (T - t[i]))
    return s


@njit(cache=True)
def _em(t, ev_m, ev_k, occ, T, L, P, eta, beta, fix_eta, max_iter, tol, kappa):
    M, K = occ.shape
    n = t.shape[0]
    cnt = np.zeros((M, K))
    ll_prev = -1e300
    ll = -1e300
    it = 0
    for it in range(max_iter):
        ll_ev, sum_q, sum_lag = _estep(t, ev_m, ev_k, L, P, eta, beta, cnt)
        comp = 0.0
        for m in range(M):
            for k in range(K):
                comp += L[m] * P[k] * occ[m, k]
        G = _sumG(t, beta, T)
        ll = ll_ev - comp - eta * G
        if abs(ll - ll_prev) < tol * n:
            break
        ll_prev = ll
        # M-step: kernel parameters
        if not fix_eta:
            eta = sum_q / G
            if eta > ETA_MAX:
                eta = ETA_MAX
            if eta < 0.0:
                eta = 0.0
        if sum_lag > 0.0 and sum_q > 0.0:
            beta = sum_q / sum_lag
            if beta > BETA_MAX:
                beta = BETA_MAX
            if beta < BETA_MIN:
                beta = BETA_MIN
        # M-step: baseline factors, three alternating sweeps
        # Gamma(kappa, kappa/Lbar) prior on the levels (kappa > 0: random-effect shrinkage, MAP update)
        lbar = 0.0
        if kappa > 0.0:
            tn = 0.0
            td = 0.0
            for m in range(M):
                for k in range(K):
                    tn += cnt[m, k]
                    td += P[k] * occ[m, k]
            lbar = tn / td if td > 0 else 1.0
        for _ in range(3):
            for m in range(M):
                num = 0.0
                den = 0.0
                for k in range(K):
                    num += cnt[m, k]
                    den += P[k] * occ[m, k]
                if kappa > 0.0:
                    L[m] = (num + kappa - 1.0) / (den + kappa / lbar) if kappa > 1.0 else (num + kappa) / (den + kappa / lbar)
                else:
                    L[m] = num / den if den > 0 else 0.0
            for k in range(K):
                num = 0.0
                den = 0.0
                for m in range(M):
                    num += cnt[m, k]
                    den += L[m] * occ[m, k]
                P[k] = num / den if den > 0 else 0.0
        # normalise P to occupancy-weighted mean one
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
    # final log-likelihood at current parameters
    ll_ev, sum_q, sum_lag = _estep(t, ev_m, ev_k, L, P, eta, beta, cnt)
    comp = 0.0
    for m in range(M):
        for k in range(K):
            comp += L[m] * P[k] * occ[m, k]
    ll = ll_ev - comp - eta * _sumG(t, beta, T)
    return eta, beta, ll, it + 1


def fit(design, eta_fixed=None, init=None, max_iter=3000, tol=1e-9, prior_kappa=0.0):
    """EM fit. prior_kappa > 0 puts a Gamma(kappa, kappa/Lbar) prior on the baseline levels L
    (empirical-Bayes random effect; used with day-level levels). Returns dict(eta, beta, inv_beta_min, L, P, ll, n_iter, n)."""
    t, ev_m, ev_k, occ, T = design["t"], design["ev_m"], design["ev_k"], design["occ"], design["T"]
    M, K = occ.shape
    n = len(t)
    if init is None:
        cm = np.bincount(ev_m, minlength=M).astype(float)
        om = occ.sum(axis=1)
        L = np.where(om > 0, cm / np.maximum(om, 1e-12), 0.0)
        P = np.ones(K)
        eta0, beta0 = 0.2, 1.0 / 15.0
    else:
        L, P, eta0, beta0 = init["L"].copy(), init["P"].copy(), init["eta"], init["beta"]
    fix = eta_fixed is not None
    if fix:
        eta0 = float(eta_fixed)
    eta, beta, ll, it = _em(t, ev_m, ev_k, occ, T, L, P, float(eta0), float(beta0), fix, max_iter, tol, float(prior_kappa))
    return dict(eta=float(eta), beta=float(beta), inv_beta_min=float(1.0 / beta), L=L, P=P, ll=float(ll),
                n_iter=int(it), n=n)


def profile_ci(design, fit0, level=0.95, tol=1e-3):
    """Profile-likelihood CI for eta: {eta : 2 (ll_hat - ll_profile(eta)) <= chi2_1(level)}."""
    from scipy.stats import chi2
    cut = chi2.ppf(level, 1) / 2.0
    ll_hat = fit0["ll"]

    def gap(e):
        f = fit(design, eta_fixed=e, init=dict(L=fit0["L"], P=fit0["P"], eta=e, beta=fit0["beta"]))
        return ll_hat - f["ll"] - cut

    eh = fit0["eta"]
    # lower bound
    if eh <= 0.0 or gap(0.0) < 0:
        lo = 0.0
    else:
        a, b = 0.0, eh  # gap(a) > 0, gap(b) < 0
        while b - a > tol:
            c = 0.5 * (a + b)
            if gap(c) > 0:
                a = c
            else:
                b = c
        lo = 0.5 * (a + b)
    if eh >= ETA_MAX or gap(ETA_MAX) < 0:
        hi = ETA_MAX
    else:
        a, b = eh, ETA_MAX
        while b - a > tol:
            c = 0.5 * (a + b)
            if gap(c) < 0:
                a = c
            else:
                b = c
        hi = 0.5 * (a + b)
    return float(lo), float(hi)


# ------------------------------------------------------------------ goodness of fit
@njit(cache=True)
def _increments(t, mcum_at_event, mcum_start, eta, beta):
    n = t.shape[0]
    out = np.empty(n)
    a = 0.0  # sum_{j <= i-1} exp(-beta (t_{i-1} - t_j))
    prev_t = 0.0
    prev_m = mcum_start
    for i in range(n):
        d = t[i] - prev_t
        inc = mcum_at_event[i] - prev_m
        if i > 0:
            inc += eta * a * (1.0 - np.exp(-beta * d))
            a = a * np.exp(-beta * d) + 1.0
        else:
            a = 1.0
        out[i] = inc
        prev_t = t[i]
        prev_m = mcum_at_event[i]
    return out


def time_rescaling(design, f):
    """Rescaled inter-arrival times (should be iid Exp(1)) and their KS statistic against Exp(1)."""
    from scipy.stats import kstest
    t = design["t"]
    rate_slot = f["L"][design["slot_month"]] * f["P"][design["slot_bin"]]
    cum = np.concatenate([[0.0], np.cumsum(rate_slot * SLOT_MIN)])
    s = np.minimum((t // SLOT_MIN).astype("int64"), len(rate_slot) - 1)
    mcum = cum[s] + rate_slot[s] * (t - s * SLOT_MIN)
    inc = _increments(t, mcum, 0.0, f["eta"], f["beta"])
    ks = kstest(inc, "expon")
    return dict(ks_stat=float(ks.statistic), ks_p=float(ks.pvalue), increments=inc)


# ------------------------------------------------------------------ Ogata thinning simulator
@njit(cache=True)
def _thin(mu_slot, eta, beta, T, seed, cap):
    np.random.seed(seed)
    mu_max = mu_slot.max()
    alpha = eta * beta
    ev = np.empty(cap)
    n = 0
    t = 0.0
    S = 0.0
    while True:
        lam_bar = mu_max + S
        w = -np.log(1.0 - np.random.random()) / lam_bar
        tn = t + w
        if tn >= T:
            break
        S *= np.exp(-beta * w)
        lam = mu_slot[int(tn // 60.0)] + S
        if np.random.random() * lam_bar <= lam:
            if n == cap:
                ev2 = np.empty(cap * 2)
                ev2[:n] = ev[:n]
                ev = ev2
                cap *= 2
            ev[n] = tn
            n += 1
            S += alpha
        t = tn
    return ev[:n]


def simulate_hawkes(mu_slot, eta, beta, seed):
    """Ogata thinning. mu_slot: baseline rate per minute on one-hour slots; returns event times (min)."""
    T = len(mu_slot) * SLOT_MIN
    cap = int(mu_slot.sum() * SLOT_MIN / max(1e-9, 1 - eta) * 1.5) + 1000
    return _thin(np.asarray(mu_slot, dtype="float64"), float(eta), float(beta), T, int(seed), cap)
