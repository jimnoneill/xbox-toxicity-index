"""Binned point-process regression (MATH_SPEC section G), phase T4.

Contents
  fit_nb2            negative binomial (NB2, log link) maximum likelihood: IRLS for beta, profile likelihood for alpha,
                     model-based and week-clustered robust (sandwich) covariance, collinear columns dropped (NaN).
  calendar_arrays    per-member daily regime arrays from calendar_member_day.parquet
  build_static       static design for one account (intercept, hour x weekday-type cells, Congress term, regimes)
  lag_features       own-history features log(1+Y_{t-1}) and log(1+sum Y_{t-2..t-L}) with a validity mask
  simulate_nb        numba simulator of the fitted model (recursive in the own lags) for validation and predictive checks
  meta_dl, meta_reg  random-effects meta-analysis (DerSimonian-Laird; REML with optional moderators)
  daily_stats        binned statistics used in the predictive checks
NB2 parametrization: Var(Y) = mu + alpha mu^2, alpha = 1/k. Interpretation: associations, not causal effects.
"""
from __future__ import annotations

import numpy as np
from numba import njit
from scipy.optimize import minimize_scalar
from scipy.special import gammaln

REGIMES = ["in_session", "absent_in_session", "primary_window", "general_window", "post_election", "aligned_president",
           "party_majority", "event_curated", "event_detected"]
CAL_COLS = ["in_office", "chamber_in_session", "voted", "primary_window", "general_window", "post_election",
            "aligned_with_president", "in_chamber_majority", "event_day"]
CAL_DAY0 = 17318  # 2017-06-01 as days since 1970-01-01


# ----------------------------------------------------------------------------- NB2 estimator
def nb_loglik(y, mu, alpha):
    r = 1.0 / alpha
    return float(np.sum(gammaln(y + r) - gammaln(r) - gammaln(y + 1.0) + r * np.log(r / (r + mu))
                        + y * np.log(mu / (r + mu) + 1e-300)))


def _ll_beta(y, eta, alpha):
    """Log likelihood up to terms that do not depend on beta."""
    mu = np.exp(eta)
    return float(np.sum(y * (np.log(alpha * mu) - np.log1p(alpha * mu)) - np.log1p(alpha * mu) / alpha))


def independent_columns(X, tol=1e-9):
    """Sweep columns in order; keep a column when its residual after the kept ones is not negligible."""
    G = X.T @ X
    d = np.diag(G).copy()
    kept = []
    for j in range(X.shape[1]):
        if d[j] <= 0:
            continue
        if not kept:
            kept.append(j)
            continue
        Gk = G[np.ix_(kept, kept)]
        gj = G[kept, j]
        try:
            sol = np.linalg.solve(Gk + 1e-12 * np.eye(len(kept)) * np.trace(Gk) / len(kept), gj)
        except np.linalg.LinAlgError:
            continue
        res = d[j] - gj @ sol
        if res > tol * d[j]:
            kept.append(j)
    return np.array(kept, dtype=int)


def fit_nb2(X, y, offset=None, cluster=None, ridge=1e-8, maxit=40, tol=1e-8, alpha_bounds=(-9.0, 4.0)):
    """NB2 ML fit. X's first column must be the intercept (all ones).
    cluster: nondecreasing integer labels (rows in time order) for the robust covariance.
    Returns dict with beta, se_model, se_robust (length p, NaN for dropped columns), alpha, ll, n, converged."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    off = np.zeros(n) if offset is None else np.asarray(offset, dtype=float)
    keep = independent_columns(X)
    Xk = X[:, keep]
    pk = len(keep)
    beta = np.zeros(pk)
    beta[0] = np.log(max(y.mean(), 1e-6)) - off.mean()
    alpha = 0.5
    ll_old = -np.inf
    converged = False
    for outer in range(30):
        eta = np.clip(Xk @ beta + off, -30, 25)
        cur = _ll_beta(y, eta, alpha)
        for it in range(maxit):
            mu = np.exp(eta)
            w = mu / (1.0 + alpha * mu)
            z = eta - off + (y - mu) / mu
            XtW = Xk.T * w
            H = XtW @ Xk
            H[np.diag_indices(pk)] += ridge * (1 + np.diag(H))
            bnew = np.linalg.solve(H, XtW @ z)
            step = bnew - beta
            t = 1.0
            for _ in range(12):
                eta_n = np.clip(Xk @ (beta + t * step) + off, -30, 25)
                new = _ll_beta(y, eta_n, alpha)
                if new >= cur - 1e-10:
                    break
                t *= 0.5
            d = np.max(np.abs(t * step))
            beta = beta + t * step
            eta, cur = eta_n, new
            if d < tol:
                break
        mu = np.exp(eta)
        res = minimize_scalar(lambda la: -nb_loglik(y, mu, np.exp(la)), bounds=alpha_bounds, method="bounded",
                              options=dict(xatol=1e-4))
        alpha = float(np.exp(res.x))
        ll = -res.fun
        if abs(ll - ll_old) < 1e-7 * (1 + abs(ll)):
            converged = True
            break
        ll_old = ll
    eta = np.clip(Xk @ beta + off, -30, 25)
    mu = np.exp(eta)
    w = mu / (1.0 + alpha * mu)
    H = (Xk.T * w) @ Xk
    H[np.diag_indices(pk)] += ridge * (1 + np.diag(H))
    Hi = np.linalg.inv(H)
    se_m = np.sqrt(np.diag(Hi))
    if cluster is None:
        cluster = np.arange(n)
    cluster = np.asarray(cluster)
    starts = np.flatnonzero(np.r_[True, cluster[1:] != cluster[:-1]])
    s = Xk * ((y - mu) / (1.0 + alpha * mu))[:, None]
    S = np.add.reduceat(s, starts, axis=0)
    G = len(starts)
    meat = S.T @ S * (G / max(G - 1, 1))
    cov_r = Hi @ meat @ Hi
    se_r = np.sqrt(np.clip(np.diag(cov_r), 0, None))
    out = dict(beta=np.full(p, np.nan), se_model=np.full(p, np.nan), se_robust=np.full(p, np.nan), alpha=alpha,
               ll=float(nb_loglik(y, mu, alpha)), n=n, converged=converged, keep=keep, n_clusters=int(G),
               mean_mu=float(mu.mean()), cov_robust=cov_r)
    out["beta"][keep] = beta
    out["se_model"][keep] = se_m
    out["se_robust"][keep] = se_r
    return out


# ----------------------------------------------------------------------------- calendar and design
def calendar_arrays(cal):
    """cal: DataFrame with date, bioguide and CAL_COLS. Returns {bioguide: int8 array [ndays_from_CAL_DAY0, 9]}."""
    import pandas as pd
    out = {}
    days = (pd.to_datetime(cal["date"]).values.astype("datetime64[D]").astype("int64") - CAL_DAY0)
    nd = int(days.max()) + 1
    bg = cal["bioguide"].astype(str).values
    vals = cal[CAL_COLS].fillna(0).values.astype("int8")
    order = np.argsort(bg, kind="stable")
    bg, days, vals = bg[order], days[order], vals[order]
    starts = np.flatnonzero(np.r_[True, bg[1:] != bg[:-1]])
    ends = np.r_[starts[1:], len(bg)]
    for a, b in zip(starts, ends):
        arr = np.zeros((nd, len(CAL_COLS)), dtype="int8")
        arr[days[a:b]] = vals[a:b]
        out[bg[a]] = arr
    return out


def local_day_from_hours(hour_idx, local_hour):
    off = (local_hour.astype(int) - (hour_idx % 24).astype(int)) % 24
    off = np.where(off > 12, off - 24, off)
    return (hour_idx + off) // 24


def build_static(cell, local_day, logc, cal_arr, detected_days, bph=1, congress="plain", exclude_regimes=False):
    """Static (non-lag) design for the PRESENT rows of one account.

    cell      : integer weekend*(24*bph) + slot of day (hour x weekday-type cell)
    local_day : local calendar day (days since 1970-01-01)
    logc      : log(1 + leave-one-out Congress count) in each bin
    cal_arr   : daily regime array of the member (CAL_COLS) or None
    detected_days : boolean array indexed by day - CAL_DAY0, True on detected event days
    congress  : 'plain' (one column), 'split' (daily mean + within-day deviation), 'none'
    Returns (X, names, in_office_mask). Column order: intercept, cells, congress, regimes (regimes last so that
    collinear regime columns are the ones dropped).
    """
    n = len(cell)
    ncell = 48 * bph
    cols, names = [np.ones(n)], ["intercept"]
    ci = np.eye(ncell, dtype=float)[cell][:, 1:]
    cols += [ci[:, j] for j in range(ncell - 1)]
    names += [f"cell{j + 1}" for j in range(ncell - 1)]
    if congress == "plain":
        cols.append(logc)
        names.append("log1p_congress")
    elif congress == "split":
        ld = (local_day - local_day.min()).astype(int)
        sums = np.bincount(ld, weights=logc)
        cnt = np.bincount(ld)
        dm = (sums / np.maximum(cnt, 1))[ld]
        cols += [dm, logc - dm]
        names += ["log1p_congress_daymean", "log1p_congress_dev"]
    di = (local_day - CAL_DAY0).astype(int)
    nd = cal_arr.shape[0] if cal_arr is not None else 0
    inr = (di >= 0) & (di < nd)
    flags = np.zeros((n, len(REGIMES)))
    inoff = np.zeros(n, dtype=bool)
    if cal_arr is not None:
        c = np.zeros((n, len(CAL_COLS)))
        c[inr] = cal_arr[di[inr]]
        inoff = c[:, 0] == 1
        flags[:, :8] = c[:, 1:9]
        # reparametrize 'voted' (collinear with in_session): absent_in_session = chamber in session and member did not vote
        flags[:, 1] = flags[:, 0] * (1 - flags[:, 1])
        dd = np.zeros(n)
        ok = (di >= 0) & (di < len(detected_days))
        dd[ok] = detected_days[di[ok]]
        flags[:, 8] = dd
    if not exclude_regimes:
        cols += [flags[:, j] for j in range(len(REGIMES))]
        names += list(REGIMES)
    return np.column_stack(cols), names, inoff


def lag_features(y_dense, present, L=24):
    """y_dense: counts on the dense bin grid (0 where absent); present: bool mask of usable bins.
    lag1 = log(1+Y_{t-1}); lagL = log(1 + sum_{j=2..L} Y_{t-j}); valid = present and all of t-1..t-L present."""
    T = len(y_dense)
    y = np.where(present, y_dense, 0).astype(float)
    cs = np.concatenate([[0.0], np.cumsum(y)])
    cp = np.concatenate([[0], np.cumsum(present.astype(int))])
    t = np.arange(T)
    lag1 = np.zeros(T)
    lag1[1:] = np.log1p(y[:-1])
    lo = np.clip(t - L, 0, T)
    hi = np.clip(t - 1, 0, T)  # sum over indices [t-L, t-2] = cs[t-1] - cs[t-L]
    s = cs[hi] - cs[lo]
    lagL = np.log1p(np.where(t >= 2, s, 0.0))
    full = (cp[t] - cp[lo]) == L  # all of [t-L, t-1] present
    valid = present & (t >= L) & full
    return lag1, lagL, valid


# ----------------------------------------------------------------------------- simulation
@njit(cache=True)
def _sim(eta0, phi1, phiL, alpha, L, seed):
    np.random.seed(seed)
    T = len(eta0)
    y = np.zeros(T)
    s = 0.0
    for t in range(T):
        if t >= 2:
            s += y[t - 2]
        if t >= L + 1:
            s -= y[t - L - 1]
        lag1 = np.log1p(y[t - 1]) if t >= 1 else 0.0
        eta = eta0[t] + phi1 * lag1 + phiL * np.log1p(s)
        if eta0[t] <= -30.0:
            y[t] = 0.0
            continue
        if eta > 9.0:
            eta = 9.0
        mu = np.exp(eta)
        if alpha < 1e-6:
            y[t] = np.random.poisson(mu)
        else:
            lam = np.random.gamma(1.0 / alpha, alpha * mu)
            y[t] = np.random.poisson(lam)
    return y


def simulate_nb(eta0, phi1, phiL, alpha, seed, L=24):
    """Simulate counts from the model on a dense grid. eta0 = static linear predictor (<= -30 forces zero)."""
    return _sim(np.asarray(eta0, dtype=float), float(phi1), float(phiL), float(alpha), int(L), int(seed))


# ----------------------------------------------------------------------------- meta-analysis
def meta_dl(y, se):
    y, se = np.asarray(y, float), np.asarray(se, float)
    ok = np.isfinite(y) & np.isfinite(se) & (se > 0)
    y, se = y[ok], se[ok]
    k = len(y)
    if k < 3:
        return dict(k=k, mu=np.nan, se=np.nan, tau2=np.nan, tau=np.nan, Q=np.nan, I2=np.nan, lo=np.nan, hi=np.nan)
    w = 1 / se ** 2
    mu_f = np.sum(w * y) / np.sum(w)
    Q = float(np.sum(w * (y - mu_f) ** 2))
    c = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max(0.0, (Q - (k - 1)) / c)
    ws = 1 / (se ** 2 + tau2)
    mu = float(np.sum(ws * y) / np.sum(ws))
    sem = float(1 / np.sqrt(np.sum(ws)))
    return dict(k=k, mu=mu, se=sem, tau2=float(tau2), tau=float(np.sqrt(tau2)), Q=Q,
                I2=float(max(0, (Q - (k - 1)) / Q)) if Q > 0 else 0.0, lo=mu - 1.96 * sem, hi=mu + 1.96 * sem)


def meta_reg(y, se, mods=None, method="REML"):
    """Random-effects meta-regression. mods: (k, q) moderator matrix WITHOUT intercept (None = overall mean).
    Returns coefficient vector (intercept first), SEs, tau2, Q, I2, k."""
    y, se = np.asarray(y, float), np.asarray(se, float)
    X = np.ones((len(y), 1)) if mods is None else np.column_stack([np.ones(len(y)), np.asarray(mods, float)])
    ok = np.isfinite(y) & np.isfinite(se) & (se > 0) & np.all(np.isfinite(X), axis=1)
    y, se, X = y[ok], se[ok], X[ok]
    k, q = X.shape
    v = se ** 2

    def fitw(t2):
        w = 1 / (v + t2)
        A = X.T @ (X * w[:, None])
        b = np.linalg.solve(A, X.T @ (w * y))
        r = y - X @ b
        return b, A, r, w

    def nreml(t2):
        b, A, r, w = fitw(t2)
        return 0.5 * (np.sum(np.log(v + t2)) + np.linalg.slogdet(A)[1] + np.sum(w * r ** 2))

    b0, A0, r0, w0 = fitw(0.0)
    Q = float(np.sum(w0 * r0 ** 2))
    if method == "DL":
        P = np.sum(w0) - np.trace(np.linalg.solve(A0, X.T @ (X * (w0 ** 2)[:, None])))
        t2 = max(0.0, (Q - (k - q)) / P)
    else:
        hi_ = max(10 * np.var(y), 10 * v.max(), 1e-6)
        res = minimize_scalar(nreml, bounds=(0.0, hi_), method="bounded", options=dict(xatol=1e-12))
        t2 = float(res.x) if nreml(0.0) > res.fun + 1e-9 else 0.0
    b, A, r, w = fitw(t2)
    cov = np.linalg.inv(A)
    return dict(k=int(k), coef=b, se=np.sqrt(np.diag(cov)), cov=cov, tau2=float(t2), tau=float(np.sqrt(t2)), Q=Q,
                I2=float(max(0.0, (Q - (k - q)) / Q)) if Q > 0 else 0.0)


# ----------------------------------------------------------------------------- predictive statistics
def daily_stats(y, present, local_hour, local_day, detected_days, bph=1):
    """Binned statistics on a dense grid. y, present, local_hour, local_day: dense arrays (absent bins ignored)."""
    pres = present.astype(bool)
    yy = np.where(pres, y, 0.0)
    tot = yy.sum()
    nb_days = pres.sum() / (24.0 * bph)
    prev = np.r_[0.0, yy[:-1]]
    prev_ok = np.r_[False, pres[:-1]]
    runs = np.sum((yy > 0) & ~((prev > 0) & prev_ok))
    out = dict(events=float(tot), sessions_per_day=float(runs / nb_days) if nb_days > 0 else np.nan,
               night_share=float(yy[local_hour < 6].sum() / tot) if tot > 0 else np.nan)
    blk = np.arange(len(yy)) // (168 * bph)
    wk = np.bincount(blk, weights=yy)
    wp = np.bincount(blk, weights=pres.astype(float))
    good = wp >= 0.9 * 168 * bph
    out["weekly_mean"] = float(wk[good].mean()) if good.any() else np.nan
    out["weekly_cv"] = float(wk[good].std() / wk[good].mean()) if good.sum() > 3 and wk[good].mean() > 0 else np.nan
    ld = (local_day - local_day.min()).astype(int)
    dsum = np.bincount(ld[pres], weights=yy[pres], minlength=ld.max() + 1)
    dcnt = np.bincount(ld[pres], minlength=ld.max() + 1)
    okd = dcnt >= 0.9 * 24 * bph
    absday = np.arange(len(dsum)) + int(local_day.min())
    di = absday - CAL_DAY0
    ev = np.zeros(len(dsum), dtype=bool)
    m_ = (di >= 0) & (di < len(detected_days))
    ev[m_] = detected_days[di[m_]]
    ev &= okd
    base_ok = okd & ~ev
    ratios = []
    for d in np.flatnonzero(ev):
        lo, hi = max(0, d - 28), min(len(dsum), d + 29)
        idx = np.arange(lo, hi)
        idx = idx[(np.abs(idx - d) >= 7) & base_ok[idx]]
        if len(idx) >= 6:
            b = dsum[idx].mean()
            if b > 0:
                ratios.append(dsum[d] / b)
    out["event_day_ratio"] = float(np.mean(ratios)) if len(ratios) >= 3 else np.nan
    out["n_event_days"] = len(ratios)
    return out
