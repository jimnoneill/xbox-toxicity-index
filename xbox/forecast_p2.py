"""P2 volume and burst forecasting (MATH_SPEC section H, task P2). Library module.

Everything is causal. Time axis: hours since 2017-06-21 04:00 UTC (hour_idx - 416116). A forecast origin t uses data
through hour t only; the forecast targets hours t+1.. . Forecast times lie on a stride-5 grid (coprime with 24 and 168).

Targets: S1 = Y[t+1]; S24 = sum Y[t+1..t+24]; S168 = sum Y[t+1..t+168]; burst = max Y[t+1..t+6] >= theta_a, where theta_a is
max(2, 90th percentile of the account's positive hourly counts in the training window).
Monthly origins T0: all models are refit (or recalibrated) with data whose target window ends before T0; level states
(exponentially weighted levels, lagged counts) are updated at every forecast time inside the month.
Calendar regime flags used: only those fixed in advance (primary window, general-election window, post-election window,
federal holiday, State of the Union day). Session/roll-call flags are NOT used (not known in advance).
Congress-wide activity enters only through lags (leave-one-out, own events removed), never same-hour.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import lfilter
from scipy.special import betainc, gammaln, expit

SEED = 20261010
HOUR0 = 416116
T0_TS = pd.Timestamp("2017-06-21 04:00:00")
STRIDE = 5
CAL_STRIDE = 25
HORIZONS = {"h1": 1, "d24": 24, "w168": 168}
BURST_H = 6
TEST_START = pd.Timestamp("2022-07-01 00:00:00")
DEV_START = pd.Timestamp("2019-07-01 00:00:00")
HLS = (7 * 24, 21 * 24, 60 * 24)          # level half-lives in hours
MIN_HIST = 90 * 24
EPS = 1e-9


def hidx(ts):
    return int((pd.Timestamp(ts) - T0_TS) / pd.Timedelta(hours=1))


# ------------------------------------------------------------------ panel
def et_offset_hours(utc):
    """US Eastern offset (-4 during DST, -5 otherwise) for a naive-UTC DatetimeIndex (no tz database needed)."""
    out = np.full(len(utc), -5.0)
    for y in range(utc.year.min(), utc.year.max() + 1):
        mar = pd.Timestamp(f"{y}-03-01")
        start = mar + pd.Timedelta(days=(6 - mar.dayofweek) % 7 + 7) + pd.Timedelta(hours=7)   # 2nd Sunday 02:00 EST = 07:00 UTC
        nov = pd.Timestamp(f"{y}-11-01")
        end = nov + pd.Timedelta(days=(6 - nov.dayofweek) % 7) + pd.Timedelta(hours=6)          # 1st Sunday 02:00 EDT = 06:00 UTC
        out[(utc >= start) & (utc < end)] = -4.0
    return out


class Panel:
    """Dense account x hour matrices. Y is NaN outside the account's span (first to last event) and in outage hours."""

    def __init__(self, hc: pd.DataFrame, cong: pd.DataFrame, cal: pd.DataFrame | None = None):
        self.accounts = sorted(hc["account"].unique())
        a_ix = {a: i for i, a in enumerate(self.accounts)}
        self.T = int(cong["hour_idx"].max() - HOUR0 + 1)
        A, T = len(self.accounts), self.T
        self.Y = np.full((A, T), np.nan, np.float32)
        self.P = np.full((A, T), np.nan, np.float32)
        ai = hc["account"].map(a_ix).values
        ti = (hc["hour_idx"].values - HOUR0).astype(np.int64)
        self.Y[ai, ti] = hc["count"].values
        self.P[ai, ti] = hc["count_posts"].values
        st = hc.groupby("account").agg(typ=("account_type", "first"), party=("party", "first"), chamber=("chamber", "first"),
                                        bio=("bioguide", "first"), h0=("hour_idx", "min"), h1=("hour_idx", "max"),
                                        tot=("count", "sum")).loc[self.accounts]
        self.static = st
        self.span0 = (st["h0"].values - HOUR0).astype(np.int64)
        self.span1 = (st["h1"].values - HOUR0).astype(np.int64)
        c = cong.sort_values("hour_idx")
        self.C = np.zeros(T, np.float64)
        self.C[(c["hour_idx"].values - HOUR0)] = c["total_events"].values
        self.obs_cong = np.zeros(T, bool)
        self.obs_cong[(c["hour_idx"].values - HOUR0)] = c["observed"].values
        utc = pd.DatetimeIndex(T0_TS + pd.to_timedelta(np.arange(T), unit="h"))
        et = utc + pd.to_timedelta(et_offset_hours(utc), unit="h")
        self.how = (et.dayofweek * 24 + et.hour).values.astype(np.int64)
        self.etday = ((et.normalize() - pd.Timestamp("1970-01-01")) // pd.Timedelta(days=1)).values.astype(np.int64)
        self.Y0 = np.nan_to_num(self.Y, nan=0.0)
        self.cs = np.concatenate([np.zeros((A, 1)), np.cumsum(self.Y0, axis=1, dtype=np.float64)], axis=1)
        self.cnt_valid = np.concatenate([np.zeros((A, 1)), np.cumsum(np.isfinite(self.Y), axis=1, dtype=np.int32)], axis=1)
        self.flags = self._calendar(cal) if cal is not None else None

    def _calendar(self, cal):
        """Per-account daily regime flags known in advance, indexed by ET day (days since 1970)."""
        d0, d1 = int(self.etday.min()), int(self.etday.max())
        nd = d1 - d0 + 1
        A = len(self.accounts)
        names = ["primary_window", "general_window", "post_election", "sotu"]
        out = {n: np.zeros((A, nd), np.float32) for n in names}
        a_by_bio = {}
        for i, b in enumerate(self.static["bio"].values):
            a_by_bio.setdefault(str(b), []).append(i)
        cal = cal[cal["bioguide"].astype(str).isin(a_by_bio)]
        di = ((cal["date"] - pd.Timestamp("1970-01-01")) // pd.Timedelta(days=1)).values.astype(np.int64) - d0
        ok = (di >= 0) & (di < nd)
        bios = cal["bioguide"].astype(str).values
        for n in names:
            vals = cal[n].values.astype(np.float32)
            for b, rows in a_by_bio.items():
                m = (bios == b) & ok
                if m.any():
                    for r in rows:
                        out[n][r, di[m]] = vals[m]
        from pandas.tseries.holiday import USFederalHolidayCalendar
        hol = USFederalHolidayCalendar().holidays(start="2017-01-01", end="2024-01-01")
        hd = ((hol - pd.Timestamp("1970-01-01")) // pd.Timedelta(days=1)).values.astype(np.int64) - d0
        h = np.zeros(nd, np.float32)
        h[hd[(hd >= 0) & (hd < nd)]] = 1
        out["holiday"] = h
        out["d0"] = d0
        return out

    def window_sum(self, ai, lo, hi):
        """sum of Y over hours lo..hi-1 (inclusive lo) for account indices ai and hour arrays lo/hi (clipped to [0,T])."""
        lo = np.clip(lo, 0, self.T)
        hi = np.clip(hi, 0, self.T)
        return self.cs[ai, hi] - self.cs[ai, lo]

    def window_valid(self, ai, lo, hi):
        """True if every hour in lo..hi-1 is observed and inside the account's span."""
        ok = (lo >= 0) & (hi <= self.T)
        lo_c, hi_c = np.clip(lo, 0, self.T), np.clip(hi, 0, self.T)
        full = (self.cnt_valid[ai, hi_c] - self.cnt_valid[ai, lo_c]) == (hi - lo)
        return ok & full


def ew_levels(panel: Panel, hl_hours, prior_rate=0.065, prior_w=24.0):
    """Causal exponentially weighted hourly rate per account at every hour (information through that hour)."""
    d = 0.5 ** (1.0 / hl_hours)
    obs = np.isfinite(panel.Y).astype(np.float64)
    num = lfilter([1.0], [1.0, -d], panel.Y0.astype(np.float64), axis=1)
    den = lfilter([1.0], [1.0, -d], obs, axis=1)
    return ((num + prior_rate * prior_w) / (den + prior_w)).astype(np.float32)


def congress_level(panel: Panel, hl_hours):
    d = 0.5 ** (1.0 / hl_hours)
    num = lfilter([1.0], [1.0, -d], panel.C)
    den = lfilter([1.0], [1.0, -d], panel.obs_cong.astype(float))
    return (num + 1.0) / (den + 1.0)


# ------------------------------------------------------------------ profiles
def hourly_profile(panel: Panel, ai_rows, t_end, last_hours=730 * 24, kappa=20.0):
    """Account hour-of-week profile (mean 1) from data before t_end, shrunk toward the pooled shape of the same set.

    Returns prof (A,168), ybar (A,) mean rate, plus the pooled shape g(168).
    """
    lo = max(0, t_end - last_hours)
    Yw = panel.Y[:, lo:t_end]
    how = panel.how[lo:t_end]
    ok = np.isfinite(Yw)
    OH = np.zeros((t_end - lo, 168), np.float32)
    OH[np.arange(t_end - lo), how] = 1.0
    S = np.nan_to_num(Yw).astype(np.float32) @ OH
    N = ok.astype(np.float32) @ OH
    S, N = S.astype(np.float64), N.astype(np.float64)
    n_tot = N.sum(1)
    ybar = np.where(n_tot > 0, S.sum(1) / np.maximum(n_tot, 1), 0.0)
    rel = S / np.maximum(ybar[:, None], 1e-6)           # sum of (y / ybar) in each cell
    w = ok.any(axis=1) & (ybar > 0)
    g = rel[w].sum(0) / np.maximum(N[w].sum(0), 1)
    g = g / g.mean() if w.any() else np.ones(168)
    prof = (rel + kappa * g[None, :]) / (N + kappa)
    prof = prof / prof.mean(axis=1, keepdims=True)
    prof[~w] = 1.0
    return prof, ybar, g


def cong_profile(panel: Panel, t_end, last_hours=730 * 24):
    lo = max(0, t_end - last_hours)
    c = panel.C[lo:t_end]
    how = panel.how[lo:t_end]
    ob = panel.obs_cong[lo:t_end]
    prof = np.ones(168)
    lv = congress_level(panel, 21 * 24)[lo:t_end]
    r = np.where(ob, c / np.maximum(lv, 1e-6), np.nan)
    for h in range(168):
        v = r[how == h]
        v = v[np.isfinite(v)]
        if len(v):
            prof[h] = v.mean()
    return prof / prof.mean()


def cum_profile(prof):
    """(A,337) cumulative profile so window sums over any 'how' range of length <=168 are differences."""
    p2 = np.concatenate([prof, prof, prof[:, :1]], axis=1)
    return np.concatenate([np.zeros((prof.shape[0], 1)), np.cumsum(p2, axis=1)], axis=1)[:, :338]


def prof_window(cp, ai, how_start, length):
    hs = how_start % 168
    return cp[ai, hs + length] - cp[ai, hs]


# ------------------------------------------------------------------ count distributions
def nb_logpmf(y, mu, alpha):
    mu = np.maximum(mu, 1e-9)
    alpha = np.maximum(alpha, 1e-6)
    r = 1.0 / alpha
    return gammaln(y + r) - gammaln(r) - gammaln(y + 1.0) + r * np.log(r / (r + mu)) + y * np.log(mu / (r + mu))


def nb_cdf(k, mu, alpha):
    """P(Y<=k) for NB2 with mean mu and variance mu + alpha mu^2 (k may be array)."""
    mu = np.maximum(mu, 1e-9)
    r = 1.0 / np.maximum(alpha, 1e-6)
    p = r / (r + mu)
    return betainc(r, np.floor(k) + 1.0, p)


def nb_crps(y, mu, alpha, chunk=200000):
    """CRPS for NB2 predictive distribution and integer outcome y: sum_k (F(k)-1[y<=k])^2 over k>=0 (exact for integer support)."""
    n = len(y)
    out = np.empty(n)
    var = mu + alpha * mu ** 2
    upper = np.maximum(y, mu + 12 * np.sqrt(var) + 5)
    order = np.argsort(upper)
    for s in range(0, n, chunk):
        ix = order[s:s + chunk]
        K = int(np.ceil(upper[ix].max())) + 1
        ks = np.arange(K)[None, :]
        F = nb_cdf(ks, mu[ix][:, None], alpha[ix][:, None])
        ind = (ks >= y[ix][:, None]).astype(float)
        out[ix] = ((F - ind) ** 2).sum(1)
    return out


def nb_pit(y, mu, alpha, u):
    """Randomized PIT for integer outcomes."""
    lo = nb_cdf(y - 1, mu, alpha) * (y > 0)
    hi = nb_cdf(y, mu, alpha)
    return lo + u * (hi - lo)


def fit_alpha_accounts(y, mu, acct, n_acct, pool_prior=1.0, grid=None):
    """Per-account NB dispersion by penalized profile likelihood on calibration rows.
    Prior: log alpha ~ N(log alpha_pool, pool_prior^2); alpha_pool from the summed likelihood."""
    grid = np.linspace(-3.5, 3.0, 27) if grid is None else grid
    al = np.exp(grid)
    ll = np.zeros((n_acct, len(al)))
    m = np.isfinite(y) & np.isfinite(mu)
    y, mu, acct = y[m], mu[m], acct[m]
    for j, a in enumerate(al):
        np.add.at(ll[:, j], acct, nb_logpmf(y, mu, a))
    pool = ll.sum(0)
    lp = grid[np.argmax(pool)]
    pen = -0.5 * ((grid - lp) / pool_prior) ** 2
    return np.exp(grid[np.argmax(ll + pen[None, :], axis=1)]), float(np.exp(lp))


# ------------------------------------------------------------------ negative binomial regression
def fit_nb(X, y, off, alpha0=1.0, ridge=1e-3, iters=2, w=None):
    """NB2 GLM with log link, log offset, ridge on non-intercept coefficients (column 0 is the intercept). Alternates beta (L-BFGS)
    and a scalar alpha. Returns beta, alpha."""
    from scipy.optimize import minimize, minimize_scalar
    n, p = X.shape
    w = np.ones(n) if w is None else w
    beta = np.zeros(p)
    beta[0] = np.log(max((y * w).sum() / max((np.exp(off) * w).sum(), 1e-9), 1e-6))
    alpha = alpha0
    pen = np.full(p, ridge)
    pen[0] = 0.0

    def f(b, a):
        eta = np.clip(off + X @ b, -20, 14)
        mu = np.exp(eta)
        r = 1.0 / a
        ll = (w * (r * np.log(r / (r + mu)) + y * np.log(mu / (r + mu)))).sum()   # constants omitted (alpha fixed)
        g = X.T @ (w * (y - mu) / (1.0 + a * mu))
        return -ll / n + 0.5 * (pen * b * b).sum() / n, -g / n + pen * b / n

    for _ in range(iters):
        res = minimize(f, beta, args=(alpha,), jac=True, method="L-BFGS-B", options={"maxiter": 200})
        beta = res.x
        mu = np.exp(np.clip(off + X @ beta, -20, 14))
        o = minimize_scalar(lambda la: -(w * nb_logpmf(y, mu, np.exp(la))).sum(), bounds=(-5, 4), method="bounded")
        alpha = float(np.exp(o.x))
    return beta, alpha


def fit_logit(X, y, ridge=1e-2, w=None):
    from scipy.optimize import minimize
    n, p = X.shape
    w = np.ones(n) if w is None else w
    pen = np.full(p, ridge)
    pen[0] = 0.0

    def f(b):
        z = X @ b
        ll = (w * (y * z - np.logaddexp(0, z))).sum()
        g = X.T @ (w * (y - expit(z)))
        return -ll / n + 0.5 * (pen * b * b).sum() / n, -g / n + pen * b / n

    b0 = np.zeros(p)
    pb = np.clip((w * y).sum() / w.sum(), 1e-4, 1 - 1e-4)
    b0[0] = np.log(pb / (1 - pb))
    return minimize(f, b0, jac=True, method="L-BFGS-B", options={"maxiter": 300}).x


# ------------------------------------------------------------------ retooled Hawkes (discrete hourly, linear, two-exponential kernels)
# lambda_t = base_t * (zeta + sum_i g_i Q_i(t)) + sched_t * sum_i a_i S_i(t),      base_t = L_slow(t-1) * s_a(how_t)
#   S_i(t) = sum_{j>=1} exp(-beta_i (j-1)) m_{t-j},  m = own count + eta * (extra thread posts)   (mark: thread size)
#   Q_i(t) = sum_{j>=1} exp(-rho_i (j-1)) z_{t-j},  z = relative excess of the leave-one-out Congress count over its expectation
#            (Congress as a second, exogenous process; its lagged excess raises the account's intensity in proportion to its baseline)
# scheduler accounts get own excitation scaled by exp(kappa_s). Observation: NB2 with dispersion alpha.
HK_NAMES = ["logit_zeta", "log_a1", "log_a2", "log_b1", "log_b2", "log_g1", "log_g2", "log_r1", "log_r2", "log_eta", "kappa_s",
            "log_alpha", "logit_w_fast"]
HK_BOUNDS = [(-4, 6), (-9, 1), (-9, 1), (np.log(0.5), np.log(6)), (np.log(0.004), np.log(0.5)), (-9, 1), (-9, 1),
             (np.log(0.5), np.log(6)), (np.log(0.01), np.log(0.5)), (-5, 2), (-2, 2), (-4, 3), (-5, 5)]
HK_INIT = np.array([2.0, np.log(0.05), np.log(0.01), np.log(1.5), np.log(0.05), np.log(0.05), np.log(0.02), np.log(1.5),
                    np.log(0.1), np.log(0.1), 0.0, 0.0, 0.0])
ZCLIP = (-1.0, 5.0)


def hk_unpack(th):
    zeta = expit(th[0])
    a = np.exp(th[1:3])
    b = np.exp(th[3:5])
    g = np.exp(th[5:7])
    rho = np.exp(th[7:9])
    eta = np.exp(th[9])
    ks = th[10]
    alpha = np.exp(th[11])
    w = expit(th[12])
    return zeta, a, b, g, rho, eta, ks, alpha, w


def cong_z(C, Y0_rows, Lc_prev, Ls_prev, prof_rows_how, cprof_how):
    """Relative excess of the leave-one-out Congress count over its expectation, using only the level known one hour earlier."""
    exp_loo = np.maximum(Lc_prev * cprof_how - Ls_prev * prof_rows_how, 0.0)
    loo = np.maximum(C - Y0_rows, 0.0)
    return np.clip((loo - exp_loo) / np.maximum(exp_loo, 0.5), *ZCLIP)


def hk_states(m, z, b, rho):
    S = [lfilter([0.0, 1.0], [1.0, -np.exp(-bi)], m, axis=1) for bi in b]
    Q = [lfilter([0.0, 1.0], [1.0, -np.exp(-ri)], z, axis=1) for ri in rho]
    return S, Q


def hk_lambda(th, base, m, z, sched):
    zeta, a, b, g, rho, eta, ks, alpha, w = hk_unpack(th)
    S, Q = hk_states(m, z, b, rho)
    sm = np.exp(ks * sched)[:, None]
    lam = base * np.maximum(zeta + g[0] * Q[0] + g[1] * Q[1], 0.02) + sm * (a[0] * S[0] + a[1] * S[1])
    return np.maximum(lam, 1e-8)


def hk_branching(th):
    zeta, a, b, g, rho, eta, ks, alpha, w = hk_unpack(th)
    return float((a / (1 - np.exp(-b))).sum())


def hk_fit(panel: Panel, rows, t_lo, t_hi, Ls, Lf, Lc, prof, cprof, sched, th0=None, maxiter=40):
    """Fit pooled Hawkes parameters on the hourly series of accounts `rows` over hours [t_lo, t_hi) by NB maximum likelihood."""
    from scipy.optimize import minimize
    th0 = HK_INIT.copy() if th0 is None else th0.copy()
    sl = slice(t_lo, t_hi)
    pv = slice(t_lo - 1, t_hi - 1)
    Y = panel.Y[rows, sl].astype(np.float64)
    valid = np.isfinite(Y)
    Y0 = np.nan_to_num(Y)
    ex = np.maximum(np.nan_to_num(panel.P[rows, sl].astype(np.float64)) - Y0, 0)
    how = panel.how[sl]
    Lp = Ls[rows, pv].astype(np.float64)
    lLs, lLf = np.log(np.maximum(Lp, 1e-4)), np.log(np.maximum(Lf[rows, pv].astype(np.float64), 1e-4))
    pr = prof[rows][:, how]
    z = cong_z(panel.C[sl][None, :], Y0, Lc[pv][None, :], Lp, prof[rows][:, how], cprof[how][None, :])
    sc = sched[rows]

    def pen_obj(th):
        m = Y0 + np.exp(th[9]) * ex
        wf = expit(th[12])
        base = np.exp((1 - wf) * lLs + wf * lLf) * pr
        lam = hk_lambda(th, base, m, z, sc)
        nll = -F_nbll(Y[valid], lam[valid], np.exp(th[11])) / valid.sum()
        return nll + 50.0 * max(0.0, hk_branching(th) - 0.95) ** 2

    res = minimize(pen_obj, th0, method="L-BFGS-B", bounds=HK_BOUNDS, options={"maxiter": maxiter, "eps": 1e-4})
    return res.x, float(res.fun)


def F_nbll(y, mu, alpha):
    return nb_logpmf(y, mu, alpha).sum()


def hk_forecast(th, panel: Panel, ai, t, Ls, Lf, Lc, prof, cprof, phi, sched, mark_ratio, H=168):
    """Multi-step expected hourly intensities lambda_{t+1..t+H} for forecast rows (account index ai, time index t, information
    through t only). Future own counts are replaced by their expectations (exact for a linear Hawkes); future Congress excess decays
    as the last-3-hour average excess times phi^k. Returns (n,H)."""
    zeta, a, b, g, rho, eta, ks, alpha, w = hk_unpack(th)
    n = len(ai)
    W = 24 * 14
    idx = np.clip(t[:, None] - np.arange(W)[None, :], 1, panel.T - 1)
    y_w = panel.Y0[ai[:, None], idx]
    p_w = np.nan_to_num(panel.P[ai[:, None], idx])
    m_w = y_w + eta * np.maximum(p_w - y_w, 0)
    z_w = cong_z(panel.C[idx], y_w, Lc[idx - 1], Ls[ai[:, None], idx - 1], prof[ai[:, None], panel.how[idx]], cprof[panel.how[idx]])
    j = np.arange(W)[None, :]
    S = [(m_w * np.exp(-bi * j)).sum(1) for bi in b]
    Q = [(z_w * np.exp(-ri * j)).sum(1) for ri in rho]
    dev = z_w[:, :3].mean(1)
    Lt = np.exp((1 - w) * np.log(np.maximum(Ls[ai, t].astype(np.float64), 1e-4)) + w * np.log(np.maximum(Lf[ai, t].astype(np.float64), 1e-4)))
    sm = np.exp(ks * sched[ai])
    how_f = (panel.how[t][:, None] + np.arange(1, H + 1)[None, :]) % 168
    base = Lt[:, None] * prof[ai[:, None], how_f]
    mm = 1.0 + eta * mark_ratio[ai]
    ea = [np.exp(-bi) for bi in b]
    er = [np.exp(-ri) for ri in rho]
    lam = np.empty((n, H))
    for k in range(H):
        l = base[:, k] * np.maximum(zeta + g[0] * Q[0] + g[1] * Q[1], 0.02) + sm * (a[0] * S[0] + a[1] * S[1])
        l = np.maximum(l, 1e-8)
        lam[:, k] = l
        mexp = l * mm
        zf = dev * phi ** (k + 1)
        S = [ea[0] * S[0] + mexp, ea[1] * S[1] + mexp]
        Q = [er[0] * Q[0] + zf, er[1] * Q[1] + zf]
    return lam
