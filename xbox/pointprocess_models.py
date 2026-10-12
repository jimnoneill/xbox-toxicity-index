"""Candidate point-process models for posting times, with likelihood, AIC/BIC and exact PIT residuals.

Time unit: minutes since the first one-hour slot. Every model is scored by the log-likelihood of the
event times CONDITIONAL ON THE FIRST EVENT, over [t_1, T] (density per minute, survival term to T), so
log-likelihoods, AIC and BIC are comparable across models. Each model returns u_i in (0,1), i = 2..n,
which are i.i.d. Uniform(0,1) under the model (probability integral transform of the gap given the past
and, for hidden-state models, the filtered state). For the Hawkes reference u_i = 1 - exp(-Lambda increment),
n values from slot0 (identical to the pipeline's time-rescaling KS).

Baselines: 'hw' = 48-bin hour-of-week profile P (occupancy mean one, 47 free parameters, plug-in Poisson MLE
with 0.5 pseudo-count). Hawkes models use month x hour-of-week levels (nm + 47), as in the pipeline.
All formulas are listed in FORMULAS (bottom of file).
"""
from __future__ import annotations

import numpy as np
from numba import njit
from scipy import optimize, special
from scipy.stats import kstest

from . import hawkes as hk

SLOT = 60.0
TINY = 1e-12


# ====================================================================== data container
class Data:
    """One entity's events (minutes, jittered) and the slot calendar (local hour-of-week bin per slot)."""

    def __init__(self, t, slot_month, slot_bin, nm):
        self.t = np.sort(np.asarray(t, dtype="float64"))
        self.slot_month = np.asarray(slot_month, dtype="int64")
        self.slot_bin = np.asarray(slot_bin, dtype="int64")
        self.nm = int(nm)
        self.ns = len(self.slot_month)
        self.T = self.ns * SLOT
        self.n = len(self.t)
        self.slot = np.minimum((self.t // SLOT).astype("int64"), self.ns - 1)
        self._hw = None
        self._hdes = None

    # -- design for the Hawkes code
    def hawkes_design(self):
        if self._hdes is None:
            occ = np.zeros((self.nm, 48))
            np.add.at(occ, (self.slot_month, self.slot_bin), SLOT)
            self._hdes = dict(t=self.t, ev_m=self.slot_month[self.slot], ev_k=self.slot_bin[self.slot], occ=occ,
                              T=self.T, slot_month=self.slot_month, slot_bin=self.slot_bin)
        return self._hdes

    # -- hour-of-week profile and the time change
    def hw(self):
        if self._hw is None:
            cnt = np.bincount(self.slot_bin[self.slot], minlength=48).astype(float)
            occ = np.bincount(self.slot_bin, minlength=48) * SLOT
            P = (cnt + 0.5) / np.maximum(occ, 1e-9)
            P = np.where(occ > 0, P, 0.0)
            P = P / (np.sum(P * occ) / occ.sum())          # occupancy-weighted mean one
            P = np.maximum(P, 1e-6)
            rate = P[self.slot_bin]                         # per slot
            cum = np.concatenate([[0.0], np.cumsum(rate * SLOT)])
            s = self.slot
            lam_t = cum[s] + rate[s] * (self.t - s * SLOT)
            self._hw = dict(P=P, rate=rate, cum=cum, lam_t=lam_t, logP_t=np.log(rate[s]), lam_T=cum[-1])
        return self._hw

    def lam_inv(self, x):
        """Inverse of the time change Lambda(t) = integral of P over [0, t] (vectorised)."""
        h = self.hw()
        cum, rate = h["cum"], h["rate"]
        s = np.clip(np.searchsorted(cum, x, side="right") - 1, 0, self.ns - 1)
        return s * SLOT + (x - cum[s]) / rate[s]


def calendar_from_cache(c):
    return Data(c["t"], c["slot_month"], c["slot_bin"], c["nm"])


def sim_to_data(tmin, tmpl, seed, round_seconds=True):
    """Round simulated times to whole seconds then add uniform sub-second jitter (as the pipeline does)."""
    rng = np.random.default_rng(seed)
    tm = np.asarray(tmin, dtype="float64")
    tm = tm[(tm >= 0) & (tm < tmpl.T)]
    if round_seconds:
        tm = np.floor(tm * 60.0) / 60.0
    tm = tm + rng.uniform(0, 1.0 / 60.0, size=len(tm))
    tm = tm[tm < tmpl.T]
    return Data(tm, tmpl.slot_month, tmpl.slot_bin, tmpl.nm)


# ====================================================================== GOF helpers
def ks_u(u):
    k = kstest(u, "uniform")
    return float(k.statistic), float(k.pvalue)


def pit_mad(u):
    x = np.sort(u)
    n = len(x)
    return float(np.mean(np.abs(x - (np.arange(1, n + 1) - 0.5) / n)))


def ks_stat_fast(u):
    x = np.sort(u)
    n = len(x)
    i = np.arange(1, n + 1)
    return float(max(np.max(i / n - x), np.max(x - (i - 1) / n)))


class Res(dict):
    """ll (conditional on first event), k (parameters), u (PIT values), params (dict), state (for simulation)."""


def finish(res, n_obs):
    res["aic"] = 2 * res["k"] - 2 * res["ll"]
    res["bic"] = res["k"] * np.log(n_obs) - 2 * res["ll"]
    res["D"], res["p"] = ks_u(res["u"])
    res["mad"] = pit_mad(res["u"])
    return res


# ====================================================================== 1. exponential Hawkes (reference)
def fit_hawkes_exp(data):
    d = data.hawkes_design()
    f = hk.fit(d)
    rate_slot = f["L"][d["slot_month"]] * f["P"][d["slot_bin"]]
    cum = np.concatenate([[0.0], np.cumsum(rate_slot * SLOT)])
    s = data.slot
    mcum = cum[s] + rate_slot[s] * (data.t - s * SLOT)
    inc = hk._increments(data.t, mcum, 0.0, f["eta"], f["beta"])
    first = np.log(rate_slot[s[0]]) - mcum[0]
    ll = f["ll"] - first
    u = -np.expm1(-inc)
    return Res(ll=float(ll), k=data.nm + 47 + 2, u=u,
               params=dict(eta=f["eta"], inv_beta_min=f["inv_beta_min"]),
               state=dict(L=f["L"], P=f["P"], eta=f["eta"], beta=f["beta"]))


def sim_hawkes_exp(data, res, seed):
    st = res["state"]
    mu = st["L"][data.slot_month] * st["P"][data.slot_bin]
    return hk.simulate_hawkes(mu, st["eta"], st["beta"], seed)


# ====================================================================== 5. Hawkes, sum of exponentials on a fixed grid
SOE_INV_BETA = np.array([1 / 12, 0.5, 3.0, 20.0, 120.0, 720.0, 4320.0])   # 5 s ... 3 d, in minutes
SOE_BETA = 1.0 / SOE_INV_BETA


@njit(cache=True)
def _estep_soe(t, ev_m, ev_k, L, P, eta, beta, cnt, sq):
    n = t.shape[0]
    J = beta.shape[0]
    cnt[:, :] = 0.0
    sq[:] = 0.0
    R = np.zeros(J)
    ll = 0.0
    prev = t[0]
    for i in range(n):
        if i > 0:
            d = t[i] - prev
            for j in range(J):
                R[j] = np.exp(-beta[j] * d) * (1.0 + R[j])
        mu = max(L[ev_m[i]] * P[ev_k[i]], 1e-300)
        ex = 0.0
        for j in range(J):
            ex += eta[j] * beta[j] * R[j]
        lam = mu + ex
        ll += np.log(lam)
        cnt[ev_m[i], ev_k[i]] += mu / lam
        for j in range(J):
            sq[j] += eta[j] * beta[j] * R[j] / lam
        prev = t[i]
    return ll


@njit(cache=True)
def _em_soe(t, ev_m, ev_k, occ, T, L, P, eta, beta, max_iter, tol):
    M, K = occ.shape
    n = t.shape[0]
    J = beta.shape[0]
    cnt = np.zeros((M, K))
    sq = np.zeros(J)
    Gs = np.zeros(J)
    for j in range(J):
        for i in range(n):
            Gs[j] += 1.0 - np.exp(-beta[j] * (T - t[i]))
    llp = -1e300
    ll = -1e300
    for it in range(max_iter):
        ll_ev = _estep_soe(t, ev_m, ev_k, L, P, eta, beta, cnt, sq)
        comp = 0.0
        for m in range(M):
            for k in range(K):
                comp += L[m] * P[k] * occ[m, k]
        ll = ll_ev - comp
        for j in range(J):
            ll -= eta[j] * Gs[j]
        if abs(ll - llp) < tol * n:
            break
        llp = ll
        s = 0.0
        for j in range(J):
            eta[j] = sq[j] / Gs[j] if Gs[j] > 0 else 0.0
            s += eta[j]
        if s > 0.999:
            for j in range(J):
                eta[j] *= 0.999 / s
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
        ws = 0.0
        for m in range(M):
            for k in range(K):
                tot += P[k] * occ[m, k]
                ws += occ[m, k]
        sc = tot / ws if ws > 0 else 1.0
        if sc > 0:
            for k in range(K):
                P[k] /= sc
            for m in range(M):
                L[m] *= sc
    # final ll
    ll_ev = _estep_soe(t, ev_m, ev_k, L, P, eta, beta, cnt, sq)
    comp = 0.0
    for m in range(M):
        for k in range(K):
            comp += L[m] * P[k] * occ[m, k]
    ll = ll_ev - comp
    for j in range(J):
        ll -= eta[j] * Gs[j]
    return ll


@njit(cache=True)
def _incr_soe(t, mcum, eta, beta):
    n = t.shape[0]
    J = beta.shape[0]
    out = np.empty(n)
    a = np.zeros(J)
    prev_t = 0.0
    prev_m = 0.0
    for i in range(n):
        inc = mcum[i] - prev_m
        if i > 0:
            d = t[i] - prev_t
            for j in range(J):
                e = np.exp(-beta[j] * d)
                inc += eta[j] * a[j] * (1.0 - e)
                a[j] = a[j] * e + 1.0
        else:
            for j in range(J):
                a[j] = 1.0
        out[i] = inc
        prev_t = t[i]
        prev_m = mcum[i]
    return out


def fit_hawkes_soe(data):
    d = data.hawkes_design()
    f1 = hk.fit(d)
    L, P = f1["L"].copy(), f1["P"].copy()
    eta = np.full(len(SOE_BETA), max(f1["eta"], 0.05) / len(SOE_BETA))
    ll = _em_soe(d["t"], d["ev_m"], d["ev_k"], d["occ"], d["T"], L, P, eta, SOE_BETA.copy(), 1500, 1e-9)
    rate_slot = L[d["slot_month"]] * P[d["slot_bin"]]
    cum = np.concatenate([[0.0], np.cumsum(rate_slot * SLOT)])
    s = data.slot
    mcum = cum[s] + rate_slot[s] * (data.t - s * SLOT)
    inc = _incr_soe(data.t, mcum, eta, SOE_BETA)
    first = np.log(rate_slot[s[0]]) - mcum[0]
    return Res(ll=float(ll - first), k=data.nm + 47 + len(SOE_BETA), u=-np.expm1(-inc),
               params=dict(eta_total=float(eta.sum()), **{f"eta_{i}": float(e) for i, e in enumerate(eta)}),
               state=dict(L=L, P=P, eta=eta.copy()))


@njit(cache=True)
def _thin_soe(mu_slot, eta, beta, T, seed, cap):
    np.random.seed(seed)
    J = beta.shape[0]
    mu_max = mu_slot.max()
    S = np.zeros(J)
    ev = np.empty(cap)
    n = 0
    t = 0.0
    while True:
        tot = 0.0
        for j in range(J):
            tot += S[j]
        lam_bar = mu_max + tot
        w = -np.log(1.0 - np.random.random()) / lam_bar
        tn = t + w
        if tn >= T:
            break
        tot = 0.0
        for j in range(J):
            S[j] *= np.exp(-beta[j] * w)
            tot += S[j]
        lam = mu_slot[int(tn // 60.0)] + tot
        if np.random.random() * lam_bar <= lam:
            if n == cap:
                ev2 = np.empty(cap * 2)
                ev2[:n] = ev[:n]
                ev = ev2
                cap *= 2
            ev[n] = tn
            n += 1
            for j in range(J):
                S[j] += eta[j] * beta[j]
        t = tn
    return ev[:n]


def sim_hawkes_soe(data, res, seed):
    st = res["state"]
    mu = st["L"][data.slot_month] * st["P"][data.slot_bin]
    cap = int(mu.sum() * SLOT / max(1e-3, 1 - st["eta"].sum()) * 1.5) + 1000
    return _thin_soe(mu.astype("float64"), st["eta"], SOE_BETA, data.T, int(seed), cap)


# ====================================================================== 2. renewal processes (gap distributions)
class Dist:
    """Gap distribution on a scale-free variable. Subclasses: logpdf(x), sf(x), rvs(rng, m)."""
    k = 0


class LogNormal(Dist):
    name = "lognormal"

    def __init__(self, x):
        lg = np.log(np.maximum(x, TINY))
        self.m, self.s, self.k = lg.mean(), max(lg.std(), 1e-3), 2

    def logpdf(self, x):
        lg = np.log(np.maximum(x, TINY))
        return -lg - np.log(self.s) - 0.5 * np.log(2 * np.pi) - 0.5 * ((lg - self.m) / self.s) ** 2

    def sf(self, x):
        return special.ndtr(-(np.log(np.maximum(x, TINY)) - self.m) / self.s)

    def rvs(self, rng, m):
        return np.exp(self.m + self.s * rng.standard_normal(m))

    def params(self):
        return dict(mu=self.m, sigma=self.s)


class Weibull(Dist):
    name = "weibull"

    def __init__(self, x):
        x = np.maximum(x, TINY)
        lg = np.log(x)

        def g(k):
            xk = np.exp(k * (lg - lg.max()))
            return (xk * lg).sum() / xk.sum() - 1.0 / k - lg.mean()
        try:
            self.c = optimize.brentq(g, 0.02, 30.0, xtol=1e-8)
        except ValueError:
            self.c = 0.02 if g(0.02) < 0 else 30.0
        self.lam = np.exp(lg.max()) * (np.mean(np.exp(self.c * (lg - lg.max())))) ** (1.0 / self.c)
        self.k = 2

    def logpdf(self, x):
        x = np.maximum(x, TINY)
        return np.log(self.c) - np.log(self.lam) + (self.c - 1) * (np.log(x) - np.log(self.lam)) - (x / self.lam) ** self.c

    def sf(self, x):
        return np.exp(-(np.maximum(x, TINY) / self.lam) ** self.c)

    def rvs(self, rng, m):
        return self.lam * (-np.log(1 - rng.random(m))) ** (1.0 / self.c)

    def params(self):
        return dict(shape=self.c, scale=self.lam)


class Gamma(Dist):
    name = "gamma"

    def __init__(self, x):
        x = np.maximum(x, TINY)
        s = np.log(x.mean()) - np.log(x).mean()
        a = (3 - s + np.sqrt((s - 3) ** 2 + 24 * s)) / (12 * s) if s > 1e-9 else 1e3
        for _ in range(60):
            f = np.log(a) - special.digamma(a) - s
            fp = 1.0 / a - special.polygamma(1, a)
            an = a - f / fp
            if an <= 0:
                an = a / 2
            if abs(an - a) < 1e-10 * a:
                a = an
                break
            a = an
        self.a, self.th, self.k = a, x.mean() / a, 2

    def logpdf(self, x):
        x = np.maximum(x, TINY)
        return (self.a - 1) * np.log(x) - x / self.th - special.gammaln(self.a) - self.a * np.log(self.th)

    def sf(self, x):
        return special.gammaincc(self.a, np.maximum(x, TINY) / self.th)

    def rvs(self, rng, m):
        return rng.gamma(self.a, self.th, m)

    def params(self):
        return dict(shape=self.a, scale=self.th)


def mix_exp_em(Z, logjac, w, r, iters=400, tol=1e-9):
    """EM for f(i) = sum_j w_j r_j exp(-r_j Z_ij) exp(logjac_ij). Z, logjac: (m, k). Returns w, r, ll."""
    m, k = Z.shape
    llp = -np.inf
    for _ in range(iters):
        lf = np.log(w)[None, :] + np.log(r)[None, :] - r[None, :] * Z + logjac
        mx = lf.max(axis=1, keepdims=True)
        e = np.exp(lf - mx)
        sm = e.sum(axis=1, keepdims=True)
        ll = float((mx[:, 0] + np.log(sm[:, 0])).sum())
        if abs(ll - llp) < tol * m:
            break
        llp = ll
        resp = e / sm
        nj = resp.sum(axis=0)
        w = np.maximum(nj / m, 1e-8)
        w /= w.sum()
        r = np.clip(nj / np.maximum((resp * Z).sum(axis=0), 1e-300), 1e-9, 1e9)
    return w, r, ll


def _he_inits(x, k):
    xs = np.sort(x)
    chunks = np.array_split(xs, k)
    out = [np.array([1.0 / max(c.mean(), TINY) for c in chunks])]
    q = np.quantile(xs, [0.02, 0.98])
    out.append(1.0 / np.geomspace(max(q[0], TINY), max(q[1] * 3, 10 * TINY), k))
    out.append(1.0 / np.geomspace(max(np.quantile(xs, 0.2), TINY), max(xs[-1], 10 * TINY), k))
    return [(np.full(k, 1.0 / k), np.sort(r)[::-1].copy() if False else r.copy()) for r in out]


class HyperExp(Dist):
    def __init__(self, x, kc):
        self.kc = kc
        Z = np.repeat(x[:, None], kc, axis=1)
        best = None
        for w0, r0 in _he_inits(x, kc):
            w, r, ll_ = mix_exp_em(Z, np.zeros_like(Z), w0, r0, iters=600)
            if best is None or ll_ > best[2]:
                best = (w, r, ll_)
        w, r, _ = best
        o = np.argsort(r)
        self.w, self.r = w[o], r[o]
        self.k = 2 * kc - 1
        self.name = f"hyperexp{kc}"

    def logpdf(self, x):
        return np.log((self.w[None, :] * self.r[None, :] * np.exp(-self.r[None, :] * x[:, None])).sum(axis=1) + 1e-300)

    def sf(self, x):
        return (self.w[None, :] * np.exp(-self.r[None, :] * x[:, None])).sum(axis=1)

    def rvs(self, rng, m):
        j = rng.choice(len(self.w), size=m, p=self.w)
        return rng.exponential(1.0 / self.r[j])

    def params(self):
        d = {}
        for i in range(self.kc):
            d[f"w{i}"] = float(self.w[i])
            d[f"rate{i}"] = float(self.r[i])
        return d


def make_dist(fam, x):
    return {"ln": lambda: LogNormal(x), "wb": lambda: Weibull(x), "gm": lambda: Gamma(x),
            "he1": lambda: HyperExp(x, 1), "he2": lambda: HyperExp(x, 2), "he3": lambda: HyperExp(x, 3)}[fam]()


def _renewal_gaps(data, circ):
    if circ:
        h = data.hw()
        tau = h["lam_t"]
        tauT = h["lam_T"]
        logjac = h["logP_t"][1:]
        base_k = 47
    else:
        tau = data.t
        tauT = data.T
        logjac = np.zeros(data.n - 1)
        base_k = 0
    g = np.maximum(np.diff(tau), TINY)
    c = max(tauT - tau[-1], TINY)
    return g, c, logjac, base_k


def fit_renewal(data, fam, circ):
    g, c, logjac, base_k = _renewal_gaps(data, circ)
    sc = g.mean()
    x = g / sc
    D = make_dist(fam, x)
    lp = D.logpdf(x) - np.log(sc)
    ll = float(lp.sum() + logjac.sum() + np.log(max(D.sf(np.array([c / sc]))[0], 1e-300)))
    u = 1.0 - D.sf(x)
    prm = D.params()
    return Res(ll=ll, k=D.k + base_k, u=u, params=prm, state=dict(dist=D, sc=sc, circ=circ))


def sim_renewal(data, res, seed):
    rng = np.random.default_rng(seed)
    st = res["state"]
    D, sc, circ = st["dist"], st["sc"], st["circ"]
    horizon = data.hw()["lam_T"] if circ else data.T
    # mean gap in scale-free units is about 1; draw generously
    m0 = int(horizon / sc * 1.3) + 1000
    out = []
    tot = 0.0
    while tot < horizon:
        gs = D.rvs(rng, m0) * sc
        cs = tot + np.cumsum(gs)
        out.append(cs)
        tot = cs[-1]
    tau = np.concatenate(out)
    # first event: start from a stationary-ish draw (condition on first event is dropped in scoring)
    tau = tau[tau < horizon]
    return data.lam_inv(tau) if circ else tau


# ====================================================================== 3. cascading NHPP (non-overlapping sessions)
def fit_cascade(data, kc):
    """Sessions start by an NHPP with rate s * P(t) measured from the last post (clock paused while a session is
    running); after each post the next post is either a within-session post (probability w_j, Exp(r_j) real-time
    gap, j = 1..kc-1) or the start of a new session (probability w_0, gap Exp(s) in time-changed units).
    kc = 2 is the geometric-session-size model; kc = 3 has two within-session scales."""
    h = data.hw()
    t = data.t
    d_real = np.maximum(np.diff(t), TINY)
    d_lam = np.maximum(np.diff(h["lam_t"]), TINY)
    c_real = max(data.T - t[-1], TINY)
    c_lam = max(h["lam_T"] - h["lam_t"][-1], TINY)
    m = len(d_real)
    Z = np.empty((m, kc))
    J = np.zeros((m, kc))
    Z[:, 0] = d_lam
    J[:, 0] = h["logP_t"][1:]
    for j in range(1, kc):
        Z[:, j] = d_real
    # init: split gaps by rank of real gap; slow component from the slowest chunk (time-changed units)
    order = np.argsort(d_real)
    chunks = np.array_split(order, kc)
    r = np.empty(kc)
    r[0] = 1.0 / max(d_lam[chunks[-1]].mean(), TINY)
    for j in range(1, kc):
        r[j] = 1.0 / max(d_real[chunks[j - 1]].mean(), TINY)
    w = np.full(kc, 1.0 / kc)
    starts = [(w, r)]
    r2 = r.copy()
    r2[1:] = r2[1:] * 3.0
    starts.append((w.copy(), r2))
    if kc == 3:
        r3 = r.copy()
        r3[1] = r[1] / 20.0
        r3[2] = r[2] * 5.0
        starts.append((w.copy(), r3))
    best = None
    for w0, r0 in starts:
        w_, r_, ll_ = mix_exp_em(Z, J, w0, r0, iters=600)
        if best is None or ll_ > best[2]:
            best = (w_, r_, ll_)
    w, r, ll = best
    Zc = np.array([c_lam] + [c_real] * (kc - 1))
    surv_c = float((w * np.exp(-r * Zc)).sum())
    ll = ll + np.log(max(surv_c, 1e-300))
    S = (w[None, :] * np.exp(-r[None, :] * Z)).sum(axis=1)
    u = 1.0 - S
    p_fast = 1.0 - w[0]
    prm = dict(session_rate_per_day=float(r[0] * 1440.0), posts_per_session=float(1.0 / max(w[0], 1e-9)),
               within_rate_per_min=float(np.sum(w[1:] * r[1:]) / max(w[1:].sum(), 1e-12)),
               idle_gap_share=float(w[0]), w_idle=float(w[0]))
    for j in range(1, kc):
        prm[f"w{j}"] = float(w[j])
        prm[f"rate{j}"] = float(r[j])
    return Res(ll=float(ll), k=(2 * kc - 1) + 47, u=u, params=prm, state=dict(w=w, r=r, kc=kc))


@njit(cache=True)
def _sim_cascade(w, r, cum, rate, T, seed, cap):
    np.random.seed(seed)
    kc = w.shape[0]
    ns = rate.shape[0]
    ev = np.empty(cap)
    n = 0
    t = 0.0
    lam_end = cum[ns]
    lam = 0.0
    first = True
    while True:
        u = np.random.random()
        j = 0
        acc = w[0]
        while j < kc - 1 and u > acc:
            j += 1
            acc += w[j]
        if first or j == 0:
            lam += -np.log(1.0 - np.random.random()) / r[0]
            if lam >= lam_end:
                break
            s = np.searchsorted(cum, lam, side="right") - 1
            if s > ns - 1:
                s = ns - 1
            t = s * 60.0 + (lam - cum[s]) / rate[s]
            first = False
        else:
            t = t + (-np.log(1.0 - np.random.random()) / r[j])
            if t >= T:
                break
            s = int(t // 60.0)
            lam = cum[s] + rate[s] * (t - s * 60.0)
        if n == cap:
            ev2 = np.empty(cap * 2)
            ev2[:n] = ev[:n]
            ev = ev2
            cap *= 2
        ev[n] = t
        n += 1
    return ev[:n]


def sim_cascade(data, res, seed):
    h = data.hw()
    st = res["state"]
    return _sim_cascade(st["w"].astype("float64"), st["r"].astype("float64"), h["cum"], h["rate"], data.T, int(seed), 100000)


# ====================================================================== 4. Markov-modulated Poisson process (HMM on events)
@njit(cache=True)
def _mm(A, B, out, k):
    for i in range(k):
        for j in range(k):
            s = 0.0
            for l in range(k):
                s += A[i, l] * B[l, j]
            out[i, j] = s


@njit(cache=True)
def _expm(A, out, k):
    """exp(A) for a small matrix by scaling and squaring with a Taylor series."""
    nrm = 0.0
    for j in range(k):
        c = 0.0
        for i in range(k):
            c += abs(A[i, j])
        if c > nrm:
            nrm = c
    sq = 0
    if nrm > 0.25:
        sq = int(np.ceil(np.log2(nrm / 0.25)))
    sc = 2.0 ** sq
    B = A / sc
    term = np.eye(k)
    res = np.eye(k)
    tmp = np.empty((k, k))
    for m in range(1, 14):
        _mm(term, B, tmp, k)
        for i in range(k):
            for j in range(k):
                term[i, j] = tmp[i, j] / m
                res[i, j] += term[i, j]
    for _ in range(sq):
        _mm(res, res, tmp, k)
        for i in range(k):
            for j in range(k):
                res[i, j] = tmp[i, j]
    for i in range(k):
        for j in range(k):
            out[i, j] = res[i, j]


@njit(cache=True)
def _vm(a, E, k, out):
    for j in range(k):
        s = 0.0
        for i in range(k):
            s += a[i] * E[i, j]
        out[j] = s


@njit(cache=True)
def _mmpp_forward(t, T, slot_bin, Pbin, lam, Q, circ, want_u):
    k = lam.shape[0]
    n = t.shape[0]
    nb = Pbin.shape[0]
    # per-bin generator minus emission
    A = np.zeros((nb, k, k))
    for b in range(nb):
        mult = Pbin[b] if circ else 1.0
        for i in range(k):
            tot = 0.0
            for j in range(k):
                if i != j:
                    q = Q[i, j] * (mult if i == 0 else 1.0)
                    A[b, i, j] = q
                    tot += q
            A[b, i, i] = -tot - lam[i]
    E60 = np.zeros((nb, k, k))
    tmpm = np.empty((k, k))
    if circ:
        for b in range(nb):
            _expm(A[b] * 60.0, tmpm, k)
            E60[b] = tmpm
    # stationary distribution of Q at P = 1
    Q0 = np.zeros((k, k))
    for i in range(k):
        tot = 0.0
        for j in range(k):
            if i != j:
                Q0[i, j] = Q[i, j]
                tot += Q[i, j]
        Q0[i, i] = -tot
    big = np.empty((k, k))
    _expm(Q0 * 1e5, big, k)
    alpha = np.empty(k)
    z = 0.0
    for s in range(k):
        alpha[s] = big[0, s] * lam[s]
        z += alpha[s]
    for s in range(k):
        alpha[s] /= z
    u = np.empty(n - 1) if want_u else np.empty(1)
    ll = 0.0
    nxt = np.empty(k)
    E = np.empty((k, k))
    A1 = np.empty((k, k))
    for i in range(1, n + 1):
        a_t = t[i - 1]
        b_t = t[i] if i < n else T
        logsc = 0.0
        if circ:
            pos = a_t
            while pos < b_t - 1e-12:
                s = int(pos // 60.0)
                endp = min(b_t, (s + 1) * 60.0)
                Ln = endp - pos
                bi = slot_bin[s]
                if Ln >= 60.0 - 1e-9:
                    _vm(alpha, E60[bi], k, nxt)
                else:
                    for x in range(k):
                        for y in range(k):
                            A1[x, y] = A[bi, x, y] * Ln
                    _expm(A1, E, k)
                    _vm(alpha, E, k, nxt)
                sm = 0.0
                for x in range(k):
                    sm += nxt[x]
                if sm < 1e-300:
                    sm = 1e-300
                logsc += np.log(sm)
                for x in range(k):
                    alpha[x] = nxt[x] / sm
                pos = endp
        else:
            for x in range(k):
                for y in range(k):
                    A1[x, y] = A[0, x, y] * (b_t - a_t)
            _expm(A1, E, k)
            _vm(alpha, E, k, nxt)
            sm = 0.0
            for x in range(k):
                sm += nxt[x]
            if sm < 1e-300:
                sm = 1e-300
            logsc = np.log(sm)
            for x in range(k):
                alpha[x] = nxt[x] / sm
        if i < n:
            f = 0.0
            for x in range(k):
                f += alpha[x] * lam[x]
            if f < 1e-300:
                f = 1e-300
            ll += logsc + np.log(f)
            if want_u:
                u[i - 1] = -np.expm1(logsc)
            for x in range(k):
                alpha[x] = alpha[x] * lam[x] / f
        else:
            ll += logsc
    return ll, u


def _unpack(theta, k):
    lam = np.exp(theta[:k])
    Q = np.zeros((k, k))
    c = k
    for i in range(k):
        for j in range(k):
            if i != j:
                Q[i, j] = np.exp(theta[c])
                c += 1
    return lam, Q


def _pack(lam, Q):
    k = len(lam)
    th = list(np.log(lam))
    for i in range(k):
        for j in range(k):
            if i != j:
                th.append(np.log(Q[i, j]))
    return np.array(th)


def _mmpp_init(data, k, circ, scale_q, scale_l):
    """Heuristic start from a hyperexponential fit of the gaps: state j emits at the j-th rate."""
    g = np.maximum(np.diff(data.t), TINY)
    sc = g.mean()
    D = HyperExp(g / sc, k)
    r = D.r / sc                      # per minute, ascending
    w = D.w
    lam = np.array(r) * scale_l
    lam[0] = r[0] * 0.05 * scale_l    # idle state emits slowly
    Q = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            if j == i + 1:
                Q[i, j] = r[i] * 1.0
            elif j == i - 1:
                Q[i, j] = r[i] * max(w[j] / max(w[i], 1e-6), 0.05)
            else:
                Q[i, j] = r[min(i, j)] * 0.1
    return lam, np.maximum(Q * scale_q, 1e-6)


def fit_mmpp(data, k, circ, starts=((1.0, 1.0), (0.3, 1.0), (3.0, 1.0)), maxiter=200):
    h = data.hw()
    t, T, sb, Pb = data.t, data.T, data.slot_bin, h["P"]
    Pb_use = Pb if circ else np.ones(48)
    best = None

    def nll(th):
        lam, Q = _unpack(th, k)
        ll, _ = _mmpp_forward(t, T, sb, Pb_use, lam, Q, circ, False)
        return -ll if np.isfinite(ll) else 1e30
    bnds = [(-16.0, 7.0)] * (k + k * (k - 1))
    for sq_, sl_ in starts:
        lam0, Q0 = _mmpp_init(data, k, circ, sq_, sl_)
        th0 = np.clip(_pack(lam0, Q0), -15.5, 6.5)
        try:
            r = optimize.minimize(nll, th0, method="L-BFGS-B", bounds=bnds, options=dict(maxiter=maxiter, maxfun=4000))
        except Exception:
            continue
        if best is None or r.fun < best.fun:
            best = r
    lam, Q = _unpack(best.x, k)
    # order states by emission rate (state 0 must stay the circadian idle state; keep optimiser's labelling,
    # but report sorted summaries)
    ll, u = _mmpp_forward(t, T, sb, Pb_use, lam, Q, circ, True)
    prm = {}
    for i in range(k):
        prm[f"lam{i}"] = float(lam[i])
        for j in range(k):
            if i != j:
                prm[f"q{i}{j}"] = float(Q[i, j])
    # stationary distribution (P = 1)
    Q0 = Q.copy()
    for i in range(k):
        Q0[i, i] = -Q[i].sum()
    w_, v_ = np.linalg.eig(Q0.T)
    pi = np.real(v_[:, np.argmin(np.abs(w_))])
    pi = np.abs(pi) / np.abs(pi).sum()
    for i in range(k):
        prm[f"pi{i}"] = float(pi[i])
    nk = k + k * (k - 1) + (47 if circ else 0)
    return Res(ll=float(ll), k=nk, u=u, params=prm, state=dict(lam=lam, Q=Q, circ=circ, k=k))


@njit(cache=True)
def _sim_mmpp(lam, Q, rate, circ, pmax, T, seed, cap):
    np.random.seed(seed)
    k = lam.shape[0]
    ev = np.empty(cap)
    n = 0
    t = 0.0
    # initial state from the stationary law
    Q0 = np.zeros((k, k))
    for i in range(k):
        tot = 0.0
        for j in range(k):
            if i != j:
                Q0[i, j] = Q[i, j]
                tot += Q[i, j]
        Q0[i, i] = -tot
    big = np.empty((k, k))
    _expm(Q0 * 1e5, big, k)
    u = np.random.random()
    s = 0
    acc = big[0, 0]
    while s < k - 1 and u > acc:
        s += 1
        acc += big[0, s]
    while True:
        mult_max = pmax if circ else 1.0
        tot_q = 0.0
        for j in range(k):
            if j != s:
                tot_q += Q[s, j] * (mult_max if s == 0 else 1.0)
        bound = lam[s] + tot_q
        t += -np.log(1.0 - np.random.random()) / bound
        if t >= T:
            break
        mult = rate[int(t // 60.0)] if (circ and s == 0) else 1.0
        u = np.random.random() * bound
        if u < lam[s]:
            if n == cap:
                ev2 = np.empty(cap * 2)
                ev2[:n] = ev[:n]
                ev = ev2
                cap *= 2
            ev[n] = t
            n += 1
        else:
            u -= lam[s]
            for j in range(k):
                if j == s:
                    continue
                qq = Q[s, j] * mult
                if u < qq:
                    s = j
                    break
                u -= qq
    return ev[:n]


def sim_mmpp(data, res, seed):
    st = res["state"]
    h = data.hw()
    return _sim_mmpp(st["lam"], st["Q"], h["rate"], st["circ"], float(h["rate"].max()), data.T, int(seed), 100000)


# ====================================================================== registry
def _renew(fam, circ):
    return lambda d: fit_renewal(d, fam, circ)


MODELS = {
    "hawkes_exp": (fit_hawkes_exp, sim_hawkes_exp),
    "hawkes_soe": (fit_hawkes_soe, sim_hawkes_soe),
    "nhpp_hw": (_renew("he1", True), sim_renewal),
    "ren_ln": (_renew("ln", False), sim_renewal), "renc_ln": (_renew("ln", True), sim_renewal),
    "ren_wb": (_renew("wb", False), sim_renewal), "renc_wb": (_renew("wb", True), sim_renewal),
    "ren_gm": (_renew("gm", False), sim_renewal), "renc_gm": (_renew("gm", True), sim_renewal),
    "ren_he2": (_renew("he2", False), sim_renewal), "renc_he2": (_renew("he2", True), sim_renewal),
    "ren_he3": (_renew("he3", False), sim_renewal), "renc_he3": (_renew("he3", True), sim_renewal),
    "casc2": (lambda d: fit_cascade(d, 2), sim_cascade),
    "casc3": (lambda d: fit_cascade(d, 3), sim_cascade),
    "mmpp2": (lambda d: fit_mmpp(d, 2, False), sim_mmpp), "mmpp2c": (lambda d: fit_mmpp(d, 2, True), sim_mmpp),
    "mmpp3": (lambda d: fit_mmpp(d, 3, False), sim_mmpp), "mmpp3c": (lambda d: fit_mmpp(d, 3, True), sim_mmpp),
}


def fit_model(name, data):
    res = MODELS[name][0](data)
    return finish(res, data.n)


def simulate(name, data, res, seed):
    return MODELS[name][1](data, res, seed)


FORMULAS = r"""
Notation: events t_1<...<t_n in [0,T], gaps d_i = t_i - t_{i-1}; mu(t) baseline; P(t) hour-of-week profile with
occupancy mean one; Lambda_P(t) = int_0^t P(s) ds; log-likelihoods are conditional on t_1.

H1 exponential Hawkes: lambda(t) = mu(t) + eta*beta*sum_{t_i<t} exp(-beta (t - t_i)), mu = L[month] P[bin].
   logL = sum_i log lambda(t_i) - int_0^T mu dt - eta * sum_i (1 - exp(-beta (T - t_i))).
   Compensator over (t_{i-1}, t_i]: Delta_i = int mu + eta * A_{i-1} * (1 - exp(-beta d_i)),
   A_i = exp(-beta d_i) A_{i-1} + 1 (A_1 = 1). PIT u_i = 1 - exp(-Delta_i).
H5 sum-of-exponentials Hawkes: lambda = mu + sum_j eta_j beta_j sum exp(-beta_j (t - t_i)), beta_j fixed grid
   (5 s ... 3 d). logL = sum log lambda(t_i) - int mu - sum_j eta_j sum_i (1-exp(-beta_j (T - t_i))).
   Delta_i = int mu + sum_j eta_j A^j_{i-1}(1 - exp(-beta_j d_i)).
R renewal with time change tau = Lambda_P(t) (or tau = t): gap g_i = tau_i - tau_{i-1} ~ F, density f:
   logL = sum_{i>=2} [log f(g_i) + log P(t_i)] + log S(tau_T - tau_n); PIT u_i = F(g_i) = 1 - S(g_i).
   lognormal f = exp(-(ln g - m)^2/(2 s^2)) / (g s sqrt(2 pi)); Weibull S = exp(-(g/lam)^c);
   gamma S = Q(a, g/theta); hyperexponential S = sum_j w_j exp(-r_j g), f = sum_j w_j r_j exp(-r_j g).
C cascading NHPP (non-overlapping sessions): after each post the next gap is a within-session gap with prob w_j
   (Exp(r_j) in real time, j>=1) or a new-session wait (prob w_0, density s P(t_i) exp(-s (Lambda_P(t_i)-Lambda_P(t_{i-1})))).
   f(i) = sum_{j>=1} w_j r_j exp(-r_j d_i) + w_0 s P(t_i) exp(-s DeltaLambda_i); S(i) = sum_{j>=1} w_j exp(-r_j d_i)
   + w_0 exp(-s DeltaLambda_i); u_i = 1 - S(i). Posts per session = 1/w_0; session rate = s per mean-profile minute.
M MMPP: hidden CTMC with generator Q (row 0 multiplied by P(t) if circadian), emission rates lam_s,
   A(t) = Q(t) - diag(lam). Filter alpha (normalised) after event i-1. Gap survival S_i = alpha E(t_{i-1},t_i) 1 with
   E = ordered product of expm(A(slot) * length); density f_i = alpha E lam; logL = sum log f_i + log S(T - t_n);
   alpha_i = (alpha E) .* lam / f_i; u_i = 1 - S_i (exact conditional PIT).
KS: D = sup |F_n(u) - u|, p = kstest(u,'uniform'); PIT MAD = mean |u_(i) - (i - 1/2)/n|.
AIC = 2k - 2 logL, BIC = k ln n - 2 logL. k counts baseline: Hawkes nm + 47 + kernel; circadian renewal/cascade/MMPP 47 + shape.
"""
