"""Phase P1 library: does posting rhythm predict the provocation (TI7) of the next post?  (MATH_SPEC section H, P1)

Contents
  account_features   causal per-account rhythm and persistence features (strictly earlier events only)
  congress_hour_features  causal lagged Congress-wide volume and spike features (hours strictly before)
  MixedLM            linear / PQL-logistic mixed model, account random intercept + random slopes, exact sufficient-stat EM
  target_encode      account target encoding (out-of-fold on training rows, full-train for test rows)
  metrics, bootstraps over accounts

Seed 20261010.  Predictive associations only.
"""
import numpy as np
import pandas as pd

SEED = 20261010
DIMS7 = ["offensive", "toxic", "neg_sentiment", "irony", "ragebait", "tribal", "performative"]
SESSION_S = 1800
TEST_START = pd.Timestamp("2022-07-01")
TEST_END = pd.Timestamp("2023-07-12")  # exclusive; corpus ends 2023-07-11

PERSIST = ["prev_ti7", "roll10", "roll50"]
OWN = ["log_dt", "sess_pos", "sess_el", "n1h", "n24h"]
TOD = ["hour", "hsin", "hcos", "h2sin", "h2cos", "night", "weekend"]
SCHED = ["sched_share"]
CONG = ["cong_p1", "cong_p24", "cong_spike_lr", "cong_spike"]
RHYTHM = OWN + TOD + SCHED + CONG
RHYTHM_NOCONG = OWN + TOD + SCHED
REGIME = ["primary_window", "general_window", "post_election", "lame_duck", "on_ballot", "aligned_pres", "in_majority",
          "sotu", "recess"]
TE = ["te_mean", "te_sd"]
# post type. Congresstweets has no quote field, so reply (text starts with '@') and multi-post thread are the only types.
# Thread size of the TARGET event is not known when its first post is made, so only the first post's reply flag is a
# legal own-type feature; the previous event's reply flag and thread size are known.
TYPE_PREV = ["prev_reply", "prev_multi", "reply_share10"]
TYPE_OWN = ["own_reply"]
LMM_SLOPES = ["log_dt", "sess_pos", "n1h", "night"]


# ----------------------------------------------------------------------------------------------- causal features
def account_features(t, ti7, tool, reply=None, multi=None):
    """Causal features for one account. t: sorted int64 UTC seconds; ti7: float (NaN = not scored); tool: bool.

    Every output for event i is a function of events j with t_j < t_i (strictly earlier in time) and of t_i itself.
    Events sharing the timestamp of i, and all later events, never enter.  Returns dict of arrays (length n).
    """
    t = np.asarray(t, dtype="int64")
    ti7 = np.asarray(ti7, dtype="float64")
    tool = np.asarray(tool, dtype=bool)
    n = len(t)
    assert n == 0 or (np.diff(t) >= 0).all()
    cnt = np.searchsorted(t, t, side="left")  # number of events strictly earlier than event i
    has_prev = cnt > 0
    pidx = np.maximum(cnt - 1, 0)
    dt_prev = np.where(has_prev, t - t[pidx], np.nan)
    # 30-minute sessions on the sorted sequence; position counts strictly earlier events of the same session
    brk = np.ones(n, dtype=bool)
    if n > 1:
        brk[1:] = np.diff(t) > SESSION_S
    sid = np.cumsum(brk) - 1
    start = np.flatnonzero(brk)[sid] if n else np.zeros(0, dtype=int)
    in_sess = cnt > start
    sess_pos = np.where(in_sess, cnt - start, 0)
    sess_el = np.where(in_sess, t - t[start], 0).astype(float)
    n1h = cnt - np.searchsorted(t, t - 3600, side="left")
    n24h = cnt - np.searchsorted(t, t - 86400, side="left")
    # persistence from the scored subsequence among strictly earlier events
    sc = ~np.isnan(ti7)
    m = np.concatenate([[0], np.cumsum(sc)])[cnt]
    sv = ti7[sc]
    cs = np.concatenate([[0.0], np.cumsum(sv)])

    def roll(k):
        lo = np.maximum(m - k, 0)
        den = np.minimum(m, k)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(m > 0, (cs[m] - cs[lo]) / np.maximum(den, 1), np.nan)

    last_scored = np.where(m > 0, sv[np.maximum(m - 1, 0)] if len(sv) else np.nan, np.nan)
    prev_scored = has_prev & sc[pidx]
    prev_ti7 = np.where(prev_scored, ti7[pidx], np.nan)
    ctool = np.concatenate([[0.0], np.cumsum(tool)])
    sched_share = np.where(has_prev, ctool[cnt] / np.maximum(cnt, 1), np.nan)
    # post-type features: previous event's type, reply share of the last 10 events, the event's own first-post type
    rp = np.zeros(n) if reply is None else np.asarray(reply, dtype=float)
    mu = np.zeros(n) if multi is None else np.asarray(multi, dtype=float)
    prev_reply = np.where(has_prev, rp[pidx], np.nan)
    prev_multi = np.where(has_prev, mu[pidx], np.nan)   # thread of the previous event is complete before this event starts
    crp = np.concatenate([[0.0], np.cumsum(rp)])
    lo10 = np.maximum(cnt - 10, 0)
    reply_share10 = np.where(has_prev, (crp[cnt] - crp[lo10]) / np.maximum(cnt - lo10, 1), np.nan)
    return dict(cnt=cnt, dt_prev=dt_prev, sess_pos=sess_pos, sess_el=sess_el, n1h=n1h, n24h=n24h, n_scored_prior=m,
                prev_ti7=prev_ti7, prev_scored=prev_scored, last_scored=last_scored, roll10=roll(10),
                roll50=roll(50), sched_share=sched_share, prev_idx=np.where(has_prev, pidx, -1),
                prev_reply=prev_reply, prev_multi=prev_multi, reply_share10=reply_share10, own_reply=rp)


def _shift(x, s):
    out = np.full(len(x), np.nan)
    if s < len(x):
        out[s:] = x[: len(x) - s]
    return out


def congress_hour_features(c, bad, nweeks=4):
    """Lagged Congress-wide features per clock hour h using only hours < h.

    c: hourly event counts (all matched accounts); bad: bool, hour in a collection outage (count invalid).
    Expected count for an hour = mean of the same hour-of-week in the previous `nweeks` weeks (valid hours only).
    Outage hours inside the lag windows are replaced by their expected value.
    Returns dict of arrays indexed like c: cong_p1 (log1p count hour h-1), cong_p24 (log1p sum h-24..h-1),
    cong_spike_lr (log ratio of the last 3 h to expectation), cong_spike (indicator: ratio > 1.5 and excess > 3 sd).
    """
    c = np.asarray(c, dtype="float64").copy()
    c[np.asarray(bad, dtype=bool)] = np.nan
    stack = np.vstack([_shift(c, 168 * k) for k in range(1, nweeks + 1)])
    with np.errstate(all="ignore"):
        valid = ~np.isnan(stack)
        exp = np.where(valid.sum(0) > 0, np.nansum(stack, 0) / np.maximum(valid.sum(0), 1), np.nan)
    cf = np.where(np.isnan(c), exp, c)  # cf[h] uses c[h] and weeks before h only
    cf0 = np.nan_to_num(cf)
    e0 = np.nan_to_num(exp)
    cs_c = np.concatenate([[0.0], np.cumsum(cf0)])
    cs_e = np.concatenate([[0.0], np.cumsum(e0)])
    n = len(c)
    h = np.arange(n)

    def wsum(cs, w):
        lo = np.maximum(h - w, 0)
        return cs[h] - cs[lo]  # sum over hours h-w .. h-1

    p1 = np.where(h >= 1, cf0[np.maximum(h - 1, 0)], np.nan)
    s24 = wsum(cs_c, 24)
    s3, e3 = wsum(cs_c, 3), wsum(cs_e, 3)
    lr = np.log((s3 + 1.0) / (e3 + 1.0))
    spike = ((s3 > 1.5 * e3) & ((s3 - e3) > 3.0 * np.sqrt(e3 + 1.0))).astype(float)
    ok = h >= 168 * nweeks  # expectations need the full lookback
    out = dict(cong_p1=np.log1p(p1), cong_p24=np.log1p(s24), cong_spike_lr=lr, cong_spike=spike)
    for k in out:
        out[k] = np.where(ok, out[k], np.nan)
    return out


# ----------------------------------------------------------------------------------------------- target encoding
def target_encode(g, y, J, folds=5, m=20.0, seed=SEED, test_g=None):
    """Account mean and sd of y, smoothed to the global mean/sd with weight m.
    Training rows: out-of-fold (a row never sees its own label).  test_g given: returns full-train encoding for those
    account codes (accounts unseen in training get the global value).  g are integer codes 0..J-1."""
    y = np.asarray(y, dtype="float64")
    def enc(sum_, sq_, cnt_, gm, gs):
        mean = (sum_ + m * gm) / (cnt_ + m)
        var = (sq_ - sum_ ** 2 / np.maximum(cnt_, 1)) / np.maximum(cnt_ - 1, 1)
        sd = (np.where(cnt_ > 1, var, gs ** 2) * np.maximum(cnt_ - 1, 0) + m * gs ** 2) / (np.maximum(cnt_ - 1, 0) + m)
        return mean, np.sqrt(np.maximum(sd, 0))

    S = np.bincount(g, y, J)
    Q = np.bincount(g, y * y, J)
    C = np.bincount(g, minlength=J).astype(float)
    if test_g is not None:
        mu, sd = enc(S, Q, C, y.mean(), y.std())
        return mu[test_g], sd[test_g]
    rng = np.random.default_rng(seed)
    f = rng.integers(0, folds, len(y))
    te_m = np.empty(len(y))
    te_s = np.empty(len(y))
    for k in range(folds):
        sel = f == k
        Sk = S - np.bincount(g[sel], y[sel], J)
        Qk = Q - np.bincount(g[sel], y[sel] ** 2, J)
        Ck = C - np.bincount(g[sel], minlength=J)
        yk = y[~sel]     # the prior (global mean/sd) also excludes the held-out fold
        mu, sd = enc(Sk, Qk, Ck, yk.mean(), yk.std())
        te_m[sel], te_s[sel] = mu[g[sel]], sd[g[sel]]
    return te_m, te_s


# ----------------------------------------------------------------------------------------------- mixed model
def _acct_stats(X, y, w, g, J):
    """Per-account sufficient statistics: Sxx (J,p,p) = X'WX, Sxy (J,p), syy (J,), n (J,)."""
    p = X.shape[1]
    order = np.argsort(g, kind="stable")
    gs = g[order]
    Xs, ys, ws = X[order], y[order], w[order]
    bounds = np.searchsorted(gs, np.arange(J + 1))
    Sxx = np.zeros((J, p, p))
    Sxy = np.zeros((J, p))
    syy = np.zeros(J)
    n = np.zeros(J)
    for j in range(J):
        a, b = bounds[j], bounds[j + 1]
        if b == a:
            continue
        xj, yj, wj = Xs[a:b], ys[a:b], ws[a:b]
        xw = xj * wj[:, None]
        Sxx[j] = xj.T @ xw
        Sxy[j] = xw.T @ yj
        syy[j] = float((wj * yj * yj).sum())
        n[j] = b - a
    return Sxx, Sxy, syy, n


class MixedLM:
    """y_ij = x_ij' beta + z_ij' u_j + e_ij,  u_j ~ N(0, G),  e_ij ~ N(0, s2 / w_ij);  Z = X[:, zidx] (zidx includes the
    intercept column 0).  Exact EM using per-account sufficient statistics (no row loop).
    family 'gaussian': s2 estimated.  family 'binomial': penalized quasi-likelihood (PQL), s2 = 1, working weights
    p(1-p) re-derived in `n_outer` outer iterations."""

    def __init__(self, zidx, family="gaussian", n_em=40, n_outer=4, ridge=1e-6):
        self.zidx = list(zidx)
        self.family = family
        self.n_em, self.n_outer, self.ridge = n_em, n_outer, ridge

    def _em(self, S, beta, G, s2, fix_s2, iters):
        Sxx, Sxy, syy, n = S
        zi = self.zidx
        p, q = Sxx.shape[1], len(zi)
        Szz = Sxx[:, zi][:, :, zi]
        Szx = Sxx[:, zi, :]
        Szy = Sxy[:, zi]
        N = n.sum()
        act = n > 0
        for _ in range(iters):
            Gi = np.linalg.inv(G)
            Mi = np.linalg.inv(s2 * Gi[None] + Szz)
            Sx = Sxx - np.einsum("jqp,jqr,jrk->jpk", Szx, Mi, Szx)
            Sy = Sxy - np.einsum("jqp,jqr,jr->jp", Szx, Mi, Szy)
            beta = np.linalg.solve(Sx.sum(0) + self.ridge * np.eye(p), Sy.sum(0))
            rz = Szy - np.einsum("jqp,p->jq", Szx, beta)
            C = np.linalg.inv(Gi[None] + Szz / s2)
            mm = np.einsum("jqr,jr->jq", C, rz) / s2
            mm[~act] = 0
            C[~act] = G
            G = (np.einsum("jq,jr->qr", mm[act], mm[act]) + C[act].sum(0)) / act.sum()
            G = 0.5 * (G + G.T) + 1e-8 * np.eye(q)
            if not fix_s2:
                rr = syy - 2 * Sxy @ beta + np.einsum("p,jpk,k->j", beta, Sxx, beta)
                e2 = rr - 2 * np.einsum("jq,jq->j", mm, rz) + np.einsum("jq,jqr,jr->j", mm, Szz, mm) \
                    + np.einsum("jqr,jrq->j", Szz, C)
                s2 = max(float(e2[act].sum() / N), 1e-8)
        # final posterior means with final beta/G
        Gi = np.linalg.inv(G)
        rz = Szy - np.einsum("jqp,p->jq", Szx, beta)
        C = np.linalg.inv(Gi[None] + Szz / s2)
        mm = np.einsum("jqr,jr->jq", C, rz) / s2
        mm[~act] = 0
        return beta, G, s2, mm

    def fit(self, X, y, g, J, w=None):
        X = np.asarray(X, dtype="float64")
        y = np.asarray(y, dtype="float64")
        g = np.asarray(g)
        w1 = np.ones(len(y)) if w is None else np.asarray(w, dtype="float64")
        q = len(self.zidx)
        if self.family == "gaussian":
            S = _acct_stats(X, y, w1, g, J)
            b0 = np.linalg.solve(S[0].sum(0) + self.ridge * np.eye(X.shape[1]), S[1].sum(0))
            s2 = max(float(((y - X @ b0) ** 2).mean()), 1e-6)
            G = np.eye(q) * 0.1 * s2
            self.beta, self.G, self.s2, self.u = self._em(S, b0, G, s2, False, self.n_em)
        else:
            p0 = float(np.clip(y.mean(), 1e-3, 1 - 1e-3))
            eta = np.full(len(y), np.log(p0 / (1 - p0)))
            G = np.eye(q) * 0.3
            beta = np.zeros(X.shape[1])
            beta[0] = eta[0]
            u = np.zeros((J, q))
            for o in range(self.n_outer):
                mu = 1 / (1 + np.exp(-eta))
                ww = np.clip(mu * (1 - mu), 1e-4, None)
                z = eta + (y - mu) / ww
                S = _acct_stats(X, z, ww, g, J)
                beta, G, _, u = self._em(S, beta, G, 1.0, True, self.n_em if o == 0 else max(self.n_em // 2, 8))
                eta = X @ beta + np.einsum("ij,ij->i", X[:, self.zidx], u[g])
            self.beta, self.G, self.s2, self.u = beta, G, 1.0, u
        return self

    def predict(self, X, g, use_re=True, link=True):
        X = np.asarray(X, dtype="float64")
        eta = X @ self.beta
        if use_re:
            g = np.asarray(g)
            ok = (g >= 0) & (g < len(self.u))
            re = np.zeros(len(eta))
            re[ok] = np.einsum("ij,ij->i", X[ok][:, self.zidx], self.u[g[ok]])
            eta = eta + re
        if self.family == "binomial" and link:
            return 1 / (1 + np.exp(-eta))
        return eta


# ----------------------------------------------------------------------------------------------- metrics
def ece(p, y, bins=10):
    q = np.minimum((np.asarray(p) * bins).astype(int), bins - 1)
    tot = 0.0
    for b in range(bins):
        s = q == b
        if s.any():
            tot += s.mean() * abs(p[s].mean() - y[s].mean())
    return float(tot)


def auc(score, y):
    from scipy.stats import rankdata
    y = np.asarray(y).astype(bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(score)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def skill_boot(acct_sum_model, acct_sum_base, B=2000, seed=SEED):
    """Skill = 1 - sum(model)/sum(base) over accounts (additive per-account loss sums), cluster bootstrap over accounts."""
    a = np.asarray(acct_sum_model, dtype=float)
    b = np.asarray(acct_sum_base, dtype=float)
    J = len(a)
    rng = np.random.default_rng(seed)
    cnt = rng.multinomial(J, np.full(J, 1.0 / J), size=B)
    sk = 1 - (cnt @ a) / (cnt @ b)
    pt = 1 - a.sum() / b.sum()
    return float(pt), float(np.percentile(sk, 2.5)), float(np.percentile(sk, 97.5))
