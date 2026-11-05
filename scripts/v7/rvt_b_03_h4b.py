import numpy as np, pandas as pd, json
from scipy.stats import binomtest
from numpy.lib.stride_tricks import sliding_window_view as swv
O = '/home/joneill/x_box/pipeline/output/v7/rvt_b'
rng = np.random.default_rng(20261013)
mw = pd.read_parquet(O + '/member_week.parquet')
mo = pd.read_parquet(O + '/member_week_office.parquet')
WK0, WK1 = pd.Timestamp('2017-06-19'), pd.Timestamp('2023-07-10')
mo = mo[(mo.week >= WK0) & (mo.week <= WK1)]
EXC = [pd.Timestamp(x) for x in '2018-11-06 2019-01-03 2020-11-03 2021-01-05 2021-01-20 2022-11-08 2023-01-03 2024-11-05 2025-01-03 2025-01-20'.split()]


def near_exc(w0, w1=None):
    w1 = w1 if w1 is not None else w0
    return any((e >= w0 - pd.Timedelta(days=21)) and (e <= w1 + pd.Timedelta(days=6 + 21)) for e in EXC)


def pelt2(y, pen, ms=8):
    n = len(y)
    cs = np.r_[0, np.cumsum(y)]
    cs2 = np.r_[0, np.cumsum(y ** 2)]
    F = np.full(n + 1, np.inf)
    F[0] = -pen
    last = np.zeros(n + 1, int)
    for t in range(ms, n + 1):
        s = np.arange(0, t - ms + 1)
        s = s[np.isfinite(F[s])]
        if len(s) == 0:
            continue
        L = t - s
        c = (cs2[t] - cs2[s]) - (cs[t] - cs[s]) ** 2 / L
        v = F[s] + c + pen
        i = v.argmin()
        F[t] = v[i]
        last[t] = s[i]
    cps = []
    t = n
    while t > 0:
        s = last[t]
        if s > 0:
            cps.append(s)
        t = s
    return sorted(cps)


def detr(x):
    t = np.arange(len(x))
    A = np.c_[np.ones(len(x)), t]
    b = np.linalg.lstsq(A, x, rcond=None)[0]
    return x - A @ b


def detr_batch(X):
    n = X.shape[1]
    t = np.arange(n)
    A = np.c_[np.ones(n), t]
    P = A @ np.linalg.pinv(A)
    return X - X @ P.T


def kt(Y):
    k = Y.shape[1]
    iu = np.triu_indices(k, 1)
    d = Y[:, iu[1]] - Y[:, iu[0]]
    return np.sign(d).sum(1) / len(iu[0])


RW = 13


def ind(X):
    S = swv(X, RW, axis=1)
    v = S.var(axis=2)
    a = S[:, :, :-1]
    b = S[:, :, 1:]
    a = a - a.mean(2, keepdims=True)
    b = b - b.mean(2, keepdims=True)
    ac = (a * b).sum(2) / np.sqrt((a ** 2).sum(2) * (b ** 2).sum(2) + 1e-12)
    return kt(v), kt(ac)


NS = 1000


def test(x):
    r = detr(x)
    tv, ta = ind(r[None])
    phi = np.clip(np.corrcoef(r[:-1], r[1:])[0, 1], -.95, .95)
    sd = r.std(ddof=1) * np.sqrt(1 - phi ** 2)
    e = rng.normal(0, sd, (NS, 26 + 50))
    S = np.zeros_like(e)
    for i in range(1, e.shape[1]):
        S[:, i] = phi * S[:, i - 1] + e[:, i]
    S = detr_batch(S[:, 50:])
    sv, sa = ind(S)
    F = np.fft.rfft(r)
    ph = rng.uniform(0, 2 * np.pi, (NS, len(F)))
    ph[:, 0] = 0
    if len(r) % 2 == 0:
        ph[:, -1] = 0
    Q = np.fft.irfft(np.abs(F)[None] * np.exp(1j * ph), n=len(r))
    Q = detr_batch(Q)
    qv, qa = ind(Q)
    return dict(tau_var=tv[0], tau_ac=ta[0], p_var_ar=(sv >= tv[0]).mean(), p_ac_ar=(sa >= ta[0]).mean(),
                p_var_ph=(qv >= tv[0]).mean(), p_ac_ph=(qa >= ta[0]).mean())


def runs_from(vals, valid):
    v = vals.copy()
    idx = np.where(valid)[0]
    runs = []
    if len(idx) == 0:
        return runs
    st = idx[0]
    prev = idx[0]
    for i in idx[1:]:
        if i - prev - 1 > 2:
            runs.append((st, prev))
            st = i
        prev = i
    runs.append((st, prev))
    out = []
    for a, b in runs:
        seg = v[a:b + 1].copy()
        ok = valid[a:b + 1]
        seg = np.interp(np.arange(len(seg)), np.where(ok)[0], seg[ok])
        out.append((a, b, seg))
    return out


res = []
ctrl = []
stats = {}
for stype in ('ti7', 'vol'):
    nrun = ncp = ncp_keep = nq = 0
    for b, g in mo.groupby('bioguide'):
        weeks = pd.date_range(g.week.min(), g.week.max(), freq='7D')
        w = pd.DataFrame(index=weeks)
        m = mw[mw.bioguide == b].set_index('week')
        w['n'] = m.n.reindex(weeks).fillna(0).values
        w['ti7'] = m.ti7.reindex(weeks).values
        if stype == 'ti7':
            vals = w.ti7.values
            valid = (w.n.values >= 5)
        else:
            vals = np.log1p(w.n.values)
            valid = np.ones(len(w), bool)
        for a, bb, seg in runs_from(vals, valid):
            if len(seg) < 60:
                continue
            nrun += 1
            dif = np.diff(seg)
            sig = np.median(np.abs(dif - np.median(dif))) / 0.6745 / np.sqrt(2)
            if sig <= 0:
                continue
            pen = 4 * sig ** 2 * np.log(len(seg))
            cps = pelt2(seg, pen)
            sdrun = seg.std(ddof=1)
            bounds = [0] + cps + [len(seg)]
            ncp += len(cps)
            wk = weeks[a:bb + 1]
            for j, c in enumerate(cps):
                prev = seg[bounds[j]:c].mean()
                nxt = seg[c:bounds[j + 2]].mean()
                if abs(nxt - prev) <= sdrun:
                    continue
                ncp_keep += 1
                if near_exc(wk[c]):
                    continue
                if c < 26:
                    continue
                if any((p >= c - 26) and (p < c) for p in cps):
                    continue
                nq += 1
                r = test(seg[c - 26:c])
                r.update(member=b, stype=stype, week=wk[c], shift=(nxt - prev) / sdrun, kind='event')
                res.append(r)
                cand = [e for e in range(26, len(seg) - 8 + 1)
                        if not any((p >= e - 26) and (p <= e + 8) for p in cps)
                        and not near_exc(wk[e - 26], wk[min(e + 8, len(wk) - 1)])]
                if cand:
                    for e in rng.choice(cand, min(5, len(cand)), replace=False):
                        r2 = test(seg[e - 26:e])
                        r2.update(member=b, stype=stype, week=wk[e], shift=np.nan, kind='control', event_week=wk[c])
                        ctrl.append(r2)
    stats[stype] = dict(runs=nrun, cps=ncp, cps_big=ncp_keep, qualifying=nq)
    print(stype, stats[stype], flush=True)
E = pd.DataFrame(res)
Cn = pd.DataFrame(ctrl)
E.to_csv(O + '/h4b_events.csv', index=False)
Cn.to_csv(O + '/h4b_controls.csv', index=False)
out = []
B = 2000
for stype in ('ti7', 'vol'):
    e = E[E.stype == stype]
    c = Cn[Cn.stype == stype]
    for ind_, tau, pa, pp in (('variance', 'tau_var', 'p_var_ar', 'p_var_ph'), ('ac1', 'tau_ac', 'p_ac_ar', 'p_ac_ph')):
        n = len(e)
        if n == 0:
            continue
        k = int((e[pa] < 0.05).sum())
        bt = binomtest(k, n, 0.05, alternative='greater')
        kp = int((e[pp] < 0.05).sum())
        btp = binomtest(kp, n, 0.05, alternative='greater')
        sc = (c[pa] < 0.05).mean() if len(c) else np.nan
        mem = e.member.unique()
        eg = {m: e[tau].values[e.member.values == m] for m in mem}
        cg = {m: c[tau].values[c.member.values == m] for m in mem}
        diffs = []
        for _ in range(B):
            ms = rng.choice(mem, len(mem))
            a = np.concatenate([eg[m] for m in ms])
            b_ = np.concatenate([cg[m] for m in ms])
            diffs.append(a.mean() - b_.mean() if len(b_) else np.nan)
        diffs = np.array(diffs)
        out.append(dict(stype=stype, indicator=ind_, n_events=n, n_members=len(mem), n_controls=len(c), share_sig_ar=k / n,
                        binom_p_ar=bt.pvalue, share_sig_phase=kp / n, binom_p_phase=btp.pvalue, control_share_sig_ar=sc,
                        mean_tau_events=e[tau].mean(), mean_tau_controls=c[tau].mean(), tau_diff=e[tau].mean() - c[tau].mean(),
                        diff_lo=np.nanpercentile(diffs, 2.5), diff_hi=np.nanpercentile(diffs, 97.5), up_share=(e['shift'] > 0).mean()))
S = pd.DataFrame(out)
p = S.binom_p_ar.values
o = np.argsort(p)
h = np.empty(len(p))
run = 0
for r_, i in enumerate(o):
    run = max(run, min(1, (len(p) - r_) * p[i]))
    h[i] = run
S['holm_p'] = h
S.to_csv(O + '/h4b_summary.csv', index=False)
pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 40)
print(S.round(4).to_string())
