"""RVT-E test E4: step at the election and step at the seating. Follows scripts/v7/rvt_e_DESIGN.md.
Run from the scripts/v7 folder: the member-day state is built by the first block of rvt_c_01_state.py.
Addition to the design, decided before any E4 estimate was seen: the same structure moved by exactly one year
(same season, no election and no change of control), where it lies inside the data."""
import json
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sst

exec(open("rvt_c_01_state.py").read().split("# ---------------------------------------------------------------- responses to the general-election window")[0])
O = ROOT + "/rvt_e"
os.makedirs(O, exist_ok=True)
rng = np.random.default_rng(20261112)
W = 120
PAIRS = [("2018-11-06", "2019-01-03"), ("2020-11-03", "2021-01-20"), ("2022-11-08", "2023-01-03")]
REAL = [pd.Timestamp(x) for p in PAIRS for x in p]
D["lev"] = np.log1p(D.events)
D["ti7d"] = np.where(D.n > 0, D.ti_sum / D.n.where(D.n > 0), np.nan)
DMIN, DMAX = D.date.min(), D.date.max()
byday = D.set_index("date").sort_index()


def groups(seat):
    p0 = cal[cal.date == seat - pd.Timedelta(days=1)].set_index("bioguide")
    p1 = cal[cal.date == seat].set_index("bioguide")
    dp = ((p1.aligned_with_president + p1.in_chamber_majority) - (p0.aligned_with_president + p0.in_chamber_majority)).dropna()
    return set(dp[dp < 0].index), set(dp[dp > 0].index)


def series(lost, gained, e, s):
    lo, hi = e - pd.Timedelta(days=W), s + pd.Timedelta(days=W)
    x = byday.loc[lo:hi]
    ndays = (hi - lo).days + 1
    cnt = x.groupby("bioguide").size()
    stay = set(cnt[cnt >= ndays].index)
    out, sizes = {}, {}
    for nm, g in (("lost", lost & stay), ("gained", gained & stay)):
        y = x[x.bioguide.isin(g)]
        out[nm + "_ti7"] = y.groupby(level=0).ti7d.mean()
        out[nm + "_lev"] = y.groupby(level=0).lev.mean()
        sizes[nm] = len(g)
    S = pd.DataFrame(out)
    S["diff_ti7"], S["diff_lev"] = S.lost_ti7 - S.gained_ti7, S.lost_lev - S.gained_lev
    return S[(S.index != e) & (S.index != s)], sizes


def fit(y, e, s):
    y = y.dropna()
    t = (y.index - e).days.values.astype(float)
    dow = pd.get_dummies(y.index.dayofweek, drop_first=True).astype(float).values
    base = np.column_stack([np.ones(len(t)), t, dow])
    se_, ss_ = (y.index > e).astype(float), (y.index > s).astype(float)
    k = base.shape[1]
    full = np.column_stack([base, se_, ss_])
    r = sm.OLS(y.values, full).fit()
    rh = sm.OLS(y.values, full).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
    bic = dict(neither=sm.OLS(y.values, base).fit().bic, election_only=sm.OLS(y.values, np.column_stack([base, se_])).fit().bic,
               seating_only=sm.OLS(y.values, np.column_stack([base, ss_])).fit().bic, both=r.bic)
    A, B, C = y[y.index < e].mean(), y[(y.index > e) & (y.index < s)].mean(), y[y.index > s].mean()
    return dict(step_election=float(r.params[k]), se_election=float(rh.bse[k]), step_seating=float(r.params[k + 1]), se_seating=float(rh.bse[k + 1]),
                bic={a: float(b) for a, b in bic.items()}, best_bic=min(bic, key=bic.get), n_days=int(len(y)),
                mean_before_election=float(A), mean_between=float(B), mean_after_seating=float(C))


RES = {}
for e, s in PAIRS:
    e, s = pd.Timestamp(e), pd.Timestamp(s)
    gap = (s - e).days
    lost, gained = groups(s)
    S, sizes = series(lost, gained, e, s)
    S.to_csv(O + f"/e4_series_{s.date()}.csv")
    # placebo anchors
    cand = [d for d in pd.date_range(DMIN + pd.Timedelta(days=W), DMAX - pd.Timedelta(days=W + gap))
            if all(abs((d - x).days) >= 21 and abs((d + pd.Timedelta(days=gap) - x).days) >= 21 for x in REAL)]
    anchors = [pd.Timestamp(d) for d in rng.choice(np.array(cand, dtype="datetime64[ns]"), 50, replace=False)]
    PL = {}
    for a in anchors:
        Sp, sz = series(lost, gained, a, a + pd.Timedelta(days=gap))
        if min(sz.values()) < 20:
            continue
        PL[a] = Sp
    yearshift = {}
    for sh in (-365, 365):
        a = e + pd.Timedelta(days=sh)
        if a - pd.Timedelta(days=W) >= DMIN and a + pd.Timedelta(days=gap + W) <= DMAX:
            Sp, sz = series(lost, gained, a, a + pd.Timedelta(days=gap))
            if min(sz.values()) >= 20:
                yearshift[sh] = (a, Sp, sz)
    R = dict(election=str(e.date()), seating=str(s.date()), gap_days=gap, members=sizes, n_placebo=len(PL))
    for col in ("diff_ti7", "diff_lev", "lost_ti7", "gained_ti7", "lost_lev", "gained_lev"):
        f = fit(S[col], e, s)
        pf = [fit(Sp[col], a, a + pd.Timedelta(days=gap)) for a, Sp in PL.items()]
        pe, ps = np.array([x["step_election"] for x in pf]), np.array([x["step_seating"] for x in pf])
        f["placebo_p_election"] = float((1 + (np.abs(pe) >= abs(f["step_election"])).sum()) / (len(pe) + 1))
        f["placebo_p_seating"] = float((1 + (np.abs(ps) >= abs(f["step_seating"])).sum()) / (len(ps) + 1))
        f["placebo_sd_election"], f["placebo_sd_seating"] = float(pe.std(ddof=1)), float(ps.std(ddof=1))
        f["same_season_other_year"] = {str(sh): dict(anchor=str(a.date()), members=sz, **{k: v for k, v in fit(Sp[col], a, a + pd.Timedelta(days=gap)).items()
                                                                                         if k.startswith(("step", "se_"))})
                                       for sh, (a, Sp, sz) in yearshift.items()}
        R[col] = f
    RES[str(s.date())] = R
    print(json.dumps(R, indent=1), flush=True)

pooled = {}
for col in ("diff_ti7", "diff_lev"):
    pooled[col] = {}
    for ev in ("election", "seating"):
        b = np.array([RES[k][col]["step_" + ev] for k in RES])
        se = np.array([RES[k][col]["se_" + ev] for k in RES])
        w = 1 / se ** 2
        ps = np.array([RES[k][col]["placebo_p_" + ev] for k in RES])
        chi = -2 * np.log(ps).sum()
        pooled[col][ev] = dict(mean=float((w * b).sum() / w.sum()), se=float(np.sqrt(1 / w.sum())), steps=[float(x) for x in b],
                               placebo_ps=[float(x) for x in ps], fisher_chi2=float(chi), fisher_p=float(sst.chi2.sf(chi, 2 * len(ps))))
    fe, fs = pooled[col]["election"]["fisher_p"], pooled[col]["seating"]["fisher_p"]
    pooled[col]["reading"] = ("at the seating" if fs < 0.05 <= fe else "at the election" if fe < 0.05 <= fs else "both" if max(fe, fs) < 0.05 else "neither")
RES["pooled"] = pooled
print(json.dumps(pooled, indent=1))
json.dump(RES, open(O + "/e4_dating.json", "w"), indent=1)
