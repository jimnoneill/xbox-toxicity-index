"""RVT-C tests C1 to C5. Follows scripts/v7/rvt_c_DESIGN.md (copied to output/v7/rvt_c/DESIGN.md before estimation)."""
import json
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

ROOT = os.path.expanduser("~/x_box/pipeline/output/v7")
O = ROOT + "/rvt_c"
os.makedirs(O, exist_ok=True)
rng = np.random.default_rng(20261110)
RES = {}

# ---------------------------------------------------------------- member-day state
md = pd.read_parquet(ROOT + "/rvt_b/member_day.parquet", columns=["bioguide", "date", "ti_sum", "n"])
md["date"] = pd.to_datetime(md.date)
ev = pd.read_parquet(ROOT + "/t1full/events.parquet", columns=["bioguide", "t"]).dropna(subset=["bioguide"])
ev["date"] = pd.to_datetime(ev.t, unit="s", utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
ed = ev.groupby(["bioguide", "date"]).size().rename("events").reset_index()
cal = pd.read_parquet(ROOT + "/t3/calendar_member_day.parquet",
                      columns=["date", "bioguide", "in_office", "party", "chamber", "general_window", "on_ballot_this_cycle", "cycle_year",
                               "aligned_with_president", "in_chamber_majority"])
cal["date"] = pd.to_datetime(cal.date)
cal["bioguide"] = cal.bioguide.astype(str)
cal = cal[(cal.in_office == 1) & cal.party.isin(["D", "R"]) & (cal.date >= "2017-06-21") & (cal.date <= "2023-07-11")]
D = cal.merge(md, on=["bioguide", "date"], how="left").merge(ed, on=["bioguide", "date"], how="left")
D[["ti_sum", "n", "events"]] = D[["ti_sum", "n", "events"]].fillna(0)
print("member-days", len(D), "members", D.bioguide.nunique(), flush=True)


def state(g):
    """TI7 (post-weighted) and log event rate over the rows of g; TI7 is NaN under 10 posts."""
    n = g.n.sum()
    return pd.Series(dict(ti7=g.ti_sum.sum() / n if n >= 10 else np.nan, lvol=np.log((g.events.sum() + 0.5) / len(g)), days=len(g), posts=n))


def ols(y, X, groups):
    ok = np.isfinite(y) & np.isfinite(X).all(1)
    r = sm.OLS(y[ok], X[ok]).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(groups[ok])[0]})
    return r, int(ok.sum()), int(pd.Series(groups[ok]).nunique())


def holm(ps):
    return [float(x) for x in multipletests(ps, method="holm")[1]]


# ---------------------------------------------------------------- responses to the general-election window
rows = []
for c, (lab, elec) in {2018: ("2018-09-03", "2018-11-06"), 2020: ("2020-09-07", "2020-11-03"), 2022: ("2022-09-05", "2022-11-08")}.items():
    lab, elec = pd.Timestamp(lab), pd.Timestamp(elec)
    win = D[(D.date >= lab) & (D.date <= elec)]
    pre = D[(D.date >= lab - pd.Timedelta(days=56)) & (D.date < lab)]
    a = win.groupby("bioguide").apply(state)
    b = pre.groupby("bioguide").apply(state)
    info = win.groupby("bioguide").agg(party=("party", "first"), chamber=("chamber", "first"), ballot=("on_ballot_this_cycle", "max"))
    m = a.join(b, lsuffix="_w", rsuffix="_p", how="inner").join(info)
    m = m[(m.days_w >= 0.9 * ((elec - lab).days + 1)) & (m.days_p >= 50)]
    m["R_ti7"] = m.ti7_w - m.ti7_p
    m["R_lvol"] = m.lvol_w - m.lvol_p
    m["cycle"] = c
    rows.append(m.reset_index())
R = pd.concat(rows, ignore_index=True)
h3 = pd.read_csv(ROOT + "/rvt_b/h3_data.csv", usecols=["bioguide", "trans", "margin", "margin_missing"])
h3 = h3[h3.margin_missing == 0]
h3["mcycle"] = h3.trans.str[:4].astype(int) - 1 + (h3.trans.str[:4].astype(int) % 2 == 0) * 0
h3["mcycle"] = h3.trans.map({"2019-01-03": 2018, "2021-01-20": 2020, "2023-01-03": 2022})
mg = h3.set_index(["bioguide", "mcycle"]).margin
R["margin_prev"] = [mg.get((b, c - 2), np.nan) for b, c in zip(R.bioguide, R.cycle)]
R["margin_this"] = [mg.get((b, c), np.nan) for b, c in zip(R.bioguide, R.cycle)]
R.to_csv(O + "/election_responses.csv", index=False)
print(R.groupby(["cycle", "ballot"]).size(), flush=True)

cyc = pd.get_dummies(R.cycle, prefix="c", drop_first=True).astype(float)
for test in ("C1", "C2", "C3"):
    RES[test] = {}
ps = {}
for out in ("R_ti7", "R_lvol"):
    # C1
    X = pd.concat([pd.Series(1.0, index=R.index, name="const"), R.ballot.astype(float).rename("ballot"), cyc], axis=1)
    r, n, g = ols(R[out].values, X.values, R.bioguide.values)
    RES["C1"][out] = dict(b=float(r.params[1]), se=float(r.bse[1]), p=float(r.pvalues[1]), n=n, members=g,
                          mean_on_ballot=float(R[R.ballot == 1][out].mean()), mean_off_ballot=float(R[R.ballot == 0][out].mean()),
                          sd_response=float(R[out].std()))
    # C2
    S = R[(R.ballot == 1) & R.margin_prev.notna()].copy()
    S["z"] = (S.margin_prev - S.margin_prev.mean()) / S.margin_prev.std()
    X = pd.concat([pd.Series(1.0, index=S.index), S.z, (S.party == "R").astype(float), (S.chamber == "senate").astype(float),
                   pd.get_dummies(S.cycle, drop_first=True).astype(float)], axis=1)
    r, n, g = ols(S[out].values, X.values, S.bioguide.values)
    RES["C2"][out] = dict(b_per_sd_margin=float(r.params[1]), se=float(r.bse[1]), p=float(r.pvalues[1]), n=n, members=g, sd_margin=float(S.margin_prev.std()))
    # C3
    W = R[R.ballot == 1].pivot(index="bioguide", columns="cycle", values=out)
    recs = []
    for c in (2020, 2022):
        if c in W and c - 2 in W:
            d = (W[c] - W[c - 2]).dropna()
            for b, v in d.items():
                recs.append(dict(bioguide=b, cycle=c, dR=v, prev=W.loc[b, c - 2], cur=W.loc[b, c], margin_prev=mg.get((b, c - 2), np.nan)))
    T = pd.DataFrame(recs)
    U = T[T.margin_prev.notna()].copy()
    U["z"] = (U.margin_prev - U.margin_prev.mean()) / U.margin_prev.std()
    X = pd.concat([pd.Series(1.0, index=U.index), U.z, (U.cycle == 2022).astype(float)], axis=1)
    r, n, g = ols(U.dR.values, X.values, U.bioguide.values)
    RES["C3"][out] = dict(b_per_sd_margin=float(r.params[1]), se=float(r.bse[1]), p=float(r.pvalues[1]), n=n, members=g,
                          corr_successive=float(np.corrcoef(T.prev, T.cur)[0, 1]), n_pairs=int(len(T)))
for test, key in (("C1", "p"), ("C2", "p"), ("C3", "p")):
    hp = holm([RES[test]["R_ti7"][key], RES[test]["R_lvol"][key]])
    RES[test]["R_ti7"]["p_holm"], RES[test]["R_lvol"]["p_holm"] = hp

# ---------------------------------------------------------------- C4 displacement of the state at changes in party control
TR = ["2019-01-03", "2021-01-20", "2023-01-03"]


def displacement(dates, shift_days=0):
    out = []
    for tr in dates:
        t = pd.Timestamp(tr) - pd.Timedelta(days=shift_days)
        t0 = pd.Timestamp(tr)
        before = D[(D.date >= t - pd.Timedelta(days=180)) & (D.date < t)]
        after = D[(D.date >= t) & (D.date < t + pd.Timedelta(days=180))]
        a, b = after.groupby("bioguide").apply(state), before.groupby("bioguide").apply(state)
        m = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
        m = m[(m.days_a >= 170) & (m.days_b >= 170)]
        # power on the day before and the day of the real change
        p0 = cal[cal.date == t0 - pd.Timedelta(days=1)].set_index("bioguide")
        p1 = cal[cal.date == t0].set_index("bioguide")
        dp = ((p1.aligned_with_president + p1.in_chamber_majority) - (p0.aligned_with_president + p0.in_chamber_majority)).dropna()
        m = m.join(dp.rename("dpower"), how="inner")
        m["d_ti7"], m["d_lvol"], m["trans"] = m.ti7_a - m.ti7_b, m.lvol_a - m.lvol_b, tr
        out.append(m.reset_index())
    return pd.concat(out, ignore_index=True)


RES["C4"] = {}
for name, sh in (("real", 0), ("placebo_one_year_before", 365)):
    M = displacement(TR, sh)
    M = M[M.dpower != 0].copy()
    M["lost"] = (M.dpower < 0).astype(float)
    M.to_csv(O + f"/displacement_{name}.csv", index=False)
    RES["C4"][name] = {}
    for out in ("d_ti7", "d_lvol"):
        X = pd.concat([pd.Series(1.0, index=M.index), M.lost, pd.get_dummies(M.trans, drop_first=True).astype(float)], axis=1)
        r, n, g = ols(M[out].values, X.values, M.bioguide.values)
        RES["C4"][name][out] = dict(lost_minus_gained=float(r.params[1]), se=float(r.bse[1]), p=float(r.pvalues[1]), n=n, members=g,
                                    by_change={tr: float(M[(M.trans == tr) & (M.lost == 1)][out].mean() - M[(M.trans == tr) & (M.lost == 0)][out].mean()) for tr in TR},
                                    sd=float(M[out].std()))
    hp = holm([RES["C4"][name]["d_ti7"]["p"], RES["C4"][name]["d_lvol"]["p"]])
    RES["C4"][name]["d_ti7"]["p_holm"], RES["C4"][name]["d_lvol"]["p_holm"] = hp

# ---------------------------------------------------------------- C5 step and ramp in the losing party's daily log event count
LOSER = {"2019-01-03": "R", "2021-01-20": "R", "2023-01-03": "D"}
W = 120
D["lev"] = np.log1p(D.events)
pm_ = D.groupby(["party", "date"]).lev.mean()


def fit(s, center):
    t = (s.index - center).days.values.astype(float)
    post = (t >= 0).astype(float)
    dow = pd.get_dummies(s.index.dayofweek, drop_first=True).astype(float).values
    base = np.column_stack([np.ones(len(t)), t, dow])
    r1 = sm.OLS(s.values, np.column_stack([base, post])).fit()
    r2 = sm.OLS(s.values, np.column_stack([base, t * post])).fit()
    rh = sm.OLS(s.values, np.column_stack([base, post])).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
    k = base.shape[1]
    return float(r1.params[k]), float(rh.bse[k]), float(r2.bic - r1.bic)


RES["C5"] = {}
alld = pd.DatetimeIndex(sorted(D.date.unique()))
for tr, party in LOSER.items():
    c = pd.Timestamp(tr)
    s = pm_[party]
    s = s[(s.index >= c - pd.Timedelta(days=W)) & (s.index <= c + pd.Timedelta(days=W)) & (s.index != c)]
    step, se, dbic = fit(s, c)
    cand = [d for d in alld if d - pd.Timedelta(days=W) >= alld[0] and d + pd.Timedelta(days=W) <= alld[-1]
            and all(abs((d - pd.Timestamp(x)).days) >= 21 for x in list(LOSER) + ["2020-11-03"])]
    pl = []
    for d in rng.choice(np.array(cand, dtype="datetime64[ns]"), 50, replace=False):
        d = pd.Timestamp(d)
        q = pm_[party]
        q = q[(q.index >= d - pd.Timedelta(days=W)) & (q.index <= d + pd.Timedelta(days=W)) & (q.index != d)]
        pl.append(fit(q, d)[0])
    pl = np.array(pl)
    RES["C5"][tr] = dict(party=party, step=step, se=se, dbic_step_vs_ramp=dbic, placebo_p_two_sided=float((1 + (np.abs(pl) >= abs(step)).sum()) / 51),
                         placebo_p_up=float((1 + (pl >= step).sum()) / 51), placebo_p95_abs=float(np.quantile(np.abs(pl), 0.95)))

json.dump(RES, open(O + "/rvt_c_results.json", "w"), indent=1)
print(json.dumps(RES, indent=1))
