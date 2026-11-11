"""RVT-D tests D1 to D3. Follows scripts/v7/rvt_d_DESIGN.md (copied to output/v7/rvt_d/DESIGN.md before estimation).
Run from the scripts/v7 folder: the member-day state is built by the first block of rvt_c_01_state.py."""
import json
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

exec(open("rvt_c_01_state.py").read().split("# ---------------------------------------------------------------- responses to the general-election window")[0])
O = ROOT + "/rvt_d"
os.makedirs(O, exist_ok=True)

# post-level sum of squares of TI7 by member-day, built as in rvt_b_01_build.py
D7 = ["offensive", "toxic", "neg_sentiment", "irony", "ragebait", "tribal", "performative"]
pp = pd.read_parquet(ROOT + "/full/post_probs_full.parquet", columns=["screen_name", "created_at", "post_type"] + [f"p_{d}" for d in D7])
pp = pp[pp.post_type != "retweet"]
pp["ti7"] = pp[[f"p_{d}" for d in D7]].mean(axis=1).astype("float64")
am = pd.read_csv(ROOT + "/ti/account_member_map.csv")[["screen_name", "bioguide"]].dropna(subset=["bioguide"])
pp = pp[["screen_name", "created_at", "ti7"]].merge(am, on="screen_name")
pp["date"] = pd.to_datetime(pp.created_at, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
pp["sq"] = pp.ti7 ** 2
g = pp.groupby(["bioguide", "date"]).agg(ti_sq=("sq", "sum"), ti_s2=("ti7", "sum"), n2=("ti7", "size")).reset_index()
g["bioguide"] = g.bioguide.astype(str)
del pp
ld = pd.read_parquet(ROOT + "/t3/calendar_member_day.parquet", columns=["date", "bioguide", "lame_duck"])
ld["date"] = pd.to_datetime(ld.date)
ld["bioguide"] = ld.bioguide.astype(str)
D = D.merge(g, on=["bioguide", "date"], how="left").merge(ld, on=["bioguide", "date"], how="left")
D[["ti_sq", "ti_s2", "n2", "lame_duck"]] = D[["ti_sq", "ti_s2", "n2", "lame_duck"]].fillna(0)
print("check: posts by the two builds", D.n.sum(), D.n2.sum(), flush=True)


def state2(x):
    n, E, days = x.n2.sum(), x.events.sum(), len(x)
    m = E / days
    return pd.Series(dict(
        ti7=x.ti_s2.sum() / n if n >= 10 else np.nan,
        lvol=np.log((E + 0.5) / days),
        ti_sd=np.sqrt((x.ti_sq.sum() - x.ti_s2.sum() ** 2 / n) / (n - 1)) if n >= 20 else np.nan,
        xd=(x.events.var(ddof=1) - m) / m ** 2 if E >= 20 else np.nan,
        days=days, posts=n, events=E))


rows = []
for c, (lab, elec) in {2018: ("2018-09-03", "2018-11-06"), 2020: ("2020-09-07", "2020-11-03"), 2022: ("2022-09-05", "2022-11-08")}.items():
    lab, elec, end = pd.Timestamp(lab), pd.Timestamp(elec), pd.Timestamp(f"{c + 1}-01-02")
    per = dict(p=D[(D.date >= lab - pd.Timedelta(days=56)) & (D.date < lab)], w=D[(D.date >= lab) & (D.date <= elec)],
               q=D[(D.date > elec) & (D.date <= end)])
    S = {k: v.groupby("bioguide").apply(state2).add_suffix("_" + k) for k, v in per.items()}
    info = per["w"].groupby("bioguide").agg(party=("party", "first"), chamber=("chamber", "first"), ballot=("on_ballot_this_cycle", "max"))
    dep = per["q"].groupby("bioguide").lame_duck.max().rename("depart")
    m = S["w"].join(S["p"], how="inner").join(S["q"], how="left").join(info).join(dep)
    m = m[(m.days_w >= 0.9 * ((elec - lab).days + 1)) & (m.days_p >= 50)]
    m["has_post"] = (m.days_q >= 50).astype(int)
    m["cycle"] = c
    rows.append(m.reset_index())
R = pd.concat(rows, ignore_index=True)
for v in ("ti7", "lvol", "ti_sd", "xd"):
    R["R_" + v] = R[v + "_w"] - R[v + "_p"]
    R["L_" + v] = np.where(R.has_post == 1, R[v + "_q"] - R[v + "_w"], np.nan)
R["depart"] = R.depart.fillna(0)
R["group"] = np.select([(R.ballot == 1) & (R.depart == 1), (R.ballot == 1) & (R.depart == 0), (R.ballot == 0) & (R.depart == 1)],
                       ["released", "returned", "departing_no_ballot"], "continuing")
R.to_csv(O + "/release_width_responses.csv", index=False)


def wins(s):
    lo, hi = s.quantile(0.01), s.quantile(0.99)
    return s.clip(lo, hi)


def holm(ps):
    return [float(x) for x in multipletests(ps, method="holm")[1]]


def group_model(y, T):
    T = T[np.isfinite(T[y])].copy()
    X = pd.DataFrame({"const": 1.0, "released": (T.group == "released").astype(float),
                      "departing_no_ballot": (T.group == "departing_no_ballot").astype(float),
                      "continuing": (T.group == "continuing").astype(float), "rep": (T.party == "R").astype(float),
                      "senate": (T.chamber == "senate").astype(float), "c2020": (T.cycle == 2020).astype(float),
                      "c2022": (T.cycle == 2022).astype(float)}, index=T.index)
    r = sm.OLS(T[y].values, X.values).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(T.bioguide)[0]})
    cb = np.zeros(X.shape[1]); cb[1], cb[2] = 1, -1
    tb = r.t_test(cb)
    return dict(a_released_minus_returned=float(r.params[1]), a_se=float(r.bse[1]), a_p=float(r.pvalues[1]),
                b_released_minus_departing_no_ballot=float(tb.effect[0]), b_se=float(tb.sd[0][0]), b_p=float(tb.pvalue),
                departing_no_ballot_minus_returned=float(r.params[2]), dnb_se=float(r.bse[2]), dnb_p=float(r.pvalues[2]),
                n=int(len(T)), members=int(T.bioguide.nunique()),
                n_by_group={k: int(v) for k, v in T.group.value_counts().items()},
                mean_by_group={k: float(v) for k, v in T.groupby("group")[y].mean().items()})


RES = {"counts": {str(k): int(v) for k, v in R.groupby(["cycle", "group"]).size().items()}}
P = R[R.has_post == 1].copy()
for y in ("L_ti_sd", "L_xd"):
    P[y + "_w"] = wins(P[y])
RES["D1"] = {y: group_model(y, P) for y in ("L_ti7", "L_lvol")}
RES["D1_pretrend"] = {y: group_model(y, P) for y in ("R_ti7", "R_lvol")}
RES["D3"] = {y: group_model(y + "_w", P) for y in ("L_ti_sd", "L_xd")}
for t, ys in (("D1", ("L_ti7", "L_lvol")), ("D3", ("L_ti_sd", "L_xd"))):
    hp = holm([RES[t][y]["a_p"] for y in ys])
    for y, h in zip(ys, hp):
        RES[t][y]["a_p_holm"] = h

RES["D2"] = {}
for y in ("R_ti_sd", "R_xd"):
    T = R[np.isfinite(R[y])].copy()
    T["y"] = wins(T[y])
    X = pd.DataFrame({"const": 1.0, "ballot": T.ballot.astype(float), "c2020": (T.cycle == 2020).astype(float),
                      "c2022": (T.cycle == 2022).astype(float)}, index=T.index)
    r = sm.OLS(T.y.values, X.values).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(T.bioguide)[0]})
    RES["D2"][y] = dict(b=float(r.params[1]), se=float(r.bse[1]), p=float(r.pvalues[1]), n=int(len(T)), members=int(T.bioguide.nunique()),
                        mean_on_ballot=float(T[T.ballot == 1].y.mean()), mean_off_ballot=float(T[T.ballot == 0].y.mean()),
                        sd_response=float(T.y.std()), mean_width_pre=float(T[y[2:] + "_p"].mean()),
                        mean_width_window=float(T[y[2:] + "_w"].mean()))
hp = holm([RES["D2"][y]["p"] for y in ("R_ti_sd", "R_xd")])
for y, h in zip(("R_ti_sd", "R_xd"), hp):
    RES["D2"][y]["p_holm"] = h
json.dump(RES, open(O + "/rvt_d_results.json", "w"), indent=1)
print(json.dumps(RES, indent=1))
