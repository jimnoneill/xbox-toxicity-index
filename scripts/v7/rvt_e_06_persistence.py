"""RVT-E description E6: how long a member's own deviation lasts. Follows scripts/v7/rvt_e_DESIGN.md.
Run from the scripts/v7 folder: the member-day state is built by the first block of rvt_c_01_state.py.
The party-week and member means are computed once on the full sample and not recomputed inside the bootstrap."""
import json
import os

import numpy as np
import pandas as pd

exec(open("rvt_c_01_state.py").read().split("# ---------------------------------------------------------------- responses to the general-election window")[0])
O = ROOT + "/rvt_e"
os.makedirs(O, exist_ok=True)
rng = np.random.default_rng(20261112)
LAGS = [1, 2, 4, 8, 13, 26, 52, 104]
NB = 500

D["week"] = D.date - pd.to_timedelta(D.date.dt.dayofweek, unit="D")
D["odd"] = (D.date.dt.dayofweek % 2 == 1)
for tag, sel in (("", slice(None)), ("_o", D.odd), ("_e", ~D.odd)):
    x = D[sel] if tag else D
    g = x.groupby(["bioguide", "week"]).agg(**{"ti_sum" + tag: ("ti_sum", "sum"), "n" + tag: ("n", "sum"), "events" + tag: ("events", "sum"),
                                              "days" + tag: ("date", "size")})
    Wk = g if not tag else Wk.join(g)
Wk = Wk.join(D.groupby(["bioguide", "week"]).party.first()).reset_index()
Wk = Wk[Wk.days == 7].copy()
Wk["ti7"] = np.where(Wk.n >= 5, Wk.ti_sum / Wk.n.where(Wk.n > 0), np.nan)
Wk["ti7_o"] = np.where(Wk.n_o >= 3, Wk.ti_sum_o / Wk.n_o.where(Wk.n_o > 0), np.nan)
Wk["ti7_e"] = np.where(Wk.n_e >= 3, Wk.ti_sum_e / Wk.n_e.where(Wk.n_e > 0), np.nan)
Wk["lev"] = np.log1p(Wk.events)
print("member-weeks", len(Wk), "with TI7", int(Wk.ti7.notna().sum()), "members", Wk.bioguide.nunique(), flush=True)


def resid(col, ref=None):
    """subtract the party-week mean and then the member mean (those of ref when given)"""
    ref = ref or col
    pw = Wk.groupby(["party", "week"])[ref].transform("mean")
    u = Wk[col] - pw
    mm = (Wk[ref] - pw).groupby(Wk.bioguide).transform("mean")
    return u - mm


Wk["u_ti7"], Wk["u_lev"] = resid("ti7"), resid("lev")
Wk["u_ti7_o"], Wk["u_ti7_e"] = resid("ti7_o", "ti7"), resid("ti7_e", "ti7")


def suff(x, y, ids):
    """per-member sufficient statistics of the pairs (x, y)"""
    ok = np.isfinite(x) & np.isfinite(y)
    d = pd.DataFrame({"b": ids[ok], "n": 1.0, "x": x[ok], "y": y[ok], "xx": x[ok] ** 2, "yy": y[ok] ** 2, "xy": x[ok] * y[ok]})
    return d.groupby("b").sum()


def r_of(S):
    n, sx, sy, sxx, syy, sxy = S.n.sum(), S.x.sum(), S.y.sum(), S.xx.sum(), S.yy.sum(), S.xy.sum()
    return (sxy - sx * sy / n) / np.sqrt((sxx - sx ** 2 / n) * (syy - sy ** 2 / n))


def with_boot(S):
    r = float(r_of(S))
    idx = np.arange(len(S))
    A = S.values
    cols = list(S.columns)
    bs = []
    for _ in range(NB):
        t = A[rng.choice(idx, len(idx), replace=True)].sum(0)
        n, sx, sy, sxx, syy, sxy = (t[cols.index(k)] for k in ("n", "x", "y", "xx", "yy", "xy"))
        bs.append((sxy - sx * sy / n) / np.sqrt((sxx - sx ** 2 / n) * (syy - sy ** 2 / n)))
    return dict(r=r, ci95=[float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))], n_pairs=int(S.n.sum()), members=int(len(S)))


RES = {}
# reliability of the weekly TI7 residual from odd and even days of the week
sh = with_boot(suff(Wk.u_ti7_o.values, Wk.u_ti7_e.values, Wk.bioguide.values))
RES["ti7_weekly_split_half"] = dict(sh, reliability=2 * sh["r"] / (1 + sh["r"]),
                                    reliability_ci95=[2 * x / (1 + x) for x in sh["ci95"]])
rel = RES["ti7_weekly_split_half"]["reliability"]
for v in ("u_ti7", "u_lev"):
    RES[v] = {}
    base = Wk[["bioguide", "week", v]]
    for k in LAGS:
        sh_ = base.assign(week=base.week + pd.Timedelta(weeks=k)).rename(columns={v: "lagged"})
        m = base.merge(sh_, on=["bioguide", "week"])
        d = with_boot(suff(m[v].values, m.lagged.values, m.bioguide.values))
        if v == "u_ti7":
            d["r_over_reliability"] = d["r"] / rel
        RES[v][str(k)] = d
    first = next((k for k in LAGS if RES[v][str(k)]["ci95"][0] <= 0), None)
    RES[v]["first_lag_with_interval_including_zero_weeks"] = first
RES["sample"] = dict(member_weeks=int(len(Wk)), member_weeks_with_ti7=int(Wk.ti7.notna().sum()), members=int(Wk.bioguide.nunique()),
                     sd_u_ti7=float(Wk.u_ti7.std()), sd_u_lev=float(Wk.u_lev.std()))
json.dump(RES, open(O + "/e6_persistence.json", "w"), indent=1)
print(json.dumps(RES, indent=1))
