"""RVT-E test E5 (exploratory): feedback and the next day's posting on the one account with engagement counts.
Follows scripts/v7/rvt_e_DESIGN.md. Rule added for construction, before any estimate: the trailing 28-day median of TI7
needs at least 20 posts, and the trailing 7-day means need at least 3 days with posts."""
import json
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

ROOT = os.path.expanduser("~/x_box/pipeline/output/v7")
DATA = os.path.expanduser("~/x_box/pipeline/data")
O = ROOT + "/rvt_e"
os.makedirs(O, exist_ok=True)
D7 = ["offensive", "toxic", "neg_sentiment", "irony", "ragebait", "tribal", "performative"]

pp = pd.read_parquet(ROOT + "/ti/post_probs_lee.parquet", columns=["id", "created_at", "type"] + [f"p_{d}" for d in D7])
ar = pd.read_parquet(DATA + "/BasedMikeLee_archive_v2.parquet", columns=["id", "like_count", "retweet_count"])
pp["id"], ar["id"] = pp.id.astype(str), ar.id.astype(str)
pp = pp.merge(ar.drop_duplicates("id"), on="id", how="inner")
pp = pp[pp.type != "retweet"].copy()
pp["ti7"] = pp[[f"p_{d}" for d in D7]].mean(axis=1)
ts = pd.to_datetime(pp.created_at, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
pp["date"], pp["ym"] = ts.dt.normalize(), ts.dt.to_period("M").astype(str)
pp["f"] = np.log1p(pp.like_count.fillna(0) + pp.retweet_count.fillna(0))
pp["fr"] = pp.f - pp.groupby(["type", "ym"]).f.transform("mean")
pp = pp.sort_values("date")
print("non-repost posts", len(pp), pp.date.min().date(), pp.date.max().date(), pp.type.value_counts().to_dict(), flush=True)

days = pd.date_range(pp.date.min(), pp.date.max())
g = pp.groupby("date")
T = pd.DataFrame({"n": g.size(), "ti7": g.ti7.mean(), "F": g.fr.mean()}).reindex(days)
T["n"] = T.n.fillna(0)
T["ln"] = np.log1p(T.n)

# premium of provocation: posts above the trailing 28-day median of TI7 against the rest
prem = {}
bydate = {d: x for d, x in pp.groupby("date")}
for d in days:
    if d not in bydate:
        continue
    past = pp[(pp.date >= d - pd.Timedelta(days=28)) & (pp.date < d)]
    if len(past) < 20:
        continue
    x = bydate[d]
    hi = x.ti7 > past.ti7.median()
    if hi.sum() >= 2 and (~hi).sum() >= 2:
        prem[d] = x.fr[hi].mean() - x.fr[~hi].mean()
T["P"] = pd.Series(prem)


def trail(s, minobs=3):
    return s.shift(1).rolling(7, min_periods=minobs).mean()


T["S"] = T.F - trail(T.F)
T["ti7_trail"], T["ln_trail"] = trail(T.ti7), trail(T.ln)
T.to_csv(O + "/e5_daily.csv")


def model(y_next, pred, controls, need_two_posts):
    X = pd.DataFrame({"x": pred})
    for k, v in controls.items():
        X[k] = v
    y = y_next.shift(-1)                      # outcome on day t+1, row is day t
    ok = y.notna() & X.notna().all(1)
    if need_two_posts:
        ok &= (T.n >= 2) & (T.n.shift(-1) >= 2)
    idx = T.index[ok]
    nxt = idx + pd.Timedelta(days=1)
    dow = pd.get_dummies(nxt.dayofweek, drop_first=True).astype(float).values
    mon = pd.get_dummies(nxt.to_period("M").astype(str), drop_first=True).astype(float).values
    Z = np.column_stack([np.ones(len(idx)), X.loc[idx].values, dow, mon])
    r = sm.OLS(y.loc[idx].values, Z).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
    sx, sy = float(X.loc[idx, "x"].std()), float(y.loc[idx].std())
    return dict(b=float(r.params[1]), se=float(r.bse[1]), p=float(r.pvalues[1]), n_days=int(len(idx)), sd_predictor=sx, sd_outcome=sy,
                effect_of_1sd_in_outcome_sd=float(r.params[1] * sx / sy), effect_of_1sd_raw=float(r.params[1] * sx))


RES = {}
c_content = lambda: {"y_t": T.ti7, "y_trail": T.ti7_trail, "ln_t": T.ln}
c_volume = lambda: {"y_t": T.ln, "y_trail": T.ln_trail}
RES["content_lag"] = model(T.ti7, T.P, c_content(), True)
RES["content_lead_placebo"] = model(T.ti7, T.P.shift(-2), c_content(), True)
RES["volume_lag"] = model(T.ln, T.S, c_volume(), False)
RES["volume_lead_placebo"] = model(T.ln, T.S.shift(-2), c_volume(), False)
hp = multipletests([RES["content_lag"]["p"], RES["volume_lag"]["p"]], method="holm")[1]
RES["content_lag"]["p_holm"], RES["volume_lag"]["p_holm"] = float(hp[0]), float(hp[1])
RES["sample"] = dict(posts=int(len(pp)), first_day=str(pp.date.min().date()), last_day=str(pp.date.max().date()), days=int(len(T)),
                     days_with_premium=int(T.P.notna().sum()), days_with_surprise=int(T.S.notna().sum()),
                     by_type={k: int(v) for k, v in pp.type.value_counts().items()})
json.dump(RES, open(O + "/e5_feedback_single_account.json", "w"), indent=1)
print(json.dumps(RES, indent=1))
