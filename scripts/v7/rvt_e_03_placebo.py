"""RVT-E test E3: arbitrary text features and all scores in the C4 model. Follows scripts/v7/rvt_e_DESIGN.md.
Run from the scripts/v7 folder: the member-day state is built by the first block of rvt_c_01_state.py."""
import json
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

exec(open("rvt_c_01_state.py").read().split("# ---------------------------------------------------------------- responses to the general-election window")[0])
O = ROOT + "/rvt_e"
os.makedirs(O, exist_ok=True)
DATA = os.path.expanduser("~/x_box/pipeline/data")
D7 = ["offensive", "toxic", "neg_sentiment", "irony", "ragebait", "tribal", "performative"]
OTHER = ["advocacy", "genuine_anger", "emotion_anger", "hate", "epistemic", "engbait", "agency"]
ARB = ["P1_xyz", "P2_n_to_z", "P3_vowels", "P4_odd_letters"]
CACHE = O + "/e3_member_day_features.parquet"

if not os.path.exists(CACHE):
    pp = pd.read_parquet(ROOT + "/full/post_probs_full.parquet", columns=["id", "screen_name", "created_at", "post_type"] + [f"p_{d}" for d in D7 + OTHER])
    pp = pp[pp.post_type != "retweet"]
    am = pd.read_csv(ROOT + "/ti/account_member_map.csv")[["screen_name", "bioguide"]].dropna(subset=["bioguide"])
    pp = pp.merge(am, on="screen_name")
    tx = pd.read_parquet(DATA + "/congresstweets_full/congresstweets_full.parquet", columns=["id", "text"]).drop_duplicates("id")
    pp = pp.merge(tx, on="id", how="left")
    del tx
    print("non-repost posts mapped", len(pp), "with text", int(pp.text.notna().sum()), flush=True)
    s = pp.text.fillna("").str.lower()
    s = s.str.replace(r"https?://\S+|www\.\S+|@\w+|&[a-z#0-9]+;", " ", regex=True)
    nlet = s.str.count(r"[a-z]")
    feats = {"P1_xyz": r"[xyz]", "P2_n_to_z": r"[n-z]", "P3_vowels": r"[aeiou]", "P4_odd_letters": r"[acegikmoqsuwy]"}
    ok = nlet >= 20
    for k, pat in feats.items():
        pp[k] = np.where(ok, s.str.count(pat) / nlet.where(ok), np.nan)
    pp["loglen"] = np.where(nlet >= 1, np.log(nlet.where(nlet >= 1)), np.nan)
    del s
    pp["ti7"] = pp[[f"p_{d}" for d in D7]].mean(axis=1).astype("float64")
    pp["date"] = pd.to_datetime(pp.created_at, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    cols = ["ti7"] + [f"p_{d}" for d in D7 + OTHER] + ARB + ["loglen"]
    a = {}
    for c in cols:
        a["s_" + c] = (c, "sum")
        a["n_" + c] = (c, "count")
    G = pp.groupby(["bioguide", "date"]).agg(**a).reset_index()
    G["bioguide"] = G.bioguide.astype(str)
    G.to_parquet(CACHE)
    del pp
G = pd.read_parquet(CACHE)
COLS = [c[2:] for c in G.columns if c.startswith("s_")]
F = D[["bioguide", "date", "events"]].merge(G, on=["bioguide", "date"], how="left").fillna(0)
print("check: posts in the two builds", int(D.n.sum()), int(F.n_ti7.sum()), flush=True)


def means(df):
    g = df.groupby("bioguide")[[c for c in df.columns if c[:2] in ("s_", "n_")]].sum()
    out = pd.DataFrame({c: np.where(g["n_" + c] >= 10, g["s_" + c] / g["n_" + c].where(g["n_" + c] > 0), np.nan) for c in COLS}, index=g.index)
    out["days"] = df.groupby("bioguide").size()
    return out


def displacement(shift):
    out = []
    for tr in ("2019-01-03", "2021-01-20", "2023-01-03"):
        t0 = pd.Timestamp(tr)
        t = t0 - pd.Timedelta(days=shift)
        a = means(F[(F.date >= t) & (F.date < t + pd.Timedelta(days=180))])
        b = means(F[(F.date >= t - pd.Timedelta(days=180)) & (F.date < t)])
        m = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
        m = m[(m.days_a >= 170) & (m.days_b >= 170)]
        p0 = cal[cal.date == t0 - pd.Timedelta(days=1)].set_index("bioguide")
        p1 = cal[cal.date == t0].set_index("bioguide")
        dp = ((p1.aligned_with_president + p1.in_chamber_majority) - (p0.aligned_with_president + p0.in_chamber_majority)).dropna()
        m = m.join(dp.rename("dpower"), how="inner")
        m = m[m.dpower != 0]
        for c in COLS:
            m["d_" + c] = m[c + "_a"] - m[c + "_b"]
        m["trans"], m["lost"] = tr, (m.dpower < 0).astype(float)
        out.append(m.reset_index()[["bioguide", "trans", "lost"] + ["d_" + c for c in COLS]])
    return pd.concat(out, ignore_index=True)


RES = {}
sd_real = {}
for name, sh in (("real", 0), ("placebo_one_year_before", 365)):
    M = displacement(sh)
    M.to_csv(O + f"/e3_displacement_{name}.csv", index=False)
    RES[name] = {}
    for c in COLS:
        y = M["d_" + c].values
        X = pd.concat([pd.Series(1.0, index=M.index), M.lost, pd.get_dummies(M.trans, drop_first=True).astype(float)], axis=1).values
        ok = np.isfinite(y)
        r = sm.OLS(y[ok], X[ok]).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(M.bioguide[ok])[0]})
        if name == "real":
            sd_real[c] = float(np.nanstd(y, ddof=1))
        RES[name][c] = dict(lost_minus_gained=float(r.params[1]), se=float(r.bse[1]), t=float(r.tvalues[1]), p=float(r.pvalues[1]),
                            n=int(ok.sum()), members=int(M.bioguide[ok].nunique()), sd_displacement=float(np.nanstd(y, ddof=1)),
                            standardized=float(r.params[1] / sd_real[c]),
                            by_change={tr: float(M[(M.trans == tr) & (M.lost == 1)]["d_" + c].mean() - M[(M.trans == tr) & (M.lost == 0)]["d_" + c].mean())
                                       for tr in ("2019-01-03", "2021-01-20", "2023-01-03")})
    hp = multipletests([RES[name][c]["p"] for c in ARB], method="holm")[1]
    for c, h in zip(ARB, hp):
        RES[name][c]["p_holm_arbitrary"] = float(h)
stored = json.load(open(ROOT + "/rvt_c/rvt_c_results.json"))["C4"]
RES["check_reproduces_C4"] = dict(ti7_here=RES["real"]["ti7"]["lost_minus_gained"], ti7_stored=stored["real"]["d_ti7"]["lost_minus_gained"],
                                  ti7_placebo_here=RES["placebo_one_year_before"]["ti7"]["lost_minus_gained"],
                                  ti7_placebo_stored=stored["placebo_one_year_before"]["d_ti7"]["lost_minus_gained"])
big = max(abs(RES["real"][c]["standardized"]) for c in ARB)
RES["reading"] = dict(any_arbitrary_holm_sig=bool(any(RES["real"][c]["p_holm_arbitrary"] < 0.05 for c in ARB)),
                      largest_abs_standardized_arbitrary=big, ti7_abs_standardized=abs(RES["real"]["ti7"]["standardized"]),
                      ti7_multiple_of_largest_arbitrary=abs(RES["real"]["ti7"]["standardized"]) / big)
json.dump(RES, open(O + "/e3_placebo_outcomes.json", "w"), indent=1)
print(json.dumps(RES, indent=1))
