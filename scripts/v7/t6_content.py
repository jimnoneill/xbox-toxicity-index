"""T6 task 3: content (7-dimension Toxicity Index and each dimension) by regime and over time, personal and official.
TI7 = mean of the probabilities of offensive, toxic, neg_sentiment, irony, ragebait, tribal, performative (spec D, equal weights).
Primary post set: non-retweet posts (own text: tweet, quote, reply). Secondary: type == tweet only.
Regime contrasts: difference in mean score between flagged and unflagged days, 95% CI by day-cluster bootstrap (2000 resamples
of days); adjusted: OLS of post score on flags + day-of-week + natural spline of time (df 5), day-clustered SEs.
Outputs: content_regime_diff.csv, content_ols.csv, content_monthly.csv, content_numbers.json, fig2_content_trend.png
Associations only; no causal reading.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from t6_common import *  # noqa
import statsmodels.api as sm
from patsy import dmatrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

DIMS = ["offensive", "toxic", "neg_sentiment", "irony", "ragebait", "tribal", "performative"]
reg, ev = regime_frame()
FLAGS = ["in_session", "lee_voted", "recess", "shutdown", "own_race_2022", "post_2022", "natl_general_2024", "post_2024",
         "inaug_2wk", "r_president", "event_day"]
rng = np.random.default_rng(20261009)

pp = pd.read_parquet(ROOT / "output/v7/ti/post_probs_lee.parquet")
pp["id"] = pp["id"].astype(str)
P = load_posts("personal")[["id", "type", "et_date", "t"]]
P["id"] = P["id"].astype(str)
P = P.merge(pp[[f"p_{d}" for d in DIMS] + ["id", "ti_prob"]], on="id", how="left")
chk = float(np.abs(P[[f"p_{d}" for d in DIMS]].mean(axis=1) - P.ti_prob).max())

raw = pd.read_parquet(ROOT / "data/SenMikeLee_2022_2026.parquet", columns=["id"])
off = pd.DataFrame({"id": raw.id.astype(str)})
for d in DIMS:
    off[f"p_{d}"] = np.load(T6 / f"_probs_official/{d}.npy")
O = load_posts("official")[["id", "type", "et_date", "t"]]
O["id"] = O["id"].astype(str)
O = O.merge(off, on="id", how="left")
for D in (P, O):
    D["TI7"] = D[[f"p_{d}" for d in DIMS]].mean(axis=1)
SCORES = ["TI7"] + [f"p_{d}" for d in DIMS]
num = dict(ti_prob_equals_mean7_max_abs_diff=chk, personal_scored=int(P.TI7.notna().sum()), official_scored=int(O.TI7.notna().sum()))

rows, ols_rows, monthly = [], [], []
sp = dmatrix("cr(tday, df=5) - 1", {"tday": reg.tday.values}, return_type="dataframe")
sp.columns = [f"s{i}" for i in range(5)]
sp.index = reg.index
dowd = pd.get_dummies(reg.dow, prefix="dow", drop_first=True).astype(float)
Xday = pd.concat([reg[FLAGS].astype(float), dowd, sp], axis=1)

for acct, D in [("personal", P), ("official", O)]:
    for setname, mask in [("nonretweet", D.type != "retweet"), ("tweet_only", D.type == "tweet")]:
        d = D[mask & D.et_date.between(reg.index[0], reg.index[-1])].copy()
        d["day"] = (d.et_date - reg.index[0]).dt.days
        for sc in SCORES:
            ds = d.groupby("day")[sc].agg(["sum", "count"]).reindex(range(len(reg)), fill_value=0)
            S, C = ds["sum"].values, ds["count"].values
            tot_mean = S.sum() / C.sum()
            num[f"{acct}_{setname}_{sc}_overall"] = float(tot_mean)
            for fl in FLAGS:
                f = reg[fl].values.astype(bool)
                if C[f].sum() < 30:
                    continue
                diff = S[f].sum() / C[f].sum() - S[~f].sum() / C[~f].sum()
                bs = []
                for _ in range(2000):
                    k = rng.integers(0, len(f), len(f))
                    ff, Sk, Ck = f[k], S[k], C[k]
                    if Ck[ff].sum() == 0 or Ck[~ff].sum() == 0:
                        continue
                    bs.append(Sk[ff].sum() / Ck[ff].sum() - Sk[~ff].sum() / Ck[~ff].sum())
                lo, hi = np.percentile(bs, [2.5, 97.5])
                rows.append(dict(account=acct, postset=setname, score=sc, flag=fl, n_posts_flagged=int(C[f].sum()),
                                 mean_flagged=float(S[f].sum() / C[f].sum()), mean_unflagged=float(S[~f].sum() / C[~f].sum()),
                                 diff=float(diff), lo=float(lo), hi=float(hi)))
            if setname == "nonretweet":
                X = Xday.iloc[d.day.values].copy()
                X = X.loc[:, X.std() > 0]
                m = sm.OLS(d[sc].values, X.values).fit(cov_type="cluster", cov_kwds={"groups": d.day.values})
                ci = m.conf_int()
                for i, c in enumerate(X.columns):
                    if c in FLAGS:
                        ols_rows.append(dict(account=acct, score=sc, flag=c, coef=float(m.params[i]), lo=float(ci[i, 0]),
                                             hi=float(ci[i, 1]), p=float(m.pvalues[i]), n=len(d)))
        if setname == "nonretweet":
            d["month"] = d.et_date.dt.to_period("M").dt.to_timestamp()
            g = d.groupby("month")[SCORES].mean()
            g["n"] = d.groupby("month").size()
            g["account"] = acct
            monthly.append(g.reset_index())
            d["year"] = d.et_date.dt.year
            gy = d.groupby("year")[SCORES].mean()
            gy["n"] = d.groupby("year").size()
            num[f"{acct}_by_year"] = {int(y): {k: round(float(v), 4) for k, v in r.items()} for y, r in gy.iterrows()}

R = pd.DataFrame(rows)
R.to_csv(T6 / "content_regime_diff.csv", index=False)
pd.DataFrame(ols_rows).to_csv(T6 / "content_ols.csv", index=False)
M = pd.concat(monthly)
M.to_csv(T6 / "content_monthly.csv", index=False)
# headline regime rows for TI7
head = R[(R.postset == "nonretweet") & (R.score == "TI7")]
num["ti7_regime_diff"] = {a: {r.flag: [round(r["diff"], 4), round(r.lo, 4), round(r.hi, 4), r.n_posts_flagged] for _, r in g.iterrows()}
                          for a, g in head.groupby("account")}
oo = pd.DataFrame(ols_rows)
oo = oo[oo.score == "TI7"]
num["ti7_ols"] = {a: {r.flag: [round(r.coef, 4), round(r.lo, 4), round(r.hi, 4), round(r.p, 4)] for _, r in g.iterrows()}
                  for a, g in oo.groupby("account")}
# dimension with largest flagged-minus-unflagged difference per regime (personal)
(T6 / "content_numbers.json").write_text(json.dumps(num, indent=1, default=float))

# ------------------------------------------------------------------ Figure 2
dimcol = {"offensive": "#E69F00", "toxic": "#000000", "neg_sentiment": "#56B4E9", "irony": "#009E73", "ragebait": "#D55E00",
          "tribal": "#0072B2", "performative": "#CC79A7"}
fig, axs = plt.subplots(2, 1, figsize=(13, 8.5), sharex=True, gridspec_kw=dict(hspace=0.1))
mp = M[M.account == "personal"].set_index("month")
mo = M[M.account == "official"].set_index("month")
mp_ok, mo_ok = mp[mp.n >= 30], mo[mo.n >= 15]
axs[0].plot(mp_ok.index, mp_ok.TI7, color=PAL["personal"], lw=2, label="@BasedMikeLee (months with 30+ non-RT posts)")
axs[0].plot(mo_ok.index, mo_ok.TI7, color=PAL["official"], lw=2, label="@SenMikeLee (months with 15+ posts)")
axs[0].set_ylabel("Toxicity Index (7 dimensions), monthly mean")
axs[0].legend(fontsize=8, loc="upper left")
for dn in DIMS:
    axs[1].plot(mp_ok.index, mp_ok[f"p_{dn}"], color=dimcol[dn], lw=1.5, label=dn.replace("_", " "))
axs[1].set_ylabel("Mean probability, @BasedMikeLee")
axs[1].legend(fontsize=7, ncol=4, loc="upper left")
for ax in axs:
    ax.grid(alpha=0.25)
    ax.axvline(pd.Timestamp("2025-01-20"), color="k", lw=1.2)
    ax.axvspan(pd.Timestamp("2022-09-05"), pd.Timestamp("2022-11-08"), color=PAL["yellow"], alpha=0.3, lw=0)
    ax.axvspan(pd.Timestamp("2024-09-02"), pd.Timestamp("2024-11-05"), color=PAL["sky"], alpha=0.3, lw=0)
axs[1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
axs[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
axs[0].set_title("Content over time, non-retweet posts. Amber: own 2022 general window; blue: 2024 general window; black: inauguration 2025-01-20",
                 fontsize=9)
fig.savefig(T6 / "fig2_content_trend.png", dpi=160, bbox_inches="tight")
print(json.dumps({k: v for k, v in num.items() if k.endswith("by_year") or "ti7" in k or k.startswith(("ti_", "personal_sc", "official_sc"))}, indent=1)[:6000])
