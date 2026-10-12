"""Summaries for the model comparison. Reads members_models.csv, bootstrap.csv, member_features.csv."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/v7/temporal/model_compare"
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 50)

mm = pd.read_csv(OUT / "members_models.csv")
if "err" in mm:
    print("errors:", mm.err.notna().sum())
    print(mm[mm.err.notna()].groupby("model").size())
    mm = mm[mm.err.isna()]
lee = mm[mm.bg == "LEE"]
mm = mm[mm.bg != "LEE"].copy()
mf = pd.read_csv(ROOT / "output/v7/temporal/member_features.csv")
mf = mf[(mf.level == "member") & (mf.window == "full") & (mf.postset == "nonrt")].set_index("id")
mm["aic_rank"] = mm.groupby("bg").aic.rank()
mm["bic_rank"] = mm.groupby("bg").bic.rank()
ref = mm[mm.model == "hawkes_exp"].set_index("bg")
mm["d_aic_vs_hawkes"] = mm.aic - mm.bg.map(ref.aic)
mm["pass05"] = mm.p > 0.05
mm["pass01"] = mm.p > 0.01
nmem = mm.bg.nunique()
print("members", nmem)

summ = mm.groupby("model").agg(
    n=("bg", "size"), pass05=("pass05", "mean"), pass01=("pass01", "mean"), medD=("D", "median"), medMAD=("mad", "median"),
    med_aic_rank=("aic_rank", "median"), mean_aic_rank=("aic_rank", "mean"), med_bic_rank=("bic_rank", "median"),
    med_dAIC_vs_hawkes=("d_aic_vs_hawkes", "median"), med_k=("k", "median"), med_secs=("secs", "median"))
best = mm.loc[mm.groupby("bg").aic.idxmin()].model.value_counts()
bestb = mm.loc[mm.groupby("bg").bic.idxmin()].model.value_counts()
summ["aic_best_share"] = (best / nmem).reindex(summ.index).fillna(0)
summ["bic_best_share"] = (bestb / nmem).reindex(summ.index).fillna(0)
summ = summ.sort_values("med_aic_rank")
summ.round(4).to_csv(OUT / "summary_by_model.csv")
print(summ.round(3).to_string())

# bootstrap
bf = OUT / "bootstrap.csv"
if bf.exists():
    bt = pd.read_csv(bf)
    if "err" in bt:
        bt = bt[bt.err.isna()]
    bs = bt.groupby("model").agg(n=("bg", "size"), ks_pass=("D", lambda x: np.nan), boot_pass05=("boot_p", lambda x: (x > 0.05).mean()),
                                 boot_pass01=("boot_p", lambda x: (x > 0.01).mean()), madboot_pass05=("boot_mad_p", lambda x: (x > 0.05).mean()),
                                 med_boot_p=("boot_p", "median"))
    kk = mm[mm.bg.isin(bt.bg.unique())].groupby("model").pass05.mean()
    bs["ks_pass05_same_members"] = kk
    bs = bs.drop(columns="ks_pass").sort_values("boot_pass05", ascending=False)
    bs.round(3).to_csv(OUT / "bootstrap_summary.csv")
    print("\nBOOTSTRAP (B per member =", int(bt.B.median()), ", min possible p =", round(1 / (1 + bt.B.median()), 3), ")")
    print(bs.round(3).to_string())

# groups
feat = mf[["n", "B", "rapid_share_300", "share_phone", "share_tool", "party", "chamber", "hawkes_eta"]]
mm = mm.join(feat.rename(columns={"n": "n_feat"}), on="bg")
mm["vol"] = pd.qcut(mm.n, 3, labels=["low", "mid", "high"])
mm["src"] = np.where(mm.share_phone >= 0.6, "phone", np.where(mm.share_tool >= 0.6, "tool", "mixed/other"))
print("\nmembers per volume tercile:", mm[mm.model == "hawkes_exp"].vol.value_counts().to_dict())
print("members per source group:", mm[mm.model == "hawkes_exp"].src.value_counts().to_dict())
for grp in ("vol", "src"):
    t = mm.groupby([grp, "model"], observed=True).agg(pass05=("pass05", "mean"), med_rank=("aic_rank", "median"), medMAD=("mad", "median")).reset_index()
    pv = t.pivot(index="model", columns=grp, values=["pass05", "med_rank"])
    pv.round(3).to_csv(OUT / f"by_{grp}.csv")
    print(f"\nBY {grp}")
    print(pv.round(3).sort_values(("med_rank", pv["med_rank"].columns[0])).to_string())
    bw = mm.loc[mm.groupby("bg").aic.idxmin()].groupby([grp, "model"], observed=True).size().unstack(0).fillna(0)
    print("AIC winner counts by", grp)
    print(bw.astype(int).to_string())

# Lee
print("\nLEE (n =", int(lee.n.iloc[0]), ")")
lee = lee.copy()
lee["aic_rank"] = lee.aic.rank()
print(lee[["model", "ll", "k", "aic", "bic", "D", "p", "mad", "aic_rank", "secs"]].sort_values("aic").round(4).to_string(index=False))
lee.to_csv(OUT / "lee_models.csv", index=False)

# parameters of interpretable models
def param_frame(model):
    s = mm[mm.model == model].set_index("bg")
    P = pd.DataFrame([json.loads(x) for x in s.params], index=s.index)
    return s, P


for model in ("casc2", "casc3"):
    s, P = param_frame(model)
    P["posts_per_session"] = P.posts_per_session
    X = P.join(mf[["B", "rapid_share_300", "hawkes_eta"]]).join(s[["n", "pass05", "D"]])
    X["logn"] = np.log(X.n)
    cols = ["session_rate_per_day", "posts_per_session", "within_rate_per_min", "idle_gap_share"]
    print(f"\nPARAMETERS {model}: quantiles over members")
    print(X[cols].quantile([.1, .25, .5, .75, .9]).round(3).to_string())
    print("Spearman with burstiness B, rapid_share_300, n, hawkes_eta")
    rows = []
    for c in cols:
        rows.append([c] + [round(spearmanr(X[c], X[o], nan_policy="omit")[0], 3) for o in ("B", "rapid_share_300", "n", "hawkes_eta")])
    print(pd.DataFrame(rows, columns=["param", "B", "rapid300", "n", "eta"]).to_string(index=False))
    X.round(5).to_csv(OUT / f"params_{model}.csv")
# Lee cascade params
for model in ("casc2", "casc3"):
    r = lee[lee.model == model]
    if len(r):
        print("Lee", model, r.params.iloc[0])
