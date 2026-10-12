"""Post-hoc robustness for (a) (labelled exploratory: written after the final test) + figures + merged numbers."""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE.parents[1] / "xbox"))
sys.path.insert(0, str(Path.home() / "x_box/pylib_v7"))
import numpy as np, pandas as pd
import statsmodels.api as sm
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import kistner_ext as K
OUT = Path.home() / "x_box/pipeline/output/v7/kistner"

p = K.build_panel(("Twitter",)); p = p[(p.n >= K.MIN_N) & (p.Year >= 2010)].copy(); p["chamber"] = p.chamber.fillna("House")
pw = []
for Y, g in p.groupby("Year"):
    f = K.power_features(Y, g.party.values, g.chamber.values); f.index = g.index; pw.append(f)
p = pd.concat([p, pd.concat(pw)], axis=1)
p["out_pres"] = 1 - p.p_pres; p["out_ch"] = 1 - p.p_ch; p["out_oth"] = 1 - p.p_oth
p["t"] = p.Year - 2010; p["isR"] = (p.party == "R").astype(float)


def fe_fit(d, xs, trend=False):
    X = d[xs].copy()
    if trend:
        X["t"] = d.t
    Xd = X - X.groupby(d.MemberICPSR).transform("mean"); yd = d.share - d.share.groupby(d.MemberICPSR).transform("mean")
    keep = []
    for c in Xd.columns:
        if Xd[c].std() > 1e-9 and np.linalg.matrix_rank(Xd[keep + [c]].values) == len(keep) + 1:
            keep.append(c)
    m = sm.OLS(yd.values, Xd[keep].values).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.MemberICPSR)[0]})
    return {c: (float(m.params[keep.index(c)]), float(m.bse[keep.index(c)])) if c in keep else None for c in xs}


ex = {}
# by party (symmetry): own-party linear trend allowed
for lab, d in [("D_only", p[p.party == "D"]), ("R_only", p[p.party == "R"])]:
    ex[lab] = {"no_trend": fe_fit(d, ["out_pres", "out_ch", "out_oth"]), "linear_trend": fe_fit(d, ["out_pres", "out_ch", "out_oth"], True)}
# leave-one-year-out jackknife for out_pres (S2-style with party trends)
jk = {}
for Y in range(2010, 2023):
    d = p[p.Year != Y].copy()
    d["tR"] = d.t * d.isR; d["tD"] = d.t * (1 - d.isR)
    X = d[["out_pres", "out_ch", "out_oth", "tR", "tD"]]
    Xd = X - X.groupby(d.MemberICPSR).transform("mean"); yd = d.share - d.share.groupby(d.MemberICPSR).transform("mean")
    m = sm.OLS(yd.values, Xd.values).fit()
    jk[Y] = {"out_pres": float(m.params[0]), "out_ch": float(m.params[1]), "out_oth": float(m.params[2])}
ex["jackknife_drop_year_S2"] = jk
# regime-level (party-year) cell means: out-pres contrast with party-year cluster bootstrap (cells = party x year)
cells = p.groupby(["Year", "party"]).agg(s=("share", "mean"), op=("out_pres", "first")).reset_index()
cells["isR"] = (cells.party == "R").astype(int)
ex["party_year_cells"] = {"n_cells": int(len(cells)), "mean_out": float(cells[cells.op == 1].s.mean()), "mean_in": float(cells[cells.op == 0].s.mean())}
# cell-level regression with party FE + year FE, cluster by year (13 clusters)
Xc = pd.get_dummies(cells.Year.astype(int), prefix="y", drop_first=True).astype(float)
Xc["R"] = cells.isR.values; Xc["op"] = cells.op.values
mc = sm.OLS(cells.s.values, sm.add_constant(Xc).values).fit(cov_type="cluster", cov_kwds={"groups": cells.Year.values})
ex["cell_regression_partyFE_yearFE_cluster_year"] = {"out_pres": float(mc.params[-1]), "se": float(mc.bse[-1])}
json.dump(ex, open(OUT / "assoc_extra.json", "w"), indent=1)
print(json.dumps(ex)[:3000])

# ---- figures
fig, ax = plt.subplots(figsize=(7.5, 4))
g = p.groupby(["Year", "party"]).apply(lambda x: x.neg.sum() / x.n.sum(), include_groups=False).unstack()
ax.plot(g.index, g["D"], color="#1f4e9c", marker="o", label="Democrats"); ax.plot(g.index, g["R"], color="#b5282e", marker="o", label="Republicans")
for y0, y1, lab in [(2010, 2010.5, ""), (2017, 2021, "R hold presidency"), (2021, 2023, "D hold presidency")]:
    pass
ax.axvspan(2016.5, 2020.5, color="#b5282e", alpha=0.07); ax.axvspan(2020.5, 2022.5, color="#1f4e9c", alpha=0.07)
ax.axvspan(2009.5, 2016.5, color="#1f4e9c", alpha=0.07)
ax.axvline(2018.5, ls="--", color="k", lw=1); ax.text(2018.6, ax.get_ylim()[1] * 0.92, "test block", fontsize=8)
ax.set_ylabel("share of tweets classified negative-partisan"); ax.legend(frameon=False); ax.set_title("Twitter negative-partisan share by party (shading = party holding presidency)")
fig.tight_layout(); fig.savefig(OUT / "fig1_party_year.png", dpi=200); fig.savefig(OUT / "fig1_party_year.pdf"); plt.close(fig)

r = json.load(open(OUT / "numbers_kistner_final.json"))["tw"]
ms = ["RTM", "R_ctrl", "R_power", "R_both", "G_ctrl", "G_both"]
fig, ax = plt.subplots(figsize=(6.5, 3.8))
v = [r[m]["skill_vs_persist"] for m in ms]; lo = [r[m]["skill_vs_persist"] - r[m]["lo"] for m in ms]; hi = [r[m]["hi"] - r[m]["skill_vs_persist"] for m in ms]
ax.bar(ms, v, yerr=[lo, hi], color=["#999", "#999", "#2a7f62", "#2a7f62", "#999", "#2a7f62"], capsize=3)
ax.axhline(0, color="k", lw=0.8); ax.set_ylabel("skill vs persistence (RMSE)"); ax.set_title("Final test 2019-2022, Twitter measure (green = uses power status)")
fig.tight_layout(); fig.savefig(OUT / "fig2_final_skill.png", dpi=200); fig.savefig(OUT / "fig2_final_skill.pdf"); plt.close(fig)

dt = pd.read_csv(OUT / "direction_final_tw.csv")
fig, ax = plt.subplots(figsize=(6, 4))
for pty, col in [("D", "#1f4e9c"), ("R", "#b5282e")]:
    ax.scatter(dt[f"R_both_{pty}"], dt[f"act_{pty}"], color=col, label=pty)
    for _, rr in dt.iterrows():
        ax.annotate(int(rr.Y), (rr[f"R_both_{pty}"], rr[f"act_{pty}"]), fontsize=8, xytext=(3, 3), textcoords="offset points")
lim = [-0.15, 0.2]; ax.plot(lim, lim, "k--", lw=0.8); ax.axhline(0, lw=0.5, color="grey"); ax.axvline(0, lw=0.5, color="grey")
ax.set_xlabel("predicted mean change (R_both)"); ax.set_ylabel("realized mean change"); ax.legend(frameon=False); ax.set_title("Party-year mean change, test years")
fig.tight_layout(); fig.savefig(OUT / "fig3_direction.png", dpi=200); fig.savefig(OUT / "fig3_direction.pdf"); plt.close(fig)

# merged numbers
num = {"assoc": json.load(open(OUT / "assoc.json")), "assoc_extra": ex, "dev": json.load(open(OUT / "dev_results.json")),
       "frozen": json.load(open(OUT / "frozen.json")), "final": json.load(open(OUT / "numbers_kistner_final.json")),
       "leak_test": json.load(open(OUT / "leak_test.json")), "panel_counts": json.load(open(OUT / "panel_counts.json"))}
json.dump(num, open(OUT / "numbers_kistner.json", "w"), indent=1, default=float)
