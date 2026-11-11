"""Figures 6 and 7 redrawn with skill in percent and labels spelled out. Run on hpcf inside
manuscript/epjds_submission_work/fig_relabel_v42. The data files and the plotted values are those of
scripts/v7/figs_03_forecast.py (its Fig 5 block is manuscript Figure 6) and fig_relabel_v30/fig7_v3.py."""
import json
import sys
from pathlib import Path

H = Path(__file__).resolve().parent
sys.path.insert(0, "/home/joneill/x_box/pipeline/scripts/v7")
import figs_common  # noqa: E402

figs_common.OUT = str(H)
from figs_common import *  # noqa: E402,F401,F403

style()

# ------------------------------------------------------------------ Figure 6 (volume forecasts and bursts)
P2 = json.load(open(V7 + "/p2/numbers_p2.json"))["sample_test"]
hz = [("h1", "Next hour"), ("d24", "Next 24 hours"), ("w168", "Next week")]
models = [("nb_nocong", "Negative binomial", SKY), ("nb", "Negative binomial with\nCongress-wide activity", BLUE),
          ("gbm", "Gradient-boosted trees", ORANGE), ("hawkes", "Hawkes", GREEN)]
base_names = {"profmean": "hour-of-week\nprofile", "pers": "preceding\nweek"}
fig, axs = plt.subplots(1, 2, figsize=(W2, 3.4), gridspec_kw={"width_ratios": [2.3, 1], "wspace": 0.3})
ax = axs[0]
labs = []
for hi, (h, hl) in enumerate(hz):
    bl = P2["ls"][h]["all"]["nb"]["baseline"]
    labs.append(f"{hl}\n(baseline: {base_names[bl]})")
    for mi, (m, ml, c) in enumerate(models):
        r = P2["ls"][h]["all"][m]
        x = hi + (mi - 1.5) * 0.17
        ax.errorbar(x, 100 * r["skill"], yerr=[[100 * (r["skill"] - r["lo"])], [100 * (r["hi"] - r["skill"])]], fmt="o", ms=4.5, color=c,
                    lw=1.2, capsize=0, label=ml if hi == 0 else None)
        print("fig6", h, m, round(100 * r["skill"], 2), round(100 * r["lo"], 2), round(100 * r["hi"], 2), "baseline", bl)
ax.axhline(0, color="black", lw=0.7)
ax.set_xticks(range(3))
ax.set_xticklabels(labs, fontsize=6.8)
ax.set_ylabel("Log-score skill over baseline (%)")
ax.set_ylim(-0.5, 12.5)
ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.3), ncol=2, fontsize=7, columnspacing=1.2, handletextpad=0.2, frameon=False)
plab(ax, "a", dx=-0.13)
ax = axs[1]
bm = [("snaive", "Same window\none week earlier", GREY), ("pers", "Persistence", GREY), ("logit", "Logistic\nregression", PINK),
      ("gbm", "Gradient-\nboosted trees", ORANGE), ("hawkes", "Hawkes", GREEN)]
for i, (m, ml, c) in enumerate(bm):
    r = P2["burst"][m]
    ax.errorbar(i, r["auc_macro"], yerr=[[r["auc_macro"] - r["auc_macro_lo"]], [r["auc_macro_hi"] - r["auc_macro"]]], fmt="o", ms=4.5,
                color=c, lw=1.2, capsize=0)
    print("fig6 burst", m, round(r["auc_macro"], 4))
ax.axhline(0.5, color="black", lw=0.7, ls=":")
ax.set_xticks(range(len(bm)))
ax.set_xticklabels([b[1] for b in bm], rotation=45, ha="right", fontsize=7)
ax.set_ylabel("Burst AUC, mean across accounts")
ax.set_ylim(0.45, 0.9)
plab(ax, "b", dx=-0.3)
save(fig, "fig6_v42")

# ------------------------------------------------------------------ Figure 7 (annual change)
p3 = pd.read_csv(V7 + "/p3/full/skill_reg_final.csv")
p3 = p3[(p3.metric == "rmse") & (p3.versus == "persistence")].set_index(["target", "model"])
pb = pd.read_csv(V7 + "/p3b/py/skill_reg_final.csv")
pb = pb[(pb.set == "final") & (pb.variant == "ii") & (pb.versus == "persistence") & (pb.n_rows == 498)].set_index(["target", "model"])
spec = [("Profile model", lambda t: p3.loc[(t, "gbt")], GREY),
        ("Power-status model", lambda t: pb.loc[(t, "gbt_core_pw" if t == "d_logvol" else "ridge_core_pw")], BLUE),
        ("Group-year oracle", lambda t: pb.loc[(t, "party_year_ORACLE")], ORANGE)]
tg = [("d_logvol", "Change in log event rate"), ("d_ti7", "Change in mean TI7")]
fig, axs = plt.subplots(1, 2, figsize=(W2, 3.2), gridspec_kw={"width_ratios": [1.3, 1], "wspace": 0.5})
ax = axs[0]
for ti, (t, tl) in enumerate(tg):
    for si, (sl, f, c) in enumerate(spec):
        r = f(t)
        x = ti + (si - 1) * 0.26
        ax.bar(x, 100 * r.skill, width=0.24, color=c, label=sl if ti == 0 else None)
        ax.errorbar(x, 100 * r.skill, yerr=[[100 * (r.skill - r.lo)], [100 * (r.hi - r.skill)]], color="black", lw=1, capsize=2)
        print("fig7", t, sl, round(100 * r.skill, 2))
ax.axhline(0, color="black", lw=0.7)
ax.set_xticks(range(2))
ax.set_xticklabels([t[1] for t in tg])
ax.set_ylabel("RMSE skill over\npredicting no change (%)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=7, columnspacing=0.8, handletextpad=0.3)
plab(ax, "a", dx=-0.15)
dr = pd.read_csv(V7 + "/p3b/py/direction_final.csv")
dr = dr[(dr.variant == "ii") & dr.party.isin(["R", "D"])]
ax = axs[1]
for pi, p in enumerate(["R", "D"]):
    c = R_COL if p == "R" else D_COL
    q = dr[dr.party == p]

    def wm(model, col):
        z = q[q.model == model]
        return np.average(z[col], weights=z.n)
    vals = [("Observed", wm("persistence", "y"), 1.0, None), ("Power-status model", wm("ridge_core_pw", "pred"), 0.45, None),
            ("Training party-mean baseline", wm("party_mean_change", "pred"), 0.0, "//")]
    for bi, (lab, v, al, hatch) in enumerate(vals):
        x = pi + (bi - 1) * 0.27
        ax.bar(x, v, width=0.25, color=c, alpha=max(al, 0.15) if hatch is None else 0.25, hatch=hatch, edgecolor=c, lw=0.8)
ax.axhline(0, color="black", lw=0.7)
ax.set_xticks([0, 1])
ax.set_xticklabels(["Republicans", "Democrats"])
ax.set_ylabel("Change in party-mean TI7,\npolitical year 2021 to 2022")
from matplotlib.patches import Patch  # noqa: E402
ax.legend(handles=[Patch(fc="#666", alpha=1, label="Observed"), Patch(fc="#666", alpha=0.45, label="Power-status model"),
                   Patch(fc="white", ec="#666", hatch="//", label="Training party-mean baseline")], loc="lower center", fontsize=7)
ax.set_ylim(-0.06, 0.035)
plab(ax, "b", dx=-0.2)
save(fig, "fig7_v42")
