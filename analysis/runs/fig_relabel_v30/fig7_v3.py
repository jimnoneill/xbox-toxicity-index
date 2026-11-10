import sys, json
sys.path.insert(0, '/home/joneill/x_box/pipeline/scripts/v7')
import figs_common
figs_common.OUT = '/home/joneill/x_box/pipeline/manuscript/epjds_submission_work/fig_relabel_v30'
from figs_common import *
style()
# ------------------------------------------------------------------ Fig 7
p3 = pd.read_csv(V7 + "/p3/full/skill_reg_final.csv")
p3 = p3[(p3.metric == "rmse") & (p3.versus == "persistence")].set_index(["target", "model"])
pb = pd.read_csv(V7 + "/p3b/py/skill_reg_final.csv")
pb = pb[(pb.set == "final") & (pb.variant == "ii") & (pb.versus == "persistence") & (pb.n_rows == 498)].set_index(["target", "model"])
spec = [("Profile model", lambda t: p3.loc[(t, "gbt")], GREY),
        ("Power-status model", lambda t: pb.loc[(t, "gbt_core_pw" if t == "d_logvol" else "ridge_core_pw")], BLUE),
        ("Group-year oracle", lambda t: pb.loc[(t, "party_year_ORACLE")], ORANGE)]
tg = [("d_logvol", "Volume change (log)"), ("d_ti7", "TI7 change")]
fig, axs = plt.subplots(1, 2, figsize=(W2, 3.2), gridspec_kw={"width_ratios": [1.3, 1], "wspace": 0.5})
ax = axs[0]
for ti, (t, tl) in enumerate(tg):
    for si, (sl, f, c) in enumerate(spec):
        r = f(t)
        x = ti + (si - 1) * 0.26
        ax.bar(x, r.skill, width=0.24, color=c, label=sl if ti == 0 else None)
        ax.errorbar(x, r.skill, yerr=[[r.skill - r.lo], [r.hi - r.skill]], color="black", lw=1, capsize=2)
        print("fig7", t, sl, round(r.skill, 4), round(r.lo, 4), round(r.hi, 4))
ax.axhline(0, color="black", lw=0.7)
ax.set_xticks(range(2)); ax.set_xticklabels([t[1] for t in tg])
ax.set_ylabel("RMSE skill over\npredicting no change")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=7, columnspacing=0.8, handletextpad=0.3)
plab(ax, "a", dx=-0.15)

dr = pd.read_csv(V7 + "/p3b/py/direction_final.csv")
dr = dr[(dr.variant == "ii") & dr.party.isin(["R", "D"])]
pr = pd.read_csv(V7 + "/p3b/py/preds_final_ii.csv")
pr = pr[(pr.target == "d_ti7") & (pr.model == "persistence") & pr.party.isin(["R", "D"])]
chk = pr.groupby("party").y.mean()
ax = axs[1]
for pi, p in enumerate(["R", "D"]):
    c = R_COL if p == "R" else D_COL
    q = dr[dr.party == p]
    def wm(model, col):
        z = q[q.model == model]
        return np.average(z[col], weights=z.n)
    vals = [("Observed", wm("persistence", "y"), 1.0, None), ("Power-status model", wm("ridge_core_pw", "pred"), 0.45, None), ("Training party-mean baseline", wm("party_mean_change", "pred"), 0.0, "//")]
    for bi, (lab, v, al, hatch) in enumerate(vals):
        x = pi + (bi - 1) * 0.27
        ax.bar(x, v, width=0.25, color=c, alpha=max(al, 0.15) if hatch is None else 0.25, hatch=hatch, edgecolor=c, lw=0.8)
    print("fig7b", p, [round(v[1], 4) for v in vals], "check mean y from preds", round(chk[p], 4))
ax.axhline(0, color="black", lw=0.7)
ax.set_xticks([0, 1]); ax.set_xticklabels(["Republicans", "Democrats"])
ax.set_ylabel("Change in party-mean TI7,\npolitical year 2021 to 2022")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(fc="#666", alpha=1, label="Observed"), Patch(fc="#666", alpha=0.45, label="Power-status model"), Patch(fc="white", ec="#666", hatch="//", label="Training party-mean baseline")],
          loc="lower center", fontsize=7)
ax.set_ylim(-0.06, 0.035)
plab(ax, "b", dx=-0.2)
save(fig, "fig7_v3")
