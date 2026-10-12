"""Fig 5 (P2 forecasting ladder), Fig 6 (P1 rhythm skill), Fig 7 (P3/P3b trajectory skill)."""
import json
from figs_common import *
style()

# ------------------------------------------------------------------ Fig 5
P2 = json.load(open(V7 + "/p2/numbers_p2.json"))["sample_test"]
hz = [("h1", "Next hour"), ("d24", "Next 24 h"), ("w168", "Next week")]
models = [("nb_nocong", "NB, no Congress", SKY), ("nb", "NB + Congress", BLUE), ("gbm", "GBT", ORANGE), ("hawkes", "Hawkes", GREEN)]
base_names = {"profmean": "profile mean", "pers": "persistence", "snaive": "seasonal naive", "hist": "history", "nb": "NB"}
fig, axs = plt.subplots(1, 2, figsize=(W2, 3.1), gridspec_kw={"width_ratios": [2.3, 1], "wspace": 0.3})
ax = axs[0]
labs = []
for hi, (h, hl) in enumerate(hz):
    bl = P2["ls"][h]["all"]["nb"]["baseline"]
    labs.append(f"{hl}\n(vs {base_names.get(bl, bl)})")
    for mi, (m, ml, c) in enumerate(models):
        r = P2["ls"][h]["all"][m]
        x = hi + (mi - 1.5) * 0.17
        ax.errorbar(x, r["skill"], yerr=[[r["skill"] - r["lo"]], [r["hi"] - r["skill"]]], fmt="o", ms=4.5, color=c, lw=1.2, capsize=0, label=ml if hi == 0 else None)
        print("fig5", h, m, round(r["skill"], 4), round(r["lo"], 4), round(r["hi"], 4), "baseline", bl)
ax.axhline(0, color="black", lw=0.7)
ax.set_xticks(range(3)); ax.set_xticklabels(labs)
ax.set_ylabel("Log-score skill vs best baseline")
ax.set_ylim(-0.005, 0.14)
ax.legend(loc="upper left", ncol=4, fontsize=7.5, columnspacing=0.8, handletextpad=0.2)
plab(ax, "a", dx=-0.13)
ax = axs[1]
bm = [("snaive", "Seasonal\nnaive", GREY), ("pers", "Persistence", GREY), ("logit", "Logit", PINK), ("gbm", "GBT", ORANGE), ("hawkes", "Hawkes", GREEN)]
for i, (m, ml, c) in enumerate(bm):
    r = P2["burst"][m]
    ax.errorbar(i, r["auc_macro"], yerr=[[r["auc_macro"] - r["auc_macro_lo"]], [r["auc_macro_hi"] - r["auc_macro"]]], fmt="o", ms=4.5, color=c, lw=1.2, capsize=0)
    print("fig5 burst", m, round(r["auc_macro"], 4), round(r["auc_macro_lo"], 4), round(r["auc_macro_hi"], 4), "pooled", round(r["auc_pooled"], 4))
ax.axhline(0.5, color="black", lw=0.7, ls=":")
ax.set_xticks(range(len(bm))); ax.set_xticklabels([b[1] for b in bm], rotation=45, ha="right", fontsize=7.5)
ax.set_ylabel("Burst AUC (account-averaged)")
ax.set_ylim(0.45, 0.9)
plab(ax, "b", dx=-0.3)
save(fig, "fig5_forecast_ladder")

# ------------------------------------------------------------------ Fig 6
sk = pd.read_csv(V7 + "/p1/skill_test.csv")
het = pd.read_csv(V7 + "/p1/hetero_test.csv")
abl = pd.read_csv(V7 + "/p1/ablation.csv")
head = sk[(sk.task == "reg") & (sk.model == "gbm_reg_R") & (sk.baseline == "gbm_reg_P")].iloc[0]
print("fig6 headline", head.skill, head.lo, head.hi)
fig = plt.figure(figsize=(W2, 3.4))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.95)
ax = fig.add_subplot(gs[0])
rowsA = [("GBT rhythm", "gbm_reg_R", "gbm_reg_P"), ("GBT + time of day", "gbm_reg_RT", "gbm_reg_PT"),
         ("GBT + post type", "gbm_reg_RTO", "gbm_reg_PTO"), ("Mixed model", "lmm_reg_R", "lmm_reg_P")]
for i, (lab, m, b) in enumerate(rowsA):
    r = sk[(sk.task == "reg") & (sk.model == m) & (sk.baseline == b)].iloc[0]
    ax.errorbar(100 * r.skill, i, xerr=[[100 * (r.skill - r.lo)], [100 * (r.hi - r.skill)]], fmt="o", ms=5 if i == 0 else 4, color=BLACK if i == 0 else GREY, lw=1.3, capsize=0)
ax.axvline(0, color="black", lw=0.7)
ax.set_yticks(range(len(rowsA))); ax.set_yticklabels([r[0] for r in rowsA], fontsize=7); ax.invert_yaxis()
ax.set_xlabel("MSE skill (%)")
plab(ax, "a", dx=-0.9)

ax = fig.add_subplot(gs[1])
h = het[(het.model == "gbm_reg_R")].copy()
order = [("party", "D", "Democrat"), ("party", "R", "Republican"), ("chamber", "house", "House"), ("chamber", "senate", "Senate"),
         ("account_type", "office", "Office"), ("account_type", "campaign", "Campaign"), ("sched", "False", "Not scheduler"), ("sched", "True", "Scheduler")]
y = 0; yt = []
for i, (v, l, lab) in enumerate(order):
    r = h[(h.variable == v) & (h.level.astype(str) == l)].iloc[0]
    c = D_COL if l == "D" else R_COL if l == "R" else GREY
    ax.errorbar(100 * r.skill, y, xerr=[[100 * (r.skill - r.lo)], [100 * (r.hi - r.skill)]], fmt="o", ms=4, color=c, lw=1.2, capsize=0)
    yt.append((y, lab)); y += 1 if i % 2 == 0 else 1.5
ax.axvline(100 * head.skill, color=BLACK, lw=0.8, ls="--")
ax.axvline(0, color="black", lw=0.7)
ax.set_yticks([a for a, b in yt]); ax.set_yticklabels([b for a, b in yt]); ax.invert_yaxis()
ax.set_xlabel("MSE skill (%)")
plab(ax, "b", dx=-0.7)

ax = fig.add_subplot(gs[2])
comp = abl.iloc[6:].copy().sort_values("skill_vs_persistence_model", ascending=False)
rename = {"own timing": "Own timing", "time of day": "Time of day", "scheduler share": "Scheduler share", "lagged Congress volume": "Lagged Congress",
          "regime flags": "Regime flags", "previous post type": "Previous post type", "own post type (first post)": "Own post type", "rhythm minus Congress": "Rhythm, no Congress"}
for i, (_, r) in enumerate(comp.iterrows()):
    ax.errorbar(100 * r.skill_vs_persistence_model, i, xerr=[[100 * (r.skill_vs_persistence_model - r.lo)], [100 * (r.hi - r.skill_vs_persistence_model)]], fmt="o", ms=4, color=BLUE, lw=1.2, capsize=0)
ax.axvline(0, color="black", lw=0.7)
ax.set_yticks(range(len(comp))); ax.set_yticklabels([rename.get(a, a) for a in comp.added_to_persistence], fontsize=7.5); ax.invert_yaxis()
ax.set_xlabel("MSE skill (%)")
plab(ax, "c", dx=-0.9)
save(fig, "fig6_rhythm_skill")

# ------------------------------------------------------------------ Fig 7
p3 = pd.read_csv(V7 + "/p3/full/skill_reg_final.csv")
p3 = p3[(p3.metric == "rmse") & (p3.versus == "persistence")].set_index(["target", "model"])
pb = pd.read_csv(V7 + "/p3b/py/skill_reg_final.csv")
pb = pb[(pb.set == "final") & (pb.variant == "ii") & (pb.versus == "persistence") & (pb.n_rows == 498)].set_index(["target", "model"])
spec = [("Control features, GBT", lambda t: p3.loc[(t, "gbt")], GREY),
        ("Control plus power features", lambda t: pb.loc[(t, "gbt_core_pw" if t == "d_logvol" else "ridge_core_pw")], BLUE),
        ("Party-year oracle", lambda t: pb.loc[(t, "party_year_ORACLE")], ORANGE)]
tg = [("d_logvol", "Volume change (log)"), ("d_ti7", "TI7 change")]
fig, axs = plt.subplots(1, 2, figsize=(W2, 3.2), gridspec_kw={"width_ratios": [1.3, 1], "wspace": 0.3})
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
ax.set_ylabel("Next-year RMSE skill vs persistence")
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
    vals = [("Observed", wm("persistence", "y"), 1.0, None), ("Power-feature model", wm("ridge_core_pw", "pred"), 0.45, None), ("Party-mean baseline", wm("party_mean_change", "pred"), 0.0, "//")]
    for bi, (lab, v, al, hatch) in enumerate(vals):
        x = pi + (bi - 1) * 0.27
        ax.bar(x, v, width=0.25, color=c, alpha=max(al, 0.15) if hatch is None else 0.25, hatch=hatch, edgecolor=c, lw=0.8)
    print("fig7b", p, [round(v[1], 4) for v in vals], "check mean y from preds", round(chk[p], 4))
ax.axhline(0, color="black", lw=0.7)
ax.set_xticks([0, 1]); ax.set_xticklabels(["Republicans", "Democrats"])
ax.set_ylabel("Change in party-mean TI7, 2021 to 2022")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(fc="#666", alpha=1, label="Observed"), Patch(fc="#666", alpha=0.45, label="Power-feature model"), Patch(fc="white", ec="#666", hatch="//", label="Party-mean baseline")],
          loc="lower center", fontsize=7)
ax.set_ylim(-0.06, 0.035)
plab(ax, "b", dx=-0.2)
save(fig, "fig7_trajectory_skill")
