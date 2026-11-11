"""Figure 4 redrawn with the manuscript's vocabulary. Same data and layout as scripts/v7/figs_02_effects.py; labels only."""
import sys
sys.path.insert(0, '/home/joneill/x_box/pipeline/scripts/v7')
from figs_common import *
from pathlib import Path
HERE = Path(__file__).resolve().parent
style()
ta = [("primary_window", "Own primary window"), ("general_window", "General-election window"), ("post_election", "Post-election window"),
      ("chamber_in_session", "Session day"), ("aligned_with_president", "Same-party president"),
      ("in_chamber_majority", "Own-party chamber majority"), ("event_day", "Political-event day")]
tb = [("primary_window", "Own primary window"), ("general_window", "General-election window"), ("post_election", "Post-election window"),
      ("in_session", "Session day"), ("aligned_president", "Same-party president"), ("party_majority", "Own-party chamber majority"),
      ("event_curated", "Political-event day"), ("event_detected", "Shared-spike day")]
adj = pd.read_csv(V7 + "/t5/regime_coefficients_adjusted.csv")
pg = pd.read_csv(V7 + "/t4b/pooled_groups.csv")
pg = pg[(pg.group_type == "party") & pg.group.isin(["R", "D"])]
po = pd.read_csv(V7 + "/t4b/pooled_overall.csv")
po = po[po.variant == "full"]
col = {"R": R_COL, "D": D_COL}
off = {"R": 0.24, "D": -0.24}
fig, axs = plt.subplots(1, 2, figsize=(W2, 3.8), gridspec_kw={"wspace": 0.95, "width_ratios": [1, 1]})
ax = axs[0]
rows = []
for i, (t, lab) in enumerate(ta):
    for spec, dy, c, ms, lw in [("common_trend", 0.17, "#9a9a9a", 3.5, 1.0), ("party_trend", -0.1, BLACK, 4.5, 1.5)]:
        r = adj[(adj.spec == spec) & (adj.term == t)].iloc[0]
        ax.errorbar(r.coef, i + dy, xerr=[[r.coef - r.ci_lo], [r.ci_hi - r.coef]], fmt="o", ms=ms, color=c, lw=lw, capsize=0)
        rows.append(("a", lab, spec, r.coef, r.ci_lo, r.ci_hi))
ax.axvline(0, color="black", lw=0.7)
ax.set_yticks(range(len(ta))); ax.set_yticklabels([t[1] for t in ta]); ax.invert_yaxis()
ax.set_xlabel("Difference in daily TI7\n(95% CI)"); ax.set_xticks([-0.06, -0.04, -0.02, 0, 0.02])
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=BLACK, label="Party-specific trends"),
                   Line2D([], [], marker="o", ls="", color="#9a9a9a", ms=3.5, label="Common trend")], loc="center left", bbox_to_anchor=(0.0, 0.47), fontsize=7.5)
plab(ax, "a", dx=-0.78)
ax = axs[1]
for i, (t, lab) in enumerate(tb):
    r = po[po.term == t].iloc[0]
    ax.hlines(i, r.RR_pi_lo, r.RR_pi_hi, color=GREY, lw=0.8, alpha=0.8)
    ax.errorbar(r.RR_mu, i, xerr=[[r.RR_mu - r.RR_lo], [r.RR_hi - r.RR_mu]], fmt="o", ms=4.5, color=BLACK, lw=1.6, capsize=0)
    rows.append(("b", lab, "all", r.RR_mu, r.RR_lo, r.RR_hi))
    for g in ["R", "D"]:
        q = pg[(pg.term == t) & (pg.group == g)]
        if len(q):
            q = q.iloc[0]
            ax.errorbar(q.RR_mu, i + off[g], xerr=[[q.RR_mu - q.RR_lo], [q.RR_hi - q.RR_mu]], fmt="o", ms=3.5, color=col[g], lw=1.1, capsize=0)
ax.axvline(1, color="black", lw=0.7)
ax.set_xscale("log"); ax.set_xlim(0.25, 3.2)
ax.set_xticks([0.25, 0.5, 1, 2, 3]); ax.set_xticklabels(["0.25", "0.5", "1", "2", "3"])
ax.set_yticks(range(len(tb))); ax.set_yticklabels([t[1] for t in tb]); ax.invert_yaxis()
ax.set_xlabel("Posting-rate ratio\n(95% CI; gray: 95% PI)")
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=c, label=l) for c, l in [(BLACK, "All accounts"), (R_COL, "Republicans"), (D_COL, "Democrats")]], loc="lower left", fontsize=7)
plab(ax, "b", dx=-0.78)
fig.savefig(HERE / "fig4_v37.png", dpi=600, bbox_inches="tight"); fig.savefig(HERE / "fig4_v37.pdf", bbox_inches="tight")
pd.DataFrame(rows, columns=["panel", "label", "spec", "est", "lo", "hi"]).to_csv(HERE / "fig4_v37_plotted.csv", index=False)
print(pd.DataFrame(rows).round(4).to_string())
print(po.columns.tolist()); print(po[["term", "RR_mu", "RR_lo", "RR_hi", "RR_pi_lo", "RR_pi_hi"]].round(3).to_string())
