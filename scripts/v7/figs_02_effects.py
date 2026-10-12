"""Fig 4: regime effects. (a) T5 member-FE daily TI7 coefficients (party-trend-adjusted primary, common-trend secondary);
(b) T4b pooled volume rate ratios with prediction intervals."""
from figs_common import *
style()

ta = [("primary_window", "Own primary window"), ("general_window", "General-election window"), ("post_election", "Post-election"),
      ("chamber_in_session", "In session"), ("aligned_with_president", "Aligned president"), ("in_chamber_majority", "Party in majority"),
      ("event_day", "Event day")]
tb = [("primary_window", "Own primary window"), ("general_window", "General-election window"), ("post_election", "Post-election"),
      ("in_session", "In session"), ("aligned_president", "Aligned president"), ("party_majority", "Party in majority"),
      ("event_curated", "Curated event day"), ("event_detected", "Detected event day")]
adj = pd.read_csv(V7 + "/t5/regime_coefficients_adjusted.csv")
pg = pd.read_csv(V7 + "/t4b/pooled_groups.csv")
pg = pg[(pg.group_type == "party") & pg.group.isin(["R", "D"])]
po = pd.read_csv(V7 + "/t4b/pooled_overall.csv")
po = po[po.variant == "full"]

col = {"R": R_COL, "D": D_COL}
off = {"R": 0.24, "D": -0.24}
fig, axs = plt.subplots(1, 2, figsize=(W2, 3.8), gridspec_kw={"wspace": 0.75, "width_ratios": [1, 1]})
ax = axs[0]
for i, (t, lab) in enumerate(ta):
    for spec, dy, c, ms, lw in [("common_trend", 0.17, "#9a9a9a", 3.5, 1.0), ("party_trend", -0.1, BLACK, 4.5, 1.5)]:
        r = adj[(adj.spec == spec) & (adj.term == t)].iloc[0]
        ax.errorbar(r.coef, i + dy, xerr=[[r.coef - r.ci_lo], [r.ci_hi - r.coef]], fmt="o", ms=ms, color=c, lw=lw, capsize=0)
ax.axvline(0, color="black", lw=0.7)
ax.set_yticks(range(len(ta))); ax.set_yticklabels([t[1] for t in ta]); ax.invert_yaxis()
ax.set_xlabel("Change in daily TI7 (95% CI)")
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=BLACK, label="Party-specific trends"),
                   Line2D([], [], marker="o", ls="", color="#9a9a9a", ms=3.5, label="Common trend")], loc="center left", bbox_to_anchor=(0.0, 0.47), fontsize=7.5)
plab(ax, "a", dx=-0.62)

ax = axs[1]
for i, (t, lab) in enumerate(tb):
    r = po[po.term == t].iloc[0]
    ax.hlines(i, r.RR_pi_lo, r.RR_pi_hi, color=GREY, lw=0.8, alpha=0.8)
    ax.errorbar(r.RR_mu, i, xerr=[[r.RR_mu - r.RR_lo], [r.RR_hi - r.RR_mu]], fmt="o", ms=4.5, color=BLACK, lw=1.6, capsize=0)
    for g in ["R", "D"]:
        q = pg[(pg.term == t) & (pg.group == g)]
        if len(q):
            q = q.iloc[0]
            ax.errorbar(q.RR_mu, i + off[g], xerr=[[q.RR_mu - q.RR_lo], [q.RR_hi - q.RR_mu]], fmt="o", ms=3.5, color=col[g], lw=1.1, capsize=0)
ax.axvline(1, color="black", lw=0.7)
ax.set_xscale("log"); ax.set_xlim(0.25, 3.2)
ax.set_xticks([0.25, 0.5, 1, 2, 3]); ax.set_xticklabels(["0.25", "0.5", "1", "2", "3"])
ax.set_yticks(range(len(tb))); ax.set_yticklabels([t[1] for t in tb]); ax.invert_yaxis()
ax.set_xlabel("Posting-rate ratio (CI; grey: PI)")
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=c, label=l) for c, l in [(BLACK, "All members"), (R_COL, "Republicans"), (D_COL, "Democrats")]], loc="lower left", fontsize=7)
plab(ax, "b", dx=-0.62)
save(fig, "fig4_regime_effects")
