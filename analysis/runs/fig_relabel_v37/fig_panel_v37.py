"""Yearly negative-partisan share by party in the 2009 to 2022 panel, with party control (new main-text figure).

Member-year share of tweets labeled negative-partisan (Kistner et al. replication data, Twitter only), members with at
least 50 tweets in the year, 2010 to 2022: the association panel of scripts/v7/kistner_02_assoc.py. Party means over
members with 95 percent intervals (mean +/- 1.96 SE), as in Figure 3."""
import sys
sys.path.insert(0, '/home/joneill/x_box/pipeline/scripts/v7'); sys.path.insert(0, '/home/joneill/x_box/pipeline'); sys.path.insert(0, '/home/joneill/x_box/pipeline/xbox')
from figs_common import *
from pathlib import Path
import kistner_ext as K
HERE = Path(__file__).resolve().parent
style()
p = K.build_panel(("Twitter",)); p = p[(p.n >= K.MIN_N) & (p.Year >= 2010) & p.party.isin(["R", "D"])].copy()
print("member-years", len(p), "members", p.MemberICPSR.nunique(), "years", p.Year.min(), p.Year.max(), "MIN_N", K.MIN_N)
rows = []
for (y, pt), g in p.groupby(["Year", "party"]):
    mu, lo, hi = mean_ci(g.share.values)
    rows.append(dict(year=int(y), party=pt, mean=mu, lo=lo, hi=hi, n_members=len(g)))
t = pd.DataFrame(rows); t.to_csv(HERE / "fig_panel_v37_plotted.csv", index=False); print(t.round(4).to_string())
print("PRES", K.PRES); print("HOUSE", K.HOUSE); print("SENATE", K.SENATE)
x0, x1 = 2009.5, 2022.5
fig = plt.figure(figsize=(W2, 3.9))
gsp = fig.add_gridspec(4, 1, height_ratios=[0.16, 0.16, 0.16, 2.6], hspace=0.12)
strips = [fig.add_subplot(gsp[i]) for i in range(3)]
ax = fig.add_subplot(gsp[3])
for sa, (lab, tab) in zip(strips, [("President", K.PRES), ("House", K.HOUSE), ("Senate", K.SENATE)]):
    for y in range(2010, 2023):
        v = tab[y]
        sa.axvspan(y - 0.5, y + 0.5, color=R_COL if v == "R" else D_COL, alpha=0.75, lw=0)
        if lab == "President":
            ax.axvspan(y - 0.5, y + 0.5, color=R_COL if v == "R" else D_COL, alpha=0.07, lw=0)
    sa.set_xlim(x0, x1); sa.set_yticks([]); sa.set_xticks([])
    sa.set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=8)
    for sp in sa.spines.values():
        sp.set_visible(False)
for pt, c, nm in [("D", D_COL, "Democrats"), ("R", R_COL, "Republicans")]:
    q = t[t.party == pt].sort_values("year")
    ax.fill_between(q.year, q.lo, q.hi, color=c, alpha=0.2, lw=0)
    ax.plot(q.year, q["mean"], "-o", color=c, lw=1.5, ms=4, label=nm)
ax.axvline(2018.5, color="black", lw=0.8, ls="--")
ax.text(2018.42, 0.335, "development years", ha="right", va="top", fontsize=7.5, color=GREY)
ax.text(2018.58, 0.335, "test years", ha="left", va="top", fontsize=7.5, color=GREY)
ax.set_xlim(x0, x1); ax.set_xticks(range(2010, 2023)); ax.set_ylim(0, 0.35)
ax.set_ylabel("Member-year mean negative-partisan\nshare (95% CI)")
ax.set_xlabel("Calendar year")
ax.legend(loc="upper left", ncol=1)
fig.savefig(HERE / "fig_panel_v37.png", dpi=600, bbox_inches="tight"); fig.savefig(HERE / "fig_panel_v37.pdf", bbox_inches="tight")
print("saved")
