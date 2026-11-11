"""Figure 2 (specification curve) redrawn with the manuscript's vocabulary.

Same data and layout as scripts/v7/figs_01_party.py (Fig 2 block); only the labels change.
Input: spec_curve_full.csv (copy of output/v7/stats_full/spec_curve_full.csv). Output: fig2_v36.png (600 dpi), fig2_v36.pdf.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
BLUE, ORANGE, BLACK, GREY = "#0072B2", "#D55E00", "#222222", "#8a8a8a"
W2 = 7.2
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7, "xtick.major.size": 3, "ytick.major.size": 3,
    "lines.linewidth": 1.2, "pdf.fonttype": 42, "ps.fonttype": 42, "figure.dpi": 100,
})

s = pd.read_csv(HERE / "spec_curve_full.csv")
s = s[s.measure == "ti7"].copy()
meta = {  # step -> (unit, posts, minimum posts, accounts)
    "A-all1": ("account", "all", 50, "roster"), "A-all3": ("account", "all", 50, "attributed"),
    "A-all4": ("account", "all", 50, "office"), "A-all5": ("account", "all", 50, "campaign"),
    "A-nonrt1": ("account", "nonrt", 50, "roster"), "A-nonrt3": ("account", "nonrt", 50, "attributed"),
    "A-nonrt4": ("account", "nonrt", 50, "office"), "A-nonrt5": ("account", "nonrt", 50, "campaign"),
    "A-orig1": ("account", "orig", 50, "roster"), "A-orig3": ("account", "orig", 50, "attributed"),
    "M1": ("member", "all", 50, "attributed"), "M2": ("member", "nonrt", 50, "attributed"),
    "M3": ("member", "nonrt", 200, "attributed"), "M3a": ("member", "all", 200, "attributed"),
    "M4b": ("member", "nonrt", 50, "switchers"), "M5": ("member", "nonrt", 50, "office"),
    "M6": ("member", "nonrt", 50, "campaign"), "M7": ("member", "orig", 50, "attributed"),
}
assert set(s.step) == set(meta), set(s.step) ^ set(meta)
s = s.sort_values("g").reset_index(drop=True)
assert len(s) == 18 and s.is_primary.sum() == 1 and s[s.is_primary].step.iloc[0] == "M2"
print("n specs", len(s), "median g", round(s.g.median(), 4), "range", round(s.g.min(), 4), round(s.g.max(), 4))
rows = [("Unit", [("member", "member"), ("account", "account")]),
        ("Posts", [("all", "all posts"), ("nonrt", "non-repost posts"), ("orig", "original posts")]),
        ("Minimum posts", [(50, "50"), (200, "200")]),
        ("Accounts", [("attributed", "all accounts attributed to members"), ("roster", "119th Congress roster accounts"),
                      ("office", "office accounts"), ("campaign", "campaign accounts"),
                      ("switchers", "party switchers under original party")])]
names = [r[0] for r in rows]
fig = plt.figure(figsize=(W2, 5.4))
gsp = fig.add_gridspec(2, 1, height_ratios=[1.1, 1.25], hspace=0.06)
ax = fig.add_subplot(gsp[0]); ax2 = fig.add_subplot(gsp[1], sharex=ax)
x = np.arange(len(s))
prim = s.is_primary.values.astype(bool)
ax.vlines(x, s.ci_lo, s.ci_hi, color=GREY, lw=1.2)
ax.scatter(x[~prim], s.g[~prim], s=22, color=BLACK, zorder=3)
ax.scatter(x[prim], s.g[prim], s=50, color=ORANGE, zorder=4, edgecolor="black", lw=0.6)
ax.axhline(0, color="black", lw=0.7)
med = s.g.median()
ax.axhline(med, color=BLUE, lw=0.9, ls="--")
ax.text(len(s) - 1.2, med - 0.012, f"median {med:.2f}", color=BLUE, ha="right", va="top", fontsize=8)
ax.set_ylabel("Party gap in TI7, Hedges g\n(Republican minus Democratic)")
plt.setp(ax.get_xticklabels(), visible=False)
ax.set_xlim(-0.8, len(s) - 0.2)
ax.tick_params(axis="x", length=0)
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=ORANGE, mec="black", label="Primary specification"),
                   Line2D([], [], marker="o", ls="", color=BLACK, label="Other specifications (95% confidence interval)")],
          loc="upper left")
ypos, sepl, y = [], [], 0
for rname, opts in rows:
    for key, lab in opts:
        for xi, st in enumerate(s.step):
            if meta[st][names.index(rname)] == key:
                ax2.scatter(xi, y, s=16, color=ORANGE if prim[xi] else BLACK, zorder=3)
        ypos.append((y, lab)); y -= 1
    sepl.append(y + 0.5); y -= 0.6
ax2.set_yticks([p[0] for p in ypos]); ax2.set_yticklabels([p[1] for p in ypos])
for yy in sepl[:-1]:
    ax2.axhline(yy - 0.3, color="#cccccc", lw=0.6)
ax2.set_ylim(min(p[0] for p in ypos) - 0.8, 0.8)
ax2.set_xticks([])
for sp in ["left", "bottom"]:
    ax2.spines[sp].set_visible(False)
ax2.tick_params(axis="y", length=0)
k = 0
for rname, opts in rows:
    mid = [p[0] for p in ypos[k:k + len(opts)]]
    ax2.text(len(s) - 0.1, np.mean(mid), rname, ha="left", va="center", fontsize=8, color=GREY, style="italic")
    k += len(opts)
ax2.set_xlabel("Specifications ordered by party gap")
fig.savefig(HERE / "fig2_v36.pdf", bbox_inches="tight")
fig.savefig(HERE / "fig2_v36.png", dpi=600, bbox_inches="tight")
s[["step", "spec", "g", "ci_lo", "ci_hi", "is_primary"]].to_csv(HERE / "fig2_v36_plotted.csv", index=False)
print("saved")
