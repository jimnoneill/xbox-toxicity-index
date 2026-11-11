"""Figure S1 (v40): timing models for member event streams.

a  share of members for whom each of the 19 variants has the lowest BIC
b  KS distance between the rescaled intervals and the fitted model, by variant
c  strength of self-excitation with and without the shared-activity term in the background rate
d  share of members passing the parametric-bootstrap time-rescaling test, by model
Usage: python figS1_v40.py   (reads output/v7/temporal_events/model_compare and .../shared)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = Path.home() / "x_box/pipeline"
sys.path.insert(0, str(ROOT / "scripts/v7"))
from figs_common import style, plab, BLUE, ORANGE, GREY, W2  # noqa: E402

style()
MC, SH = ROOT / "output/v7/temporal_events/model_compare", ROOT / "output/v7/temporal_events/shared"
mm = pd.read_csv(MC / "members_models.csv")
mm = mm[(mm.bg != "LEE") & mm.ll.notna()]
nmem = mm.bg.nunique()
lab = {"casc3": "Cascade (3 levels)", "hawkes_soe": "Hawkes, several decay rates", "hawkes_exp": "Hawkes, one decay rate",
       "mmpp3c": "State switching (3), clock", "mmpp3": "State switching (3)", "renc_he3": "Renewal, 3 rates, clock",
       "casc2": "Cascade (2 levels)", "renc_he2": "Renewal, 2 rates, clock", "ren_he2": "Renewal, 2 rates",
       "ren_gm": "Renewal, gamma", "ren_wb": "Renewal, Weibull", "renc_wb": "Renewal, Weibull, clock",
       "ren_ln": "Renewal, lognormal", "renc_ln": "Renewal, lognormal, clock", "renc_gm": "Renewal, gamma, clock",
       "mmpp2": "State switching (2)", "mmpp2c": "State switching (2), clock", "ren_he3": "Renewal, 3 rates",
       "nhpp_hw": "Poisson, clock", "hawkes_shared": "Hawkes, one decay rate,\nshared-activity term"}
piv = mm.pivot_table(index="bg", columns="model", values="bic")
best = piv.idxmin(axis=1).value_counts() / len(piv)
s2 = best.reindex(sorted(mm.model.unique())).fillna(0).sort_values()
fig = plt.figure(figsize=(W2, 6.3))
outer = fig.add_gridspec(2, 1, height_ratios=[1.45, 1], hspace=0.42)
gs = outer[0].subgridspec(1, 2, width_ratios=[1.1, 1], wspace=0.08)
gs2 = outer[1].subgridspec(1, 2, width_ratios=[1, 1], wspace=1.0)
ax = fig.add_subplot(gs[0])
cols = [BLUE if m == "casc3" else ORANGE if m.startswith("hawkes") else GREY for m in s2.index]
ax.barh(range(len(s2)), 100 * s2.values, color=cols)
ax.set_yticks(range(len(s2)))
ax.set_yticklabels([lab.get(m, m) for m in s2.index], fontsize=7)
ax.set_xlabel(f"Members best fit by BIC (%), n = {nmem}")
plab(ax, "a", dx=-0.8)
axb = fig.add_subplot(gs[1])
order = list(s2.index)
bp = axb.boxplot([mm[mm.model == m].D.dropna().values for m in order], vert=False, positions=range(len(order)), widths=0.6,
                 whis=(5, 95), showfliers=False, patch_artist=True, medianprops=dict(color="black", lw=1.0),
                 whiskerprops=dict(lw=0.7), capprops=dict(lw=0.7), boxprops=dict(lw=0.7))
for patch, c in zip(bp["boxes"], cols):
    patch.set_facecolor(c)
    patch.set_alpha(0.75)
axb.set_xscale("log")
axb.set_yticks(range(len(order)))
axb.set_yticklabels([])
axb.set_ylim(ax.get_ylim())
crit = 1.358 / np.sqrt(mm.drop_duplicates("bg").n.median())
axb.axvline(crit, color="black", lw=0.7, ls="--")
axb.annotate("5% critical value at the median stream length", xy=(crit, 1.0), xycoords=("data", "axes fraction"),
             xytext=(0, 3), textcoords="offset points", fontsize=6.5, ha="left", va="bottom")
axb.set_xticks([0.02, 0.05, 0.1, 0.2])
axb.set_xticklabels(["0.02", "0.05", "0.1", "0.2"])
axb.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
axb.set_xlabel("Distance between rescaled intervals\nand the fitted model (KS D)")
plab(axb, "b", dx=-0.12)
# c: self-excitation with and without the shared-activity term
sh = pd.read_csv(SH / "members_shared.csv")
sh = sh[sh.ll.notna()]
e = sh.pivot_table(index="bg", columns="model", values="eta")
axc = fig.add_subplot(gs2[0])
axc.scatter(e["hawkes_exp"], e["hawkes_shared"], s=4, color=ORANGE, alpha=0.5, lw=0)
lim = float(np.nanquantile(e[["hawkes_exp", "hawkes_shared"]].values, 0.995)) * 1.05
axc.plot([0, lim], [0, lim], color="black", lw=0.7, ls="--")
axc.set_xlim(0, lim)
axc.set_ylim(0, lim)
axc.set_xlabel("Self-excitation without\nthe shared-activity term")
axc.set_ylabel("Self-excitation with\nthe shared-activity term")
axc.text(0.05, 0.93, f"lower for {100 * (e.hawkes_shared < e.hawkes_exp).mean():.0f}% of members", transform=axc.transAxes, fontsize=7, va="top")
plab(axc, "c", dx=-0.42)
# d: parametric-bootstrap pass rates for every member
bt = pd.read_csv(SH / "bootstrap_all.csv")
bt = bt[bt.B > 0]
pr = (100 * bt.assign(ok=bt.boot_p > 0.05).groupby("model").ok.mean()).sort_values()
axd = fig.add_subplot(gs2[1])
dcol = [BLUE if m == "casc3" else ORANGE if m.startswith("hawkes") else GREY for m in pr.index]
axd.barh(range(len(pr)), pr.values, color=dcol)
axd.set_yticks(range(len(pr)))
axd.set_yticklabels([lab.get(m, m) for m in pr.index], fontsize=7)
axd.set_xlim(0, 100)
axd.set_xlabel(f"Members passing the\nbootstrap test (%), n = {bt.bg.nunique()}")
plab(axd, "d", dx=-0.95)
fig.savefig(HERE / "figS1_v40.png", dpi=600, bbox_inches="tight")
fig.savefig(HERE / "figS1_v40.pdf", bbox_inches="tight")
json.dump({"bic_best": (100 * best).round(1).to_dict(), "boot_pass": pr.round(1).to_dict(), "boot_members": int(bt.bg.nunique())},
          open(HERE / "figS1_v40.json", "w"), indent=1)
print("figure saved", dict(pr.round(1)))
