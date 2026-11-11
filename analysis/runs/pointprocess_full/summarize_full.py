"""Summary numbers and Figure S1 for the point-process model comparison.

Usage: python summarize_full.py DIR [--fig]
DIR is a model_compare output directory (members_models.csv, bootstrap.csv, cache.pkl). Run on
output/v7/temporal/model_compare it reproduces the every-third-day numbers; run on
output/v7/temporal_full/model_compare it gives the numbers for the whole corpus. Writes summary_full.json (and
figS1_v37.png/.pdf with --fig) next to this script, suffixed by the directory's parent name.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
D = Path(sys.argv[1])
tag = D.parent.name
mm = pd.read_csv(D / "members_models.csv")
if "err" in mm:
    print("fit errors:", int(mm.err.notna().sum()))
    mm = mm[mm.err.isna()]
mm = mm[mm.bg != "LEE"].copy()
nmem = mm.bg.nunique()
out = {"dir": str(D), "members": int(nmem), "models": int(mm.model.nunique())}
mm["pass05"] = mm.p > 0.05
ks = mm.groupby("model").pass05.mean().sort_values(ascending=False)
out["ks_pass05_by_model_pct"] = (100 * ks).round(1).to_dict()
out["ks_fail_hawkes_exp_pct"] = round(100 * (1 - ks["hawkes_exp"]), 1)
piv_b = mm.pivot_table(index="bg", columns="model", values="bic")
piv_a = mm.pivot_table(index="bg", columns="model", values="aic")
d = (piv_b["casc3"] - piv_b["hawkes_exp"]).dropna()
out["bic_casc3_over_hawkes_exp_pct"] = round(100 * (d < 0).mean(), 1)
d = (piv_a["casc3"] - piv_a["hawkes_exp"]).dropna()
out["median_dAIC_casc3_minus_hawkes_exp"] = round(float(d.median()), 1)
d = (piv_a["casc3"] - piv_a["hawkes_soe"]).dropna()
out["aic_casc3_lower_than_hawkes_soe_pct"] = round(100 * (d < 0).mean(), 1)
best = piv_b.idxmin(axis=1).value_counts() / len(piv_b)
out["bic_best_share_pct"] = (100 * best).round(1).to_dict()
bf = D / "bootstrap.csv"
if bf.exists():
    bt = pd.read_csv(bf)
    if "err" in bt:
        print("boot errors:", int(bt.err.notna().sum()))
        bt = bt[bt.err.isna()]
    bp = bt.groupby("model").boot_p.apply(lambda x: (x > 0.05).mean()).sort_values(ascending=False)
    out["boot_members"] = int(bt.bg.nunique())
    out["boot_B_median"] = float(bt.B.median())
    out["boot_pass05_by_model_pct"] = (100 * bp).round(1).to_dict()
    out["boot_pass05_max_pct"] = round(100 * float(bp.max()), 1)


# size of misfit: KS distance D between the rescaled intervals and the uniform distribution (0 = perfect), by model
Dq = mm.groupby("model").D.quantile([.25, .5, .75]).unstack().sort_values(0.5)
out["ks_D_quartiles_by_model"] = {m: [round(float(v), 4) for v in r] for m, r in Dq.iterrows()}
mm["K"] = mm.D * np.sqrt(mm.n)   # scaled statistic; 5 percent critical value 1.358
out["ks_scaled_median_by_model"] = mm.groupby("model").K.median().round(2).sort_values().to_dict()
out["ks_D_below_005_pct_by_model"] = (100 * mm.assign(c=mm.D < 0.05).groupby("model").c.mean()).round(1).sort_values(ascending=False).to_dict()
out["ks_D_below_010_pct_by_model"] = (100 * mm.assign(c=mm.D < 0.10).groupby("model").c.mean()).round(1).sort_values(ascending=False).to_dict()
out["bic_soe_over_exp_pct"] = round(100 * float(((piv_b["hawkes_soe"] - piv_b["hawkes_exp"]).dropna() < 0).mean()), 1)
three = ["mmpp3c", "mmpp3", "hawkes_soe", "renc_he3", "ren_he3", "casc3"]
out["bic_best_three_scale_pct"] = round(100 * float(piv_b.idxmin(axis=1).isin(three).mean()), 1)
out["bic_best_clock_pct"] = round(100 * float(piv_b.idxmin(axis=1).isin(
    [m for m in piv_b.columns if m.startswith(("renc", "hawkes", "casc")) or m.endswith("c") or m == "nhpp_hw"]).mean()), 1)
out["aic_soe_over_exp_pct"] = round(100 * float(((piv_a["hawkes_soe"] - piv_a["hawkes_exp"]).dropna() < 0).mean()), 1)
out["ks_crit_D_at_median_n"] = round(1.358 / float(np.sqrt(mm.drop_duplicates("bg").n.median())), 4)
vf = D / "validation_sims.csv"
if vf.exists():
    v = pd.read_csv(vf)
    rj = 100 * v.assign(r=v.p < 0.05).groupby("model").r.mean()
    out["validation_templates"] = int(v.bg.nunique())
    out["validation_sims"] = int(len(v))
    out["validation_reject05_by_model_pct"] = rj.round(1).sort_values().to_dict()
    out["validation_reject05_max_pct"] = round(float(rj.max()), 1)
    out["validation_models_at_or_below_5pct"] = int((rj <= 5).sum())
out["n_median"] = float(mm.drop_duplicates("bg").n.median())
out["n_total"] = int(mm.drop_duplicates("bg").n.sum())


def params(model):
    s = mm[mm.model == model].set_index("bg")
    return pd.DataFrame([json.loads(x) for x in s.params], index=s.index), s


P, s = params("casc3")
out["casc3_param_names"] = list(P.columns)
out["casc3_medians"] = {c: round(float(P[c].median()), 4) for c in P.columns}
for a, b in [("rate1", "scale1_min"), ("rate2", "scale2_min")]:
    if a in P:
        out["casc3_medians"][b] = round(float((1 / P[a]).median()), 3)
H, hs = params("hawkes_exp")
out["hawkes_exp_param_names"] = list(H.columns)
if "inv_beta_min" in H:
    lo = H.inv_beta_min.min()
    out["hawkes_exp_inv_beta_min_floor"] = float(lo)
    out["hawkes_exp_at_floor_pct"] = round(100 * float((H.inv_beta_min <= lo * 1.01).mean()), 1)
    out["hawkes_exp_inv_beta_quantiles_min"] = H.inv_beta_min.quantile([.1, .25, .5, .75, .9]).round(3).to_dict()
out["hawkes_exp_medians"] = {c: round(float(H[c].median()), 4) for c in H.columns}
S, ss = params("hawkes_soe")
out["hawkes_soe_param_names"] = list(S.columns)
out["hawkes_soe_medians"] = {c: round(float(S[c].median()), 4) for c in S.columns}
# burstiness and volume from the cached streams
cache = pickle.load(open(D / "cache.pkl", "rb"))
rows = []
for bg in P.index:
    t = np.asarray(cache[bg]["t"], float)
    g = np.diff(t); g = g[g > 0] if (np.diff(t) > 0).any() else g
    r = g.std() / g.mean()
    rows.append((bg, (r - 1) / (r + 1), len(t)))
B = pd.DataFrame(rows, columns=["bg", "B", "n"]).set_index("bg")
out["rho_posts_per_session_burstiness"] = round(float(spearmanr(P.posts_per_session, B.B.loc[P.index])[0]), 2)
out["rho_posts_per_session_volume"] = round(float(spearmanr(P.posts_per_session, B.n.loc[P.index])[0]), 2)
out["posts_total"] = int(B.n.sum())
out["posts_median_per_member"] = float(B.n.median())
json.dump(out, open(HERE / f"summary_{tag}.json", "w"), indent=1)
print(json.dumps(out, indent=1))

if "--fig" in sys.argv:
    import matplotlib
    import matplotlib.ticker
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(Path.home() / "x_box/pipeline/scripts/v7"))
    from figs_common import style, plab, BLUE, ORANGE, GREEN, GREY, W2
    style()
    lab = {"casc3": "Cascade (3 levels)", "hawkes_soe": "Hawkes, sum of exponentials", "hawkes_exp": "Hawkes, exponential",
           "mmpp3c": "MMPP-3, clock", "mmpp3": "MMPP-3", "renc_he3": "Renewal, hyperexp.-3, clock", "casc2": "Cascade (2 levels)",
           "renc_he2": "Renewal, hyperexp.-2, clock", "ren_he2": "Renewal, hyperexp.-2", "ren_gm": "Renewal, gamma",
           "ren_wb": "Renewal, Weibull", "renc_wb": "Renewal, Weibull, clock", "ren_ln": "Renewal, lognormal",
           "renc_ln": "Renewal, lognormal, clock", "renc_gm": "Renewal, gamma, clock", "mmpp2": "MMPP-2", "mmpp2c": "MMPP-2, clock",
           "ren_he3": "Renewal, hyperexp.-3", "nhpp_hw": "Poisson, clock"}
    s2 = best.reindex(sorted(mm.model.unique())).fillna(0).sort_values()
    fig, axs = plt.subplots(1, 2, figsize=(W2, 3.6), gridspec_kw={"width_ratios": [1.1, 1], "wspace": 0.08})
    ax = axs[0]
    cols = [BLUE if m == "casc3" else ORANGE if m.startswith("hawkes") else GREY for m in s2.index]
    ax.barh(range(len(s2)), 100 * s2.values, color=cols)
    ax.set_yticks(range(len(s2))); ax.set_yticklabels([lab.get(m, m) for m in s2.index], fontsize=7)
    ax.set_xlabel(f"Members best fit by BIC (%), n = {nmem}")
    plab(ax, "a", dx=-0.8)
    ax = axs[1]
    order = list(s2.index)
    data = [mm[mm.model == m].D.dropna().values for m in order]
    bp = ax.boxplot(data, vert=False, positions=range(len(order)), widths=0.6, whis=(5, 95), showfliers=False, patch_artist=True,
                    medianprops=dict(color="black", lw=1.0), whiskerprops=dict(lw=0.7), capprops=dict(lw=0.7), boxprops=dict(lw=0.7))
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c); patch.set_alpha(0.75)
    ax.set_xscale("log"); ax.set_yticks(range(len(order))); ax.set_yticklabels([])
    ax.set_ylim(axs[0].get_ylim())
    crit = 1.358 / np.sqrt(out["n_median"])
    ax.axvline(crit, color="black", lw=0.7, ls="--")
    ax.annotate("5% critical value at the median stream length", xy=(crit, 1.0), xycoords=("data", "axes fraction"),
                xytext=(0, 3), textcoords="offset points", fontsize=6.5, ha="left", va="bottom")
    ax.set_xticks([0.02, 0.05, 0.1, 0.2]); ax.set_xticklabels(["0.02", "0.05", "0.1", "0.2"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Distance between rescaled intervals\nand the fitted model (KS D)")
    plab(ax, "b", dx=-0.12)
    fig.savefig(HERE / f"figS1_{tag}.png", dpi=600, bbox_inches="tight"); fig.savefig(HERE / f"figS1_{tag}.pdf", bbox_inches="tight")
    print("figure saved")
