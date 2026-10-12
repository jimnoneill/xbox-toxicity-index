"""P2 analysis: skill vs best baseline with bootstrap CIs over accounts, winners by horizon and account group, burst metrics,
PIT histograms, figures, numbers_p2.json.

usage: p2_analyze.py --run dev_sample_full [--run test_sample_full] [--tag sample]
Skill = 1 - loss_model / loss_best_baseline (loss: negative log score, CRPS, MAE). Account-level averages, bootstrap over accounts
(B=1000, seed 20261010). Baselines: pers, snaive, hist, profmean (profmean = historical hour-of-week profile mean).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/v7/p2"
SEED = 20261010
BASE_C = ["pers", "snaive", "hist", "profmean"]
BASE_B = ["base", "pers", "snaive"]
H_ORDER = ["h1", "d24", "w168"]
MODEL_ORDER = ["pers", "snaive", "hist", "profmean", "nb0", "nb_nocong", "nb", "gbm", "hawkes"]
LAB = {"pers": "Persistence", "snaive": "Seasonal naive", "hist": "Historical mean", "profmean": "Hour-of-week mean",
       "nb0": "NB level+profile", "nb_nocong": "NB own lags", "nb": "NB + Congress lags", "gbm": "Boosted trees", "hawkes": "Hawkes",
       "base": "Base rate", "logit": "Logistic (NB features)"}
B = 1000


def load_run(run):
    d = OUT / run
    agg = pd.read_parquet(d / "agg.parquet")
    burst = pd.read_parquet(d / "burst.parquet")
    pit = pd.read_parquet(d / "pit.parquet")
    acc = json.load(open(d / "accounts.json"))
    st = pd.DataFrame(acc["static"])
    st["tot"] = st["tot"].astype(float)
    st["vol"] = pd.qcut(st.tot, 3, labels=["low", "mid", "high"]).astype(str)
    return agg, burst, pit, st


def acct_means(agg, metric):
    g = agg.groupby(["h", "model", "account"]).agg(n=("n", "sum"), v=(metric, "sum"))
    g["m"] = g.v / g.n
    return g["m"].unstack("account")


def loss(agg, metric):
    m = acct_means(agg, metric)
    if metric == "ls":
        m = -m
    return m


def boot_idx(A, rng):
    return rng.integers(0, A, (B, A))


def skill_table(L, h, models, bases, accts, rng):
    """L: DataFrame (h, model) x account loss. returns dict model -> (skill, lo, hi, base_used, winfrac)"""
    Lh = L.loc[h]
    cols = [a for a in accts if a in Lh.columns]
    Lh = Lh[cols]
    base_means = Lh.loc[[b for b in bases if b in Lh.index]].mean(axis=1)
    bb = base_means.idxmin()
    idx = boot_idx(len(cols), rng)
    out = {}
    X = Lh.values
    bi = list(Lh.index).index(bb)
    for mname in models:
        if mname not in Lh.index:
            continue
        mi = list(Lh.index).index(mname)
        sk = 1 - X[mi].mean() / X[bi].mean()
        bs = 1 - X[mi][idx].mean(1) / X[bi][idx].mean(1)
        out[mname] = dict(skill=float(sk), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)), baseline=bb,
                          loss=float(X[mi].mean()), baseline_loss=float(X[bi].mean()),
                          frac_accounts_better=float((X[mi] < X[bi]).mean()))
    return out


def group_defs(st):
    g = {"all": st.index.astype(int)}
    for col in ("typ", "party", "chamber", "vol"):
        for v in sorted(st[col].unique()):
            g[f"{col}={v}"] = st.index[st[col] == v].astype(int)
    return g


def auc(y, p):
    from scipy.stats import rankdata
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(p)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def ece(y, p, nb=10):
    q = np.quantile(p, np.linspace(0, 1, nb + 1))
    q[0], q[-1] = -1, 2
    b = np.clip(np.searchsorted(q, p, side="right") - 1, 0, nb - 1)
    t = 0.0
    tab = []
    for k in range(nb):
        m = b == k
        if m.sum():
            t += m.mean() * abs(p[m].mean() - y[m].mean())
            tab.append((float(p[m].mean()), float(y[m].mean()), int(m.sum())))
    return float(t), tab


def burst_metrics(bu, st, rng, models):
    out = {}
    accts = np.sort(bu.account.unique())
    A = len(accts)
    pos = {a: i for i, a in enumerate(accts)}
    ai = bu.account.map(pos).values
    y = bu.y.values.astype(int)
    brier = {}
    macro_auc = {}
    for m in models:
        if m not in bu:
            continue
        p = bu[m].values
        sq = (p - y) ** 2
        brier[m] = np.bincount(ai, sq, A) / np.bincount(ai, None, A)
        macro = np.full(A, np.nan)
        order = np.argsort(ai, kind="stable")
        bounds = np.searchsorted(ai[order], np.arange(A + 1))
        for i in range(A):
            sl = order[bounds[i]:bounds[i + 1]]
            sc = np.zeros(len(p)) if m == "base" else (bu[m + "_i"].values if (m + "_i") in bu else p)
            macro[i] = auc(y[sl], sc[sl])
        macro_auc[m] = macro
    bases = [b for b in BASE_B if b in brier]
    bb = min(bases, key=lambda b: brier[b].mean())
    idx = boot_idx(A, rng)
    for m in brier:
        pool_auc = auc(y, bu[m].values)
        e, tab = ece(y, bu[m].values)
        sk = 1 - brier[m].mean() / brier[bb].mean()
        bs = 1 - brier[m][idx].mean(1) / brier[bb][idx].mean(1)
        ma = macro_auc[m]
        ok = np.isfinite(ma)
        mb = np.array([np.nanmean(ma[ix]) for ix in idx[:300]])
        out[m] = dict(brier=float(brier[m].mean()), brier_skill=float(sk), skill_lo=float(np.percentile(bs, 2.5)),
                      skill_hi=float(np.percentile(bs, 97.5)), baseline=bb, auc_pooled=float(pool_auc),
                      auc_macro=float(np.nanmean(ma)), auc_macro_lo=float(np.percentile(mb, 2.5)),
                      auc_macro_hi=float(np.percentile(mb, 97.5)), ece=e, reliability=tab, base_rate=float(y.mean()), n=int(len(y)))
    return out, brier, accts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--tag", default="sample")
    a = ap.parse_args()
    numbers = {}
    figdir = OUT / "figs"
    figdir.mkdir(exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for run in a.run:
        period = run.split("_")[0]
        agg, burst, pit, st = load_run(run)
        rng = np.random.default_rng(SEED)
        num = {"n_accounts": int(agg.account.nunique()), "n_forecasts": {h: int(agg[(agg.h == h) & (agg.model == "nb")].n.sum()) for h in H_ORDER}}
        models = [m for m in MODEL_ORDER if m in set(agg.model)]
        groups = group_defs(st)
        for metric in ("ls", "crps", "ae"):
            L = loss(agg, metric)
            num[metric] = {}
            for h in H_ORDER:
                if h not in L.index.get_level_values(0):
                    continue
                num[metric][h] = {}
                for gname, ids in groups.items():
                    t = skill_table(L, h, models, BASE_C, ids, np.random.default_rng(SEED))
                    num[metric][h][gname] = t
        # winners
        L = loss(agg, "ls")
        C = loss(agg, "crps")
        num["winners"] = {}
        for h in H_ORDER:
            num["winners"][h] = {}
            for gname, ids in groups.items():
                row = {}
                for nm, LL in (("ls", L), ("crps", C)):
                    cols = [x for x in ids if x in LL.loc[h].columns]
                    mm = LL.loc[h][cols].mean(axis=1)
                    mm = mm[[m for m in models if m in mm.index]]
                    row[nm] = mm.idxmin()
                    # share of accounts for which each model is best
                    best = LL.loc[h][cols].loc[mm.index].idxmin()
                    row[nm + "_share"] = best.value_counts(normalize=True).round(3).to_dict()
                num["winners"][h][gname] = row
        # paired difference of the leading models (gbm, nb, hawkes vs nb)
        num["paired"] = {}
        for h in H_ORDER:
            Lh = L.loc[h]
            idx = boot_idx(Lh.shape[1], np.random.default_rng(SEED))
            for m1, m2 in (("nb", "nb0"), ("nb", "nb_nocong"), ("gbm", "nb"), ("hawkes", "nb"), ("hawkes", "gbm")):
                if m1 in Lh.index and m2 in Lh.index:
                    d = (Lh.loc[m2].values - Lh.loc[m1].values)       # positive: m1 better (lower loss)
                    bs = d[idx].mean(1)
                    num["paired"][f"{h}:{m1}_vs_{m2}"] = dict(mean_ls_gain=float(d.mean()), lo=float(np.percentile(bs, 2.5)),
                                                              hi=float(np.percentile(bs, 97.5)))
        # burst
        bmodels = [m for m in BASE_B + ["logit", "gbm", "hawkes"] if m in burst]
        bm, brier, accts = burst_metrics(burst, st, np.random.default_rng(SEED), bmodels)
        num["burst"] = bm
        num["burst_groups"] = {}
        for gname, ids in groups.items():
            if gname == "all":
                continue
            sub = burst[burst.account.isin(ids)]
            if len(sub) > 500 and sub.y.sum() > 20:
                g_, _, _ = burst_metrics(sub, st, np.random.default_rng(SEED), bmodels)
                num["burst_groups"][gname] = {m: dict(brier_skill=v["brier_skill"], lo=v["skill_lo"], hi=v["skill_hi"],
                                                      auc_macro=v["auc_macro"], baseline=v["baseline"]) for m, v in g_.items()}
        # PIT
        pi = pit.groupby(["h", "model", "bin"]).agg(size=("size", "sum")).reset_index()
        num["pit"] = {f"{h}:{m}": (d.sort_values("bin")["size"] / d["size"].sum()).round(4).tolist()
                      for (h, m), d in pi.groupby(["h", "model"])}
        numbers[period] = num
        # ---------------- figures
        fig, axs = plt.subplots(1, 3, figsize=(13, 3.8), sharey=False)
        cols = {"nb0": "#9ecae1", "nb_nocong": "#6baed6", "nb": "#2171b5", "gbm": "#e6550d", "hawkes": "#31a354"}
        for ax, h in zip(axs, H_ORDER):
            for k, m in enumerate([m for m in cols if m in models]):
                s = num["ls"][h]["all"].get(m)
                if s is None:
                    continue
                ax.errorbar(k, s["skill"], yerr=[[s["skill"] - s["lo"]], [s["hi"] - s["skill"]]], fmt="o", color=cols[m], capsize=3)
                c = num["crps"][h]["all"][m]
                ax.errorbar(k + 0.2, c["skill"], yerr=[[c["skill"] - c["lo"]], [c["hi"] - c["skill"]]], fmt="s", color=cols[m], mfc="white", capsize=3)
            ax.axhline(0, color="grey", lw=.8)
            ax.set_xticks(range(len([m for m in cols if m in models])))
            ax.set_xticklabels([LAB[m] for m in cols if m in models], rotation=30, ha="right", fontsize=8)
            ax.set_title({"h1": "Next hour", "d24": "Next 24 h", "w168": "Next week"}[h])
        axs[0].set_ylabel("Skill vs best baseline\n(circle: log score, square: CRPS)")
        fig.tight_layout()
        fig.savefig(figdir / f"fig_skill_{period}_{a.tag}.png", dpi=150)
        plt.close(fig)
        # skill by account type
        fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
        for ax, h in zip(axs, H_ORDER):
            for k, m in enumerate([m for m in ("nb", "gbm", "hawkes") if m in models]):
                for j, g in enumerate(("typ=office", "typ=campaign")):
                    s = num["ls"][h][g].get(m)
                    if s:
                        ax.errorbar(k + j * .25, s["skill"], yerr=[[s["skill"] - s["lo"]], [s["hi"] - s["skill"]]], fmt="o",
                                    color=["#2171b5", "#d94801"][j], capsize=3, label=g.split("=")[1] if k == 0 else None)
            ax.axhline(0, color="grey", lw=.8)
            ax.set_xticks(range(3))
            ax.set_xticklabels(["NB + Congress", "Boosted trees", "Hawkes"], fontsize=8)
            ax.set_title({"h1": "Next hour", "d24": "Next 24 h", "w168": "Next week"}[h])
        axs[0].legend(fontsize=8)
        axs[0].set_ylabel("Log-score skill vs best baseline")
        fig.tight_layout()
        fig.savefig(figdir / f"fig_skill_by_type_{period}_{a.tag}.png", dpi=150)
        plt.close(fig)
        # burst reliability + AUC
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        for m, v in bm.items():
            r = np.array(v["reliability"])
            axs[0].plot(r[:, 0], r[:, 1], "o-", ms=3, label=LAB.get(m, m))
        mx = max(np.array(v["reliability"])[:, 0].max() for v in bm.values())
        axs[0].plot([0, mx], [0, mx], "k--", lw=.8)
        axs[0].set_xscale("symlog", linthresh=1e-3)
        axs[0].set_yscale("symlog", linthresh=1e-3)
        axs[0].set_xlabel("Predicted burst probability (decile mean)")
        axs[0].set_ylabel("Observed frequency")
        axs[0].legend(fontsize=7)
        ms = list(bm)
        axs[1].bar(range(len(ms)), [bm[m]["auc_macro"] for m in ms], yerr=[[bm[m]["auc_macro"] - bm[m]["auc_macro_lo"] for m in ms],
                   [bm[m]["auc_macro_hi"] - bm[m]["auc_macro"] for m in ms]], color="#6baed6", capsize=3)
        axs[1].set_xticks(range(len(ms)))
        axs[1].set_xticklabels([LAB.get(m, m) for m in ms], rotation=30, ha="right", fontsize=8)
        axs[1].set_ylabel("Within-account AUC (mean over accounts)")
        axs[1].set_ylim(0.4, 1)
        fig.tight_layout()
        fig.savefig(figdir / f"fig_burst_{period}_{a.tag}.png", dpi=150)
        plt.close(fig)
        # PIT
        keys = [k for k in num["pit"] if k.split(":")[1] in ("nb", "gbm", "hawkes")]
        fig, axs = plt.subplots(3, 3, figsize=(9, 7), sharey=True)
        for i, h in enumerate(H_ORDER):
            for j, m in enumerate(("nb", "gbm", "hawkes")):
                k = f"{h}:{m}"
                if k in num["pit"]:
                    axs[i, j].bar(np.arange(10) + .5, num["pit"][k], width=.9, color="#6baed6")
                    axs[i, j].axhline(.1, color="k", lw=.8, ls="--")
                axs[i, j].set_title(f"{h} {LAB[m]}", fontsize=8)
        fig.tight_layout()
        fig.savefig(figdir / f"fig_pit_{period}_{a.tag}.png", dpi=150)
        plt.close(fig)
    p = OUT / f"numbers_p2_{a.tag}.json"
    json.dump(numbers, open(p, "w"), indent=1, default=float)
    print("wrote", p)


if __name__ == "__main__":
    main()
