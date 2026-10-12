"""P3 step 2: rolling-origin trajectory forecasts for congress members (political-year panel).

Origins: 2018, 2019, 2020 are DEVELOPMENT origins (each fit only on earlier transitions). Origin 2021 (target PY2022,
1 July 2022 to 30 June 2023 = the untouched final test period) runs only with --final, after the development
results and the model list are frozen.
Usage: python p3_run.py --ti sample|full [--final]
Outputs in output/v7/p3/{ti}/ : transitions.csv, attrition.csv, preds_dev.csv, preds_final.csv, hyperparams*.csv,
skill_*.csv, reliability_*.csv, importance_*.csv, shap_*.csv, numbers_p3_{ti}.json, fig_*.png
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from xbox import forecast_p3 as fp  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEV = [2018, 2019, 2020]
FINAL = [2021]
PAL = {"model": "#0072B2", "base": "#7f7f7f", "alt": "#D55E00", "g": "#009E73"}
REG_MODELS = ["ridge", "enet", "gbt", "ridge_core", "gbt_core"]
BIN_MODELS = ["logit", "gbt", "logit_core", "gbt_core"]
DEPLOY_BASE = ["persistence", "regression_to_mean", "member_history_mean", "party_mean_change", "small_linear"]


def summarize(preds, tag, best_base, outdir):
    """Skill tables for regression targets and binary scores on the given prediction rows."""
    rows = []
    for tg in fp.TARGETS:
        d = preds[preds.target == tg]
        if d.empty:
            continue
        for m in REG_MODELS:
            bt = d[d.model.isin(DEPLOY_BASE)].assign(se=lambda x: (x.y - x.pred) ** 2).groupby("model").se.mean().idxmin()
            for bname, b in (("persistence", "persistence"), ("best_baseline", best_base[tg]),
                             ("best_on_eval_rows", bt)):
                for metric in ("rmse", "mae"):
                    r = fp.skill_ci(d, tg, m, b, metric=metric)
                    rows.append(dict(set=tag, target=tg, model=m, versus=bname, base_model=b, metric=metric, **r))
        # baselines vs persistence and oracle diagnostic
        for m in DEPLOY_BASE[1:] + ["party_year_ORACLE"]:
            if ((d.model == m).any()):
                r = fp.skill_ci(d, tg, m, "persistence")
                rows.append(dict(set=tag, target=tg, model=m, versus="persistence", base_model="persistence",
                                 metric="rmse", **r))
    sk = pd.DataFrame(rows)
    brow = []
    for e in fp.ESC:
        d = preds[preds.target == e]
        if d.empty:
            continue
        for m in BIN_MODELS + ["persistence_score", "rtm_score", "base_rate"]:
            dm = d[d.model == m]
            if dm.empty:
                continue
            a = fp.auc_ci(d, e, m)
            rel, ece = fp.reliability(dm.y.values, dm.pred.values)
            if m in ("persistence_score", "rtm_score"):
                brier = np.nan
            else:
                brier = float(np.mean((dm.y - dm.pred) ** 2))
            rr = {}
            if m in BIN_MODELS:
                r = fp.skill_ci(d, e, m, "base_rate", metric="brier")
                rr = dict(brier_skill=r["skill"], brier_lo=r["lo"], brier_hi=r["hi"])
            brow.append(dict(set=tag, target=e, model=m, auc=a["auc"], auc_lo=a["lo"], auc_hi=a["hi"], brier=brier,
                             ece=ece, prevalence=float(d[d.model == "base_rate"].y.mean()), n=len(dm), **rr))
            rel.assign(set=tag, target=e, model=m).to_csv(outdir / f"reliability_{tag}_{e}_{m}.csv", index=False)
    bn = pd.DataFrame(brow)
    sk.to_csv(outdir / f"skill_reg_{tag}.csv", index=False)
    bn.to_csv(outdir / f"skill_bin_{tag}.csv", index=False)
    return sk, bn


def calib_table(preds, tag, outdir):
    rows = []
    for tg in fp.TARGETS:
        for m in REG_MODELS:
            d = preds[(preds.target == tg) & (preds.model == m)]
            if d.empty:
                continue
            c = fp.calib_slope(d.y.values, d.pred.values)
            rows.append(dict(set=tag, target=tg, model=m, pred_sd=float(d.pred.std()), y_sd=float(d.y.std()),
                             corr=float(np.corrcoef(d.pred, d.y)[0, 1]) if d.pred.std() > 0 else np.nan, **c))
    t = pd.DataFrame(rows)
    t.to_csv(outdir / f"calibration_reg_{tag}.csv", index=False)
    return t


def importance_and_shap(T, origins, outdir, tag, n_rep=10):
    """Permutation importance and SHAP (gradient-boosted trees, full features) on the test rows of each origin."""
    imp_rows, shap_rows = [], []
    for t in origins:
        Ttr = T[T.origin <= t - 1].reset_index(drop=True)
        Tte = T[T.origin == t].reset_index(drop=True)
        if Ttr.origin.nunique() < 2:
            continue
        for tg in list(fp.TARGETS) + list(fp.ESC):
            binary = tg in fp.ESC
            p, hp, mdl = fp.fit_predict("gbt", Ttr, Tte, fp.FEATURES_FULL, tg, binary=binary, return_model=True)
            fn = (lambda X, m=mdl: m.predict_proba(X[fp.FEATURES_FULL])[:, 1]) if binary else \
                 (lambda X, m=mdl: m.predict(X[fp.FEATURES_FULL]))
            imp = fp.perm_importance(fn, Tte, fp.FEATURES_FULL, tg, n_rep=n_rep, binary=binary)
            imp.insert(0, "origin", t)
            imp.insert(1, "target", tg)
            imp_rows.append(imp)
            booster = mdl.booster_
            sv = booster.predict(Tte[fp.FEATURES_FULL], pred_contrib=True)[:, :-1]
            s = pd.DataFrame(dict(feature=fp.FEATURES_FULL, mean_abs_shap=np.abs(sv).mean(0),
                                  mean_shap=sv.mean(0)))
            s.insert(0, "origin", t)
            s.insert(1, "target", tg)
            shap_rows.append(s)
            # per-row signed relation: correlation between feature value and its SHAP (direction of association)
            dirs = [np.corrcoef(Tte[f].fillna(Tte[f].median()).values, sv[:, i])[0, 1]
                    if sv[:, i].std() > 0 and Tte[f].nunique() > 1 else np.nan
                    for i, f in enumerate(fp.FEATURES_FULL)]
            shap_rows[-1]["direction_corr"] = dirs
    imp = pd.concat(imp_rows, ignore_index=True)
    sh = pd.concat(shap_rows, ignore_index=True)
    imp.to_csv(outdir / f"importance_{tag}.csv", index=False)
    sh.to_csv(outdir / f"shap_{tag}.csv", index=False)
    return imp, sh


def noise_ceiling(T):
    out = {}
    for tg, se_cols in (("d_ti7", ("ti7_se", "ti7_se_next")),):
        noise = np.nanmean(T[se_cols[0]] ** 2 + T[se_cols[1]] ** 2)
        v = T[tg].var()
        out[tg] = dict(var_change=float(v), mean_sampling_var=float(noise), reliability=float(1 - noise / v))
    return out


def figs(sk, bn, imp, sh, outdir, tag, preds):
    # skill bars versus best baseline
    d = sk[(sk.versus == "best_baseline") & (sk.metric == "rmse") & sk.model.isin(["ridge", "gbt", "ridge_core", "gbt_core"])]
    if not d.empty:
        tg_list = list(fp.TARGETS)
        fig, ax = plt.subplots(figsize=(9, 4.2))
        w = 0.2
        cols = {"ridge": PAL["model"], "gbt": PAL["alt"], "ridge_core": "#56B4E9", "gbt_core": "#E69F00"}
        for k, m in enumerate(["ridge", "gbt", "ridge_core", "gbt_core"]):
            dd = d[d.model == m].set_index("target").reindex(tg_list)
            x = np.arange(len(tg_list)) + (k - 1.5) * w
            ax.bar(x, dd.skill, w, color=cols[m], label=m)
            ax.errorbar(x, dd.skill, yerr=[dd.skill - dd.lo, dd.hi - dd.skill], fmt="none", ecolor="k", lw=1)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(np.arange(len(tg_list)))
        ax.set_xticklabels([fp.TARGETS[t][1] for t in tg_list])
        ax.set_ylabel("RMSE skill vs best baseline (95% CI over members)")
        ax.set_title(f"Next-year change: skill, {tag} set")
        ax.legend(frameon=False, fontsize=8, ncol=4)
        fig.tight_layout()
        fig.savefig(outdir / f"fig_skill_{tag}.png", dpi=160)
        plt.close(fig)
    # reliability
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.8))
    for ax, e in zip(axs, fp.ESC):
        for m, c in (("logit", PAL["model"]), ("gbt", PAL["alt"])):
            f = outdir / f"reliability_{tag}_{e}_{m}.csv"
            if f.exists():
                r = pd.read_csv(f)
                ax.plot(r.pred, r.obs, "o-", color=c, label=m)
        ax.plot([0, 0.6], [0, 0.6], color="k", lw=0.8, ls=":")
        ax.set_title({"esc_ti7": "TI7 escalation", "esc_vol": "Volume escalation"}[e])
        ax.set_xlabel("predicted probability")
        ax.set_ylabel("observed frequency")
    axs[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / f"fig_reliability_{tag}.png", dpi=160)
    plt.close(fig)
    # importance (SHAP mean abs + permutation) for TI7 and volume
    if imp is not None:
        fig, axs = plt.subplots(2, 2, figsize=(11, 8))
        for j, tg in enumerate(["d_ti7", "d_logvol"]):
            a = imp[(imp.target == tg) & ~imp.feature.str.startswith("GROUP")].groupby("feature").importance.mean()
            a = a.sort_values().tail(12)
            axs[0, j].barh(a.index, a.values, color=PAL["model"])
            axs[0, j].set_title(f"Permutation importance: {fp.TARGETS[tg][1]}")
            axs[0, j].set_xlabel("increase in RMSE")
            b = sh[sh.target == tg].groupby("feature").mean_abs_shap.mean().sort_values().tail(12)
            axs[1, j].barh(b.index, b.values, color=PAL["alt"])
            axs[1, j].set_title(f"Mean |SHAP|: {fp.TARGETS[tg][1]}")
        fig.tight_layout()
        fig.savefig(outdir / f"fig_importance_{tag}.png", dpi=160)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ti", default="sample", choices=["sample", "full"])
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--min_scored", type=int, default=None)
    a = ap.parse_args()
    out = ROOT / "output/v7/p3" / a.ti
    out.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(ROOT / f"output/v7/p3/panel_member_py_{a.ti}.csv")
    panel["seniority_congresses"] = panel.seniority_congresses.clip(lower=0)
    el = pd.read_csv(ROOT / "output/v7/t3/elections.csv")
    su = fp.seat_lookup(el)
    ms = a.min_scored if a.min_scored is not None else (25 if a.ti == "sample" else 75)
    T, att = fp.make_transitions(panel, su, min_scored=ms)
    T.to_csv(out / "transitions.csv", index=False)
    att.to_csv(out / "attrition.csv", index=False)
    print(att.to_string(), flush=True)
    print("transitions", T.groupby("origin").size().to_dict(), flush=True)

    dev, hp_dev = fp.rolling(T, DEV)
    dev.to_csv(out / "preds_dev.csv", index=False)
    hp_dev.to_csv(out / "hyperparams_dev.csv", index=False)
    # best deployable baseline per target decided on development origins only
    best = {}
    for tg in fp.TARGETS:
        d = dev[(dev.target == tg) & dev.model.isin(DEPLOY_BASE)]
        best[tg] = d.assign(se=(d.y - d.pred) ** 2).groupby("model").se.mean().idxmin()
    sk_dev, bn_dev = summarize(dev, "dev", best, out)
    calib_table(dev, "dev", out)
    nums = dict(ti_source=a.ti, min_scored=ms, best_baseline_dev=best, noise_ceiling=noise_ceiling(T),
                transitions_per_origin={int(k): int(v) for k, v in T.groupby("origin").size().items()},
                attrition=att.to_dict("records"))
    imp = sh = None
    if a.final:
        fin, hp_fin = fp.rolling(T, FINAL)
        fin.to_csv(out / "preds_final.csv", index=False)
        hp_fin.to_csv(out / "hyperparams_final.csv", index=False)
        sk_fin, bn_fin = summarize(fin, "final", best, out)
        calib_table(fin, "final", out)
        both = pd.concat([dev, fin])
        sk_all, bn_all = summarize(both, "all", best, out)
        imp, sh = importance_and_shap(T, [2019, 2020, 2021], out, "test", n_rep=10)
        figs(sk_fin, bn_fin, imp, sh, out, "final", both)
        nums["final_skill_reg"] = sk_fin.to_dict("records")
        nums["final_skill_bin"] = bn_fin.to_dict("records")
        nums["all_skill_reg"] = sk_all.to_dict("records")
        nums["all_skill_bin"] = bn_all.to_dict("records")
        nums["importance_top"] = {tg: imp[(imp.target == tg) & ~imp.feature.str.startswith("GROUP")]
                                  .groupby("feature").importance.mean().sort_values(ascending=False).head(8).round(5).to_dict()
                                  for tg in list(fp.TARGETS) + list(fp.ESC)}
        nums["importance_groups"] = {tg: imp[(imp.target == tg) & imp.feature.str.startswith("GROUP")]
                                     .groupby("feature").importance.mean().sort_values(ascending=False).round(5).to_dict()
                                     for tg in list(fp.TARGETS) + list(fp.ESC)}
        nums["shap_top"] = {tg: sh[sh.target == tg].groupby("feature").mean_abs_shap.mean()
                            .sort_values(ascending=False).head(8).round(5).to_dict()
                            for tg in list(fp.TARGETS) + list(fp.ESC)}
    figs(sk_dev, bn_dev, imp, sh, out, "dev", dev)
    nums["dev_skill_reg"] = sk_dev.to_dict("records")
    nums["dev_skill_bin"] = bn_dev.to_dict("records")
    (out / f"numbers_p3_{a.ti}{'_final' if a.final else '_dev'}.json").write_text(json.dumps(nums, indent=1, default=float))
    print("done", flush=True)


if __name__ == "__main__":
    main()
