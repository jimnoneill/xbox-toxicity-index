"""T4 pooling: random-effects meta-analysis of account coefficients (REML primary, DerSimonian-Laird comparison), overall and
by party, chamber, account type and scheduler flag; adjusted and unadjusted group contrasts (meta-regression); heterogeneity
(tau, I2, 95% prediction interval); sensitivity variants; likelihood-ratio comparisons that show how much clustering the
Congress term and the own lags absorb; and assembly of numbers_t4.json (validation, predictive and 15-minute results are
merged when their files exist).
Standard errors: week-clustered robust (primary); model-based as a sensitivity (overall only).
Outputs: output/v7/t4/pooled_overall.csv, pooled_groups.csv, contrasts.csv, pooled_variants.csv, lr_tests.csv, numbers_t4.json
Usage: python t4_meta.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t1full_common import ROOT, T1  # noqa: E402
from xbox import binned_model as bm  # noqa: E402

T4 = ROOT / "output/v7/t4"
SE_MAX, B_MAX = 2.0, 8.0
DOUBLING = {"log1p_congress", "log1p_lag1", "log1p_lag2_24", "log1p_congress_daymean", "log1p_congress_dev"}

fits = pd.read_csv(T4 / "account_fits.csv")
summ = pd.read_csv(T4 / "account_summary.csv")
ok = summ[summ.status == "ok"].copy()
meta = ok[ok.variant == "full"][["account", "bioguide", "party", "chamber", "account_type", "n_events", "n_bins"]].copy()
sched = pd.read_csv(T1 / "scheduler_accounts.csv")[["screen_name", "flag_minute"]].rename(columns={"screen_name": "account"})
meta = meta.merge(sched, on="account", how="left")
meta["flag_minute"] = meta.flag_minute.fillna(False).astype(bool)
fits = fits.merge(meta[["account", "party", "chamber", "account_type", "flag_minute"]], on="account", how="inner")


def pool(d, se_col="se_robust"):
    d = d[np.isfinite(d.beta) & np.isfinite(d[se_col]) & (d[se_col] > 0) & (d[se_col] < SE_MAX) & (d.beta.abs() < B_MAX)]
    if len(d) < 5:
        return None
    r = bm.meta_reg(d.beta.values, d[se_col].values, method="REML")
    dl = bm.meta_dl(d.beta.values, d[se_col].values)
    mu, se, tau = float(r["coef"][0]), float(r["se"][0]), r["tau"]
    pi = 1.96 * np.sqrt(tau ** 2 + se ** 2)
    return dict(k=r["k"], mu=mu, se=se, lo=mu - 1.96 * se, hi=mu + 1.96 * se, tau=tau, I2=r["I2"],
                pi_lo=mu - pi, pi_hi=mu + pi, dl_mu=dl["mu"], dl_tau=dl["tau"], unweighted_median=float(d.beta.median()),
                share_positive=float((d.beta > 0).mean()))


def add_rr(row, term):
    f = 2.0 if term in DOUBLING else 1.0
    for a in ("mu", "lo", "hi", "pi_lo", "pi_hi"):
        row[f"RR_{a}"] = float(np.exp(row[a] * (np.log(2.0) if f == 2.0 else 1.0)))
    row["rr_basis"] = "per doubling of (1+x)" if f == 2.0 else "indicator on vs off"
    return row


GROUPS = [("party", ["D", "R"]), ("chamber", ["house", "senate"]), ("account_type", ["office", "campaign"]),
          ("flag_minute", [False, True])]

# ---- overall by variant and term
ov = []
for (v, t), d in fits.groupby(["variant", "term"]):
    r = pool(d)
    if r:
        r2 = pool(d, "se_model")
        r.update(se_model_based_mu=r2["mu"] if r2 else np.nan, se_model_based_se=r2["se"] if r2 else np.nan,
                 tau_model_based=r2["tau"] if r2 else np.nan)
        ov.append(add_rr(dict(variant=v, term=t, accounts_with_term=int(len(d)), **r), t))
ov = pd.DataFrame(ov)
ov.to_csv(T4 / "pooled_variants.csv", index=False)
ovf = ov[ov.variant == "full"]
ovf.to_csv(T4 / "pooled_overall.csv", index=False)

# ---- groups (full)
gr, ct = [], []
ff = fits[fits.variant == "full"]
for t, d in ff.groupby("term"):
    for gname, levels in GROUPS:
        for lv in levels:
            r = pool(d[d[gname] == lv])
            if r:
                gr.append(add_rr(dict(term=t, group_type=gname, group=str(lv), **r), t))
        # unadjusted contrast: second level minus first
        dd = d[d[gname].isin(levels)]
        dd = dd[np.isfinite(dd.beta) & np.isfinite(dd.se_robust) & (dd.se_robust > 0) & (dd.se_robust < SE_MAX) & (dd.beta.abs() < B_MAX)]
        if len(dd) > 20 and dd[gname].nunique() == 2:
            m = (dd[gname] == levels[1]).astype(float).values[:, None]
            r = bm.meta_reg(dd.beta.values, dd.se_robust.values, m, "REML")
            diff, s = float(r["coef"][1]), float(r["se"][1])
            ct.append(dict(term=t, contrast=f"{gname}: {levels[1]} minus {levels[0]}", adjusted=False, k=r["k"], diff=diff,
                           se=s, lo=diff - 1.96 * s, hi=diff + 1.96 * s, p=float(2 * norm.sf(abs(diff / s))),
                           tau_resid=r["tau"], ratio=float(np.exp(diff))))
    # adjusted joint model
    dd = d[d.party.isin(["D", "R"]) & d.chamber.isin(["house", "senate"]) & d.account_type.isin(["office", "campaign"])]
    dd = dd[np.isfinite(dd.beta) & np.isfinite(dd.se_robust) & (dd.se_robust > 0) & (dd.se_robust < SE_MAX) & (dd.beta.abs() < B_MAX)]
    if len(dd) > 40:
        M = np.column_stack([(dd.party == "R"), (dd.chamber == "senate"), (dd.account_type == "campaign"),
                             dd.flag_minute]).astype(float)
        if all(M[:, j].std() > 0 for j in range(4)):
            r = bm.meta_reg(dd.beta.values, dd.se_robust.values, M, "REML")
            for j, lab in enumerate(["party: R minus D", "chamber: senate minus house", "account_type: campaign minus office",
                                     "scheduler flag: on minus off"], start=1):
                diff, s = float(r["coef"][j]), float(r["se"][j])
                ct.append(dict(term=t, contrast=lab + " (adjusted for the other three)", adjusted=True, k=r["k"], diff=diff,
                               se=s, lo=diff - 1.96 * s, hi=diff + 1.96 * s, p=float(2 * norm.sf(abs(diff / s))),
                               tau_resid=r["tau"], ratio=float(np.exp(diff))))
pd.DataFrame(gr).to_csv(T4 / "pooled_groups.csv", index=False)
ctd = pd.DataFrame(ct)
ctd.to_csv(T4 / "contrasts.csv", index=False)

# ---- LR tests and dispersion
piv_ll = ok.pivot(index="account", columns="variant", values="ll")
piv_a = ok.pivot(index="account", columns="variant", values="alpha")
conv = ok.groupby("variant").converged.mean().to_dict()
lr = []
for small, df_ in (("noC", 1), ("nolag", 2), ("base", 3), ("noreg", 9)):
    if small in piv_ll:
        x = piv_ll[["full", small]].dropna()
        stat = 2 * (x["full"] - x[small])
        p = chi2.sf(stat.clip(lower=0), df_)
        lr.append(dict(full_vs=small, df=df_, accounts=int(len(x)), total_ll_gain=float((x["full"] - x[small]).sum()),
                       median_stat=float(stat.median()), share_p_lt_05=float((p < 0.05).mean()),
                       share_p_lt_05_bonferroni=float((p < 0.05 / len(x)).mean())))
lr = pd.DataFrame(lr)
lr.to_csv(T4 / "lr_tests.csv", index=False)
alpha_med = {v: dict(median=float(piv_a[v].median()), q25=float(piv_a[v].quantile(.25)), q75=float(piv_a[v].quantile(.75)))
             for v in piv_a.columns}

# ---- relationships among account coefficients
wide = ff.pivot(index="account", columns="term", values="beta").join(meta.set_index("account")[["n_events"]])
cors = {}
for a, b in (("log1p_congress", "log1p_lag1"), ("log1p_congress", "log1p_lag2_24"), ("log1p_lag1", "log1p_lag2_24"),
             ("log1p_congress", "n_events"), ("log1p_lag1", "n_events")):
    x = wide[[a, b]].dropna()
    cors[f"{a} ~ {b}"] = dict(n=int(len(x)), spearman=float(spearmanr(x[a], x[b])[0]))
# coefficient shift when the Congress term / lags are dropped (same accounts)
shift = {}
f2 = fits.pivot_table(index=["account", "term"], columns="variant", values="beta")
for t in bm.REGIMES + ["log1p_lag1", "log1p_lag2_24"]:
    if t in f2.index.get_level_values("term"):
        x = f2.xs(t, level="term")
        for v in ("noC", "nolag", "base", "noreg"):
            if v in x and "full" in x:
                z = x[["full", v]].dropna()
                if len(z) > 10:
                    shift[f"{t}|{v}"] = dict(n=int(len(z)), median_full=float(z.full.median()), median_alt=float(z[v].median()),
                                             median_abs_diff=float((z.full - z[v]).abs().median()))

# ---- assemble numbers_t4.json
def rr_rows(df):
    keep = ["term", "k", "mu", "lo", "hi", "tau", "I2", "pi_lo", "pi_hi", "RR_mu", "RR_lo", "RR_hi", "rr_basis", "dl_mu",
            "dl_tau", "se_model_based_mu", "se_model_based_se", "tau_model_based", "share_positive"]
    return df[[c for c in keep if c in df]].round(4).to_dict("records")


num = dict(
    design=dict(bin_minutes=60, min_events_per_account=500, accounts_in_hourly_file=int(summ.account.nunique()),
                accounts_fitted_full=int((ok.variant == "full").sum()),
                accounts_skipped=int(((summ.variant == "full") & (summ.status == "skipped")).sum()),
                accounts_error=int(summ.status.str.startswith("error").sum()),
                events_in_model_sample=int(meta.n_events.sum()), bins_in_model_sample=int(meta.n_bins.sum()),
                converged_share_by_variant=conv, standard_errors="week-clustered robust (primary)",
                pooling="REML random effects, DerSimonian-Laird comparison",
                by_party=meta.party.value_counts().to_dict(), by_chamber=meta.chamber.value_counts().to_dict(),
                by_account_type=meta.account_type.value_counts().to_dict(),
                scheduler_flagged=int(meta.flag_minute.sum()),
                regime_columns_dropped_as_unidentified=ok[ok.variant == "full"].n_dropped.describe().round(2).to_dict()),
    pooled_full=rr_rows(ovf),
    pooled_by_group=pd.DataFrame(gr).pipe(lambda d: d[d.term.isin(["log1p_congress", "log1p_lag1", "log1p_lag2_24", "in_session",
                                                                    "primary_window", "general_window", "event_curated",
                                                                    "event_detected"])]).round(4).to_dict("records") if gr else [],
    contrasts_adjusted=ctd[ctd.adjusted & ctd.term.isin(["log1p_congress", "log1p_lag1", "log1p_lag2_24", "in_session",
                                                          "general_window", "event_detected"])].round(4).to_dict("records"),
    contrasts_unadjusted=ctd[~ctd.adjusted & ctd.term.isin(["log1p_congress", "log1p_lag1", "log1p_lag2_24"])].round(4).to_dict("records"),
    sensitivity_pooled_other_variants={v: rr_rows(ov[ov.variant == v]) for v in ("nolag", "noC", "base", "csplit", "noreg")},
    lr_tests=lr.round(4).to_dict("records"), dispersion_alpha_median_by_variant=alpha_med,
    coefficient_correlations=cors, coefficient_shifts_when_terms_dropped=shift,
    guardrail="delta and phi are associations (responsiveness and persistence), not causal effects or diagnostic indicators")
for name, f in (("validation", "validation_summary.json"), ("predictive_checks", "predictive_summary.json"),
                ("sensitivity_15min", "sens15_summary.json")):
    if (T4 / f).exists():
        num[name] = json.load(open(T4 / f))
(T4 / "numbers_t4.json").write_text(json.dumps(num, indent=1, default=float))
pd.set_option("display.width", 250)
print(ovf[["term", "k", "mu", "lo", "hi", "tau", "I2", "RR_mu", "RR_lo", "RR_hi", "dl_mu"]].round(3).to_string())
print(lr.round(3).to_string())
print(alpha_med)
print(ctd[ctd.adjusted & ctd.term.isin(["log1p_congress", "log1p_lag1", "log1p_lag2_24"])].round(3).to_string())

# ---- wide per-account coefficient table (full model)
wb = ff.pivot(index="account", columns="term", values="beta").add_prefix("b_")
ws = ff.pivot(index="account", columns="term", values="se_robust").add_prefix("se_")
(meta.set_index("account")[["bioguide", "party", "chamber", "account_type", "flag_minute", "n_events", "n_bins"]]
 .join(wb).join(ws).join(piv_a["full"].rename("alpha")).reset_index().to_csv(T4 / "account_coefficients.csv", index=False))
