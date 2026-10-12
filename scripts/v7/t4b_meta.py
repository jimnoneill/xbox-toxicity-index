"""T4b pooling and assembly (review fixes). REML random-effects pooling of account coefficients with week-clustered robust SEs.
Reports for every term: pooled estimate and CI, tau, I2, 95% prediction interval (t with k-2 df), share of accounts with a
positive coefficient, Wald p; Holm correction across the 8-term regime family; per-account distributions (CSV for forest and
density plots, with empirical-Bayes shrunken values); group and adjusted contrasts (office vs campaign, Senate vs House, R vs D,
scheduler flag); trend sensitivity (variant "trend") against the main model; comparison with the superseded T4 pooled results;
lag-stability disclosure; LR tests. Associations only.
Outputs: output/v7/t4b/*.csv and numbers_t4b.json (old T4 values under "superseded").
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm, spearmanr
from scipy.stats import t as tdist

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t1full_common import ROOT, T1  # noqa: E402
from xbox import binned_model as bm  # noqa: E402

T4 = ROOT / "output/v7/t4"
T4B = ROOT / "output/v7/t4b"
SE_MAX, B_MAX = 2.0, 8.0
DOUBLING = {"log1p_congress", "log1p_lag1", "log1p_lag2_24", "log1p_congress_daymean", "log1p_congress_dev"}
REG_B = ["in_session", "primary_window", "general_window", "post_election", "aligned_president", "party_majority",
         "event_curated", "event_detected"]

fits = pd.read_csv(T4B / "account_fits.csv")
summ = pd.read_csv(T4B / "account_summary.csv")
ok = summ[summ.status == "ok"].copy()
fs = ok[ok.variant == "full"]
meta = fs[["account", "bioguide", "party", "chamber", "account_type", "n_events", "n_bins", "capped", "branching_free",
           "sim_ratio_max"]].copy()
sched = pd.read_csv(T1 / "scheduler_accounts.csv")[["screen_name", "flag_minute"]].rename(columns={"screen_name": "account"})
meta = meta.merge(sched, on="account", how="left")
meta["flag_minute"] = meta.flag_minute.fillna(False).astype(bool)
fits = fits.merge(meta[["account", "party", "chamber", "account_type", "flag_minute", "capped"]], on="account", how="inner")


def clean(d, se_col="se_robust"):
    return d[np.isfinite(d.beta) & np.isfinite(d[se_col]) & (d[se_col] > 0) & (d[se_col] < SE_MAX) & (d.beta.abs() < B_MAX)]


def pool(d, se_col="se_robust"):
    d = clean(d, se_col)
    if len(d) < 5:
        return None
    r = bm.meta_reg(d.beta.values, d[se_col].values, method="REML")
    dl = bm.meta_dl(d.beta.values, d[se_col].values)
    mu, se, tau = float(r["coef"][0]), float(r["se"][0]), r["tau"]
    k = r["k"]
    tq = float(tdist.ppf(0.975, max(k - 2, 1)))
    pi = tq * np.sqrt(tau ** 2 + se ** 2)
    z = mu / se if se > 0 else np.nan
    return dict(k=k, mu=mu, se=se, lo=mu - 1.96 * se, hi=mu + 1.96 * se, p=float(2 * norm.sf(abs(z))), tau=tau, I2=r["I2"],
                pi_lo=mu - pi, pi_hi=mu + pi, dl_mu=dl["mu"], dl_tau=dl["tau"], unweighted_median=float(d.beta.median()),
                share_positive=float((d.beta > 0).mean()))


def add_rr(row, term):
    dbl = term in DOUBLING
    for a in ("mu", "lo", "hi", "pi_lo", "pi_hi"):
        row[f"RR_{a}"] = float(np.exp(row[a] * (np.log(2.0) if dbl else 1.0)))
    row["rr_basis"] = "per doubling of (1+x)" if dbl else "indicator on vs off"
    return row


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


GROUPS = [("party", ["D", "R"]), ("chamber", ["house", "senate"]), ("account_type", ["office", "campaign"]),
          ("flag_minute", [False, True])]

# ---- overall by variant and term
ov = []
for (v, t), d in fits.groupby(["variant", "term"]):
    if t.startswith("trend") or t == "intercept":
        continue
    r = pool(d)
    if r:
        r2 = pool(d, "se_model")
        r.update(se_model_based_mu=r2["mu"] if r2 else np.nan, se_model_based_se=r2["se"] if r2 else np.nan,
                 tau_model_based=r2["tau"] if r2 else np.nan)
        ov.append(add_rr(dict(variant=v, term=t, accounts_with_term=int(len(d)), **r), t))
ov = pd.DataFrame(ov)
# Holm across the regime family, per variant
ov["p_holm_regime_family"] = np.nan
for v, g in ov[ov.term.isin(REG_B)].groupby("variant"):
    ov.loc[g.index, "p_holm_regime_family"] = holm(g.p.values)
ov["survives_holm_05"] = (ov.p_holm_regime_family < 0.05).astype(object)
ov.loc[~ov.term.isin(REG_B), "survives_holm_05"] = None
ov.to_csv(T4B / "pooled_variants.csv", index=False)
ovf = ov[ov.variant == "full"]
ovf.to_csv(T4B / "pooled_overall.csv", index=False)
ovf[ovf.term.isin(REG_B)][["term", "k", "mu", "p", "p_holm_regime_family", "survives_holm_05", "RR_mu", "RR_lo", "RR_hi",
                          "RR_pi_lo", "RR_pi_hi", "tau", "I2", "share_positive"]].to_csv(T4B / "holm_regime_family.csv", index=False)

# ---- per-account distributions (forest / density plots)
ff = fits[fits.variant == "full"]
dist, qrows = [], []
for t, d in ff.groupby("term"):
    if t == "intercept":
        continue
    d = clean(d).copy()
    r = ovf[ovf.term == t]
    if r.empty:
        continue
    mu, tau = float(r.mu.iloc[0]), float(r.tau.iloc[0])
    d["pooled_mu"] = mu
    d["blup"] = mu + (tau ** 2 / (tau ** 2 + d.se_robust ** 2)) * (d.beta - mu)
    f = np.log(2.0) if t in DOUBLING else 1.0
    d["rr_raw"] = np.exp(d.beta * f)
    d["rr_blup"] = np.exp(d.blup * f)
    dist.append(d[["account", "term", "beta", "se_robust", "blup", "rr_raw", "rr_blup", "pooled_mu", "party", "chamber",
                   "account_type", "capped"]])
    q = d.beta.quantile([.05, .25, .5, .75, .95]).values
    qrows.append(dict(term=t, k=len(d), q05=q[0], q25=q[1], q50=q[2], q75=q[3], q95=q[4], sd_raw=d.beta.std(),
                      sd_blup=d.blup.std(), share_positive=float((d.beta > 0).mean()),
                      share_blup_positive=float((d.blup > 0).mean())))
pd.concat(dist).to_csv(T4B / "account_coefficients_long.csv", index=False)
pd.DataFrame(qrows).to_csv(T4B / "distribution_quantiles.csv", index=False)

# ---- groups and contrasts (full)
gr, ct = [], []
for t, d in ff.groupby("term"):
    if t == "intercept":
        continue
    for gname, levels in GROUPS:
        for lv in levels:
            r = pool(d[d[gname] == lv])
            if r:
                gr.append(add_rr(dict(term=t, group_type=gname, group=str(lv), **r), t))
        dd = clean(d[d[gname].isin(levels)])
        if len(dd) > 20 and dd[gname].nunique() == 2:
            m = (dd[gname] == levels[1]).astype(float).values[:, None]
            r = bm.meta_reg(dd.beta.values, dd.se_robust.values, m, "REML")
            diff, s = float(r["coef"][1]), float(r["se"][1])
            ct.append(dict(term=t, contrast=f"{gname}: {levels[1]} minus {levels[0]}", adjusted=False, k=r["k"], diff=diff,
                           se=s, lo=diff - 1.96 * s, hi=diff + 1.96 * s, p=float(2 * norm.sf(abs(diff / s))),
                           tau_resid=r["tau"], ratio=float(np.exp(diff))))
    dd = clean(d[d.party.isin(["D", "R"]) & d.chamber.isin(["house", "senate"]) & d.account_type.isin(["office", "campaign"])])
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
gr = pd.DataFrame(gr)
gr.to_csv(T4B / "pooled_groups.csv", index=False)
ctd = pd.DataFrame(ct)
ctd.to_csv(T4B / "contrasts.csv", index=False)
old_ct = pd.read_csv(T4 / "contrasts.csv")
cmp_ct = ctd[ctd.adjusted].merge(old_ct[old_ct.adjusted][["term", "contrast", "diff", "lo", "hi", "p"]],
                                 on=["term", "contrast"], how="left", suffixes=("_new", "_old"))
cmp_ct.to_csv(T4B / "contrasts_new_vs_old.csv", index=False)

# ---- trend sensitivity (same accounts as the full fit, paired where both exist)
tv = []
fw = fits[fits.variant == "full"].set_index(["account", "term"])
tw = fits[fits.variant == "trend"].set_index(["account", "term"])
for t in ["log1p_congress", "log1p_lag1", "log1p_lag2_24"] + REG_B:
    a = ovf[ovf.term == t]
    b = ov[(ov.variant == "trend") & (ov.term == t)]
    if a.empty or b.empty:
        continue
    j = fw.xs(t, level="term")[["beta", "se_robust"]].join(tw.xs(t, level="term")[["beta", "se_robust"]], lsuffix="_f",
                                                           rsuffix="_t").dropna()
    j = j[(j.se_robust_f < SE_MAX) & (j.se_robust_t < SE_MAX)]
    r_f = a.iloc[0]
    r_t = b.iloc[0]
    tv.append(dict(term=t, k_full=int(r_f.k), k_trend=int(r_t.k), mu_full=r_f.mu, lo_full=r_f.lo, hi_full=r_f.hi,
                   mu_trend=r_t.mu, lo_trend=r_t.lo, hi_trend=r_t.hi, RR_full=r_f.RR_mu, RR_trend=r_t.RR_mu,
                   RR_lo_trend=r_t.RR_lo, RR_hi_trend=r_t.RR_hi, diff_log=r_t.mu - r_f.mu,
                   paired_median_diff=float((j.beta_t - j.beta_f).median()) if len(j) else np.nan,
                   paired_spearman=float(spearmanr(j.beta_f, j.beta_t)[0]) if len(j) > 5 else np.nan,
                   sign_change=bool(np.sign(r_f.mu) != np.sign(r_t.mu)),
                   ci_full_excludes_0=bool(r_f.lo > 0 or r_f.hi < 0), ci_trend_excludes_0=bool(r_t.lo > 0 or r_t.hi < 0),
                   holm_survives_full=r_f.survives_holm_05 if t in REG_B else np.nan,
                   holm_survives_trend=r_t.survives_holm_05 if t in REG_B else np.nan,
                   tau_full=r_f.tau, tau_trend=r_t.tau, I2_full=r_f.I2, I2_trend=r_t.I2))
tvd = pd.DataFrame(tv)
tvd.to_csv(T4B / "trend_vs_full.csv", index=False)

# ---- comparison with superseded T4
old = pd.read_csv(T4 / "pooled_overall.csv")
oc = old[["term", "mu", "lo", "hi", "tau", "I2", "RR_mu", "RR_lo", "RR_hi", "k"]].rename(columns=lambda c: c if c == "term" else c + "_old")
oc = ovf[["term", "mu", "lo", "hi", "tau", "I2", "RR_mu", "RR_lo", "RR_hi", "k"]].merge(oc, on="term", how="left")
oc["diff_mu"] = oc.mu - oc.mu_old
oc.to_csv(T4B / "pooled_new_vs_old.csv", index=False)

# ---- LR tests and dispersion
piv_ll = ok.pivot(index="account", columns="variant", values="ll")
piv_a = ok.pivot(index="account", columns="variant", values="alpha")
conv = ok.groupby("variant").converged.mean().to_dict()
lr = []
for small, df_ in (("noC", 1), ("nolag", 2), ("base", 3), ("noreg", 8)):
    if small in piv_ll:
        x = piv_ll[["full", small]].dropna()
        stat = 2 * (x["full"] - x[small])
        p = chi2.sf(stat.clip(lower=0), df_)
        lr.append(dict(full_vs=small, df=df_, accounts=int(len(x)), total_ll_gain=float((x["full"] - x[small]).sum()),
                       median_stat=float(stat.median()), share_p_lt_05=float((p < 0.05).mean()),
                       share_p_lt_05_bonferroni=float((p < 0.05 / len(x)).mean())))
lr = pd.DataFrame(lr)
lr.to_csv(T4B / "lr_tests.csv", index=False)
alpha_med = {v: dict(median=float(piv_a[v].median()), q25=float(piv_a[v].quantile(.25)), q75=float(piv_a[v].quantile(.75)))
             for v in piv_a.columns}

# ---- lag stability disclosure
st = fs[["account", "bioguide", "party", "chamber", "account_type", "branching_free", "elasticity_sum_free", "phi1_free", "phi24_free", "capped", "capped_fixed",
         "sim_ratio_max", "n_events"]].copy()
st["unstable_after_cap"] = st.sim_ratio_max > 3
st.sort_values("branching_free", ascending=False).to_csv(T4B / "lag_stability.csv", index=False)
bf = st.branching_free.dropna()
old_unstable = ["DocAbraham", "JahanaHayesCT", "NikemaWilliams", "SteveChabot", "chuckschumer", "joekennedy"]
lag_stab = dict(
    cap=0.9, branching_definition="|phi_1| + |phi_24| (L1 norm of the two log-lag coefficients; below 1 the log-linearised feedback contracts); accounts above the cap were refitted with |phi_1| + |phi_24| = 0.9 imposed (4 sign-face attempts, then phi_1 fixed at 0 and phi_24 at +-0.9; accounts_capped_fixed_at_phi1_zero lists those)",
    unstable_after_cap_note="aocenespanol stays unstable at caps of 0.9, 0.7, 0.5 and 0.3 (simulated total 1.2 to 1.9 times observed at 0.3 to 0.5 and above 3 at 0.9): extremely sparse, heavily overdispersed campaign account (mean 0.05 events per hour, NB alpha about 40); its lag coefficients are not reliable",
    accounts_capped_fixed_at_phi1_zero=sorted(st.loc[st.capped_fixed == True, "account"]),  # noqa: E712
    elasticity_sum_free_quantiles={k: float(st.elasticity_sum_free.quantile(q)) for k, q in (("q05", .05), ("q25", .25), ("q50", .5), ("q75", .75), ("q95", .95))},
    share_capped=float((st.capped == True).mean()),  # noqa: E712
    n_unstable_before_cap_in_t4_sample=6,
    accounts=int(len(st)), free_branching_quantiles={k: float(bf.quantile(q)) for k, q in
                                                      (("q05", .05), ("q25", .25), ("q50", .5), ("q75", .75), ("q95", .95), ("max", 1))},
    accounts_free_branching_ge_cap=int((bf > 0.9).sum()), accounts_free_branching_ge_1=int((bf >= 1).sum()),
    accounts_capped=sorted(st.loc[st.capped == True, "account"]),  # noqa: E712
    unstable_after_cap_sim_ratio_gt_3=sorted(st.loc[st.unstable_after_cap, "account"]),
    previously_unstable_accounts_status=st[st.account.isin(old_unstable)][
        ["account", "branching_free", "capped", "sim_ratio_max"]].round(3).to_dict("records"),
    pooled_lags_all_final=ovf[ovf.term.isin(["log1p_lag1", "log1p_lag2_24"])][["term", "k", "mu", "lo", "hi", "RR_mu"]].round(4).to_dict("records"),
    pooled_lags_free_fit=ov[(ov.variant == "full_uncapped") & ov.term.isin(["log1p_lag1", "log1p_lag2_24"])][
        ["term", "k", "mu", "lo", "hi", "RR_mu"]].round(4).to_dict("records"))
nc = fits[(fits.variant == "full") & fits.term.isin(["log1p_lag1", "log1p_lag2_24"]) & (fits.capped == False)]  # noqa: E712
lag_stab["pooled_lags_never_capped_accounts"] = [dict(term=t, **{k: round(v, 4) for k, v in pool(d).items() if k in ("k", "mu", "lo", "hi")})
                                                 for t, d in nc.groupby("term")]

# ---- regime columns dropped
drop_freq = {}
for t in REG_B:
    d = fits[(fits.variant == "full") & (fits.term == t)]
    drop_freq[t] = float((~np.isfinite(d.beta)).mean()) if len(d) else np.nan


# ---- assemble numbers_t4b.json
def rr_rows(df):
    keep = ["term", "k", "mu", "lo", "hi", "p", "p_holm_regime_family", "survives_holm_05", "tau", "I2", "pi_lo", "pi_hi",
            "RR_mu", "RR_lo", "RR_hi", "RR_pi_lo", "RR_pi_hi", "rr_basis", "share_positive", "dl_mu", "dl_tau"]
    return df[[c for c in keep if c in df]].round(4).to_dict("records")


old_num = json.load(open(T4 / "numbers_t4.json"))
CORE = ["log1p_congress", "log1p_lag1", "log1p_lag2_24"] + REG_B
num = dict(
    design=dict(version="t4b", bin_minutes=60, min_events_per_account=500,
                congress_term="leave-one-out excluding ALL accounts of the same member (bioguide)",
                regimes=REG_B, absent_in_session="dropped (near-unidentified)",
                lag_cap=0.9, accounts_fitted_full=int(len(fs)), events_in_model_sample=int(meta.n_events.sum()),
                bins_in_model_sample=int(meta.n_bins.sum()), converged_share_by_variant=conv,
                standard_errors="week-clustered robust (primary)", pooling="REML random effects; PI uses t with k-2 df",
                by_party=meta.party.value_counts().to_dict(), by_chamber=meta.chamber.value_counts().to_dict(),
                by_account_type=meta.account_type.value_counts().to_dict(), scheduler_flagged=int(meta.flag_minute.sum()),
                trend="natural cubic spline of calendar time per account, df = max(3, round(span in years))",
                regime_term_drop_frequency=drop_freq,
                other_limits="corpus starts 2017-06-21 (first five months of the first Trump term unobserved); "
                             "event_detected days (top 60) come from Congress-wide totals that still include every account"),
    pooled_full=rr_rows(ovf[ovf.term.isin(CORE)]),
    holm_regime_family=rr_rows(ovf[ovf.term.isin(REG_B)]),
    holm_survivors=sorted(ovf[(ovf.term.isin(REG_B)) & (ovf.survives_holm_05 == True)].term),  # noqa: E712
    holm_not_surviving=sorted(ovf[(ovf.term.isin(REG_B)) & (ovf.survives_holm_05 != True)].term),  # noqa: E712
    pooled_trend_sensitivity=tvd.round(4).to_dict("records"),
    pooled_by_group=gr[gr.term.isin(CORE)].round(4).to_dict("records"),
    contrasts_adjusted=ctd[ctd.adjusted & ctd.term.isin(CORE)].round(4).to_dict("records"),
    contrasts_unadjusted=ctd[~ctd.adjusted & ctd.term.isin(CORE[:3])].round(4).to_dict("records"),
    contrasts_new_vs_old=cmp_ct[cmp_ct.term.isin(["log1p_congress", "log1p_lag1", "log1p_lag2_24", "in_session", "aligned_president"])]
    .round(4).to_dict("records"),
    sensitivity_pooled_other_variants={v: rr_rows(ov[(ov.variant == v) & ov.term.isin(CORE)])
                                       for v in ("full_uncapped", "nolag", "noC", "base", "csplit", "noreg")},
    lag_stability=lag_stab, lr_tests=lr.round(4).to_dict("records"), dispersion_alpha_median_by_variant=alpha_med,
    distribution_files=dict(per_account="account_coefficients_long.csv", quantiles="distribution_quantiles.csv"),
    guardrail="delta and phi are associations (responsiveness and persistence), not causal effects or diagnostic indicators",
    superseded=dict(note="T4 results with the focal-account-only LOO term, absent_in_session, uncapped lags (output/v7/t4)",
                    pooled_full=old_num.get("pooled_full"), contrasts_adjusted=old_num.get("contrasts_adjusted"),
                    contrasts_unadjusted=old_num.get("contrasts_unadjusted"), lr_tests=old_num.get("lr_tests"),
                    predictive_checks=old_num.get("predictive_checks"), design=old_num.get("design")))
if (T4B / "predictive_summary.json").exists():
    num["predictive_checks"] = json.load(open(T4B / "predictive_summary.json"))
if (T4B / "sens15_summary.json").exists():
    num["sensitivity_15min"] = json.load(open(T4B / "sens15_summary.json"))
(T4B / "numbers_t4b.json").write_text(json.dumps(num, indent=1, default=float))
pd.set_option("display.width", 250)
print(ovf[ovf.term.isin(CORE)][["term", "k", "mu", "lo", "hi", "tau", "I2", "RR_mu", "RR_pi_lo", "RR_pi_hi", "share_positive",
                               "p_holm_regime_family"]].round(3).to_string())
print(tvd[["term", "mu_full", "mu_trend", "RR_full", "RR_trend", "RR_lo_trend", "RR_hi_trend", "sign_change",
           "holm_survives_full", "holm_survives_trend"]].round(3).to_string())
print(lr.round(3).to_string())
print(json.dumps({k: v for k, v in lag_stab.items() if k != "accounts_capped"}, indent=1, default=float)[:3000])
print(len(lag_stab["accounts_capped"]), "capped")
print(ctd[ctd.adjusted & ctd.term.isin(["log1p_congress", "log1p_lag1", "log1p_lag2_24"])].round(3).to_string())
