"""Collect every number the manuscript cites from the temporal outputs into numbers_temporal.json
(key: {value, desc}). Deterministic; reads only files written by the other temporal_*.py scripts.

Post sets: nonrt = every post that is not a retweet (primary), all = every post (secondary).
eta is described as an "excess clustering index (upper bound on self-excitation)". Profile-likelihood CIs are NOT
in this file (see hawkes_ci_diagnostics*.csv, model-conditional). The previous version of the numbers (original-post
primary sample) is kept verbatim under the key "superseded".
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[2] / "output/v7/temporal"
OLD = OUT / "superseded_20261008_originals/numbers_temporal.json"
N = {}
ECI = "excess clustering index (upper bound on self-excitation)"


def put(k, v, d):
    if isinstance(v, (np.floating, float)):
        v = None if np.isnan(v) else round(float(v), 4)
    elif isinstance(v, (np.integer,)):
        v = int(v)
    elif isinstance(v, (np.bool_,)):
        v = bool(v)
    N[k] = dict(value=v, desc=d)


def main():
    run = json.load(open(OUT / "features_run.json"))
    put("n_members_matched", run["members"], "members with matched congresstweets accounts (TI account map)")
    put("n_accounts_matched", run["accounts"], "member accounts in the temporal analysis")
    put("n_posts_matched", run["posts_matched"], "congresstweets posts from matched member accounts")
    put("n_posts_retweets_rt_prefix", run["retweets_rt_prefix"], "posts whose text starts 'RT @' (excluded from the primary set)")
    put("n_posts_nonretweet", run["nonretweet_posts"], "non-retweet posts (primary post set)")
    put("n_members_location_model_ok", run["members_loc_model_ok"], "members with >=20 VoteView vote days (location model usable)")
    mf = pd.read_csv(OUT / "member_features.csv")
    af = pd.read_csv(OUT / "account_features.csv")
    fo = mf[(mf.window == "full") & (mf.postset == "nonrt")]
    fa = mf[(mf.window == "full") & (mf.postset == "all")]
    put("n_members_qualifying_nonrt_full", int((fo.n >= 200).sum()), "members with >=200 non-retweet posts, 2017-2023")
    put("n_members_qualifying_all_full", int((fa.n >= 200).sum()), "members with >=200 posts of any type, 2017-2023")
    put("n_accounts_qualifying_nonrt_full", int(((af.postset == "nonrt") & (af.n >= 200)).sum()), "accounts with >=200 non-retweet posts")
    for t in ("office", "campaign", "unknown"):
        put(f"n_accounts_qualifying_{t}", int(((af.postset == "nonrt") & (af.n >= 200) & (af.account_type == t)).sum()),
            f"qualifying {t} accounts")
    for w in ("w12", "y2022", "y2023"):
        x = mf[(mf.window == w) & (mf.postset == "nonrt")]
        put(f"n_members_qualifying_nonrt_{w}", int((x.n >= 200).sum()), f"members with >=200 non-retweet posts in window {w}")
    fs = pd.read_csv(OUT / "feature_summary.csv")
    for _, r in fs.iterrows():
        if r.window in ("full", "w12"):
            k = f"member_{r.window}_{r.postset}_{r.feature}"
            lab = ECI if r.feature.startswith("hawkes_eta") else r.feature
            put(k + "_median", r["median"], f"median across qualifying members: {lab}, {r.window}, {r.postset} posts")
            put(k + "_p05", r["p05"], f"5th percentile across members: {r.feature}")
            put(k + "_p95", r["p95"], f"95th percentile across members: {r.feature}")
    q = fo[fo.n >= 200]
    for c in ("cls_dc_vote", "cls_dc_bracket", "cls_home", "cls_transition"):
        put(f"member_mean_{c}", q[c].mean(), f"mean share of member non-retweet posts on {c} days")
    put("member_zero_gap_share_median", q.zero_gap_share.median(), "median share of same-second gaps, non-retweet posts")
    put("spearman_A_n_vs_n_posts", q[["A_n", "n"]].corr(method="spearman").iloc[0, 1], "Spearman, member A_n vs number of posts")

    # eta reporting (F2, M7)
    er = pd.read_csv(OUT / "eta_reporting.csv")
    for _, r in er.iterrows():
        pre = f"eta_{r.window}_{r.postset}_"
        for c in er.columns:
            if c in ("window", "postset") or pd.isna(r[c]):
                continue
            put(pre + c, r[c], f"{ECI}, {r.window} window, {r.postset} posts: {c}")
    bs = pd.read_csv(OUT / "eta_baseline_sensitivity.csv")
    for _, r in bs.iterrows():
        b = r.baseline.split(" ")[0]
        put(f"eta_baseline_{b}_median", r["median"], f"member median {ECI} under baseline '{r.baseline}' (nonrt, full)")
        put(f"eta_baseline_{b}_median_diff_vs_primary", r["median_diff_vs_primary"],
            f"median member difference in eta, baseline '{r.baseline}' minus primary (nonrt, full)")
        put(f"eta_baseline_{b}_spearman_with_logn", r["spearman_with_logn"], f"Spearman of eta with log n posts, baseline '{r.baseline}'")

    # validation: bias only (CI coverage lives in the diagnostics file)
    v = pd.read_csv(OUT / "hawkes_validation/summary.csv")
    for _, r in v.iterrows():
        k = f"hawkes_val_{r.scen}_n{int(r.n_target)}_eta{r.eta_true}"
        put(k + "_bias", r.bias, "bias of eta_hat (month x hour-of-week baseline)")
        put(k + "_rmse", r.rmse, "RMSE of eta_hat")
        put(k + "_const_eta_mean", r.const_eta_mean, "mean eta_hat with constant baseline")
        put(k + "_week_eta_mean", r.week_eta_mean, "mean eta_hat with week-level baseline")
        if not pd.isna(r.day_bias):
            put(k + "_day_fixed_bias", r.day_bias, "bias of eta_hat with day-level fixed-effect baseline")
            put(k + "_day_re_bias", r.dayre_bias, "bias of eta_hat with day-level random-effect baseline")
        put(k + "_ks_reject5", r.ks_reject5, "share of fits rejected by the KS time-rescaling test at 0.05")
    A = v[v.scen == "A"]
    put("hawkes_val_A_max_abs_bias", A.bias.abs().max(), "max |bias| of eta_hat over scenario A (baseline inside the model class)")
    for sc in ("C3", "C5", "C7", "S5"):
        x = v[v.scen == sc]
        if len(x):
            z = x[x.eta_true == 0.0]
            put(f"hawkes_val_{sc}_bias_at_eta0", z.bias.iloc[0], f"bias at true eta 0, day-shock scenario {sc}")
            put(f"hawkes_val_{sc}_max_bias", x.bias.max(), f"largest bias over eta in the day-shock scenario {sc}")
    for sc, sd in (("C3", 0.3), ("C5", 0.5), ("C7", 0.7), ("S5", 0.5)):
        put(f"hawkes_val_{sc}_daily_sd", sd, f"sd of the day-level log-rate shock in scenario {sc}")

    # staff
    sc_ = json.load(open(OUT / "staff_counts.json"))
    for lv, d in sc_.items():
        for k, x in d.items():
            put(f"staff_{lv}_{k}", x, f"staff signal {lv} level: {k}")
    ss = pd.read_csv(OUT / "staff_signal.csv")
    for _, r in ss.iterrows():
        k = f"staff_{r.level}_{r.feature}"
        put(k + "_g", r.g_phone_minus_tool, "Hedges g, phone-dominant minus tool-dominant")
        put(k + "_ci_lo", r.ci_lo, "bootstrap 95% CI low")
        put(k + "_ci_hi", r.ci_hi, "bootstrap 95% CI high")

    # party contrasts
    pc = pd.read_csv(OUT / "party_contrasts.csv")
    for _, r in pc.iterrows():
        fam = "" if r.family == "main" else "_wellfit"
        k = f"party_{r.postset}_{r.feature}{fam}"
        put(k + "_g", r.g_R_minus_D, "member-level Hedges g, Republican minus Democrat")
        put(k + "_ci_lo", r.ci_lo, "bootstrap 95% CI low (10,000 resamples, stratified by party)")
        put(k + "_ci_hi", r.ci_hi, "bootstrap 95% CI high")
        put(k + "_holm_p", r.holm_p, "Holm-adjusted Welch p across the temporal feature family")
        put(k + "_adj_std_coef", r.adj_std_coef_R, "standardized party coefficient adjusted for chamber and log posts (HC3 OLS)")
        put(k + "_adj_p", r.adj_p_hc3, "HC3 p value of the adjusted party coefficient")
        put(k + "_median_R", r.median_R, "median, Republicans")
        put(k + "_median_D", r.median_D, "median, Democrats")
        put(k + "_n_R", r.n_R, "n Republicans")
        put(k + "_n_D", r.n_D, "n Democrats")

    # zones
    z = json.load(open(OUT / "zone_effect.json"))
    for k in ("n_house", "n_house_in_split_states", "n_district_matched", "n_no_district_row", "n_members_zone_changed",
              "n_district_changes_across_congresses"):
        put(f"zone_{k}", z[k], f"district-level zone assignment: {k}")
    for k, v_ in z["changed_direction"].items():
        put(f"zone_changed_{k.replace('America/', '').replace(' -> ', '_to_')}", v_, f"members whose zone changes {k}")
    for f, d in z["effect_on_features"].items():
        for k, x in d.items():
            put(f"zone_effect_{f}_{k}", x, f"effect of district vs state zone on {f}: {k}")

    # Lee
    for tag in ("xlsx", "v2_parquet"):
        s = json.load(open(OUT / f"lee_location_summary_{tag}.json"))
        put(f"lee_{tag}_n_posts", s["n_posts"], "Lee archive posts")
        put(f"lee_{tag}_last_post", s["last"], "last post in archive")
        put(f"lee_{tag}_vote_days", s["lee_vote_days"], "distinct days with a cast Senate vote (S115-S119)")
        for tp, x in s["type_counts"].items():
            put(f"lee_{tag}_type_{tp}", x, f"Lee archive posts of type {tp}")
        for sc, d in s["scenarios"].items():
            for m, x in d.items():
                put(f"lee_{tag}_{sc}_{m}", x, f"Lee all posts, scenario {sc}: {m}")
        for c, x in s["day_class_posts"].items():
            put(f"lee_{tag}_posts_{c}", x, f"Lee posts on {c} days")
        lf = pd.read_csv(OUT / f"lee_features_{tag}.csv")
        cols = ("n", "A_n", "B", "entropy_model_trans_dc", "entropy_all_dc", "night_00_06_all_dc", "night_00_06_all_home",
                "night_00_06_model_trans_dc", "night_00_06_model_trans_home", "night_22_06_model_trans_dc",
                "weekend_share_model_trans_dc", "weekend_ratio_model_trans_dc", "rapid_share_300", "zero_gap_share",
                "hawkes_eta", "hawkes_inv_beta_min", "hawkes_ks_stat", "hawkes_ks_p",
                "hawkes_eta_const_baseline", "hawkes_eta_week_baseline", "hawkes_eta_day_baseline",
                "hawkes_eta_day_re_baseline")
        for _, r in lf.iterrows():
            if r.postset not in ("nonrt", "all", "tweet_quote", "tweet") or r.n < 200:
                continue
            for c in cols:
                if c in r and not pd.isna(r[c]):
                    lab = ECI if c.startswith("hawkes_eta") else c
                    put(f"lee_{tag}_{r.window}_{r.postset}_{c}", r[c], f"Lee {r.window} {r.postset}: {lab}")
        lp = pd.read_csv(OUT / f"lee_percentiles_{tag}.csv")
        for _, r in lp.iterrows():
            k = f"lee_{tag}_{r.lee_window}_vs_{r.ref_window}_{r.feature}"
            put(k + "_pct_all", r.pct_all, f"Lee percentile among n={r.n_ref} members ({r.ref_note})")
            put(k + "_pct_vm", r.pct_volume_matched, "Lee volume-matched percentile (decile of post count)")
            if r.feature == "hawkes_eta":
                for nm in ("wellfit05", "wellfit10"):
                    put(f"{k}_pct_{nm}", r[f"pct_all_{nm}"], f"Lee {ECI} percentile among well-fitting members (KS p >= "
                        f"{'0.05' if nm.endswith('05') else '0.10'}), n={r[f'n_ref_{nm}']}")
                    put(f"{k}_pct_vm_{nm}", r[f"pct_vm_{nm}"], "same, volume-matched within the well-fitting members")
    if OLD.exists():
        N["superseded"] = dict(
            desc="numbers_temporal.json before the 2026-10-08 review fixes: 'original posts' (no retweet, no '@' "
                 "prefix; Lee tweet+quote) primary sample, state-majority zones, profile-likelihood CIs. Do not cite.",
            value=json.load(open(OLD)))
    (OUT / "numbers_temporal.json").write_text(json.dumps(N, indent=1))
    print(len(N), "numbers")


if __name__ == "__main__":
    main()
