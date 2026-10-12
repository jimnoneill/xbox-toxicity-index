"""Self-excitation reporting and split-state zone effect (review items F2, M7 and m4).

eta is reported as an "excess clustering index (upper bound on self-excitation)": the EM estimate of the branching
ratio under a month x hour-of-week baseline is inflated by any day-level activity shock the baseline cannot see
(validation scenarios C3/C5/C7 quantify the bias).

Eta report (members with >= 200 posts of the post set, windows full and w12, post sets nonrt and all):
  * share of members with an acceptable time-rescaling fit: KS p >= 0.05 and p >= 0.10 (p from scipy.kstest of the
    rescaled gaps against Exp(1); parameters are estimated from the same data, so the p values are, if anything,
    generous). The earlier statistic "KS D < 0.05 / 0.10" is also reported for continuity; it is a distance, not a p.
  * distribution of eta among well-fitting members only
  * Spearman correlation of eta with log post count, all members and well-fitting members
  * sensitivity of the member median eta to the baseline (constant, week, month, day fixed, day random effect)
Zone report: how many members change zone under the district assignment, by state, and the effect on night share,
entropy and weekend share (member-level difference district-zone minus state-zone).
Output: output/v7/temporal/eta_reporting.csv, eta_baseline_sensitivity.csv, zone_effect.json, zone_effect_by_state.csv
Usage: python temporal_reports.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "output/v7/temporal"


def q(x):
    return dict(n=int(len(x)), mean=float(x.mean()), sd=float(x.std()), p05=float(x.quantile(.05)),
                p25=float(x.quantile(.25)), median=float(x.median()), p75=float(x.quantile(.75)),
                p95=float(x.quantile(.95)))


def eta_report(mf):
    rows = []
    for (w, ps), g in mf.groupby(["window", "postset"]):
        if w not in ("full", "w12"):
            continue
        g = g[(g.n >= 200) & g.hawkes_eta.notna() & g.hawkes_ks_p.notna()]
        if len(g) == 0:
            continue
        base = dict(window=w, postset=ps, n_members=len(g))
        r = dict(base)
        for thr in (0.05, 0.10):
            r[f"share_ks_p_ge_{int(thr * 100):02d}"] = float((g.hawkes_ks_p >= thr).mean())
            r[f"n_ks_p_ge_{int(thr * 100):02d}"] = int((g.hawkes_ks_p >= thr).sum())
        for thr in (0.05, 0.10):
            r[f"share_ksD_lt_{int(thr * 100):02d}"] = float((g.hawkes_ks_stat < thr).mean())
        r["spearman_eta_logn_all"] = float(stats.spearmanr(g.hawkes_eta, np.log(g.n))[0])
        for thr in (0.05, 0.10):
            wf = g[g.hawkes_ks_p >= thr]
            tag = f"wellfit{int(thr * 100):02d}"
            if len(wf) >= 10:
                r[f"spearman_eta_logn_{tag}"] = float(stats.spearmanr(wf.hawkes_eta, np.log(wf.n))[0])
                for k, v in q(wf.hawkes_eta).items():
                    r[f"eta_{tag}_{k}"] = v
                r[f"median_n_{tag}"] = float(wf.n.median())
        for k, v in q(g.hawkes_eta).items():
            r[f"eta_all_{k}"] = v
        r["median_n_all"] = float(g.n.median())
        rows.append(r)
    return pd.DataFrame(rows)


def baseline_sensitivity(mf):
    g = mf[(mf.window == "full") & (mf.postset == "nonrt") & (mf.n >= 200)]
    cols = {"hawkes_eta": "month_x_hourofweek (primary)", "hawkes_eta_const_baseline": "constant",
            "hawkes_eta_week_baseline": "week_level", "hawkes_eta_day_baseline": "day_fixed_effect",
            "hawkes_eta_day_re_baseline": "day_random_effect"}
    rows = []
    for c, nm in cols.items():
        if c not in g:
            continue
        x = g[c].dropna()
        rows.append(dict(baseline=nm, n_members=len(x), median=x.median(), mean=x.mean(), p05=x.quantile(.05),
                         p95=x.quantile(.95),
                         spearman_with_logn=stats.spearmanr(g.loc[x.index, c], np.log(g.loc[x.index, "n"]))[0],
                         median_diff_vs_primary=(x - g.loc[x.index, "hawkes_eta"]).median()))
    return pd.DataFrame(rows)


def zone_report(mf):
    za = pd.read_csv(OUT / "member_zone_assignment.csv")
    res = dict(n_members=int(len(za)), n_house=int((za.chamber == "house").sum()),
               n_house_in_split_states=int(((za.chamber == "house") & za.state.isin(
                   ["FL", "TN", "KY", "IN", "MI", "ND", "SD", "NE", "KS", "TX", "ID", "OR"])).sum()),
               n_district_matched=int(za.detail.isin(["district_stable", "district_changes_across_congresses"]).sum()),
               n_no_district_row=int((za.detail == "no_district").sum()),
               n_members_zone_changed=int(za.changed.sum()),
               n_district_changes_across_congresses=int((za.detail == "district_changes_across_congresses").sum()))
    ch = za[za.changed]
    bs = ch.groupby(["state", "state_zone", "home_zone"]).size().reset_index(name="members_changed")
    tot = za[(za.chamber == "house")].groupby("state").size().rename("house_members_in_state")
    bs = bs.merge(tot, left_on="state", right_index=True)
    bs.to_csv(OUT / "zone_effect_by_state.csv", index=False)
    res["changed_by_party"] = ch.party.value_counts().to_dict()
    g = mf[(mf.window == "full") & (mf.postset == "nonrt") & (mf.n >= 200)].merge(
        za[["bioguide", "changed"]], left_on="bioguide", right_on="bioguide")
    d = {}
    for f, a, b in (("night_00_06", "night_00_06_model_trans_dc", "night_00_06_statezone"),
                    ("night_22_06", "night_22_06_model_trans_dc", "night_22_06_statezone"),
                    ("entropy", "entropy_model_trans_dc", "entropy_statezone"),
                    ("weekend_share", "weekend_share_model_trans_dc", "weekend_share_statezone")):
        diff = g[a] - g[b]
        ch_ = diff[g.changed]
        d[f] = dict(qualifying_members=int(diff.notna().sum()),
                    changed_members_qualifying=int(ch_.notna().sum()),
                    mean_diff_changed=float(ch_.mean()) if len(ch_) else None,
                    median_diff_changed=float(ch_.median()) if len(ch_) else None,
                    min_diff_changed=float(ch_.min()) if len(ch_) else None,
                    max_diff_changed=float(ch_.max()) if len(ch_) else None,
                    mean_abs_diff_changed=float(ch_.abs().mean()) if len(ch_) else None,
                    median_all_statezone=float(g[b].median()), median_all_district=float(g[a].median()),
                    mean_all_statezone=float(g[b].mean()), mean_all_district=float(g[a].mean()),
                    max_abs_diff_unchanged=float(diff[~g.changed].abs().max()))
    res["effect_on_features"] = d
    # direction table: changed members by (state zone -> district zone)
    res["changed_direction"] = {f"{r.state_zone} -> {r.home_zone}": int(r.members_changed)
                                for r in bs.groupby(["state_zone", "home_zone"]).members_changed.sum().reset_index().itertuples()}
    return res


def main():
    mf = pd.read_csv(OUT / "member_features.csv")
    er = eta_report(mf)
    er.to_csv(OUT / "eta_reporting.csv", index=False)
    bsens = baseline_sensitivity(mf)
    bsens.to_csv(OUT / "eta_baseline_sensitivity.csv", index=False)
    z = zone_report(mf)
    (OUT / "zone_effect.json").write_text(json.dumps(z, indent=1))
    pd.set_option("display.width", 250, "display.max_columns", 60)
    print(er.T.round(3).to_string())
    print(bsens.round(3).to_string())
    print(json.dumps(z, indent=1))


if __name__ == "__main__":
    main()
