"""Empirical and volume-matched percentiles of every temporal feature among qualifying members
(n >= 200 non-retweet posts, postset 'nonrt', or all posts, 'all', in the window), per window and post set. Ranks only, no z scores.
Volume-matched = rank within the member's decile of post count (deciles cut on the qualifying members).
Output: output/v7/temporal/member_features_pct.csv, feature_summary.csv
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from xbox import temporal_v2 as tv  # noqa: E402

OUT = ROOT / "output/v7/temporal"
FEATS = ["A_n", "B", "entropy_model_trans_dc", "night_00_06_model_trans_dc", "night_22_06_model_trans_dc",
         "weekend_ratio_model_trans_dc", "rapid_share_300", "hawkes_eta", "hawkes_inv_beta_min"]


def main():
    df = pd.read_csv(OUT / "member_features.csv")
    parts = []
    for (w, ps), g in df.groupby(["window", "postset"]):
        g = g[g.n >= 200].copy()
        for f in FEATS:
            if f not in g:
                continue
            v = g[f].values
            if np.isnan(v).sum() > len(v) - 10:
                continue
            g["pct_" + f] = [tv.emp_percentile(x, v) for x in v]
            g["vm_" + f] = [np.nan if np.isnan(x) else tv.volume_matched_percentile(x, n, v, g.n.values)[0] for x, n in zip(v, g.n.values)]
        parts.append(g)
    out = pd.concat(parts)
    out.to_csv(OUT / "member_features_pct.csv", index=False)
    rows = []
    for (w, ps), g in out.groupby(["window", "postset"]):
        for f in FEATS + ["entropy_all_dc", "entropy_all_home", "night_00_06_all_dc", "night_00_06_all_home",
                          "night_00_06_model_trans_home", "weekend_share_model_trans_dc", "zero_gap_share",
                          "hawkes_ks_stat", "hawkes_eta_const_baseline", "hawkes_eta_week_baseline",
                          "hawkes_eta_day_baseline", "hawkes_eta_day_re_baseline", "hawkes_ks_p",
                          "entropy_statezone", "night_00_06_statezone", "night_22_06_statezone"]:
            if f in g and g[f].notna().any():
                x = g[f].dropna()
                rows.append(dict(window=w, postset=ps, feature=f, n_members=len(x), mean=x.mean(),
                                 sd=x.std(), p05=x.quantile(.05), p25=x.quantile(.25), median=x.median(),
                                 p75=x.quantile(.75), p95=x.quantile(.95)))
    pd.DataFrame(rows).to_csv(OUT / "feature_summary.csv", index=False)
    print(pd.DataFrame(rows).query("window=='full' and postset=='nonrt'").round(3).to_string())


if __name__ == "__main__":
    main()
