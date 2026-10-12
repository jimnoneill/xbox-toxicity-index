"""Party contrasts for the temporal features (member level, Republican minus Democrat).

Sample: members with >= 200 posts of the post set in the full window (2017-2023), party D or R (independents and the
three Libertarian accounts are excluded). Post sets: nonrt (primary, non-retweet posts) and all (secondary).
For every feature: Hedges g (R minus D, J = 1 - 3/(4 df - 1)), 95% percentile bootstrap CI from 10,000 resamples of
members stratified by party (fixed seed), Welch t test p, Holm adjustment across the feature family (the 10 features
below, within each post set), and a volume-adjusted standardized coefficient (OLS on party + chamber + log n posts,
HC3 errors, coefficient divided by the SD of the feature) because posting volume drives several of these features.
Supplementary rows (outside the Holm family): eta among members whose time-rescaling KS p >= 0.05.
Output: output/v7/temporal/party_contrasts.csv
Usage: python temporal_party.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from xbox import temporal_v2 as tv  # noqa: E402

OUT = ROOT / "output/v7/temporal"
FAMILY = ["A_n", "B", "entropy_model_trans_dc", "night_00_06_model_trans_dc", "night_22_06_model_trans_dc",
          "weekend_share_model_trans_dc", "weekend_ratio_model_trans_dc", "rapid_share_300", "hawkes_eta",
          "hawkes_inv_beta_min"]
N_BOOT = 10000


def ols_hc3(y, X):
    """OLS coefficients with HC3 standard errors; X includes the intercept."""
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    w = (e / (1 - h)) ** 2
    cov = XtX_inv @ (X.T * w) @ X @ XtX_inv
    return b, np.sqrt(np.diag(cov))


def contrast(d, f, seed):
    r = d[d.party == "R"]
    dm = d[d.party == "D"]
    b = tv.hedges_g_boot(r[f].values, dm[f].values, n_boot=N_BOOT, seed=seed)
    ok = d[f].notna()
    dd = d[ok]
    X = np.column_stack([np.ones(len(dd)), (dd.party == "R").astype(float), (dd.chamber == "senate").astype(float),
                         np.log(dd.n.astype(float))])
    y = dd[f].values.astype(float)
    bb, se = ols_hc3(y, X)
    z = bb[1] / se[1]
    return dict(feature=f, n_R=b["n_x"], n_D=b["n_y"], mean_R=float(r[f].mean()), mean_D=float(dm[f].mean()),
                median_R=float(r[f].median()), median_D=float(dm[f].median()),
                g_R_minus_D=b["g"], ci_lo=b["lo"], ci_hi=b["hi"], welch_p=tv.welch_p(r[f].values, dm[f].values),
                adj_std_coef_R=float(bb[1] / y.std(ddof=1)), adj_p_hc3=float(2 * stats.t.sf(abs(z), len(y) - 4)),
                spearman_with_log_n=float(stats.spearmanr(dd[f], np.log(dd.n))[0]))


def main():
    mf = pd.read_csv(OUT / "member_features.csv")
    rows = []
    for ps in ("nonrt", "all"):
        d = mf[(mf.window == "full") & (mf.postset == ps) & (mf.n >= 200) & mf.party.isin(["R", "D"])]
        fam = []
        for i, f in enumerate(FAMILY):
            c = contrast(d, f, seed=20261008 + i)
            c.update(postset=ps, family="main")
            fam.append(c)
        padj = tv.holm_adjust([c["welch_p"] for c in fam])
        for c, p in zip(fam, padj):
            c["holm_p"] = float(p)
            c["holm_sig05"] = bool(p < 0.05)
        rows += fam
        dw = d[d.hawkes_ks_p >= 0.05]
        c = contrast(dw, "hawkes_eta", seed=20261099)
        c.update(postset=ps, family="supplementary_eta_wellfit_ks05", holm_p=np.nan, holm_sig05=np.nan)
        rows.append(c)
    res = pd.DataFrame(rows)
    first = ["postset", "family", "feature"]
    res = res[first + [c for c in res.columns if c not in first]]
    res.to_csv(OUT / "party_contrasts.csv", index=False)
    pd.set_option("display.width", 250)
    print(res[res.postset == "nonrt"][["feature", "n_R", "n_D", "median_R", "median_D", "g_R_minus_D", "ci_lo", "ci_hi",
                                       "holm_p", "adj_std_coef_R", "adj_p_hc3"]].round(3).to_string())


if __name__ == "__main__":
    main()
