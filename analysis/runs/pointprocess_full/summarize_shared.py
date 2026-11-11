"""Summary numbers for the shared-activity models, the misfit diagnostics, the simulation and the full bootstrap.

Usage: python summarize_shared.py [SHARED_DIR]   (default output/v7/temporal_events/shared)
Writes summary_shared.json next to this script.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

HERE = Path(__file__).resolve().parent
D = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "x_box/pipeline/output/v7/temporal_events/shared"
out = {"dir": str(D)}
pct = lambda x: round(100 * float(np.mean(x)), 1)  # noqa: E731

mm = pd.read_csv(D / "members_shared.csv")
if "err" in mm:
    out["fit_errors"] = int(mm.err.notna().sum())
    mm = mm[mm.err.isna()]
out["members"] = int(mm.bg.nunique())
P = {c: mm.pivot_table(index="bg", columns="model", values=c) for c in ["ll", "bic", "aic", "D", "p", "eta", "gamma", "inv_beta_min", "lag1", "ks_at_u", "ks_sign", "n"]}
n = P["n"]["hawkes_exp"]
out["n_median"] = float(n.median())
# does the shared driver improve the Hawkes model?
dll = P["ll"]["hawkes_shared"] - P["ll"]["hawkes_exp"]
out["shared_vs_exp"] = dict(
    median_loglik_gain=round(float(dll.median()), 1),
    median_gain_per_1000_events=round(float((1000 * dll / n).median()), 1),
    lrt_p05_pct=pct(chi2.sf(2 * dll.clip(lower=0), 1) < 0.05),
    bic_prefers_shared_pct=pct(P["bic"]["hawkes_shared"] < P["bic"]["hawkes_exp"]),
    aic_prefers_shared_pct=pct(P["aic"]["hawkes_shared"] < P["aic"]["hawkes_exp"]))
g = P["gamma"]["hawkes_shared"]
out["gamma"] = dict(median=round(float(g.median()), 2), q25=round(float(g.quantile(.25)), 2), q75=round(float(g.quantile(.75)), 2),
                    positive_pct=pct(g > 0), rate_ratio_per_sd_median=round(float(np.exp(g).median()), 2))
# what happens to self-excitation once the shared driver is in the background?
e0, e1 = P["eta"]["hawkes_exp"], P["eta"]["hawkes_shared"]
out["eta"] = dict(median_exp=round(float(e0.median()), 3), median_shared=round(float(e1.median()), 3),
                  q25_exp=round(float(e0.quantile(.25)), 3), q75_exp=round(float(e0.quantile(.75)), 3),
                  q25_shared=round(float(e1.quantile(.25)), 3), q75_shared=round(float(e1.quantile(.75)), 3),
                  lower_with_shared_pct=pct(e1 < e0), median_change=round(float((e1 - e0).median()), 3),
                  median_relative_change_pct=round(100 * float(((e1 - e0) / e0.where(e0 > 1e-6)).median()), 1),
                  decay_min_median_exp=round(float(P["inv_beta_min"]["hawkes_exp"].median()), 1),
                  decay_min_median_shared=round(float(P["inv_beta_min"]["hawkes_shared"].median()), 1))
# is self-excitation still needed beside the shared driver?
d2 = P["ll"]["hawkes_shared"] - P["ll"]["poisson_shared"]
out["hawkes_shared_vs_poisson_shared"] = dict(
    median_loglik_gain=round(float(d2.median()), 1), lrt_p05_pct=pct(chi2.sf(2 * d2.clip(lower=0), 2) < 0.05),
    bic_prefers_hawkes_pct=pct(P["bic"]["hawkes_shared"] < P["bic"]["poisson_shared"]))
d3 = P["ll"]["poisson_shared"] - P["ll"]["poisson"]
out["poisson_shared_vs_poisson"] = dict(median_loglik_gain=round(float(d3.median()), 1),
                                        bic_prefers_shared_pct=pct(P["bic"]["poisson_shared"] < P["bic"]["poisson"]))
best = P["bic"][["hawkes_exp", "hawkes_shared", "poisson_shared", "poisson"]].idxmin(axis=1).value_counts(normalize=True)
out["bic_best_of_four_pct"] = (100 * best).round(1).to_dict()
# time-rescaling test and where the misfit lies
out["ks_pass05_pct"] = {m: pct(P["p"][m] > 0.05) for m in P["p"].columns}
out["ks_D_median"] = {m: round(float(P["D"][m].median()), 4) for m in P["D"].columns}
out["lag1_median"] = {m: round(float(P["lag1"][m].median()), 3) for m in P["lag1"].columns}
out["lag1_beyond_2se_pct"] = {m: pct(P["lag1"][m].abs() > 2 / np.sqrt(n)) for m in P["lag1"].columns}
out["lag1_positive_pct"] = {m: pct(P["lag1"][m] > 0) for m in P["lag1"].columns}
u, s = P["ks_at_u"]["hawkes_exp"], P["ks_sign"]["hawkes_exp"]
out["ks_location_hawkes_exp"] = dict(median_u=round(float(u.median()), 2), q25=round(float(u.quantile(.25)), 2), q75=round(float(u.quantile(.75)), 2),
                                     too_few_small_u_pct=pct(s < 0), too_many_small_u_pct=pct(s > 0))
# refits by calendar year
yy = pd.read_csv(D / "years_shared.csv")
out["years"] = {}
for m, x in yy.groupby("model"):
    full = P["p"][m]
    per = x.groupby("bg").p.apply(lambda v: (v > 0.05).mean())
    out["years"][m] = dict(segments=int(len(x)), members=int(x.bg.nunique()), n_median=float(x.n.median()),
                           pass05_pct=pct(x.p > 0.05), D_median=round(float(x.D.median()), 4),
                           scaled_D_median=round(float((x.D * np.sqrt(x.n)).median()), 2),
                           members_all_years_pass_pct=pct(per == 1), eta_median=round(float(x.eta.median()), 3),
                           eta_within_member_sd_median=round(float(x.groupby("bg").eta.std().median()), 3),
                           by_year_pass05_pct={int(y): pct(v.p > 0.05) for y, v in x.groupby("year")},
                           full_stream_pass05_same_members_pct=pct(full.reindex(x.bg.unique()) > 0.05))
out["scaled_D_median_full"] = {m: round(float((P["D"][m] * np.sqrt(n)).median()), 2) for m in P["D"].columns}
# simulation
sf = D / "sim_shared.csv"
if sf.exists():
    sv = pd.read_csv(sf)
    sv["b_exp"], sv["b_shared"] = sv.eta_exp - sv.eta_true, sv.eta_shared - sv.eta_true
    gsim = sv.groupby(["scen", "n_other", "eta_true"])
    out["sim"] = {f"{a}:{b}:{c}": dict(reps=int(len(v)), bias_exp=round(float(v.b_exp.mean()), 3), bias_shared=round(float(v.b_shared.mean()), 3),
                                        ks_reject_exp_pct=pct(v.p_exp < 0.05), ks_reject_shared_pct=pct(v.p_shared < 0.05),
                                        lrt_detects_pct=pct(chi2.sf(2 * v.dll.clip(lower=0), 1) < 0.05))
                  for (a, b, c), v in gsim}
# bootstrap for every member
bf = D / "bootstrap_all.csv"
if bf.exists():
    bt = pd.read_csv(bf)
    bt = bt[bt.B > 0]
    out["boot"] = dict(members={m: int(v.bg.nunique()) for m, v in bt.groupby("model")}, B_median=float(bt.B.median()),
                       pass05_pct={m: pct(v.boot_p > 0.05) for m, v in bt.groupby("model")},
                       excess_D_median={m: round(float((v.D - v.boot_D_med).median()), 4) for m, v in bt.groupby("model")},
                       null_D_median={m: round(float(v.boot_D_med.median()), 4) for m, v in bt.groupby("model")},
                       D_median={m: round(float(v.D.median()), 4) for m, v in bt.groupby("model")},
                       D_over_null_q95_median={m: round(float((v.D / v.boot_D_q95).median()), 2) for m, v in bt.groupby("model")})
# how much evidence a stream carries for several decay rates over one (gain in log-likelihood per 1,000 events;
# BIC needs dk ln(n) / (2 n) per event, see wolfram/q3_bic_switch.wl)
out["several_rates_gain"] = {}
for tag in ["temporal_events", "temporal_full"]:
    f = D.parents[1] / tag / "model_compare" / "members_models.csv"
    x = pd.read_csv(f)
    x = x[(x.bg != "LEE") & x.ll.notna()]
    a, b = x[x.model == "hawkes_exp"].set_index("bg"), x[x.model == "hawkes_soe"].set_index("bg")
    dk = (b.k - a.k)
    gain = 1000 * (b.ll - a.ll) / a.n
    need = 1000 * dk * np.log(a.n) / (2 * a.n)
    out["several_rates_gain"][tag] = dict(extra_parameters=int(dk.iloc[0]), median_gain_per_1000=round(float(gain.median()), 1),
                                          median_needed_per_1000=round(float(need.median()), 1),
                                          gain_below_1_per_1000_pct=pct(gain < 1), bic_prefers_pct=pct(gain > need))
# how long the leftover dependence lasts (run_memory.py): DFA exponent of the rescaled intervals, 0.5 if independent
mf = D / "memory.csv"
if mf.exists():
    me = pd.read_csv(mf)
    h = me[me.model == "hawkes_exp"]
    med = lambda c: {m: round(float(v[c].median()), 3) for m, v in me.groupby("model")}  # noqa: E731
    out["memory"] = dict(
        members=int(h.bg.nunique()), sims_per_member=float(h.B.median()),
        alpha_median=med("alpha"), alpha_small_windows_median=med("alpha_short"), alpha_large_windows_median=med("alpha_long"),
        null_alpha_median=round(float(h.null_alpha_med.median()), 3),
        null_alpha_small_windows_median=round(float(h.null_alpha_short_med.median()), 3),
        null_alpha_large_windows_median=round(float(h.null_alpha_long_med.median()), 3),
        alpha_above_all_sims_pct=pct(h.null_alpha_p < 0.05),
        alpha_small_above_all_sims_pct=pct(h.null_alpha_short_p < 0.05),
        large_windows_members=int(h.alpha_long.notna().sum()),
        alpha_large_above_all_sims_pct=pct(h.null_alpha_long_p[h.alpha_long.notna()] < 0.05),
        alpha_above_0p75_pct=pct(h.alpha > 0.75),
        acf_median={m: {c[3:]: round(float(v[c].median()), 3) for c in me.columns if c.startswith("acf")}
                    for m, v in me.groupby("model") if m != "raw_intervals"})
json.dump(out, open(HERE / "summary_shared.json", "w"), indent=1)
print(json.dumps(out, indent=1))
