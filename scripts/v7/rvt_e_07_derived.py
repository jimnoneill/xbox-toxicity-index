"""RVT-E: figures derived from the stored RVT-E results for the report. Written after the E1 to E6 results were seen;
everything here is arithmetic on stored estimates and is labelled as derived in the report."""
import json
import os

import numpy as np
from scipy import stats as sst

ROOT = os.path.expanduser("~/x_box/pipeline/output/v7")
O = ROOT + "/rvt_e"
L = lambda f: json.load(open(O + "/" + f))
E2, E3, E4, E5, E6 = L("e2_reliability.json"), L("e3_placebo_outcomes.json"), L("e4_dating.json"), L("e5_feedback_single_account.json"), L("e6_persistence.json")
H1B = json.load(open(ROOT + "/rvt_a/h1b_results.json"))
C = json.load(open(ROOT + "/rvt_c/rvt_c_results.json"))
ARB = ["P1_xyz", "P2_n_to_z", "P3_vowels", "P4_odd_letters"]
D7 = ["p_offensive", "p_toxic", "p_neg_sentiment", "p_irony", "p_ragebait", "p_tribal", "p_performative"]
SEATS = ["2019-01-03", "2021-01-20", "2023-01-03"]
R = {}

# E1: party-specific placebo difference-in-differences of H1b
R["H1b_placebos_by_party"] = {k: {p: dict(b=v[p]["b"], se=v[p]["se"], ci95=v[p]["ci_cl"]) for p in ("pooled", "D", "R") if p in v}
                              for k, v in H1B["placebos"].items()}

# E3: floor set by arbitrary features, at real and at placebo dates
floor_real = max(abs(E3["real"][c]["standardized"]) for c in ARB)
floor_plac = max(abs(E3["placebo_one_year_before"][c]["standardized"]) for c in ARB)
floor_all = max(floor_real, floor_plac)
R["E3"] = dict(floor_real=floor_real, floor_placebo_dates=floor_plac, floor_all=floor_all,
               ti7_real=E3["real"]["ti7"]["standardized"], ti7_placebo_dates=E3["placebo_one_year_before"]["ti7"]["standardized"],
               ti7_real_over_floor_all=abs(E3["real"]["ti7"]["standardized"]) / floor_all,
               ti7_placebo_over_floor_all=abs(E3["placebo_one_year_before"]["ti7"]["standardized"]) / floor_all,
               arbitrary_tests_p_below_05=int(sum(E3[d][c]["p"] < 0.05 for d in ("real", "placebo_one_year_before") for c in ARB)),
               arbitrary_tests_total=2 * len(ARB),
               scores_real_standardized={c: E3["real"][c]["standardized"] for c in E3["real"] if c.startswith("p_")},
               scores_placebo_standardized={c: E3["placebo_one_year_before"][c]["standardized"] for c in E3["real"] if c.startswith("p_")},
               scores_above_floor_all=[c for c in E3["real"] if c.startswith("p_") and abs(E3["real"][c]["standardized"]) > floor_all],
               components_above_floor_all=[c for c in D7 if abs(E3["real"][c]["standardized"]) > floor_all],
               components_not_above_floor_all=[c for c in D7 if abs(E3["real"][c]["standardized"]) <= floor_all])


# E4: the same pooling for the secondary (single-group) series; not part of the reading fixed in the design
def pool(col, ev):
    b = np.array([E4[k][col]["step_" + ev] for k in SEATS])
    se = np.array([E4[k][col]["se_" + ev] for k in SEATS])
    ps = np.array([E4[k][col]["placebo_p_" + ev] for k in SEATS])
    w = 1 / se ** 2
    chi = -2 * np.log(ps).sum()
    psd = np.array([E4[k][col]["placebo_sd_" + ev] for k in SEATS])
    return dict(steps=[float(x) for x in b], mean=float((w * b).sum() / w.sum()), se=float(np.sqrt(1 / w.sum())), placebo_ps=[float(x) for x in ps],
                unweighted_mean=float(b.mean()), se_of_unweighted_mean_from_placebo_sd=float(np.sqrt((psd ** 2).sum()) / len(psd)),
                ratio_placebo_sd_to_hac_se=float(psd.mean() / se.mean()),
                fisher_p=float(sst.chi2.sf(chi, 2 * len(ps))), mean_placebo_sd=float(np.mean([E4[k][col]["placebo_sd_" + ev] for k in SEATS])),
                mean_hac_se=float(se.mean()))


R["E4"] = {col: {ev: pool(col, ev) for ev in ("election", "seating")} for col in ("diff_ti7", "diff_lev", "lost_ti7", "gained_ti7", "lost_lev", "gained_lev")}
R["E4"]["period_means_diff_ti7"] = {k: [E4[k]["diff_ti7"][m] for m in ("mean_before_election", "mean_between", "mean_after_seating")] for k in SEATS}
R["E4"]["period_means_diff_lev"] = {k: [E4[k]["diff_lev"][m] for m in ("mean_before_election", "mean_between", "mean_after_seating")] for k in SEATS}
ys = []
for k in SEATS:
    for sh, x in E4[k]["diff_ti7"]["same_season_other_year"].items():
        ys.append(dict(pair=k, shift_days=int(sh), anchor=x["anchor"], step_election=x["step_election"], step_seating=x["step_seating"]))
R["E4"]["same_season_other_year_diff_ti7"] = ys
R["E4"]["same_season_other_year_lost_ti7_seating"] = [dict(pair=k, shift_days=int(sh), step=x["step_seating"], se=x["se_seating"])
                                                     for k in SEATS for sh, x in E4[k]["lost_ti7"]["same_season_other_year"].items()]

# E5: intervals, and the volume coefficient per doubling of feedback
ln2 = float(np.log(2))
R["E5"] = {}
for k in ("content_lag", "content_lead_placebo", "volume_lag", "volume_lead_placebo"):
    m = E5[k]
    lo, hi = m["b"] - 1.96 * m["se"], m["b"] + 1.96 * m["se"]
    d = dict(ci95=[lo, hi], per_sd_ci95=[lo * m["sd_predictor"], hi * m["sd_predictor"]],
             per_sd_in_outcome_sd_ci95=[lo * m["sd_predictor"] / m["sd_outcome"], hi * m["sd_predictor"] / m["sd_outcome"]])
    if k.startswith("volume"):
        d["pct_posts_per_doubling"] = 100 * (np.exp(m["b"] * ln2) - 1)
        d["pct_posts_per_doubling_ci95"] = [100 * (np.exp(lo * ln2) - 1), 100 * (np.exp(hi * ln2) - 1)]
    R["E5"][k] = d

# E2 and E6 side by side: what persists
R["E2_E6"] = dict(
    response_reliability={v: E2["election_min5"]["response"][v]["reliability"] for v in ("ti7", "lvol")},
    response_cross_cycle={v: E2["election_min5"]["response"][v]["cross_r_demeaned"] for v in ("ti7", "lvol")},
    level_cross_cycle={v: E2["election_min5"]["level"][v]["cross_r_demeaned"] for v in ("ti7", "lvol")},
    weekly_autocorr_ti7={k: E6["u_ti7"][k]["r"] for k in ("1", "4", "13", "26", "52", "104")},
    weekly_autocorr_lev={k: E6["u_lev"][k]["r"] for k in ("1", "4", "13", "26", "52", "104")})
R["benchmarks"] = dict(C4_ti7=C["C4"]["real"]["d_ti7"]["lost_minus_gained"], C4_ti7_by_change=C["C4"]["real"]["d_ti7"]["by_change"],
                       C4_lvol=C["C4"]["real"]["d_lvol"]["lost_minus_gained"], C4_lvol_by_change=C["C4"]["real"]["d_lvol"]["by_change"])
json.dump(R, open(O + "/e7_derived.json", "w"), indent=1)
print(json.dumps(R, indent=1))
