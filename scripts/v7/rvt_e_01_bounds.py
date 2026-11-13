"""RVT-E test E1: bounds on the tests of individual history, from stored estimates. Follows scripts/v7/rvt_e_DESIGN.md."""
import json
import os

import numpy as np

ROOT = os.path.expanduser("~/x_box/pipeline/output/v7")
O = ROOT + "/rvt_e"
os.makedirs(O, exist_ok=True)
A = json.load(open(ROOT + "/rvt_a/h1a_results.json"))
B = json.load(open(ROOT + "/rvt_a/h1b_results.json"))
C = json.load(open(ROOT + "/rvt_c/rvt_c_results.json"))


def bound(est, se, margin, bench, ci90=None, ci95=None, extra_ci90=()):
    ci90 = ci90 or [est - 1.645 * se, est + 1.645 * se]
    ci95 = ci95 or [est - 1.96 * se, est + 1.96 * se]
    inside = all(-margin < lo and hi < margin for lo, hi in [ci90, *extra_ci90])
    worst = max(abs(ci95[0]), abs(ci95[1]))
    return dict(est=est, se=se, ci90=ci90, ci95=ci95, margin=margin, benchmark=bench, bounded=bool(inside),
                largest_effect_in_ci95=worst, largest_as_share_of_benchmark=worst / abs(bench), mde_2p8se=2.8 * se)


def fisher_ci(r, n):
    z, s = np.arctanh(r), 1 / np.sqrt(n - 3)
    return [float(np.tanh(z - 1.96 * s)), float(np.tanh(z + 1.96 * s))]


RES = {}
m = A["M1_primary"]
RES["H1a_earlier_time_out_of_power"] = dict(
    bound(m["b"], m["se"], 0.03, 0.10, m["ci90_cl"], m["ci_cl"], extra_ci90=[m["ci90_wild"]]),
    units="change in negative-partisan share", n=m["n"], members=m["members"], ci90_wild=m["ci90_wild"], ci95_wild=m["ci_wild"],
    random_effects_meta=dict(est=A["meta_RE"]["re"], ci95=A["meta_RE"]["ci_re"]))
for k in ("pooled", "D", "R"):
    m = B["primary"][k]
    RES["H1b_did_" + k] = dict(bound(m["b"], m["se"], 0.03, 0.10, m["ci90_cl"], m["ci_cl"], extra_ci90=[m["ci90_wild"]]),
                               units="negative-partisan share", n_rows=m["n_rows"], ci90_wild=m["ci90_wild"], ci95_wild=m["ci_wild"],
                               p_cluster=m["p_cl"], p_wild=m["p_wild"])
RES["H1b_placebos_pooled"] = {k: dict(b=v["pooled"]["b"], se=v["pooled"]["se"], ci95=v["pooled"]["ci_cl"]) for k, v in B["placebos"].items()}

bench_ti7 = C["C4"]["real"]["d_ti7"]["lost_minus_gained"]
bench_lvol = C["C1"]["R_lvol"]["b"]
for out, bench, units in (("R_ti7", bench_ti7, "TI7 per SD of previous margin"), ("R_lvol", bench_lvol, "log event rate per SD of previous margin")):
    m = C["C3"][out]
    RES["C3_" + out] = dict(bound(m["b_per_sd_margin"], m["se"], 0.3 * abs(bench), bench), units=units, n=m["n"], members=m["members"],
                            corr_successive=m["corr_successive"], n_pairs=m["n_pairs"],
                            corr_ci95=fisher_ci(m["corr_successive"], m["n_pairs"]))
RES["benchmarks"] = dict(panel_opposition_effect=0.10, C4_ti7_lost_minus_gained=bench_ti7, C1_lvol_ballot=bench_lvol,
                         C4_lvol_lost_minus_gained=C["C4"]["real"]["d_lvol"]["lost_minus_gained"])
json.dump(RES, open(O + "/e1_bounds.json", "w"), indent=1)
print(json.dumps(RES, indent=1))
