"""P3 step 4: assemble output/v7/p3/numbers_p3.json from the stage outputs of one TI source.
Usage: python p3_numbers.py --ti sample|full"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ti", default="sample")
    a = ap.parse_args()
    o = ROOT / "output/v7/p3" / a.ti
    n = json.loads((o / f"numbers_p3_{a.ti}_final.json").read_text())
    out = dict(ti_source=a.ti, min_scored=n["min_scored"], transitions_per_origin=n["transitions_per_origin"],
               attrition=n["attrition"], noise_ceiling_d_ti7=n["noise_ceiling"]["d_ti7"],
               best_baseline_chosen_on_dev=n["best_baseline_dev"])
    panel = pd.read_csv(ROOT / f"output/v7/p3/panel_member_py_{a.ti}.csv")
    out["panel"] = dict(member_py_rows=int(len(panel)), members=int(panel.bioguide.nunique()),
                        rows_by_py={int(k): int(v) for k, v in panel.groupby("py").size().items()})
    for tag in ("dev", "final", "all"):
        sk = pd.read_csv(o / f"skill_reg_{tag}.csv")
        cal = pd.read_csv(o / f"calibration_reg_{'final' if tag == 'final' else 'dev'}.csv") if tag != "all" else None
        res = {}
        for tg, g in sk[sk.metric == "rmse"].groupby("target"):
            r = {}
            for m in ("ridge", "enet", "gbt", "ridge_core", "gbt_core"):
                gm = g[g.model == m]
                r[m] = {v: dict(skill=float(x.skill), lo=float(x.lo), hi=float(x.hi), rmse_model=float(x.score_model),
                                rmse_base=float(x.score_base), base=x.base_model)
                        for v, x in gm.set_index("versus").iterrows()}
            r["baselines_vs_persistence"] = {m: dict(skill=float(x.skill), lo=float(x.lo), hi=float(x.hi))
                                             for m, x in g[(g.versus == "persistence") & ~g.model.isin(
                                                 ["ridge", "enet", "gbt", "ridge_core", "gbt_core"])]
                                             .set_index("model").iterrows()}
            res[tg] = r
        out[f"reg_{tag}"] = res
        bn = pd.read_csv(o / f"skill_bin_{tag}.csv")
        out[f"bin_{tag}"] = {e: {x.model: {k: (None if pd.isna(x[k]) else float(x[k])) for k in
                                           ("auc", "auc_lo", "auc_hi", "brier", "ece", "brier_skill", "brier_lo", "brier_hi",
                                            "prevalence")}
                                 for _, x in g.iterrows()} for e, g in bn.groupby("target")}
    for tag in ("dev", "final"):
        c = pd.read_csv(o / f"calibration_reg_{tag}.csv")
        out[f"calibration_slope_{tag}"] = {tg: {x.model: dict(slope=float(x.slope), corr=float(x["corr"]))
                                                for _, x in g.iterrows()} for tg, g in c.groupby("target")}
    for k in ("importance_top", "importance_groups", "shap_top"):
        out[k] = n[k]
    lee = json.loads((o / "numbers_lee.json").read_text()) if (o / "numbers_lee.json").exists() else None
    if lee:
        out["lee"] = lee
    (ROOT / "output/v7/p3" / f"numbers_p3_{a.ti}.json").write_text(json.dumps(out, indent=1, default=float))
    if a.ti == "sample" and not (ROOT / "output/v7/ti/post_probs_full_marker").exists():
        pass
    print("written", a.ti)


if __name__ == "__main__":
    main()
