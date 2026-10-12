"""FINAL TEST, runs exactly once (lock file). Train: targets 2011..2018. Test: targets 2019..2022. Frozen hyperparameters."""
import sys, json, os
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE.parents[1] / "xbox"))
import numpy as np, pandas as pd
import kistner_ext as K
OUT = Path.home() / "x_box/pipeline/output/v7/kistner"
LOCK = OUT / "FINAL_RAN.lock"
if LOCK.exists():
    sys.exit("final test already ran; refusing to run again")
LOCK.write_text("started\n")
frozen = json.load(open(OUT / "frozen.json"))
NPERM = 500
res = {}
for name in ["tw", "twfb"]:
    t = pd.read_csv(f"{K.DATA}/transitions_{name}.csv")
    tr = t[t.Y <= 2018].reset_index(drop=True); te = t[t.Y >= 2019].reset_index(drop=True)
    y = te.target.values; z = np.zeros(len(te)); mem = te.MemberICPSR.values
    r = {"n_train": int(len(tr)), "n_test": int(len(te)), "n_test_members": int(te.MemberICPSR.nunique()),
         "test_per_year": te.groupby("Y").size().to_dict()}
    r["rmse_persist"] = K.rmse(y, z)
    orc = K.oracle_pred(te)
    r["rmse_oracle_partychamberyear"] = K.rmse(y, orc)
    P = {}
    rt = K.rtm_predict(tr, te); P["RTM"] = rt
    for m, (kind, setn) in K.MODELS.items():
        P[m] = K.fit_predict(kind, K.SETS[setn], tr, te, frozen[name][m])
    for m, p in P.items():
        pt, lo, hi, _ = K.boot_skill(y, p, z, mem, B=1000)
        d = {"rmse": K.rmse(y, p), "skill_vs_persist": pt, "lo": lo, "hi": hi,
             "oracle_gap_share": (r["rmse_persist"] - K.rmse(y, p)) / (r["rmse_persist"] - r["rmse_oracle_partychamberyear"])}
        if m != "RTM":
            pt2, lo2, hi2, _ = K.boot_skill(y, p, rt, mem, B=1000)
            d.update({"skill_vs_rtm": pt2, "lo_rtm": lo2, "hi_rtm": hi2})
        # subsets
        for lab, msk in [("transition_years_2019_2021", te.Y.isin([2019, 2021]).values), ("nontransition_2020_2022", te.Y.isin([2020, 2022]).values)]:
            a, b, c, _ = K.boot_skill(y[msk], p[msk], z[msk], mem[msk], B=1000)
            d[lab] = {"skill": a, "lo": b, "hi": c}
        d["by_year"] = {int(Y): 1 - K.rmse(y[(te.Y == Y).values], p[(te.Y == Y).values]) / K.rmse(y[(te.Y == Y).values], 0.0 * y[(te.Y == Y).values]) for Y in sorted(te.Y.unique())}
        r[m] = d
    # contrast: R_both minus R_ctrl skill (paired bootstrap on member clusters)
    pt, lo, hi, _ = K.boot_skill(y, P["R_both"], P["R_ctrl"], mem, B=1000)
    r["R_both_vs_R_ctrl_rmse_ratio_skill"] = {"skill": pt, "lo": lo, "hi": hi}
    # direction tables
    dt = K.direction_table(te, {k: P[k] for k in ["R_ctrl", "R_power", "R_both", "G_both"]})
    dt.to_csv(OUT / f"direction_final_{name}.csv", index=False)
    # predictions for figures
    pd.concat([te[["MemberICPSR", "Y", "party", "chamber", "target"]], pd.DataFrame(P)], axis=1).to_csv(OUT / f"preds_final_{name}.csv", index=False)
    # placebo: permute year -> power configuration mapping, refit with frozen alphas, score on test
    rng = np.random.default_rng(K.SEED)
    years = list(range(2011, 2023))
    null = {"R_power": [], "R_both": []}
    for i in range(NPERM):
        perm = rng.permutation(years); mp = {Y: int(p_) for Y, p_ in zip(years, perm)}
        tt = K.with_power(t, mp)
        a = tt[tt.Y <= 2018].reset_index(drop=True); b = tt[tt.Y >= 2019].reset_index(drop=True)
        for m in null:
            kind, setn = K.MODELS[m]
            pp = K.fit_predict(kind, K.SETS[setn], a, b, frozen[name][m])
            null[m].append(1 - K.rmse(b.target.values, pp) / K.rmse(b.target.values, 0 * pp))
    for m in null:
        arr = np.array(null[m]); obs = r[m]["skill_vs_persist"]
        r[f"placebo_{m}"] = {"mean": float(arr.mean()), "p2.5": float(np.percentile(arr, 2.5)), "p97.5": float(np.percentile(arr, 97.5)),
                             "p_one_sided_ge_observed": float((1 + (arr >= obs).sum()) / (NPERM + 1)), "n_perm": NPERM}
    res[name] = r
    print(name, json.dumps({k: (v["skill_vs_persist"], v["lo"], v["hi"]) if isinstance(v, dict) and "skill_vs_persist" in v else None for k, v in r.items() if isinstance(v, dict) and "skill_vs_persist" in v}))
json.dump(res, open(OUT / "numbers_kistner_final.json", "w"), indent=1, default=float)
LOCK.write_text("done\n")
