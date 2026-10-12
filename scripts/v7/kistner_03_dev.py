"""Dev stage: rolling origins T=2015..2018 (train on targets 2011..T-1). Selects hyperparameters, writes frozen.json.
Never reads test targets for scoring (targets of 2019+ are excluded from every frame here)."""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE.parents[1] / "xbox"))
import numpy as np, pandas as pd
import kistner_ext as K
OUT = Path.home() / "x_box/pipeline/output/v7/kistner"
DEV_ORIGINS = [2015, 2016, 2017, 2018]
frozen = {}; devres = {}
for name, modes in [("tw", ("Twitter",)), ("twfb", ("Twitter", "Facebook"))]:
    t = pd.read_csv(f"{K.DATA}/transitions_{name}.csv")
    t = t[t.Y <= 2018].reset_index(drop=True)  # test years (2019+) removed before ANY modelling
    frozen[name] = {}; devres[name] = {}
    # pooled dev predictions for each model x hp
    preds = {}
    for mname, (kind, setn) in K.MODELS.items():
        cols = K.SETS[setn]
        for hp in K.grid(kind):
            pr = []
            for T in DEV_ORIGINS:
                tr = t[t.Y < T]; te = t[t.Y == T]
                pr.append(pd.Series(K.fit_predict(kind, cols, tr, te, hp), index=te.index))
            preds[(mname, hp)] = pd.concat(pr)
    dev = t[t.Y.isin(DEV_ORIGINS)]
    r0 = K.rmse(dev.target, np.zeros(len(dev)))
    devres[name]["rmse_persist"] = r0
    rtm = pd.concat([pd.Series(K.rtm_predict(t[t.Y < T], t[t.Y == T]), index=t[t.Y == T].index) for T in DEV_ORIGINS])
    devres[name]["rtm"] = {"rmse": K.rmse(dev.target, rtm.loc[dev.index]), "skill": 1 - K.rmse(dev.target, rtm.loc[dev.index]) / r0}
    for mname, (kind, setn) in K.MODELS.items():
        sc = {hp: K.rmse(dev.target, preds[(mname, hp)].loc[dev.index]) for hp in K.grid(kind)}
        best = min(sc, key=sc.get)
        frozen[name][mname] = int(best)
        p = preds[(mname, best)].loc[dev.index]
        pt, lo, hi, _ = K.boot_skill(dev.target, p, np.zeros(len(dev)), dev.MemberICPSR, B=1000)
        devres[name][mname] = {"hp": int(best), "grid_rmse": {str(k): v for k, v in sc.items()}, "skill": pt, "lo": lo, "hi": hi}
        # per-origin skill
        devres[name][mname]["by_origin"] = {int(T): 1 - K.rmse(dev[dev.Y == T].target, p[dev.Y == T]) / K.rmse(dev[dev.Y == T].target, 0 * dev[dev.Y == T].target) for T in DEV_ORIGINS}
        if name == "tw":
            devres[name].setdefault("_p", {})[mname] = p
    # direction on dev origins 2015, 2016, 2017, 2018 (rolling forecasts)
    if name == "tw":
        d2 = dev.reset_index(drop=True)
        P = {m: devres[name]["_p"][m].loc[dev.index].values for m in ["R_ctrl", "R_power", "R_both", "G_both"]}
        dtab = K.direction_table(d2, P)
        dtab.to_csv(OUT / "direction_dev.csv", index=False)
        del devres[name]["_p"]
        # leave-one-transition-year-out diagnostic (NOT a forecast: trains on other years incl. later ones), years 2011..2018
        loyo = {m: np.zeros(len(t)) for m in ["R_ctrl", "R_power", "R_both"]}
        for Y in sorted(t.Y.unique()):
            tr = t[t.Y != Y]; te = t[t.Y == Y]
            for m in loyo:
                kind, setn = K.MODELS[m]
                loyo[m][te.index] = K.fit_predict(kind, K.SETS[setn], tr, te, frozen[name][m])
        K.direction_table(t, loyo).to_csv(OUT / "direction_loyo_diag.csv", index=False)
    print(name, json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk in ("hp", "skill", "lo", "hi")}) for k, v in devres[name].items()}, default=float))
json.dump(frozen, open(OUT / "frozen.json", "w"), indent=1)
json.dump(devres, open(OUT / "dev_results.json", "w"), indent=1, default=float)
