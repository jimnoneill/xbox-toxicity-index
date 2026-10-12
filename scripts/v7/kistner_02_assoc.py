"""(a) Association: negshare on out-of-power status, member FE, optional party-specific trends / year FE."""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE.parents[1] / "xbox"))
sys.path.insert(0, str(Path.home() / "x_box/pylib_v7"))
import numpy as np, pandas as pd
import statsmodels.api as sm
import kistner_ext as K
OUT = Path.home() / "x_box/pipeline/output/v7/kistner"
res = {}
for name, modes in [("tw", ("Twitter",)), ("twfb", ("Twitter", "Facebook"))]:
    p = K.build_panel(modes); p = p[(p.n >= K.MIN_N) & (p.Year >= 2010)].copy()
    p["chamber"] = p.chamber.fillna("House")
    pw = []
    for Y, g in p.groupby("Year"):
        f = K.power_features(Y, g.party.values, g.chamber.values); f.index = g.index; pw.append(f)
    p = pd.concat([p, pd.concat(pw)], axis=1)
    p["out_pres"] = 1 - p.p_pres; p["out_ch"] = 1 - p.p_ch; p["out_oth"] = 1 - p.p_oth; p["out_n"] = 3 - p.p_tri
    p["isR"] = (p.party == "R").astype(float); p["t"] = p.Year - 2010
    p["tR"] = p.t * p.isR; p["tD"] = p.t * (1 - p.isR)
    y = p.share
    sd_within = float((y - y.groupby(p.MemberICPSR).transform("mean")).std())
    res[name] = {"n_obs": int(len(p)), "n_members": int(p.MemberICPSR.nunique()), "years": [2010, 2022],
                 "sd_share": float(y.std()), "sd_within_member": sd_within, "mean_share": float(y.mean())}
    yfe = pd.get_dummies(p.Year.astype(int), prefix="y", drop_first=True).astype(float)
    specs = {
        "S1_FE": (["out_pres", "out_ch", "out_oth"], []),
        "S2_FE_partytrend": (["out_pres", "out_ch", "out_oth"], ["tR", "tD"]),
        "S3_FE_yearFE": (["out_pres", "out_ch", "out_oth"], "yfe"),
        "S4_FE_yearFE_partytrend": (["out_pres", "out_ch", "out_oth"], "yfe+tr"),
        "N1_FE_n_out": (["out_n"], []),
        "N2_FE_n_out_partytrend": (["out_n"], ["tR", "tD"]),
        "P1_FE_pres_only": (["out_pres"], []),
        "P2_FE_pres_only_partytrend": (["out_pres"], ["tR", "tD"]),
    }
    for sn, (xs, extra) in specs.items():
        X = p[xs].copy()
        if isinstance(extra, str) and extra == "yfe":
            X = pd.concat([X, yfe], axis=1)
        elif isinstance(extra, str) and extra == "yfe+tr":
            X = pd.concat([X, yfe, p[["tR", "tD"]]], axis=1)
        elif extra:
            X = pd.concat([X, p[extra]], axis=1)
        Xd = X - X.groupby(p.MemberICPSR).transform("mean"); yd = y - y.groupby(p.MemberICPSR).transform("mean")
        keep = [c for c in Xd.columns if Xd[c].std() > 1e-9]
        # sequentially drop perfectly collinear columns
        sel = []
        for c in keep:
            trial = sel + [c]
            if np.linalg.matrix_rank(Xd[trial].values) == len(trial):
                sel.append(c)
        keep = sel
        m = sm.OLS(yd.values, Xd[keep].values).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(p.MemberICPSR)[0]})
        out = {}
        for c in xs:
            if c in keep:
                i = keep.index(c); b, se = float(m.params[i]), float(m.bse[i])
                out[c] = {"b": b, "se": se, "lo": b - 1.96 * se, "hi": b + 1.96 * se, "b_in_within_sd": b / sd_within}
            else:
                out[c] = "dropped_collinear"
        res[name][sn] = out
    res[name]["party_year_mean_share"] = {str(k): v for k, v in p.groupby(["Year", "party"]).share.mean().unstack().round(4).to_dict("index").items()}
    print(name, json.dumps({k: v for k, v in res[name].items() if k != "party_year_mean_share"}, indent=1))
json.dump(res, open(OUT / "assoc.json", "w"), indent=1)
