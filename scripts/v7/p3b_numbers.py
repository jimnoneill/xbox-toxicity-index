"""P3b numbers: reads the saved prediction tables (no refitting of the forecasting models) and writes
output/v7/p3b/numbers_p3b.json. Post-hoc items are labelled as such."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
from xbox import forecast_p3 as fp  # noqa: E402

ROOT = HERE.parents[1]
OUT = ROOT / "output/v7/p3b"
SEED = 20261011
V = ["i", "ii", "iii"]
TG = list(fp.TARGETS)
DEPLOY = ["persistence", "regression_to_mean", "member_history_mean", "party_mean_change", "small_linear"]
PAIRS = [("ridge_pw", "ridge"), ("gbt_pw", "gbt"), ("ridge_core_pw", "ridge_core"), ("gbt_core_pw", "gbt_core")]


def sk(df, tg, m, b, B=2000):
    r = fp.skill_ci(df, tg, m, b, B=B, seed=SEED)
    return dict(skill=round(r["skill"], 4), lo=round(r["lo"], 4), hi=round(r["hi"], 4))


def best_on_rows(df, tg):
    d = df[(df.target == tg) & df.model.isin(DEPLOY)]
    return d.assign(se=(d.y - d.pred) ** 2).groupby("model").se.mean().idxmin()


def one_panel(pk):
    d = OUT / pk
    frozen = json.loads((d / "frozen.json").read_text())
    res = dict(frozen=frozen)
    preds = {}
    for v in V:
        dev = pd.read_csv(d / f"preds_dev_{v}.csv")
        fin = pd.read_csv(d / f"preds_final_{v}.csv")
        preds[v] = dict(dev=dev, final=fin)
    res["skill"] = {}
    res["increment_over_matching_control"] = {}
    for v in V:
        for st in ("dev", "final"):
            df = preds[v][st]
            for tg in TG:
                ch = frozen["chosen"][f"{v}:{tg}"]
                be = best_on_rows(df, tg)
                key = f"{v}|{st}|{tg}"
                row = dict(chosen_on_dev=ch, best_deployable_baseline_on_these_rows=be,
                           chosen_vs_persistence=sk(df, tg, ch, "persistence"),
                           chosen_vs_dev_best_baseline=sk(df, tg, ch, frozen["best_baseline"][tg]),
                           chosen_vs_best_baseline_on_rows=sk(df, tg, ch, be),
                           oracle_vs_persistence=sk(df, tg, "party_year_ORACLE", "persistence"),
                           control_gbt_vs_persistence=sk(df, tg, "gbt", "persistence"),
                           control_ridge_vs_persistence=sk(df, tg, "ridge", "persistence"),
                           power_only_ridge_vs_persistence=sk(df, tg, "ridge_pwonly", "persistence"),
                           power_plus_level_ridge_vs_persistence=sk(df, tg, "ridge_pwlvl", "persistence"),
                           power_plus_level_gbt_vs_persistence=sk(df, tg, "gbt_pwlvl", "persistence"),
                           regression_to_mean_vs_persistence=sk(df, tg, "regression_to_mean", "persistence"))
                res["skill"][key] = row
                if tg in ("d_ti7", "d_logvol", "d_tribal"):
                    res["increment_over_matching_control"][key] = {
                        f"{a}_over_{b}": sk(df, tg, a, b, B=1000) for a, b in PAIRS}
    # recovery table from the saved csvs
    rc = pd.concat([pd.read_csv(d / "recovery_dev.csv"), pd.read_csv(d / "recovery_final.csv")])
    rc = rc[rc.model.str.contains("pw")]
    res["recovered_fraction_of_oracle_gap"] = {
        f"{r.variant}|{r.set}|{r.target}|{r.model}": [round(r.recovered, 3), round(r.rec_lo, 3), round(r.rec_hi, 3)]
        for r in rc.itertuples() if r.set in ("dev", "final")}
    # escalation AUC
    bn = pd.concat([pd.read_csv(d / "skill_bin_dev.csv"), pd.read_csv(d / "skill_bin_final.csv")])
    bn = bn[bn.set.isin(["dev", "final"])]
    res["auc"] = {f"{r.variant}|{r.set}|{r.target}|{r.model}": [round(r.auc, 3), round(r.auc_lo, 3), round(r.auc_hi, 3),
                                                              round(r.ece, 3)] for r in bn.itertuples()}
    # direction of the final-year forecast by party (TI7)
    dr = pd.read_csv(d / "direction_final.csv")
    dr = dr[dr.model.isin(["ridge", "gbt", "ridge_pw", "gbt_pw", "ridge_pwonly", "ridge_pwlvl", "gbt_pwlvl",
                           "party_year_ORACLE", "persistence"])]
    res["direction_final_ti7"] = {}
    for v in V:
        t = dr[(dr.variant == v) & dr.party.isin(["R", "D"])].groupby(["party", "model"]).apply(
            lambda g: pd.Series(dict(pred=np.average(g.pred, weights=g.n), actual=np.average(g.y, weights=g.n))))
        res["direction_final_ti7"][v] = {f"{p}|{m}": [round(x.pred, 4), round(x.actual, 4)] for (p, m), x in t.iterrows()}
        t2 = dr[(dr.variant == v) & dr.party.isin(["R", "D"]) & dr.model.isin(["ridge_pwonly", "ridge_pw", "gbt_pwlvl", "gbt"])]
        res.setdefault("direction_final_ti7_by_chamber", {})[v] = {
            f"{r.party}|{r.chamber}|{r.model}": [round(r.pred, 4), round(r.y, 4), int(r.n)] for r in t2.itertuples()}
    # power change by party and chamber per origin, and realized TI7 change
    Tb = pd.read_csv(d / "transitions_base.csv")
    P = pd.read_csv(d / "power_features_raw.csv")
    import p3b_power as pw
    ch = {}
    for v in V:
        X = Tb.merge(pw.variant_table(P, v), on=["bioguide", "origin"])
        g = X[X.party.isin(["R", "D"])].groupby(["origin", "party", "chamber"]).agg(
            d_a=("d_a", "mean"), d_m=("d_m", "mean"), d_a_end=("d_a_end", "mean"), d_m_end=("d_m_end", "mean"),
            d_ti7=("d_ti7", "mean"), n=("d_ti7", "size")).round(3).reset_index()
        ch[v] = g.to_dict("records")
    res["power_change_by_origin_party_chamber"] = ch
    # learned mechanism (post hoc, training transitions only: origins up to 2020), pooled OLS with origin dummies
    mech = {}
    for v in ("ii", "iii", "i"):
        X = Tb.merge(pw.variant_table(P, v), on=["bioguide", "origin"])
        X = X[X.origin <= 2020].copy()
        for t in sorted(X.origin.unique())[1:]:
            X[f"o{t}"] = (X.origin == t).astype(float)
        cols = ["d_a", "d_m"] + [c for c in X.columns if c.startswith("o") and c[1:].isdigit()]
        for tg in ("d_ti7", "d_tribal", "d_performative", "d_ragebait"):
            A = np.column_stack([np.ones(len(X))] + [X[c].values for c in cols])
            y = X[tg].values
            beta = np.linalg.lstsq(A, y, rcond=None)[0]
            bs = []
            rng = np.random.default_rng(SEED)
            ids = X.bioguide.values
            uniq = np.unique(ids)
            idx = {u: np.where(ids == u)[0] for u in uniq}
            for _ in range(300):
                pick = rng.choice(uniq, len(uniq))
                ii = np.concatenate([idx[u] for u in pick])
                bs.append(np.linalg.lstsq(A[ii], y[ii], rcond=None)[0][1:3])
            bs = np.array(bs)
            mech[f"{v}|{tg}"] = dict(b_d_a=[round(beta[1], 4), round(np.percentile(bs[:, 0], 2.5), 4), round(np.percentile(bs[:, 0], 97.5), 4)],
                                     b_d_m=[round(beta[2], 4), round(np.percentile(bs[:, 1], 2.5), 4), round(np.percentile(bs[:, 1], 97.5), 4)])
    res["mechanism_ols_training_origins_le_2020_posthoc"] = mech
    res["leak_tests"] = json.loads((d / "leak_tests.json").read_text())
    res["transitions_per_origin"] = {int(k): int(v) for k, v in Tb.groupby("origin").size().items()}
    return res


if __name__ == "__main__":
    out = {}
    for pk in ("py", "cy"):
        out[pk] = one_panel(pk)
    (OUT / "numbers_p3b.json").write_text(json.dumps(out, indent=1, default=float))
    print("written")
