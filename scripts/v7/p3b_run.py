"""P3b: does knowing the political power situation of the forecast year make next-year provocation change predictable?

Same pipeline as P3 (xbox/forecast_p3.py: baselines, ridge/elastic net/LightGBM/logit, rolling origins, member-cluster
bootstrap), with known-in-advance power features added under three timing variants:
  i   honest      status on the forecast date held for the whole target year
  ii  postelec    status known at the mid-target-year update (adds elections; Georgia 2021 runoffs unresolved)
  iii oracle      realized average status over the target year (upper bound)
Stages: python p3b_run.py --panel py|cy            development origins only; writes frozen.json
        python p3b_run.py --panel py|cy --final    runs ONCE (lock file) the final origin with the frozen choices.
Predictive associations only. Seed 20261011.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
from xbox import forecast_p3 as fp  # noqa: E402
import p3b_power as pw  # noqa: E402

fp.SEED = 20261011
SEED = 20261011
ROOT = HERE.parents[1]
OUTR = ROOT / "output/v7/p3b"
POW = ["a_next", "m_next", "tri_next", "d_a", "d_m", "d_tri", "d_a_end", "d_m_end", "d_a_x_R", "d_m_x_R",
       "a_cur_avg", "m_cur_avg"]
VARIANTS = {"i": "honest", "ii": "postelec", "iii": "oracle"}
DEPLOY_BASE = ["persistence", "regression_to_mean", "member_history_mean", "party_mean_change", "small_linear"]
PW_REG = ["ridge_pw", "enet_pw", "gbt_pw", "ridge_core_pw", "gbt_core_pw", "ridge_pwonly", "ridge_pwlvl", "gbt_pwlvl"]
CTL_REG = ["ridge", "enet", "gbt", "ridge_core", "gbt_core"]
PW_BIN = ["logit_pw", "gbt_pw", "logit_core_pw", "gbt_core_pw", "logit_pwlvl"]
CTL_BIN = ["logit", "gbt", "logit_core", "gbt_core"]
ESC_LVL = {"esc_ti7": "ti7", "esc_vol": "log_rate"}
CFG = {"py": dict(dev=[2018, 2019, 2020], final=[2021], mode="py", panel="panel_member_py_full.csv", base="output/v7/p3"),
       "cy": dict(dev=[2019, 2020], final=[2021], mode="cy", panel="panel_member_cy_full.csv", base="output/v7/p3b")}


def _prep_fixed_power_scale(Xtr, Xte, feats):
    """Same as fp._prep, except the power columns (bounded in [-1, 1] by construction) use a fixed scale of 0.5 instead
    of the training SD. Training SDs of change flags are near zero in years with no power change, and dividing by them
    extrapolates catastrophically when a new kind of change appears in the test year (found in development)."""
    med = Xtr[feats].median().fillna(0.0)
    a = Xtr[feats].fillna(med)
    b = Xte[feats].fillna(med)
    mu, sd = a.mean(), a.std().replace(0, 1).fillna(1)
    for f in feats:
        if f in POW:
            sd[f] = 0.5
    return ((a - mu) / sd).values, ((b - mu) / sd).values


fp._prep = _prep_fixed_power_scale


def cy_seat_flags(su, bg, t):
    ev = t % 2 == 0
    nx = (t + 1) % 2 == 0
    return {"seat_up_general_cur": su.get((bg, t), np.nan) if ev else 0,
            "seat_up_primary_cur": su.get((bg, t), np.nan) if ev else 0,
            "seat_up_general_next": su.get((bg, t + 1), np.nan) if nx else 0,
            "seat_up_primary_next": su.get((bg, t + 1), np.nan) if nx else 0,
            "election_year_next": int(nx)}


def build_T(panel_key):
    cfg = CFG[panel_key]
    panel = pd.read_csv(ROOT / cfg["base"] / cfg["panel"])
    panel["seniority_congresses"] = panel.seniority_congresses.clip(lower=0)
    if panel_key == "cy":
        panel = panel[(panel.py >= 2018) & (panel.py <= 2022)].copy()
        fp.seat_flags = cy_seat_flags
    su = fp.seat_lookup(pd.read_csv(ROOT / "output/v7/t3/elections.csv"))
    T, att = fp.make_transitions(panel, su, min_scored=75)
    origins = sorted(T.origin.unique())
    P, side = pw.build(ROOT, cfg["mode"], origins)
    return T, att, P


def with_power(T, P, v):
    d = pw.variant_table(P, v)
    X = T.merge(d, on=["bioguide", "origin"], how="left")
    pr = X.party_R.fillna(0.0)
    X["d_a_x_R"] = X.d_a * pr
    X["d_m_x_R"] = X.d_m * pr
    return X


def specs(tg, ctl, power):
    binary = tg in fp.ESC
    lvl = ESC_LVL[tg] if binary else fp.TARGETS[tg][0]
    F, C = fp.FEATURES_FULL, fp.FEATURES_CORE
    s = []
    if binary:
        if ctl:
            s += [("logit", "logit", F), ("gbt", "gbt", F), ("logit_core", "logit", C), ("gbt_core", "gbt", C)]
        if power:
            s += [("logit_pw", "logit", F + POW), ("gbt_pw", "gbt", F + POW), ("logit_core_pw", "logit", C + POW),
                  ("gbt_core_pw", "gbt", C + POW), ("logit_pwlvl", "logit", POW + [lvl])]
    else:
        if ctl:
            s += [("ridge", "ridge", F), ("enet", "enet", F), ("gbt", "gbt", F), ("ridge_core", "ridge", C),
                  ("gbt_core", "gbt", C)]
        if power:
            s += [("ridge_pw", "ridge", F + POW), ("enet_pw", "enet", F + POW), ("gbt_pw", "gbt", F + POW),
                  ("ridge_core_pw", "ridge", C + POW), ("gbt_core_pw", "gbt", C + POW),
                  ("ridge_pwonly", "ridge", POW), ("ridge_pwlvl", "ridge", POW + [lvl]),
                  ("gbt_pwlvl", "gbt", POW + [lvl])]
    return s


def roll(T, origins, ctl, power, targets=None):
    targets = targets or list(fp.TARGETS) + list(fp.ESC)
    recs = []
    for t in origins:
        Ttr = T[T.origin <= t - 1].reset_index(drop=True)
        Tte = T[T.origin == t].reset_index(drop=True)
        if len(Ttr) == 0 or len(Tte) == 0:
            continue
        for tg in targets:
            binary = tg in fp.ESC
            if not binary:
                out = fp.baselines_reg(Ttr, Tte, tg, fp.TARGETS[tg][0]) if ctl else {}
            else:
                out = {}
                if ctl:
                    out = {"base_rate": np.full(len(Tte), float(Ttr[tg].mean())),
                           "persistence_score": Tte[{"esc_ti7": "lag_d_ti7", "esc_vol": "lag_d_log_rate"}[tg]].values,
                           "rtm_score": -Tte[ESC_LVL[tg]].values}
            for tag, kind, feats in specs(tg, ctl, power):
                p, hp, _ = fp.fit_predict(kind, Ttr, Tte, feats, tg, binary=binary)
                out[tag] = p
            for m, p in out.items():
                recs.append(pd.DataFrame(dict(bioguide=Tte.bioguide, origin=t, target=tg, model=m, pred=p,
                                              y=Tte[tg].values, party=Tte.party, chamber=Tte.chamber)))
    return pd.concat(recs, ignore_index=True)


def member_parts(df, target, model):
    d = df[(df.target == target) & (df.model == model)]
    g = d.assign(se=(d.y - d.pred) ** 2).groupby("bioguide").agg(se=("se", "sum"), n=("se", "size"))
    return g


def recov(df, target, model, base="persistence", oracle="party_year_ORACLE", B=2000):
    a, b, o = (member_parts(df, target, m) for m in (model, base, oracle))
    j = a.join(b, lsuffix="_m", rsuffix="_b").join(o.rename(columns={"se": "se_o", "n": "n_o"}), how="inner")
    n = j.n_m.values.astype(float)
    sm, sb, so = j.se_m.values, j.se_b.values, j.se_o.values

    def f(w):
        den = (w * n).sum()
        rb = np.sqrt((w * sb).sum() / den)
        sk_m = 1 - np.sqrt((w * sm).sum() / den) / rb
        sk_o = 1 - np.sqrt((w * so).sum() / den) / rb
        return sk_m, sk_o

    sk_m, sk_o = f(np.ones(len(j)))
    rng = np.random.default_rng(SEED)
    W = rng.multinomial(len(j), np.ones(len(j)) / len(j), size=B).astype(float)
    den = W @ n
    rb = np.sqrt(W @ sb / den)
    skm = 1 - np.sqrt(W @ sm / den) / rb
    sko = 1 - np.sqrt(W @ so / den) / rb
    r = skm / sko
    return dict(skill_model=float(sk_m), skill_oracle=float(sk_o), recovered=float(sk_m / sk_o) if sk_o else np.nan,
                rec_lo=float(np.percentile(r, 2.5)), rec_hi=float(np.percentile(r, 97.5)))


def score(preds, variant, tag, best):
    rows, brow, rrow = [], [], []
    models = set(preds.model)
    for tg in fp.TARGETS:
        d = preds[preds.target == tg]
        for m in sorted(models):
            if ((d.model == m).sum() == 0) or m == "persistence":
                continue
            for vs_name, b in (("persistence", "persistence"), ("best_baseline", best[tg])):
                r = fp.skill_ci(d, tg, m, b, B=2000, seed=SEED)
                rows.append(dict(variant=variant, set=tag, target=tg, model=m, versus=vs_name, base_model=b, **r))
            if m not in DEPLOY_BASE and m != "party_year_ORACLE":
                rrow.append(dict(variant=variant, set=tag, target=tg, model=m, **recov(d, tg, m)))
    for e in fp.ESC:
        d = preds[preds.target == e]
        for m in sorted(set(d.model)):
            dm = d[d.model == m]
            a = fp.auc_ci(d, e, m, B=500, seed=SEED)
            rel, ece = fp.reliability(dm.y.values, dm.pred.values)
            brier = np.nan if m in ("persistence_score", "rtm_score") else float(np.mean((dm.y - dm.pred) ** 2))
            rr = {}
            if m in CTL_BIN + PW_BIN:
                r = fp.skill_ci(d, e, m, "base_rate", metric="brier", B=2000, seed=SEED)
                rr = dict(brier_skill=r["skill"], brier_lo=r["lo"], brier_hi=r["hi"])
            brow.append(dict(variant=variant, set=tag, target=e, model=m, auc=a["auc"], auc_lo=a["lo"], auc_hi=a["hi"],
                             brier=brier, ece=ece, n=len(dm), **rr))
    return pd.DataFrame(rows), pd.DataFrame(brow), pd.DataFrame(rrow)


def direction(preds, T, tag):
    """Mean predicted vs realized change by party x chamber, for the TI7 target."""
    d = preds[(preds.target == "d_ti7")]
    g = d.groupby(["origin", "party", "chamber", "model"]).agg(pred=("pred", "mean"), y=("y", "mean"),
                                                                   n=("y", "size")).reset_index()
    g.insert(0, "set", tag)
    return g


def placebo_and_leak(T, P, origins, outdir):
    """Leakage perturbation tests extended to the new features (development origins only, d_ti7 and d_tribal)."""
    res = {}
    rng = np.random.default_rng(SEED)
    # structural checks
    for v in VARIANTS:
        X = with_power(T, P, v)
        if v == "i":
            res["i_change_from_end_is_zero"] = bool(np.nanmax(np.abs(X[["d_a_end", "d_m_end"]].values)) < 1e-12)
    # within party x chamber x origin: correlation of demeaned power features with demeaned target
    for v in VARIANTS:
        X = with_power(T, P, v)
        out = {}
        for f in ("d_a", "d_m", "m_next", "a_next"):
            g = X.groupby(["origin", "party", "chamber"])
            fr = X[f] - g[f].transform("mean")
            yr = X["d_ti7"] - g["d_ti7"].transform("mean")
            out[f] = dict(within_sd=float(fr.std()), corr=float(np.corrcoef(fr.fillna(0), yr)[0, 1]) if fr.std() > 1e-9 else None)
        res[f"within_group_{v}"] = out
    # placebo target and shuffled power features, variant ii and iii
    for v in ("ii", "iii"):
        X = with_power(T, P, v)
        # (a) placebo: shuffle d_ti7 across members within origin (train and test) -> no real signal exists
        Xp = X.copy()
        for t, g in Xp.groupby("origin"):
            Xp.loc[g.index, "d_ti7"] = rng.permutation(g.d_ti7.values)
        pr = roll(Xp, origins, ctl=True, power=True, targets=["d_ti7"])
        # (b) shuffle the power columns across members within origin (breaks the link to party/chamber)
        Xs = X.copy()
        for t, g in Xs.groupby("origin"):
            perm = rng.permutation(len(g))
            for c in POW:
                Xs.loc[g.index, c] = g[c].values[perm]
        ps = roll(Xs, origins, ctl=False, power=True, targets=["d_ti7"])
        pa = roll(X, origins, ctl=False, power=True, targets=["d_ti7"])
        for nm, pp in (("placebo_target", pr), ("shuffled_power", ps), ("real_power", pa)):
            out = {}
            for m in ("ridge_pw", "gbt_pw", "ridge_pwonly", "ridge_pwlvl"):
                dm = pp[(pp.model == m) & (pp.target == "d_ti7")]
                dd = pd.concat([dm, dm.assign(model="persistence", pred=0.0)], ignore_index=True)
                r = fp.skill_ci(dd, "d_ti7", m, "persistence", B=1000, seed=SEED)
                out[m] = dict(skill_vs_persistence=r["skill"], lo=r["lo"], hi=r["hi"])
            res[f"{nm}_{v}"] = out
    (outdir / "leak_tests.json").write_text(json.dumps(res, indent=1, default=float))
    return res


def run(panel_key, final):
    cfg = CFG[panel_key]
    out = OUTR / panel_key
    out.mkdir(parents=True, exist_ok=True)
    T, att, P = build_T(panel_key)
    T.to_csv(out / "transitions_base.csv", index=False)
    att.to_csv(out / "attrition.csv", index=False)
    P.to_csv(out / "power_features_raw.csv", index=False)
    print(panel_key, "transitions", T.groupby("origin").size().to_dict(), flush=True)
    origins = cfg["final"] if final else cfg["dev"]
    lock = out / "FINAL_RAN.lock"
    frozen_path = out / "frozen.json"
    if final:
        if lock.exists():
            sys.exit("final test already run once; refusing")
        frozen = json.loads(frozen_path.read_text())
    tag = "final" if final else "dev"
    ctl_done = None
    all_sk, all_bn, all_rc, all_dir = [], [], [], []
    for v in VARIANTS:
        X = with_power(T, P, v)
        pr = roll(X, origins, ctl=(v == "i"), power=True)
        pr.insert(0, "variant", v)
        if v == "i":
            ctl_done = pr[~pr.model.isin(PW_REG + PW_BIN)].copy()
        pr = pd.concat([ctl_done.assign(variant=v), pr[pr.model.isin(PW_REG + PW_BIN)]], ignore_index=True) \
            if v != "i" else pr
        pr.to_csv(out / f"preds_{tag}_{v}.csv", index=False)
        if final:
            dev_pr = pd.read_csv(out / f"preds_dev_{v}.csv")
            best = frozen["best_baseline"]
            both = pd.concat([dev_pr, pr], ignore_index=True)
            sets = [("final", pr), ("all", both)]
        else:
            best = {}
            for tg in fp.TARGETS:
                dd = pr[(pr.target == tg) & pr.model.isin(DEPLOY_BASE)]
                best[tg] = dd.assign(se=(dd.y - dd.pred) ** 2).groupby("model").se.mean().idxmin()
            sets = [("dev", pr)]
        for st, pp in sets:
            sk, bn, rc = score(pp, v, st, best)
            all_sk.append(sk)
            all_bn.append(bn)
            all_rc.append(rc)
        all_dir.append(direction(pr, T, tag).assign(variant=v))
        print("variant", v, "done", flush=True)
    sk, bn, rc = pd.concat(all_sk), pd.concat(all_bn), pd.concat(all_rc)
    sk.to_csv(out / f"skill_reg_{tag}.csv", index=False)
    bn.to_csv(out / f"skill_bin_{tag}.csv", index=False)
    rc.to_csv(out / f"recovery_{tag}.csv", index=False)
    pd.concat(all_dir).to_csv(out / f"direction_{tag}.csv", index=False)
    if not final:
        # freeze: per variant and target, the power model with the lowest pooled dev RMSE; baseline per target
        frozen = {"best_baseline": best, "chosen": {}}
        for v in VARIANTS:
            for tg in fp.TARGETS:
                d = sk[(sk.variant == v) & (sk.set == "dev") & (sk.target == tg) & (sk.versus == "persistence")
                       & sk.model.isin(PW_REG)]
                frozen["chosen"][f"{v}:{tg}"] = d.sort_values("skill", ascending=False).model.iloc[0]
        frozen_path.write_text(json.dumps(frozen, indent=1))
        placebo_and_leak(T, P, origins, out)
    else:
        lock.write_text("final test executed once\n")
    print("stage", tag, "complete", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", choices=["py", "cy"], required=True)
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    run(a.panel, a.final)
