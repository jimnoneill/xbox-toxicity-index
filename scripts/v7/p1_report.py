"""P1 step 3: scores, skill with account-cluster bootstrap CIs, horizon, heterogeneity, calibration, figures.

Reads  <out>/preds_<stage>/*.parquet (+ features.parquet for account attributes)
Writes <out>/metrics_<stage>.json, skill_<stage>.csv, horizon_<stage>.csv, hetero_<stage>.csv, origin_skill_<stage>.csv,
       calibration_<stage>.csv, fig1_skill_<stage>.png, fig2_calibration_<stage>.png, fig4_heterogeneity_<stage>.png
Skill = 1 - score_model / score_baseline (MSE, Brier, log loss); AUC reported as a difference.  Bootstrap: 2000 resamples
of accounts (AUC: 300).  Predictive associations only.  Seed 20261010.
Usage: python p1_report.py --source subset|full --stage dev|test
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p1_common import *  # noqa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

a = get_args(lambda ap: ap.add_argument("--stage", choices=["dev", "test"], required=True))
OUT = outdir(a.source)
ST = a.stage
PAL = dict(base="#7f7f7f", P="#0072B2", R="#D55E00", F="#009E73", lmm="#CC79A7")
rng = np.random.default_rng(fp.SEED)

P = pd.concat([pd.read_parquet(f) for f in sorted((OUT / f"preds_{ST}").glob("origin_*.parquet"))], ignore_index=True)
F = pd.read_parquet(OUT / "features.parquet", columns=["acct", "party", "chamber", "account_type", "flag_strong",
                                                       "flag_minute", "flag_tool"]).drop_duplicates("acct").set_index("acct")
for c in ["party", "chamber", "account_type"]:
    P[c] = P.acct.map(F[c])
P["sched"] = P.acct.map(F.flag_strong).fillna(False).astype(bool)
acc_codes, ainv = np.unique(P.acct.values, return_inverse=True)
J = len(acc_codes)
y = P.ti7.to_numpy("float64")
print(ST, "test rows", len(P), "accounts", J, flush=True)

TYP = ["PT", "RT", "PTO", "RTO"]
REG = ["b_global", "b_mean", "b_persist", "b_roll", "lmm_reg_P", "lmm_reg_R", "gbm_reg_P", "gbm_reg_R", "gbm_reg_F"] + [f"{m}_reg_{s}" for m in ("lmm", "gbm") for s in TYP]
INC = [(f"{m}_reg_{a_}", f"{m}_reg_{b_}") for m in ("gbm", "lmm") for a_, b_ in [("R", "P"), ("RT", "PT"), ("RTO", "PTO"), ("PT", "P"), ("PTO", "PT")]] + [("gbm_reg_F", "gbm_reg_R")]
BASE_REG = ["b_global", "b_mean", "b_persist", "b_roll"]


def acct_sum(v, idx=ainv, n=J):
    return np.bincount(idx, weights=v, minlength=n)


def reg_loss(c):
    return (y - P[c].to_numpy("float64")) ** 2


def bin_loss(c, task, kind):
    yy = P[f"y_{task}"].to_numpy("float64")
    p = np.clip(P[c].to_numpy("float64"), 1e-4, 1 - 1e-4)
    return (yy - p) ** 2 if kind == "brier" else -(yy * np.log(p) + (1 - yy) * np.log(1 - p))


res = dict(stage=ST, source=a.source, n_rows=int(len(P)), n_accounts=int(J),
           test_start=str(P.when.min()), test_end=str(P.when.max()), prevalence_high_acct=float(P.y_bina.mean()),
           prevalence_high_global=float(P.y_binb.mean()), ti7_mean=float(y.mean()), ti7_sd=float(y.std()))
rows = []


def add(task, score, model, base, la, lb, note=""):
    pt, lo, hi = fp.skill_boot(acct_sum(la), acct_sum(lb), B=2000)
    rows.append(dict(task=task, score=score, model=model, baseline=base, skill=pt, lo=lo, hi=hi,
                     score_model=float(la.sum() / len(la)), score_base=float(lb.sum() / len(lb)), note=note))
    return pt


# ---- regression
L = {c: reg_loss(c) for c in REG}
mse = {c: float(L[c].mean()) for c in REG}
best = min(BASE_REG, key=lambda c: mse[c])
res["reg_mse"] = mse
res["reg_mae"] = {c: float(np.abs(y - P[c]).mean()) for c in REG}
res["reg_best_baseline"] = best
for m in REG:
    if m != best:
        add("reg", "mse", m, best, L[m], L[best])
for m_r, m_p in INC:
    add("reg", "mse", m_r, m_p, L[m_r], L[m_p], note="incremental")
add("reg", "mse", "gbm_reg_R", "lmm_reg_P", L["gbm_reg_R"], L["lmm_reg_P"], note="rhythm GBM vs persistence mixed model")
res["reg_r2_vs_global_mean"] = {c: float(1 - mse[c] / mse["b_global"]) for c in REG}
# ---- binary
for task in ("bina", "binb"):
    yy = P[f"y_{task}"].to_numpy()
    mods = [c for c in [f"b_rate_{task}", f"b_persist_{task}", f"lmm_{task}_P", f"lmm_{task}_R"] + [f"gbm_{task}_{s}" for s in ["P", "R"] + TYP] if c in P.columns]
    for kind in ("brier", "logloss"):
        LB = {c: bin_loss(c, task, kind) for c in mods}
        bb = min(mods[:2], key=lambda c: LB[c].mean())
        res[f"{task}_{kind}"] = {c: float(LB[c].mean()) for c in mods}
        res[f"{task}_{kind}_best_baseline"] = bb
        for m in mods:
            if m != bb:
                add(task, kind, m, bb, LB[m], LB[bb])
        for m_r, m_p in [(f"gbm_{task}_{x}", f"gbm_{task}_{z}") for x, z in [("R", "P"), ("RT", "PT"), ("RTO", "PTO"), ("PT", "P"), ("PTO", "PT")]] + [(f"lmm_{task}_R", f"lmm_{task}_P")]:
            if m_r not in P.columns or m_p not in P.columns:
                continue
            add(task, kind, m_r, m_p, LB[m_r], LB[m_p], note="incremental")
    res[f"{task}_ece"] = {c: fp.ece(np.clip(P[c].to_numpy(), 0, 1), yy) for c in mods}
    res[f"{task}_auc"] = {c: fp.auc(P[c].to_numpy(), yy) for c in mods}
    # AUC difference with account-cluster bootstrap
    idx_by = [np.flatnonzero(ainv == k) for k in range(J)]
    pairs = [(f"gbm_{task}_R", f"gbm_{task}_P"), (f"lmm_{task}_R", f"lmm_{task}_P"), (f"gbm_{task}_P", f"b_persist_{task}"), (f"gbm_{task}_RT", f"gbm_{task}_PT"), (f"gbm_{task}_RTO", f"gbm_{task}_PTO")]
    pairs = [(x, z) for x, z in pairs if x in P.columns and z in P.columns]
    d = {}
    for ma, mb in pairs:
        sa, sb = P[ma].to_numpy(), P[mb].to_numpy()
        pt = fp.auc(sa, yy) - fp.auc(sb, yy)
        bs = []
        for _ in range(300):
            ii = np.concatenate([idx_by[k] for k in rng.integers(0, J, J)])
            bs.append(fp.auc(sa[ii], yy[ii]) - fp.auc(sb[ii], yy[ii]))
        d[f"{ma}_minus_{mb}"] = dict(diff=float(pt), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)))
    res[f"{task}_auc_diff"] = d
SK = pd.DataFrame(rows)
SK.to_csv(OUT / f"skill_{ST}.csv", index=False)

# ---- horizon: days since the forecast origin
hb = pd.cut(P.horizon_days, [-1, 7, 14, 21, 40], labels=["0-7", "7-14", "14-21", "21+"])
hr = []
for lab in hb.cat.categories:
    s = (hb == lab).to_numpy()
    if s.sum() < 500:
        continue
    for m_r, m_p in [("gbm_reg_R", "gbm_reg_P"), ("lmm_reg_R", "lmm_reg_P"), ("gbm_reg_RT", "gbm_reg_PT"), ("lmm_reg_RT", "lmm_reg_PT")]:
        pt, lo, hi = fp.skill_boot(acct_sum(L[m_r] * s), acct_sum(L[m_p] * s), B=1000)
        hr.append(dict(horizon_days=lab, model=m_r, vs=m_p, n=int(s.sum()), skill=pt, lo=lo, hi=hi))
pd.DataFrame(hr).to_csv(OUT / f"horizon_{ST}.csv", index=False)

# ---- by origin month
orows = []
for o, s in P.groupby("origin").groups.items():
    s = np.asarray(P.index.get_indexer(s))
    for m_r, m_p in [("gbm_reg_R", "gbm_reg_P"), ("lmm_reg_R", "lmm_reg_P"), ("gbm_reg_RT", "gbm_reg_PT"), ("lmm_reg_RT", "lmm_reg_PT")]:
        orows.append(dict(origin=o, model=m_r, n=len(s), skill=1 - L[m_r][s].sum() / L[m_p][s].sum()))
OR = pd.DataFrame(orows)
OR.to_csv(OUT / f"origin_skill_{ST}.csv", index=False)

# ---- heterogeneity: incremental skill (GBM rhythm vs GBM persistence) by group
hrows = []
for var in ["party", "account_type", "chamber", "sched"]:
    for lev in P[var].dropna().unique():
        s = (P[var] == lev).to_numpy()
        if s.sum() < 1000 or len(np.unique(ainv[s])) < 10:
            continue
        for task, ml, mb in [("reg", "gbm_reg_R", "gbm_reg_P"), ("reg", "lmm_reg_R", "lmm_reg_P"), ("reg", "gbm_reg_RT", "gbm_reg_PT")]:
            pt, lo, hi = fp.skill_boot(acct_sum(L[ml] * s), acct_sum(L[mb] * s), B=1000)
            hrows.append(dict(variable=var, level=str(lev), model=ml, vs=mb, n=int(s.sum()),
                              accounts=int(len(np.unique(ainv[s]))), skill=pt, lo=lo, hi=hi,
                              mse_persist=float(L[mb][s].mean())))
        Lb = bin_loss("gbm_bina_R", "bina", "brier"), bin_loss("gbm_bina_P", "bina", "brier")
        pt, lo, hi = fp.skill_boot(acct_sum(Lb[0] * s), acct_sum(Lb[1] * s), B=1000)
        hrows.append(dict(variable=var, level=str(lev), model="gbm_bina_R", vs="gbm_bina_P", n=int(s.sum()),
                          accounts=int(len(np.unique(ainv[s]))), skill=pt, lo=lo, hi=hi, mse_persist=float(Lb[1][s].mean())))
HT = pd.DataFrame(hrows)
HT.to_csv(OUT / f"hetero_{ST}.csv", index=False)

# ---- calibration
cal = []
for c in [f"gbm_reg_{s}" for s in "PRF"] + ["lmm_reg_P", "lmm_reg_R"]:
    q = pd.qcut(P[c], 10, labels=False, duplicates="drop")
    g = pd.DataFrame({"p": P[c], "y": y, "q": q}).groupby("q").mean()
    for k, r in g.iterrows():
        cal.append(dict(task="reg", model=c, bin=int(k), pred=r.p, obs=r.y))
for task in ("bina", "binb"):
    for c in [f"gbm_{task}_P", f"gbm_{task}_R", f"lmm_{task}_P", f"lmm_{task}_R", f"b_persist_{task}"]:
        q = pd.qcut(P[c].rank(method="first"), 10, labels=False)
        g = pd.DataFrame({"p": P[c], "y": P[f"y_{task}"], "q": q}).groupby("q").mean()
        for k, r in g.iterrows():
            cal.append(dict(task=task, model=c, bin=int(k), pred=r.p, obs=r.y))
CAL = pd.DataFrame(cal)
CAL.to_csv(OUT / f"calibration_{ST}.csv", index=False)

# ---- figures
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 2, figsize=(11, 4 + 0.2 * len(SK[SK.note == "incremental"])))
for m, col in [("gbm_reg_R", PAL["R"]), ("lmm_reg_R", PAL["lmm"])]:
    d = OR[OR.model == m].sort_values("origin")
    ax[0].plot(pd.to_datetime(d.origin), d.skill * 100, "-o", ms=3, color=col, label={"gbm_reg_R": "boosted trees", "lmm_reg_R": "mixed model"}[m])
ax[0].axhline(0, color="k", lw=0.6)
ax[0].set_ylabel("incremental skill of rhythm features over persistence-only (%, MSE)")
ax[0].set_title("By forecast origin")
ax[0].legend(frameon=False)
sel = SK[(SK.note == "incremental")]
lab, pts, los, his = [], [], [], []
for _, r in sel.iterrows():
    lab.append(f"{r.task} {r.score}\n{r.model.replace('_', ' ')}")
    pts.append(r.skill * 100); los.append(r.lo * 100); his.append(r.hi * 100)
yy_ = np.arange(len(lab))
ax[1].errorbar(pts, yy_, xerr=[np.array(pts) - np.array(los), np.array(his) - np.array(pts)], fmt="o", color=PAL["R"], capsize=2)
ax[1].axvline(0, color="k", lw=0.6)
ax[1].set_yticks(yy_)
ax[1].set_yticklabels(lab, fontsize=6)
ax[1].set_xlabel("incremental skill (%) with 95% CI over accounts")
ax[1].set_title("Overall")
plt.tight_layout()
plt.savefig(OUT / f"fig1_skill_{ST}.png", dpi=140)
plt.close()

fig, ax = plt.subplots(1, 3, figsize=(12, 4))
for k, (task, ttl) in enumerate([("reg", "TI7 (regression)"), ("bina", "High vs account 80th pct"), ("binb", "High vs global 80th pct")]):
    for c, col, ls in ([(f"gbm_reg_P", PAL["P"], "-"), (f"gbm_reg_R", PAL["R"], "-"), (f"lmm_reg_R", PAL["lmm"], "--")] if task == "reg" else
                       [(f"gbm_{task}_P", PAL["P"], "-"), (f"gbm_{task}_R", PAL["R"], "-"), (f"lmm_{task}_R", PAL["lmm"], "--")]):
        d = CAL[(CAL.task == task) & (CAL.model == c)]
        ax[k].plot(d.pred, d.obs, ls, marker="o", ms=3, color=col, label=c.replace("_", " "))
    lim = [min(CAL[CAL.task == task].pred.min(), CAL[CAL.task == task].obs.min()), max(CAL[CAL.task == task].pred.max(), CAL[CAL.task == task].obs.max())]
    ax[k].plot(lim, lim, color="k", lw=0.6)
    ax[k].set_xlabel("predicted (decile mean)")
    ax[k].set_ylabel("observed")
    ax[k].set_title(ttl)
ax[0].legend(frameon=False, fontsize=7)
plt.tight_layout()
plt.savefig(OUT / f"fig2_calibration_{ST}.png", dpi=140)
plt.close()

H = HT[(HT.model == "gbm_reg_R")].reset_index(drop=True)
if len(H):
    fig, ax = plt.subplots(figsize=(6, 4))
    yv = np.arange(len(H))[::-1]
    ax.errorbar(H.skill * 100, yv, xerr=[(H.skill - H.lo) * 100, (H.hi - H.skill) * 100], fmt="o", color=PAL["R"], capsize=2)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_yticks(yv)
    ax.set_yticklabels([f"{r.variable}: {r.level} (n={r.accounts} acct)" for _, r in H.iterrows()], fontsize=7)
    ax.set_xlabel("incremental skill of rhythm features, MSE (%)")
    plt.tight_layout()
    plt.savefig(OUT / f"fig4_heterogeneity_{ST}.png", dpi=140)
    plt.close()

res["skill_table_rows"] = len(SK)
(OUT / f"metrics_{ST}.json").write_text(json.dumps(res, indent=1, default=float))
print(json.dumps({k: res[k] for k in ["reg_mse", "reg_best_baseline"]}, indent=1))
print(SK[(SK.note == "incremental") | (SK.model.str.startswith("gbm"))].round(4).to_string())
