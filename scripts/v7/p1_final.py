"""P1 step 4 (run after the final test stage): models trained on all rows before 2022-07-01, scored once on the final test
period 2022-07-01 .. 2023-07-11: feature-block ablations, partial dependence, mixed-model coefficients, gain importance.
Writes <out>/final.json, ablation.csv, pdp.csv, empirical_bins.csv, lmm_coefficients.csv, fig3_pdp.png
Usage: python p1_final.py --source subset|full
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p1_models import *  # noqa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

a = get_args()
OUT = outdir(a.source)
rng = np.random.default_rng(fp.SEED)
hp = json.loads((OUT / "hp.json").read_text())
F = pd.read_parquet(OUT / "features.parquet")
F["when"] = pd.to_datetime(F["when"])
tr, te, J, thr_g = prepare_origin(F, fp.TEST_START, fp.TEST_END)
print("train", len(tr), "test", len(te), "accounts", J, flush=True)
ac, ainv = np.unique(te.acct.values, return_inverse=True)
JA = len(ac)
y = te.ti7.to_numpy("float64")
out = dict(n_train=len(tr), n_test=len(te), accounts=int(JA))


def acct_sum(v):
    return np.bincount(ainv, weights=v, minlength=JA)


# ---- ablations: persistence plus one feature block (boosted trees, reg), hyperparameters of the rhythm model
BLOCKS = {"own timing": fp.OWN, "time of day": fp.TOD, "scheduler share": fp.SCHED, "lagged Congress volume": fp.CONG,
          "regime flags": fp.REGIME, "previous post type": fp.TYPE_PREV, "own post type (first post)": fp.TYPE_OWN}
sets = dict(SETS)
for k, v in BLOCKS.items():
    sets[k] = fp.PERSIST + v + fp.TE
sets["rhythm minus Congress"] = fp.PERSIST + fp.RHYTHM_NOCONG + fp.TE
hpa = dict(hp)
for k in [k for k in sets if k not in SETS]:
    hpa[f"reg_{k}"] = hp["reg_R"]
preds = {}
models = {}
for k, cols in sets.items():
    if cols is None:
        continue
    models[k] = fit_gbm(tr, "reg", k, hpa, sets)
    preds[k] = models[k].predict(te[cols])
L = {k: (y - v) ** 2 for k, v in preds.items()}
rows = []
for k in preds:
    if k == "P":
        continue
    pt, lo, hi = fp.skill_boot(acct_sum(L[k]), acct_sum(L["P"]), B=2000)
    rows.append(dict(added_to_persistence=k, mse=float(L[k].mean()), skill_vs_persistence_model=pt, lo=lo, hi=hi))
pd.DataFrame(rows).to_csv(OUT / "ablation.csv", index=False)
out["ablation"] = rows
out["mse_persistence_gbm"] = float(L["P"].mean())
imp = pd.Series(models["R"].feature_importance("gain"), index=SETS["R"])
out["gain_importance_share_R"] = (imp / imp.sum()).sort_values(ascending=False).round(4).to_dict()

# ---- mixed-model coefficients (standardized predictors; final fit before the test period)
rowsc = []
for task, s in [("reg", "R"), ("binb", "R")]:
    l = LMM(task, s).fit(tr, tr.ccode.to_numpy(), J)
    names = ["intercept"] + l.cols
    for n_, b_ in zip(names, l.m.beta):
        rowsc.append(dict(task=task, term=n_, coef=float(b_)))
    out[f"lmm_{task}_R_random_sd"] = dict(zip([names[i] for i in l.zidx], np.sqrt(np.diag(l.m.G)).round(4).tolist()))
    out[f"lmm_{task}_R_resid_sd"] = float(np.sqrt(l.m.s2))
pd.DataFrame(rowsc).to_csv(OUT / "lmm_coefficients.csv", index=False)

# ---- partial dependence (boosted trees with rhythm features), account-cluster bootstrap bands
S = te.sample(min(30000, len(te)), random_state=fp.SEED).copy()
Sa, Sinv = np.unique(S.acct.values, return_inverse=True)
bstr = models["R"]
bstb = fit_gbm(tr, "bina", "R", hp)
cols_r = SETS["R"]


def grid_set(S, name):
    d = S.copy()
    if name == "dt_s":
        return [(v, {"log_dt": np.log1p(v)}) for v in [5, 15, 30, 60, 180, 600, 1800, 3600, 3 * 3600, 6 * 3600, 12 * 3600, 86400, 3 * 86400]]
    if name == "sess_pos":
        return [(v, {"sess_pos": np.log1p(v)}) for v in [0, 1, 2, 3, 5, 8, 12, 20]]
    if name == "events_past_hour":
        return [(v, {"n1h": np.log1p(v), "n24h": np.maximum(d.n24h.values, np.log1p(v))}) for v in [0, 1, 2, 3, 5, 8, 12]]
    if name == "local_hour":
        out_ = []
        for h in range(24):
            ang = 2 * np.pi * h / 24
            out_.append((h, {"hour": float(h), "hsin": np.sin(ang), "hcos": np.cos(ang), "h2sin": np.sin(2 * ang),
                             "h2cos": np.cos(2 * ang), "night": float(h < 6)}))
        return out_
    if name == "congress_spike_prev3h":
        sp = tr.loc[tr.cong_spike > 0, "cong_spike_lr"].median()
        ns = tr.loc[tr.cong_spike == 0, "cong_spike_lr"].median()
        return [(0, {"cong_spike": 0.0, "cong_spike_lr": ns}), (1, {"cong_spike": 1.0, "cong_spike_lr": sp})]


pdp = []
for name in ["dt_s", "sess_pos", "events_past_hour", "local_hour", "congress_spike_prev3h"]:
    for val, setd in grid_set(S, name):
        d = S[cols_r].copy()
        for c, v in setd.items():
            d[c] = v
        for mname, mdl in [("ti7", bstr), ("p_high_acct80", bstb)]:
            pr = mdl.predict(d[cols_r])
            cnt = rng.multinomial(len(Sa), np.full(len(Sa), 1 / len(Sa)), size=300)
            sums = np.bincount(Sinv, weights=pr, minlength=len(Sa))
            ns_ = np.bincount(Sinv, minlength=len(Sa))
            bs = (cnt @ sums) / (cnt @ ns_)
            pdp.append(dict(feature=name, value=val, outcome=mname, pdp=float(pr.mean()), lo=float(np.percentile(bs, 2.5)),
                            hi=float(np.percentile(bs, 97.5))))
PD = pd.DataFrame(pdp)
PD.to_csv(OUT / "pdp.csv", index=False)

# ---- empirical bins (account-demeaned TI7 and observed high rate) with account-cluster bootstrap
te["dev"] = y - te.te_mean.to_numpy()
edges = {"dt_s": [0, 10, 30, 60, 300, 900, 1800, 3600, 4 * 3600, 12 * 3600, 86400, 10 ** 9],
         "sess_pos": [-1, 0, 1, 2, 4, 8, 16, 10 ** 6], "events_past_hour": [-1, 0, 1, 2, 4, 8, 10 ** 6]}
raw = {"dt_s": np.expm1(te.log_dt.to_numpy()), "sess_pos": np.expm1(te.sess_pos.to_numpy()),
       "events_past_hour": np.expm1(te.n1h.to_numpy()), "local_hour": te.hour.to_numpy(),
       "congress_spike_prev3h": te.cong_spike.to_numpy()}
erows = []
for name, v in raw.items():
    if name in edges:
        b = pd.cut(v, edges[name], labels=False, right=True)
        labels = {i: f"({edges[name][i]},{edges[name][i + 1]}]" for i in range(len(edges[name]) - 1)}
    else:
        b = pd.Series(v).astype(int).to_numpy()
        labels = {i: str(i) for i in np.unique(b)}
    for k in np.unique(b[~pd.isna(b)]):
        s = b == k
        if s.sum() < 200:
            continue
        for oc, vals in [("ti7_dev_from_account_mean", te.dev.to_numpy()), ("high_acct80", te.y_bina.to_numpy())]:
            ssum = np.bincount(ainv, weights=vals * s, minlength=JA)
            sn = np.bincount(ainv, weights=s.astype(float), minlength=JA)
            cnt = rng.multinomial(JA, np.full(JA, 1 / JA), size=500)
            bs = (cnt @ ssum) / np.maximum(cnt @ sn, 1)
            erows.append(dict(feature=name, bin=labels[int(k)], bin_index=int(k), outcome=oc, n=int(s.sum()),
                              mean=float(ssum.sum() / sn.sum()), lo=float(np.percentile(bs, 2.5)),
                              hi=float(np.percentile(bs, 97.5))))
EB = pd.DataFrame(erows)
EB.to_csv(OUT / "empirical_bins.csv", index=False)

# ---- figure
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
feats = ["dt_s", "sess_pos", "events_past_hour", "local_hour", "congress_spike_prev3h"]
tt = {"dt_s": "Time since last post (s, log scale)", "sess_pos": "Position in session (earlier posts)",
      "events_past_hour": "Posts in the past hour", "local_hour": "Local hour", "congress_spike_prev3h": "Congress-wide spike, previous 3 h"}
fig, axs = plt.subplots(2, 5, figsize=(17, 6))
for j, f in enumerate(feats):
    for i, oc in enumerate(["ti7", "p_high_acct80"]):
        d = PD[(PD.feature == f) & (PD.outcome == oc)]
        ax = axs[i, j]
        ax.plot(d.value, d.pdp, "-o", ms=3, color="#D55E00")
        ax.fill_between(d.value, d.lo, d.hi, color="#D55E00", alpha=0.2, lw=0)
        if f == "dt_s":
            ax.set_xscale("log")
        if f == "congress_spike_prev3h":
            ax.set_xticks([0, 1])
        ax.set_title(tt[f], fontsize=8) if i == 0 else None
        ax.set_ylabel("mean predicted TI7" if oc == "ti7" else "mean predicted P(high provocation)")
plt.tight_layout()
plt.savefig(OUT / "fig3_pdp.png", dpi=140)
plt.close()
(OUT / "final.json").write_text(json.dumps(out, indent=1, default=float))
print(json.dumps({k: out[k] for k in ["ablation", "gain_importance_share_R"]}, indent=1, default=float))
