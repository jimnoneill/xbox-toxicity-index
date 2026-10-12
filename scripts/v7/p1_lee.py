"""P1 step 5: out-of-sample application to Lee (personal @BasedMikeLee, official @SenMikeLee).

Models are trained on congress members only (all rows before 2023-07-12) and applied to Lee's 2022-07 to 2026-10 events with
no refit.  Lee's own rows never enter training, thresholds or tuning.  Account random effects are unavailable for a new
account (population-level mixed model); account target encoding is replaced by the expanding mean of his own earlier events
(at least 20 earlier events, otherwise the congress global mean), a causal quantity.  Congress lagged features exist only up
to 2023-07-11, so the 2022-07 .. 2026-10 application uses models without them; the overlap window also reports the full model.
Bootstrap: calendar weeks (block) resampled.  One case, predictive association only.
Writes <out>/lee.json, lee_predictions.parquet, lee_monthly.csv, fig5_lee.png.   Usage: python p1_lee.py --source subset|full
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p1_models import *  # noqa
from t6_common import load_posts, T6  # noqa
import t1_common as c1  # noqa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

a = get_args()
OUT = outdir(a.source)
rng = np.random.default_rng(fp.SEED)
hp = json.loads((OUT / "hp.json").read_text())
F = pd.read_parquet(OUT / "features.parquet")
F["when"] = pd.to_datetime(F["when"])
tr, _, J, thr_g = prepare_origin(F, fp.TEST_END, fp.TEST_END + pd.Timedelta(days=1))
print("train rows", len(tr), "accounts", J, flush=True)
GM, GSD = float(tr.ti7.mean()), float(tr.ti7.std())
TE_SD = float(tr.te_sd.mean())

# ---- models trained on congress members only
G = {}
for task in ("reg", "binb"):
    G[f"{task}_P"] = fit_gbm(tr, task, "P", hp)
    G[f"{task}_R"] = fit_gbm(tr, task, "R", hp, SETS_NOCONG)
    G[f"{task}_Rc"] = fit_gbm(tr, task, "R", hp)             # with lagged Congress features
    for s in ("PT", "RT") + (("PTO", "RTO") if task == "reg" else ()):
        G[f"{task}_{s}"] = fit_gbm(tr, task, s, hp, SETS_NOCONG)   # post-type variants (Lee: reply flag only)
Lm = {}
for task in ("reg", "binb"):
    Lm[f"{task}_P"] = LMM(task, "P").fit(tr, tr.ccode.to_numpy(), J)
    Lm[f"{task}_R"] = LMM(task, "R", nocong=True).fit(tr, tr.ccode.to_numpy(), J)
    Lm[f"{task}_Rc"] = LMM(task, "R").fit(tr, tr.ccode.to_numpy(), J)
q = tr.ti7.quantile(0.8)
prev_hi = (tr.prev_ti7 > q).to_numpy()
r1, r0 = tr.y_binb.to_numpy()[prev_hi].mean(), tr.y_binb.to_numpy()[~prev_hi].mean()
base_rate = float(tr.y_binb.mean())

# ---- congress lag features by hour
ch = pd.read_parquet(c1.ROOT / "output/v7/t2full/congress_hourly.parquet")
hmin, hmax = int(ch.hour_idx.min()), int(ch.hour_idx.max())
cc = np.zeros(hmax - hmin + 1)
bad = np.ones(hmax - hmin + 1, dtype=bool)
cc[ch.hour_idx.values - hmin] = ch.total_events.values
bad[ch.hour_idx.values - hmin] = (~ch.observed.values) | ch.outage.values
CG = fp.congress_hour_features(cc, bad)

md, cd, mch = c1.lc.load_vote_days(c1.ROOT / "data/voteview")
ml = c1.lc.MemberLocation(md[c1.LEE_ICPSR], cd["S"], "UT", name="Lee")
DIM_COLS = [f"p_{d}" for d in fp.DIMS7]


def collapse(df, self_id):
    nonrt = c1.tv.nonretweet_mask_lee(df.type.values)
    x = df[nonrt].reset_index(drop=True)
    n = len(x)
    txt = x.text.fillna("").values
    mk = c1.hy.thread_markers(txt)
    forbid = np.array(c1.hy.addressee_mismatch(txt), dtype=bool)
    forbid[0] = False
    t = x.t.values
    ids = x.id.astype(str).values
    irt = x.in_reply_to_user_id.astype(str).values
    conv = x.conversation_id.astype(str).values
    ref = x.referenced_tweet_ids.astype(str).values
    typ = x.type.values
    extra = np.zeros(n, dtype=bool)
    selfreply = (typ == "reply") & (irt == self_id)
    for k in range(1, 6):
        prev_id = np.concatenate([[""] * k, ids[:-k]])
        dt = np.concatenate([[10 ** 9] * k, t[k:] - t[:-k]])
        extra |= selfreply & (ref == prev_id) & (dt < 3600)
    pc = np.concatenate([[""], conv[:-1]])
    dt1 = np.concatenate([[10 ** 9], np.diff(t)])
    extra |= selfreply & (conv == pc) & (conv != "nan") & (dt1 < 3600)
    g = np.zeros(n, dtype=object)
    g[:] = "a"
    tid, _ = c1.hy.collapse_threads(g, t, mk.num.values, mk.tot.values, mk.kw.values, extra_link=extra, forbid=forbid)
    x["thread_id"] = tid
    return x


def lee_events(which):
    D = load_posts(which)
    if which == "personal":
        pp = pd.read_parquet(c1.ROOT / "output/v7/ti/post_probs_lee.parquet", columns=["id"] + DIM_COLS)
        pp["id"] = pp.id.astype(str)
        D["id"] = D.id.astype(str)
        D = D.merge(pp, on="id", how="left")
        sid = c1.LEE_ID
    else:
        raw = pd.read_parquet(c1.ROOT / "data/SenMikeLee_2022_2026.parquet", columns=["id"])
        pr = pd.DataFrame({"id": raw.id.astype(str)})
        for d in fp.DIMS7:
            pr[f"p_{d}"] = np.load(T6 / f"_probs_official/{d}.npy")
        D["id"] = D.id.astype(str)
        D = D.merge(pr, on="id", how="left")
        from collections import Counter
        import glob
        cn = Counter()
        for fn in glob.glob(str(c1.ROOT / "data/senmikelee_raw_20261009/page_*.json")):
            for tw in json.load(open(fn)).get("data", []):
                cn[tw["author_id"]] += 1
        sid = cn.most_common(1)[0][0]
    D["ti7"] = D[DIM_COLS].mean(axis=1)
    D["cclass"] = c1.tv.client_class(D.source.values) if "source" in D else "other"
    x = collapse(D, sid)
    ev = x.groupby("thread_id").agg(t=("t", "first"), ti7=("ti7", "mean"), size=("thread_id", "size"),
                                    tool=("cclass", lambda s: s.iloc[0] == "tool"),
                                    reply=("type", lambda s: float(s.iloc[0] == "reply"))).reset_index()
    ev = ev.sort_values(["t", "thread_id"]).reset_index(drop=True)
    return ev


def features(ev):
    t = ev.t.values
    f = fp.account_features(t, ev.ti7.values, ev.tool.values, ev.reply.values, (ev["size"].values >= 2).astype(float))
    cls = ml.classes(t)
    h, dow, _ = ml.local_hour_dow(t, (c1.lc.PRIMARY,), cls)[c1.lc.PRIMARY]
    d = pd.DataFrame(dict(t=t, when=pd.to_datetime(t, unit="s"), ti7=ev.ti7.values, size=ev["size"].values))
    for k in ["prev_ti7", "roll10", "roll50", "sched_share", "n_scored_prior", "prev_scored", "dt_prev",
              "prev_reply", "prev_multi", "reply_share10", "own_reply"]:
        d[k] = f[k]
    d["log_dt"] = np.log1p(f["dt_prev"])
    d["sess_pos"] = np.log1p(f["sess_pos"])
    d["sess_el"] = np.log1p(f["sess_el"])
    d["n1h"] = np.log1p(f["n1h"])
    d["n24h"] = np.log1p(f["n24h"])
    d["hour"] = h.astype(float)
    ang = 2 * np.pi * d.hour / 24
    d["hsin"], d["hcos"], d["h2sin"], d["h2cos"] = np.sin(ang), np.cos(ang), np.sin(2 * ang), np.cos(2 * ang)
    d["night"] = (d.hour < 6).astype(float)
    d["weekend"] = (dow >= 5).astype(float)
    hh = t // 3600 - hmin
    inr = (hh >= 0) & (hh <= hmax - hmin)
    for k, v in CG.items():
        d[k] = np.where(inr, v[np.clip(hh, 0, len(v) - 1)], np.nan)
    # expanding own mean as account encoding (causal), cold start variant = congress mean
    sc = ~np.isnan(ev.ti7.values)
    cs = np.concatenate([[0.0], np.cumsum(np.where(sc, ev.ti7.values, 0.0))])
    cn_ = np.concatenate([[0], np.cumsum(sc)])
    cnt = f["cnt"]
    n_pr, s_pr = cn_[cnt], cs[cnt]
    d["te_mean"] = np.where(n_pr >= 20, (s_pr + 20 * GM) / (n_pr + 20), GM)
    d["te_sd"] = TE_SD
    d["te_mean_cold"] = GM
    d["y_binb"] = (d.ti7 > thr_g).astype(float)
    return d


def block_boot(la, lb, wk, B=1000):
    keys, inv = np.unique(wk, return_inverse=True)
    return fp.skill_boot(np.bincount(inv, weights=la, minlength=len(keys)), np.bincount(inv, weights=lb, minlength=len(keys)), B=B)


def evaluate(d, label, mask_name, mask):
    d = d[mask]
    if len(d) < 300:
        return None
    y_ = d.ti7.to_numpy()
    wk = d.when.dt.to_period("W").astype(str).to_numpy()
    r = dict(window=mask_name, n=int(len(d)), ti7_mean=float(y_.mean()), high_rate=float(d.y_binb.mean()))
    P = {"b_persist": d.prev_ti7.to_numpy(), "b_roll50": d.roll50.to_numpy(), "b_exp_mean": d.te_mean.to_numpy(),
         "b_congress_mean": np.full(len(d), GM)}
    for k in ("P", "R"):
        P[f"gbm_{k}"] = G[f"reg_{k}"].predict(d[SETS[k] if k == "P" else SETS_NOCONG["R"]])
        P[f"lmm_{k}"] = Lm[f"reg_{k}"].predict(d, np.full(len(d), -1))
    dc = d.assign(te_mean=d.te_mean_cold)
    for k in ("PT", "RT", "PTO", "RTO"):
        P[f"gbm_{k}"] = G[f"reg_{k}"].predict(d[SETS_NOCONG[k]])
    P["gbm_P_coldTE"] =G["reg_P"].predict(dc[SETS["P"]])
    P["gbm_R_coldTE"] = G["reg_R"].predict(dc[SETS_NOCONG["R"]])
    if d.cong_p1.notna().all():
        P["gbm_Rc"] = G["reg_Rc"].predict(d[SETS["R"]])
        P["lmm_Rc"] = Lm["reg_Rc"].predict(d, np.full(len(d), -1))
    mse = {k: float(np.mean((y_ - v) ** 2)) for k, v in P.items()}
    best = min([k for k in P if k.startswith("b_")], key=lambda k: mse[k])
    r["mse"] = mse
    r["best_baseline"] = best
    r["mean_pred"] = {k: float(np.mean(v)) for k, v in P.items()}
    lo_ = {k: (y_ - v) ** 2 for k, v in P.items()}
    sk = {}
    for k in P:
        if k != best:
            pt, lo, hi = block_boot(lo_[k], lo_[best], wk)
            sk[k] = dict(skill=pt, lo=lo, hi=hi)
    for k, b_ in [("gbm_R", "gbm_P"), ("lmm_R", "lmm_P"), ("gbm_Rc", "gbm_P"), ("lmm_Rc", "lmm_P"),
                 ("gbm_RT", "gbm_PT"), ("gbm_RTO", "gbm_PTO"), ("gbm_PT", "gbm_P"), ("gbm_PTO", "gbm_PT")]:
        if k in P:
            pt, lo, hi = block_boot(lo_[k], lo_[b_], wk)
            sk[f"{k}_vs_{b_}"] = dict(skill=pt, lo=lo, hi=hi)
    r["skill"] = sk
    # binary (congress-global 80th percentile threshold)
    yb = d.y_binb.to_numpy()
    if 0 < yb.sum() < len(yb):
        B = {"b_rate": np.full(len(d), base_rate), "b_persist": np.where(d.prev_ti7.to_numpy() > q, r1, r0),
             "gbm_P": G["binb_P"].predict(d[SETS["P"]]), "gbm_R": G["binb_R"].predict(d[SETS_NOCONG["R"]]),
             "lmm_P": Lm["binb_P"].predict(d, np.full(len(d), -1)), "lmm_R": Lm["binb_R"].predict(d, np.full(len(d), -1))}
        bs = {k: float(np.mean((yb - np.clip(v, 1e-4, 1 - 1e-4)) ** 2)) for k, v in B.items()}
        r["bin_brier"] = bs
        r["bin_auc"] = {k: fp.auc(v, yb) for k, v in B.items() if k != "b_rate"}
        r["bin_ece"] = {k: fp.ece(np.clip(v, 0, 1), yb) for k, v in B.items()}
        bb = min(["b_rate", "b_persist"], key=lambda k: bs[k])
        sb = {}
        for k in B:
            if k != bb:
                pt, lo, hi = block_boot((yb - B[k]) ** 2, (yb - B[bb]) ** 2, wk)
                sb[k] = dict(skill=pt, lo=lo, hi=hi)
        for k, b_ in [("gbm_R", "gbm_P"), ("lmm_R", "lmm_P")]:
            pt, lo, hi = block_boot((yb - B[k]) ** 2, (yb - B[b_]) ** 2, wk)
            sb[f"{k}_vs_{b_}"] = dict(skill=pt, lo=lo, hi=hi)
        r["bin_brier_skill"] = sb
        r["bin_best_baseline"] = bb
        # AUC difference rhythm vs persistence (week blocks)
        keys, inv = np.unique(wk, return_inverse=True)
        idx_by = [np.flatnonzero(inv == j) for j in range(len(keys))]
        bsamp = []
        for _ in range(200):
            ii = np.concatenate([idx_by[j] for j in rng.integers(0, len(keys), len(keys))])
            bsamp.append(fp.auc(B["gbm_R"][ii], yb[ii]) - fp.auc(B["gbm_P"][ii], yb[ii]))
        r["bin_auc_diff_gbm_R_minus_P"] = dict(diff=r["bin_auc"]["gbm_R"] - r["bin_auc"]["gbm_P"],
                                               lo=float(np.percentile(bsamp, 2.5)), hi=float(np.percentile(bsamp, 97.5)))
    return r


res = {}
allpred = []
for which in ("personal", "official"):
    ev = lee_events(which)
    d = features(ev)
    ok = d.ti7.notna() & d.prev_scored & (d.n_scored_prior >= MIN_PRIOR_SCORED)
    d = d[ok].reset_index(drop=True)
    d["account"] = which
    yr = d.when.dt.year
    t0_ = d.when.min()
    out = dict(events_scored=int(len(d)), first=str(t0_), last=str(d.when.max()))
    wins = {"all_2022_2026": np.ones(len(d), bool),
            "congress_overlap_to_2023-07-11": (d.when <= pd.Timestamp("2023-07-11 23:59")).to_numpy(),
            "first_12_months": (d.when < t0_ + pd.Timedelta(days=365)).to_numpy()}
    for y_ in range(2022, 2027):
        wins[f"year_{y_}"] = (yr == y_).to_numpy()
    out["windows"] = {k: evaluate(d, which, k, m) for k, m in wins.items()}
    res[which] = out
    d["pred_gbm_P"] = G["reg_P"].predict(d[SETS["P"]])
    d["pred_gbm_R"] = G["reg_R"].predict(d[SETS_NOCONG["R"]])
    allpred.append(d[["account", "when", "ti7", "prev_ti7", "roll50", "pred_gbm_P", "pred_gbm_R", "y_binb"]])
    print(which, len(d), flush=True)
pr = pd.concat(allpred)
pr.to_parquet(OUT / "lee_predictions.parquet", index=False)
res["training"] = dict(rows=int(len(tr)), accounts=int(J), global_mean=GM, global_threshold_80=float(thr_g))
(OUT / "lee.json").write_text(json.dumps(res, indent=1, default=float))
mon = pr.assign(month=pr.when.dt.to_period("M").dt.to_timestamp()).groupby(["account", "month"]).agg(
    n=("ti7", "size"), obs=("ti7", "mean"), gbm_P=("pred_gbm_P", "mean"), gbm_R=("pred_gbm_R", "mean"), persist=("prev_ti7", "mean")).reset_index()
mon.to_csv(OUT / "lee_monthly.csv", index=False)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for k, which in enumerate(("personal", "official")):
    m = mon[(mon.account == which) & (mon.n >= 20)]
    ax[k].plot(m.month, m.obs, "-", color="k", lw=1.2, label="observed")
    ax[k].plot(m.month, m.gbm_P, "-", color="#0072B2", lw=1, label="persistence-only model")
    ax[k].plot(m.month, m.gbm_R, "-", color="#D55E00", lw=1, label="persistence + rhythm model")
    ax[k].set_title(f"{which} account, monthly mean TI7")
ax[0].legend(frameon=False, fontsize=7)
plt.tight_layout()
plt.savefig(OUT / "fig5_lee.png", dpi=140)
print(json.dumps({w: {k: (v["skill"] if v else None) for k, v in res[w]["windows"].items() if k in ("all_2022_2026",)} for w in ("personal", "official")}, indent=1, default=float)[:3000])
