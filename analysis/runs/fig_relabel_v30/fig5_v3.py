"""Strict-v2 (corrected-design) counterparts of the archived RQ1 Figure 5 diagnostics.

Read-only on everything outside this directory. Reuses the frozen P1 LightGBM recipe and the
aligned strict-v2 cohort/feature repair from ../rq1_gbt_v2/run_gbt.py without modification.

Stages (run in order):
  monthly : 13 monthly origins, exact aligned cohort (422,529 test events); refits P and R and
            verifies them against ../rq1_gbt_v2/predictions_aligned, then fits PT/RT/PTO/RTO and the
            single-block ablation models (persistence + one block, reg_R hyperparameters as archived).
  lmm     : population+account mixed models P and R at the same 13 origins (archived panel a row 4).
  final   : single final origin 2022-07-01 (archived p1_final procedure): ablation, partial dependence
            (risk curves), empirical bins, on the corrected cohort/features.
  lee     : Mike Lee transfer at the final origin 2023-07-12. Mode 'archived' reproduces lee.json from
            the legacy inputs (pipeline check); mode 'strict' rebuilds Lee inputs with the corrected clock.
  report  : all numbers + CIs -> results.json
  figure  : fig5_strict_v2.png/.pdf
Usage: PYTHONPATH=$HOME/x_box/pyextra ~/x_box/venv/bin/python -u fig5_v2.py --stage <stage>
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
assert ROOT.name == "pipeline" and HERE.name in ("rq1_strict_v2_fig5", "fig_relabel_v30")
GBT = ROOT / "manuscript/epjds_submission_work/rq1_gbt_v2"
STRICT = ROOT / "manuscript/epjds_submission_work/rq1_strict_v2"
sys.path.insert(0, str(GBT))
import run_gbt as rg  # noqa: E402  (definitions only; nothing in it runs or writes at import)

m = rg.original_models()
fp = m.fp
SEED = rg.SEED
assert SEED == 20261010 == fp.SEED
BOOT = 2000

BLOCKS = {"own timing": fp.OWN, "time of day": fp.TOD, "scheduler share": fp.SCHED,
          "lagged Congress volume": fp.CONG, "regime flags": fp.REGIME,
          "previous post type": fp.TYPE_PREV, "own post type (first post)": fp.TYPE_OWN}
SETS = dict(m.SETS)
for _k, _v in BLOCKS.items():
    SETS[_k] = fp.PERSIST + _v + fp.TE
SETS["rhythm minus Congress"] = fp.PERSIST + fp.RHYTHM_NOCONG + fp.TE
MAIN = ["P", "R", "PT", "RT", "PTO", "RTO"]
ABL = list(BLOCKS) + ["rhythm minus Congress"]


def frozen():
    hp = rg.frozen_hp()
    hpa = dict(hp)
    for k in SETS:
        if k not in m.SETS:
            hpa[f"reg_{k}"] = hp["reg_R"]  # archived p1_final: blocks use the rhythm model's hyperparameters
    return hp, hpa


def col(name):
    return "m_" + name.replace(" ", "_").replace("(", "").replace(")", "")


def load_frames():
    """Aligned strict-v2 frames (run_gbt.aligned_repaired_frames) plus closure-aware prev_multi."""
    frame = pd.read_parquet(rg.P1 / "features.parquet")
    frame["when"] = pd.to_datetime(frame["when"])
    needed = (["thread_id", "ti7", "t_end", "strict_n_completed_prior", "strict_prev_multi",
               "prev_reply", "reply_share10", "own_reply"] + ["strict_" + c for c in rg.ALIGNED_COLS])
    ref = pd.read_parquet(STRICT / "strict_features.parquet", columns=needed)
    legacy, repaired = rg.aligned_repaired_frames(frame, ref)
    audit = {}
    for c in ("prev_reply", "reply_share10", "own_reply"):  # start-observable type columns: must be identical
        audit[f"n_diff_{c}_p1_vs_strict_file"] = int((~np.isclose(frame[c], ref[c], rtol=0, atol=0,
                                                                   equal_nan=True)).sum())
    repaired["prev_multi"] = ref.strict_prev_multi.to_numpy(dtype="float32")
    audit["n_changed_prev_multi_closure"] = int((~np.isclose(frame.prev_multi, repaired.prev_multi, rtol=0,
                                                             atol=0, equal_nan=True)).sum())
    return legacy, repaired, audit


def strict_select(legacy, repaired, origin, end):
    elig = legacy.strict_n_completed_prior.to_numpy() >= 10
    return rg.prepare_matched(legacy.loc[elig], repaired.loc[elig], origin, end, embargo=7200)


# ----------------------------------------------------------------------------- monthly
def run_monthly():
    t0 = time.time()
    legacy, repaired, audit = load_frames()
    hp, hpa = frozen()
    d = HERE / "preds_monthly"
    d.mkdir(exist_ok=True)
    attrs = pd.read_parquet(rg.P1 / "features.parquet", columns=["acct", "party", "chamber", "account_type",
                                                                  "flag_strong"]).drop_duplicates("acct").set_index("acct")
    checks = {}
    for origin in rg.months():
        key = f"{origin:%Y-%m}"
        path = d / f"origin_{key}.parquet"
        if path.exists():
            print(key, "exists", flush=True)
            continue
        tr, te, trr, ter = rg.select_aligned_origin(legacy, repaired, origin)
        ref = pd.read_parquet(GBT / "predictions_aligned" / f"origin_{key}.parquet")
        if not np.array_equal(ter.thread_id.to_numpy(), ref.thread_id.to_numpy()):
            raise AssertionError("cohort differs from predictions_aligned " + key)
        out = ter[["thread_id", "acct", "ti7", "y_bina"]].copy()
        out["origin"] = origin
        out["roll50"] = ter.roll50.to_numpy(dtype="float64")
        out["orig_p"] = ref.orig_p.to_numpy()
        out["orig_r"] = ref.orig_r.to_numpy()
        for c in ("party", "chamber", "account_type"):
            out[c] = out.acct.map(attrs[c]).to_numpy()
        out["sched"] = out.acct.map(attrs.flag_strong).fillna(False).astype(bool).to_numpy()
        for k in MAIN + ABL:
            out[col(k)] = m.fit_gbm(trr, "reg", k, hpa, SETS).predict(ter[SETS[k]]).astype("float32")
        dp = float(np.max(np.abs(out.m_P.to_numpy() - ref.aligned_p.to_numpy())))
        dr = float(np.max(np.abs(out.m_R.to_numpy() - ref.aligned_r.to_numpy())))
        if max(dp, dr) > 0:
            raise AssertionError(f"P/R refit does not reproduce predictions_aligned {key}: {dp} {dr}")
        checks[key] = {"n_train": len(trr), "n_test": len(ter), "max_abs_diff_P": dp, "max_abs_diff_R": dr}
        out.to_parquet(path, index=False)
        print(key, checks[key], round(time.time() - t0), "s", flush=True)
    p = HERE / "monthly_checks.json"
    old = json.loads(p.read_text()) if p.exists() else {}
    old.update(checks)
    old["feature_audit"] = audit
    p.write_text(json.dumps(old, indent=2))


def run_lmm():
    t0 = time.time()
    legacy, repaired, _ = load_frames()
    d = HERE / "preds_lmm"
    d.mkdir(exist_ok=True)
    for origin in rg.months():
        key = f"{origin:%Y-%m}"
        path = d / f"origin_{key}.parquet"
        if path.exists():
            continue
        tr, te, trr, ter = rg.select_aligned_origin(legacy, repaired, origin)
        J = int(trr.ccode.max()) + 1
        out = ter[["thread_id"]].copy()
        for s in ("P", "R"):
            l = m.LMM("reg", s).fit(trr, trr.ccode.to_numpy(), J)
            out[f"lmm_{s}"] = np.asarray(l.predict(ter, ter.ccode.to_numpy()), dtype="float32")
        out.to_parquet(path, index=False)
        print(key, "lmm", len(out), round(time.time() - t0), "s", flush=True)


# ----------------------------------------------------------------------------- final origin
def run_final():
    """Archived p1_final.py procedure (single origin 2022-07-01) on corrected inputs."""
    legacy, repaired, _ = load_frames()
    hp, hpa = frozen()
    rng = np.random.default_rng(fp.SEED)
    tr, te_l, trr, te = strict_select(legacy, repaired, fp.TEST_START, fp.TEST_END)
    print("final train", len(trr), "test", len(te), flush=True)
    y = te.ti7.to_numpy("float64")
    preds, models = {}, {}
    for k in MAIN + ABL:
        models[k] = m.fit_gbm(trr, "reg", k, hpa, SETS)
        preds[k] = models[k].predict(te[SETS[k]])
    out = te[["thread_id", "acct", "ti7"]].copy()
    for k, v in preds.items():
        out[col(k)] = v.astype("float32")
    out.to_parquet(HERE / "preds_final_origin.parquet", index=False)
    ac, ainv = np.unique(te.acct.values, return_inverse=True)
    JA = len(ac)
    # ---- partial dependence (copy of archived grid / bootstrap) on the corrected rhythm models
    S = te.sample(min(30000, len(te)), random_state=fp.SEED).copy()
    Sa, Sinv = np.unique(S.acct.values, return_inverse=True)
    bstr = models["R"]
    bstb = m.fit_gbm(trr, "bina", "R", hp)
    cols_r = m.SETS["R"]

    def grid_set(S, name):
        if name == "dt_s":
            return [(v, {"log_dt": np.log1p(v)}) for v in [5, 15, 30, 60, 180, 600, 1800, 3600, 3 * 3600, 6 * 3600, 12 * 3600, 86400, 3 * 86400]]
        if name == "sess_pos":
            return [(v, {"sess_pos": np.log1p(v)}) for v in [0, 1, 2, 3, 5, 8, 12, 20]]
        if name == "events_past_hour":
            return [(v, {"n1h": np.log1p(v), "n24h": np.maximum(S.n24h.values, np.log1p(v))}) for v in [0, 1, 2, 3, 5, 8, 12]]
        if name == "local_hour":
            o = []
            for h in range(24):
                ang = 2 * np.pi * h / 24
                o.append((h, {"hour": float(h), "hsin": np.sin(ang), "hcos": np.cos(ang), "h2sin": np.sin(2 * ang),
                              "h2cos": np.cos(2 * ang), "night": float(h < 6)}))
            return o
        if name == "congress_spike_prev3h":
            sp = trr.loc[trr.cong_spike > 0, "cong_spike_lr"].median()
            ns = trr.loc[trr.cong_spike == 0, "cong_spike_lr"].median()
            return [(0, {"cong_spike": 0.0, "cong_spike_lr": ns}), (1, {"cong_spike": 1.0, "cong_spike_lr": sp})]

    pdp = []
    for name in ["dt_s", "sess_pos", "events_past_hour", "local_hour", "congress_spike_prev3h"]:
        for val, setd in grid_set(S, name):
            dd = S[cols_r].copy()
            for c, v in setd.items():
                dd[c] = v
            for mname, mdl in [("ti7", bstr), ("p_high_acct80", bstb)]:
                pr = mdl.predict(dd[cols_r])
                cnt = rng.multinomial(len(Sa), np.full(len(Sa), 1 / len(Sa)), size=300)
                sums = np.bincount(Sinv, weights=pr, minlength=len(Sa))
                ns_ = np.bincount(Sinv, minlength=len(Sa))
                bs = (cnt @ sums) / (cnt @ ns_)
                pdp.append(dict(feature=name, value=val, outcome=mname, pdp=float(pr.mean()),
                                lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5))))
    pd.DataFrame(pdp).to_csv(HERE / "pdp_strict_v2.csv", index=False)
    # ---- empirical bins (archived definition; hour is the corrected strict hour)
    dev = y - te.te_mean.to_numpy()
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
            for oc, vals in [("ti7_dev_from_account_mean", dev), ("high_acct80", te.y_bina.to_numpy())]:
                ssum = np.bincount(ainv, weights=vals * s, minlength=JA)
                sn = np.bincount(ainv, weights=s.astype(float), minlength=JA)
                cnt = rng.multinomial(JA, np.full(JA, 1 / JA), size=500)
                bs = (cnt @ ssum) / np.maximum(cnt @ sn, 1)
                erows.append(dict(feature=name, bin=labels[int(k)], bin_index=int(k), outcome=oc, n=int(s.sum()),
                                  mean=float(ssum.sum() / sn.sum()), lo=float(np.percentile(bs, 2.5)),
                                  hi=float(np.percentile(bs, 97.5))))
    pd.DataFrame(erows).to_csv(HERE / "empirical_bins_strict_v2.csv", index=False)
    (HERE / "final_origin_info.json").write_text(json.dumps(
        {"origin": str(fp.TEST_START), "end": str(fp.TEST_END), "n_train": len(trr), "n_test": len(te),
         "accounts_test": int(JA), "pdp_sample": len(S)}, indent=2))
    print("final done", flush=True)


# ----------------------------------------------------------------------------- Lee
def lee_loaders():
    sys.path.insert(0, str(rg.SCRIPT))
    from t6_common import load_posts, T6
    import t1_common as c1
    return load_posts, T6, c1


def lee_events(which):
    """Verbatim archived p1_lee.collapse/lee_events, plus thread end time t_end (last post)."""
    load_posts, T6, c1 = lee_loaders()
    DIM_COLS = [f"p_{d}" for d in fp.DIMS7]
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
        for dd in fp.DIMS7:
            pr[f"p_{dd}"] = np.load(T6 / f"_probs_official/{dd}.npy")
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
    nonrt = c1.tv.nonretweet_mask_lee(D.type.values)
    x = D[nonrt].reset_index(drop=True)
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
    selfreply = (typ == "reply") & (irt == sid)
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
    ev = x.groupby("thread_id").agg(t=("t", "first"), t_end=("t", "max"), ti7=("ti7", "mean"),
                                    size=("thread_id", "size"),
                                    tool=("cclass", lambda s: s.iloc[0] == "tool"),
                                    reply=("type", lambda s: float(s.iloc[0] == "reply"))).reset_index()
    return ev.sort_values(["t", "thread_id"]).reset_index(drop=True)


def lee_features(ev, mode, GM, TE_SD, thr_g):
    load_posts, T6, c1 = lee_loaders()
    t = ev.t.values.astype("int64")
    f = fp.account_features(t, ev.ti7.values, ev.tool.values, ev.reply.values, (ev["size"].values >= 2).astype(float))
    md, cd, _ = c1.lc.load_vote_days(c1.ROOT / "data/voteview")
    d = pd.DataFrame(dict(t=t, when=pd.to_datetime(t, unit="s"), ti7=ev.ti7.values, size=ev["size"].values))
    for k in ["prev_ti7", "roll10", "roll50", "sched_share", "n_scored_prior", "prev_scored", "dt_prev",
              "prev_reply", "prev_multi", "reply_share10", "own_reply"]:
        d[k] = f[k]
    d["log_dt"] = np.log1p(f["dt_prev"])
    d["sess_pos"] = np.log1p(f["sess_pos"])
    d["sess_el"] = np.log1p(f["sess_el"])
    d["n1h"] = np.log1p(f["n1h"])
    d["n24h"] = np.log1p(f["n24h"])
    if mode == "archived":
        ml = c1.lc.MemberLocation(md[c1.LEE_ICPSR], cd["S"], "UT", name="Lee")
        h, dow, _ = ml.local_hour_dow(t, (c1.lc.PRIMARY,), ml.classes(t))[c1.lc.PRIMARY]
        weekend = (dow >= 5).astype(float)
        sc = ~np.isnan(ev.ti7.values)
        cs = np.concatenate([[0.0], np.cumsum(np.where(sc, ev.ti7.values, 0.0))])
        cn_ = np.concatenate([[0], np.cumsum(sc)])
        n_pr, s_pr = cn_[f["cnt"]], cs[f["cnt"]]
        eligible = d.ti7.notna() & d.prev_scored & (d.n_scored_prior >= m.MIN_PRIOR_SCORED)
    else:
        sys.path.insert(0, str(STRICT))
        import strict_features as sf
        lag = sf.repair_lags_strict(ev.assign(screen_name="lee")[["thread_id", "screen_name", "t", "t_end", "size", "ti7"]])
        lag = lag.set_index("thread_id").loc[ev.thread_id]
        for k in ("prev_ti7", "roll10", "roll50", "prev_multi"):
            d[k] = lag[k].to_numpy()
        d["n_completed_prior"] = lag.n_completed_prior.to_numpy()
        h, wk_, cls = sf.local_clock(t, md[c1.LEE_ICPSR], c1.lc.STATE_ZONE["UT"])
        weekend = np.asarray(wk_, dtype=float)
        d["strict_loc_class_transition"] = (cls == c1.lc.TRANSITION)
        # account encoding: expanding mean over closed (t_end + 7200 < t) scored events only
        order = np.lexsort((ev.thread_id.values, ev.t_end.values))
        cut = ev.t_end.values[order].astype("int64") + sf.CLOSURE_SECONDS
        cnt = np.searchsorted(cut, t, side="left")
        v = ev.ti7.values[order]
        sc = np.isfinite(v)
        n_pr = np.r_[0, np.cumsum(sc)][cnt]
        s_pr = np.r_[0.0, np.cumsum(np.where(sc, v, 0.0))][cnt]
        eligible = d.ti7.notna() & (d.n_completed_prior >= 10) & np.isfinite(d.prev_ti7)
    d["hour"] = np.asarray(h, dtype=float)
    ang = 2 * np.pi * d.hour / 24
    d["hsin"], d["hcos"], d["h2sin"], d["h2cos"] = np.sin(ang), np.cos(ang), np.sin(2 * ang), np.cos(2 * ang)
    d["night"] = (d.hour < 6).astype(float)
    d["weekend"] = weekend
    d["te_mean"] = np.where(n_pr >= 20, (s_pr + 20 * GM) / (n_pr + 20), GM)
    d["te_sd"] = TE_SD
    d["y_binb"] = (d.ti7 > thr_g).astype(float)
    return d[eligible.to_numpy()].reset_index(drop=True)


def week_skill(la, lb, wk, B=1000):
    keys, inv = np.unique(wk, return_inverse=True)
    pt, lo, hi = fp.skill_boot(np.bincount(inv, weights=la, minlength=len(keys)),
                               np.bincount(inv, weights=lb, minlength=len(keys)), B=B)
    return {"skill": pt, "ci95": [lo, hi], "n_weeks": int(len(keys))}


def run_lee(mode):
    hp, _ = frozen()
    if mode == "archived":
        F = pd.read_parquet(rg.P1 / "features.parquet")
        F["when"] = pd.to_datetime(F["when"])
        tr, _, J, thr_g = m.prepare_origin(F, fp.TEST_END, fp.TEST_END + pd.Timedelta(days=1))
    else:
        legacy, repaired, _ = load_frames()
        _, _, tr, _ = strict_select(legacy, repaired, fp.TEST_END, fp.TEST_END + pd.Timedelta(days=1))
        thr_g = float(tr.ti7.quantile(0.8))
        if tr.t_end.max() + 7200 >= int(fp.TEST_END.timestamp()):
            raise AssertionError("Lee training closure violated")
    GM, TE_SD = float(tr.ti7.mean()), float(tr.te_sd.mean())
    print(mode, "Lee training rows", len(tr), flush=True)
    G = {"P": m.fit_gbm(tr, "reg", "P", hp), "R": m.fit_gbm(tr, "reg", "R", hp, m.SETS_NOCONG)}
    for s in ("PT", "RT", "PTO", "RTO"):
        G[s] = m.fit_gbm(tr, "reg", s, hp, m.SETS_NOCONG)
    res = {"training": {"rows": len(tr), "accounts": int(tr.acct.nunique()), "global_mean": GM,
                        "global_threshold_80": thr_g, "te_sd_mean": TE_SD}}
    allp = []
    for which in ("personal", "official"):
        ev = lee_events(which)
        d = lee_features(ev, mode, GM, TE_SD, thr_g)
        y = d.ti7.to_numpy()
        wk = d.when.dt.to_period("W").astype(str).to_numpy()
        P = {"P": G["P"].predict(d[m.SETS["P"]])}
        for s in ("R", "PT", "RT", "PTO", "RTO"):
            P[s] = G[s].predict(d[m.SETS_NOCONG[s]])
        P["b_roll50"] = d.roll50.to_numpy()
        L = {k: (y - v) ** 2 for k, v in P.items()}
        r = {"n_events": int(len(d)), "first": str(d.when.min()), "last": str(d.when.max()),
             "mse": {k: float(v.mean()) for k, v in L.items()}}
        for a, b in (("R", "P"), ("RT", "PT"), ("RTO", "PTO"), ("PT", "P"), ("PTO", "PT"), ("P", "b_roll50")):
            r[f"{a}_vs_{b}"] = week_skill(L[a], L[b], wk)
        if mode == "strict":
            r["n_dc_transition_rows"] = int(d.strict_loc_class_transition.sum())
        res[which] = r
        allp.append(d.assign(account=which, pred_P=P["P"], pred_R=P["R"])[
            ["account", "when", "ti7", "prev_ti7", "roll50", "pred_P", "pred_R"]])
        print(mode, which, r["n_events"], r["R_vs_P"], r["RTO_vs_PTO"], flush=True)
    pd.concat(allp).to_parquet(HERE / f"lee_predictions_{mode}.parquet", index=False)
    (HERE / f"lee_{mode}.json").write_text(json.dumps(res, indent=2, default=float))


# ----------------------------------------------------------------------------- report
def boot_pairs(d, pairs, B=BOOT, seed=SEED):
    """Account-blocked bootstrap (aligned_summary scheme: one multinomial weight matrix, seed 20261010)."""
    y = d.ti7.to_numpy("float64")
    names = sorted({c for p in pairs for c in p})
    loss = pd.DataFrame({c: (y - d[c].to_numpy("float64")) ** 2 for c in names}).assign(acct=d.acct.to_numpy())
    g = loss.groupby("acct")[names].sum()
    sums = g.to_numpy()
    rng = np.random.default_rng(seed)
    w = rng.multinomial(len(sums), np.full(len(sums), 1 / len(sums)), size=B)
    tot = w @ sums
    idx = {c: i for i, c in enumerate(names)}
    out, draws = {}, {}
    for a, b in pairs:
        pt = 1 - loss[a].mean() / loss[b].mean()
        bs = 1 - tot[:, idx[a]] / tot[:, idx[b]]
        draws[(a, b)] = bs
        out[f"{a}_vs_{b}"] = {"skill_pct": 100 * float(pt), "ci95_pct": (100 * np.percentile(bs, [2.5, 97.5])).tolist(),
                              "mse_model": float(loss[a].mean()), "mse_reference": float(loss[b].mean()),
                              "n": int(len(d)), "accounts": int(len(sums))}
    return out, draws


def run_report():
    files = [HERE / "preds_monthly" / f"origin_{o:%Y-%m}.parquet" for o in rg.months()]
    d = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    if len(files) != 13 or d.thread_id.duplicated().any():
        raise AssertionError("13 monthly files, unique events required")
    lmm_files = [HERE / "preds_lmm" / f"origin_{o:%Y-%m}.parquet" for o in rg.months()]
    have_lmm = all(p.exists() for p in lmm_files)
    if have_lmm:
        lm = pd.concat([pd.read_parquet(p) for p in lmm_files], ignore_index=True)
        if not np.array_equal(lm.thread_id.to_numpy(), d.thread_id.to_numpy()):
            raise AssertionError("lmm rows misaligned")
        d["lmm_P"], d["lmm_R"] = lm.lmm_P.to_numpy(), lm.lmm_R.to_numpy()
    res = {"n_test_events": int(len(d)), "accounts": int(d.acct.nunique()), "origins": 13, "bootstrap": {
        "draws": BOOT, "seed": SEED, "unit": "account (all months of an account resampled together)"}}
    # reproduction gate: identical numbers to rq1_gbt_v2/aligned_evaluation.json
    ali = rg.aligned_summary(d.assign(aligned_p=d.m_P, aligned_r=d.m_R))
    ref = json.loads((GBT / "aligned_evaluation.json").read_text())
    res["reproduction"] = {"repaired_skill_pct": 100 * ali["models"]["repaired"]["skill"],
                           "repaired_ci95_pct": [100 * x for x in ali["models"]["repaired"]["ci95_skill"]],
                           "reference_skill_pct": 100 * ref["models"]["repaired"]["skill"],
                           "exact_match": ali["models"]["repaired"]["skill"] == ref["models"]["repaired"]["skill"]
                           and ali["models"]["repaired"]["ci95_skill"] == ref["models"]["repaired"]["ci95_skill"]}
    if not res["reproduction"]["exact_match"]:
        raise AssertionError("harness does not reproduce aligned headline")
    # panel a
    pairs_a = [("m_R", "m_P"), ("m_RT", "m_PT"), ("m_RTO", "m_PTO"), ("m_PT", "m_P"), ("m_PTO", "m_PT"),
               ("m_P", "roll50"), ("orig_r", "orig_p")]  # roll50 here is the corrected closure-aware roll50
    if have_lmm:
        pairs_a += [("lmm_R", "lmm_P")]
    a, dr = boot_pairs(d, pairs_a)
    diff = dr[("m_RT", "m_PT")] - dr[("m_R", "m_P")]
    a["rhythm_gain_change_with_prev_type_controls_pp"] = {
        "point_pp": a["m_RT_vs_m_PT"]["skill_pct"] - a["m_R_vs_m_P"]["skill_pct"],
        "ci95_pp": (100 * np.percentile(diff, [2.5, 97.5])).tolist()}
    diff2 = dr[("m_RTO", "m_PTO")] - dr[("m_R", "m_P")]
    a["rhythm_gain_change_with_prev_and_own_type_pp"] = {
        "point_pp": a["m_RTO_vs_m_PTO"]["skill_pct"] - a["m_R_vs_m_P"]["skill_pct"],
        "ci95_pp": (100 * np.percentile(diff2, [2.5, 97.5])).tolist()}
    res["panel_a"] = a
    # panel b
    b = {}
    for var, lev in [("party", "D"), ("party", "R"), ("chamber", "house"), ("chamber", "senate"),
                     ("account_type", "office"), ("account_type", "campaign"), ("sched", "False"), ("sched", "True")]:
        s = d[d[var].astype(str) == lev]
        r, _ = boot_pairs(s, [("m_R", "m_P")])
        b[f"{var}={lev}"] = r["m_R_vs_m_P"]
    res["panel_b"] = b
    # panel c: persistence + one block vs persistence (13 monthly origins)
    c, _ = boot_pairs(d, [(col(k), "m_P") for k in ABL])
    res["panel_c_monthly"] = {k: c[f"{col(k)}_vs_m_P"] for k in ABL}
    # panel c secondary: archived single-origin procedure on corrected inputs
    fo = HERE / "preds_final_origin.parquet"
    if fo.exists():
        f = pd.read_parquet(fo)
        cf, _ = boot_pairs(f, [(col(k), "m_P") for k in ABL] + [("m_R", "m_P")])
        res["panel_c_single_final_origin"] = {k: cf[f"{col(k)}_vs_m_P"] for k in ABL}
        res["panel_c_single_final_origin"]["R (all rhythm)"] = cf["m_R_vs_m_P"]
        res["final_origin_info"] = json.loads((HERE / "final_origin_info.json").read_text())
    # risk curves
    pdp_p = HERE / "pdp_strict_v2.csv"
    if pdp_p.exists():
        res["risk_curves"] = risk_statements(pd.read_csv(pdp_p), pd.read_csv(HERE / "empirical_bins_strict_v2.csv"),
                                             pd.read_csv(rg.P1 / "pdp.csv"))
    for mode in ("archived", "strict"):
        p = HERE / f"lee_{mode}.json"
        if p.exists():
            res[f"lee_{mode}_inputs"] = json.loads(p.read_text())
    res["monthly_checks"] = json.loads((HERE / "monthly_checks.json").read_text())
    (HERE / "results.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({k: v for k, v in res.items() if k not in ("monthly_checks",)}, indent=1, default=float)[:20000])


def risk_statements(pdp, eb, pdp_arch):
    out = {}
    for tag, P in (("strict_v2", pdp), ("archived", pdp_arch)):
        q = P[P.outcome == "p_high_acct80"]
        dt = q[q.feature == "dt_s"].set_index("value").pdp
        sp = q[q.feature == "sess_pos"].set_index("value").pdp
        hr = q[q.feature == "local_hour"].set_index("value").pdp
        within_min = dt[dt.index <= 60]
        out[tag] = {
            "pdp_dt": dt.round(5).to_dict(), "pdp_sess_pos": sp.round(5).to_dict(), "pdp_hour": hr.round(5).to_dict(),
            "S1_within_minute_lowest": bool(within_min.max() <= dt[dt.index > 60].min()),
            "S2_rises_after_3min": bool(dt.loc[180] > within_min.max()),
            "S3_falls_deeper_into_sessions": bool(sp.loc[20] < sp.loc[0] and sp.iloc[-1] == sp.min()),
            "S3_monotone_nonincreasing": bool((np.diff(sp.to_numpy()) <= 0).all()),
            "S4_peak_hour": int(hr.idxmax()), "S4_late_evening_peak": bool(hr.idxmax() in (20, 21, 22, 23)),
        }
    e = eb[(eb.outcome == "high_acct80")]
    out["strict_v2_empirical_high_acct80"] = {f: e[e.feature == f][["bin", "n", "mean", "lo", "hi"]].round(5).to_dict("records")
                                              for f in ("dt_s", "sess_pos", "local_hour")}
    return out


# ----------------------------------------------------------------------------- figure
def run_figure():
    sys.path.insert(0, str(rg.SCRIPT))
    import figs_common as fc
    fc.style()
    plt = fc.plt
    plt.rcParams["font.family"] = "Arial"  # requested house font (figs_common default is DejaVu Sans)
    R = json.loads((HERE.parent / "rq1_strict_v2_fig5" / "results.json").read_text())
    a, b, c = R["panel_a"], R["panel_b"], R["panel_c_monthly"]
    fig = plt.figure(figsize=(fc.W2, 3.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.95)
    ax = fig.add_subplot(gs[0])
    rowsA = [("Rhythm added", "m_R_vs_m_P"),
             ("Rhythm added, both\nmodels also given\nprevious-event type", "m_RT_vs_m_PT"),
             ("Rhythm added, both\nalso given first-post\nreply status", "m_RTO_vs_m_PTO")]
    if "lmm_R_vs_lmm_P" in a:
        rowsA.append(("Rhythm added,\nmixed model", "lmm_R_vs_lmm_P"))
    for i, (lab, k) in enumerate(rowsA):
        r = a[k]
        lo, hi = r["ci95_pct"]
        ax.errorbar(r["skill_pct"], i, xerr=[[r["skill_pct"] - lo], [hi - r["skill_pct"]]], fmt="o",
                    ms=5 if i == 0 else 4, color=fc.BLACK if i == 0 else fc.GREY, lw=1.3, capsize=0)
    ax.axvline(0, color="black", lw=0.7)
    ax.set_yticks(range(len(rowsA))); ax.set_yticklabels([r[0] for r in rowsA], fontsize=7); ax.invert_yaxis()
    ax.set_xlabel("MSE skill over content\nhistory (%)")
    fc.plab(ax, "a", dx=-0.9)
    ax = fig.add_subplot(gs[1])
    order = [("party=D", "Democrat", fc.D_COL), ("party=R", "Republican", fc.R_COL), ("chamber=house", "House", fc.GREY),
             ("chamber=senate", "Senate", fc.GREY), ("account_type=office", "Office", fc.GREY),
             ("account_type=campaign", "Campaign", fc.GREY), ("sched=False", "Not scheduled-\nlooking", fc.GREY),
             ("sched=True", "Scheduled-\nlooking", fc.GREY)]
    y = 0; yt = []
    for i, (k, lab, colr) in enumerate(order):
        r = b[k]
        lo, hi = r["ci95_pct"]
        ax.errorbar(r["skill_pct"], y, xerr=[[r["skill_pct"] - lo], [hi - r["skill_pct"]]], fmt="o", ms=4, color=colr, lw=1.2, capsize=0)
        yt.append((y, lab)); y += 1 if i % 2 == 0 else 1.5
    ax.axvline(a["m_R_vs_m_P"]["skill_pct"], color=fc.BLACK, lw=0.8, ls="--")
    ax.axvline(0, color="black", lw=0.7)
    ax.set_yticks([p for p, _ in yt]); ax.set_yticklabels([q for _, q in yt]); ax.invert_yaxis()
    ax.set_xlabel("Rhythm MSE skill over\ncontent history (%)")
    fc.plab(ax, "b", dx=-0.7)
    ax = fig.add_subplot(gs[2])
    rename = {"own timing": "Own timing", "time of day": "Time of day", "scheduler share": "Prior management-\nclient use",
              "lagged Congress volume": "Lagged Congress-\nwide activity", "regime flags": "Calendar indicators",
              "previous post type": "Previous event type", "own post type (first post)": "First post is a\nreply (diagnostic)",
              "rhythm minus Congress": "Rhythm without\nCongress-wide activity"}
    comp = sorted(c.items(), key=lambda kv: -kv[1]["skill_pct"])
    for i, (k, r) in enumerate(comp):
        lo, hi = r["ci95_pct"]
        diag = k.startswith("own post type")
        ax.errorbar(r["skill_pct"], i, xerr=[[r["skill_pct"] - lo], [hi - r["skill_pct"]]], fmt="o", ms=4,
                    color=fc.GREY if diag else fc.BLUE, mfc="white" if diag else fc.BLUE, lw=1.2, capsize=0)
    ax.axvline(0, color="black", lw=0.7)
    ax.set_yticks(range(len(comp))); ax.set_yticklabels([rename.get(k, k) for k, _ in comp], fontsize=7.5); ax.invert_yaxis()
    ax.set_xlabel("MSE skill over content\nhistory (%)")
    fc.plab(ax, "c", dx=-0.9)
    fig.savefig(HERE / "fig5_v3.pdf", bbox_inches="tight")
    fig.savefig(HERE / "fig5_v3.png", dpi=300, bbox_inches="tight")
    print("saved fig5_strict_v2")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["monthly", "lmm", "final", "lee", "report", "figure"])
    ap.add_argument("--mode", choices=["archived", "strict"], default="strict")
    args = ap.parse_args()
    t0 = time.time()
    {"monthly": run_monthly, "lmm": run_lmm, "final": run_final, "lee": lambda: run_lee(args.mode),
     "report": run_report, "figure": run_figure}[args.stage]()
    print("stage", args.stage, "seconds", round(time.time() - t0), flush=True)
