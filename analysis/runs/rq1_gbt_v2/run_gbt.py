"""Isolated full-corpus P1 LightGBM reproduction and as-of sensitivity.

Run with PYTHONPATH=~/x_box/pyextra ~/x_box/venv/bin/python run_gbt.py
--stage original|sensitivity|report|manifest in that order. Never tunes.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/v7"


HERE = Path(__file__).resolve().parent
P1 = ROOT / "output/v7/p1"
RQ1 = ROOT / "manuscript/epjds_submission_work/rq1"
RQ1_STRICT = ROOT / "manuscript/epjds_submission_work/rq1_strict_v2"
assert ROOT.name == "pipeline" and HERE == RQ1.parent / "rq1_gbt_v2"
SEED = 20261010


def original_models():
    sys.path.insert(0, str(SCRIPT))
    import p1_models
    return p1_models


LAGS = ("prev_ti7", "roll10", "roll50")


ALIGNED_COLS = LAGS + ("hour", "hsin", "hcos", "h2sin", "h2cos", "night", "weekend",
                       "cong_p1", "cong_p24", "cong_spike_lr", "cong_spike")


def aligned_repaired_frames(original, strict):
    """Read-only strict-ridge feature source, P1 learner and exact shared row order."""
    if not original.thread_id.reset_index(drop=True).equals(strict.thread_id.reset_index(drop=True)):
        raise AssertionError("aligned P1/strict-ridge feature IDs differ")
    if not np.allclose(original.ti7, strict.ti7, rtol=0, atol=2e-6):
        raise AssertionError("aligned P1/strict-ridge target labels differ")
    legacy = original.copy()
    legacy["t_end"] = strict.t_end.to_numpy(dtype="int64")
    legacy["strict_n_completed_prior"] = strict.strict_n_completed_prior.to_numpy(dtype="int32")
    repaired = legacy.copy()
    for col in ALIGNED_COLS:
        repaired[col] = strict["strict_" + col].to_numpy(dtype="float32")
    return legacy, repaired


def select_aligned_origin(legacy, repaired, origin):
    """Exact strict-ridge test/train cohort, then original P1 target encoding."""
    m = original_models()
    eligible = legacy.strict_n_completed_prior.to_numpy() >= 10
    if not np.array_equal(eligible, repaired.strict_n_completed_prior.to_numpy() >= 10):
        raise AssertionError("repaired cohort eligibility differs")
    return prepare_matched(legacy.loc[eligible], repaired.loc[eligible], origin,
                           m.window_end(origin), embargo=7200)


def embargo_lags(events, seconds=7200):
    """Completed-event score history only after a strict closure embargo."""
    chunks = []
    for _, g in events.groupby("screen_name", sort=False):
        starts = g.sort_values(["t", "thread_id"], kind="stable")
        done = g.sort_values(["t_end", "thread_id"], kind="stable")
        cutoff = done.t_end.to_numpy(dtype="int64") + seconds
        count = np.searchsorted(cutoff, starts.t.to_numpy(dtype="int64"), side="left")
        vals = done.ti7.to_numpy(dtype="float64")
        scored = np.isfinite(vals)
        n = np.r_[0, scored.cumsum()][count]
        sv = vals[scored]
        cs = np.r_[0., sv.cumsum()]
        res = starts[["thread_id"]].copy()
        res["prev_ti7"] = np.where(n > 0, sv[np.maximum(n - 1, 0)] if len(sv) else np.nan, np.nan)
        for k in (10, 50):
            lo = np.maximum(n - k, 0)
            res[f"roll{k}"] = np.where(n > 0, (cs[n] - cs[lo]) / np.maximum(n - lo, 1), np.nan)
        chunks.append(res)
    return pd.concat(chunks, ignore_index=True)


def prior_vote_dc(et_day, vote_days, min_votes=20):
    """Only prior Eastern-date member votes; no same-day/date-only or future votes."""
    day = np.asarray(et_day, dtype="int64")
    vd = np.asarray(vote_days, dtype="int64")
    i = np.searchsorted(vd, day, side="left")
    if not len(vd):
        return np.zeros(len(day), dtype=bool)
    prev = vd[np.maximum(i - 1, 0)]
    weekday = (day + 3) % 7 < 5
    return (i >= min_votes) & weekday & (day - prev == 1)


def local_hour_for_member(t, vote_days, state, lc, min_votes=20):
    """Conservative date-only prior-vote location; fixed state home zone.

    Split-state districts intentionally fall back to the state zone; no
    retrospective choice weighted by future posting dates is permitted.
    """
    h_dc, dow_dc, et_day, _, _ = lc.to_local(t, lc.ET)
    zone = lc.STATE_ZONE[state]
    if zone == lc.ET:
        return h_dc, dow_dc
    h_home, dow_home, _, _, _ = lc.to_local(t, zone)
    observed = np.searchsorted(np.asarray(vote_days, dtype="int64"), et_day, side="left")
    use_dc = prior_vote_dc(et_day, vote_days, min_votes) | (observed < min_votes)
    return np.where(use_dc, h_dc, h_home), np.where(use_dc, dow_dc, dow_home)


def strict_features(original, corrected):
    """Two nested designs on identical row order: closure lags, then vote-safe hours."""
    if not original.thread_id.reset_index(drop=True).equals(corrected.thread_id.reset_index(drop=True)):
        raise AssertionError("strict feature alignment differs")
    lag = original.copy()
    for c in LAGS:
        lag[c] = corrected[c].to_numpy(dtype="float32")
    vote = lag.copy()
    vote["hour"] = corrected.hour.to_numpy(dtype="float32")
    vote["weekend"] = corrected.weekend.to_numpy(dtype="float32")
    angle = 2 * np.pi * vote.hour.to_numpy() / 24
    for name, arr in (("hsin", np.sin(angle)), ("hcos", np.cos(angle)),
                      ("h2sin", np.sin(2 * angle)), ("h2cos", np.cos(2 * angle)),
                      ("night", (vote.hour.to_numpy() < 6).astype(float))):
        vote[name] = arr.astype("float32")
    return lag, vote


def attach_arrival_lags(original, arrival):
    """Index-aligned read-only audit; never replace labels or rhythm features."""
    if not original.thread_id.reset_index(drop=True).equals(arrival.thread_id.reset_index(drop=True)):
        raise AssertionError("arrival/original thread alignment differs")
    for col in LAGS:
        if not np.allclose(original[col], arrival["naive_" + col], rtol=0, atol=0, equal_nan=True):
            raise AssertionError("arrival/original naive lag mismatch: " + col)
    naive = original.copy()
    naive["t_end"] = arrival.t_end.to_numpy()
    safe = naive.copy()
    for col in LAGS:
        safe[col] = arrival[col].to_numpy()
    return naive, safe


def paired_summary(d, boot=2000, seed=20261010):
    """Account-blocked, paired CI; each resample retains all months per account."""
    if d.empty:
        raise ValueError("empty prediction set")
    cols = [f"{branch}_{kind}" for branch in ("orig", "cut", "safe") for kind in ("p", "r")]
    y = d.ti7.to_numpy(dtype="float64")
    losses = np.column_stack([(y - d[c].to_numpy(dtype="float64")) ** 2 for c in cols])
    sums = pd.DataFrame(losses, columns=cols).assign(acct=d.acct.to_numpy()).groupby("acct", sort=True)[cols].sum().to_numpy()
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(sums), np.full(len(sums), 1 / len(sums)), size=boot)
    bl = counts @ sums
    means = losses.mean(axis=0)
    out = {"n": len(d), "accounts": len(sums)}
    skill = {}
    bs = {}
    for i, branch in enumerate(("original", "cutoff", "asof")):
        k = 2 * i
        a, b = means[k], means[k + 1]
        skill[branch] = 1 - b / a
        bs[branch] = 1 - np.divide(bl[:, k + 1], bl[:, k],
                                    out=np.full(boot, np.nan), where=bl[:, k] > 0)
        valid = bs[branch][np.isfinite(bs[branch])]
        if not len(valid):
            raise ValueError("bootstrap has no nonzero persistence loss")
        out[branch] = {"mse_p": float(a), "mse_r": float(b), "delta_mse": float(a - b),
                       "skill": float(skill[branch]),
                       "ci95_skill": np.percentile(valid, [2.5, 97.5]).tolist()}
    differences = {"asof_minus_original": ("asof", "original"),
                   "asof_minus_cutoff": ("asof", "cutoff"),
                   "cutoff_minus_original": ("cutoff", "original")}
    out["paired_skill_change"] = {name: float(skill[a] - skill[b]) for name, (a, b) in differences.items()}
    out["ci95_paired_skill_change"] = {name: np.percentile((bs[a] - bs[b])[np.isfinite(bs[a] - bs[b])],
                                                               [2.5, 97.5]).tolist()
                                        for name, (a, b) in differences.items()}
    out["paired_mse_difference"] = {"asof_r_minus_original_r": float(means[5] - means[1]),
                                    "asof_r_minus_cutoff_r": float(means[5] - means[3])}
    return out


def month_block_skill_change(d, boot=2000, seed=SEED):
    """Secondary paired resampling of intact monthly origin windows."""
    cols = ("orig_p", "orig_r", "aligned_p", "aligned_r")
    y = d.ti7.to_numpy(dtype="float64")
    losses = pd.DataFrame({c: (y - d[c].to_numpy(dtype="float64")) ** 2 for c in cols})
    sums = losses.assign(origin=d.origin.to_numpy()).groupby("origin")[list(cols)].sum().to_numpy()
    if len(sums) < 2:
        raise AssertionError("at least two month origins required")
    total = sums.sum(axis=0)
    point = (1 - total[3] / total[2]) - (1 - total[1] / total[0])
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(sums), np.full(len(sums), 1 / len(sums)), size=boot)
    sampled = weights @ sums
    draws = (1 - sampled[:, 3] / sampled[:, 2]) - (1 - sampled[:, 1] / sampled[:, 0])
    return {"n_origins": len(sums), "paired_skill_change": float(point),
            "ci95_paired_skill_change": np.percentile(draws, [2.5, 97.5]).tolist()}


def aligned_summary(d, boot=2000, seed=SEED):
    """Paired original-vs-repaired GBM on only the strict closed-history rows."""
    y = d.ti7.to_numpy(dtype="float64")
    cols = ("orig_p", "orig_r", "aligned_p", "aligned_r")
    loss = np.column_stack([(y - d[c].to_numpy(dtype="float64")) ** 2 for c in cols])
    groups = pd.DataFrame(loss, columns=cols).assign(acct=d.acct.to_numpy()).groupby("acct")
    sums = groups[list(cols)].sum().to_numpy()
    sizes = groups.size().to_numpy()
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(sums), np.full(len(sums), 1 / len(sums)), size=boot)
    totals = weights @ sums
    ns = weights @ sizes
    models = {}
    for name, k in (("legacy_matched", 0), ("repaired", 2)):
        p, r = float(loss[:, k].mean()), float(loss[:, k + 1].mean())
        models[name] = {"mse_p": p, "mse_r": r, "delta_mse": p - r,
                        "skill": 1 - r / p,
                        "ci95_skill": np.percentile(1 - totals[:, k + 1] / totals[:, k], [2.5, 97.5]).tolist(),
                        "ci95_delta_mse": np.percentile((totals[:, k] - totals[:, k + 1]) / ns,
                                                        [2.5, 97.5]).tolist()}
    skill_difference = (1 - totals[:, 3] / totals[:, 2]) - (1 - totals[:, 1] / totals[:, 0])
    delta_difference = ((totals[:, 2] - totals[:, 3]) - (totals[:, 0] - totals[:, 1])) / ns
    return {"n": len(d), "accounts": len(sums), "models": models,
            "paired_skill_change_repaired_minus_legacy": models["repaired"]["skill"] - models["legacy_matched"]["skill"],
            "ci95_paired_skill_change": np.percentile(skill_difference, [2.5, 97.5]).tolist(),
            "paired_delta_mse_change_repaired_minus_legacy": models["repaired"]["delta_mse"] - models["legacy_matched"]["delta_mse"],
            "ci95_paired_delta_mse_change": np.percentile(delta_difference, [2.5, 97.5]).tolist()}


def strict_summary(d, boot=2000, seed=SEED):
    """Nested ablations with identical account resamples across all designs."""
    names = ("original", "cutoff", "asof", "embargo", "strict")
    prefixes = ("orig", "cut", "safe", "emb", "strict")
    y = d.ti7.to_numpy(dtype="float64")
    cols = [f"{tag}_{s}" for tag in prefixes for s in ("p", "r")]
    loss = np.column_stack([(y - d[c].to_numpy(dtype="float64")) ** 2 for c in cols])
    groups = pd.DataFrame(loss, columns=cols).assign(acct=d.acct.to_numpy()).groupby("acct")
    sums = groups[cols].sum().to_numpy()
    sizes = groups.size().to_numpy()
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(sums), np.full(len(sums), 1 / len(sums)), size=boot)
    totals = weights @ sums
    sampled_n = weights @ sizes
    means = loss.mean(axis=0)
    models, samples = {}, {}
    for i, name in enumerate(names):
        j = 2 * i
        sk = 1 - means[j + 1] / means[j]
        bs = 1 - totals[:, j + 1] / totals[:, j]
        models[name] = {"mse_p": float(means[j]), "mse_r": float(means[j + 1]),
                        "delta_mse": float(means[j] - means[j + 1]), "skill": float(sk),
                        "ci95_delta_mse": np.percentile((totals[:, j] - totals[:, j + 1]) / sampled_n,
                                                        [2.5, 97.5]).tolist(),
                        "ci95_skill": np.percentile(bs, [2.5, 97.5]).tolist()}
        samples[name] = bs
    pairs = (("cutoff", "original"), ("asof", "cutoff"), ("embargo", "asof"),
             ("strict", "embargo"), ("strict", "original"))
    labels = {("strict", "embargo"): "vote_minus_embargo"}
    return {"n": len(d), "accounts": len(sums), "models": models,
            "paired_skill_change": {labels.get((a, b), f"{a}_minus_{b}"):
                                    models[a]["skill"] - models[b]["skill"] for a, b in pairs},
            "paired_r_mse_difference": {labels.get((a, b), f"{a}_minus_{b}"):
                                        models[a]["mse_r"] - models[b]["mse_r"] for a, b in pairs},
            "ci95_paired_r_mse_difference": {labels.get((a, b), f"{a}_minus_{b}"):
                                             np.percentile((totals[:, 2 * names.index(a) + 1] -
                                                            totals[:, 2 * names.index(b) + 1]) / sampled_n,
                                                           [2.5, 97.5]).tolist()
                                             for a, b in pairs},
            "ci95_paired_skill_change": {labels.get((a, b), f"{a}_minus_{b}"):
                                         np.percentile(samples[a] - samples[b], [2.5, 97.5]).tolist()
                                         for a, b in pairs}}


def compare_with_ridge(gbt, ridge, boot=2000, seed=SEED):
    """Paired account CI for the *descriptive* cross-estimator skill gap."""
    if (len(gbt) != len(ridge) or gbt.thread_id.duplicated().any() or ridge.thread_id.duplicated().any()
            or not gbt.thread_id.isin(ridge.thread_id).all()):
        raise AssertionError("GBM/ridge row IDs do not match")
    ordered = ridge.set_index("thread_id").loc[gbt.thread_id]
    if not np.array_equal(gbt.acct.to_numpy(), ordered.acct.to_numpy()):
        raise AssertionError("GBM/ridge accounts do not match")
    if not np.allclose(gbt.ti7.to_numpy(), ordered.ti7.to_numpy(), rtol=0, atol=1e-6):
        raise AssertionError("GBM/ridge labels do not match")
    y = gbt.ti7.to_numpy(dtype="float64")
    losses = np.column_stack([(y - v) ** 2 for v in (
        gbt.safe_p.to_numpy(dtype="float64"), gbt.safe_r.to_numpy(dtype="float64"),
        ordered.content.to_numpy(dtype="float64"), ordered.rhythm.to_numpy(dtype="float64"))])
    sums = pd.DataFrame(losses, columns=list("abcd")).assign(acct=gbt.acct.to_numpy()).groupby("acct")[["a", "b", "c", "d"]].sum().to_numpy()
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(sums), np.full(len(sums), 1 / len(sums)), size=boot)
    totals = counts @ sums
    sk_g = 1 - losses[:, 1].sum() / losses[:, 0].sum()
    sk_r = 1 - losses[:, 3].sum() / losses[:, 2].sum()
    sampled_gap = totals[:, 3] / totals[:, 2] - totals[:, 1] / totals[:, 0]
    return {"n": len(gbt), "accounts": len(sums), "gbm_skill": float(sk_g),
            "ridge_skill": float(sk_r), "gap_gbm_minus_ridge": float(sk_g - sk_r),
            "ci95_gap_gbm_minus_ridge": np.percentile(sampled_gap, [2.5, 97.5]).tolist()}


def summarize_predictions(full, matched, boot=2000):
    """Never conflate the published full-row skill with a matched-row effect."""
    copy = full.copy()
    for tag in ("cut", "safe"):
        for s in ("p", "r"):
            copy[f"{tag}_{s}"] = copy[f"orig_{s}"]
    overall = paired_summary(copy, boot=boot)
    return {"original_full": {"n": overall["n"], "accounts": overall["accounts"],
                               **overall["original"]},
            "matched": paired_summary(matched, boot=boot)}


def aligned_month(legacy, repaired, origin, hp, original_rows):
    """Same closed-history cohort and repaired features as strict ridge; P1 GBM."""
    tr, te, tr_rep, te_rep = select_aligned_origin(legacy, repaired, origin)
    if tr.empty or te.empty or tr.t_end.max() + 7200 >= int(origin.timestamp()):
        raise AssertionError("aligned training closure/eligibility violation")
    if not tr.thread_id.equals(tr_rep.thread_id) or not te.thread_id.equals(te_rep.thread_id):
        raise AssertionError("aligned P/R rows differ")
    old = original_rows.set_index("thread_id")
    if old.index.has_duplicates or not te.thread_id.isin(old.index).all():
        raise AssertionError("aligned test rows missing from original P1")
    pred = fit_p_r(tr_rep, te_rep, hp)
    out = te[["thread_id", "acct", "when", "ti7", "t_end"]].copy()
    out["origin"] = origin
    for c in ("orig_p", "orig_r"):
        out[c] = old.loc[te.thread_id, c].to_numpy()
    for s in ("P", "R"):
        out[f"aligned_{s.lower()}"] = pred[s].astype("float32")
    if not np.isfinite(out[["aligned_p", "aligned_r"]]).all().all():
        raise AssertionError("nonfinite aligned GBM predictions")
    return out, {"n_train": len(tr), "n_test": len(te), "accounts": int(tr.acct.nunique()),
                 "n_test_excluded_vs_legacy": len(original_rows) - len(te),
                 "n_changed_hour_test": int((te.hour.to_numpy() != te_rep.hour.to_numpy()).sum()),
                 "n_changed_congress_spike_lr_test": int((~np.isclose(te.cong_spike_lr,
                     te_rep.cong_spike_lr, atol=1e-6, rtol=0, equal_nan=True)).sum())}


def strict_month(naive, lag, vote, origin, hp, previous_rows):
    """Closure-embargoed lag-only pair, then vote-safe R on same rows."""
    m = original_models()
    tr, te, tr_lag, te_lag = prepare_matched(naive, lag, origin, m.window_end(origin), embargo=7200)
    if tr.empty or te.empty or tr.t_end.max() + 7200 >= int(origin.timestamp()):
        raise AssertionError("uncertified/empty embargoed training set")
    tr_vote, te_vote = vote.loc[tr.index].copy(), vote.loc[te.index].copy()
    for src, dst in ((tr_lag, tr_vote), (te_lag, te_vote)):
        if not src.thread_id.equals(dst.thread_id) or not src.ti7.equals(dst.ti7):
            raise AssertionError("vote-safe and lag-only rows disagree")
        for c in ("ccode", "thr_a", "y_bina", "y_binb", "te_mean", "te_sd"):
            dst[c] = src[c].to_numpy()
    emb = fit_p_r(tr_lag, te_lag, hp)
    vote_r = m.fit_gbm(tr_vote, "reg", "R", hp).predict(te_vote[m.SETS["R"]])
    old = previous_rows.set_index("thread_id")
    if old.index.has_duplicates or not te.thread_id.isin(old.index).all():
        raise AssertionError("embargoed rows not in prior matched test set")
    r = te[["thread_id", "acct", "when", "ti7", "t_end"]].copy()
    r["origin"] = origin
    for c in ("orig_p", "orig_r", "cut_p", "cut_r", "safe_p", "safe_r"):
        r[c] = old.loc[te.thread_id, c].to_numpy()
    r["emb_p"] = emb["P"].astype("float32")
    r["emb_r"] = emb["R"].astype("float32")
    r["strict_p"] = r.emb_p.to_numpy()
    r["strict_r"] = vote_r.astype("float32")
    changed_hour = te_vote.hour.to_numpy() != te_lag.hour.to_numpy()
    r["changed_hour"] = changed_hour
    if not np.isfinite(r[["emb_p", "emb_r", "strict_p", "strict_r"]]).all().all():
        raise AssertionError("non-finite strict predictions")
    return r, {"n_train": len(tr), "n_test": len(te), "accounts": int(tr.acct.nunique()),
               "n_excluded_test_from_previous": len(previous_rows) - len(te),
               "n_changed_hour_test": int(changed_hour.sum()),
               "n_changed_lag_test": int(np.any(~np.isclose(te[list(LAGS)], te_lag[list(LAGS)],
                                                      rtol=0, atol=1e-6, equal_nan=True), axis=1).sum())}


def matched_month(naive, safe, origin, hp, original_rows):
    """Train cutoff-only and repaired-lag P/R on precisely the same rows."""
    m = original_models()
    tr, te, tr_safe, te_safe = prepare_matched(naive, safe, origin, m.window_end(origin))
    if tr.empty or te.empty or tr.t_end.max() >= int(origin.timestamp()):
        raise AssertionError("empty/unfinished matched origin")
    if not np.array_equal(tr.thread_id, tr_safe.thread_id) or not np.array_equal(te.thread_id, te_safe.thread_id):
        raise AssertionError("matched training/test IDs disagree")
    old = original_rows.set_index("thread_id")
    if old.index.has_duplicates or not te.thread_id.isin(old.index).all():
        raise AssertionError("matched test rows missing in original P1")
    p_cut = fit_p_r(tr, te, hp)
    p_safe = fit_p_r(tr_safe, te_safe, hp)
    r = te[["thread_id", "acct", "when", "ti7", "t_end"]].copy()
    r["origin"] = origin
    r["orig_p"] = old.loc[te.thread_id, "orig_p"].to_numpy()
    r["orig_r"] = old.loc[te.thread_id, "orig_r"].to_numpy()
    for s in ("P", "R"):
        r[f"cut_{s.lower()}"] = p_cut[s].astype("float32")
        r[f"safe_{s.lower()}"] = p_safe[s].astype("float32")
    changed_tr = np.any(~np.isclose(tr[list(LAGS)], tr_safe[list(LAGS)], equal_nan=True, rtol=0, atol=1e-6), axis=1)
    changed_te = np.any(~np.isclose(te[list(LAGS)], te_safe[list(LAGS)], equal_nan=True, rtol=0, atol=1e-6), axis=1)
    r["changed_lag"] = changed_te
    if not np.isfinite(r[["orig_p", "orig_r", "cut_p", "cut_r", "safe_p", "safe_r"]]).all().all():
        raise AssertionError("non-finite predictions")
    return r, {"n_train": len(tr), "n_test": len(te), "accounts": tr.acct.nunique(),
               "n_changed_train_lags": int(changed_tr.sum()), "n_changed_test_lags": int(changed_te.sum())}


def original_month(frame, origin, hp, archived):
    """Fit P1 exactly as originally written, verify row-level archived outputs."""
    m = original_models()
    stop = m.window_end(origin)
    tr, te, j, _ = m.prepare_origin(frame, origin, stop)
    if tr.when.max() >= origin or te.when.min() < origin or te.when.max() >= stop:
        raise AssertionError("original temporal origin violation")
    preds = fit_p_r(tr, te, hp)
    out = te[["thread_id", "acct", "when", "ti7"]].copy()
    for s in ("P", "R"):
        out[f"orig_{s.lower()}"] = preds[s].astype("float32")
    audit = verify_archive(out, archived)
    return out, {"n_train": len(tr), "n_test": len(te), "accounts": j, "archive": audit}


def verify_archive(current, archived, tolerance=2e-6):
    """Assert the refit matches persisted P1 predictions, not just aggregate skill."""
    if not np.array_equal(current.thread_id.to_numpy(), archived.thread_id.to_numpy()):
        raise AssertionError("original rows differ from archive")
    if not np.array_equal(current.ti7.to_numpy(), archived.ti7.to_numpy()):
        raise AssertionError("original labels differ from archive")
    diffs = {s: float(np.max(np.abs(current[f"orig_{s.lower()}"].to_numpy() -
                                     archived[f"gbm_reg_{s}"].to_numpy()))) for s in ("P", "R")}
    if max(diffs.values()) > tolerance:
        raise AssertionError(f"original prediction mismatch: {diffs}")
    return {"max_abs_prediction_error": max(diffs.values()), "per_model": diffs,
            "n": len(current)}


def fit_p_r(tr, te, hp):
    """Delegate exactly to frozen P1 feature sets, seed, cap, LightGBM recipe."""
    m = original_models()
    return {s: m.fit_gbm(tr, "reg", s, hp).predict(te[m.SETS[s]]) for s in ("P", "R")}


def prepare_matched(naive, safe, origin, end, min_train=100, embargo=0):
    """Same completed-label rows, account codes and target encoding for both fits."""
    m = original_models()
    tr, te = select_with_cutoff(naive, origin, end, min_train, embargo)
    if not tr.index.equals(safe.loc[tr.index].index) or not te.index.equals(safe.loc[te.index].index):
        raise AssertionError("as-of frame index mismatch")
    tr_safe, te_safe = safe.loc[tr.index].copy(), safe.loc[te.index].copy()
    for a, b in ((tr, tr_safe), (te, te_safe)):
        if not a.thread_id.equals(b.thread_id) or not a.ti7.equals(b.ti7):
            raise AssertionError("matched row/label alignment failed")
    cmap = {acct: i for i, acct in enumerate(sorted(tr.acct.unique()))}
    j = len(cmap)
    for d in (tr, te):
        d["ccode"] = d.acct.map(cmap).astype("int32")
    m.add_labels(tr, te)
    a_tr, a_te = tr.acct.to_numpy(copy=True), te.acct.to_numpy(copy=True)
    tr["acct"], te["acct"] = tr.ccode.to_numpy(), te.ccode.to_numpy()
    m.add_te(tr, te, j)
    tr["acct"], te["acct"] = a_tr, a_te
    for src, dst in ((tr, tr_safe), (te, te_safe)):
        for c in ("ccode", "thr_a", "y_bina", "y_binb", "te_mean", "te_sd"):
            dst[c] = src[c].to_numpy()
    return tr, te, tr_safe, te_safe


def select_with_cutoff(frame, origin, end, min_train=100, embargo=0):
    """Completed-label training at a monthly origin; same arrival test universe."""
    tr = frame[(frame.when < origin) & (frame.t_end + embargo < int(origin.timestamp()))]
    te = frame[(frame.when >= origin) & (frame.when < end)]
    counts = tr.acct.value_counts()
    ok = counts.index[counts >= min_train]
    return tr[tr.acct.isin(ok)].copy(), te[te.acct.isin(ok)].copy()


def build_strict_artifact():
    """Rebuild all-event embargoed lags and previous-day-only local hours."""
    original_models()  # imports P1 shared module, adding read-only pipeline root to sys.path
    from xbox import location as lc
    sys.path.insert(0, str(RQ1))
    from rq1_arrival import score_events
    f = pd.read_parquet(P1 / "features.parquet",
                        columns=["thread_id", "screen_name", "bioguide", "t", "ti7", "hour", "weekend"])
    source = ROOT / "output/v7/t1full"
    ev = pd.read_parquet(source / "events.parquet",
                         columns=["thread_id", "screen_name", "t", "t_end", "size"])
    posts = pd.read_parquet(source / "posts_threads.parquet", columns=["id", "thread_id"])
    prob = [f"p_{d}" for d in original_models().fp.DIMS7]
    scores = pd.read_parquet(ROOT / "output/v7/full/post_probs_full.parquet", columns=["id"] + prob)
    ev = score_events(ev, posts, scores, prob)
    del posts, scores
    if ev.thread_id.duplicated().any():
        raise AssertionError("duplicate event ID")
    ref = ev.set_index("thread_id").loc[f.thread_id, "ti7"].to_numpy()
    err = float(np.max(np.abs(ref - f.ti7.to_numpy())))
    if not np.isfinite(err) or err > 2e-6:
        raise AssertionError(f"reconstructed event score disagrees with P1: {err}")
    print("event labels verified", len(ev), "max error", err, flush=True)
    history_n = len(ev)
    lag = embargo_lags(ev).set_index("thread_id").loc[f.thread_id]
    del ev
    amap = pd.read_csv(ROOT / "output/v7/ti/account_member_map.csv")
    mm = (amap[amap.entity_type == "member"].sort_values("n_tweets", ascending=False)
          .groupby("bioguide").agg(icpsr=("icpsr", "first"), state=("state", "first")))
    member_days, _, _ = lc.load_vote_days(ROOT / "data/voteview")
    h = np.empty(len(f), dtype="float32")
    weekend = np.empty(len(f), dtype="float32")
    for bg, indices in f.groupby("bioguide").indices.items():
        if bg not in mm.index:
            raise AssertionError(f"missing member metadata {bg}")
        row = mm.loc[bg]
        state = row.state
        if state not in lc.STATE_ZONE:
            raise AssertionError(f"unknown home state {state}")
        votes = member_days.get(int(row.icpsr), np.array([], dtype="int64"))
        hh, dow = local_hour_for_member(f.t.to_numpy()[indices], votes, state, lc)
        h[indices] = hh
        weekend[indices] = dow >= 5
    out = pd.DataFrame({"thread_id": f.thread_id, "prev_ti7": lag.prev_ti7.to_numpy(dtype="float32"),
                        "roll10": lag.roll10.to_numpy(dtype="float32"),
                        "roll50": lag.roll50.to_numpy(dtype="float32"),
                        "hour": h, "weekend": weekend})
    if not out.thread_id.equals(f.thread_id) or out[list(LAGS)].isna().any().any():
        raise AssertionError("strict feature alignment / missing lag")
    out.to_parquet(HERE / "strict_features.parquet", index=False)
    m = {"n_rows": len(f), "n_events_history": history_n, "label_max_abs_error": err,
         "n_changed_hour": int((h != f.hour.to_numpy()).sum()),
         "n_changed_weekend": int((weekend != f.weekend.to_numpy()).sum()),
         "n_changed_embargo_lag_vs_original": int(np.any(~np.isclose(out[list(LAGS)],
             pd.read_parquet(P1 / "features.parquet", columns=list(LAGS)),
             rtol=0, atol=1e-6, equal_nan=True), axis=1).sum()),
         "n_changed_hour_final": int(((h != f.hour.to_numpy()) &
                                      (f.t.to_numpy() >= int(pd.Timestamp("2022-07-01").timestamp()))).sum()),
         "vote_rule": "only member vote days strictly before Eastern event day; date-only same-day excluded; if <20 prior votes ET fallback; state home zone in split states (no future-post-weighted district choice)",
         "closure_rule": "event.t_end + 7200 < event first-post t; equal second and exact 7200 boundary excluded"}
    (HERE / "strict_audit.json").write_text(json.dumps(m, indent=2))
    print("strict artifact", m, flush=True)


def months():
    m = original_models()
    return m.origins("test")


def archive_month(origin):
    return pd.read_parquet(P1 / f"preds_test/origin_{origin:%Y-%m}.parquet",
                           columns=["thread_id", "acct", "when", "ti7", "gbm_reg_P", "gbm_reg_R"])


def frozen_hp():
    hp = json.loads((P1 / "hp.json").read_text())
    for s in ("P", "R"):
        if f"reg_{s}" not in hp or hp[f"reg_{s}"]["rounds"] < 1:
            raise AssertionError("original frozen hyperparameters missing")
    return hp


def update_diag(origin, key, value):
    path = HERE / "origin_diagnostics.json"
    d = json.loads(path.read_text()) if path.exists() else {}
    d.setdefault(f"{origin:%Y-%m}", {})[key] = value
    path.write_text(json.dumps(d, indent=2))


def run_original():
    frame = pd.read_parquet(P1 / "features.parquet")
    frame["when"] = pd.to_datetime(frame["when"])
    hp = frozen_hp()
    directory = HERE / "predictions_original"
    directory.mkdir(exist_ok=True)
    for origin in months():
        path = directory / f"origin_{origin:%Y-%m}.parquet"
        archive = archive_month(origin)
        if path.exists():
            rows = pd.read_parquet(path)
            audit = verify_archive(rows, archive)
            print(origin.strftime("%Y-%m"), "verified existing", audit, flush=True)
            continue
        rows, diag = original_month(frame, origin, hp, archive)
        rows.to_parquet(path, index=False)
        update_diag(origin, "original", diag)
        print(origin.strftime("%Y-%m"), "reproduced", diag, flush=True)


def run_sensitivity():
    frame = pd.read_parquet(P1 / "features.parquet")
    frame["when"] = pd.to_datetime(frame["when"])
    cols = ["thread_id", "t_end"] + list(LAGS) + ["naive_" + c for c in LAGS]
    arrival = pd.read_parquet(RQ1 / "arrival_features.parquet", columns=cols)
    naive, safe = attach_arrival_lags(frame, arrival)
    del frame, arrival
    hp = frozen_hp()
    directory = HERE / "predictions_matched"
    directory.mkdir(exist_ok=True)
    for origin in months():
        path = directory / f"origin_{origin:%Y-%m}.parquet"
        origpath = HERE / "predictions_original" / f"origin_{origin:%Y-%m}.parquet"
        if not origpath.is_file():
            raise FileNotFoundError(f"original refit missing: {origpath}")
        orig = pd.read_parquet(origpath)
        verify_archive(orig, archive_month(origin))
        if path.exists():
            print(origin.strftime("%Y-%m"), "existing matched", flush=True)
            continue
        rows, diag = matched_month(naive, safe, origin, hp, orig)
        rows.to_parquet(path, index=False)
        update_diag(origin, "matched", diag)
        print(origin.strftime("%Y-%m"), "matched", diag, flush=True)


def run_strict(start=None, end=None):
    frame = pd.read_parquet(P1 / "features.parquet")
    frame["when"] = pd.to_datetime(frame["when"])
    arrival = pd.read_parquet(RQ1 / "arrival_features.parquet", columns=["thread_id", "t_end"])
    if not frame.thread_id.equals(arrival.thread_id):
        raise AssertionError("original and arrival rows misaligned")
    frame["t_end"] = arrival.t_end.to_numpy()
    corrected = pd.read_parquet(HERE / "strict_features.parquet")
    lag, vote = strict_features(frame, corrected)
    del corrected, arrival
    directory = HERE / "predictions_strict"
    directory.mkdir(exist_ok=True)
    hp = frozen_hp()
    for origin in months():
        if start and f"{origin:%Y-%m}" < start:
            continue
        if end and f"{origin:%Y-%m}" > end:
            continue
        path = directory / f"origin_{origin:%Y-%m}.parquet"
        old_path = HERE / "predictions_matched" / f"origin_{origin:%Y-%m}.parquet"
        if not old_path.is_file():
            raise FileNotFoundError(f"prior P1 matched predictions required: {old_path}")
        if path.is_file():
            print(f"{origin:%Y-%m} verified existing strict file", len(pd.read_parquet(path)), flush=True)
            continue
        old = pd.read_parquet(old_path)
        rows, info = strict_month(frame, lag, vote, origin, hp, old)
        rows.to_parquet(path, index=False)
        update_diag(origin, "strict", info)
        print(f"{origin:%Y-%m} strict", info, flush=True)


def run_aligned(start=None, end=None):
    """Repaired cohort/features shared with read-only rq1_strict; P1 LightGBM."""
    frame = pd.read_parquet(P1 / "features.parquet")
    frame["when"] = pd.to_datetime(frame["when"])
    needed = ["thread_id", "ti7", "t_end", "strict_n_completed_prior"] + ["strict_" + c for c in ALIGNED_COLS]
    reference = pd.read_parquet(RQ1_STRICT / "strict_features.parquet", columns=needed)
    legacy, repaired = aligned_repaired_frames(frame, reference)
    del frame, reference
    expected = json.loads((RQ1_STRICT / "monthly_metrics.json").read_text())["combined_conditional_strict"]
    by_month = {x["origin"][:7]: x for x in expected}
    directory = HERE / "predictions_aligned"
    directory.mkdir(exist_ok=True)
    hp = frozen_hp()
    for origin in months():
        key = f"{origin:%Y-%m}"
        if start and key < start:
            continue
        if end and key > end:
            continue
        path = directory / f"origin_{key}.parquet"
        origpath = HERE / "predictions_original" / f"origin_{key}.parquet"
        if not origpath.is_file():
            raise FileNotFoundError(origpath)
        orig = pd.read_parquet(origpath)
        verify_archive(orig, archive_month(origin))
        refids = pd.read_parquet(RQ1_STRICT / "predictions/combined_conditional_strict" /
                                 f"origin_{key}.parquet", columns=["thread_id"])
        if path.is_file():
            rows = pd.read_parquet(path)
            if not np.array_equal(rows.thread_id, refids.thread_id):
                raise AssertionError(f"existing aligned cohort differs from strict ridge {key}")
            print(key, "verified existing aligned", len(rows), flush=True)
            continue
        rows, info = aligned_month(legacy, repaired, origin, hp, orig)
        if (not np.array_equal(rows.thread_id, refids.thread_id)
                or info["n_train"] != by_month[key]["n_train_after_account_filter"]):
            raise AssertionError(f"GBM/ridge train/test row mismatch {key}: {info}")
        rows.to_parquet(path, index=False)
        update_diag(origin, "aligned", info)
        print(key, "aligned", info, flush=True)


def run_aligned_report():
    os_ = months()
    files = [HERE / "predictions_aligned" / f"origin_{o:%Y-%m}.parquet" for o in os_]
    ridge_files = [RQ1_STRICT / "predictions/combined_conditional_strict" /
                   f"origin_{o:%Y-%m}.parquet" for o in os_]
    if len(files) != 13 or not all(p.is_file() for p in files + ridge_files):
        raise AssertionError("13 paired aligned GBM/ridge monthly outputs required")
    parts = [pd.read_parquet(p) for p in files]
    d = pd.concat(parts, ignore_index=True)
    if d.thread_id.duplicated().any():
        raise AssertionError("duplicate aligned test IDs")
    result = aligned_summary(d)
    result["month_block_sensitivity"] = month_block_skill_change(d)
    ridge = pd.concat([pd.read_parquet(p, columns=["thread_id", "acct", "ti7", "content", "rhythm"])
                       for p in ridge_files], ignore_index=True)
    comparison = compare_with_ridge(d.assign(safe_p=d.aligned_p, safe_r=d.aligned_r), ridge)
    ref = json.loads((RQ1_STRICT / "evaluation.json").read_text())["combined_conditional_strict"]["primary"]
    if abs(comparison["ridge_skill"] - ref["relative_mse_reduction"]) > 1e-6:
        raise AssertionError("strict ridge prediction/metric mismatch")
    result["strict_ridge_paired_comparison"] = comparison
    result["strict_ridge_reference"] = ref
    original_full = json.loads((HERE / "evaluation.json").read_text())["original_full"]
    result["n_legacy_test"] = original_full["n"]
    result["n_excluded_from_legacy_test"] = original_full["n"] - len(d)
    diag = json.loads((HERE / "origin_diagnostics.json").read_text())
    result["n_training_row_origin_exclusions_vs_legacy"] = int(sum(
        info["original"]["n_train"] - info["aligned"]["n_train"] for info in diag.values()))
    result["n_changed_test_hour"] = int(sum(info["aligned"]["n_changed_hour_test"] for info in diag.values()))
    result["n_changed_test_congress_spike_lr"] = int(sum(
        info["aligned"]["n_changed_congress_spike_lr_test"] for info in diag.values()))
    result["cohort_note"] = (
        "Legacy original GBM +1.620694% pooled on 422587 P1 test rows; legacy_matched is its "
        "persisted predictions restricted to the 422529 strict-ridge-eligible test rows. "
        "Repaired GBM is refit on the 7200s-closed training subset with >=10 closed prior events "
        "and its test rows exactly match strict ridge month by month. The paired comparison changes "
        "training closure, score-lag eligibility, prior-date vote clock, active-Congress district zone "
        "and no-hindsight-imputation Congress features together; it is not an isolated component effect. "
        "Strict ridge uses additional reply/type controls, one-hot account effects and ridge learner. "
        "Both retain retrospectively defined P1 event universe/outage mask; no unconditional operational as-of claim.")
    (HERE / "aligned_evaluation.json").write_text(json.dumps(result, indent=2))
    monthly = []
    for o, part in zip(os_, parts):
        r = aligned_summary(part, boot=300)
        monthly.append({"origin": f"{o:%Y-%m}", "n": len(part),
                        "legacy_matched_skill": r["models"]["legacy_matched"]["skill"],
                        "repaired_skill": r["models"]["repaired"]["skill"]})
    (HERE / "aligned_monthly_metrics.json").write_text(json.dumps(monthly, indent=2))
    print(json.dumps(result, indent=2), flush=True)


def run_strict_report():
    os_ = months()
    paths = [HERE / "predictions_strict" / f"origin_{o:%Y-%m}.parquet" for o in os_]
    if len(paths) != 13 or not all(p.is_file() for p in paths):
        raise AssertionError("13 full strict monthly predictions required")
    chunks = [pd.read_parquet(p) for p in paths]
    rows = pd.concat(chunks, ignore_index=True)
    if rows.thread_id.duplicated().any():
        raise AssertionError("repeated final test event")
    result = strict_summary(rows)
    full = json.loads((HERE / "evaluation.json").read_text())["original_full"]
    if len(rows) == full["n"] and abs(result["models"]["original"]["skill"] - full["skill"]) > 1e-8:
        raise AssertionError("original frozen refit changed in strict comparison")
    metrics = []
    for o, chunk in zip(os_, chunks):
        s = strict_summary(chunk, boot=300)
        metrics.append({"origin": f"{o:%Y-%m}", "n": len(chunk),
                        "skills": {key: val["skill"] for key, val in s["models"].items()}})
    diag = json.loads((HERE / "origin_diagnostics.json").read_text())
    audit = json.loads((HERE / "strict_audit.json").read_text())
    result["n_test_excluded_vs_original"] = int(full["n"] - len(rows))
    result["n_training_row_origin_exclusions_vs_original"] = int(sum(
        entry["original"]["n_train"] - entry["strict"]["n_train"] for entry in diag.values()))
    result["count_discrepancy_note"] = (
        "The separately reported 1,502 t_end-within-2h row-origin incidences are not reconciled "
        "with the 1,772 exclusions of model-eligible training rows computed here. "
        "Do not equate counts until the other audit's row universe and time boundary are identified.")
    result["n_changed_test_local_hour"] = int(rows.changed_hour.sum())
    result["n_changed_test_closure_lags"] = int(sum(entry["strict"]["n_changed_lag_test"] for entry in diag.values()))
    result["feature_audit"] = audit
    result["ridge_reference_legacy_not_same_asof_design"] = json.loads((RQ1 / "evaluation.json").read_text())["primary"]
    result["interpretation_limits"] = (
        "Conservative as-of sensitivity, not proven operational real-time pipeline: date-only prior-day VoteView "
        "is assumed visible by the next Eastern day (publication delays unverified); a 7200s idle gap is used "
        "as retrospective thread closure, not a real-time classifier proof; original thread-collapse, classifier "
        "scoring, outage mask and other P1 rhythms are not independently operationally validated. "
        "Split-state home zone is the fixed state zone rather than future-post-weighted district choice. "
        "Frozen 2019 hyperparameters were tuned on legacy features, with no final-window retuning. "
        "The separate ridge +0.44% uses legacy local hours and no 7200s closure embargo, and different "
        "learner, type controls and account encoding; do not treat it as the strict design.")
    (HERE / "strict_monthly_metrics.json").write_text(json.dumps(metrics, indent=2))
    (HERE / "strict_evaluation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


def run_report():
    os_ = months()
    full_files = [HERE / "predictions_original" / f"origin_{o:%Y-%m}.parquet" for o in os_]
    pair_files = [HERE / "predictions_matched" / f"origin_{o:%Y-%m}.parquet" for o in os_]
    if not all(p.is_file() for p in full_files + pair_files):
        raise AssertionError("all 13 original and matched month files required")
    full = pd.concat([pd.read_parquet(p) for p in full_files], ignore_index=True)
    pair = pd.concat([pd.read_parquet(p) for p in pair_files], ignore_index=True)
    if full.thread_id.duplicated().any() or pair.thread_id.duplicated().any() or not pair.thread_id.isin(full.thread_id).all():
        raise AssertionError("duplicate/nonmatched test row")
    for o, f in zip(os_, full_files):
        verify_archive(pd.read_parquet(f), archive_month(o))
    report = summarize_predictions(full, pair)
    published = json.loads((P1 / "metrics_test.json").read_text())["reg_mse"]
    for s in ("P", "R"):
        actual = report["original_full"][f"mse_{s.lower()}"]
        if abs(actual - published[f"gbm_reg_{s}"]) > 1e-8:
            raise AssertionError(f"original aggregate MSE mismatch for {s}: {actual}")
    monthly = []
    for o in os_:
        d = pair.loc[pair.origin == o]
        x = paired_summary(d, boot=300)
        monthly.append({"origin": f"{o:%Y-%m}", "n": x["n"],
                        "original_skill": x["original"]["skill"],
                        "cutoff_skill": x["cutoff"]["skill"],
                        "asof_skill": x["asof"]["skill"],
                        "paired_asof_minus_original": x["paired_skill_change"]["asof_minus_original"]})
    (HERE / "monthly_metrics.json").write_text(json.dumps(monthly, indent=2))
    ridge = json.loads((RQ1 / "evaluation.json").read_text())["primary"]
    ridge_files = [RQ1 / "predictions" / f"origin_{o:%Y-%m}.parquet" for o in os_]
    if not all(p.is_file() for p in ridge_files):
        raise AssertionError("all 13 ridge prediction files required for reconciliation")
    ridge_rows = pd.concat([pd.read_parquet(p, columns=["thread_id", "acct", "ti7", "content", "rhythm"])
                            for p in ridge_files], ignore_index=True)
    report["ridge_paired_reconciliation"] = compare_with_ridge(pair, ridge_rows)
    if abs(report["ridge_paired_reconciliation"]["ridge_skill"] - ridge["relative_mse_reduction"]) > 1e-6:
        raise AssertionError("recomputed ridge skill differs from rq1 evaluation")
    report["ridge_reference_not_comparable"] = ridge
    report["published_original_mse"] = {s: published[f"gbm_reg_{s}"] for s in ("P", "R")}
    report["n_original_rows_not_matched"] = len(full) - len(pair)
    report["n_changed_matched_test_lags"] = int(pair.changed_lag.sum())
    report["design"] = {"original": "P vs R, monthly expanding origin, first-arrival target rows, original P1 start-order lags, all prior labels, account threshold >=100, original OOF target encodings; original frozen LightGBM hyperparameters and seed",
                        "matched_cutoff": "same original P1 rows / sets / seed / hyperparameters / target; only train on events with t_end < origin, account threshold >=100 on completed labels; test intersection with original",
                        "matched_asof": "identical training and test rows to matched_cutoff; only prev_ti7, roll10, roll50 replaced using prior events with t_end < arrival, all other P1 features unchanged",
                        "ridge_difference": "ridge RQ1 adds reply/type features to both sets, account fixed effects, standardized ridge alpha=10, and requires >=10 completed prior events; it is NOT the P1 LightGBM estimator or the same feature contrast",
                        "limitations": "eventual thread TI7 and retrospective event collapse / classifier availability are assumed; first-post reply verified in RQ1; timing/rhythm remain original; same frozen hyperparameters are transported without retuning; association not causal"}
    (HERE / "evaluation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k in ("original_full", "matched", "n_original_rows_not_matched", "n_changed_matched_test_lags")}, indent=2), flush=True)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return {"bytes": path.stat().st_size, "sha256": h.hexdigest()}


def run_manifest():
    import lightgbm
    import pyarrow
    files = [p for p in HERE.rglob("*") if p.is_file() and p.name not in ("manifest.json", "manifest.log") and "__pycache__" not in p.parts and ".pytest_cache" not in p.parts]
    inputs = ([P1 / x for x in ("hp.json", "features.parquet", "metrics_test.json")]
              + [SCRIPT / x for x in ("p1_common.py", "p1_models.py", "p1_run.py", "p1_report.py",
                                      "p1_build.py", "t1_common.py", "t1full_common.py")]
              + [ROOT / "xbox" / x for x in ("location.py", "forecast_p1.py")]
              + [RQ1 / x for x in ("arrival_features.parquet", "evaluation.json", "rq1_arrival.py")]
              + [RQ1_STRICT / x for x in ("strict_features.parquet", "strict_features.py",
                                            "run_strict.py", "evaluation.json", "monthly_metrics.json",
                                            "manifest.json")]
              + [RQ1_STRICT / "predictions/combined_conditional_strict" /
                 f"origin_{o:%Y-%m}.parquet" for o in months()]
              + [ROOT / "output/v7/t1full" / x for x in ("events.parquet", "posts_threads.parquet",
                                                         "outage_mask.parquet")]
              + [ROOT / "scripts/v7/t1full_build.py", ROOT / "xbox/hygiene.py"]
              + [ROOT / "output/v7/full/post_probs_full.parquet",
                 ROOT / "output/v7/ti/account_member_map.csv"]
              + sorted((ROOT / "data/voteview").glob("*.csv"))
              + [RQ1 / "predictions" / f"origin_{o:%Y-%m}.parquet" for o in months()]
              + [P1 / "preds_test" / f"origin_{o:%Y-%m}.parquet" for o in months()])
    manifest = {"inputs_read_only": {str(p): sha(p) for p in inputs},
                "outputs": {str(p.relative_to(HERE)): sha(p) for p in sorted(files)},
                "versions": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__,
                             "lightgbm": lightgbm.__version__, "pyarrow": pyarrow.__version__},
                "seed": SEED, "fixed_frozen_hp": {s: frozen_hp()[f"reg_{s}"] for s in ("P", "R")}}
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("manifest", len(manifest["inputs_read_only"]), "input hashes", len(manifest["outputs"]), "output hashes", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["original", "sensitivity", "build-strict", "strict", "aligned", "report", "strict-report", "aligned-report", "manifest"], required=True)
    parser.add_argument("--from-month", help="strict/aligned stage lower inclusive YYYY-MM")
    parser.add_argument("--through-month", help="strict/aligned stage upper inclusive YYYY-MM")
    args = parser.parse_args()
    if args.stage == "strict":
        run_strict(args.from_month, args.through_month)
    elif args.stage == "aligned":
        run_aligned(args.from_month, args.through_month)
    else:
        {"original": run_original, "sensitivity": run_sensitivity,
         "build-strict": build_strict_artifact,
         "report": run_report, "strict-report": run_strict_report,
         "aligned-report": run_aligned_report,
         "manifest": run_manifest}[args.stage]()
