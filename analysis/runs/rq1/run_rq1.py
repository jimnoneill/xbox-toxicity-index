"""Full-corpus as-of audit and matched monthly RQ1 ridge evaluation.

Run on HPCF: ~/x_box/venv/bin/python run_rq1.py --stage audit
             ~/x_box/venv/bin/python run_rq1.py --stage eval
Only writes beside this script. No access to shared output paths for writing.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rq1_arrival import assemble_frame, metric_summary, score_events, select_origin, fit_pair, previous_start_inflight

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]  # .../pipeline/manuscript/epjds_submission_work/rq1 -> pipeline
assert ROOT.name == "pipeline", ROOT
T1 = ROOT / "output/v7/t1full"
P1 = ROOT / "output/v7/p1"
DIMS = ["offensive", "toxic", "neg_sentiment", "irony", "ragebait", "tribal", "performative"]
PROB = [f"p_{x}" for x in DIMS]
CONTENT = ["prev_ti7", "roll10", "roll50", "prev_reply", "prev_multi", "reply_share10", "own_reply"]
RHYTHM = ["log_dt", "sess_pos", "sess_el", "n1h", "n24h", "hour", "hsin", "hcos", "h2sin", "h2cos",
          "night", "weekend", "sched_share", "cong_p1", "cong_p24", "cong_spike_lr", "cong_spike"]
SEED = 20261010


def audit():
    events = pd.read_parquet(T1 / "events.parquet", columns=["thread_id", "screen_name", "t", "t_end", "size", "first_id"])
    posts = pd.read_parquet(T1 / "posts_threads.parquet", columns=["id", "thread_id"])
    probs = pd.read_parquet(ROOT / "output/v7/full/post_probs_full.parquet", columns=["id"] + PROB)
    events = score_events(events, posts, probs, PROB)
    del posts, probs
    original = pd.read_parquet(P1 / "features.parquet")
    ref = events.set_index("thread_id").loc[original.thread_id]
    err = np.abs(ref.ti7.to_numpy() - original.ti7.to_numpy())
    if not (np.isfinite(err).all() and np.max(err) < 2e-6):
        raise AssertionError(f"P1 target event-label mismatch: max {np.nanmax(err)}")
    assert np.array_equal(ref.t.to_numpy(), original.t.to_numpy())
    assert np.array_equal(ref["size"].to_numpy(), original["size"].to_numpy())
    # Reply is defined on the first post's text only, observable at event arrival.
    raw = pd.read_parquet(ROOT / "data/congresstweets_full/congresstweets_full.parquet", columns=["id", "text"])
    first = events.first_id.to_numpy()
    reply = raw.drop_duplicates("id").set_index("id").text.reindex(first)
    if reply.isna().any():
        raise AssertionError("first-post text absent in raw archive")
    reply_flag = reply.str.lstrip().str.startswith("@").to_numpy(dtype=float)
    events["first_reply"] = reply_flag
    observed_reply = events.set_index("thread_id").first_reply.reindex(original.thread_id).to_numpy()
    if not np.array_equal(observed_reply, original.own_reply.to_numpy()):
        raise AssertionError("own_reply is not the first-post arrival reply flag")
    del raw, reply, ref
    print("loaded", len(events), "events", len(original), "original P1 rows; label/reply checks pass", flush=True)
    frame = assemble_frame(events, original)
    # Diagnostic distinction: differences can reflect completion-order rather than leakage.
    flag = previous_start_inflight(events)
    frame = frame.merge(flag, on="thread_id", how="left", validate="one_to_one")
    assert (frame.naive_prev_unavailable <= (frame.prior_inflight > 0)).all()
    frame["prior_inflight"] = frame.prior_inflight.astype("int32")
    frame["n_completed_prior"] = frame.n_completed_prior.astype("int32")
    affected = frame[frame.naive_prev_unavailable == 1][["thread_id", "screen_name", "t", "t_end", "size", "ti7",
        "prev_start_thread_id", "prev_start_end", "naive_prev_ti7", "prev_ti7", "naive_roll10", "roll10",
        "naive_prev_multi", "prev_multi", "prior_inflight"]].copy()
    prior = events[["thread_id", "t", "size", "ti7"]].rename(columns={
        "thread_id": "prev_start_thread_id", "t": "prev_start_t", "size": "prev_start_size", "ti7": "prev_start_ti7"})
    affected = affected.merge(prior, on="prev_start_thread_id", how="left", validate="many_to_one")
    if affected.prev_start_t.isna().any() or (affected.prev_start_end < affected.t).any():
        raise AssertionError("interleaved case evidence inconsistent")
    affected.to_csv(HERE / "interleaved_cases.csv", index=False)
    for col in ["prev_ti7", "roll10", "roll50", "prev_multi"]:
        a, b = frame["naive_" + col].to_numpy(), frame[col].to_numpy()
        frame["changed_" + col] = ~(np.isclose(a, b, atol=1e-6, rtol=0, equal_nan=True))
    audit_info = {
        "event_universe": int(len(events)), "original_p1_rows": int(len(original)),
        "original_label_max_abs_error": float(err.max()), "first_post_reply_verified_rows": int(len(original)),
        "multi_events": int((events["size"] > 1).sum()),
        "multi_event_duration_positive": int(((events["size"] > 1) & (events.t_end > events.t)).sum()),
        "rows_with_any_inflight_prior": int((frame.prior_inflight > 0).sum()),
        "rows_with_naive_immediate_prev_unavailable": int(frame.naive_prev_unavailable.sum()),
        "rows_with_less_than_10_completed_prior": int((frame.n_completed_prior < 10).sum()),
        "rows_with_changed_prev_ti7": int(frame.changed_prev_ti7.sum()),
        "rows_with_changed_roll10": int(frame.changed_roll10.sum()),
        "rows_with_changed_roll50": int(frame.changed_roll50.sum()),
        "rows_with_changed_prev_multi": int(frame.changed_prev_multi.sum()),
        "inflight_prior_test_rows": int(((frame.t >= pd.Timestamp("2022-07-01").timestamp()) & (frame.prior_inflight > 0)).sum()),
        "asof_rule": "end timestamp strictly earlier than arrival; score/size only from completed prior events",
        "event_label": "eventual completed-thread mean over post-wise mean of seven probabilities",
        "own_reply_rule": "first post text only, verified against raw first_id text",
        "limits": "retrospective thread reconstruction and timestamp-second resolution; first-post type observed, target label eventual; no claim of causal effect",
        "input_paths": [str(T1 / "events.parquet"), str(T1 / "posts_threads.parquet"),
                        str(ROOT / "output/v7/full/post_probs_full.parquet"), str(P1 / "features.parquet")],
    }
    frame.to_parquet(HERE / "arrival_features.parquet", index=False)
    (HERE / "audit.json").write_text(json.dumps(audit_info, indent=2))
    print(json.dumps({k: v for k, v in audit_info.items() if isinstance(v, (int, float))}), flush=True)


def evaluate():
    frame = pd.read_parquet(HERE / "arrival_features.parquet")
    start = pd.Timestamp("2022-07-01")
    end = pd.Timestamp("2023-07-12")
    pred_dir = HERE / "predictions"
    pred_dir.mkdir(exist_ok=True)
    months = pd.date_range(start, end - pd.Timedelta(days=1), freq="MS")
    per_month = []
    for o in months:
        stop = min(o + pd.offsets.MonthBegin(1), end)
        origin, limit = int(o.timestamp()), int(stop.timestamp())
        tr, te = select_origin(frame, origin, limit, min_train=100)
        if te.empty:
            continue
        assert tr.t_end.max() < origin <= te.t.min() and te.t.max() < limit
        p = fit_pair(tr, te, CONTENT, RHYTHM)
        r = te[["thread_id", "acct", "t", "t_end", "size", "ti7", "prior_inflight", "naive_prev_unavailable"]].copy()
        r["content"], r["rhythm"] = p["content"], p["rhythm"]
        r["origin"] = str(o.date())
        r.to_parquet(pred_dir / f"origin_{o:%Y-%m}.parquet", index=False)
        row = {"origin": str(o.date()), "n_train_eligible": int(len(tr)), "n_test": int(len(r)),
               "accounts_train": int(tr.acct.nunique()), **metric_summary(r, boot=300, seed=SEED)}
        per_month.append(row)
        print(row["origin"], "train", row["n_train_eligible"], "test", row["n_test"],
              "delta MSE", round(row["delta_mse"], 7), flush=True)
    (HERE / "monthly_metrics.json").write_text(json.dumps(per_month, indent=2))
    predictions = pd.concat([pd.read_parquet(pred_dir / f"origin_{o:%Y-%m}.parquet") for o in months if (pred_dir / f"origin_{o:%Y-%m}.parquet").exists()], ignore_index=True)
    if predictions.thread_id.duplicated().any():
        raise AssertionError("test row repeated across monthly origins")
    outputs = {"primary": metric_summary(predictions, seed=SEED),
               "single_post": metric_summary(predictions[predictions["size"] == 1], seed=SEED),
               "no_inflight_prior": metric_summary(predictions[predictions.prior_inflight == 0], seed=SEED),
               "single_post_no_inflight": metric_summary(predictions[(predictions["size"] == 1) & (predictions.prior_inflight == 0)], seed=SEED),
               "naive_prev_available": metric_summary(predictions[predictions.naive_prev_unavailable == 0], seed=SEED)}
    (HERE / "evaluation.json").write_text(json.dumps(outputs, indent=2))
    print(json.dumps(outputs, indent=2), flush=True)


def manifest():
    items = {}
    for p in [HERE / "README.md", HERE / "audit.log", HERE / "eval.log", HERE / "rq1_arrival.py", HERE / "run_rq1.py", HERE / "test_rq1.py",
              HERE / "audit.json", HERE / "evaluation.json", HERE / "monthly_metrics.json", HERE / "interleaved_cases.csv", HERE / "arrival_features.parquet"] + sorted((HERE / "predictions").glob("*.parquet")):
        if p.exists():
            items[str(p.relative_to(HERE))] = {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    (HERE / "manifest.json").write_text(json.dumps({"inputs_read_only": [str(T1), str(P1), str(ROOT / "output/v7/full"), str(ROOT / "data/congresstweets_full")],
                                                 "files": items, "seed": SEED, "design": "monthly expanding-origin 2022-07 through 2023-07; 600000 deterministic training cap, ridge alpha=10; 1000 account bootstrap replicates"}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["audit", "eval"], required=True)
    a = ap.parse_args()
    if a.stage == "audit":
        audit()
    else:
        evaluate()
    manifest()
