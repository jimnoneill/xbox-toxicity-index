"""Point-process model comparison on member event streams (threads merged), congressional corpus.

Same models and code as run_full.py. The stream of a member is the start time of each event in
output/v7/t1full/events.parquet (non-repost posts of one account joined into an event by the thread rules of
xbox.hygiene.collapse_threads), pooled over the member's accounts. Members: at least 200 events and a Voteview
identifier. Output: output/v7/temporal_events/model_compare. Nothing else is modified.

Stages: cache, fit, validate, boot.   Usage: python run_events.py STAGE [--workers 16] [--dir temporal_events]
(--dir temporal_full runs validate on the post-level fits of run_full.py.)
"""
import argparse
import pickle
import shutil
import sys
import time

import numpy as np
import pandas as pd

import run_full as rf

ROOT = rf.ROOT
MIN_N = 200


def stage_cache(out):
    import temporal_features as tf
    import temporal_hawkes_diag as dg
    lo, hi = tf.WINDOWS["full"]
    ev = pd.read_parquet(ROOT / "output/v7/t1full/events.parquet", columns=["bioguide", "t"])
    ev = ev[(ev.t >= lo) & (ev.t < hi)]
    elig = rf.eligible_members()                      # members with >= 200 non-repost posts and a Voteview id
    n = ev.groupby("bioguide").size()
    keep = [b for b in elig.id if n.get(b, 0) >= MIN_N]
    rf.ELIG = elig[elig.id.isin(keep)].copy()
    rf.ELIG.assign(n_events=[int(n[b]) for b in rf.ELIG.id]).to_csv(out / "eligible_members.csv", index=False)
    print("eligible members", len(rf.ELIG), flush=True)
    pd.read_parquet, pd.read_csv = rf.read_parquet, rf.read_csv
    t0 = time.time()
    pick = dg._load(10 ** 6)
    pd.read_parquet, pd.read_csv = rf._read_parquet, rf._read_csv
    te = {b: g.t.values for b, g in ev.groupby("bioguide")}
    res, skipped = {}, []
    for bg in pick:
        try:
            t, d, seed = dg.member_design(bg, t_override=te[bg])
            res[bg] = dict(t=d["t"], slot_month=d["slot_month"], slot_bin=d["slot_bin"], nm=int(d["occ"].shape[0]),
                           seed=int(seed), n=len(t))
        except Exception as e:  # noqa
            skipped.append((bg, str(e)[:120]))
    pickle.dump(res, open(out / "cache.pkl", "wb"))
    shutil.copy(rf.OLD / "cache_lee.pkl", out / "cache_lee.pkl")
    print("cached", len(res), "skipped", skipped, round(time.time() - t0), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("--dir", default="temporal_events")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--B", type=int, default=30)
    ap.add_argument("--nboot", type=int, default=100)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--ntempl", type=int, default=5)
    ap.add_argument("--models", default="")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    out = ROOT / "output/v7" / a.dir / "model_compare"
    out.mkdir(parents=True, exist_ok=True)
    if a.stage == "cache":
        assert a.dir == "temporal_events"
        return stage_cache(out)
    import temporal_model_compare as mc
    mc.OUT = out
    {"fit": mc.stage_fit, "boot": mc.stage_boot, "validate": mc.stage_validate}[a.stage](a)


if __name__ == "__main__":
    main()
