"""Point-process model comparison refitted on the congressional corpus (every collected day).

The comparison in scripts/v7/temporal_model_compare*.py read data/congresstweets_2017_2023.parquet, which holds
every third day of each month. This wrapper reruns the same code, unmodified, on
data/congresstweets_full/congresstweets_full.parquet and writes to output/v7/temporal_full/model_compare.
Nothing under output/v7/temporal is modified.

Stages: cache, fit, boot.   Usage: python run_full.py STAGE [--workers 16]
Members: every member-mapped member with at least 200 non-repost posts in the corpus and a Voteview identifier.
"""
import argparse
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.home() / "x_box/pipeline"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/v7"))
OLD = ROOT / "output/v7/temporal/model_compare"
OUT = ROOT / "output/v7/temporal_full/model_compare"
SAMPLE = str(ROOT / "data/congresstweets_2017_2023.parquet")
FULL = str(ROOT / "data/congresstweets_full/congresstweets_full.parquet")
DIAG = str(ROOT / "output/v7/temporal/hawkes_ci_diagnostics.csv")
MIN_N = 200

_read_parquet = pd.read_parquet
_read_csv = pd.read_csv


def read_parquet(path, *a, **k):
    if str(path) == SAMPLE:
        path = FULL
    return _read_parquet(path, *a, **k)


def eligible_members():
    import temporal_features as tf
    amap = _read_csv(ROOT / "output/v7/ti/account_member_map.csv", dtype={"user_id": str})
    amap = amap[amap.entity_type == "member"].copy()
    tw = _read_parquet(FULL, columns=["id", "screen_name", "created_at", "text"]).drop_duplicates("id")
    tw = tw.merge(amap[["screen_name", "bioguide"]], on="screen_name", how="inner")
    t = tw["created_at"].dt.tz_convert("UTC").dt.tz_localize(None).values.astype("datetime64[s]").astype("int64")
    lo, hi = tf.WINDOWS["full"]
    keep = tf.tv.nonretweet_mask_congresstweets(tw["text"].values) & (t >= lo) & (t < hi)
    n = tw[keep].groupby("bioguide").size()
    icpsr = amap.sort_values("n_tweets", ascending=False).groupby("bioguide").icpsr.first()
    n = n[(n >= MIN_N) & n.index.isin(icpsr[icpsr.notna()].index)]
    return pd.DataFrame(dict(level="member", window="full", postset="nonrt", id=n.index, n=n.values))


def read_csv(path, *a, **k):
    if str(path) == DIAG:
        return ELIG.copy()
    return _read_csv(path, *a, **k)


def stage_cache(a):
    global ELIG
    import temporal_hawkes_diag as dg
    ELIG = eligible_members()
    ELIG.to_csv(OUT / "eligible_members.csv", index=False)
    print("eligible members", len(ELIG), flush=True)
    pd.read_parquet, pd.read_csv = read_parquet, read_csv
    t0 = time.time()
    pick = dg._load(10 ** 6)
    pd.read_parquet, pd.read_csv = _read_parquet, _read_csv
    out, skipped = {}, []
    for bg in pick:
        try:
            t, d, seed = dg.member_design(bg)
            out[bg] = dict(t=d["t"], slot_month=d["slot_month"], slot_bin=d["slot_bin"], nm=int(d["occ"].shape[0]),
                           seed=int(seed), n=len(t))
        except Exception as e:  # noqa
            skipped.append((bg, str(e)[:120]))
    pickle.dump(out, open(OUT / "cache.pkl", "wb"))
    shutil.copy(OLD / "cache_lee.pkl", OUT / "cache_lee.pkl")
    print("cached", len(out), "skipped", skipped, round(time.time() - t0), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--B", type=int, default=30)
    ap.add_argument("--nboot", type=int, default=100)
    ap.add_argument("--models", default="")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage == "cache":
        return stage_cache(a)
    import temporal_model_compare as mc
    mc.OUT = OUT
    {"fit": mc.stage_fit, "boot": mc.stage_boot}[a.stage](a)


if __name__ == "__main__":
    main()
