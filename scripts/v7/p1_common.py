"""P1 shared paths, source switch and rolling-origin calendar (MATH_SPEC section H).

Source flag: --source subset (default; development on the 1-in-3-day scored subset, output/v7/p1/dev) or
--source full (output/v7/full/post_probs_full.parquet, output/v7/p1).  Everything else is identical.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from xbox import forecast_p1 as fp  # noqa: E402

PROBS = {"subset": ROOT / "output/v7/ti/post_probs.parquet",
         "full": ROOT / "output/v7/full/post_probs_full.parquet"}
P1 = ROOT / "output/v7/p1"
FIRST_ROW = pd.Timestamp("2017-08-01")      # congress lag features need 4 weeks of history
DEV_FIRST_ORIGIN = pd.Timestamp("2019-07-01")
TUNE_TRAIN_END, TUNE_VAL_END = pd.Timestamp("2019-01-01"), DEV_FIRST_ORIGIN     # tuning uses data before every origin
MIN_TRAIN_ROWS = 100                         # an account is evaluated at an origin only with this many training rows
MIN_PRIOR_SCORED = 10
THREADS = 16


def get_args(extra=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["subset", "full"], default="subset")
    if extra:
        extra(ap)
    a = ap.parse_args()
    return a


def outdir(source):
    d = P1 / "dev" if source == "subset" else P1
    d.mkdir(parents=True, exist_ok=True)
    return d


def origins(kind):
    """Monthly forecast origins.  dev: 2019-07-01 .. 2022-06-01.  test: 2022-07-01 .. 2023-07-01."""
    if kind == "dev":
        return list(pd.date_range(DEV_FIRST_ORIGIN, fp.TEST_START - pd.offsets.MonthBegin(1), freq="MS"))
    return list(pd.date_range(fp.TEST_START, fp.TEST_END - pd.Timedelta(days=1), freq="MS"))


def window_end(o):
    """Exclusive end of the test window of origin o (next month start; capped at the corpus end)."""
    return min(o + pd.offsets.MonthBegin(1), fp.TEST_END)


def ts_epoch(ts):
    return int(pd.Timestamp(ts).value // 10 ** 9)
