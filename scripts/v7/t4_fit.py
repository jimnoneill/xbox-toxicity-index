"""T4 step: per-account negative binomial fits on 60-minute bins (MATH_SPEC G), five variants.

Variants: full, nolag (no own-lag terms), noC (no Congress term), base (neither), csplit (Congress term split into the
daily mean and the within-day deviation), plus noreg (no regime flags; reveals what the regime flags absorb).
Accounts need >= 500 events inside the model sample (in office, lags available).
Outputs (output/v7/t4/): account_fits.csv (long: account, variant, term, beta, se_model, se_robust),
account_summary.csv (one row per account and variant), fit_progress.log
Usage: python t4_fit.py [--workers 16] [--variants full,nolag,noC,base,csplit,noreg]
"""
import argparse
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t4_data import *  # noqa: E402,F401,F403

warnings.filterwarnings("ignore")
FIT_TERMS = ["intercept", "log1p_congress", "log1p_congress_daymean", "log1p_congress_dev", "log1p_lag1",
             "log1p_lag2_24"] + bm.REGIMES


def task(args):
    i, variants = args
    rows, summ = [], []
    for v in variants:
        try:
            des = account_design(i, v)
            n_ev = float(des["y"].sum())
            if len(des["y"]) < 1000 or n_ev < MIN_EVENTS_T4 or not des["cal_ok"]:
                summ.append(dict(account=G["accounts"][i], variant=v, status="skipped", n_bins=len(des["y"]),
                                 n_events=n_ev, cal_ok=des["cal_ok"]))
                break
            fit = bm.fit_nb2(des["X"], des["y"], cluster=des["cluster"])
        except Exception as e:  # keep going, record
            summ.append(dict(account=G["accounts"][i], variant=v, status="error:" + str(e)[:80]))
            continue
        names = des["names"]
        for t in FIT_TERMS:
            if t in names:
                j = names.index(t)
                rows.append(dict(account=G["accounts"][i], variant=v, term=t, beta=fit["beta"][j],
                                 se_model=fit["se_model"][j], se_robust=fit["se_robust"][j]))
        summ.append(dict(account=G["accounts"][i], variant=v, status="ok", n_bins=fit["n"], n_events=n_ev,
                         alpha=fit["alpha"], ll=fit["ll"], converged=fit["converged"], n_clusters=fit["n_clusters"],
                         n_cols=len(names), n_dropped=len(names) - len(fit["keep"]), mean_mu=fit["mean_mu"],
                         bioguide=G["bioguide"][i], party=G["party"][i], chamber=G["chamber"][i],
                         account_type=G["atype"][i]))
    return rows, summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--variants", default="full,nolag,noC,base,csplit,noreg")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    T4.mkdir(parents=True, exist_ok=True)
    n = load_all()
    variants = a.variants.split(",")
    idx = list(range(n))
    if a.limit:
        idx = idx[:a.limit]
    # largest first for load balance
    idx.sort(key=lambda i: -(G["ends"][i] - G["starts"][i]))
    print("accounts in hourly file", n, "loaded", round(time.time() - t0), "s", flush=True)
    R, S = [], []
    with Pool(a.workers) as p:
        for k, (r, s) in enumerate(p.imap_unordered(task, [(i, variants) for i in idx], chunksize=1)):
            R += r
            S += s
            if k % 50 == 0:
                print(k, len(idx), round(time.time() - t0), "s", flush=True)
    pd.DataFrame(R).to_csv(T4 / "account_fits.csv", index=False)
    pd.DataFrame(S).to_csv(T4 / "account_summary.csv", index=False)
    print("done", len(R), len(S), round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()
