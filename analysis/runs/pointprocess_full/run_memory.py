"""How long does the leftover dependence in the rescaled intervals last?

For each member the rescaled intervals u of a fitted model are independent and uniform if the model is correct.
Detrended fluctuation analysis (DFA, linear detrending) of their normal scores gives an exponent alpha: 0.5 for
independent values, above 0.5 for persistent ones. A short-memory process shows alpha above 0.5 only at small window
sizes and returns to 0.5 at large ones; a long-memory process keeps alpha above 0.5 at all sizes. The exponent is
computed over all window sizes, over small ones (8 to 64 intervals) and over large ones (64 intervals and more), for
the exponential Hawkes model with and without the shared-activity term, and is calibrated against streams simulated
from the fitted exponential Hawkes model and refitted (B per member). Autocorrelations at fixed lags are also stored.

    python run_memory.py [--workers 16] [--B 20]      writes output/v7/temporal_events/shared/memory.csv
"""
import argparse
import time
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

import run_shared as rs
from run_shared import pm, sm

warnings.filterwarnings("ignore")
LAGS = [1, 2, 5, 10, 20, 50, 100]
SPLIT = 64


def normal_scores(u):
    return norm.ppf(rankdata(u) / (len(u) + 1.0))


def dfa(x):
    """Fluctuation function of DFA-1 on log-spaced window sizes from 8 to n/8. Returns sizes, F."""
    n = len(x)
    if n < 128:
        return np.array([]), np.array([])
    y = np.cumsum(x - x.mean())
    sizes = np.unique(np.round(np.exp(np.linspace(np.log(8), np.log(n // 8), 18))).astype(int))
    F = []
    for s in sizes:
        k = n // s
        tt = np.arange(s) - (s - 1) / 2.0
        den = (tt ** 2).sum()
        v = []
        for seg in (y[:k * s].reshape(k, s), y[n - k * s:].reshape(k, s)):
            m = seg.mean(axis=1, keepdims=True)
            b = (seg * tt).sum(axis=1, keepdims=True) / den
            v.append(((seg - m - b * tt) ** 2).mean(axis=1))
        F.append(np.sqrt(np.mean(np.concatenate(v))))
    return sizes, np.array(F)


def slope(s, F, lo=None, hi=None):
    ok = np.ones(len(s), bool)
    if lo is not None:
        ok &= s >= lo
    if hi is not None:
        ok &= s <= hi
    if ok.sum() < 4:
        return np.nan
    return float(np.polyfit(np.log(s[ok]), np.log(F[ok]), 1)[0])


def alphas(x):
    s, F = dfa(x)
    if len(s) == 0:
        return np.nan, np.nan, np.nan
    return slope(s, F), slope(s, F, hi=SPLIT), slope(s, F, lo=SPLIT)


def acf(x, lag):
    if len(x) <= lag + 10:
        return np.nan
    a, b = x[:-lag], x[lag:]
    return float(np.corrcoef(a, b)[0, 1])


def describe(u):
    x = normal_scores(u)
    a, a_s, a_l = alphas(x)
    out = dict(alpha=a, alpha_short=a_s, alpha_long=a_l)
    out.update({f"acf{k}": acf(x, k) for k in LAGS})
    return out


def task(args):
    bg, B = args
    try:
        d, z = rs.member(bg)
        r0, r1 = pm.fit_model("hawkes_exp", d), sm.fit_shared(d, z)
        rows = [dict(bg=bg, model="hawkes_exp", n=d.n, **describe(r0["u"])),
                dict(bg=bg, model="hawkes_shared", n=d.n, **describe(r1["u"]))]
        raw = np.log(np.maximum(np.diff(d.t), 1e-6))
        a, a_s, a_l = alphas(normal_scores(raw))
        rows.append(dict(bg=bg, model="raw_intervals", n=d.n, alpha=a, alpha_short=a_s, alpha_long=a_l))
        N = []
        for b in range(B):
            t = pm.simulate("hawkes_exp", d, r0, 7919 * (b + 1) + 11)
            ds = pm.sim_to_data(t, d, 613 * (b + 1) + 5)
            if ds.n < 128:
                continue
            N.append(alphas(normal_scores(pm.fit_model("hawkes_exp", ds)["u"])))
        if N:
            N = np.array(N)
            obs = np.array([rows[0]["alpha"], rows[0]["alpha_short"], rows[0]["alpha_long"]])
            for j, nm in enumerate(["alpha", "alpha_short", "alpha_long"]):
                col = N[:, j][np.isfinite(N[:, j])]
                if len(col) and np.isfinite(obs[j]):
                    rows[0][f"null_{nm}_med"] = float(np.median(col))
                    rows[0][f"null_{nm}_q95"] = float(np.quantile(col, 0.95))
                    rows[0][f"null_{nm}_p"] = float((1 + (col >= obs[j]).sum()) / (1 + len(col)))
            rows[0]["B"] = len(N)
        return rows
    except Exception as e:  # noqa
        return [dict(bg=bg, model="error", err=str(e)[:200])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--B", type=int, default=20)
    a = ap.parse_args()
    rs.load()
    bgs = sorted(rs.CACHE, key=lambda b: -rs.CACHE[b]["n"])
    task((bgs[-1], 1))
    t0 = time.time()
    R = []
    with Pool(a.workers, maxtasksperchild=8) as p:
        for i, r in enumerate(p.imap_unordered(task, [(b, a.B) for b in bgs], chunksize=1)):
            R += r
            if i % 100 == 0:
                print(i, len(bgs), round(time.time() - t0), flush=True)
    pd.DataFrame(R).to_csv(rs.OUT / "memory.csv", index=False)
    print("memory done", round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
