"""Are a member's timing rules fixed over the years, or do they change from year to year?

For every member with yearly fits in all five calendar years 2018 to 2022 (years_shared.csv), the exponential Hawkes model
and the Hawkes model with the shared-activity term are fitted once to the five years together (rules fixed) and compared
with the five yearly fits already stored (rules refitted each year). Both versions have a separate level for every
month; the yearly version also has its own hour-of-week schedule, self-excitation, decay rate and shared-activity
coefficient in each year.

  LR   = 2 (sum of yearly log-likelihoods - fixed log-likelihood), df = sum of yearly k - fixed k
  BIC  = -2 logL + k log(n), with n the member's events in the five years, for both versions

Writes drift.csv (one row per member and model) and drift_summary.json to output/v7/temporal_events/shared.
Usage: python run_drift.py [--workers 16]
"""
import argparse
import json
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import chi2

import run_shared as rs
from run_shared import pm, sm

OUT = rs.OUT
Y = {}


def pooled_data(bg, d, zraw):
    ts = rs.SLOT0[bg] + 3600 * np.arange(d.ns, dtype="int64")
    y0 = pd.Timestamp("2018-01-01", tz="UTC").timestamp()
    y1 = pd.Timestamp("2023-01-01", tz="UTC").timestamp()
    idx = np.nonzero((ts >= y0) & (ts < y1))[0]
    s0, s1 = idx[0], idx[-1] + 1
    t = d.t[(d.t >= s0 * 60.0) & (d.t < s1 * 60.0)] - s0 * 60.0
    mo = np.unique(d.slot_month[s0:s1], return_inverse=True)[1]
    return pm.Data(t, mo, d.slot_bin[s0:s1], int(mo.max()) + 1), sm.standardize(zraw[s0:s1])


def task(bg):
    try:
        d, _ = rs.member(bg)
        zraw = rs.others_log(bg, d.ns)
        dp, zp = pooled_data(bg, d, zraw)
        out = []
        for model, r in [("hawkes_exp", pm.fit_model("hawkes_exp", dp)), ("hawkes_shared", sm.fit_shared(dp, zp))]:
            y = Y[(bg, model)]
            n = len(dp.t)
            ll_y, k_y = float(y.ll.sum()), int(y.k.sum())
            lr, df = 2 * (ll_y - r["ll"]), k_y - r["k"]
            par = {a: float(b) for a, b in r["params"].items()}
            row = dict(bg=bg, model=model, n=n, n_yearly=int(y.n.sum()), ll_fixed=r["ll"], k_fixed=r["k"], ll_yearly=ll_y, k_yearly=k_y,
                       lr=lr, df=df, p_lr=float(chi2.sf(lr, df)) if df > 0 else np.nan,
                       bic_fixed=-2 * r["ll"] + r["k"] * np.log(n), bic_yearly=-2 * ll_y + k_y * np.log(n),
                       aic_fixed=-2 * r["ll"] + 2 * r["k"], aic_yearly=-2 * ll_y + 2 * k_y,
                       D_fixed=r["D"], D_yearly_median=float(y.D.median()), D_yearly_max=float(y.D.max()),
                       eta_fixed=par.get("eta"), eta_min=float(y.eta.min()), eta_max=float(y.eta.max()),
                       gamma_fixed=par.get("gamma"), gamma_min=float(y.gamma.min()) if model == "hawkes_shared" else np.nan,
                       gamma_max=float(y.gamma.max()) if model == "hawkes_shared" else np.nan)
            out.append(row)
        return out
    except Exception as e:  # noqa
        return [dict(bg=bg, err=str(e)[:200])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    rs.load()
    yy = pd.read_csv(OUT / "years_shared.csv")
    yy = yy[yy.year.isin(rs.YEARS)]
    c = yy[yy.model == "hawkes_exp"].groupby("bg").year.nunique()
    bgs = sorted(c[c == len(rs.YEARS)].index)
    for (bg, model), g in yy[yy.bg.isin(bgs)].groupby(["bg", "model"]):
        Y[(bg, model)] = g
    print(len(bgs), "members with all five yearly fits", flush=True)
    t0 = time.time()
    R = []
    with Pool(a.workers, maxtasksperchild=8) as p:
        for i, r in enumerate(p.imap_unordered(task, bgs, chunksize=1)):
            R += r
            if i % 50 == 0:
                print(i, round(time.time() - t0), flush=True)
    df = pd.DataFrame(R)
    df.to_csv(OUT / "drift.csv", index=False)
    S = dict(members=len(bgs), errors=int(df.err.notna().sum()) if "err" in df else 0)
    for model, g in df[df.ll_fixed.notna()].groupby("model"):
        S[model] = dict(
            n_members=int(len(g)),
            median_events=float(g.n.median()),
            n_match_pct=float(100 * (g.n == g.n_yearly).mean()),
            median_df=float(g.df.median()),
            lr_reject_fixed_5pct=float(100 * (g.p_lr < 0.05).mean()),
            lr_reject_fixed_bonferroni_pct=float(100 * (g.p_lr < 0.05 / len(g)).mean()),
            bic_prefers_yearly_pct=float(100 * (g.bic_yearly < g.bic_fixed).mean()),
            aic_prefers_yearly_pct=float(100 * (g.aic_yearly < g.aic_fixed).mean()),
            median_lr_over_df=float((g.lr / g.df).median()),
            median_gain_per_event=float(((g.ll_yearly - g.ll_fixed) / g.n).median()),
            median_D_fixed=float(g.D_fixed.median()), median_D_yearly=float(g.D_yearly_median.median()),
            median_eta_fixed=float(g.eta_fixed.median()), median_eta_range=float((g.eta_max - g.eta_min).median()),
            median_gamma_range=float((g.gamma_max - g.gamma_min).median()) if model == "hawkes_shared" else None)
    json.dump(S, open(OUT / "drift_summary.json", "w"), indent=1)
    print(json.dumps(S, indent=1))
    print("done", round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
