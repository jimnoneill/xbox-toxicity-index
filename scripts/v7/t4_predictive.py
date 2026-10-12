"""T4 predictive checks: simulate from fitted models for a random sample of 100 accounts and compare observed with
predicted (a) binned sessions per day (runs of consecutive non-empty hours), (b) night share (local 00-06),
(c) weekly volume (mean, coefficient of variation of weekly totals), (d) response on detected event days
(mean daily count on the top-60 event days divided by the mean on non-event days 7-28 days away).
Models refitted per account: full, noC (no Congress term), nolag (no own lags), base (neither). Simulations are conditional
on the observed covariates and the observed Congress-wide counts (as in the fit), 200 per model and account.
For each statistic: observed value, simulated mean and 95% interval, probability integral transform (PIT).
Outputs: output/v7/t4/predictive_checks.csv, predictive_summary.json, predictive_sample.csv
Usage: python t4_predictive.py [--n 100] [--sims 200] [--workers 16]
"""
import argparse
import json
import sys
import time
import warnings
import zlib
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t4_data import *  # noqa: E402,F401,F403

warnings.filterwarnings("ignore")
MODELS = ["full", "noC", "nolag", "base"]
STATS = ["sessions_per_day", "night_share", "weekly_mean", "weekly_cv", "event_day_ratio"]


def sample_accounts(n, seed=SEED):
    s = pd.read_csv(T4 / "account_summary.csv")
    ok = s[(s.variant == "full") & (s.status == "ok") & (s.converged == True)]  # noqa: E712
    names = sorted(ok.account.unique())
    rng = np.random.default_rng(seed)
    pick = list(rng.choice(names, size=min(n, len(names)), replace=False))
    return pick


def dense_helpers(des):
    pos = des["pos"]
    T = des["T"]
    ld = np.full(T, -1, dtype="int64")
    ld[pos] = des["ld"]
    ld = np.maximum.accumulate(ld)
    lh = np.zeros(T, dtype="int16")
    lh[pos] = des["local_hour"]
    return ld, lh


def task(args):
    i, sims = args
    rows = []
    obs_cache = None
    for model in MODELS:
        des = account_design(i, model)
        names = des["names"]
        fit = bm.fit_nb2(des["X"], des["y"], cluster=des["cluster"])
        beta = np.where(np.isfinite(fit["beta"]), fit["beta"], 0.0)
        lagn = ("log1p_lag1", "log1p_lag2_24")
        sidx = [j for j, nm in enumerate(names) if nm not in lagn]
        p1 = beta[names.index(lagn[0])] if lagn[0] in names else 0.0
        pL = beta[names.index(lagn[1])] if lagn[1] in names else 0.0
        eta = np.full(des["T"], -40.0)
        e = des["Xall"][:, sidx] @ beta[sidx]
        eta[des["pos"][des["inoff"]]] = e[des["inoff"]]
        ld, lh = dense_helpers(des)
        if obs_cache is None:
            obs_cache = bm.daily_stats(des["ydense"], des["present"], lh, ld, G["detected"])
        stats = {s: [] for s in STATS}
        for k in range(sims):
            y = bm.simulate_nb(eta, p1, pL, fit["alpha"], SEED + zlib.crc32(f"{i}|{model}".encode()) % 99991 + 17 * k)
            st = bm.daily_stats(y, des["present"], lh, ld, G["detected"])
            for s in STATS:
                stats[s].append(st[s])
        for s in STATS:
            v = np.array(stats[s], dtype=float)
            v = v[np.isfinite(v)]
            o = obs_cache[s]
            if len(v) < 20 or not np.isfinite(o):
                continue
            pit = float((v < o).mean() + 0.5 * (v == o).mean())
            rows.append(dict(account=G["accounts"][i], model=model, stat=s, obs=o, sim_mean=float(v.mean()),
                             sim_lo=float(np.percentile(v, 2.5)), sim_hi=float(np.percentile(v, 97.5)),
                             sim_sd=float(v.std()), pit=pit, inside=bool(np.percentile(v, 2.5) <= o <= np.percentile(v, 97.5)),
                             alpha=fit["alpha"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    t0 = time.time()
    n = load_all()
    pick = sample_accounts(a.n)
    idx = {nm: k for k, nm in enumerate(G["accounts"])}
    pd.DataFrame({"account": pick}).to_csv(T4 / "predictive_sample.csv", index=False)
    rows = []
    with Pool(a.workers) as p:
        for q, r in enumerate(p.imap_unordered(task, [(idx[nm], a.sims) for nm in pick], chunksize=1)):
            rows += r
            if q % 10 == 0:
                print(q, len(pick), round(time.time() - t0), "s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(T4 / "predictive_checks.csv", index=False)
    summ = {}
    # unstable simulation: the fitted own-lag feedback is supercritical and the simulated weekly volume exceeds 3x observed
    wk = df[df.stat == "weekly_mean"].assign(ratio=lambda x: x.sim_mean / x.obs)
    unstable = {m: set(g.loc[g.ratio > 3, "account"]) for m, g in wk.groupby("model")}
    summ["unstable_accounts"] = {m: sorted(v) for m, v in unstable.items()}
    for (m, s), d in df.groupby(["model", "stat"]):
        d = d[~d.account.isin(unstable.get(m, set()))]
        rel = (d.sim_mean / d.obs - 1).replace([np.inf, -np.inf], np.nan)
        pit = d.pit.values
        summ[f"{m}|{s}"] = dict(accounts=int(len(d)), accounts_unstable_excluded=len(unstable.get(m, set())),
                                coverage_95=float(d.inside.mean()),
                                median_pred_over_obs_minus1=float(rel.median()),
                                median_abs_rel_error=float(rel.abs().median()),
                                share_within_10pct=float((rel.abs() < 0.10).mean()),
                                mean_pit=float(pit.mean()), share_pit_extreme=float(((pit < 0.025) | (pit > 0.975)).mean()),
                                share_obs_above_pred=float((d.obs > d.sim_mean).mean()),
                                median_obs=float(d.obs.median()), median_pred=float(d.sim_mean.median()))
    (T4 / "predictive_summary.json").write_text(json.dumps(summ, indent=1))
    t = pd.DataFrame({k: v for k, v in summ.items() if k != 'unstable_accounts'}).T
    pd.set_option("display.width", 250)
    print(t.round(3).to_string())
    print("done", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
