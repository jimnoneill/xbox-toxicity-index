"""T4b predictive checks, in-sample and held-out, for the same 100 accounts as T4 (predictive_sample.csv).
Models: full (member-LOO Congress term, lags capped for stability) and noC (no Congress term).
Periods:
  insample : fit on all data, simulate the whole account span (labelled IN-SAMPLE).
  heldout  : fit on hours before 2022-01-01 (UTC), simulate 2022-01-01 to 2023-07-11 starting from the observed last 24 bins
             before the window (labelled HELD-OUT). The full model is conditional on the observed Congress-wide counts and
             calendar of the window; noC is not.
Simulation modes: plugin (fitted coefficients fixed) and param (coefficients drawn from N(beta, robust covariance) for every
simulation, so parameter uncertainty is propagated). 200 simulations per account, model, period and mode.
Quantities: sessions/day (runs of non-empty hours), night share (local 00-06), weekly volume mean, weekly CV, event-day response.
Coverage = share of accounts whose observed value lies in the central 95% interval of the simulated values.
Outputs: output/v7/t4b/predictive_checks.csv, predictive_summary.json
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
from t4b_data import *  # noqa: E402,F401,F403

warnings.filterwarnings("ignore")
MODELS = ["full", "noC"]
STATS = ["sessions_per_day", "night_share", "weekly_mean", "weekly_cv", "event_day_ratio"]


def dense_helpers(des):
    pos, T = des["pos"], des["T"]
    ld = np.full(T, -1, dtype="int64")
    ld[pos] = des["ld"]
    ld = np.maximum.accumulate(ld)
    lh = np.zeros(T, dtype="int16")
    lh[pos] = des["local_hour"]
    return ld, lh


def chol(cov):
    c = cov + np.diag(1e-10 * np.diag(cov) + 1e-14)
    try:
        return np.linalg.cholesky(c)
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(c)
        return V * np.sqrt(np.clip(w, 0, None))


def task(args):
    i, sims = args
    rows = []
    acct = G["accounts"][i]
    for period in ("insample", "heldout"):
        for model in MODELS:
            try:
                end_hour = HOUR_2022 if period == "heldout" else None
                des = account_design_b(i, model, end_hour=end_hour)
                if len(des["y"]) < 1000 or des["y"].sum() < MIN_EVENTS_T4:
                    continue
                r = fit_account(des)
            except Exception:
                continue
            fit, cov = r["fit"], r["cov"]
            beta = np.where(np.isfinite(fit["beta"]), fit["beta"], 0.0)
            ld, lh = dense_helpers(des)
            lo = des["lo"]
            if period == "heldout":
                s0 = max(HOUR_2022 - lo - 24, 0)
                n0 = max(HOUR_2022 - lo - s0, 0)
                s1 = min(HOUR_END_HELD - lo, des["T"])
                if s1 - (s0 + n0) < 24 * 60:
                    continue
            else:
                s0, n0, s1 = 0, 0, des["T"]
            sl = slice(s0, s1)
            a_ = slice(s0 + n0, s1)
            pres = des["present"]
            if period == "heldout" and pres[a_].sum() < 24 * 60:
                continue
            obs = bm.daily_stats(des["ydense"][a_], pres[a_], lh[a_], ld[a_], G["detected"])
            if not np.isfinite(obs["night_share"]):
                continue
            y0 = des["ydense"][sl]
            for mode in ("plugin", "param"):
                stats = {s: [] for s in STATS}
                rng = np.random.default_rng(SEED + zlib.crc32(f"{i}|{model}|{period}|{mode}".encode()) % 1000003)
                tot_ratio = []
                Lc = chol(cov) if mode == "param" else None
                for k in range(sims):
                    b = beta if mode == "plugin" else beta + Lc @ rng.standard_normal(len(beta))
                    eta = eta_static(des, b)
                    p1, pL = lag_coefs(des, b)
                    ys = bmb.simulate_nb2(eta[sl], p1, pL, fit["alpha"],
                                          SEED + 7919 * k + zlib.crc32(f"{i}{model}".encode()) % 9973, y0=y0, n0=n0)
                    ys = ys[n0:]
                    st = bm.daily_stats(ys, pres[a_], lh[a_], ld[a_], G["detected"])
                    for s in STATS:
                        stats[s].append(st[s])
                    tot_ratio.append(ys[pres[a_]].sum() / max(des["ydense"][a_][pres[a_]].sum(), 1.0))
                unstable = bool(np.median(tot_ratio) > 3)
                for s in STATS:
                    v = np.array(stats[s], dtype=float)
                    v = v[np.isfinite(v)]
                    o = obs[s]
                    if len(v) < 20 or not np.isfinite(o):
                        continue
                    lo_, hi_ = np.percentile(v, [2.5, 97.5])
                    pit = float((v < o).mean() + 0.5 * (v == o).mean())
                    rows.append(dict(account=acct, period=period, model=model, mode=mode, stat=s, obs=o,
                                     sim_mean=float(v.mean()), sim_median=float(np.median(v)), sim_lo=float(lo_),
                                     sim_hi=float(hi_), pit=pit, inside=bool(lo_ <= o <= hi_), capped=r["capped"],
                                     branching_free=r["branching_free"], unstable=unstable,
                                     sim_total_ratio_median=float(np.median(tot_ratio))))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    load_all_b()
    T4B.mkdir(parents=True, exist_ok=True)
    pick = list(pd.read_csv(T4 / "predictive_sample.csv").account)
    if a.limit:
        pick = pick[:a.limit]
    idx = {nm: k for k, nm in enumerate(G["accounts"])}
    rows = []
    with Pool(a.workers) as p:
        for q, r in enumerate(p.imap_unordered(task, [(idx[nm], a.sims) for nm in pick], chunksize=1)):
            rows += r
            if q % 10 == 0:
                print(q, len(pick), round(time.time() - t0), "s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(T4B / "predictive_checks.csv", index=False)
    summ = {}
    for (per, m, mode, s), d in df.groupby(["period", "model", "mode", "stat"]):
        rel = (d.sim_median / d.obs - 1).replace([np.inf, -np.inf], np.nan)
        summ[f"{per}|{m}|{mode}|{s}"] = dict(
            label="IN-SAMPLE" if per == "insample" else "HELD-OUT 2022-01-01..2023-07-11", accounts=int(len(d)),
            coverage_95=float(d.inside.mean()), median_pred_over_obs_minus1=float(rel.median()),
            median_abs_rel_error=float(rel.abs().median()), mean_pit=float(d.pit.mean()),
            share_pit_extreme=float(((d.pit < 0.025) | (d.pit > 0.975)).mean()),
            share_obs_above_pred_median=float((d.obs > d.sim_median).mean()), median_obs=float(d.obs.median()),
            median_pred=float(d.sim_median.median()), unstable_accounts=int(d.unstable.sum()))
    st = df[(df["mode"] == "plugin") & (df.stat == "weekly_mean")]
    summ["unstable_accounts"] = {f"{per}|{m}": sorted(g.loc[g.unstable, "account"]) for (per, m), g in st.groupby(["period", "model"])}
    summ["accounts_evaluated"] = {f"{per}|{m}": int(g.account.nunique()) for (per, m), g in df.groupby(["period", "model"])}
    (T4B / "predictive_summary.json").write_text(json.dumps(summ, indent=1))
    t = pd.DataFrame({k: v for k, v in summ.items() if k.count("|") == 3}).T
    pd.set_option("display.width", 250)
    print(t[["accounts", "coverage_95", "median_pred_over_obs_minus1", "mean_pit", "unstable_accounts"]].to_string())
    print(json.dumps(summ["unstable_accounts"], indent=1))
    print("done", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
