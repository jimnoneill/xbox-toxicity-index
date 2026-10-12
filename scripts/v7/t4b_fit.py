"""T4b: per-account NB fits with the review fixes (member-level LOO Congress term, absent_in_session dropped, lag cap,
optional time trend). Variants: full (capped where free branching > CAP), nolag, noC, base, csplit, noreg, trend.
"full_uncapped" rows keep the free lag fits. Stability: two plug-in simulations of the final full model; unstable when the
simulated total exceeds 3x the observed total.
Outputs (output/v7/t4b/): account_fits.csv, account_summary.csv
Usage: python t4b_fit.py [--workers 16] [--variants ...]
"""
import argparse
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t4b_data import *  # noqa: E402,F401,F403

warnings.filterwarnings("ignore")
FIT_TERMS = ["intercept", "log1p_congress", "log1p_congress_daymean", "log1p_congress_dev", "log1p_lag1",
             "log1p_lag2_24"] + REG_B


def rows_for(acct, v, names, fit):
    out = []
    for t in FIT_TERMS:
        if t in names:
            j = names.index(t)
            out.append(dict(account=acct, variant=v, term=t, beta=fit["beta"][j], se_model=fit["se_model"][j],
                            se_robust=fit["se_robust"][j]))
    for j, nm in enumerate(names):
        if nm.startswith("trend"):
            out.append(dict(account=acct, variant=v, term=nm, beta=fit["beta"][j], se_model=fit["se_model"][j],
                            se_robust=fit["se_robust"][j]))
    return out


def task(args):
    i, variants = args
    acct = G["accounts"][i]
    rows, summ = [], []
    for v in variants:
        try:
            des = account_design_b(i, v)
            n_ev = float(des["y"].sum())
            if len(des["y"]) < 1000 or n_ev < MIN_EVENTS_T4 or not des["cal_ok"]:
                summ.append(dict(account=acct, variant=v, status="skipped", n_bins=len(des["y"]), n_events=n_ev,
                                 cal_ok=des["cal_ok"]))
                break
            r = fit_account(des)
            fit = r["fit"]
        except Exception as e:
            summ.append(dict(account=acct, variant=v, status="error:" + str(e)[:80]))
            continue
        names = des["names"]
        rows += rows_for(acct, v, names, fit)
        sim_ratio = np.nan
        if v == "full":
            rows += rows_for(acct, "full_uncapped", names, r["uncapped"])
            beta = fit["beta"]
            eta = eta_static(des, beta)
            p1, pL = lag_coefs(des, beta)
            obs_tot = float(des["ydense"][des["present"]].sum())
            rat = []
            for k in range(2):
                ys = bmb.simulate_nb2(eta, p1, pL, fit["alpha"], SEED + 31 * k + i)
                rat.append(float(ys[des["present"]].sum() / max(obs_tot, 1.0)))
            sim_ratio = max(rat)
        p1f = p2f = np.nan
        if des["use_lags"]:
            p1f = r["uncapped"]["beta"][names.index(LAGN[0])]
            p2f = r["uncapped"]["beta"][names.index(LAGN[1])]
        summ.append(dict(account=acct, variant=v, status="ok", n_bins=fit["n"], n_events=n_ev, alpha=fit["alpha"],
                         ll=fit["ll"], converged=fit["converged"], n_clusters=fit["n_clusters"], n_cols=len(names),
                         n_dropped=len(names) - len(fit["keep"]), mean_mu=fit["mean_mu"],
                         bioguide=G["bioguide"][i], party=G["party"][i], chamber=G["chamber"][i],
                         account_type=G["atype"][i], branching_free=r["branching_free"], elasticity_sum_free=r["elasticity_sum_free"], capped=r["capped"], capped_fixed=r["capped_fixed"],
                         phi1_free=p1f, phi24_free=p2f, sim_ratio_max=sim_ratio, ll_free=r["uncapped"]["ll"]))
    return rows, summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--variants", default="full,nolag,noC,base,csplit,noreg,trend")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    t0 = time.time()
    T4B.mkdir(parents=True, exist_ok=True)
    n = load_all_b()
    variants = a.variants.split(",")
    idx = list(range(n))
    if a.limit:
        idx = idx[:a.limit]
    idx.sort(key=lambda i: -(G["ends"][i] - G["starts"][i]))
    print("accounts", n, "loaded", round(time.time() - t0), "s", flush=True)
    R, S = [], []
    with Pool(a.workers) as p:
        for k, (r, s) in enumerate(p.imap_unordered(task, [(i, variants) for i in idx], chunksize=1)):
            R += r
            S += s
            if k % 50 == 0:
                print(k, len(idx), round(time.time() - t0), "s", flush=True)
    tag = a.out
    pd.DataFrame(R).to_csv(T4B / f"account_fits{tag}.csv", index=False)
    pd.DataFrame(S).to_csv(T4B / f"account_summary{tag}.csv", index=False)
    print("done", len(R), len(S), round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()
