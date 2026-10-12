"""T4b sensitivity: 15-minute bins for the 100-account predictive sample, with the review fixes (member-level leave-one-out
Congress term, absent_in_session dropped, lag stability cap at |phi_1| + |phi_24| <= 0.9). Variants: full, noC.
Bins: 15 minutes in the account's local time; own lags are the previous 15-minute bin and the sum over the 2nd to 96th previous bins.
Outputs: output/v7/t4b/sens15_fits.csv, sens15_summary.json
"""
import argparse
import json
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t4b_data import *  # noqa: E402,F401,F403

warnings.filterwarnings("ignore")
TERMS = ["intercept", "log1p_congress", "log1p_lag1", "log1p_lag2_24"] + REG_B


def design15(i, variant):
    d = account_arrays(i)
    bg = G["bioguide"][i]
    acct = G["accounts"][i]
    hrs = d["hour_idx"]
    n = len(hrs)
    lo = int(hrs.min())
    ev15 = G["ev15"][acct]
    T15 = 4 * (int(hrs.max()) - lo + 1)
    y15 = np.bincount(ev15 - 4 * lo, minlength=T15)[:T15].astype(float)
    hr_rep = np.repeat(hrs, 4)
    q = np.tile(np.arange(4), n)
    b15 = 4 * hr_rep + q
    lh = np.repeat(d["local_hour"], 4)
    ldow = np.repeat(d["local_dow"], 4)
    ld = bm.local_day_from_hours(hr_rep, lh)
    slot = lh.astype(int) * 4 + q
    cell = (ldow >= 5).astype(int) * 96 + slot
    mem = np.bincount(G["mem15"].get(bg, np.zeros(0, dtype="int64")), minlength=4 * G["nh"])
    loo = G["tot15"][b15 - 4 * G["hour0"]] - mem[b15 - 4 * G["hour0"]]
    assert loo.min() >= 0
    logc = np.log1p(loo.astype(float))
    congress = "plain" if variant == "full" else "none"
    Xs, names, inoff = bm.build_static(cell, ld, logc, G["cal"].get(bg), G["detected"], 4, congress)
    j = names.index("absent_in_session")
    Xs = np.delete(Xs, j, axis=1)
    names = names[:j] + names[j + 1:]
    pos = (b15 - 4 * lo).astype(int)
    present = np.zeros(T15, dtype=bool)
    present[pos[inoff]] = True
    lag1, lagL, valid = bm.lag_features(y15, present, 96)
    fit_rows = valid[pos] & inoff
    ncore = 1 + 191 + (1 if congress == "plain" else 0)
    X = np.hstack([Xs[:, :ncore], np.column_stack([lag1[pos], lagL[pos]]), Xs[:, ncore:]])[fit_rows]
    nms = names[:ncore] + list(LAGN) + names[ncore:]
    return dict(X=X, names=nms, y=y15[pos][fit_rows], cluster=(b15[fit_rows] // 672), use_lags=True)


def task(i):
    rows = []
    for v in ("full", "noC"):
        des = design15(i, v)
        r = fit_account(des)
        fit = r["fit"]
        for t in TERMS:
            if t in des["names"]:
                j = des["names"].index(t)
                rows.append(dict(account=G["accounts"][i], variant=v, term=t, beta=fit["beta"][j],
                                 se_model=fit["se_model"][j], se_robust=fit["se_robust"][j], alpha=fit["alpha"],
                                 ll=fit["ll"], n_bins=fit["n"], capped=r["capped"], branching_free=r["branching_free"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    t0 = time.time()
    load_all_b()
    m, hour0, obs, cum = load_mask()
    ev = pd.read_parquet(T1 / "events.parquet", columns=["screen_name", "bioguide", "t"])
    b = ev.t.values // 900
    G["tot15"] = np.bincount(b - 4 * hour0, minlength=4 * len(m)).astype("int64")
    G["ev15"] = {k: g.t.values // 900 for k, g in ev.groupby("screen_name")}
    G["mem15"] = {str(k): (g.t.values // 900 - 4 * hour0).astype("int64") for k, g in ev.groupby("bioguide")}
    pick = list(pd.read_csv(T4 / "predictive_sample.csv").account)[:a.n]
    idx = {nm: k for k, nm in enumerate(G["accounts"])}
    order = sorted(pick, key=lambda nm: -(G["ends"][idx[nm]] - G["starts"][idx[nm]]))
    rows = []
    with Pool(a.workers) as p:
        for q, r in enumerate(p.imap_unordered(task, [idx[nm] for nm in order], chunksize=1)):
            rows += r
            if q % 5 == 0:
                print(q, len(order), round(time.time() - t0), "s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(T4B / "sens15_fits.csv", index=False)
    f60 = pd.read_csv(T4B / "account_fits.csv")
    f60 = f60[f60.account.isin(pick) & f60.variant.isin(["full", "noC"])]
    out = {}
    from scipy.stats import spearmanr
    for v in ("full", "noC"):
        for t in TERMS[1:]:
            x = df[(df.variant == v) & (df.term == t)].set_index("account")
            y = f60[(f60.variant == v) & (f60.term == t)].set_index("account")
            j = x[["beta", "se_robust"]].join(y[["beta", "se_robust"]], lsuffix="_15", rsuffix="_60").dropna()
            j = j[(j.se_robust_15 < 2) & (j.se_robust_60 < 2) & (j.se_robust_15 > 0) & (j.se_robust_60 > 0)]
            if len(j) < 10:
                continue
            r15 = bm.meta_reg(j.beta_15.values, j.se_robust_15.values)
            r60 = bm.meta_reg(j.beta_60.values, j.se_robust_60.values)
            f = np.log(2) if t.startswith("log1p") else 1.0
            out[f"{v}|{t}"] = dict(k=int(len(j)), mu_15=float(r15["coef"][0]), se_15=float(r15["se"][0]), tau_15=r15["tau"],
                                   mu_60=float(r60["coef"][0]), se_60=float(r60["se"][0]), tau_60=r60["tau"],
                                   spearman_account_coef=float(spearmanr(j.beta_15, j.beta_60)[0]),
                                   RR_15=float(np.exp(r15["coef"][0] * f)), RR_60=float(np.exp(r60["coef"][0] * f)))
    summ = dict(accounts=len(pick), bin_minutes=15, pooled_vs_60min=out,
                median_alpha_15min=df[df.term == "intercept"].groupby("variant").alpha.median().to_dict(),
                note="lag and Congress coefficients live on different time scales at 15 and 60 minutes; regime coefficients should agree",
                elapsed_s=round(time.time() - t0))
    (T4B / "sens15_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1)[:2500])


if __name__ == "__main__":
    main()
