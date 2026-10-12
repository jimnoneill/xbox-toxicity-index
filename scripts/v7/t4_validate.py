"""T4 validation of the estimator on simulated counts with known coefficients (bias, coverage), plus validation of the
random-effects pooling (DerSimonian-Laird and REML) on simulated account coefficients.

Part 1: for three real accounts (low, median, high volume) the real design (hour x weekday cells, Congress term, regime
flags, observation gaps) is kept, the coefficients are set to known values, counts are simulated recursively (own lags
feed back), and the NB model is refitted R times.
  scenario A: truth = coefficients fitted to the real account (realistic mix).
  scenario B: truth = fitted regime coefficients, but delta = 0.25, phi_1 = 0.30, phi_24 = 0.15 (round known values);
              intercept re-tuned so the simulated event count matches the real one.
Reports bias, RMSE, coverage of 95% Wald intervals with model-based and week-clustered robust SEs, SE calibration.
Part 2: pooling validation (K accounts, true mean 0.2, tau in {0, .05, .15}, SEs resampled from real fits if present).
Outputs: output/v7/t4/validation_nb.csv, validation_nb_summary.csv, validation_meta.csv, validation_summary.json
Usage: python t4_validate.py [--reps 200] [--workers 16]
"""
import argparse
import json
import sys
import time
import zlib
import warnings
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t4_data import *  # noqa: E402,F401,F403

warnings.filterwarnings("ignore")
TERMS = ["log1p_congress", "log1p_lag1", "log1p_lag2_24", "in_session", "absent_in_session", "primary_window", "general_window",
         "post_election", "aligned_president", "party_majority", "event_curated", "event_detected", "intercept"]
TRUTH = {}


def make_truth(i, scenario):
    des = account_design(i, "full")
    fit = bm.fit_nb2(des["X"], des["y"], cluster=des["cluster"])
    beta = np.where(np.isfinite(fit["beta"]), fit["beta"], 0.0)
    names = des["names"]
    if scenario == "B":
        beta[names.index("log1p_congress")] = 0.25
        beta[names.index("log1p_lag1")] = 0.30
        beta[names.index("log1p_lag2_24")] = 0.15
    keep = fit["keep"]
    alpha = fit["alpha"]
    target = float(des["y"].sum())
    p1, pL = beta[names.index("log1p_lag1")], beta[names.index("log1p_lag2_24")]
    # Xall columns: same order as names (cells, congress, lags, regimes)
    Xs = np.delete(des["Xall"], [names.index("log1p_lag1"), names.index("log1p_lag2_24")], axis=1)
    sidx = [j for j in range(len(names)) if names[j] not in ("log1p_lag1", "log1p_lag2_24")]
    for it in range(5):
        eta = np.full(des["T"], -40.0)
        e = Xs @ beta[sidx]
        eta[des["pos"][des["inoff"]]] = e[des["inoff"]]
        ys = bm.simulate_nb(eta, p1, pL, alpha, seed=SEED + it)
        got = ys[des["pos"]][des["fit_rows"]].sum()
        beta[0] += float(np.log(target / max(got, 1.0))) * 0.9
        if abs(got / target - 1) < 0.03:
            break
    eta = np.full(des["T"], -40.0)
    e = Xs @ beta[sidx]
    eta[des["pos"][des["inoff"]]] = e[des["inoff"]]
    return dict(i=i, names=names, beta=beta, keep=keep, alpha=alpha, eta0=eta, p1=p1, pL=pL,
                account=G["accounts"][i], scenario=scenario)


def rep_task(args):
    key, r = args
    tr = TRUTH[key]
    y = bm.simulate_nb(tr["eta0"], tr["p1"], tr["pL"], tr["alpha"], seed=SEED % 1000003 + zlib.crc32(key.encode()) % 100003 + 7919 * r)
    des = account_design(tr["i"], "full", y_override=y)
    f = bm.fit_nb2(des["X"], des["y"], cluster=des["cluster"])
    rows = []
    for t in TERMS:
        j = tr["names"].index(t)
        if not np.isfinite(f["beta"][j]) or j not in set(tr["keep"]):
            continue
        rows.append(dict(account=tr["account"], scenario=tr["scenario"], rep=r, term=t, truth=tr["beta"][j],
                         est=f["beta"][j], se_model=f["se_model"][j], se_robust=f["se_robust"][j]))
    rows.append(dict(account=tr["account"], scenario=tr["scenario"], rep=r, term="log_alpha",
                     truth=np.log(tr["alpha"]), est=np.log(f["alpha"]), se_model=np.nan, se_robust=np.nan))
    return rows


def meta_validation(rng, se_pool):
    out = []
    for K in (100, 400, 1000):
        for tau in (0.0, 0.05, 0.15):
            res = {m: dict(mu=[], cov=[], tau=[], tcov=[]) for m in ("DL", "REML")}
            for _ in range(500):
                se = rng.choice(se_pool, K)
                th = 0.2 + rng.normal(0, tau, K)
                y = th + rng.normal(0, se)
                for m in ("DL", "REML"):
                    r = bm.meta_reg(y, se, method=m)
                    mu, s = r["coef"][0], r["se"][0]
                    res[m]["mu"].append(mu)
                    res[m]["cov"].append(abs(mu - 0.2) < 1.96 * s)
                    res[m]["tau"].append(r["tau"])
            for m in res:
                out.append(dict(K=K, tau_true=tau, method=m, bias_mu=float(np.mean(res[m]["mu"]) - 0.2),
                                coverage_mu=float(np.mean(res[m]["cov"])), mean_tau_hat=float(np.mean(res[m]["tau"])),
                                bias_tau=float(np.mean(res[m]["tau"]) - tau)))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    t0 = time.time()
    n = load_all()
    ev = np.array([G["count"][G["starts"][i]:G["ends"][i]].sum() for i in range(n)])
    ok = np.flatnonzero((ev >= 600) & np.array([G["bioguide"][i] in G["cal"] for i in range(n)]))
    order = ok[np.argsort(ev[ok])]
    picks = [int(order[int(q * (len(order) - 1))]) for q in (0.1, 0.5, 0.9)]
    print("accounts", [(G["accounts"][i], int(ev[i])) for i in picks], flush=True)
    keys = []
    for i in picks:
        for sc in ("A", "B"):
            k = f"{G['accounts'][i]}|{sc}"
            TRUTH[k] = make_truth(i, sc)
            keys.append(k)
    print("truth built", round(time.time() - t0), "s", flush=True)
    tasks = [(k, r) for k in keys for r in range(a.reps)]
    rows = []
    with Pool(a.workers) as p:
        for q, r in enumerate(p.imap_unordered(rep_task, tasks, chunksize=2)):
            rows += r
            if q % 100 == 0:
                print(q, len(tasks), round(time.time() - t0), "s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(T4 / "validation_nb.csv", index=False)
    df["err"] = df.est - df.truth
    for s in ("model", "robust"):
        df[f"cov_{s}"] = (df.err.abs() < 1.96 * df[f"se_{s}"]).astype(float)
    g = df.groupby(["scenario", "account", "term"]).agg(
        truth=("truth", "first"), bias=("err", "mean"), rmse=("err", lambda x: float(np.sqrt(np.mean(x ** 2)))),
        sd_est=("est", "std"), mean_se_model=("se_model", "mean"), mean_se_robust=("se_robust", "mean"),
        cov_model=("cov_model", "mean"), cov_robust=("cov_robust", "mean"), reps=("est", "size")).reset_index()
    g.to_csv(T4 / "validation_nb_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(g.round(3).to_string())
    # headline per scenario/term pooled over accounts
    h = df[df.term != "log_alpha"].groupby(["scenario", "term"]).agg(
        bias=("err", "mean"), cov_model=("cov_model", "mean"), cov_robust=("cov_robust", "mean")).reset_index()
    # pooling validation: SEs from real robust SEs if available, else a plausible range
    se_pool = np.linspace(0.02, 0.12, 50)
    p_ = T4 / "account_fits.csv"
    if p_.exists():
        f = pd.read_csv(p_)
        s = f[(f.variant == "full") & (f.term == "log1p_congress")].se_robust.dropna().values
        if len(s) > 50:
            se_pool = s
    rng = np.random.default_rng(SEED)
    mv = meta_validation(rng, se_pool)
    mv.to_csv(T4 / "validation_meta.csv", index=False)
    print(mv.round(4).to_string())
    summ = dict(reps=a.reps, accounts=[G["accounts"][i] for i in picks],
                events=[int(ev[i]) for i in picks],
                coverage_by_term=h.round(4).to_dict("records"),
                worst_abs_bias=float(g.bias.abs().max()),
                coverage_model_mean=float(g.cov_model.mean()), coverage_robust_mean=float(g.cov_robust.mean()),
                coverage_robust_min=float(g.cov_robust.min()), coverage_model_min=float(g.cov_model.min()),
                key_terms=g[g.term.isin(["log1p_congress", "log1p_lag1", "log1p_lag2_24"])].round(4).to_dict("records"),
                meta=mv.round(4).to_dict("records"), elapsed_s=round(time.time() - t0))
    (T4 / "validation_summary.json").write_text(json.dumps(summ, indent=1, default=float))


if __name__ == "__main__":
    main()
