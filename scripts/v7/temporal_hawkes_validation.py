"""Hawkes fitter validation on simulated data (spec C5). Deterministic (fixed seeds).

Simulates exponential-kernel Hawkes processes by Ogata thinning with a month-level x 48-bin
hour-of-week baseline, rounds timestamps to whole seconds (as in the real data), refits with
xbox.hawkes (structured baseline) and with a constant baseline, and reports bias, RMSE and
profile-likelihood CI coverage (model-conditional; diagnostics only).
Also refits with a day-level baseline (fixed effect per day, "day_fixed") and with a day-level random effect
(Gamma(2) shrinkage prior on the day levels, "day_re") to test whether either removes the inflation.

Scenarios
  A  baseline inside the model class, 1/beta = 15 min, eta in {0,.2,.5,.8}, n in {2000,5000,20000}
  B  short kernel, 1/beta = 2 min, n = 5000
  S0/S5   sparse posting (n = 1500, about 4 posts per day) without / with day shocks (sd 0.5), eta in {0, .2, .5}
  C3/C5/C7  misspecified baseline: extra day-level random activity (sd 0.3 / 0.5 / 0.7 on the log rate),
            n = 5000, eta in {0, 0.2, 0.5}. Eta from the monthly x hour-of-week fit is then an upper bound:
            the bias reported here is the quantitative size of the upper-bound property.
Output: output/v7/temporal/hawkes_validation/{reps.csv, summary.csv, *.png}
Usage: python temporal_hawkes_validation.py [--reps 100] [--workers 16]
"""
import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from xbox import hawkes as hk  # noqa: E402

OUT = ROOT / "output/v7/temporal/hawkes_validation"
N_SLOTS = 12 * 720  # 12 "months" of 30 days
ETAS = [0.0, 0.2, 0.5, 0.8]


def true_profile():
    h = np.arange(24)
    wd = 0.15 + np.exp(1.5 * np.cos(2 * np.pi * (h - 15) / 24))
    we = 0.6 * (0.15 + np.exp(1.2 * np.cos(2 * np.pi * (h - 17) / 24)))
    p = np.concatenate([wd, we])
    return p / p.mean()


def slot_structure():
    s = np.arange(N_SLOTS)
    hour = s % 24
    day = (s // 24) % 7
    k = hour + 24 * (day >= 5)
    m = s // 720
    return m, k


def make_mu(n_target, eta, rng, daily_sd=0.0):
    m, k = slot_structure()
    lev = np.exp(rng.normal(0, 0.4, size=12))
    raw = lev[m] * true_profile()[k]
    if daily_sd > 0:
        dl = np.exp(rng.normal(0, daily_sd, size=N_SLOTS // 24))
        raw = raw * np.repeat(dl, 24)
    c = n_target * (1 - eta) / (raw.sum() * 60.0)
    return raw * c


def one(task):
    scen, n_target, eta, inv_beta, rep, seed, daily_sd = task
    rng = np.random.default_rng(seed)
    mu = make_mu(n_target, eta, rng, daily_sd)
    t = hk.simulate_hawkes(mu, eta, 1.0 / inv_beta, seed)
    t_sec = np.sort(np.floor(t * 60).astype("int64"))
    m, k = slot_structure()
    d = hk.make_design(t_sec, m, k, 0, 12, jitter_seed=seed)
    f = hk.fit(d)
    lo, hi = hk.profile_ci(d, f)
    ks = hk.time_rescaling(d, f)
    dc = hk.constant_baseline_design(d)
    fc = hk.fit(dc)
    dw = hk.make_design(t_sec, np.arange(N_SLOTS) // 168, k, 0, N_SLOTS // 168 + 1, jitter_seed=seed)
    fw = hk.fit(dw)  # sensitivity: week-level instead of month-level baseline
    if n_target <= 5000:
        dd = hk.make_design(t_sec, np.arange(N_SLOTS) // 24, k, 0, N_SLOTS // 24, jitter_seed=seed)
        eta_day = hk.fit(dd)["eta"]
        eta_dayre = hk.fit(dd, prior_kappa=2.0)["eta"]
    else:
        eta_day = eta_dayre = np.nan
    return dict(scen=scen, n_target=n_target, eta_true=eta, inv_beta_true=inv_beta, rep=rep, n=len(t),
                eta_hat=f["eta"], ci_lo=lo, ci_hi=hi, inv_beta_hat=f["inv_beta_min"],
                ks_stat=ks["ks_stat"], ks_p=ks["ks_p"], eta_hat_const=fc["eta"],
                inv_beta_hat_const=fc["inv_beta_min"], eta_hat_week=fw["eta"], eta_hat_day=eta_day, eta_hat_dayre=eta_dayre,
                n_iter=f["n_iter"], daily_sd=daily_sd)


def summarize(df):
    g = df.groupby(["scen", "n_target", "inv_beta_true", "eta_true"])
    rows = []
    for key, x in g:
        rows.append(dict(
            scen=key[0], n_target=key[1], inv_beta_true=key[2], eta_true=key[3], reps=len(x),
            mean_n=x.n.mean(),
            eta_mean=x.eta_hat.mean(), bias=(x.eta_hat - x.eta_true).mean(),
            median_bias=(x.eta_hat - x.eta_true).median(),
            rmse=np.sqrt(((x.eta_hat - x.eta_true) ** 2).mean()),
            coverage95=((x.ci_lo <= x.eta_true) & (x.eta_true <= x.ci_hi)).mean(),
            ci_width=(x.ci_hi - x.ci_lo).mean(),
            inv_beta_median=x.inv_beta_hat.median(),
            ks_stat_mean=x.ks_stat.mean(), ks_reject5=(x.ks_p < 0.05).mean(),
            const_eta_mean=x.eta_hat_const.mean(), const_bias=(x.eta_hat_const - x.eta_true).mean(),
            week_eta_mean=x.eta_hat_week.mean(), week_bias=(x.eta_hat_week - x.eta_true).mean(),
            ks_reject10=(x.ks_p < 0.10).mean(),
            day_eta_mean=x.eta_hat_day.mean(), day_bias=(x.eta_hat_day - x.eta_true).mean(),
            day_rmse=np.sqrt(((x.eta_hat_day - x.eta_true) ** 2).mean()),
            dayre_eta_mean=x.eta_hat_dayre.mean(), dayre_bias=(x.eta_hat_dayre - x.eta_true).mean(),
            dayre_rmse=np.sqrt(((x.eta_hat_dayre - x.eta_true) ** 2).mean()),
        ))
    return pd.DataFrame(rows)


def plots(df, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    a = df[df.scen == "A"]
    ns = sorted(a.n_target.unique())
    fig, axs = plt.subplots(1, len(ns), figsize=(4 * len(ns), 3.8), sharey=True)
    for ax, n in zip(np.atleast_1d(axs), ns):
        x = a[a.n_target == n]
        data = [x[x.eta_true == e].eta_hat.values for e in ETAS]
        ax.boxplot(data, positions=range(len(ETAS)), widths=0.5, showfliers=False)
        ax.plot(range(len(ETAS)), ETAS, "r_", ms=18, mew=2, label="truth")
        ax.set_xticks(range(len(ETAS)))
        ax.set_xticklabels([str(e) for e in ETAS])
        ax.set_title(f"n about {n}")
        ax.set_xlabel("true eta")
    np.atleast_1d(axs)[0].set_ylabel("estimated eta (structured baseline)")
    np.atleast_1d(axs)[0].legend()
    fig.tight_layout()
    fig.savefig(out / "recovery_eta.png", dpi=150)
    plt.close(fig)

    p = df[(df.scen == "A") & (df.eta_true == 0.0)]
    fig, ax = plt.subplots(figsize=(5, 3.8))
    pos = np.arange(len(ns))
    ax.boxplot([p[p.n_target == n].eta_hat_const.values for n in ns], positions=pos - 0.18, widths=0.3,
               showfliers=False, boxprops=dict(color="tab:red"), medianprops=dict(color="tab:red"))
    ax.boxplot([p[p.n_target == n].eta_hat.values for n in ns], positions=pos + 0.18, widths=0.3,
               showfliers=False, boxprops=dict(color="tab:blue"), medianprops=dict(color="tab:blue"))
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(pos)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("events (true eta = 0, inhomogeneous Poisson)")
    ax.set_ylabel("estimated eta")
    ax.set_title("red: constant baseline, blue: month x hour-of-week baseline", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "constant_baseline_inflation.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = []
    cell = 0
    for n in (2000, 5000, 20000):
        for e in ETAS:
            cell += 1
            reps = a.reps if n < 20000 else max(40, a.reps // 2)
            tasks += [("A", n, e, 15.0, r, 100000 * cell + r, 0.0) for r in range(reps)]
    for e in ETAS:
        cell += 1
        tasks += [("B", 5000, e, 2.0, r, 100000 * cell + r, 0.0) for r in range(max(40, a.reps // 2))]
    for scen, sd in (("C3", 0.3), ("C5", 0.5), ("C7", 0.7)):
        for e in (0.0, 0.2, 0.5):
            cell += 1
            tasks += [(scen, 5000, e, 15.0, r, 100000 * cell + r, sd) for r in range(max(40, a.reps // 2))]
    # sparse posting (about 4 posts per day, like a typical member): day-level baselines have little data per day
    for scen, sd in (("S0", 0.0), ("S5", 0.5)):
        for e in (0.0, 0.2, 0.5):
            cell += 1
            tasks += [(scen, 1500, e, 15.0, r, 100000 * cell + r, sd) for r in range(max(40, a.reps // 2))]
    t0 = time.time()
    # warm numba cache in the parent so workers do not race on compilation
    one(("A", 500, 0.2, 15.0, 0, 1, 0.0))
    with Pool(a.workers) as p:
        res = p.map(one, tasks, chunksize=4)
    df = pd.DataFrame(res)
    df.to_csv(OUT / "reps.csv", index=False)
    s = summarize(df)
    s.to_csv(OUT / "summary.csv", index=False)
    plots(df, OUT)
    print(s.round(3).to_string())
    print("elapsed", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
