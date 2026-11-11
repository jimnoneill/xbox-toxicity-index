"""Shared-activity models, misfit diagnostics and the full bootstrap for member event streams.

Stages (python run_shared.py STAGE [--workers 16]); everything is written to output/v7/temporal_events/shared and
nothing else is modified.
  prep  hourly counts of events started by all members (events.parquet) and each member's first slot
  fit   per member: exponential Hawkes (reference), Hawkes with the shared driver, Poisson with and without the
        driver; lag-1 autocorrelation and location of the largest gap of the rescaled intervals; refits per
        calendar year 2018 to 2022 (members_shared.csv, years_shared.csv)
  sim   synthetic populations with day-level shocks that are shared, private, or both: bias of the branching ratio
        with and without the shared driver (sim_shared.csv)
  boot  parametric bootstrap of the time-rescaling test for every member (bootstrap_all.csv; --models, --resume)
"""
import argparse
import json
import pickle
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import run_full as rf

ROOT = rf.ROOT
sys.path.insert(0, str(ROOT))
from xbox import hawkes as hk  # noqa: E402
from xbox import pointprocess_models as pm  # noqa: E402
import shared_models as sm  # noqa: E402

warnings.filterwarnings("ignore")
EV = ROOT / "output/v7/temporal_events/model_compare"
OUT = ROOT / "output/v7/temporal_events/shared"
CACHE, HOURLY, SLOT0 = {}, {}, {}
YEARS = [2018, 2019, 2020, 2021, 2022]
MIN_YEAR_N = 150


def stage_prep(a):
    import temporal_features as tf
    lo, hi = tf.WINDOWS["full"]
    ev = pd.read_parquet(ROOT / "output/v7/t1full/events.parquet", columns=["bioguide", "t"])
    ev = ev[(ev.t >= lo) & (ev.t < hi)]
    h0 = int(ev.t.min() // 3600 * 3600)
    hh = ((ev.t.values - h0) // 3600).astype("int64")
    np.save(OUT / "congress_hourly.npy", np.bincount(hh))
    cache = pickle.load(open(EV / "cache.pkl", "rb"))
    first = ev.groupby("bioguide").t.min()
    own = {}
    for bg in cache:
        own[bg] = np.bincount(hh[ev.bioguide.values == bg], minlength=hh.max() + 1).astype("int32")
    pickle.dump(own, open(OUT / "own_hourly.pkl", "wb"))
    json.dump(dict(h0=h0, slot0={bg: int(first[bg] // 3600 * 3600) for bg in cache}), open(OUT / "slot0.json", "w"))
    print("prep done", len(cache), "members;", int(np.bincount(hh).sum()), "events", flush=True)


def load():
    CACHE.update(pickle.load(open(EV / "cache.pkl", "rb")))
    CACHE.pop("LEE", None)
    j = json.load(open(OUT / "slot0.json"))
    SLOT0.update(j["slot0"])
    HOURLY["h0"] = j["h0"]
    HOURLY["all"] = np.load(OUT / "congress_hourly.npy")
    HOURLY["own"] = pickle.load(open(OUT / "own_hourly.pkl", "rb"))


def others_log(bg, ns):
    """log(1 + events started by all other members) for each of the member's one-hour slots."""
    i0 = (SLOT0[bg] - HOURLY["h0"]) // 3600
    tot, own = HOURLY["all"], HOURLY["own"][bg]
    idx = i0 + np.arange(ns)
    ok = (idx >= 0) & (idx < len(tot))
    x = np.zeros(ns)
    x[ok] = tot[idx[ok]] - own[idx[ok]]
    return np.log1p(np.maximum(x, 0))


def member(bg):
    d = pm.calendar_from_cache(CACHE[bg])
    return d, sm.standardize(others_log(bg, d.ns))


def year_data(bg, d, zraw, y):
    ts = SLOT0[bg] + 3600 * np.arange(d.ns, dtype="int64")
    y0, y1 = pd.Timestamp(f"{y}-01-01", tz="UTC").timestamp(), pd.Timestamp(f"{y + 1}-01-01", tz="UTC").timestamp()
    idx = np.nonzero((ts >= y0) & (ts < y1))[0]
    if len(idx) < 24 * 200:
        return None
    s0, s1 = idx[0], idx[-1] + 1
    t = d.t[(d.t >= s0 * 60.0) & (d.t < s1 * 60.0)] - s0 * 60.0
    if len(t) < MIN_YEAR_N:
        return None
    mo = np.unique(d.slot_month[s0:s1], return_inverse=True)[1]
    return pm.Data(t, mo, d.slot_bin[s0:s1], int(mo.max()) + 1), sm.standardize(zraw[s0:s1])


def row(bg, model, r, extra=None):
    loc, sign = sm.ks_location(r["u"])
    out = dict(bg=bg, model=model, n=len(r["u"]) + 1, ll=r["ll"], k=r["k"], aic=r["aic"], bic=r["bic"], D=r["D"], p=r["p"],
               lag1=sm.lag1(r["u"]), ks_at_u=loc, ks_sign=sign, **{a: float(b) for a, b in r["params"].items()})
    out.update(extra or {})
    return out


def fit_task(bg):
    try:
        d, z = member(bg)
        zraw = others_log(bg, d.ns)
        rows = [row(bg, "hawkes_exp", pm.fit_model("hawkes_exp", d)), row(bg, "hawkes_shared", sm.fit_shared(d, z)),
                row(bg, "poisson_shared", sm.fit_shared(d, z, poisson=True)),
                row(bg, "poisson", sm.fit_shared(d, z, poisson=True, fix_gamma=0.0))]
        yrows = []
        for y in YEARS:
            yd = year_data(bg, d, zraw, y)
            if yd is None:
                continue
            dy, zy = yd
            yrows.append(row(bg, "hawkes_exp", pm.fit_model("hawkes_exp", dy), dict(year=y)))
            yrows.append(row(bg, "hawkes_shared", sm.fit_shared(dy, zy), dict(year=y)))
        return rows, yrows
    except Exception as e:  # noqa
        return [dict(bg=bg, err=str(e)[:200])], []


def stage_fit(a):
    load()
    bgs = sorted(CACHE, key=lambda b: -CACHE[b]["n"])
    fit_task(bgs[-1])
    t0 = time.time()
    R, Y = [], []
    with Pool(a.workers, maxtasksperchild=8) as p:
        for i, (r, y) in enumerate(p.imap_unordered(fit_task, bgs, chunksize=1)):
            R += r
            Y += y
            if i % 50 == 0:
                print(i, len(bgs), round(time.time() - t0), flush=True)
                pd.DataFrame(R).to_csv(OUT / "members_shared.csv", index=False)
                pd.DataFrame(Y).to_csv(OUT / "years_shared.csv", index=False)
    pd.DataFrame(R).to_csv(OUT / "members_shared.csv", index=False)
    pd.DataFrame(Y).to_csv(OUT / "years_shared.csv", index=False)
    print("fit done", round(time.time() - t0), flush=True)


# ---------------------------------------------------------------- synthetic populations with shared shocks
N_SLOTS = 12 * 720


def _profile():
    h = np.arange(24)
    wd = 0.15 + np.exp(1.5 * np.cos(2 * np.pi * (h - 15) / 24))
    we = 0.6 * (0.15 + np.exp(1.2 * np.cos(2 * np.pi * (h - 17) / 24)))
    p = np.concatenate([wd, we])
    return p / p.mean()


def sim_task(args):
    scen, eta, rep, seed, n_other = args
    rng = np.random.default_rng(seed)
    s = np.arange(N_SLOTS)
    k = s % 24 + 24 * (((s // 24) % 7) >= 5)
    m = s // 720
    sd_shared, sd_own = dict(shared=(0.5, 0.0), private=(0.0, 0.5), both=(0.5, 0.3))[scen]
    shared = np.repeat(np.exp(rng.normal(0, 0.5, N_SLOTS // 24)), 24)          # the rest of Congress always has shocks

    def mu(n_target, e, use_shared, own_sd):
        raw = np.exp(rng.normal(0, 0.4, 12))[m] * _profile()[k]
        if use_shared:
            raw = raw * shared
        if own_sd > 0:
            raw = raw * np.repeat(np.exp(rng.normal(0, own_sd, N_SLOTS // 24)), 24)
        return raw * n_target * (1 - e) / (raw.sum() * 60.0)

    others = np.zeros(N_SLOTS)
    for j in range(n_other):
        t = hk.simulate_hawkes(mu(3000, 0.0, True, 0.0), 0.0, 1 / 15.0, seed * 100 + j + 1)
        others += np.bincount((t // 60).astype("int64"), minlength=N_SLOTS)[:N_SLOTS]
    t = hk.simulate_hawkes(mu(5000, eta, sd_shared > 0, sd_own), eta, 1 / 15.0, seed)
    tmpl = pm.Data(np.array([0.0, 1.0]), m, k, 12)
    d = pm.sim_to_data(t, tmpl, seed + 7)
    z = sm.standardize(np.log1p(others))
    r0, r1 = pm.fit_model("hawkes_exp", d), sm.fit_shared(d, z)
    return dict(scen=scen, eta_true=eta, rep=rep, n_other=n_other, n=d.n, eta_exp=r0["params"]["eta"], eta_shared=r1["params"]["eta"],
                gamma=r1["params"]["gamma"], p_exp=r0["p"], p_shared=r1["p"], D_exp=r0["D"], D_shared=r1["D"],
                dll=r1["ll"] - r0["ll"])


def stage_sim(a):
    tasks, cell = [], 0
    for scen in ("shared", "private", "both"):
        for e in (0.0, 0.2, 0.5):
            cell += 1
            for n_other in (30, 150):
                tasks += [(scen, e, r, 100000 * cell + r + 50000 * (n_other == 150), n_other) for r in range(a.reps)]
    sim_task(("shared", 0.2, 0, 1, 2))
    t0 = time.time()
    with Pool(a.workers) as p:
        res = p.map(sim_task, tasks, chunksize=2)
    df = pd.DataFrame(res)
    df.to_csv(OUT / "sim_shared.csv", index=False)
    g = df.assign(b_exp=df.eta_exp - df.eta_true, b_shared=df.eta_shared - df.eta_true).groupby(["scen", "n_other", "eta_true"])
    print(g[["b_exp", "b_shared", "gamma", "dll"]].mean().round(3).to_string())
    print("sim done", round(time.time() - t0), flush=True)


# ---------------------------------------------------------------- bootstrap for every member
def boot_task(args):
    bg, model, B = args
    try:
        d, z = member(bg)
        shared = model == "hawkes_shared"
        r0 = sm.fit_shared(d, z) if shared else pm.fit_model(model, d)
        Ds = []
        for b in range(B):
            t = sm.simulate_shared(d, r0, 104729 * (b + 1) + 7) if shared else pm.simulate(model, d, r0, 104729 * (b + 1) + 7)
            ds = pm.sim_to_data(t, d, 977 * (b + 1) + 3)
            if ds.n < 50:
                continue
            Ds.append((sm.fit_shared(ds, z) if shared else pm.fit_model(model, ds))["D"])
        Ds = np.array(Ds)
        return dict(bg=bg, model=model, n=d.n, D=r0["D"], p=r0["p"], B=len(Ds),
                    boot_p=float((1 + (Ds >= r0["D"]).sum()) / (1 + len(Ds))) if len(Ds) else np.nan,
                    boot_D_med=float(np.median(Ds)) if len(Ds) else np.nan,
                    boot_D_q95=float(np.quantile(Ds, 0.95)) if len(Ds) else np.nan)
    except Exception as e:  # noqa
        print("boot error", bg, model, str(e)[:200], flush=True)
        return dict(bg=bg, model=model, n=np.nan, D=np.nan, p=np.nan, B=0, boot_p=np.nan, boot_D_med=np.nan, boot_D_q95=np.nan)


def stage_boot(a):
    load()
    models = a.models.split(",")
    outf = OUT / "bootstrap_all.csv"
    done = set()
    if a.resume and outf.exists():
        x = pd.read_csv(outf)
        done = set(zip(x.bg, x.model))
    bgs = sorted(CACHE, key=lambda b: -CACHE[b]["n"])
    tasks = [(b, m, a.B) for b in bgs for m in models if (b, m) not in done]
    print("boot tasks", len(tasks), flush=True)
    first = not (a.resume and outf.exists())
    t0 = time.time()
    with Pool(a.workers, maxtasksperchild=8) as p:
        for i, r in enumerate(p.imap_unordered(boot_task, tasks, chunksize=1)):
            pd.DataFrame([r]).to_csv(outf, mode="w" if first else "a", header=first, index=False)
            first = False
            if i % 100 == 0:
                print(i, len(tasks), round(time.time() - t0), flush=True)
    print("boot done", round(time.time() - t0), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--B", type=int, default=30)
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--models", default="hawkes_exp,hawkes_shared,hawkes_soe,renc_he3,casc3")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dict(prep=stage_prep, fit=stage_fit, sim=stage_sim, boot=stage_boot)[a.stage](a)


if __name__ == "__main__":
    main()
