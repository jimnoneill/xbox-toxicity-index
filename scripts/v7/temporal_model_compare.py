"""Point-process model comparison for member posting times (stages: validate, fit, boot, analyze).

Usage: python temporal_model_compare.py STAGE [--workers 16] [--B 30] [--nboot 100] [--reps 20]
Outputs in output/v7/temporal/model_compare/. Existing outputs are never modified.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from xbox import pointprocess_models as pm  # noqa: E402

warnings.filterwarnings("ignore")
OUT = ROOT / "output/v7/temporal/model_compare"
ALL = list(pm.MODELS)
CACHE = {}


def load_cache():
    CACHE.update(pickle.load(open(OUT / "cache.pkl", "rb")))
    CACHE.update(pickle.load(open(OUT / "cache_lee.pkl", "rb")))


def row_of(bg, name, res, secs):
    return dict(bg=bg, model=name, n=CACHE[bg]["n"], ll=res["ll"], k=res["k"], aic=res["aic"], bic=res["bic"],
                D=res["D"], p=res["p"], mad=res["mad"], secs=round(secs, 2),
                params=json.dumps({a: float(b) for a, b in res["params"].items()}))


# ---------------------------------------------------------------- fit stage
def fit_task(bg):
    d = pm.calendar_from_cache(CACHE[bg])
    rows = []
    for m in ALL:
        t0 = time.time()
        try:
            r = pm.fit_model(m, d)
            rows.append(row_of(bg, m, r, time.time() - t0))
        except Exception as e:  # noqa
            rows.append(dict(bg=bg, model=m, n=CACHE[bg]["n"], err=str(e)[:200]))
    return rows


def stage_fit(a):
    load_cache()
    bgs = sorted(CACHE, key=lambda b: -CACHE[b]["n"])
    outf = OUT / "members_models.csv"
    done = set(pd.read_csv(outf).bg) if (outf.exists() and a.resume) else set()
    todo = [b for b in bgs if b not in done]
    print("fit", len(todo), "entities", flush=True)
    first = not outf.exists() or not a.resume
    t0 = time.time()
    with Pool(a.workers, maxtasksperchild=4) as p:
        for i, rows in enumerate(p.imap_unordered(fit_task, todo, chunksize=1)):
            pd.DataFrame(rows).to_csv(outf, mode="w" if first else "a", header=first, index=False)
            first = False
            if i % 20 == 0:
                print(i, round(time.time() - t0), "s", flush=True)
    print("fit done", round(time.time() - t0), flush=True)


# ---------------------------------------------------------------- validation by simulation
def val_task(args):
    bg, model, rep, nrep = args
    d = pm.calendar_from_cache(CACHE[bg])
    r0 = pm.fit_model(model, d)
    t = pm.simulate(model, d, r0, 7919 * (rep + 1) + 13)
    ds = pm.sim_to_data(t, d, 31 * rep + 5)
    out = dict(bg=bg, model=model, rep=rep, n_real=d.n, n_sim=ds.n)
    if ds.n < 100:
        return out
    r1 = pm.fit_model(model, ds)
    out.update(D=r1["D"], p=r1["p"], mad=r1["mad"], ll=r1["ll"], D_real=r0["D"])
    for key, v in r0["params"].items():
        if key in ("session_rate_per_day", "posts_per_session", "within_rate_per_min", "idle_gap_share", "eta", "inv_beta_min", "eta_total"):
            out[f"true_{key}"] = v
            out[f"est_{key}"] = r1["params"][key]
    return out


def stage_validate(a):
    load_cache()
    ids = [b for b in CACHE if b != "LEE"]
    ids = sorted(ids, key=lambda b: CACHE[b]["n"])
    nn = [CACHE[b]["n"] for b in ids]
    # templates: members with n between 1500 and 15000 spread over volume
    cand = [b for b in ids if 1500 <= CACHE[b]["n"] <= 15000]
    templ = [cand[i] for i in np.linspace(0, len(cand) - 1, a.ntempl).round().astype(int)]
    print("templates", [(b, CACHE[b]["n"]) for b in templ], flush=True)
    tasks = [(b, m, r, a.reps) for m in ALL for b in templ for r in range(a.reps)]
    rng = np.random.default_rng(0)
    rng.shuffle(tasks)
    t0 = time.time()
    res = []
    with Pool(a.workers, maxtasksperchild=8) as p:
        for i, r in enumerate(p.imap_unordered(val_safe, tasks, chunksize=1)):
            res.append(r)
            if i % 50 == 0:
                print(i, len(tasks), round(time.time() - t0), flush=True)
                pd.DataFrame(res).to_csv(OUT / "validation_sims.csv", index=False)
    pd.DataFrame(res).to_csv(OUT / "validation_sims.csv", index=False)
    print("validate done", round(time.time() - t0), flush=True)


def val_safe(x):
    try:
        return val_task(x)
    except Exception as e:  # noqa
        return dict(bg=x[0], model=x[1], rep=x[2], err=str(e)[:200])


def boot_safe(x):
    try:
        return boot_task(x)
    except Exception as e:  # noqa
        return dict(bg=x[0], model=x[1], err=str(e)[:200])


# ---------------------------------------------------------------- parametric bootstrap
def boot_task(args):
    bg, model, B = args
    d = pm.calendar_from_cache(CACHE[bg])
    r0 = pm.fit_model(model, d)
    Ds, Ms = [], []
    for b in range(B):
        t = pm.simulate(model, d, r0, 104729 * (b + 1) + 7)
        ds = pm.sim_to_data(t, d, 977 * (b + 1) + 3)
        if ds.n < 50:
            continue
        r1 = pm.fit_model(model, ds)
        Ds.append(r1["D"])
        Ms.append(r1["mad"])
    Ds, Ms = np.array(Ds), np.array(Ms)
    return dict(bg=bg, model=model, n=d.n, D=r0["D"], mad=r0["mad"], B=len(Ds),
                boot_p=float((1 + (Ds >= r0["D"]).sum()) / (1 + len(Ds))) if len(Ds) else np.nan,
                boot_mad_p=float((1 + (Ms >= r0["mad"]).sum()) / (1 + len(Ms))) if len(Ms) else np.nan,
                boot_D_med=float(np.median(Ds)) if len(Ds) else np.nan, boot_D_q95=float(np.quantile(Ds, 0.95)) if len(Ds) else np.nan)


def stage_boot(a):
    load_cache()
    ids = sorted([b for b in CACHE if b != "LEE"], key=lambda b: CACHE[b]["n"])
    pick = [ids[i] for i in np.linspace(0, len(ids) - 1, a.nboot).round().astype(int)]
    models = a.models.split(",") if a.models else ALL
    tasks = [(b, m, a.B) for b in sorted(pick, key=lambda b: -CACHE[b]["n"]) for m in models]
    print("boot tasks", len(tasks), flush=True)
    outf = OUT / "bootstrap.csv"
    first = True
    t0 = time.time()
    with Pool(a.workers, maxtasksperchild=8) as p:
        for i, r in enumerate(p.imap_unordered(boot_safe, tasks, chunksize=1)):
            pd.DataFrame([r]).to_csv(outf, mode="w" if first else "a", header=first, index=False)
            first = False
            if i % 50 == 0:
                print(i, len(tasks), round(time.time() - t0), flush=True)
    print("boot done", round(time.time() - t0), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--B", type=int, default=30)
    ap.add_argument("--nboot", type=int, default=100)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--ntempl", type=int, default=5)
    ap.add_argument("--models", default="")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"fit": stage_fit, "validate": stage_validate, "boot": stage_boot}[a.stage](a)


if __name__ == "__main__":
    main()
