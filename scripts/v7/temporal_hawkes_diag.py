"""Diagnostics for the Hawkes time-rescaling KS rejection (spec C5). Deterministic seeds.

Stages on a stratified sample of members (full window, non-retweet posts, exactly the pipeline design):
 1 reproduce pipeline KS, gap shares (ties), compensator check by numerical integration
 2 tie variants: collapse same-second posts, drop follow-ups within 60 s
 3 parametric bootstrap p value and effect size (mean absolute PIT deviation)
 4 two-exponential kernel, day-level baseline, 12-month window
Output: output/v7/temporal/hawkes_diag/{members.csv, null_calibration.csv, summary.txt}
Usage: python temporal_hawkes_diag.py [--workers 12] [--nmem 48] [--boot 20]
"""
import argparse
import sys
import time
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit
from scipy.stats import kstest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/v7"))
from xbox import hawkes as hk  # noqa: E402
import temporal_features as tf  # noqa: E402

OUT = ROOT / "output/v7/temporal/hawkes_diag"
G = tf.G


# ------------------------------------------------------------------ two-exponential EM
@njit(cache=True)
def _estep2(t, ev_m, ev_k, L, P, eta, beta, cnt):
    n = t.shape[0]
    cnt[:, :] = 0.0
    R = np.zeros(2)
    Rp = np.zeros(2)
    ll = 0.0
    sq = np.zeros(2)
    sl = np.zeros(2)
    prev = t[0]
    for i in range(n):
        if i > 0:
            d = t[i] - prev
            for j in range(2):
                e = np.exp(-beta[j] * d)
                Rp[j] = e * (Rp[j] + d * (1.0 + R[j]))
                R[j] = e * (1.0 + R[j])
        mu = max(L[ev_m[i]] * P[ev_k[i]], 1e-300)
        e0 = eta[0] * beta[0] * R[0]
        e1 = eta[1] * beta[1] * R[1]
        lam = mu + e0 + e1
        ll += np.log(lam)
        cnt[ev_m[i], ev_k[i]] += mu / lam
        sq[0] += e0 / lam
        sq[1] += e1 / lam
        sl[0] += eta[0] * beta[0] * Rp[0] / lam
        sl[1] += eta[1] * beta[1] * Rp[1] / lam
        prev = t[i]
    return ll, sq, sl


@njit(cache=True)
def _em2(t, ev_m, ev_k, occ, T, L, P, eta, beta, max_iter, tol, bmax):
    M, K = occ.shape
    n = t.shape[0]
    cnt = np.zeros((M, K))
    llp = -1e300
    ll = -1e300
    for it in range(max_iter):
        ll_ev, sq, sl = _estep2(t, ev_m, ev_k, L, P, eta, beta, cnt)
        comp = 0.0
        for m in range(M):
            for k in range(K):
                comp += L[m] * P[k] * occ[m, k]
        Gs = np.zeros(2)
        for j in range(2):
            for i in range(n):
                Gs[j] += 1.0 - np.exp(-beta[j] * (T - t[i]))
        ll = ll_ev - comp - eta[0] * Gs[0] - eta[1] * Gs[1]
        if abs(ll - llp) < tol * n:
            break
        llp = ll
        for j in range(2):
            if sq[j] > 0 and Gs[j] > 0:
                eta[j] = sq[j] / Gs[j]
            if sl[j] > 0 and sq[j] > 0:
                beta[j] = min(max(sq[j] / sl[j], 1.0 / (7 * 24 * 60.0)), bmax)
        s = eta[0] + eta[1]
        if s > 0.999:
            eta[0] *= 0.999 / s
            eta[1] *= 0.999 / s
        for _ in range(3):
            for m in range(M):
                num = 0.0
                den = 0.0
                for k in range(K):
                    num += cnt[m, k]
                    den += P[k] * occ[m, k]
                L[m] = num / den if den > 0 else 0.0
            for k in range(K):
                num = 0.0
                den = 0.0
                for m in range(M):
                    num += cnt[m, k]
                    den += L[m] * occ[m, k]
                P[k] = num / den if den > 0 else 0.0
        tot = 0.0
        ws = 0.0
        for m in range(M):
            for k in range(K):
                tot += P[k] * occ[m, k]
                ws += occ[m, k]
        sc = tot / ws if ws > 0 else 1.0
        if sc > 0:
            for k in range(K):
                P[k] /= sc
            for m in range(M):
                L[m] *= sc
    return ll


def trunc_ks(inc, delta):
    """KS of (inc - delta | inc > delta) against Exp(1): exact under the null by memorylessness."""
    x = inc[inc > delta] - delta
    if len(x) < 30:
        return np.nan, np.nan
    k = kstest(x, "expon")
    return float(k.statistic), float(k.pvalue)


def fit2(d, f1, bmax=1 / 0.05):
    """Two-exponential fit, initialised from the single-exponential fit (slow) plus a seconds-scale component."""
    L, P = f1["L"].copy(), f1["P"].copy()
    eta = np.array([max(f1["eta"] * 0.7, 0.01), 0.15])
    beta = np.array([min(f1["beta"], 1 / 5.0), 1 / 0.25])
    ll = _em2(d["t"], d["ev_m"], d["ev_k"], d["occ"], d["T"], L, P, eta, beta, 3000, 1e-9, bmax)
    o = np.argsort(beta)
    return dict(L=L, P=P, eta=eta[o], beta=beta[o], ll=float(ll))


@njit(cache=True)
def _incr2(t, mcum, eta, beta):
    n = t.shape[0]
    out = np.empty(n)
    a = np.zeros(2)
    prev_t = 0.0
    prev_m = 0.0
    for i in range(n):
        inc = mcum[i] - prev_m
        if i > 0:
            d = t[i] - prev_t
            for j in range(2):
                inc += eta[j] * a[j] * (1.0 - np.exp(-beta[j] * d))
                a[j] = a[j] * np.exp(-beta[j] * d) + 1.0
        else:
            a[0] = 1.0
            a[1] = 1.0
        out[i] = inc
        prev_t = t[i]
        prev_m = mcum[i]
    return out


def mcum_of(d, f):
    t = d["t"]
    rate = f["L"][d["slot_month"]] * f["P"][d["slot_bin"]]
    cum = np.concatenate([[0.0], np.cumsum(rate * hk.SLOT_MIN)])
    s = np.minimum((t // hk.SLOT_MIN).astype("int64"), len(rate) - 1)
    return cum[s] + rate[s] * (t - s * hk.SLOT_MIN)


def rescale2(d, f2):
    inc = _incr2(d["t"], mcum_of(d, f2), f2["eta"], f2["beta"])
    k = kstest(inc, "expon")
    return float(k.statistic), float(k.pvalue), inc


# ------------------------------------------------------------------ helpers
def pit_mad(inc):
    """Scale-free effect size: mean |F(inc_(i)) - (i-.5)/n| with F the Exp(1) cdf."""
    x = np.sort(inc)
    n = len(x)
    return float(np.mean(np.abs(1 - np.exp(-x) - (np.arange(1, n + 1) - 0.5) / n)))


def inc_profile(inc):
    return dict(f_lt001=float((inc < 0.01).mean()), f_lt01=float((inc < 0.1).mean()),
                f_gt3=float((inc > 3).mean()), f_gt5=float((inc > 5).mean()), inc_mean=float(inc.mean()))


def member_design(bg, window="full", kind="month", t_override=None):
    df = G["posts"][bg]
    lo, hi = tf.WINDOWS[window]
    t = df["t"].values
    sel = (t >= lo) & (t < hi) & df["nonrt"].values
    t = np.sort(t[sel]) if t_override is None else np.sort(t_override)
    ml = G["loc"][bg]
    loc_ok = tf.lc.model_usable(ml)
    ml_ = ml if loc_ok else None
    seed = zlib.crc32(f"{bg}|{window}|nonrt".encode()) & 0x7FFFFFFF
    slot0, m, k, nm = tf.slot_arrays(ml_, t[0], t[-1], tf.PRIMARY, day=(kind == "day"))
    return t, hk.make_design(t, m, k, slot0, nm, jitter_seed=seed), seed


def comp_check(d, f, n_check=150, seed=0):
    """Compare the rescaled increments with direct numerical integration of lambda(t)."""
    from scipy.integrate import quad
    t = d["t"]
    inc = hk._increments(t, mcum_of(d, f), 0.0, f["eta"], f["beta"])
    rate = f["L"][d["slot_month"]] * f["P"][d["slot_bin"]]
    ns = len(rate)
    rng = np.random.default_rng(seed)
    idx = rng.choice(np.arange(1, len(t)), size=min(n_check, len(t) - 1), replace=False)
    eta, beta = f["eta"], f["beta"]
    errs = []
    for i in idx:
        a, b = t[i - 1], t[i]
        past = t[:i]

        def lam(x):
            s = min(int(x // 60.0), ns - 1)
            return rate[s] + eta * beta * np.exp(-beta * (x - past)).sum()
        pts = [60.0 * s for s in range(int(a // 60.0) + 1, int(b // 60.0) + 1)]
        if len(pts) > 400:
            continue
        v = quad(lam, a, b, points=pts or None, limit=max(50, 2 * len(pts) + 10), epsabs=1e-10, epsrel=1e-8)[0]
        errs.append(abs(v - inc[i]) / max(inc[i], 1e-9))
    return dict(cc_n=len(errs), cc_max_rel=float(np.max(errs)) if errs else np.nan)


def boot_ks(d, f, B, seed0):
    """Parametric bootstrap of the KS statistic: simulate from the fitted model (second-rounded), refit, test."""
    mu = f["L"][d["slot_month"]] * f["P"][d["slot_bin"]]
    nm = d["occ"].shape[0]
    Ds, Ms, Ns = [], [], []
    for b in range(B):
        sd = seed0 + b
        t = hk.simulate_hawkes(mu, f["eta"], f["beta"], sd)
        if len(t) < 50:
            continue
        t_sec = np.sort(np.floor(t * 60).astype("int64"))
        ds = hk.make_design(t_sec, d["slot_month"], d["slot_bin"], 0, nm, jitter_seed=sd)
        fs = hk.fit(ds)
        r = hk.time_rescaling(ds, fs)
        Ds.append(r["ks_stat"])
        Ms.append(pit_mad(r["increments"]))
        Ns.append(len(t))
    return np.array(Ds), np.array(Ms), np.array(Ns)


# ------------------------------------------------------------------ per-member task
def task(args):
    bg, B = args
    out = dict(bg=bg)
    t, d, seed = member_design(bg)
    n = len(t)
    f = hk.fit(d)
    r = hk.time_rescaling(d, f)
    gaps = np.diff(t)
    out.update(n=n, eta=f["eta"], inv_beta=f["inv_beta_min"], D=r["ks_stat"], p=r["ks_p"], D_crit=1.36 / np.sqrt(n),
               g0=float((gaps == 0).mean()), g_lt5=float((gaps < 5).mean()), g_lt60=float((gaps < 60).mean()),
               mad=pit_mad(r["increments"]), **inc_profile(r["increments"]))
    out.update(comp_check(d, f, seed=seed))
    for dl in (0.01, 0.05, 0.1):
        kd, kp = trunc_ks(r["increments"], dl)
        out.update({f"tr{dl}_D": kd, f"tr{dl}_p": kp})
    # ties: collapse same-second, drop follow-ups within 60 s of the previous kept post
    tu = np.unique(t)
    keep = np.concatenate([[True], np.diff(tu) >= 60])
    for name, tt in (("collapse", tu), ("thin60", tu[keep])):
        if len(tt) >= 200:
            _, dd, _ = member_design(bg, t_override=tt)
            ff = hk.fit(dd)
            rr = hk.time_rescaling(dd, ff)
            out.update({f"{name}_n": len(tt), f"{name}_D": rr["ks_stat"], f"{name}_p": rr["ks_p"],
                        f"{name}_eta": ff["eta"], f"{name}_mad": pit_mad(rr["increments"])})
    # day-level baseline
    for nm_, kap in (("dayfix", 0.0), ("dayre", 2.0)):
        _, dd, _ = member_design(bg, kind="day")
        ff = hk.fit(dd, prior_kappa=kap)
        rr = hk.time_rescaling(dd, ff)
        out.update({f"{nm_}_D": rr["ks_stat"], f"{nm_}_p": rr["ks_p"], f"{nm_}_eta": ff["eta"], f"{nm_}_mad": pit_mad(rr["increments"])})
    # two exponentials (month baseline)
    f2 = fit2(d, f)
    D2, p2, inc2 = rescale2(d, f2)
    out.update(two_D=D2, two_p=p2, two_mad=pit_mad(inc2), two_eta_slow=f2["eta"][0], two_eta_fast=f2["eta"][1],
               two_inv_beta_slow=1 / f2["beta"][0], two_inv_beta_fast=1 / f2["beta"][1], two_ll=f2["ll"], one_ll=f["ll"])
    f2r = fit2(d, f, bmax=1 / 0.005)
    D2r, p2r, inc2r = rescale2(d, f2r)
    out.update(two_relax_D=D2r, two_relax_p=p2r, two_relax_inv_fast=1 / f2r["beta"][1], two_relax_eta_fast=f2r["eta"][1])
    for dl in (0.01, 0.05):
        kd, kp = trunc_ks(inc2r, dl)
        out.update({f"two_relax_tr{dl}_p": kp})
    # two exponentials on day-level baseline (random effect off)
    _, dd, _ = member_design(bg, kind="day")
    fd = hk.fit(dd)
    f2d = fit2(dd, fd)
    D2d, p2d, inc2d = rescale2(dd, f2d)
    out.update(two_day_D=D2d, two_day_p=p2d, two_day_mad=pit_mad(inc2d))
    # 12-month window
    try:
        tw, dw, _ = member_design(bg, window="w12")
        if len(tw) >= 200:
            fw = hk.fit(dw)
            rw = hk.time_rescaling(dw, fw)
            out.update(w12_n=len(tw), w12_D=rw["ks_stat"], w12_p=rw["ks_p"], w12_eta=fw["eta"])
    except Exception as e:  # noqa
        out["w12_err"] = str(e)[:60]
    # parametric bootstrap
    Ds, Ms, Ns = boot_ks(d, f, B, seed * 7 % 1000003)
    out.update(boot_n=len(Ds), boot_D_med=float(np.median(Ds)), boot_D_max=float(Ds.max()),
               boot_p=float((1 + (Ds >= r["ks_stat"]).sum()) / (1 + len(Ds))), boot_mad_med=float(np.median(Ms)),
               boot_p_pipeline_ks=float(np.mean(Ds > 1.36 / np.sqrt(np.median(Ns)))))
    return out


# ------------------------------------------------------------------ null calibration (in class, structured baseline)
def null_task(args):
    """Simulate from the model class with member-like structure, fit, run the exact KS code."""
    bg, seed, eta_over = args
    t, d, _ = member_design(bg)
    f = hk.fit(d)
    eta = f["eta"] if eta_over is None else eta_over
    mu = f["L"][d["slot_month"]] * f["P"][d["slot_bin"]]
    nm = d["occ"].shape[0]
    tt = hk.simulate_hawkes(mu, eta, f["beta"], seed)
    t_sec = np.sort(np.floor(tt * 60).astype("int64"))
    ds = hk.make_design(t_sec, d["slot_month"], d["slot_bin"], 0, nm, jitter_seed=seed)
    fs = hk.fit(ds)
    r = hk.time_rescaling(ds, fs)
    o = dict(bg=bg, seed=seed, n=len(tt), eta_true=eta, eta_hat=fs["eta"], D=r["ks_stat"], p=r["ks_p"], mad=pit_mad(r["increments"]))
    for dl in (0.01, 0.05, 0.1):
        o[f"tr{dl}_p"] = trunc_ks(r["increments"], dl)[1]
    return o


def load(nmem):
    return _load(nmem)


def _load(nmem):
    """Same loading as temporal_features.main, restricted to a stratified sample of members."""
    import xbox.location as lc
    diag = pd.read_csv(ROOT / "output/v7/temporal/hawkes_ci_diagnostics.csv")
    diag = diag[(diag.level == "member") & (diag.window == "full") & (diag.postset == "nonrt")].sort_values("n")
    pick = diag.iloc[np.linspace(0, len(diag) - 1, nmem).round().astype(int)]["id"].tolist() if nmem < len(diag) else diag["id"].tolist()
    amap = pd.read_csv(ROOT / "output/v7/ti/account_member_map.csv", dtype={"user_id": str})
    amap = amap[amap.entity_type == "member"].copy()
    tw = pd.read_parquet(ROOT / "data/congresstweets_2017_2023.parquet", columns=["id", "screen_name", "created_at", "text", "source"])
    tw = tw.drop_duplicates("id").merge(amap[["screen_name", "bioguide", "account_type"]], on="screen_name", how="inner")
    tw = tw[tw.bioguide.isin(pick)].copy()
    tw["t"] = tw["created_at"].dt.tz_convert("UTC").dt.tz_localize(None).values.astype("datetime64[s]").astype("int64")
    tw["nonrt"] = tf.tv.nonretweet_mask_congresstweets(tw["text"].values)
    tw = tw[["screen_name", "bioguide", "account_type", "t", "nonrt"]].sort_values("t")
    member_days, chamber_days, mchamb = lc.load_vote_days(ROOT / "data/voteview")
    mm = (amap.sort_values("n_tweets", ascending=False).groupby("bioguide")
          .agg(icpsr=("icpsr", "first"), party=("party", "first"), chamber=("chamber", "first"), state=("state", "first")))
    G["posts"] = {bg: g.reset_index(drop=True) for bg, g in tw.groupby("bioguide")}
    allm = pd.read_csv(ROOT / "data/voteview_all_members.csv")
    dtab = lc.load_district_table(ROOT / "data/district_zone/district_zone.csv")
    G["loc"] = {}
    for bg in pick:
        r = mm.loc[bg]
        icpsr = int(r["icpsr"])
        ch = sorted(mchamb.get(icpsr, {"H"}))
        cd = np.unique(np.concatenate([chamber_days[c] for c in ch if c in chamber_days]))
        vd = member_days.get(icpsr, np.array([], dtype="int64"))
        hz = None
        if r["chamber"] == "house":
            hz, _ = lc.district_zone_for_member(icpsr, r["state"], allm, dtab, G["posts"][bg]["t"].values)
        G["loc"][bg] = lc.MemberLocation(vd, cd, r["state"], name=bg, home_zone=hz)
    return pick


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--nmem", type=int, default=48)
    ap.add_argument("--boot", type=int, default=20)
    ap.add_argument("--null_reps", type=int, default=4)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    pick = load(a.nmem)
    print("loaded", len(pick), round(time.time() - t0), "s", flush=True)
    d0 = hk.make_design(np.arange(0, 20000, 37), np.zeros(10, int), np.zeros(10, int), 0, 1)
    f0 = hk.fit(d0)
    hk.time_rescaling(d0, f0)
    fit2(d0, f0)
    with Pool(a.workers) as p:
        nul = p.map(null_task, [(bg, 1000 + r, None) for bg in pick for r in range(a.null_reps)], chunksize=1)
        pd.DataFrame(nul).to_csv(OUT / "null_calibration.csv", index=False)
        print("null done", round(time.time() - t0), "s", flush=True)
        res = p.map(task, [(bg, a.boot) for bg in pick], chunksize=1)
    pd.DataFrame(res).to_csv(OUT / "members.csv", index=False)
    print("done", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
