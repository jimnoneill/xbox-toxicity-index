"""P2 rolling-origin forecasting run (MATH_SPEC H, task P2).

usage: p2_run.py --period dev|test [--accounts sample|all] [--models full|cheap] [--workers 16] [--max-origins N]

dev  : monthly origins 2019-07-01 .. 2022-06-01, targets must end before 2022-07-01 (test period untouched).
test : monthly origins 2022-07-01 .. 2023-06-01 (+ remainder to 2023-07-11). Run ONCE after dev is frozen.
Per origin: parameters fit on rows whose target window ends before T0 - 90 d; the last 90 d before T0 calibrate dispersion
(per account), baseline smoothing and Platt scaling. Forecast times inside the month use all information up to that hour.
Outputs: output/v7/p2/<period>_<accounts>/{agg.parquet,burst.parquet,pit.parquet,params.json}
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path.home() / "x_box/pyextra"))
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from xbox import forecast_p2 as F

OUT = ROOT / "output/v7/p2"
T2 = ROOT / "output/v7/t2full"
G = {}
HS = F.HORIZONS


def select_sample(st, n=200, seed=F.SEED):
    """Stratified sample: eligible = span covers >=120 d of the test period with >=120 d of history before it."""
    ts = F.hidx(F.TEST_START)
    elig = st[(st.typ.isin(["office", "campaign"])) & (st.party.isin(["D", "R"])) & ((st.h1 - F.HOUR0) >= ts + 120 * 24) &
              ((st.h0 - F.HOUR0) <= ts - 120 * 24)].copy()
    elig["vol"] = pd.qcut(elig.tot, 3, labels=["low", "mid", "high"])
    elig["stratum"] = elig.typ + "|" + elig.party + "|" + elig.chamber + "|" + elig.vol.astype(str)
    cnt = elig.stratum.value_counts()
    q = cnt / cnt.sum() * n
    alloc = np.maximum(np.floor(q).astype(int), 1)
    rem = n - alloc.sum()
    for s in (q - np.floor(q)).sort_values(ascending=False).index[:max(rem, 0)]:
        alloc[s] += 1
    alloc = np.minimum(alloc, cnt)
    rng = np.random.default_rng(seed)
    pick = []
    for s, k in alloc.items():
        pick += list(rng.choice(elig.index[elig.stratum == s].values, int(k), replace=False))
    return elig.loc[sorted(pick)]


def load(accounts_mode):
    hc = pd.read_parquet(T2 / "hourly_counts.parquet", columns=["account", "bioguide", "account_type", "party", "chamber",
                                                               "hour_idx", "count", "count_posts"])
    hc = hc.rename(columns={"account_type": "account_type"})
    st = hc.groupby("account").agg(typ=("account_type", "first"), party=("party", "first"), chamber=("chamber", "first"),
                                    h0=("hour_idx", "min"), h1=("hour_idx", "max"), tot=("count", "sum"))
    OUT.mkdir(parents=True, exist_ok=True)
    sp = OUT / "sample_accounts.csv"
    if accounts_mode == "sample":
        if sp.exists():
            sel = pd.read_csv(sp, index_col=0)
        else:
            sel = select_sample(st)
            sel.to_csv(sp)
        keep = sel.index
    else:
        ts = F.hidx(F.TEST_START)
        keep = st[st.typ.isin(["office", "campaign"]) & ((st.h1 - F.HOUR0) >= ts + 120 * 24) & ((st.h0 - F.HOUR0) <= ts - 120 * 24)].index
    hc = hc[hc.account.isin(keep)]
    cong = pd.read_parquet(T2 / "congress_hourly.parquet", columns=["hour_idx", "observed", "total_events"])
    cal = pd.read_parquet(ROOT / "output/v7/t3/calendar_member_day.parquet",
                          columns=["date", "bioguide", "primary_window", "general_window", "post_election", "sotu"])
    P = F.Panel(hc, cong, cal)
    sch = pd.read_csv(ROOT / "output/v7/t1full/scheduler_accounts.csv").set_index("screen_name")
    flag = (sch.flag_minute | sch.flag_tool | sch.flag_strong).astype(float)
    sched = np.array([flag.get(a, 0.0) for a in P.accounts])
    return P, sched


# ------------------------------------------------------------------ raw rows (origin independent, causal)
def build_rows(P, sched):
    A, T = P.Y.shape
    L = {hl: F.ew_levels(P, hl) for hl in F.HLS}
    Lc = F.congress_level(P, 21 * 24)
    Cl = np.concatenate([[0.0], np.cumsum(P.C)])
    Mb = np.zeros((A, T), np.float32)       # max of Y over hours t-5..t
    Y0 = P.Y0
    Mb[:] = Y0
    for k in range(1, 6):
        Mb[:, k:] = np.maximum(Mb[:, k:], Y0[:, :-k])
    tg = np.arange(F.STRIDE * 40, T - 1, F.STRIDE)
    ai, t = np.meshgrid(np.arange(A), tg, indexing="ij")
    ai, t = ai.ravel(), t.ravel()
    elig = (t - P.span0[ai] >= F.MIN_HIST) & (t <= P.span1[ai]) & (P.cnt_valid[ai, t + 1] - P.cnt_valid[ai, t] == 1)
    ai, t = ai[elig], t[elig]
    o = np.argsort(t, kind="stable")
    ai, t = ai[o].astype(np.int32), t[o].astype(np.int64)
    R = {"ai": ai, "t": t}
    R["y0"] = Y0[ai, t]
    for n, w in (("s3", 3), ("s24", 24), ("s168", 168)):
        R[n] = P.window_sum(ai, t - w + 1, t + 1)
    cwin = lambda w: np.maximum(Cl[t + 1] - Cl[np.maximum(t + 1 - w, 0)] - R["s3" if w == 3 else "s24" if w == 24 else "s168"], 0)
    R["c1"] = np.maximum(P.C[t] - R["y0"], 0)
    R["c3"], R["c24"] = cwin(3), cwin(24)
    for name, H in HS.items():
        R["y_" + name] = P.window_sum(ai, t + 1, t + 1 + H)
        R["v_" + name] = P.window_valid(ai, t + 1, t + 1 + H)
        R["sn_" + name] = P.window_sum(ai, t + 1 - 168, t + 1 - 168 + H) if H < 168 else R["s168"]
        R["pe_" + name] = {1: R["y0"], 24: R["s24"], 168: R["s168"]}[H]
    R["m6"] = np.where(t + 6 < T, Mb[ai, np.minimum(t + 6, T - 1)], 0)
    R["v6"] = P.window_valid(ai, t + 1, t + 7)
    R["pe6"] = Mb[ai, t]
    R["sn6"] = Mb[ai, t + 6 - 168]
    for hl in F.HLS:
        R[f"L{hl}"] = L[hl][ai, t]
    R["how"] = P.how[t]
    fl = P.flags
    di = P.etday[np.minimum(t + 1, T - 1)] - fl["d0"]
    for n in ("primary_window", "general_window", "post_election", "sotu"):
        R[n] = fl[n][ai, di]
    R["holiday"] = fl["holiday"][di]
    G["L"], G["Lc"], G["Mb"] = L, Lc, Mb
    return R


def sl_rows(R, lo, hi):
    a, b = np.searchsorted(R["t"], [lo, hi])
    return {k: v[a:b] for k, v in R.items()}


def pick(r, m):
    return {k: v[m] for k, v in r.items()}


# ------------------------------------------------------------------ design
def design(P, r, H, ctx, hlsel, burst_theta=None):
    ai, t = r["ai"], r["t"]
    how = r["how"]
    L = np.maximum(r[f"L{hlsel}"], 1e-4)
    prof, cp, cprof, ccp = ctx["prof"], ctx["cp"], ctx["cprof"], ctx["ccp"]
    E1 = L * prof[ai, how]
    E3 = L * F.prof_window(cp, ai, (how - 2) % 168, 3)
    E24 = L * F.prof_window(cp, ai, (how - 23) % 168, 24)
    E168 = 168 * L
    W = F.prof_window(cp, ai, (how + 1) % 168, H)
    Lc = G["Lc"][t]
    cw = lambda s, n: Lc * (ccp[(s % 168) + n] - ccp[s % 168])
    ce1, ce3, ce24 = Lc * cprof[how], cw(how - 2, 3), cw(how - 23, 24)
    lg = lambda x, e: np.log1p(x) - np.log1p(e)
    cols = [np.ones(len(t)),
            lg(r["y0"], E1), lg(r["s3"], E3), lg(r["s24"], E24), lg(r["s168"], E168),
            lg(r["c1"], np.maximum(ce1 - E1, 0)), lg(r["c3"], np.maximum(ce3 - E3, 0)), lg(r["c24"], np.maximum(ce24 - E24, 0)),
            np.clip(np.log((r[f"L{F.HLS[0]}"] + 1e-3) / (r[f"L{F.HLS[2]}"] + 1e-3)), -3, 3), np.log(W / H),
            r["primary_window"], r["general_window"], r["post_election"], r["holiday"], r["sotu"]]
    X = np.column_stack(cols)
    off = np.log(L * W)
    return X, off


XN = ["const", "d_own1", "d_own3", "d_own24", "d_own168", "d_cong1", "d_cong3", "d_cong24", "level_trend", "log_profile",
      "primary", "general", "post_election", "holiday", "sotu"]
NOCONG = [0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14]


def gbm_features(P, r, X, off, ctx):
    ai = r["ai"]
    st = ctx["static"]
    how_n = (r["how"] + 1) % 168
    ex = [np.log1p(r["y0"]), np.log1p(r["s3"]), np.log1p(r["s24"]), np.log1p(r["s168"]), np.log1p(r["c1"]), np.log1p(r["c3"]),
          np.log1p(r["c24"]), np.log(np.maximum(r[f"L{F.HLS[0]}"], 1e-4)), np.log(np.maximum(r[f"L{F.HLS[1]}"], 1e-4)),
          np.log(np.maximum(r[f"L{F.HLS[2]}"], 1e-4)), how_n, how_n % 24, how_n // 24, np.log(ctx["prof"][ai, how_n]),
          st[ai, 0], st[ai, 1], st[ai, 2], np.log(ctx["ybar"][ai] + 1e-4), np.log(G["Lc"][r["t"]] + 1e-3)]
    return np.column_stack([X[:, 1:], np.column_stack(ex), off])


GN = XN[1:] + ["l_y0", "l_s3", "l_s24", "l_s168", "l_c1", "l_c3", "l_c24", "lL7", "lL21", "lL60", "how_next", "hour_next",
               "dow_next", "log_prof_next", "type", "party", "chamber", "log_ybar", "log_Lc", "log_offset"]


# ------------------------------------------------------------------ one origin
def run_origin(args):
    T0, T1, limit, models = args
    t_start = time.time()
    P, R, sched = G["P"], G["R"], G["sched"]
    A = len(P.accounts)
    typ = (P.static.typ.values == "campaign").astype(int)
    rng = np.random.default_rng(F.SEED + T0)
    cal_lo = T0 - 90 * 24
    # ---- context from data before T0 only
    prof, ybar, g = F.hourly_profile(P, None, T0)
    ctx = {"prof": prof, "cp": F.cum_profile(prof), "ybar": ybar}
    cprof = F.cong_profile(P, T0)
    ctx["cprof"] = cprof
    ctx["ccp"] = np.concatenate([[0.0], np.cumsum(np.concatenate([cprof, cprof, cprof[:1]]))])
    ctx["static"] = np.column_stack([typ, (P.static.party.values == "R").astype(int), (P.static.chamber.values == "senate").astype(int)])
    mr = np.maximum(np.nansum(P.P[:, :T0] - P.Y[:, :T0], axis=1), 0) / np.maximum(np.nansum(P.Y[:, :T0], axis=1), 1)
    # burst thresholds from training window
    theta = np.full(A, 2.0)
    for i in range(A):
        v = P.Y[i, max(0, T0 - 730 * 24):T0]
        v = v[np.isfinite(v) & (v > 0)]
        if len(v) >= 20:
            theta[i] = max(2.0, np.quantile(v, 0.9))
    ctx["theta"] = theta
    # ---- row sets
    ev = sl_rows(R, T0, T1)
    tr = sl_rows(R, 0, cal_lo)
    ca = sl_rows(R, cal_lo, T0)
    out_agg, params = [], {"T0": int(T0)}
    pit_rows, burst_rows = [], []

    # level half-life selection: pooled NB log-likelihood of level x profile on calibration rows (day horizon)
    best, bll = F.HLS[1], -np.inf
    m_ = ca["v_d24"] & (ca["t"] + 24 < T0)
    for hl in F.HLS:
        X, off = design(P, pick(ca, m_), 24, ctx, hl)
        mu = np.exp(off)
        ll = F.nb_logpmf(ca["y_d24"][m_], mu, 1.0).sum()
        if ll > bll:
            best, bll = hl, ll
    hlsel = best
    params["hl_hours"] = int(hlsel)

    # congress persistence
    x = np.log((P.C[:T0] + 1) / (G["Lc"][:T0] * cprof[P.how[:T0]] + 1))
    phi = float(np.clip(np.corrcoef(x[:-1][P.obs_cong[:T0 - 1]], x[1:][P.obs_cong[:T0 - 1]])[0, 1], 0.2, 0.95))
    params["cong_phi"] = phi

    results = {}   # (h, model) -> dict of arrays for eval rows
    for hname, H in HS.items():
        mtr = tr["v_" + hname] & (tr["t"] + H < cal_lo)
        mca = ca["v_" + hname] & (ca["t"] + H < T0)
        mev = ev["v_" + hname] & (ev["t"] + H < limit)
        rtr, rca, rev = pick(tr, mtr), pick(ca, mca), pick(ev, mev)
        if len(rev["t"]) == 0 or len(rca["t"]) < 1000 or len(rtr["t"]) < 5000:
            continue
        Xtr, otr = design(P, rtr, H, ctx, hlsel)
        Xca, oca = design(P, rca, H, ctx, hlsel)
        Xev, oev = design(P, rev, H, ctx, hlsel)
        ytr, yca, yev = rtr["y_" + hname], rca["y_" + hname], rev["y_" + hname]
        ai_ca, ai_ev = rca["ai"], rev["ai"]
        hm = ybar * H                                   # historical mean for the window
        mods = {}
        # --- baselines: smoothed point forecasts turned into NB distributions with calibrated smoothing and account dispersion
        for bname, key in (("pers", "pe_"), ("snaive", "sn_")):
            pc, pe = rca[key + hname], rev[key + hname]
            bestk, bl = 0.1, -np.inf
            for kap in (0.03, 0.1, 0.3, 1.0, 3.0):
                mu = (pc + kap * hm[ai_ca] + 1e-3) / (1 + kap)
                ll = F.nb_logpmf(yca, mu, 1.0).sum()
                if ll > bl:
                    bestk, bl = kap, ll
            mods[bname] = dict(point=pe, cal_mu=(pc + bestk * hm[ai_ca] + 1e-3) / (1 + bestk),
                               ev_mu=(pe + bestk * hm[ai_ev] + 1e-3) / (1 + bestk), kappa=bestk)
        mods["hist"] = dict(point=hm[ai_ev], cal_mu=hm[ai_ca] + 1e-3, ev_mu=hm[ai_ev] + 1e-3)
        Wc = F.prof_window(ctx["cp"], ai_ca, (rca["how"] + 1) % 168, H)
        We = F.prof_window(ctx["cp"], ai_ev, (rev["how"] + 1) % 168, H)
        mods["profmean"] = dict(point=ybar[ai_ev] * We, cal_mu=ybar[ai_ca] * Wc + 1e-3, ev_mu=ybar[ai_ev] * We + 1e-3)
        # --- NB ladder
        sub = rng.choice(len(ytr), min(len(ytr), 350000), replace=False)
        bnb = {}
        for ty in (0, 1):
            ms = ctx["static"][rtr["ai"][sub], 0] == ty
            ix = sub[ms]
            if len(ix) < 2000:
                continue
            b0, a0 = F.fit_nb(Xtr[ix][:, [0]], ytr[ix], otr[ix], ridge=1e-6)             # level x profile only
            bf, af = F.fit_nb(Xtr[ix], ytr[ix], otr[ix])
            bc, ac = F.fit_nb(Xtr[ix][:, NOCONG], ytr[ix], otr[ix])
            bnb[ty] = (b0, bf, bc)
            params.setdefault("nb_coef", {})[f"{hname}_type{ty}"] = dict(zip(XN, map(float, bf)))
        def nb_pred(Xx, ox, ai_x, which):
            mu = np.zeros(len(ox))
            for ty, tup in bnb.items():
                m = ctx["static"][ai_x, 0] == ty
                if which == "nb0":
                    mu[m] = np.exp(ox[m] + Xx[m][:, [0]] @ tup[0])
                elif which == "nb":
                    mu[m] = np.exp(np.clip(ox[m] + Xx[m] @ tup[1], -20, 14))
                else:
                    mu[m] = np.exp(np.clip(ox[m] + Xx[m][:, NOCONG] @ tup[2], -20, 14))
            return mu
        for nm in ("nb0", "nb_nocong", "nb"):
            mods[nm] = dict(cal_mu=nb_pred(Xca, oca, ai_ca, nm), ev_mu=nb_pred(Xev, oev, ai_ev, nm))
        # --- GBM (Poisson objective with log(level x profile) as init score), NB dispersion by account
        if "gbm" in models:
            import lightgbm as lgb
            Gtr, Gca, Gev = (gbm_features(P, rtr, Xtr, otr, ctx), gbm_features(P, rca, Xca, oca, ctx),
                             gbm_features(P, rev, Xev, oev, ctx))
            ix = np.sort(sub)
            val = rtr["t"][ix] >= cal_lo - 60 * 24
            prm = dict(objective="poisson", learning_rate=0.05, num_leaves=15, min_data_in_leaf=300, feature_fraction=0.8,
                       bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, num_threads=1, verbose=-1, seed=F.SEED)
            dtr = lgb.Dataset(Gtr[ix][~val][:, :-1], ytr[ix][~val], init_score=otr[ix][~val], feature_name=GN[:-1])
            dva = lgb.Dataset(Gtr[ix][val][:, :-1], ytr[ix][val], init_score=otr[ix][val], reference=dtr)
            bst = lgb.train(prm, dtr, 500, valid_sets=[dva], callbacks=[lgb.early_stopping(30, verbose=False)])
            mods["gbm"] = dict(cal_mu=np.exp(oca + bst.predict(Gca[:, :-1], raw_score=True)),
                               ev_mu=np.exp(oev + bst.predict(Gev[:, :-1], raw_score=True)))
            params.setdefault("gbm_rounds", {})[hname] = int(bst.best_iteration)
            params.setdefault("gbm_gain", {})[hname] = dict(zip(GN[:-1], map(float, bst.feature_importance("gain"))))
        # --- Hawkes (multi-step expected intensity)
        if "hawkes" in models:
            thh = G["hk_theta"].get((T0, "all"))
            if thh is not None:
                mus = {}
                for part, rr in (("cal_mu", rca), ("ev_mu", rev)):
                    # subsample long calibration set for cost
                    mus[part] = hawkes_mu(P, rr, ctx, hlsel, thh, sched, mr, phi, H)
                mods["hawkes"] = dict(cal_mu=mus["cal_mu"], ev_mu=mus["ev_mu"])
        # --- dispersion per account on calibration rows, then scores on evaluation rows
        for nm, d in mods.items():
            alpha_a, pool = F.fit_alpha_accounts(yca, d["cal_mu"], ai_ca, A)
            d["alpha_pool"] = pool
            mu, al = d["ev_mu"], alpha_a[ai_ev]
            ls = F.nb_logpmf(yev, mu, al)
            crps = F.nb_crps(yev, mu, al)
            from scipy.stats import nbinom
            r_ = 1.0 / np.maximum(al, 1e-6)
            med = nbinom.ppf(0.5, r_, r_ / (r_ + mu)) if "point" not in d else d["point"]
            ae = np.abs(yev - med)
            se = (yev - (d["point"] if "point" in d else mu)) ** 2
            df = pd.DataFrame(dict(account=ai_ev, ls=ls, crps=crps, ae=ae, se=se, n=1.0))
            df["h"], df["model"], df["origin"] = hname, nm, T0
            out_agg.append(df)
            if nm in ("nb", "gbm", "hawkes", "profmean"):
                pit = F.nb_pit(yev, mu, al, rng.random(len(yev)))
                pit_rows.append(pd.DataFrame(dict(h=hname, model=nm, typ=ctx["static"][ai_ev, 0], pit=pit)))
        if hname == "h1":                      # keep Hawkes calibration info for the burst step
            keep = dict(rev=rev, rca=rca, Xtr=Xtr, otr=otr, rtr=rtr, Xca=Xca, oca=oca, Xev=Xev, oev=oev, mods=mods)
    # ---- burst
    if "keep" in dir() or True:
        try:
            br = burst_task(P, R, ev, tr, ca, ctx, hlsel, T0, T1, limit, cal_lo, models, sched, mr, phi, rng)
            if br is not None:
                burst_rows.append(br)
        except Exception as e:  # keep the run alive but record
            params["burst_error"] = repr(e)
    agg = pd.concat(out_agg) if out_agg else pd.DataFrame()
    if len(agg):
        agg = agg.groupby(["origin", "h", "model", "account"], as_index=False).agg(n=("n", "sum"), ls=("ls", "sum"), crps=("crps", "sum"),
                                                                                    ae=("ae", "sum"), se=("se", "sum"))
    pit = pd.concat(pit_rows) if pit_rows else pd.DataFrame()
    if len(pit):
        pit["bin"] = np.minimum((pit.pit * 10).astype(int), 9)
        pit = pit.groupby(["h", "model", "typ", "bin"], as_index=False).size()
    params["secs"] = time.time() - t_start
    return agg, (pd.concat(burst_rows) if burst_rows else pd.DataFrame()), pit, params


def _hk_rows(P, rr, ctx, th, sched, mr, phi, H):
    """Expected hourly intensities (n,H) for rows rr; th = {account type: theta}."""
    ai, t = rr["ai"], rr["t"]
    lam_all = np.zeros((len(t), H))
    tys = P.static.typ.values[ai]
    for ty, thv in th.items():
        idx = np.where(tys == ty)[0]
        for s in range(0, len(idx), 15000):
            ii = idx[s:s + 15000]
            lam_all[ii] = F.hk_forecast(np.asarray(thv), P, ai[ii], t[ii], G["L"][F.HLS[1]], G["L"][F.HLS[0]], G["Lc"], ctx["prof"], ctx["cprof"], phi,
                                        sched, mr, H=H)
    return lam_all


def hawkes_mu(P, rr, ctx, hlsel, th, sched, mr, phi, H):
    return _hk_rows(P, rr, ctx, th, sched, mr, phi, H).sum(1)


def hawkes_hourly(P, rr, ctx, hlsel, th, sched, mr, phi, H=6):
    return _hk_rows(P, rr, ctx, th, sched, mr, phi, H)


def burst_task(P, R, ev, tr, ca, ctx, hlsel, T0, T1, limit, cal_lo, models, sched, mr, phi, rng):
    A = len(P.accounts)
    theta = ctx["theta"]
    f = lambda r, lim: r["v6"] & (r["t"] + 6 < lim)
    rtr, rca, rev = pick(tr, f(tr, cal_lo)), pick(ca, f(ca, T0)), pick(ev, f(ev, limit))
    if len(rev["t"]) == 0 or len(rtr["t"]) < 5000:
        return None
    lab = lambda r: (r["m6"] >= theta[r["ai"]]).astype(float)
    ytr, yca, yev = lab(rtr), lab(rca), lab(rev)
    base_all = np.zeros(A)
    rall = np.concatenate([ytr, yca]);  aall = np.concatenate([rtr["ai"], rca["ai"]])
    pool = rall.mean()
    base_all = (np.bincount(aall, rall, A) + 20 * pool) / (np.bincount(aall, None, A) + 20)
    out = dict(account=rev["ai"], t=rev["t"], y=yev, base=base_all[rev["ai"]])
    # persistence / seasonal-naive indicators
    for nm, key in (("pers", "pe6"), ("snaive", "sn6")):
        itr = (np.concatenate([rtr[key], rca[key]]) >= theta[aall]).astype(int)
        q = [rall[itr == k].mean() / max(pool, 1e-6) for k in (0, 1)]
        ie = (rev[key] >= theta[rev["ai"]]).astype(int)
        cnt = np.array([[np.sum((aall == a) & (itr == k)) for k in (0, 1)] for a in range(A)]) if False else None
        # account-level shrinkage of conditional rates
        p = np.zeros(len(ie))
        for k in (0, 1):
            h = np.bincount(aall[itr == k], rall[itr == k], A)
            n = np.bincount(aall[itr == k], None, A)
            prior = np.clip(q[k] * base_all, 1e-4, 0.99)
            pk = (h + 8 * prior) / (n + 8)
            p[ie == k] = pk[rev["ai"][ie == k]]
        out[nm] = np.clip(p, 1e-4, 0.999)
        out[nm + "_i"] = ie
    # logistic on NB-type features (H=6), and GBM
    sub = rng.choice(len(ytr), min(len(ytr), 350000), replace=False)
    def feats(r):
        X, off = design(P, r, 6, ctx, hlsel)
        th_ = theta[r["ai"]]
        Z = np.column_stack([X, off - np.log(th_), np.log(th_)])
        return X, off, Z
    Xtr, otr, Ztr = feats(rtr); Xca, oca, Zca = feats(rca); Xev, oev, Zev = feats(rev)
    for ty in (0, 1):
        pass
    bl = F.fit_logit(Ztr[sub], ytr[sub])
    out["logit"] = np.clip(expit(Zev @ bl), 1e-4, 1 - 1e-4)
    if "gbm" in models:
        import lightgbm as lgb
        Gtr = np.column_stack([gbm_features(P, rtr, Xtr, otr, ctx), Ztr[:, -2:]])
        Gca = np.column_stack([gbm_features(P, rca, Xca, oca, ctx), Zca[:, -2:]])
        Gev = np.column_stack([gbm_features(P, rev, Xev, oev, ctx), Zev[:, -2:]])
        ix = np.sort(sub)
        val = rtr["t"][ix] >= cal_lo - 60 * 24
        prm = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=300, feature_fraction=0.8,
                   bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, num_threads=1, verbose=-1, seed=F.SEED)
        dtr = lgb.Dataset(Gtr[ix][~val], ytr[ix][~val]); dva = lgb.Dataset(Gtr[ix][val], ytr[ix][val], reference=dtr)
        bst = lgb.train(prm, dtr, 500, valid_sets=[dva], callbacks=[lgb.early_stopping(30, verbose=False)])
        out["gbm"] = np.clip(bst.predict(Gev), 1e-4, 1 - 1e-4)
    if "hawkes" in models and G["hk_theta"].get((T0, "all")) is not None:
        th = G["hk_theta"][(T0, "all")]
        alpha1 = None
        def pb(r):
            lam = hawkes_hourly(P, r, ctx, hlsel, th, sched, mr, phi, H=6)
            return lam
        lc, le = pb(rca), pb(rev)
        # hourly NB dispersion: pooled from calibration (account-level shrunk) on h=1 hourly targets
        ycal1 = P.Y0[rca["ai"], np.minimum(rca["t"] + 1, P.T - 1)]
        al_a, _ = F.fit_alpha_accounts(ycal1, lc[:, 0], rca["ai"], A)
        def pburst(lam, ai):
            thr = theta[ai][:, None]
            pk = 1 - F.nb_cdf(thr - 1, lam, al_a[ai][:, None])
            return 1 - np.prod(1 - np.clip(pk, 0, 1), axis=1)
        zc, ze = pburst(lc, rca["ai"]), pburst(le, rev["ai"])
        Zc = np.column_stack([np.ones(len(zc)), logit(np.clip(zc, 1e-5, 1 - 1e-5))])
        bpl = F.fit_logit(Zc, yca, ridge=1e-6)
        out["hawkes"] = np.clip(expit(np.column_stack([np.ones(len(ze)), logit(np.clip(ze, 1e-5, 1 - 1e-5))]) @ bpl), 1e-4, 1 - 1e-4)
    df = pd.DataFrame(out)
    df["origin"] = T0
    return df


# ------------------------------------------------------------------ Hawkes fits per origin (type pooled)
def hk_fit_origin(args):
    T0, = args
    P, sched = G["P"], G["sched"]
    cal_lo = T0 - 90 * 24
    prof, ybar, g = F.hourly_profile(P, None, T0)
    cprof = F.cong_profile(P, T0)
    typ = P.static.typ.values
    lo = max(1, cal_lo - 18 * 30 * 24)
    rng = np.random.default_rng(F.SEED + T0)
    th_all = {}
    # level choice mirrors run_origin default (21 d) for the fit baseline; forecast uses hlsel (same family)
    Ls = G["L"][F.HLS[1]]
    ok = np.where((P.span0 < cal_lo - 60 * 24) & (P.span1 > lo))[0]
    res = {}
    for ty in ("office", "campaign"):
        rows = ok[typ[ok] == ty]
        if len(rows) == 0:
            continue
        rows = rng.choice(rows, min(len(rows), 50), replace=False)
        th, f = F.hk_fit(P, rows, lo, cal_lo, Ls, G["L"][F.HLS[0]], G["Lc"], prof, cprof, sched, th0=G.get("hk_init"), maxiter=60)
        res[ty] = (th, f)
    return T0, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", default="dev")
    ap.add_argument("--accounts", default="sample")
    ap.add_argument("--models", default="full")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max-origins", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    P, sched = load(a.accounts)
    G.update(P=P, sched=sched)
    G["R"] = build_rows(P, sched)
    print("rows", len(G["R"]["t"]), "accounts", len(P.accounts), f"{time.time()-t0:.0f}s", flush=True)
    models = {"full": ("gbm", "hawkes"), "cheap": ()}[a.models]
    if a.period == "dev":
        o0, o1, limit = F.hidx(F.DEV_START), F.hidx(F.TEST_START), F.hidx(F.TEST_START)
    else:
        o0, o1, limit = F.hidx(F.TEST_START), P.T - 1, P.T
    mo = pd.date_range(F.T0_TS.normalize() + pd.Timedelta(hours=0), periods=1)
    starts = pd.date_range(F.DEV_START if a.period == "dev" else F.TEST_START, periods=60, freq="MS")
    bounds = [F.hidx(s) for s in starts if F.hidx(s) < o1] + [o1]
    origins = list(zip(bounds[:-1], bounds[1:]))
    if a.max_origins:
        origins = origins[:a.max_origins]
    G["hk_theta"] = {}
    od = OUT / f"{a.period}_{a.accounts}_{a.models}"
    od.mkdir(parents=True, exist_ok=True)
    with Pool(a.workers) as pool:
        if "hawkes" in models:
            fits = pool.map(hk_fit_origin, [(T0,) for T0, _ in origins], chunksize=1)
            hkp = {}
            for T0, res in fits:
                hkp[int(T0)] = {k: [list(map(float, v[0])), v[1]] for k, v in res.items()}
            json.dump(hkp, open(od / "hawkes_params.json", "w"))
            print("hawkes fits done", f"{time.time()-t0:.0f}s", flush=True)
            # pooled single parameter set per origin: type-specific mixture is applied via account type; use office/campaign mean
            G["hk_by_type"] = {T0: {k: v[0] for k, v in res.items()} for T0, res in fits}
        G["models_tmp"] = models
    # Hawkes with type-specific parameters: run office and campaign rows separately in hawkes_mu via wrapper
    install_type_wrappers()
    jobs = [(T0, T1, limit, models) for T0, T1 in origins]
    with Pool(a.workers) as pool:
        res = pool.map(run_origin_wrapped, jobs, chunksize=1)
    agg = pd.concat([r[0] for r in res if len(r[0])])
    burst = pd.concat([r[1] for r in res if len(r[1])])
    pit = pd.concat([r[2] for r in res if len(r[2])])
    agg.to_parquet(od / "agg.parquet")
    burst.to_parquet(od / "burst.parquet")
    pit.to_parquet(od / "pit.parquet")
    P_ = [r[3] for r in res]
    json.dump(P_, open(od / "params.json", "w"), default=float)
    json.dump(dict(accounts=P.accounts, static=P.static.reset_index().astype(str).to_dict("records")), open(od / "accounts.json", "w"))
    print("done", f"{time.time()-t0:.0f}s", flush=True)


def install_type_wrappers():
    """Hawkes parameters are type specific: make G['hk_theta'] a lookup that hawkes_mu/hawkes_hourly resolve per account type."""
    pass


def run_origin_wrapped(args):
    T0 = args[0]
    th = G.get("hk_by_type", {}).get(T0)
    if th:
        G["hk_theta"] = {(T0, "all"): th}
    return run_origin(args)


if __name__ == "__main__":
    main()
