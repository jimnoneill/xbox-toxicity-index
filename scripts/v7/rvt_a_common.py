"""RVT-A shared helpers: panel, Voteview service history, power tables, OLS with cluster SE, wild cluster bootstrap."""
import os, sys, json
import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.expanduser("~/x_box/pipeline")
sys.path.insert(0, ROOT)
from xbox import kistner_ext as K  # noqa

OUT = f"{ROOT}/output/v7/rvt_a"
SEED = 20261013
B_BOOT = 1999


def _tab(spans):
    d = {}
    for a, b, v in spans:
        for y in range(a, b + 1):
            d[y] = v
    return d

PRES = _tab([(1989, 1992, "R"), (1993, 2000, "D"), (2001, 2008, "R"), (2009, 2016, "D"), (2017, 2020, "R"), (2021, 2022, "D")])
HOUSE = _tab([(1989, 1994, "D"), (1995, 2006, "R"), (2007, 2010, "D"), (2011, 2018, "R"), (2019, 2022, "D")])
SENATE = _tab([(1989, 1994, "D"), (1995, 2000, "R"), (2001, 2002, "D"), (2003, 2006, "R"), (2007, 2014, "D"), (2015, 2020, "R"), (2021, 2022, "D")])
MAJ = {"House": HOUSE, "Senate": SENATE}


def start_year(c):
    return 1789 + 2 * (c - 1)


def load_panel():
    p = K.build_panel(("Twitter",))
    p = p[(p.n >= 50) & p.Year.between(2010, 2022)].copy()
    return p.reset_index(drop=True)


def load_service(panel):
    """Voteview service: one row per (icpsr, congress, chamber) 1989+ plus first congress ever."""
    v = pd.read_csv(f"{ROOT}/data/voteview_all_members.csv", low_memory=False,
                    usecols=["congress", "chamber", "icpsr", "party_code", "nominate_dim1"])
    v = v[v.chamber.isin(["House", "Senate"])]
    first = v.groupby("icpsr").congress.min().rename("first_c")
    ids = set(panel.MemberICPSR)
    v = v[v.icpsr.isin(ids)].copy()
    ppar = panel.groupby("MemberICPSR").party.first()
    v["party"] = np.where(v.party_code == 100, "D", np.where(v.party_code == 200, "R", v.icpsr.map(ppar)))
    v["start"] = v.congress.map(start_year)
    first = first.reset_index()
    first["first_start"] = first.first_c.map(start_year)
    return v, first.set_index("icpsr").first_start


def years_service(v):
    """dict icpsr -> list of (year, chamber, party), years 1989+."""
    d = {}
    for r in v.itertuples():
        for y in (r.start, r.start + 1):
            if y >= 1989:
                d.setdefault(r.icpsr, []).append((y, r.chamber, r.party))
    return d


def nominate_at(v, icpsr, T):
    s = v[v.icpsr == icpsr].dropna(subset=["nominate_dim1"])
    if s.empty:
        return np.nan
    c = (T - 1 - 1789) // 2 + 1
    j = (s.congress - c).abs().idxmin()
    return abs(s.loc[j, "nominate_dim1"])


def ols_cluster(X, y, cl):
    """OLS with CR1 cluster-robust SE. cl: integer codes. Returns beta, se, resid, V."""
    X = np.asarray(X, float); y = np.asarray(y, float)
    n, k = X.shape
    XtXi = np.linalg.pinv(X.T @ X)
    b = XtXi @ X.T @ y
    e = y - X @ b
    cl = np.asarray(cl)
    G = len(np.unique(cl))
    sc = X * e[:, None]
    order = np.argsort(cl, kind="stable")
    cs = cl[order]
    idx = np.r_[0, np.where(np.diff(cs) != 0)[0] + 1]
    S = np.add.reduceat(sc[order], idx, axis=0)
    meat = S.T @ S
    adj = G / (G - 1) * (n - 1) / max(n - k, 1)
    V = adj * XtXi @ meat @ XtXi
    return b, np.sqrt(np.diag(V)), e, V


def wild_boot_t(X, y, cl, j, B=B_BOOT, seed=SEED):
    """Wild cluster bootstrap (Rademacher), null-imposed on coefficient j. Returns dict with p, boot-t CI."""
    X = np.asarray(X, float); y = np.asarray(y, float); cl = np.asarray(cl)
    b, se, e, V = ols_cluster(X, y, cl)
    t0 = b[j] / se[j]
    Xr = np.delete(X, j, axis=1)
    br = np.linalg.pinv(Xr.T @ Xr) @ Xr.T @ y
    fit = Xr @ br; u = y - fit
    order = np.argsort(cl, kind="stable")
    Xs, fs, us, cs = X[order], fit[order], u[order], cl[order]
    idx = np.r_[0, np.where(np.diff(cs) != 0)[0] + 1]
    G = len(idx)
    gid = np.repeat(np.arange(G), np.diff(np.r_[idx, len(cs)]))
    n, k = Xs.shape
    P = np.linalg.pinv(Xs.T @ Xs) @ Xs.T
    XtXi = np.linalg.pinv(Xs.T @ Xs)
    adj = G / (G - 1) * (n - 1) / max(n - k, 1)
    rng = np.random.default_rng(seed)
    ts = np.empty(B)
    for i in range(B):
        w = rng.choice([-1.0, 1.0], G)[gid]
        ys = fs + w * us
        bb = P @ ys
        ee = ys - Xs @ bb
        S = np.add.reduceat(Xs * ee[:, None], idx, axis=0)
        Vb = adj * XtXi @ (S.T @ S) @ XtXi
        ts[i] = bb[j] / np.sqrt(Vb[j, j])
    p = (np.sum(np.abs(ts) >= abs(t0)) + 1) / (B + 1)
    # CI by test inversion approximated with bootstrap-t quantiles of the null distribution
    q = np.percentile(np.abs(ts), 95); q90 = np.percentile(np.abs(ts), 90)
    return dict(b=float(b[j]), se=float(se[j]), t=float(t0), p_wild=float(p), ci_wild=[float(b[j] - q * se[j]), float(b[j] + q * se[j])],
                ci90_wild=[float(b[j] - q90 * se[j]), float(b[j] + q90 * se[j])], ci90_cl=[float(b[j] - 1.645 * se[j]), float(b[j] + 1.645 * se[j])],
                ci_cl=[float(b[j] - 1.96 * se[j]), float(b[j] + 1.96 * se[j])], p_cl=float(2 * (1 - stats.norm.cdf(abs(t0)))))


def dl_meta(b, se):
    b = np.asarray(b, float); se = np.asarray(se, float)
    w = 1 / se ** 2
    fe = (w * b).sum() / w.sum()
    Q = float((w * (b - fe) ** 2).sum()); df = len(b) - 1
    C = w.sum() - (w ** 2).sum() / w.sum()
    tau2 = max(0.0, (Q - df) / C) if df > 0 else 0.0
    ws = 1 / (se ** 2 + tau2)
    re = (ws * b).sum() / ws.sum(); sre = np.sqrt(1 / ws.sum())
    return dict(fe=float(fe), re=float(re), se_re=float(sre), ci_re=[float(re - 1.96 * sre), float(re + 1.96 * sre)], Q=Q, df=df,
                p_Q=float(1 - stats.chi2.cdf(Q, df)) if df > 0 else None, I2=float(max(0, (Q - df) / Q)) if Q > 0 and df > 0 else 0.0, tau2=float(tau2))


def codes(x):
    return pd.factorize(pd.Series(x))[0]
