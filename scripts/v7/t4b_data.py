"""T4b data and fitting helpers (fixes from the adversarial review). New file; T4 files are untouched.

Changes relative to t4_data.py
  1. Leave-one-out Congress term excludes ALL accounts of the same member (bioguide): C_{-m,t} = total events of all matched
     accounts in hour t minus the events of every account of the focal member (office and campaign).
  2. absent_in_session is dropped (near-unidentified). Regimes: in_session, primary_window, general_window, post_election,
     aligned_president, party_majority, event_curated, event_detected.
  3. Optional slow time trend per account (natural cubic spline of calendar time, about one df per year of span): variant "trend".
  4. Lag stability: implied branching = phi_1 + phi_24 (elasticity sum of the two log-lag terms). When the free fit has
     branching above CAP the model is refitted with phi_1 + phi_24 = CAP imposed (equality constraint on the boundary; the
     reparametrisation phi_1*(lag1 - lag24) + CAP*lag24 is an ordinary offset GLM). The free fit is kept as "full_uncapped".
  5. fit_account returns the full-length covariance (robust) so predictive checks can propagate parameter uncertainty.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t4_data import *  # noqa: E402,F401,F403
import t4_data as _d  # noqa: E402
from xbox import binned_model_b as bmb  # noqa: E402

T4B = ROOT / "output/v7/t4b"
CAP = 0.9
REG_B = ["in_session", "primary_window", "general_window", "post_election", "aligned_president", "party_majority",
         "event_curated", "event_detected"]
LAGN = ("log1p_lag1", "log1p_lag2_24")
HOUR_2022 = int(np.datetime64("2022-01-01", "h").astype("int64"))
HOUR_END_HELD = int(np.datetime64("2023-07-12", "h").astype("int64"))


def load_all_b():
    n = _d.load_all()
    m, hour0, obs, cum = load_mask()
    ev = pd.read_parquet(T1 / "events.parquet", columns=["bioguide", "t"])
    h = ev.t.values // 3600 - hour0
    G["hour0"], G["nh"] = hour0, len(m)
    G["tot"] = np.bincount(h, minlength=len(m)).astype("int64")
    G["mem_hours"] = {str(b): g for b, g in pd.Series(h).groupby(ev.bioguide.astype(str).values)}
    G["mem_hours"] = {b: v.values.astype("int64") for b, v in G["mem_hours"].items()}
    return n


def member_loo(i):
    d = account_arrays(i)
    bg = G["bioguide"][i]
    hrs = d["hour_idx"]
    mem = np.bincount(G["mem_hours"].get(bg, np.zeros(0, dtype="int64")), minlength=G["nh"])
    loo = G["tot"][hrs - G["hour0"]] - mem[hrs - G["hour0"]]
    assert loo.min() >= 0
    return loo


def account_design_b(i, variant="full", end_hour=None, trend_df=None):
    """Design for account i. variant: full, nolag, noC, base, csplit, noreg, trend. end_hour: fit rows restricted to
    hour_idx < end_hour (held-out checks); the dense pieces still cover the whole account span."""
    d = account_arrays(i)
    bg = G["bioguide"][i]
    cal_arr = G["cal"].get(bg)
    hrs = d["hour_idx"]
    ld = bm.local_day_from_hours(hrs, d["local_hour"])
    cell = (d["local_dow"] >= 5).astype(int) * 24 + d["local_hour"].astype(int)
    loo = member_loo(i)
    logc = np.log1p(loo.astype(float))
    congress = {"full": "plain", "nolag": "plain", "noC": "none", "base": "none", "csplit": "split",
                "noreg": "plain", "trend": "plain"}[variant]
    Xs, names, inoff = bm.build_static(cell, ld, logc, cal_arr, G["detected"], 1, congress,
                                       exclude_regimes=(variant == "noreg"))
    if "absent_in_session" in names:
        j = names.index("absent_in_session")
        Xs = np.delete(Xs, j, axis=1)
        names = names[:j] + names[j + 1:]
    lo = int(hrs.min())
    T = int(hrs.max()) - lo + 1
    pos = (hrs - lo).astype(int)
    ydense = np.zeros(T)
    ydense[pos] = d["count"]
    present = np.zeros(T, dtype=bool)
    present[pos[inoff]] = True
    lag1, lagL, valid = bm.lag_features(ydense, present, 24)
    fit_rows = valid[pos] & inoff
    if end_hour is not None:
        fit_rows = fit_rows & (hrs < end_hour)
    ncore = 1 + 47 + (0 if congress == "none" else (1 if congress == "plain" else 2))
    use_lags = variant not in ("nolag", "base")
    parts, nms = [Xs[:, :ncore]], names[:ncore]
    if variant == "trend":
        span_years = (hrs[fit_rows].max() - hrs[fit_rows].min()) / 8766.0 if fit_rows.any() else 1.0
        df = trend_df or int(max(3, round(span_years)))
        tt = (hrs - hrs.min()) / float(max(hrs.max() - hrs.min(), 1))
        Ball = bmb.ns_basis(tt, df)
        parts.append(Ball)
        nms = nms + [f"trend{k}" for k in range(Ball.shape[1])]
    lag_cols = np.column_stack([lag1[pos], lagL[pos]])
    if use_lags:
        parts.append(lag_cols)
        nms = nms + list(LAGN)
    parts.append(Xs[:, ncore:])
    nms = nms + names[ncore:]
    Xall = np.hstack(parts)
    return dict(X=Xall[fit_rows], names=nms, y=ydense[pos][fit_rows], cluster=(hrs[fit_rows] // 168),
                fit_rows=fit_rows, pos=pos, lo=lo, T=T, present=present, ld=ld, Xall=Xall, ydense=ydense,
                lag_cols=lag_cols, local_hour=d["local_hour"], inoff=inoff, n_rows=int(len(hrs)),
                cal_ok=cal_arr is not None, ncore=ncore, use_lags=use_lags, hrs=hrs)


def _embed(res, p):
    cov = np.zeros((p, p))
    k = res["keep"]
    cov[np.ix_(k, k)] = res["cov_robust"]
    return cov


def fit_account(des, cap=CAP):
    """NB2 fit with the lag-stability cap. The stability measure is the L1 norm |phi_1| + |phi_24| of the two log-lag
    coefficients (a sufficient condition for contraction of the log-linearised feedback). When the free fit exceeds CAP the model
    is refitted with |phi_1| + |phi_24| = CAP imposed on the face of the free fit's signs (offset-GLM reparametrisation); if the
    constrained optimum leaves that face the signs are updated (up to 3 times), then phi_1 is fixed at 0 and phi_24 at +-CAP.
    Returns dict(fit, cov, branching_free (L1), elasticity_sum_free, capped, capped_fixed, uncapped)."""
    X, y, names = des["X"], des["y"], des["names"]
    free = bm.fit_nb2(X, y, cluster=des["cluster"])
    p = X.shape[1]
    out = dict(fit=free, cov=_embed(free, p), branching_free=np.nan, elasticity_sum_free=np.nan, capped=False,
               capped_fixed=False, uncapped=free)
    if not des["use_lags"]:
        return out
    j1, jL = names.index(LAGN[0]), names.index(LAGN[1])
    b1, bL = free["beta"][j1], free["beta"][jL]
    if not (np.isfinite(b1) and np.isfinite(bL)):
        return out
    l1 = abs(b1) + abs(bL)
    out["branching_free"] = float(l1)
    out["elasticity_sum_free"] = float(b1 + bL)
    if l1 <= cap:
        return out
    s1 = 1.0 if b1 >= 0 else -1.0
    s2 = 1.0 if bL >= 0 else -1.0
    cols = [j for j in range(p) if j != jL]
    r1 = cols.index(j1)
    for attempt in range(4):
        fixed = attempt == 3
        if not fixed:
            f = s1 * s2
            Xc = X.copy()
            Xc[:, j1] = X[:, j1] - f * X[:, jL]
            Xc = np.delete(Xc, jL, axis=1)
            off = s2 * cap * X[:, jL]
            keep_cols = cols
        else:  # phi_1 fixed at 0, phi_24 = s2 * cap
            Xc = np.delete(X, [j1, jL], axis=1)
            off = s2 * cap * X[:, jL]
            keep_cols = [j for j in range(p) if j not in (j1, jL)]
        c = bm.fit_nb2(Xc, y, offset=off, cluster=des["cluster"])
        if fixed:
            break
        phi1 = c["beta"][r1]
        if not np.isfinite(phi1):
            continue
        phi24 = s2 * cap - f * phi1
        if abs(phi1) + abs(phi24) <= cap + 1e-6:
            break
        s1 = 1.0 if phi1 >= 0 else -1.0
        s2 = 1.0 if phi24 >= 0 else -1.0
    pc = Xc.shape[1]
    T = np.zeros((p, pc))
    for r, j in enumerate(keep_cols):
        T[j, r] = 1.0
    beta = np.full(p, np.nan)
    se_m = np.full(p, np.nan)
    for r, j in enumerate(keep_cols):
        if np.isfinite(c["beta"][r]):
            beta[j] = c["beta"][r]
            se_m[j] = c["se_model"][r]
    if not fixed:
        T[jL, r1] = -s1 * s2
        if np.isfinite(c["beta"][r1]):
            beta[jL] = s2 * cap - s1 * s2 * c["beta"][r1]
            se_m[jL] = c["se_model"][r1]
    else:
        beta[j1], beta[jL] = 0.0, s2 * cap
    cov = T @ _embed(c, pc) @ T.T
    se_r = np.where(np.isfinite(beta), np.sqrt(np.clip(np.diag(cov), 0, None)), np.nan)
    if fixed:
        se_r[[j1, jL]] = np.nan
        se_m[[j1, jL]] = np.nan
    fit = dict(c)
    fit.update(beta=beta, se_robust=se_r, se_model=se_m, keep=np.flatnonzero(np.isfinite(beta)))
    out.update(fit=fit, cov=cov, capped=True, capped_fixed=bool(fixed))
    return out


def eta_static(des, beta):
    """Static linear predictor on the dense grid (lags excluded); -40 where the account is not in office / not observed."""
    names = des["names"]
    b = np.where(np.isfinite(beta), beta, 0.0)
    sidx = [j for j, nm in enumerate(names) if nm not in LAGN]
    e = des["Xall"][:, sidx] @ b[sidx]
    eta = np.full(des["T"], -40.0)
    eta[des["pos"][des["inoff"]]] = e[des["inoff"]]
    return eta


def lag_coefs(des, beta):
    names = des["names"]
    p1 = beta[names.index(LAGN[0])] if LAGN[0] in names else 0.0
    pL = beta[names.index(LAGN[1])] if LAGN[1] in names else 0.0
    return (0.0 if not np.isfinite(p1) else p1), (0.0 if not np.isfinite(pL) else pL)
