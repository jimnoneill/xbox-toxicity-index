"""Congress as a system of coupled on/off units (pairwise maximum-entropy, or Ising, form), exploratory.

Unit and time: s_i(t) = 1 if member i started at least one event in hour t. Members: those with at least 150 events in
every calendar year 2018 to 2022 (years_shared.csv). Hours: 2018-01-01 to 2022-12-31 UTC.
Each member's conditional probability of being active is a logistic function of
  M0  own schedule (168 Eastern hour-of-week levels) and own level (log of 1 + active hours in the preceding 28 days)
  M1  M0 + one collective term: log(1 + number of other members active in the same hour)
  M3  M0 + two collective terms: the same count split into own party and other parties
  M2  M0 + one coupling J_ij to every other member (pseudo-likelihood of the pairwise model, ridge penalty)
M1 to M3 take the fitted M0 logit as one input. The ridge strength of M2 is chosen on 2021 after fitting on 2018 to
2020; all models are then refitted on 2018 to 2021 and scored on 2022 (held-out log-likelihood per member-hour).
The number K of members active in the same hour is simulated for 2022 from M0 (independent draws) and from M1 and M2
(Gibbs sampling of the fitted conditionals) and compared with the observed K.

    python run_spin.py [--workers 16]     writes output/v7/temporal_events/shared/spin_*.csv|npy and spin_summary.json
"""
import argparse
import json
import pickle
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")
ROOT = Path.home() / "x_box/pipeline"
OUT = ROOT / "output/v7/temporal_events/shared"
CGRID = [0.001, 0.003, 0.01, 0.03, 0.1]
G = {}


def ll(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float((y * np.log(p) + (1 - y) * np.log(1 - p)).sum())


def setup():
    yy = pd.read_csv(OUT / "years_shared.csv")
    yy = yy[yy.model == "hawkes_exp"]
    c = yy.groupby("bg").year.nunique()
    bgs = sorted(c[c == 5].index)
    own = pickle.load(open(OUT / "own_hourly.pkl", "rb"))
    h0 = json.load(open(OUT / "slot0.json"))["h0"]
    bgs = [b for b in bgs if b in own and b != "LEE"]
    nh = len(own[bgs[0]])
    ts = pd.to_datetime(h0 + 3600 * np.arange(nh), unit="s", utc=True)
    A = np.stack([(own[b] > 0) for b in bgs], axis=1).astype("float32")           # all hours x members
    lvl = np.log1p(pd.DataFrame(A).rolling(28 * 24, min_periods=1).sum().shift(1).fillna(0).values).astype("float32")
    sel = np.asarray((ts >= "2018-01-01") & (ts < "2023-01-01"))
    loc = ts.tz_convert("America/New_York")
    how = np.asarray(loc.dayofweek * 24 + loc.hour)[sel]
    yr = np.asarray(ts.year)[sel]
    am = pd.read_csv(ROOT / "output/v7/ti/account_member_map.csv", dtype=str)
    am = am[am.entity_type == "member"].drop_duplicates("bioguide").set_index("bioguide")
    G.update(bgs=bgs, S=A[sel], lvl=lvl[sel], how=how, yr=yr, party=np.array(am.party.reindex(bgs).astype(object).fillna("NA").tolist(), dtype=object),
             chamber=np.array(am.chamber.reindex(bgs).astype(object).fillna("NA").tolist(), dtype=object),
             state=np.array(am.state.reindex(bgs).astype(object).fillna("NA").tolist(), dtype=object))
    G["H"] = np.eye(168, dtype="float32")[how]
    G["K"] = G["S"].sum(axis=1)
    pa = G["party"]
    G["Kparty"] = {p: G["S"][:, pa == p].sum(axis=1) for p in set(pa.tolist())}


def base_fit(i, tr):
    X = np.column_stack([G["H"], G["lvl"][:, i]])
    m = LogisticRegression(C=10.0, max_iter=400).fit(X[tr], G["S"][tr, i])
    return m.decision_function(X).astype("float32")


def collective(i):
    S = G["S"]
    k_all = G["K"] - S[:, i]
    k_own = G["Kparty"][G["party"][i]] - S[:, i]
    return np.log1p(k_all), np.log1p(k_own), np.log1p(k_all - k_own)


def task_val(i):
    yr, S = G["yr"], G["S"]
    tr, va = yr <= 2020, yr == 2021
    if S[tr, i].sum() < 5:
        return i, [np.nan] * len(CGRID)
    b = base_fit(i, tr)
    X = np.column_stack([b, np.delete(S, i, axis=1)])
    out = []
    for C in CGRID:
        m = LogisticRegression(C=C, max_iter=300).fit(X[tr], S[tr, i])
        out.append(ll(S[va, i], m.predict_proba(X[va])[:, 1]))
    return i, out


def task_final(args):
    i, C = args
    yr, S = G["yr"], G["S"]
    tr, te = yr <= 2021, yr == 2022
    y = S[:, i]
    b = base_fit(i, tr)
    ka, ko, kx = collective(i)
    res = dict(i=i, bg=G["bgs"][i], n_test=int(te.sum()), active_share=float(y[te].mean()))
    p0 = 1 / (1 + np.exp(-b))
    res["ll0"] = ll(y[te], p0[te])
    X1 = np.column_stack([b, ka])
    m1 = LogisticRegression(C=10.0, max_iter=300).fit(X1[tr], y[tr])
    res["ll1"], res["g1"] = ll(y[te], m1.predict_proba(X1[te])[:, 1]), float(m1.coef_[0, 1])
    X3 = np.column_stack([b, ko, kx])
    m3 = LogisticRegression(C=10.0, max_iter=300).fit(X3[tr], y[tr])
    res["ll3"], res["g_own"], res["g_other"] = ll(y[te], m3.predict_proba(X3[te])[:, 1]), float(m3.coef_[0, 1]), float(m3.coef_[0, 2])
    X2 = np.column_stack([b, np.delete(S, i, axis=1)])
    m2 = LogisticRegression(C=C, max_iter=400).fit(X2[tr], y[tr])
    res["ll2"] = ll(y[te], m2.predict_proba(X2[te])[:, 1])
    J = np.insert(m2.coef_[0, 1:], i, 0.0)
    return res, J.astype("float32"), b[te], (float(m1.intercept_[0]), float(m1.coef_[0, 0]), float(m1.coef_[0, 1])), \
        (float(m2.intercept_[0]), float(m2.coef_[0, 0]))


def gibbs(base, J, a, bcoef, sweeps, rng, mean_field=None):
    """Sample s for every test hour. base: hours x N logit of M0. Pairwise: logit_i = a_i + b_i base_i + sum_j J[i, j] s_j.
    Mean field: logit_i = a_i + b_i base_i + g_i log(1 + number of others active)."""
    T, N = base.shape
    s = (rng.random((T, N)) < 1 / (1 + np.exp(-base))).astype("float32")
    k = s.sum(axis=1)
    for _ in range(sweeps):
        for i in rng.permutation(N):
            if mean_field is None:
                z = a[i] + bcoef[i] * base[:, i] + s @ J[i]
            else:
                z = a[i] + bcoef[i] * base[:, i] + mean_field[i] * np.log1p(k - s[:, i])
            new = (rng.random(T) < 1 / (1 + np.exp(-z))).astype("float32")
            k += new - s[:, i]
            s[:, i] = new
    return s.sum(axis=1)


def kstats(k, q):
    return dict(mean=round(float(k.mean()), 2), sd=round(float(k.std()), 2),
                **{f"share_hours_at_or_above_{name}": round(100 * float((k >= v).mean()), 2) for name, v in q.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    setup()
    N = len(G["bgs"])
    print("members", N, "hours", len(G["yr"]), flush=True)
    t0 = time.time()
    with Pool(a.workers) as p:
        val = dict(p.imap_unordered(task_val, range(N), chunksize=2))
        V = np.array([val[i] for i in range(N)])
        tot = np.nansum(V, axis=0)
        C = CGRID[int(tot.argmax())]
        print("validation ll by C", dict(zip(CGRID, tot.round(0))), "chosen", C, round(time.time() - t0), flush=True)
        fin = p.map(task_final, [(i, C) for i in range(N)], chunksize=2)
    fin.sort(key=lambda r: r[0]["i"])
    R = pd.DataFrame([f[0] for f in fin])
    R["party"], R["chamber"] = G["party"], G["chamber"]
    R.to_csv(OUT / "spin_members.csv", index=False)
    Jraw = np.stack([f[1] for f in fin])
    J = (Jraw + Jraw.T) / 2
    np.save(OUT / "spin_J.npy", J)
    base = np.column_stack([f[2] for f in fin])
    m1p = np.array([f[3] for f in fin])
    m2p = np.array([f[4] for f in fin])
    print("fits done", round(time.time() - t0), flush=True)

    te = G["yr"] == 2022
    n = float(R.n_test.sum())
    tot = {m: float(R["ll" + m].sum()) for m in "0123"}
    out = dict(members=N, hours_fit=int((G["yr"] <= 2021).sum()), hours_test=int(te.sum()), ridge_C=C,
               active_share_median_pct=round(100 * float(R.active_share.median()), 1),
               party_counts=pd.Series(G["party"]).value_counts().to_dict(), chamber_counts=pd.Series(G["chamber"]).value_counts().to_dict(),
               heldout_ll_per_1000_member_hours={m: round(1000 * tot[m] / n, 2) for m in tot},
               gain_over_schedule_pct={m: round(100 * (tot[m] - tot["0"]) / abs(tot["0"]), 2) for m in "123"},
               collective_share_of_pairwise_gain_pct=round(100 * (tot["1"] - tot["0"]) / (tot["2"] - tot["0"]), 1),
               party_split_share_of_pairwise_gain_pct=round(100 * (tot["3"] - tot["0"]) / (tot["2"] - tot["0"]), 1),
               members_better_with_collective_pct=round(100 * float((R.ll1 > R.ll0).mean()), 1),
               members_better_pairwise_than_collective_pct=round(100 * float((R.ll2 > R.ll1).mean()), 1),
               members_better_party_split_than_collective_pct=round(100 * float((R.ll3 > R.ll1).mean()), 1),
               g_collective_median=round(float(R.g1.median()), 3), g_own_party_median=round(float(R.g_own.median()), 3),
               g_other_party_median=round(float(R.g_other.median()), 3),
               own_party_coefficient_larger_pct=round(100 * float((R.g_own > R.g_other).mean()), 1))
    # couplings by pair type
    iu = np.triu_indices(N, 1)
    same_p = (G["party"][:, None] == G["party"][None, :])[iu]
    same_c = (G["chamber"][:, None] == G["chamber"][None, :])[iu]
    same_s = (G["state"][:, None] == G["state"][None, :])[iu]
    j = J[iu]
    rng = np.random.default_rng(20261110)

    def diff(mask):
        d = float(j[mask].mean() - j[~mask].mean())
        bs = []
        for _ in range(400):
            idx = rng.integers(0, N, N)
            w = np.bincount(idx, minlength=N).astype(float)
            ww = (w[:, None] * w[None, :])[iu]
            a1, a0 = (ww * mask).sum(), (ww * ~mask).sum()
            if a1 > 0 and a0 > 0:
                bs.append((ww * mask * j).sum() / a1 - (ww * ~mask * j).sum() / a0)
        return dict(mean_same=round(float(j[mask].mean()), 4), mean_different=round(float(j[~mask].mean()), 4), difference=round(d, 4),
                    ci_low=round(float(np.quantile(bs, .025)), 4), ci_high=round(float(np.quantile(bs, .975)), 4), pairs_same=int(mask.sum()))

    out["couplings"] = dict(pairs=int(len(j)), positive_pct=round(100 * float((j > 0).mean()), 1), mean=round(float(j.mean()), 4),
                            sd=round(float(j.std()), 4), party=diff(same_p), chamber=diff(same_c), state=diff(same_s))
    # number of members active in the same hour, 2022
    kobs = G["K"][te]
    q = {"observed_q95": float(np.quantile(kobs, .95)), "observed_q99": float(np.quantile(kobs, .99))}
    p0 = 1 / (1 + np.exp(-base))
    k0 = np.concatenate([(rng.random(p0.shape) < p0).sum(axis=1) for _ in range(5)])
    k1 = np.concatenate([gibbs(base, None, m1p[:, 0], m1p[:, 1], 25, rng, mean_field=m1p[:, 2]) for _ in range(3)])
    k2 = np.concatenate([gibbs(base, Jraw, m2p[:, 0], m2p[:, 1], 25, rng) for _ in range(3)])
    out["K"] = dict(thresholds=q, observed=kstats(kobs, q), schedule_only=kstats(k0, q), collective_term=kstats(k1, q), pairwise=kstats(k2, q))
    json.dump(out, open(OUT / "spin_summary.json", "w"), indent=1)
    print(json.dumps(out, indent=1))
    print("spin done", round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
