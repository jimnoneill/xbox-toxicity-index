"""RVT-E test E2: split-half reliability of a member's response. Follows scripts/v7/rvt_e_DESIGN.md.
Run from the scripts/v7 folder: the member-day state is built by the first block of rvt_c_01_state.py."""
import json
import os

import numpy as np
import pandas as pd

exec(open("rvt_c_01_state.py").read().split("# ---------------------------------------------------------------- responses to the general-election window")[0])
O = ROOT + "/rvt_e"
os.makedirs(O, exist_ok=True)
rng = np.random.default_rng(20261112)
NB = 2000


def agg(df, minposts, suffix):
    g = df.groupby("bioguide").agg(ti_sum=("ti_sum", "sum"), n=("n", "sum"), events=("events", "sum"), days=("date", "size"))
    out = pd.DataFrame({"ti7": np.where(g.n >= minposts, g.ti_sum / g.n.where(g.n > 0), np.nan),
                        "lvol": np.log((g.events + 0.5) / g.days), "days": g.days}, index=g.index)
    return out.add_suffix(suffix)


def halves(df):
    """odd and even days of a period, numbered in date order"""
    k = df.date.rank(method="dense").astype(int)
    return df[k % 2 == 1], df[k % 2 == 0]


def build(periods_a, periods_b, keep, minhalf):
    """response = state(a) - state(b), full and by halves; keep(full table) applies the coverage rule"""
    ao, ae = halves(periods_a)
    bo, be = halves(periods_b)
    m = agg(periods_a, 10, "_a").join(agg(periods_b, 10, "_b"), how="inner")
    m = m[keep(m)]
    for tag, x, y in (("o", ao, bo), ("e", ae, be)):
        m = m.join(agg(x, minhalf, "_a" + tag)).join(agg(y, minhalf, "_b" + tag))
    for v in ("ti7", "lvol"):
        m["R_" + v] = m[v + "_a"] - m[v + "_b"]
        m["Ro_" + v] = m[v + "_ao"] - m[v + "_bo"]
        m["Re_" + v] = m[v + "_ae"] - m[v + "_be"]
        m["L_" + v], m["Lo_" + v], m["Le_" + v] = m[v + "_a"], m[v + "_ao"], m[v + "_ae"]
    return m


def demean(x, g):
    return x - x.groupby(g).transform("mean")


def corr(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() > 2 else np.nan, int(ok.sum())


def stats(H, P, pre):
    """H: member-unit rows with halves; P: pairs of successive units; pre = 'R' or 'L'; returns dict of statistics"""
    out = {}
    for v in ("ti7", "lvol"):
        o, e = demean(H[pre + "o_" + v], H.unit), demean(H[pre + "e_" + v], H.unit)
        rh, n = corr(o.values, e.values)
        rel = 2 * rh / (1 + rh)
        d = dict(split_half_r=rh, reliability=rel, n_units=n)
        if P is not None:
            Q = P[P.var_ == pre + "_" + v]
            rx, npair = corr(Q.prev.values, Q.cur.values)
            rxd, _ = corr(demean(Q.prev, Q.unit).values, demean(Q.cur, Q.unit).values)
            d.update(cross_r_raw=rx, cross_r_demeaned=rxd, n_pairs=npair, cross_over_reliability=rxd / rel if rel > 0 else np.nan)
        out[v] = d
    return out


def boot(H, P, pre):
    H = H.reset_index(drop=True)
    ids = np.array(sorted(set(H.bioguide)))
    hi = H.groupby("bioguide").indices
    empty = np.array([], dtype=int)
    if P is not None:
        P = P.reset_index(drop=True)
        pi = P.groupby("bioguide").indices
    draws = []
    for _ in range(NB):
        s = rng.choice(ids, len(ids), replace=True)
        Hb = H.iloc[np.concatenate([hi[b] for b in s])].reset_index(drop=True)
        Pb = P.iloc[np.concatenate([pi.get(b, empty) for b in s])].reset_index(drop=True) if P is not None else None
        draws.append(stats(Hb, Pb, pre))
    res = stats(H, P, pre)
    for v in res:
        for k in list(res[v]):
            if k.startswith("n_"):
                continue
            a = np.array([d[v][k] for d in draws], dtype=float)
            res[v][k + "_ci95"] = [float(np.nanquantile(a, 0.025)), float(np.nanquantile(a, 0.975))]
    return res


RES = {}
CYC = {2018: ("2018-09-03", "2018-11-06"), 2020: ("2020-09-07", "2020-11-03"), 2022: ("2022-09-05", "2022-11-08")}
for minhalf in (5, 10):
    rows = []
    for c, (lab, elec) in CYC.items():
        lab, elec = pd.Timestamp(lab), pd.Timestamp(elec)
        win = D[(D.date >= lab) & (D.date <= elec)]
        pre = D[(D.date >= lab - pd.Timedelta(days=56)) & (D.date < lab)]
        need = 0.9 * ((elec - lab).days + 1)
        m = build(win, pre, lambda t: (t.days_a >= need) & (t.days_b >= 50), minhalf)
        m = m.join(win.groupby("bioguide").on_ballot_this_cycle.max().rename("ballot"))
        m["unit"] = c
        rows.append(m.reset_index())
    H = pd.concat(rows, ignore_index=True)
    H = H[H.ballot == 1].copy()
    recs = []
    for var in ("R_ti7", "R_lvol", "L_ti7", "L_lvol"):
        W = H.pivot(index="bioguide", columns="unit", values=var)
        for c in (2020, 2022):
            d = W[[c - 2, c]].dropna()
            recs += [dict(bioguide=b, unit=c, var_=var, prev=p, cur=q) for b, p, q in zip(d.index, d[c - 2], d[c])]
    P = pd.DataFrame(recs)
    key = f"election_min{minhalf}"
    RES[key] = dict(response=boot(H, P, "R"), level=boot(H, P, "L"), member_cycles_on_ballot=int(len(H)), members=int(H.bioguide.nunique()))
    if minhalf == 5:
        H.to_csv(O + "/e2_election_halves.csv", index=False)
    print(key, json.dumps(RES[key], indent=1), flush=True)

# secondary: displacement at changes in party control (C4 construction)
rows = []
for tr in ("2019-01-03", "2021-01-20", "2023-01-03"):
    t = pd.Timestamp(tr)
    before = D[(D.date >= t - pd.Timedelta(days=180)) & (D.date < t)]
    after = D[(D.date >= t) & (D.date < t + pd.Timedelta(days=180))]
    m = build(after, before, lambda x: (x.days_a >= 170) & (x.days_b >= 170), 5)
    p0 = cal[cal.date == t - pd.Timedelta(days=1)].set_index("bioguide")
    p1 = cal[cal.date == t].set_index("bioguide")
    dp = ((p1.aligned_with_president + p1.in_chamber_majority) - (p0.aligned_with_president + p0.in_chamber_majority)).dropna()
    m = m.join(dp.rename("dpower"), how="inner")
    m = m[m.dpower != 0]
    m["unit"] = [tr + ("_lost" if x < 0 else "_gained") for x in m.dpower]
    rows.append(m.reset_index())
H = pd.concat(rows, ignore_index=True)
RES["displacement_min5"] = dict(response=boot(H, None, "R"), member_changes=int(len(H)), members=int(H.bioguide.nunique()))
print("displacement", json.dumps(RES["displacement_min5"], indent=1), flush=True)
json.dump(RES, open(O + "/e2_reliability.json", "w"), indent=1)
