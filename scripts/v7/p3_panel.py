"""P3 step 1: member x political-year panel (new file; reads, never writes, other tasks' outputs).

Political year PY y = 1 July y to 30 June y+1 (Eastern). PY2017..PY2022 are complete in the full corpus
(2017-06-21 to 2023-07-11); calendar 2017 and 2023 are partial, which is why PY is primary.

Per member-PY, using only that PY's data (so every feature is available at the end of the PY):
  volume       events (thread-collapsed non-retweets, all of the member's accounts pooled), in-office days,
               observed share, rate = events / (in-office days x observed share)
  descriptors  A_n, rapid-follow-up share, sessions/day, events/session, entropy, night, weekend (t1_common.describe)
  account mix  campaign share, scheduler-like share, tool-client share, phone share
  co-movement  OLS of log1p(own daily events) on log1p(leave-member-out Congress daily events), own lag-1 day,
               weekend, within the PY only (a daily, past-only stand-in for the full-period hourly NB of T4b)
  content      mean of 9 probabilities on ORIGINAL posts, TI7 mean, day-clustered SE, within-PY TI7 slope
  static       party, chamber, state, Nokken-Poole dim1 (Congress current at the end of the PY), seniority,
               seat-up flags, alignment with the president and majority at the end of the PY

Usage: python p3_panel.py [--ti sample|full] [--workers 16]
Output: output/v7/p3/panel_member_py_{ti}.csv
"""
import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t1_common as c1  # noqa: E402
from t1full_common import *  # noqa: E402,F401,F403

c1.MIN_N = 40  # yearly streams are shorter than the 200-event rule used for whole-period descriptors
OUT = ROOT / "output/v7/p3"
PYS = list(range(2017, 2023))
DIMS7 = ["p_offensive", "p_toxic", "p_neg_sentiment", "p_irony", "p_ragebait", "p_tribal", "p_performative"]
DIMS = DIMS7 + ["p_genuine_anger", "p_advocacy"]
G = {}
ET = "America/New_York"


def py_bounds(y):
    lo = pd.Timestamp(f"{y}-07-01", tz=ET).value // 10 ** 9
    hi = pd.Timestamp(f"{y + 1}-07-01", tz=ET).value // 10 ** 9
    return int(lo), int(hi)


def et_day(t):
    s = pd.to_datetime(np.asarray(t), unit="s", utc=True).tz_convert(ET).tz_localize(None)
    return s.values.astype("datetime64[D]").astype("int64")


def comove(own_days, y_lo_day, y_hi_day, cong, ok_day, in_office):
    """OLS within one PY. Returns beta on log1p(LOO Congress), beta on own lag-1, weekend coef, n days, R2."""
    d = np.arange(y_lo_day, y_hi_day)
    cnt = np.bincount(own_days - y_lo_day + 1, minlength=len(d) + 1)[: len(d) + 1]  # index 0 = day before
    own = cnt[1:]
    lag = cnt[:-1]
    loo = cong[d - G["day0"]] - own
    keep = ok_day[d - G["day0"]] & in_office[d - G["day0"]]
    keep[0] = False  # lag of the first day is outside the PY
    if keep.sum() < 120:
        return (np.nan,) * 3 + (int(keep.sum()), np.nan)
    wk = ((d + 3) % 7 >= 5).astype(float)  # 1970-01-01 was a Thursday
    X = np.column_stack([np.ones(keep.sum()), np.log1p(loo[keep]), np.log1p(lag[keep]), wk[keep]])
    yv = np.log1p(own[keep])
    beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    res = yv - X @ beta
    r2 = 1 - res.var() / yv.var() if yv.var() > 0 else np.nan
    return float(beta[1]), float(beta[2]), float(beta[3]), int(keep.sum()), float(r2)


def task(bg):
    ev = G["ev"].get(bg)
    if ev is None or len(ev) == 0:
        return []
    ml = G["loc"].get(bg)
    loc_ok = lc.model_usable(ml) if ml is not None else False
    t_all = ev.t.values
    rows = []
    cal = G["cal"].get(bg)
    for y in PYS:
        lo, hi = py_bounds(y)
        m = (t_all >= lo) & (t_all < hi)
        n = int(m.sum())
        if n == 0:
            continue
        e = ev[m]
        t = e.t.values
        cum, hour0 = G["cum"], G["hour0"]
        v = hy.gap_valid(t, cum, hour0) if len(t) > 1 else np.zeros(0, bool)
        r = c1.describe(t, ml, loc_ok, v, cum, hour0, e["size"].values)
        r = {k: r.get(k, np.nan) for k in ("A_n", "rapid_share_300", "sessions_per_day_30", "events_per_session_30",
                                            "entropy", "night_00_06", "weekend_share", "zero_gap_share",
                                            "exposure_days", "active_days")}
        rows_i = dict(bioguide=bg, py=y, n_events=n, n_posts=int(e["size"].sum()), **r)
        rows_i["campaign_share"] = float((e.account_type.values == "campaign").mean())
        rows_i["sched_share"] = float(e.sched_like.mean())
        rows_i["tool_share"] = float(e.tool_client.mean())
        rows_i["phone_share"] = float((e.cclass.values == "phone").mean())
        mid = lo + (hi - lo) // 2
        rows_i["vol_trend"] = float(np.log((np.sum(t >= mid) + 1.0) / (np.sum(t < mid) + 1.0)))
        # in-office days and observed share
        d_lo, d_hi = int(et_day([lo])[0]), int(et_day([hi])[0])
        if cal is not None:
            io = cal[d_lo - G["cal_day0"]: d_hi - G["cal_day0"]]
            rows_i["in_office_days"] = int(io.sum())
        else:
            rows_i["in_office_days"] = 0
        rows_i["obs_share"] = float(G["ok_day"][d_lo - G["day0"]: d_hi - G["day0"]].mean())
        b1, b2, b3, nd, r2 = comove(et_day(t), d_lo, d_hi, G["cong"], G["ok_day"], G["io_full"].get(bg, G["all_true"]))
        rows_i.update(comove_beta=b1, momentum_beta=b2, comove_weekend=b3, comove_days=nd, comove_r2=r2)
        rows.append(rows_i)
    return rows


def ti_table(source):
    path = ROOT / ("output/v7/ti/post_probs.parquet" if source == "sample" else "output/v7/full/post_probs_full.parquet")
    cols = ["screen_name", "created_at", "post_type"] + DIMS
    p = pd.read_parquet(path, columns=cols)
    p = p[p.post_type == "original"].copy()
    am = load_amap()[["screen_name", "bioguide"]].drop_duplicates("screen_name")
    p = p.merge(am, on="screen_name")
    t = p.created_at.dt.tz_convert("UTC").dt.tz_localize(None).values.astype("datetime64[s]").astype("int64")
    p["t"] = t
    et = pd.to_datetime(t, unit="s", utc=True).tz_convert(ET)
    p["py"] = np.where(et.month >= 7, et.year, et.year - 1)
    p["day"] = et.tz_localize(None).values.astype("datetime64[D]").astype("int64")
    p = p[(p.py >= PYS[0]) & (p.py <= PYS[-1])]
    p["ti7"] = p[DIMS7].mean(axis=1)
    lo = np.array([py_bounds(y)[0] for y in range(PYS[0], PYS[-1] + 1)])
    hi = np.array([py_bounds(y)[1] for y in range(PYS[0], PYS[-1] + 1)])
    k = p.py.values - PYS[0]
    p["frac"] = (p.t.values - lo[k]) / (hi[k] - lo[k]) - 0.5
    return p


def ti_agg(p):
    rows = []
    for (bg, y), g in p.groupby(["bioguide", "py"]):
        n = len(g)
        x = g.ti7.values
        mean = x.mean()
        # day-clustered SE of the mean (ratio estimator)
        dd = pd.DataFrame({"d": g.day.values, "r": x - mean})
        s = dd.groupby("d").r.sum().values
        se = float(np.sqrt((s ** 2).sum()) / n) if len(s) > 1 else np.nan
        f = g.frac.values
        slope = float(np.cov(f, x, bias=True)[0, 1] / f.var()) if n >= 10 and f.var() > 0 else np.nan
        r = dict(bioguide=bg, py=y, n_scored=n, n_days_scored=int(len(s)), ti7=float(mean), ti7_se=se, ti7_slope=slope)
        for d in DIMS:
            r[d[2:]] = float(g[d].mean())
        rows.append(r)
    return pd.DataFrame(rows)


def static_attrs(am):
    mm = (am.sort_values("n_tweets", ascending=False).groupby("bioguide")
          .agg(icpsr=("icpsr", "first"), party=("party", "first"), chamber=("chamber", "first"),
               state=("state", "first"), np_static=("np_dim1", "first")).reset_index())
    vv = pd.read_csv(ROOT / "data/voteview_all_members.csv", usecols=["congress", "icpsr", "nokken_poole_dim1"])
    allv = pd.read_csv(ROOT / "data/voteview_all_members.csv", usecols=["congress", "icpsr"])
    first = allv.groupby("icpsr").congress.min()
    npd = vv.groupby(["icpsr", "congress"]).nokken_poole_dim1.mean()
    return mm, first, npd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ti", default="sample", choices=["sample", "full"])
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    m, hour0, obs, cum = load_mask()
    amap = load_amap()
    ev = pd.read_parquet(T1 / "events.parquet", columns=["screen_name", "bioguide", "account_type", "t", "size",
                                                         "sched_like", "tool_client", "cclass"]).sort_values("t")
    print("events", len(ev), flush=True)
    # day axes
    day0 = int(et_day([ev.t.min()])[0]) - 3
    day1 = int(et_day([ev.t.max()])[0]) + 3
    G.update(hour0=hour0, cum=cum, day0=day0)
    ok_day = np.zeros(day1 - day0 + 1, bool)
    mm_ = m.groupby("et_day")["observed"].mean()
    ok_day[(mm_.index.values - day0).clip(0, len(ok_day) - 1)] = (mm_.values >= 0.9)
    G["ok_day"] = ok_day
    G["cong"] = np.bincount(et_day(ev.t.values) - day0, minlength=len(ok_day)).astype(float)
    G["all_true"] = np.ones(len(ok_day), bool)
    # in-office day indicators
    cal = pd.read_parquet(ROOT / "output/v7/t3/calendar_member_day.parquet",
                          columns=["date", "bioguide", "in_office", "aligned_with_president", "in_chamber_majority",
                                   "president_party"])
    cal["d"] = cal.date.values.astype("datetime64[D]").astype("int64")
    G["cal_day0"] = day0
    G["cal"], G["io_full"] = {}, {}
    for bg, g in cal.groupby("bioguide"):
        arr = np.zeros(len(ok_day), bool)
        dd = g.d.values - day0
        k = (dd >= 0) & (dd < len(arr))
        arr[dd[k]] = g.in_office.values[k] == 1
        G["cal"][bg] = arr
        G["io_full"][bg] = arr
    G["ev"] = {b: g.reset_index(drop=True) for b, g in ev.groupby("bioguide")}
    G["loc"], mm = build_locations(amap, {b: g.t.values for b, g in G["ev"].items()})
    print("setup", round(time.time() - t0), "s", flush=True)
    bgs = sorted(G["ev"], key=lambda b: -len(G["ev"][b]))
    rows = []
    with Pool(a.workers) as p:
        for i, r in enumerate(p.imap_unordered(task, bgs, chunksize=2)):
            rows += r
            if i % 100 == 0:
                print(i, len(bgs), round(time.time() - t0), "s", flush=True)
    base = pd.DataFrame(rows)
    # content
    ti = ti_agg(ti_table(a.ti))
    panel = base.merge(ti, on=["bioguide", "py"], how="left")
    # static + calendar flags
    st, first, npd = static_attrs(amap)
    panel = panel.merge(st, on="bioguide", how="left")
    cong_end = (panel.py + 1 - 1789) // 2 + 1  # Congress sitting at 30 June of PY y (year y+1)
    panel["congress_end"] = cong_end
    panel["seniority_congresses"] = cong_end - panel.icpsr.map(first).fillna(cong_end)
    vals = []
    for ic, c in zip(panel.icpsr, cong_end):
        v = npd.get((ic, c), np.nan)
        if np.isnan(v):
            v = npd.get((ic, c - 1), np.nan)
        vals.append(v)
    panel["np_dim1"] = vals
    panel["np_dim1"] = panel.np_dim1.fillna(panel.np_static)
    panel["abs_np_dim1"] = panel.np_dim1.abs()
    # end-of-PY alignment / majority from the calendar
    end_day = np.array([int(et_day([py_bounds(y)[1] - 86400])[0]) for y in range(2017, 2023)])
    cal_end = cal[cal.d.isin(end_day)][["bioguide", "d", "aligned_with_president", "in_chamber_majority"]]
    dmap = {d: y for d, y in zip(end_day, range(2017, 2023))}
    cal_end["py"] = cal_end.d.map(dmap)
    panel = panel.merge(cal_end.drop(columns="d"), on=["bioguide", "py"], how="left")
    # seat up (scheduled, known in advance): House always; Senate by class
    el = pd.read_csv(ROOT / "output/v7/t3/elections.csv")
    el["seat_up"] = np.where(el.ballot_reason == "senate_seat_not_up", 0, 1)
    su = el[el.cycle.isin([2018, 2020, 2022])].set_index(["bioguide", "cycle"]).seat_up
    # general of cycle c falls in PY c (c even); primaries in spring of c fall in PY c-1
    panel["seat_up_general"] = [su.get((b, y), np.nan) if y % 2 == 0 else 0 for b, y in zip(panel.bioguide, panel.py)]
    panel["seat_up_primary"] = [su.get((b, y + 1), np.nan) if (y + 1) % 2 == 0 else 0
                                for b, y in zip(panel.bioguide, panel.py)]
    panel["election_year"] = (panel.py % 2 == 0).astype(int)
    panel["log_rate"] = np.log(panel.n_events / (panel.in_office_days.clip(lower=1) * panel.obs_share.clip(lower=0.5)))
    panel.to_csv(OUT / f"panel_member_py_{a.ti}.csv", index=False)
    print("panel", panel.shape, round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()
