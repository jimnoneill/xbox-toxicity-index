"""FULL-corpus rerun (output/v7/t2full) of T2 step 5: hourly counts per account (local time) and the leave-one-out Congress-wide hourly count.

Counts are thread-collapsed non-retweet events (events.parquet; thread start time) in observed hours only (outage mask and
the every-third-day collection design applied). Dense within each account's span (first to last event hour), restricted
to observed hours, for accounts with at least 200 events.
Congress-wide total T_t = events of ALL matched member accounts in hour t; loo_congress_count = T_t - own count.
Same for raw non-retweet posts (count_posts, loo_congress_posts).

Outputs: output/v7/t2/hourly_counts.parquet, congress_hourly.parquet, hourly_counts_meta.json
Columns of hourly_counts: account, bioguide, account_type, party, chamber, hour_idx (UTC epoch hour), hour_utc,
  local_hour, local_dow (0=Mon, primary location scenario model_trans_dc), et_day (Eastern date ordinal), count,
  loo_congress_count, count_posts, loo_congress_posts
Usage: python t2_hourly.py [--workers 16]
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t1full_common import *  # noqa: E402,F401,F403

G = {}


def task(bg):
    ml = G["loc"][bg]
    loc_ok = lc.model_usable(ml)
    mrow = G["mm"].loc[bg]
    out = []
    ev, pt = G["ev"][bg], G["pt"][bg]
    for acct, d in ev.groupby("screen_name"):
        if len(d) < MIN_N:
            continue
        h_ev = d.t.values // 3600
        h_ps = pt.t.values[pt.screen_name.values == acct] // 3600
        lo, hi = int(h_ev.min()), int(h_ev.max())
        hrs = np.arange(lo, hi + 1)
        keep = G["obs"][hrs - G["hour0"]]
        hrs = hrs[keep]
        own = np.bincount(h_ev - G["hour0"], minlength=G["nh"])[hrs - G["hour0"]]
        own_p = np.bincount(h_ps - G["hour0"], minlength=G["nh"])[hrs - G["hour0"]]
        tot = G["tot_ev"][hrs - G["hour0"]]
        tot_p = G["tot_ps"][hrs - G["hour0"]]
        ts = hrs * 3600
        if loc_ok:
            cls = ml.classes(ts)
            lh, ldow, _ = ml.local_hour_dow(ts, (lc.PRIMARY,), cls)[lc.PRIMARY]
        else:
            lh, ldow, _, _, _ = lc.to_local(ts, lc.ET)
        et_day, _, _, _ = hy.et_fields(hrs)
        out.append(pd.DataFrame(dict(
            account=acct, bioguide=bg, account_type=d.account_type.iloc[0], party=mrow["party"],
            chamber=mrow["chamber"], hour_idx=hrs.astype("int32"), local_hour=lh.astype("int8"),
            local_dow=ldow.astype("int8"), et_day=et_day.astype("int32"), count=own.astype("int16"),
            loo_congress_count=hy.loo_counts(tot, own).astype("int32"), count_posts=own_p.astype("int16"),
            loo_congress_posts=hy.loo_counts(tot_p, own_p).astype("int32"), loc_model_ok=loc_ok)))
    return pd.concat(out, ignore_index=True) if out else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    t0 = time.time()
    T2.mkdir(parents=True, exist_ok=True)
    m, hour0, obs, cum = load_mask()
    nh = len(m)
    amap = load_amap()
    ev = pd.read_parquet(T1 / "events.parquet", columns=["screen_name", "bioguide", "account_type", "t", "size"])
    pt = pd.read_parquet(T1 / "posts_threads.parquet", columns=["screen_name", "bioguide", "t"])
    tot_ev = np.bincount(ev.t.values // 3600 - hour0, minlength=nh).astype("int64")
    tot_ps = np.bincount(pt.t.values // 3600 - hour0, minlength=nh).astype("int64")
    assert tot_ev[~obs].sum() == 0 and tot_ps[~obs].sum() == 0
    cong = pd.DataFrame(dict(hour_idx=m.hour.values.astype("int32"),
                             hour_utc=pd.to_datetime(m.hour.values * 3600, unit="s"), observed=obs,
                             in_frame=m.in_frame.values, outage=m.outage.values, et_day=m.et_day.values,
                             et_hour=m.et_hour.values, weekday=m.weekday.values, total_events=tot_ev,
                             total_posts=tot_ps, all_posts_incl_unmatched_rt=m["count"].values))
    cong.to_parquet(T2 / "congress_hourly.parquet", index=False)
    G.update(hour0=hour0, nh=nh, obs=obs, tot_ev=tot_ev, tot_ps=tot_ps)
    G["ev"] = {b: g for b, g in ev.groupby("bioguide")}
    G["pt"] = {b: g for b, g in pt.groupby("bioguide")}
    G["loc"], G["mm"] = build_locations(amap, {b: g.t.values for b, g in G["ev"].items()})
    bgs = sorted(G["ev"], key=lambda b: -len(G["ev"][b]))
    res = []
    with Pool(a.workers) as p:
        for i, r in enumerate(p.imap_unordered(task, bgs, chunksize=2)):
            if r is not None:
                res.append(r)
            if i % 100 == 0:
                print(i, len(bgs), round(time.time() - t0), "s", flush=True)
    df = pd.concat(res, ignore_index=True)
    df.insert(5, "hour_utc", pd.to_datetime(df.hour_idx.astype("int64") * 3600, unit="s"))
    df = df.sort_values(["account", "hour_idx"]).reset_index(drop=True)
    df.to_parquet(T2 / "hourly_counts.parquet", index=False)
    meta = dict(rows=int(len(df)), accounts=int(df.account.nunique()), observed_hours_total=int(obs.sum()),
                zeros_share=float((df["count"] == 0).mean()), mean_count=float(df["count"].mean()),
                total_events_all_matched=int(tot_ev.sum()), min_events_per_account=MIN_N,
                loo_negative=int((df.loo_congress_count < 0).sum()),
                note="events = thread-collapsed non-retweet posts; hours outside the outage-masked collection frame are absent")
    # integrity: sum of own counts over ALL accounts equals total (qualifying accounts only here, so <=)
    chk = df.groupby("hour_idx")["count"].sum()
    meta["qualifying_sum_le_total"] = bool((chk.values <= tot_ev[chk.index.values - hour0]).all())
    (T2 / "hourly_counts_meta.json").write_text(json.dumps(meta, indent=2))
    print(meta, round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
