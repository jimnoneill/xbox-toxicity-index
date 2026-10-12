"""Shared loaders and the descriptor function for the T1/T2 temporal hygiene scripts (new file).
Reuses xbox.location and xbox.temporal_v2; thread/outage logic lives in xbox.hygiene."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from xbox import hygiene as hy  # noqa: E402
from xbox import location as lc  # noqa: E402
from xbox import temporal_v2 as tv  # noqa: E402

SEED = 20261009
T1 = ROOT / "output/v7/t1"
T2 = ROOT / "output/v7/t2"
LEE_ID = "1551266822460162051"
LEE_ICPSR = 41110
MIN_N = 200
SESSION_MIN = (15, 30, 60)


def load_raw():
    """All congresstweets posts, de-duplicated on id. Columns: id, screen_name, t (UTC epoch s), text, source."""
    tw = pd.read_parquet(ROOT / "data/congresstweets_2017_2023.parquet",
                         columns=["id", "screen_name", "created_at", "text", "source"])
    tw = tw.drop_duplicates("id")
    tw["t"] = (tw["created_at"].dt.tz_convert("UTC").dt.tz_localize(None).values.astype("datetime64[s]")
               .astype("int64"))
    return tw.drop(columns=["created_at"])


def load_amap():
    amap = pd.read_csv(ROOT / "output/v7/ti/account_member_map.csv", dtype={"user_id": str})
    return amap[amap.entity_type == "member"].copy()


def load_mask():
    m = pd.read_parquet(T1 / "outage_mask.parquet")
    hour0 = int(m.hour.iloc[0])
    obs = m.observed.values
    cum = np.concatenate([[0], np.cumsum(obs)]).astype("int64")
    return m, hour0, obs, cum


def build_locations(amap, t_by_member):
    """Per-bioguide MemberLocation (district-aware home zone), same construction as temporal_features.py."""
    member_days, chamber_days, mchamb = lc.load_vote_days(ROOT / "data/voteview")
    mm = (amap.sort_values("n_tweets", ascending=False).groupby("bioguide")
          .agg(icpsr=("icpsr", "first"), party=("party", "first"), chamber=("chamber", "first"),
               state=("state", "first")))
    allm = pd.read_csv(ROOT / "data/voteview_all_members.csv")
    dtab = lc.load_district_table(ROOT / "data/district_zone/district_zone.csv")
    loc = {}
    for bg, r in mm.iterrows():
        if bg not in t_by_member:
            continue
        icpsr = int(r["icpsr"])
        ch = sorted(mchamb.get(icpsr, {"H"}))
        cd = np.unique(np.concatenate([chamber_days[c] for c in ch if c in chamber_days]))
        vd = member_days.get(icpsr, np.array([], dtype="int64"))
        hz = None
        if r["chamber"] == "house":
            hz, _ = lc.district_zone_for_member(icpsr, r["state"], allm, dtab, t_by_member[bg])
        loc[bg] = lc.MemberLocation(vd, cd, r["state"], name=bg, home_zone=hz)
    return loc, mm


def describe(t, ml, loc_ok, valid=None, obs_cum=None, hour0=0, sizes=None, naive_days=False):
    """Model-free descriptors for one stream of event times (UTC seconds).

    valid     : bool array of length n-1 (gap lies inside observed hours) or None (all gaps valid).
    obs_cum   : cumulative observed-hour counts (leading 0) used for the exposure denominator; None = calendar span.
    sizes     : posts per event (thread size) for posts_per_session in posts; None = events only.
    naive_days: exposure = calendar days between first and last event (the earlier convention).
    Burstiness uses valid gaps only; n for A_n is (number of valid gaps + 1).
    Circadian shares use primary location scenario (model_trans_dc); NaN if the location model is not usable.
    """
    t = np.asarray(t, dtype="int64")
    n = len(t)
    res = dict(n=n)
    if n < MIN_N:
        return res
    order = np.argsort(t, kind="stable")
    assert (order == np.arange(n)).all()
    tau_all = np.diff(t).astype(float)
    ok = np.ones(n - 1, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    tau = tau_all[ok]
    res["n_valid_gaps"] = int(len(tau))
    if len(tau) >= 2 and tau.mean() > 0:
        r = tau.std(ddof=0) / tau.mean()
        res.update(r=float(r), B=float(tv.goh_barabasi(r)), A_n=float(tv.a_n(r, len(tau) + 1)),
                   zero_gap_share=float((tau == 0).mean()), rapid_share_300=float((tau <= 300).mean()))
    if loc_ok:
        cls = ml.classes(t)
        h, dow, day = ml.local_hour_dow(t, (lc.PRIMARY,), cls)[lc.PRIMARY]
        res["entropy"] = tv.entropy_norm(h)
        res["night_00_06"] = tv.night_share(h, "00_06")
        res["night_22_06"] = tv.night_share(h, "22_06")
        ws, wr = tv.weekend_stats(dow)
        res["weekend_share"], res["weekend_ratio"] = ws, wr
        res["active_days"] = int(len(np.unique(day)))
    else:
        for k in ("entropy", "night_00_06", "night_22_06", "weekend_share", "weekend_ratio"):
            res[k] = np.nan
        _, _, day, _, _ = lc.to_local(t, lc.ET)
        res["active_days"] = int(len(np.unique(day)))
    if naive_days or obs_cum is None:
        exposure = (t[-1] - t[0]) / 86400.0 + 1.0
    else:
        a, b = int(t[0] // 3600 - hour0), int(t[-1] // 3600 - hour0)
        exposure = float(obs_cum[b + 1] - obs_cum[a]) / 24.0
    res["exposure_days"] = exposure
    for m in SESSION_MIN:
        sid = hy.sessionize(t, ok, m * 60)
        ns = int(sid[-1]) + 1
        res[f"sessions_per_day_{m}"] = ns / exposure if exposure > 0 else np.nan
        res[f"events_per_session_{m}"] = n / ns
        if sizes is not None:
            res[f"posts_per_session_{m}"] = float(np.sum(sizes)) / ns
        if m == 30:
            res["sessions_per_activeday_30"] = ns / res["active_days"]
    return res


def load_lee():
    p = ROOT / "data/BasedMikeLee_archive_v2.parquet"
    df = pd.read_parquet(p)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["created_at"])
    df["t"] = df["created_at"].dt.tz_localize(None).values.astype("datetime64[s]").astype("int64")
    return df.sort_values(["t", "id"]).reset_index(drop=True)
