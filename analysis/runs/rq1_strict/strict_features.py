"""Strict prior-date local clocks and thread-closure-aware persistence for RQ1.
Only historical row data and immutable location metadata enter predictors.
"""
import numpy as np
import pandas as pd
from xbox import location as lc

CLOSURE_SECONDS = 7200


def local_clock(t, member_vote_days, home_zone):
    """Prior Eastern calendar dates only; uncertain same-day vote is never consulted.

    On D, only a recorded vote on D-1 can imply transition/DC. Otherwise
    home, with IANA DST conversion. A same-day or next-day vote is unknown.
    """
    t = np.asarray(t, dtype=np.int64)
    et_day = lc.to_local(t, lc.ET)[2]
    votes = np.asarray(member_vote_days, dtype=np.int64)
    prev = np.searchsorted(votes, et_day, side="left") - 1
    transition = (prev >= 0) & (votes[np.maximum(prev, 0)] == et_day - 1) if len(votes) else np.zeros(len(t), bool)
    dc_h, dc_dow, _, _, _ = lc.to_local(t, lc.ET)
    if home_zone == lc.ET:
        home_h, home_dow = dc_h, dc_dow
    else:
        home_h, home_dow, _, _, _ = lc.to_local(t, home_zone)
    h = np.where(transition, dc_h, home_h).astype(np.int16)
    weekend = (np.where(transition, dc_dow, home_dow) >= 5).astype(np.int8)
    cls = np.where(transition, lc.TRANSITION, lc.HOME).astype(np.int8)
    return h, weekend, cls


def rebuild_local_columns(frame, hour, weekend):
    result = frame.copy()
    result["hour"] = np.asarray(hour, dtype=np.float32)
    angle = np.asarray(hour, dtype=np.float64) * (2*np.pi/24)
    for c, value in (("hsin", np.sin(angle)), ("hcos", np.cos(angle)),
                     ("h2sin", np.sin(2*angle)), ("h2cos", np.cos(2*angle)),
                     ("night", (np.asarray(hour) < 6).astype(float)),
                     ("weekend", np.asarray(weekend))):
        result[c] = np.asarray(value, dtype=np.float32)
    return result


def select_origin_strict(rows, origin, end, min_train=100, closure_seconds=CLOSURE_SECONDS):
    eligible = rows[rows.n_completed_prior >= 10]
    tr = eligible[(eligible.t < origin) & (eligible.t_end + closure_seconds < origin)]
    te = eligible[(eligible.t >= origin) & (eligible.t < end)]
    ok = tr.acct.value_counts()
    keep = ok.index[ok >= min_train]
    return tr[tr.acct.isin(keep)].copy(), te[te.acct.isin(keep)].copy()


def repair_lags_strict(events, closure_seconds=CLOSURE_SECONDS):
    """Complete-event score and size observable only after t_end+7200 < arrival."""
    chunks=[]
    for _, g in events.groupby("screen_name", sort=False):
        starts = g.sort_values(["t", "thread_id"], kind="stable")
        done = g.sort_values(["t_end", "thread_id"], kind="stable")
        end = done.t_end.to_numpy(dtype=np.int64) + closure_seconds
        t = starts.t.to_numpy(dtype=np.int64)
        count = np.searchsorted(end, t, side="left")
        previous_starts = np.searchsorted(t, t, side="left")
        scores = done.ti7.to_numpy(dtype=float)
        valid = np.isfinite(scores)
        n = np.r_[0, valid.cumsum()][count]
        sv = scores[valid]
        cs = np.r_[0., sv.cumsum()]
        last = np.full(len(t), np.nan)
        last[n > 0] = sv[n[n > 0] - 1]
        out = starts[["thread_id"]].copy()
        out["n_completed_prior"] = count
        out["prior_inflight"] = previous_starts - count
        out["prev_ti7"] = last
        for w in (10,50):
            lo = np.maximum(n-w,0)
            out[f"roll{w}"] = np.where(n>lo, (cs[n]-cs[lo])/np.maximum(n-lo,1), np.nan)
        multi = (done["size"].to_numpy()>1).astype(float)
        out["prev_multi"] = np.where(count>0,multi[np.maximum(count-1,0)],np.nan)
        chunks.append(out)
    return pd.concat(chunks,ignore_index=True)


def zone_for_congress(icpsr, state, chamber, congress, members, district_table):
    """Use only the contemporaneous seat's map; no future posting-count weighting."""
    home = lc.STATE_ZONE[state]
    if state not in lc.SPLIT_STATES or chamber != "house":
        return home
    match = members[(members.icpsr == icpsr) & (members.state_abbrev == state) &
                    (members.chamber == "House") & (members.congress == congress)]
    districts = set(match.district_code.dropna().astype(int))
    if len(districts) != 1:
        return home
    dist = next(iter(districts))
    zone = district_table.get((lc.CONGRESS_MAP[congress], state, 1 if dist == 0 else dist))
    return zone[0] if zone is not None else home


def clocks_for_frame(frame, amap, member_vote_days, members, district_table):
    """Zone for the seat at each arrival, then vote days strictly before ET date."""
    a = amap[amap.entity_type == "member"]
    for col in ("state", "icpsr", "chamber"):
        if (a.groupby("bioguide")[col].nunique(dropna=True)>1).any():
            raise AssertionError("ambiguous bioguide metadata: " + col)
    a = a.drop_duplicates("bioguide").set_index("bioguide")
    times = frame.t.to_numpy(dtype=np.int64)
    c = np.full(len(frame), 115, dtype=np.int16)
    for congress in (116, 117, 118):
        start = int(pd.Timestamp(f"{2017 + 2*(congress-115)}-01-03").timestamp())
        c[times >= start] = congress
    hours = np.empty(len(frame), dtype=np.int16)
    weekends = np.empty(len(frame), dtype=np.int8)
    classes = np.empty(len(frame), dtype=np.int8)
    fallback = 0
    for bg, idx in frame.groupby("bioguide", sort=False).indices.items():
        if bg not in a.index:
            raise AssertionError("member not mapped: " + str(bg))
        meta = a.loc[bg]
        if pd.isna(meta.icpsr) or pd.isna(meta.state):
            raise AssertionError("member state/icpsr absent: " + str(bg))
        days = member_vote_days.get(int(meta.icpsr), np.array([], dtype=np.int64))
        for congress in np.unique(c[idx]):
            sub = idx[c[idx] == congress]
            zone = zone_for_congress(int(meta.icpsr), str(meta.state), str(meta.chamber),
                                     int(congress), members, district_table)
            if meta.state in lc.SPLIT_STATES and meta.chamber == "house" and zone == lc.STATE_ZONE[meta.state]:
                fallback += len(sub) # may be genuine state-zone district; upper-bound fallback diagnostic
            hh, ww, cls = local_clock(times[sub], days, zone)
            hours[sub], weekends[sub], classes[sub] = hh, ww, cls
    result = rebuild_local_columns(frame, hours, weekends)
    result["strict_loc_class"] = classes
    audit = {"split_state_rows_using_state_zone": int(fallback), "dc_transition_rows": int((classes == lc.TRANSITION).sum()),
             "home_rows": int((classes == lc.HOME).sum()),
             "changed_hour": int((frame.hour.to_numpy()!=hours).sum()),
             "changed_weekend": int((frame.weekend.to_numpy()!=weekends).sum())}
    return result, audit


def causal_congress_features(hourly_event_counts):
    """Original P1 lag recipe, without full-corpus hindsight outage imputation."""
    from xbox import forecast_p1 as fp
    counts = np.asarray(hourly_event_counts, dtype=np.float64)
    return fp.congress_hour_features(counts, np.zeros(len(counts), dtype=bool))
