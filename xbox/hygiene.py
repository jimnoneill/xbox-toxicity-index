"""Data hygiene for post streams (temporal redesign phase T1/T2).

Pure numpy/pandas functions, no I/O.

  et_fields            UTC epoch-hour -> Eastern calendar day, hour, weekday, day of month
  detect_outages       collection outage mask at date-hour resolution (see the function docstring for criteria)
  gap_valid            which consecutive-event gaps lie fully inside observed hours
  thread_markers       explicit thread markers in text ("1/", "2/5", "(3/n)", thread word, thread emoji)
  collapse_threads     thread ids from the gap rule (< 60 s) plus marker rules, within account
  scheduler_stats      share of posts at minute :00/:15/:30/:45 against the expected 4/60
  sessionize           session ids from a gap threshold (breaks also at invalid gaps)
  hourly_counts, loo_counts   per-account hourly counts and the leave-one-out Congress-wide count
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ET = "America/New_York"
THREAD_EMOJI = "\U0001F9F5"

# ------------------------------------------------------------------ time helpers


def et_fields(hours):
    """hours: integer array of UTC epoch hours. Returns (et_day ordinal, et_hour, weekday Mon=0, day_of_month)."""
    hours = np.asarray(hours, dtype="int64")
    idx = pd.to_datetime(hours * 3600, unit="s", utc=True).tz_convert(ET)
    day = idx.tz_localize(None).values.astype("datetime64[D]").astype("int64")
    return day, idx.hour.values.astype("int16"), idx.dayofweek.values.astype("int16"), idx.day.values.astype("int16")


def design_days_default(dom):
    """The congresstweets parquet contains only Eastern calendar days with day-of-month in {1,4,...,28}
    (every third day; the 31st is never present). Returns True for such days."""
    dom = np.asarray(dom)
    return (dom % 3 == 1) & (dom <= 28)


# ------------------------------------------------------------------ outage detection

def _rolling_median_days(x, window, min_periods=3):
    return pd.Series(x).rolling(window, center=True, min_periods=min_periods).median().values


def _robust_sigma(v):
    v = v[np.isfinite(v)]
    if len(v) < 20:
        return np.nan, np.nan
    med = np.median(v)
    return float(med), float(1.4826 * np.median(np.abs(v - med)))


def detect_outages(counts, hour0, design=design_days_default, windows=(1, 3, 6, 12), ratio_max=0.08,
                   k_sigma=5.0, min_expected_per_hour=25.0, level_window=11, pad_hours=1):
    """Collection-outage mask for Congress-wide hourly post counts.

    counts : integer hourly counts of unique posts, index i = UTC epoch hour hour0 + i.
    design : function of day-of-month array -> bool, days that the collection design should contain. If None,
             every Eastern calendar day inside the first/last non-empty hour is a frame day.
    Frame: Eastern days that have at least one post or are design days, between the first and last non-empty hour.
    Hours outside the frame are 'unobserved by design'; they are not outages.
    Expected count: E = L_d * w_weekday * q_{weekday,hour}, with L_d a centered rolling median (level_window frame
    days with posts) of daily totals divided by the weekday factor w (iterated 3 times), and q the median over days of
    the within-day hour share (renormalized per weekday).
    Criteria (a window is flagged when all three hold):
      1. window of w consecutive hours in the same Eastern day, w in `windows` (hours) or the whole day;
      2. observed/expected ratio <= ratio_max (0.08); ordinary holiday lows are 10 to 30 percent of expected and are
         therefore not flagged, a collection failure is near zero;
      3. log((C+.5)/(E+.5)) is more than k_sigma (5) robust sigmas below its median, with the robust sigma estimated
         per window length (1.4826 * MAD) over windows with expected count >= min_expected_per_hour * w (25 per hour).
    A frame day with zero posts is an outage. Flagged hours are padded by pad_hours on each side (within the frame).
    Returns a DataFrame with one row per hour: hour, count, in_frame, expected, outage, observed, et_day, et_hour,
    weekday, and an attribute dict in .attrs['info'].
    """
    counts = np.asarray(counts, dtype="float64")
    n = len(counts)
    hours = hour0 + np.arange(n, dtype="int64")
    day, eth, dow, dom = et_fields(hours)
    nz = np.flatnonzero(counts > 0)
    first, last = nz[0], nz[-1]
    in_range = (np.arange(n) >= first) & (np.arange(n) <= last)
    udays, inv = np.unique(day, return_inverse=True)
    nd = np.bincount(inv, weights=counts * in_range, minlength=len(udays))
    d_dom = np.zeros(len(udays), dtype=int)
    d_dow = np.zeros(len(udays), dtype=int)
    d_dom[inv] = dom
    d_dow[inv] = dow
    in_range_d = np.bincount(inv, weights=in_range.astype(float), minlength=len(udays)) > 0
    frame_d = in_range_d & ((nd > 0) | (design(d_dom) if design is not None else True))
    in_frame = frame_d[inv] & in_range
    # --- daily level and weekday factors on frame days with posts
    good = frame_d & (nd > 0)
    w = np.ones(7)
    lvl = np.full(len(udays), np.nan)
    gi = np.flatnonzero(good)
    for _ in range(3):
        x = nd[gi] / w[d_dow[gi]]
        lvl_g = _rolling_median_days(x, level_window)
        lvl[gi] = lvl_g
        ratio = nd[gi] / lvl_g
        for kk in range(7):
            m = d_dow[gi] == kk
            if m.sum() >= 5:
                w[kk] = np.nanmedian(ratio[m])
        w = w / w.mean()
    # fill level for frame days with zero posts from the nearest good days
    ser = pd.Series(lvl)
    ser[~frame_d] = np.nan
    lvl = ser.interpolate(limit_direction="both").values
    # --- hour-of-day share profile per weekday (median over days)
    q = np.zeros((7, 24))
    hh = eth.astype(int)
    share = np.where(nd[inv] > 0, counts / np.maximum(nd[inv], 1), np.nan)
    for kk in range(7):
        for h in range(24):
            m = in_frame & (dow == kk) & (hh == h) & good[inv]
            q[kk, h] = np.nanmedian(share[m]) if m.sum() >= 3 else np.nan
        q[kk] = np.nan_to_num(q[kk], nan=np.nanmean(q[kk]) if np.isfinite(np.nanmean(q[kk])) else 1 / 24)
        q[kk] = q[kk] / q[kk].sum()
    exp = lvl[inv] * w[dow] * q[dow, hh]
    exp = np.where(in_frame, exp, 0.0)
    # --- window detection
    flag = np.zeros(n, dtype=bool)
    cs_c = np.concatenate([[0.0], np.cumsum(np.where(in_frame, counts, 0.0))])
    cs_e = np.concatenate([[0.0], np.cumsum(exp)])
    cs_f = np.concatenate([[0], np.cumsum(in_frame.astype(int))])
    det = {}
    for wlen in windows:
        i0 = np.arange(0, n - wlen + 1)
        i1 = i0 + wlen - 1
        ok = (cs_f[i1 + 1] - cs_f[i0] == wlen) & (inv[i0] == inv[i1])
        C = cs_c[i1 + 1] - cs_c[i0]
        E = cs_e[i1 + 1] - cs_e[i0]
        me = min_expected_per_hour * wlen
        el = ok & (E >= me)
        lr = np.log((C + 0.5) / (E + 0.5))
        med, sig = _robust_sigma(lr[el])
        det[wlen] = dict(median_lr=med, sigma=sig, n_windows=int(el.sum()))
        if not np.isfinite(sig):
            continue
        sig = max(sig, 0.05)
        bad = el & (C <= ratio_max * E) & (lr < med - k_sigma * sig)
        for s in np.flatnonzero(bad):
            flag[s:s + wlen] = True
    # whole-day windows
    ed = np.bincount(inv, weights=exp, minlength=len(udays))
    cd_ = np.bincount(inv, weights=np.where(in_frame, counts, 0.0), minlength=len(udays))
    full_day = frame_d & (ed >= 25.0 * 24)
    lrd = np.log((cd_ + 0.5) / (ed + 0.5))
    med, sig = _robust_sigma(lrd[full_day & (nd > 0)])
    det["day"] = dict(median_lr=med, sigma=sig, n_windows=int((full_day & (nd > 0)).sum()))
    sig = max(sig, 0.05) if np.isfinite(sig) else 0.05
    bad_d = full_day & (cd_ <= ratio_max * ed) & ((lrd < med - k_sigma * sig) | (cd_ == 0))
    zero_d = frame_d & (nd == 0)
    flag |= (bad_d | zero_d)[inv] & in_frame
    # padding
    if pad_hours:
        f2 = flag.copy()
        for s in range(1, pad_hours + 1):
            f2[s:] |= flag[:-s]
            f2[:-s] |= flag[s:]
        flag = f2 & in_frame
    observed = in_frame & ~flag
    out = pd.DataFrame(dict(hour=hours, count=counts.astype("int64"), in_frame=in_frame, expected=exp, outage=flag,
                            observed=observed, et_day=day, et_hour=eth, weekday=dow))
    out.attrs["info"] = dict(detection=det, weekday_factor=w.tolist(), zero_frame_days=int(zero_d.sum()),
                             frame_days=int(frame_d.sum()), outage_days_whole=int(bad_d.sum()))
    return out


def outage_runs(df):
    """Merge consecutive outage hours into runs: DataFrame start_hour, end_hour, hours, et_start, count_in_run,
    expected_in_run."""
    o = df["outage"].values
    if not o.any():
        return pd.DataFrame(columns=["start_hour", "end_hour", "hours", "et_start", "count", "expected"])
    edges = np.flatnonzero(np.diff(np.concatenate([[0], o.astype(int), [0]])))
    rows = []
    for s, e in zip(edges[::2], edges[1::2]):
        seg = df.iloc[s:e]
        rows.append(dict(start_hour=int(seg.hour.iloc[0]), end_hour=int(seg.hour.iloc[-1]), hours=e - s,
                         et_start=str(pd.Timestamp(int(seg.hour.iloc[0]) * 3600, unit="s", tz="UTC").tz_convert(ET)),
                         count=int(seg["count"].sum()), expected=float(seg["expected"].sum())))
    return pd.DataFrame(rows)


def gap_valid(t_sec, observed_cum, hour0):
    """For sorted event times t (UTC seconds), is the gap between consecutive events entirely inside observed hours?
    observed_cum: cumulative count of observed hours with a leading 0, length n_hours + 1 (index of hour h is
    h - hour0). Valid iff every hour from the earlier event's hour through the later event's hour is observed.
    Events in unobserved hours make both adjacent gaps invalid."""
    t = np.asarray(t_sec, dtype="int64")
    if len(t) < 2:
        return np.zeros(0, dtype=bool)
    h = t // 3600 - hour0
    nh = len(observed_cum) - 1
    inside = (h >= 0) & (h < nh)
    hc = np.clip(h, 0, nh - 1)
    a, b = hc[:-1], hc[1:]
    ok = (observed_cum[b + 1] - observed_cum[a]) == (b - a + 1)
    return ok & inside[:-1] & inside[1:]


# ------------------------------------------------------------------ thread markers

_URL = r"https?://\S+"
_START = r"^[\s\(\[\.\"']*(\d{1,2})\s*/\s*(\d{1,2}|[nN])?(?![\d/])"
_END = r"(?:^|[\s\(\[])(\d{1,2})\s*/\s*(\d{1,2}|[nN])[\)\]]?[\s\W]*$"


def addressee_mismatch(text):
    """True where the post and the previous post (in the given order) both start with an @handle and the handles
    differ. Order must be the same sorted order used by collapse_threads. First element is False."""
    s = pd.Series(text).fillna("").astype(str)
    h = s.str.extract(r"^@(\w+)")[0].str.lower()
    p = h.shift(1)
    out = (h.notna() & p.notna() & (h != p)).values
    return out


def thread_markers(text):
    """Explicit thread markers. Returns DataFrame num (int, -1 none), tot (int; -1 none, 0 for 'n'), kw (bool: the
    thread emoji or the word 'thread')."""
    s = pd.Series(text).fillna("").astype(str)
    clean = s.str.replace(_URL, " ", regex=True).str.strip()
    st = clean.str.extract(_START)
    en = clean.str.extract(_END)
    num = pd.to_numeric(st[0], errors="coerce")
    tot = st[1]
    use_end = num.isna()
    num = num.where(~use_end, pd.to_numeric(en[0], errors="coerce"))
    tot = tot.where(~use_end, en[1])
    tot_n = pd.to_numeric(tot, errors="coerce").where(~tot.isin(["n", "N"]), 0)
    kw = clean.str.contains(THREAD_EMOJI, regex=False) | clean.str.contains(r"(?i)\bthread\b", regex=True)
    return pd.DataFrame(dict(num=num.fillna(-1).astype("int64").values, tot=tot_n.fillna(-1).astype("int64").values,
                             kw=kw.values.astype(bool)))


RULE_NAMES = {0: "start", 1: "gap", 2: "numbered", 3: "keyword", 4: "extra"}


def collapse_threads(group, t, num, tot, kw, gap_s=60, marker_window_s=7200, kw_window_s=300, extra_link=None,
                     block=None, forbid=None):
    """Thread ids within group (account). Inputs must be sorted by (group, t) and refer to non-retweet posts only.

    A post links to the immediately preceding post of the same group when
      rule 1 (gap):      t - t_prev < gap_s (strictly less than 60 s by default);
      rule 2 (numbered): t - t_prev < marker_window_s, the post is numbered k >= 2, the previous post is numbered
                         k-1 (same declared total, or either undeclared), or the previous post carries a thread
                         keyword/emoji and this post is numbered 2;
      rule 3 (keyword):  t - t_prev < kw_window_s and the previous post carries a thread keyword/emoji;
      extra_link:        optional boolean array (already requires same group), rule code 4.
    block: optional integer array; posts never link across different block values (e.g. observation blocks).
    forbid: optional boolean array; True forbids rules 1 and 3 for that post (used for replies to a different
    addressee than the previous post: two @-replies to different handles are not one thread).
    Returns (thread_id int64 starting at 0, rule int8 per post: 0 start of a thread, 1..4 rule that linked it).
    """
    g = np.asarray(group)
    t = np.asarray(t, dtype="int64")
    n = len(t)
    rule = np.zeros(n, dtype=np.int8)
    if n == 0:
        return np.zeros(0, dtype="int64"), rule
    same = np.zeros(n, dtype=bool)
    same[1:] = g[1:] == g[:-1]
    if block is not None:
        b = np.asarray(block)
        same[1:] &= b[1:] == b[:-1]
    dt = np.zeros(n, dtype="int64")
    dt[1:] = t[1:] - t[:-1]
    num, tot, kw = np.asarray(num), np.asarray(tot), np.asarray(kw, dtype=bool)
    pn, pt, pk = (np.concatenate([[-1], num[:-1]]), np.concatenate([[-1], tot[:-1]]),
                  np.concatenate([[False], kw[:-1]]))
    fb = np.asarray(forbid, dtype=bool) if forbid is not None else np.zeros(n, dtype=bool)
    r1 = same & (dt < gap_s) & ~fb
    tot_ok = (tot == pt) | (tot <= 0) | (pt <= 0)
    r2 = same & (dt < marker_window_s) & (num >= 2) & (((pn == num - 1) & tot_ok) | (pk & (num == 2)))
    r3 = same & (dt < kw_window_s) & pk & ~fb
    r4 = (same & np.asarray(extra_link, dtype=bool)) if extra_link is not None else np.zeros(n, dtype=bool)
    rule[r4] = 4
    rule[r3] = 3
    rule[r2] = 2
    rule[r1] = 1
    link = rule > 0
    return np.cumsum(~link).astype("int64") - 1, rule


# ------------------------------------------------------------------ scheduler signature

def scheduler_stats(t_sec, tool_flag=None):
    """Minute-of-hour signature. Share of posts at minute 0/15/30/45 vs the expected 4/60, binomial z, share at
    second 0, share from management clients (tool_flag boolean array)."""
    t = np.asarray(t_sec, dtype="int64")
    n = len(t)
    if n == 0:
        return dict(n=0)
    minute = (t // 60) % 60
    sec = t % 60
    q = float(np.isin(minute, (0, 15, 30, 45)).mean())
    p0 = 4.0 / 60.0
    z = (q - p0) / np.sqrt(p0 * (1 - p0) / n)
    out = dict(n=n, quarter_share=q, quarter_ratio=q / p0, quarter_z=float(z), min00_share=float((minute == 0).mean()),
               sec0_share=float((sec == 0).mean()))
    if tool_flag is not None:
        out["tool_share"] = float(np.mean(tool_flag))
    return out


# ------------------------------------------------------------------ sessions

def sessionize(t_sec, valid, gap_s):
    """Session id per event (sorted times). A new session starts when the gap to the previous event is >= gap_s or
    the gap is not valid (spans unobserved hours)."""
    t = np.asarray(t_sec, dtype="int64")
    if len(t) == 0:
        return np.zeros(0, dtype="int64")
    d = np.diff(t)
    new = (d >= gap_s) | ~np.asarray(valid, dtype=bool)
    return np.concatenate([[0], np.cumsum(new)]).astype("int64")


# ------------------------------------------------------------------ hourly counts

def hourly_counts(t_sec, hour0, n_hours):
    """Counts per UTC hour in [hour0, hour0 + n_hours); events outside are ignored."""
    h = np.asarray(t_sec, dtype="int64") // 3600 - hour0
    h = h[(h >= 0) & (h < n_hours)]
    return np.bincount(h, minlength=n_hours).astype("int64")


def loo_counts(total, own, observed=None):
    """Leave-one-out Congress-wide count: total minus the account's own count; NaN in unobserved hours."""
    loo = np.asarray(total, dtype="float64") - np.asarray(own, dtype="float64")
    if observed is not None:
        loo = np.where(observed, loo, np.nan)
    return loo
