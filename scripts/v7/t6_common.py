"""T6 shared loaders for the Lee case study (personal @BasedMikeLee vs official @SenMikeLee).
Days are Eastern calendar dates (same convention as the T3 calendar). Analysis window: 2022-07-25 to 2026-10-05 ET
(first and last archive days are partial). Weeks start Monday.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t1_common import *  # noqa: E402,F401,F403
from t1_common import ROOT, LEE_ICPSR  # noqa: E402

T6 = ROOT / "output/v7/t6"
T3 = ROOT / "output/v7/t3"
T6.mkdir(parents=True, exist_ok=True)
D0, D1 = pd.Timestamp("2022-07-25"), pd.Timestamp("2026-10-05")
PAL = {"personal": "#0072B2", "official": "#D55E00", "grey": "#7f7f7f", "green": "#009E73", "purple": "#CC79A7",
       "yellow": "#E69F00", "sky": "#56B4E9"}


def load_posts(which):
    """Return df with id, t (UTC epoch s), et_date (Timestamp, naive), type, text, conversation_id, in_reply_to_user_id."""
    if which == "personal":
        df = load_lee()
    else:
        df = pd.read_parquet(ROOT / "data/SenMikeLee_2022_2026.parquet")
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
        df = df.dropna(subset=["created_at"])
        df["t"] = df["created_at"].dt.tz_localize(None).values.astype("datetime64[s]").astype("int64")
        df = df.sort_values(["t", "id"]).reset_index(drop=True)
    et = df["created_at"].dt.tz_convert("America/New_York")
    df["et_date"] = et.dt.tz_localize(None).dt.normalize()
    return df


def daily_index():
    return pd.date_range(D0, D1, freq="D")


def regime_frame():
    """Daily regime flags for Lee on ET dates D0..D1."""
    idx = daily_index()
    r = pd.DataFrame(index=idx)
    md = pd.read_csv(T3 / "member_days.csv", parse_dates=["date"])
    md = md[md.icpsr == LEE_ICPSR].set_index("date")
    nat = pd.read_csv(T3 / "national.csv", parse_dates=["date"]).set_index("date")
    r["in_session"] = md.chamber_in_session.reindex(idx).fillna(0).astype(int)
    r["lee_voted"] = md.voted.reindex(idx).fillna(0).astype(int)
    r["recess"] = nat.recess_senate.reindex(idx).fillna(0).astype(int)
    r["shutdown"] = nat.shutdown.reindex(idx).fillna(0).astype(int)
    w = pd.read_csv(T3 / "windows.csv", parse_dates=["general_window_start", "general_window_end", "post_election_start",
                                                      "post_election_end", "primary_window_start", "primary_window_end"])
    w = w[(w.icpsr == LEE_ICPSR) & (w.cycle == 2022)].iloc[0]
    r["own_race_2022"] = ((idx >= w.general_window_start) & (idx <= w.general_window_end)).astype(int)
    r["post_2022"] = ((idx >= w.post_election_start) & (idx <= w.post_election_end)).astype(int)
    r["own_primary_2022"] = ((idx >= w.primary_window_start) & (idx <= w.primary_window_end)).astype(int)
    # national 2024 general window: Labor Day (first Monday of September) to election day, plus 14 days after
    ld = pd.Timestamp("2024-09-02")
    ed = pd.Timestamp("2024-11-05")
    r["natl_general_2024"] = ((idx >= ld) & (idx <= ed)).astype(int)
    r["post_2024"] = ((idx > ed) & (idx <= ed + pd.Timedelta(days=14))).astype(int)
    inaug = pd.Timestamp("2025-01-20")
    r["inaug_2wk"] = ((idx >= inaug) & (idx < inaug + pd.Timedelta(days=14))).astype(int)
    r["r_president"] = (idx >= inaug).astype(int)
    r["r_senate"] = (nat.senate_majority.reindex(idx) == "R").astype(int)
    ev = pd.read_csv(T3 / "events.csv", parse_dates=["date"])
    ev = ev[(ev.date >= D0) & (ev.date <= D1)]
    r["event_day"] = 0
    for d in ev.date:
        r.loc[(r.index >= d) & (r.index <= d + pd.Timedelta(days=1)), "event_day"] = 1
    r["dow"] = r.index.dayofweek
    r["tday"] = np.arange(len(r))
    return r, ev


def daily_counts(df, idx=None, mask=None):
    idx = idx if idx is not None else daily_index()
    d = df if mask is None else df[mask]
    return d.groupby("et_date").size().reindex(idx, fill_value=0)


def weekly_counts(s):
    """Weekly sums (Monday start) for complete weeks only."""
    wk = s.groupby(s.index - pd.to_timedelta(s.index.dayofweek, unit="D")).agg(["sum", "count"])
    wk = wk[wk["count"] == 7]["sum"]
    wk.index.name = "week"
    return wk


def wjson(name, obj):
    (T6 / name).write_text(json.dumps(obj, indent=1, default=float))
