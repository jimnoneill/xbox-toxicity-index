"""Per-tweet location (timezone) model for Sen. Mike Lee (@BasedMikeLee).

Replaces the vote-distance model in jose_tz_data/process_basedmikelee_timezone_model.py.
Fixes: (1) days are Eastern calendar days, not UTC days; (2) presence comes from
Lee's own cast votes (VoteView icpsr 41110), not from any Senate roll call;
(3) weekends are never bracketed into a Washington stay; (4) days that cannot be
resolved are labelled 'transition' and reported as bounds instead of guessed.

Day classes (Eastern calendar date D):
  dc_vote     Lee cast a yea/nay/paired/present vote on D
  dc_bracket  weekday strictly between two dc_vote days at most 4 days apart
  utah        two or more days from any dc_vote day, or a weekend day not voting
  transition  everything else (day after the last vote of a week, day before
              the first vote of a week, Senate voted on D but Lee did not)

Usage:
  python lee_location_model.py --archive BasedMikeLee_full_archive.xlsx \
      --voteview-dir data/voteview --out output/lee_location
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

LEE_ICPSR = 41110
PRESENT_CODES = {1, 2, 3, 4, 5, 6, 7, 8}  # yea and nay variants (1-6), present (7-8); 9 = not voting
ET, MT = "America/New_York", "America/Denver"


def load_votes(vdir: Path):
    rows = []
    for f in sorted(vdir.glob("S1*_votes.csv")):
        c = f.stem.split("_")[0]
        votes = pd.read_csv(f)
        calls = pd.read_csv(vdir / f"{c}_rollcalls.csv", usecols=["congress", "rollnumber", "date"])
        rows.append(votes.merge(calls, on=["congress", "rollnumber"]))
    v = pd.concat(rows)
    v["date"] = pd.to_datetime(v["date"]).dt.date
    senate_days = set(v["date"])
    lee = v[v.icpsr == LEE_ICPSR]
    lee_days = set(lee.loc[lee.cast_code.isin(PRESENT_CODES), "date"])
    return senate_days, lee_days


def classify_days(days, senate_days, lee_days):
    vote_arr = np.array(sorted(pd.to_datetime(list(lee_days))))
    out = {}
    for d in days:
        ts = pd.Timestamp(d)
        if d in lee_days:
            out[d] = "dc_vote"
            continue
        i = np.searchsorted(vote_arr, ts)
        prev_gap = (ts - vote_arr[i - 1]).days if i > 0 else 999
        next_gap = (vote_arr[i] - ts).days if i < len(vote_arr) else 999
        weekday = ts.dayofweek < 5
        if weekday and prev_gap + next_gap <= 4 and not _weekend_between(ts, prev_gap, next_gap):
            out[d] = "dc_bracket"
        elif min(prev_gap, next_gap) >= 2 or not weekday:
            out[d] = "utah"
        else:
            out[d] = "transition"
        if out[d] == "utah" and d in senate_days and min(prev_gap, next_gap) <= 1:
            out[d] = "transition"
    return out


def _weekend_between(ts, prev_gap, next_gap):
    span = pd.date_range(ts - pd.Timedelta(days=prev_gap), ts + pd.Timedelta(days=next_gap))
    return bool((span.dayofweek >= 5).any())


def window_stats(local):
    h, dow = local.dt.hour, local.dt.dayofweek
    return {
        "night_00_06": round(float((h < 6).mean()), 4),
        "night_22_06": round(float(((h >= 22) | (h < 6)).mean()), 4),
        "weekend": round(float((dow >= 5).mean()), 4),
    }


def localize(utc, zones):
    """Per-row conversion; zones is a Series of IANA names aligned to utc."""
    out = pd.Series(index=utc.index, dtype="object")
    for z in zones.unique():
        m = zones == z
        out[m] = utc[m].dt.tz_convert(z).dt.tz_localize(None)
    return pd.to_datetime(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--voteview-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    tw = pd.read_excel(a.archive, dtype={"id": str})
    utc = pd.to_datetime(tw["created_at"], utc=True)
    et_date = utc.dt.tz_convert(ET).dt.date

    senate_days, lee_days = load_votes(Path(a.voteview_dir))
    cls = classify_days(sorted(set(et_date)), senate_days, lee_days)
    tw["day_class"] = et_date.map(cls)
    tw["et_date"] = et_date

    base = {"dc_vote": ET, "dc_bracket": ET, "utah": MT}
    scen = {
        "all_eastern": pd.Series(ET, index=tw.index),
        "all_mountain": pd.Series(MT, index=tw.index),
        "model_transition_eastern": tw["day_class"].map({**base, "transition": ET}),
        "model_transition_mountain": tw["day_class"].map({**base, "transition": MT}),
    }
    results = {"n_tweets": int(len(tw)), "lee_vote_days": len(lee_days),
               "day_class_tweets": tw["day_class"].value_counts().to_dict(),
               "scenarios": {}}
    for name, zones in scen.items():
        results["scenarios"][name] = window_stats(localize(utc, zones))

    tw["tz_primary"] = scen["model_transition_eastern"]
    tw["local_time_primary"] = localize(utc, tw["tz_primary"]).astype(str)
    tw[["id", "created_at", "et_date", "day_class", "tz_primary", "local_time_primary"]].to_parquet(
        out / "lee_tweet_location.parquet", index=False)
    (out / "lee_location_summary.json").write_text(json.dumps(results, indent=2, default=str))
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
