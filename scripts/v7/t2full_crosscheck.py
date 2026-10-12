"""Cross-check of the FULL-corpus detected event days (output/v7/t2full) against the curated T3 calendar
(output/v7/t3/events.csv, national.csv: SOTU, inaugurations, general election days).

For each curated event inside the corpus span: daily z (event day, +-1 day max), rank of that day among all complete days,
whether it is in the detected top 60, maximum hourly z within +-24 h. For the detected top 60: whether it lies within
1 day of a curated event, SOTU, inauguration or general election day; the unmatched days are listed with their theme words.
Outputs: output/v7/t2full/crosscheck_t3_events.csv, crosscheck_t3_top60.csv, crosscheck_summary.json
Usage: python t2full_crosscheck.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t1full_common import *  # noqa: E402,F401,F403

d = pd.read_csv(T2 / "congress_daily.csv", parse_dates=["date"])
d = d[d.complete].copy()
d["rank"] = d.z_day.rank(ascending=False, method="min").astype(int)
zh = pd.read_parquet(T2 / "congress_z_hourly.parquet")
zh["date"] = pd.to_datetime(zh.et_day, unit="D")
ev = pd.read_csv(ROOT / "output/v7/t3/events.csv", parse_dates=["date"])
nat = pd.read_csv(ROOT / "output/v7/t3/national.csv", parse_dates=["date"])
top = pd.read_csv(T2 / "event_days_top60.csv", parse_dates=["date"])
span = (d.date.min(), d.date.max())
top_set = set(top.date)
rows = []
for _, e in ev.iterrows():
    if not (span[0] <= e.date <= span[1]):
        continue
    w = d[(d.date >= e.date - pd.Timedelta(days=1)) & (d.date <= e.date + pd.Timedelta(days=1))]
    day = d[d.date == e.date]
    hz = zh[(zh.date >= e.date - pd.Timedelta(days=1)) & (zh.date <= e.date + pd.Timedelta(days=1))]
    rows.append(dict(date=e.date.date(), label=e.label, category=e.category, date_match=e.date_match,
                     z_day=float(day.z_day.iloc[0]) if len(day) else np.nan,
                     rank_day=int(day["rank"].iloc[0]) if len(day) else np.nan,
                     z_day_pm1_max=float(w.z_day.max()) if len(w) else np.nan,
                     in_top60_pm1=bool(any((e.date + pd.Timedelta(days=k)) in top_set for k in (-1, 0, 1))),
                     z_hour_max_pm1=float(hz.z_hour.max()) if len(hz) else np.nan))
ce = pd.DataFrame(rows)
ce.to_csv(T2 / "crosscheck_t3_events.csv", index=False)
anchors = set(ev.date)
for col in ("sotu", "inauguration", "general_election_day"):
    anchors |= set(nat.loc[nat[col] == 1, "date"])
near = lambda dt: any((dt + pd.Timedelta(days=k)) in anchors for k in (-1, 0, 1))
top["near_curated_or_anchor"] = top.date.map(near)
top["near_curated_event_only"] = top.date.map(lambda dt: any((dt + pd.Timedelta(days=k)) in set(ev.date) for k in (-1, 0, 1)))
top[["rank", "date", "z_day", "ratio", "distinctive_words", "near_curated_or_anchor", "near_curated_event_only"]].to_csv(
    T2 / "crosscheck_t3_top60.csv", index=False)


def spot(dt):
    r = d[d.date == pd.Timestamp(dt)]
    hz = zh[(zh.date >= pd.Timestamp(dt) - pd.Timedelta(days=1)) & (zh.date <= pd.Timestamp(dt) + pd.Timedelta(days=1))]
    return dict(date=dt, events=int(r.events.iloc[0]), expected=round(float(r.expected.iloc[0])),
                ratio=round(float(r.ratio.iloc[0]), 2), z_day=round(float(r.z_day.iloc[0]), 2),
                rank_among_complete_days=int(r["rank"].iloc[0]), in_top60=pd.Timestamp(dt) in top_set,
                z_hour_max_pm1=round(float(hz.z_hour.max()), 2))


old_top = pd.read_csv(T2OLD / "event_days_top60.csv", parse_dates=["date"])
both = set(old_top.date) & top_set
summ = dict(
    curated_events_in_span=int(len(ce)), curated_with_z_day_pm1_gt_3=int((ce.z_day_pm1_max > 3).sum()),
    curated_in_top60_pm1=int(ce.in_top60_pm1.sum()),
    curated_median_rank_of_day=float(ce.rank_day.median()),
    top60_near_curated_or_anchor=int(top.near_curated_or_anchor.sum()),
    top60_near_curated_event_only=int(top.near_curated_event_only.sum()),
    top60_unmatched_dates=[str(x.date()) for x in top.loc[~top.near_curated_or_anchor, "date"]],
    january_6_2021=spot("2021-01-06"), charlottesville_2017_08_12=spot("2017-08-12"),
    complete_days=int(len(d)), days_z_gt_3=int((d.z_day > 3).sum()),
    overlap_with_1in3_top60_days=int(len(both)),
    best_matched_curated=ce.sort_values("z_day_pm1_max", ascending=False).head(12)[
        ["date", "label", "z_day_pm1_max"]].assign(date=lambda x: x.date.astype(str)).to_dict("records"),
    lowest_curated=ce.sort_values("z_day_pm1_max").head(10)[["date", "label", "z_day_pm1_max"]].assign(
        date=lambda x: x.date.astype(str)).to_dict("records"))
(T2 / "crosscheck_summary.json").write_text(json.dumps(summ, indent=1, default=str))
print(json.dumps({k: v for k, v in summ.items() if k not in ("best_matched_curated", "lowest_curated")}, indent=1,
                 default=str))
