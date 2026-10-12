"""T3: cross-check curated events/national calendar against T2 event-day detection (output/v7/t2/event_days_top60.csv).
Output t3/crosscheck_t2.csv: each T2 day with nearest curated event (days), SOTU/election flags."""
import os, pandas as pd
R = os.path.expanduser("~/x_box/pipeline/output/v7")
t2 = pd.read_csv(R + "/t2/event_days_top60.csv", parse_dates=["date"])
ev = pd.read_csv(R + "/t3/events.csv", parse_dates=["date"])
nat = pd.read_csv(R + "/t3/national.csv", parse_dates=["date"]).set_index("date")
rows = []
for r in t2.itertuples():
    d = (ev.date - r.date).dt.days
    j = d.abs().idxmin()
    n = nat.loc[r.date] if r.date in nat.index else None
    rows.append(dict(rank=r.rank, date=r.date.date(), z_day=r.z_day, ratio=r.ratio, top_hashtags=str(r.top_hashtags)[:80],
                     nearest_event=ev.loc[j, "label"], nearest_event_date=ev.loc[j, "date"].date(), days_off=int(d[j]),
                     within_1d=int(abs(d[j]) <= 1), sotu=int(n.sotu) if n is not None else 0,
                     election_day=int(n.general_election_day) if n is not None else 0,
                     day_before_election=int(r.date + pd.Timedelta(days=1) in set(nat.index[nat.general_election_day == 1])),
                     shutdown=int(n.shutdown) if n is not None else 0))
x = pd.DataFrame(rows); x.to_csv(R + "/t3/crosscheck_t2.csv", index=False)
cov = ((x.within_1d == 1) | (x.sotu == 1) | (x.election_day == 1) | (x.day_before_election == 1)).mean()
print("T2 top days", len(x), "explained by curated event/SOTU/election (+-1d):", round(cov, 3))
print(x[(x.within_1d == 0) & (x.sotu == 0) & (x.election_day == 0) & (x.day_before_election == 0)][["rank", "date", "z_day", "top_hashtags"]].head(25).to_string())
