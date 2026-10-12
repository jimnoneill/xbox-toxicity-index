"""T3 table 7: calendar_member_day.parquet, one row per member x date, joining member_days, chamber_days, national,
windows (own race) and events. All flags are 0/1 integers unless named days_*. Window flags apply only to the member's
own race (on_ballot == 1). days_to_own_primary/general are signed (date minus target; negative = before) for the cycle
whose general election is the next one on or after the date, and are set for members on the ballot in that cycle;
state_primary_days_to is the same for the state's primary regardless of the member's own race.
Documented assumptions: see t3_elections.py and t3_national.py headers."""
import os
import pandas as pd, numpy as np
R = os.path.expanduser("~/x_box/pipeline"); OUT = R + "/output/v7/t3"
md = pd.read_csv(OUT + "/member_days.csv", parse_dates=["date"])
cd = pd.read_csv(OUT + "/chamber_days.csv", parse_dates=["date"])
nat = pd.read_csv(OUT + "/national.csv", parse_dates=["date"])
w = pd.read_csv(OUT + "/windows.csv", parse_dates=[c for c in pd.read_csv(OUT + "/windows.csv", nrows=1).columns if c.endswith(("_start", "_end", "_date"))])
ev = pd.read_csv(OUT + "/events.csv", parse_dates=["date"])
m = pd.read_csv(OUT + "/../ti/account_member_map.csv")
m = m[m.bioguide.notna()].drop_duplicates("bioguide")[["bioguide", "party", "chamber", "state", "name"]]

df = md.merge(m, on="bioguide", how="left")
for ch in ("house", "senate"):
    x = cd[cd.chamber == ch][["date", "n_votes", "in_session"]].rename(columns={"n_votes": f"{ch}_n_votes", "in_session": f"{ch}_in_session"})
    df = df.merge(x, on="date", how="left")
df["own_chamber_n_votes"] = np.where(df.term_chamber == "senate", df.senate_n_votes, df.house_n_votes)
df = df.merge(nat.drop(columns=["source"]), on="date", how="left")
df["own_chamber_recess"] = np.where(df.term_chamber == "senate", df.recess_senate, df.recess_house)
df["aligned_with_president"] = (df.party == df.president_party).astype(int)
maj = np.where(df.term_chamber == "senate", df.senate_majority, df.house_majority)
df["in_chamber_majority"] = (df.party == maj).astype(int)

# own-race windows
for c in ("primary_window", "runoff_window", "general_window", "post_election", "lame_duck"):
    df[c] = 0
df["on_ballot_this_cycle"] = 0
df["days_to_own_primary"] = np.nan; df["days_to_own_general"] = np.nan; df["state_primary_days_to"] = np.nan
df["cycle_year"] = np.nan
for y in sorted(w.cycle.unique()):
    wy = w[w.cycle == y].set_index("bioguide")
    for c in ("primary_window_start", "primary_window_end", "runoff_window_start", "runoff_window_end",
              "general_window_start", "general_window_end", "post_election_start", "post_election_end",
              "lame_duck_start", "lame_duck_end", "primary_date", "general_date"):
        df["_" + c] = df.bioguide.map(wy[c])
    df["_on"] = df.bioguide.map(wy.on_ballot).fillna(0)
    d = df.date
    ing = lambda s, e: (d >= df["_" + s]) & (d <= df["_" + e])
    df["primary_window"] |= ing("primary_window_start", "primary_window_end").astype(int)
    df["runoff_window"] |= ing("runoff_window_start", "runoff_window_end").astype(int)
    df["general_window"] |= ing("general_window_start", "general_window_end").astype(int)
    df["post_election"] |= ing("post_election_start", "post_election_end").astype(int)
    df["lame_duck"] |= ing("lame_duck_start", "lame_duck_end").astype(int)
    # cycle assignment: dates after previous general (Jan 1 of year y-1 + ...) simple rule = calendar year y or y-1 H2 .. use year == y
    inyear = d.dt.year == y
    df.loc[inyear, "cycle_year"] = y
    df.loc[inyear, "on_ballot_this_cycle"] = df.loc[inyear, "_on"].astype(int)
    onm = inyear & (df._on == 1)
    df.loc[onm, "days_to_own_primary"] = (d[onm] - df.loc[onm, "_primary_date"]).dt.days
    df.loc[onm, "days_to_own_general"] = (d[onm] - df.loc[onm, "_general_date"]).dt.days
    df.loc[inyear, "state_primary_days_to"] = (d[inyear] - df.loc[inyear, "_primary_date"]).dt.days
    df.drop(columns=[c for c in df.columns if c.startswith("_")], inplace=True)

# events
evd = ev.groupby("date").agg(event_n=("label", "size"), event_categories=("category", lambda s: ";".join(sorted(set(s))))).reset_index()
df = df.merge(evd, on="date", how="left")
df["event_n"] = df.event_n.fillna(0).astype(int); df["event_day"] = (df.event_n > 0).astype(int)
df["event_categories"] = df.event_categories.fillna("")
ed = np.sort(ev.date.unique())
dv = df.date.values
i = np.searchsorted(ed, dv)
nxt = np.where(i < len(ed), ed[np.minimum(i, len(ed) - 1)], np.datetime64("NaT"))
prv = np.where(i - 1 >= 0, ed[np.maximum(i - 1, 0)], np.datetime64("NaT"))
dn = np.abs((nxt - dv) / np.timedelta64(1, "D")); dp = np.abs((dv - prv) / np.timedelta64(1, "D"))
df["days_to_nearest_event"] = np.fmin(dn, dp)
df["event_window_pm1"] = (df.days_to_nearest_event <= 1).astype(int)
df["calendar_source"] = "t3_calendar.py joining member_days, chamber_days, national, windows, events"
for c in df.select_dtypes("object").columns:
    if c != "calendar_source": df[c] = df[c].astype("category")
df.drop(columns=["calendar_source"], inplace=True)
df.to_parquet(OUT + "/calendar_member_day.parquet", index=False)
print(df.shape); print(df[["voted", "primary_window", "general_window", "post_election", "lame_duck", "event_day", "own_chamber_recess"]].mean())
