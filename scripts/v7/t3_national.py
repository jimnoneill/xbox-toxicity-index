"""T3 table 5: national.csv, one row per date 2017-06-01..2026-10-06.
Columns: president, president_party, senate_majority, house_majority, shutdown (0/1), shutdown_type, recess_house,
recess_senate (derived: run of >= RECESS_MIN consecutive days without any roll call in that chamber, bounded by roll-call
days on both sides; trailing days after the last roll call in the files are flagged unknown, not recess), sotu,
sotu_note, inauguration, general_election_day, source columns.
Sources and verification: constants below are checked against Wikipedia text by t3_verify_national.py (output
national_checks.csv). Majority party follows the party that organized the chamber: Senate Democratic from 2021-01-20
(Harris tie-break, 50-50), Republican from 2025-01-03; House Republican to 2019-01-03, Democratic to 2023-01-03,
Republican after."""
import os, glob, argparse
import pandas as pd, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--recess_min", type=int, default=7); a = ap.parse_args()
R = os.path.expanduser("~/x_box/pipeline"); OUT = R + "/output/v7/t3"
days = pd.date_range("2017-06-01", "2026-10-06")
nat = pd.DataFrame({"date": days})

def per(rows, col, default=""):
    nat[col] = default
    for s, e, v in rows:
        nat.loc[(nat.date >= s) & (nat.date <= e), col] = v
per([("2017-06-01", "2021-01-19", "Trump"), ("2021-01-20", "2025-01-19", "Biden"), ("2025-01-20", "2026-12-31", "Trump")], "president")
per([("2017-06-01", "2021-01-19", "R"), ("2021-01-20", "2025-01-19", "D"), ("2025-01-20", "2026-12-31", "R")], "president_party")
per([("2017-06-01", "2021-01-19", "R"), ("2021-01-20", "2025-01-02", "D"), ("2025-01-03", "2026-12-31", "R")], "senate_majority")
per([("2017-06-01", "2019-01-02", "R"), ("2019-01-03", "2023-01-02", "D"), ("2023-01-03", "2026-12-31", "R")], "house_majority")

nat["shutdown"] = 0; nat["shutdown_type"] = ""
for s, e, t in [("2018-01-20", "2018-01-22", "full"), ("2018-02-09", "2018-02-09", "brief_hours"),
                ("2018-12-22", "2019-01-25", "partial"), ("2025-10-01", "2025-11-12", "full"),
                ("2026-01-31", "2026-02-03", "partial_brief"), ("2026-02-14", "2026-04-30", "partial_dhs_only")]:
    m = (nat.date >= s) & (nat.date <= e); nat.loc[m, "shutdown"] = 1; nat.loc[m, "shutdown_type"] = t

# recess from roll calls
rc = pd.concat([pd.read_csv(f, usecols=["chamber", "date"]) for f in glob.glob(R + "/data/voteview/[HS]1*_rollcalls.csv")])
rc["date"] = pd.to_datetime(rc.date)
last = rc.date.max()
for ch, col in (("House", "recess_house"), ("Senate", "recess_senate")):
    v = np.array(sorted(set(rc[rc.chamber == ch].date)), dtype="datetime64[ns]")
    nat[col] = 0
    d = nat.date.values
    i = np.searchsorted(v, d, side="right")           # index of next vote strictly after
    j = i - 1                                          # previous vote on or before
    prev = np.where(j >= 0, v[np.maximum(j, 0)], np.datetime64("NaT"))
    nxt = np.where(i < len(v), v[np.minimum(i, len(v) - 1)], np.datetime64("NaT"))
    novote = ~np.isin(d, v)
    gap = (nxt - prev) / np.timedelta64(1, "D") - 1    # no-vote days between bounding votes
    nat[col] = (novote & (gap >= a.recess_min)).astype(int)
    nat.loc[np.isnat(nxt) & novote, col] = 0
    nat[col + "_data_end_unknown"] = (np.isnat(nxt) & novote).astype(int)

sotu = {"2018-01-30": "State of the Union", "2019-02-05": "State of the Union", "2020-02-04": "State of the Union",
        "2021-04-28": "Address to joint session (not formally SOTU)", "2022-03-01": "State of the Union",
        "2023-02-07": "State of the Union", "2024-03-07": "State of the Union",
        "2025-03-04": "Address to joint session (not formally SOTU)", "2026-02-24": "State of the Union"}
nat["sotu"] = nat.date.astype(str).isin(sotu).astype(int)
nat["sotu_note"] = nat.date.astype(str).map(sotu).fillna("")
nat["inauguration"] = nat.date.isin(pd.to_datetime(["2021-01-20", "2025-01-20"])).astype(int)
gen = ["2018-11-06", "2020-11-03", "2022-11-08", "2024-11-05", "2026-11-03"]
nat["general_election_day"] = nat.date.isin(pd.to_datetime(gen)).astype(int)
nat["source"] = ("Presidency and majorities: senate.gov/house.gov party-division histories; shutdowns: CRS/Wikipedia; "
                 "recess: derived from VoteView roll calls; SOTU and inaugurations: Wikipedia, see national_checks.csv")
nat.to_csv(OUT + "/national.csv", index=False)
print(nat.shape, nat[["shutdown", "recess_house", "recess_senate"]].sum().to_dict(), "last roll call", last.date())
