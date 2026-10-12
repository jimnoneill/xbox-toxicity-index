"""T3 tables 1 and 2: chamber_days.csv and member_days.csv.
Source: VoteView rollcalls/votes (data/voteview, congresses 115-119, through latest roll call), legislators terms
(unitedstates/congress-legislators, output/v7/t3/src) for in-office status.
Definitions: in_session = at least one roll call that chamber that day (a lower bound on true session: pro forma
and no-vote days are not counted). voted = member cast code 1-8 (9 = absent, 0 = not a member).
Coverage: 2017-06-01..2023-07-31 for all members with a bioguide in account_member_map; Mike Lee (icpsr 41110)
through 2026-10-06. Deterministic."""
import os, glob, yaml, pandas as pd, numpy as np
R = os.path.expanduser("~/x_box/pipeline")
OUT = R + "/output/v7/t3"; SRC = OUT + "/src"
START, END_ALL, END_LEE = pd.Timestamp("2017-06-01"), pd.Timestamp("2023-07-31"), pd.Timestamp("2026-10-06")

rc, vt = [], []
for f in sorted(glob.glob(R + "/data/voteview/[HS]1*_rollcalls.csv")):
    rc.append(pd.read_csv(f, usecols=["congress", "chamber", "rollnumber", "date"]))
    v = pd.read_csv(f.replace("rollcalls", "votes"), usecols=["congress", "chamber", "rollnumber", "icpsr", "cast_code"])
    vt.append(v)
rc = pd.concat(rc); rc["date"] = pd.to_datetime(rc["date"])
vt = pd.concat(vt).merge(rc, on=["congress", "chamber", "rollnumber"])
vt = vt[vt.cast_code.between(1, 8)]
print("rollcalls", len(rc), rc.date.min().date(), rc.date.max().date())

# chamber_days
days = pd.date_range(START, END_LEE)
cd = []
for ch in ("House", "Senate"):
    n = rc[rc.chamber == ch].groupby("date").size()
    s = pd.DataFrame({"date": days, "chamber": ch.lower()})
    s["n_votes"] = s.date.map(n).fillna(0).astype(int)
    s["in_session"] = (s.n_votes > 0).astype(int)
    cd.append(s)
cd = pd.concat(cd)
last = rc.date.max()
cd["source"] = "VoteView rollcalls H/S115-119 (voteview.com/static/data/out/rollcalls)"
cd["data_complete"] = (cd.date <= last).astype(int)  # 0 after last roll call in the files
cd.to_csv(OUT + "/chamber_days.csv", index=False)

# terms
def load(p):
    with open(p) as fh: return yaml.load(fh, Loader=yaml.CSafeLoader)
leg = load(SRC + "/legislators-current.yaml") + load(SRC + "/legislators-historical.yaml")
terms = []
for L in leg:
    b = L["id"].get("bioguide")
    for t in L["terms"]:
        terms.append((b, t["type"], pd.Timestamp(t["start"]), pd.Timestamp(t["end"]), t["state"],
                      t.get("district"), t.get("class"), t.get("party")))
terms = pd.DataFrame(terms, columns=["bioguide", "type", "start", "end", "state", "district", "sclass", "party"])
terms.to_csv(SRC + "/terms_all.csv", index=False)

m = pd.read_csv(OUT + "/../ti/account_member_map.csv")
m = m[m.bioguide.notna()].drop_duplicates("bioguide")[["bioguide", "icpsr", "name", "party", "chamber", "state"]]
m["icpsr"] = m.icpsr.astype(int)
print("members", len(m))

vdays = vt.groupby(["icpsr", "date"]).size().rename("n_votes_cast").reset_index()
sess = {ch: set(cd[(cd.chamber == ch) & (cd.in_session == 1)].date) for ch in ("house", "senate")}
out = []
for r in m.itertuples():
    end = END_LEE if r.icpsr == 41110 else END_ALL
    d = pd.date_range(START, end)
    f = pd.DataFrame({"date": d})
    t = terms[(terms.bioguide == r.bioguide)]
    f["in_office"] = 0; f["term_chamber"] = ""
    for x in t.itertuples():
        mask = (f.date >= x.start) & (f.date <= x.end)
        f.loc[mask, "in_office"] = 1
        f.loc[mask, "term_chamber"] = "senate" if x.type == "sen" else "house"
    vv = vdays[vdays.icpsr == r.icpsr].set_index("date").n_votes_cast
    f["n_votes_cast"] = f.date.map(vv).fillna(0).astype(int)
    f["voted"] = (f.n_votes_cast > 0).astype(int)
    f["chamber_in_session"] = [1 if c and dd in sess[c] else 0 for dd, c in zip(f.date, f.term_chamber)]
    vd = f.date[f.voted == 1].values
    allv = np.array(sorted(set(vdays[vdays.icpsr == r.icpsr].date)), dtype="datetime64[ns]")
    dv = f.date.values
    if len(allv):
        i = np.searchsorted(allv, dv, side="left")
        nxt = np.where(i < len(allv), allv[np.minimum(i, len(allv) - 1)], np.datetime64("NaT"))
        j = np.searchsorted(allv, dv, side="right") - 1
        prv = np.where(j >= 0, allv[np.maximum(j, 0)], np.datetime64("NaT"))
        f["days_since_own_vote"] = ((dv - prv) / np.timedelta64(1, "D"))
        f["days_to_next_own_vote"] = ((nxt - dv) / np.timedelta64(1, "D"))
        f["days_to_nearest_own_vote"] = f[["days_since_own_vote", "days_to_next_own_vote"]].min(axis=1)
    else:
        for c in ("days_since_own_vote", "days_to_next_own_vote", "days_to_nearest_own_vote"): f[c] = np.nan
    f["bioguide"] = r.bioguide; f["icpsr"] = r.icpsr
    out.append(f)
md = pd.concat(out)
md["source"] = "VoteView votes H/S115-119 + congress-legislators terms"
md = md[["date", "bioguide", "icpsr", "in_office", "term_chamber", "chamber_in_session", "n_votes_cast", "voted",
         "days_since_own_vote", "days_to_next_own_vote", "days_to_nearest_own_vote", "source"]]
md.to_csv(OUT + "/member_days.csv", index=False)
print(md.shape, md.voted.mean(), (md.in_office == 0).mean())
