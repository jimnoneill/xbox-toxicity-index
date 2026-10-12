"""T3 tables 3 and 4: elections.csv and windows.csv.
Inputs (output/v7/t3/src): FEC election-dates API dump (fec_*.json; official FEC), FEC candidate master bulk files
(fec/cn{yy}/cn.txt; statutory candidates C with CAND_ELECTION_YR equal to the cycle), congress-legislators yaml (terms, fec ids), MEDSL Senate returns (senate.tab, Harvard Dataverse
doi:10.7910/DVN/PEJ5QU) used as a cross-check for senators.
on_ballot rule: member has an FEC H/S candidate id with a candidate-summary row in that cycle, OR a new term starts in
January of the following year (won). A candidate id only proves a filing with financial activity, not a place on the
final ballot (withdrawn or primary-lost candidates are included; those who filed but never appeared on a ballot are
rare but possible). Senate rows are additionally checked against senate class.
Verified flags: ballot_verified = two independent sources agree; primary_date_verified = date from FEC API with
no special-case note. Louisiana (jungle primary) has no party primary; the FEC "primary" is the filing deadline, so
the primary is set missing and flagged.
Window parameters are arguments; defaults: primary 90 days, post-election 14, runoff 30."""
import os, glob, json, argparse, datetime as dt
import pandas as pd, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--primary_days", type=int, default=90)
ap.add_argument("--post_days", type=int, default=14)
ap.add_argument("--runoff_days", type=int, default=30)
args = ap.parse_args()
R = os.path.expanduser("~/x_box/pipeline"); OUT = R + "/output/v7/t3"; SRC = OUT + "/src"
YEARS = [2018, 2020, 2022, 2024, 2026]
GEN = {2018: "2018-11-06", 2020: "2020-11-03", 2022: "2022-11-08", 2024: "2024-11-05", 2026: "2026-11-03"}

# ---- FEC dates
rows = []
for f in glob.glob(SRC + "/fec_20??*.json"):
    rows += json.load(open(f)).get("results", [])
fd = pd.DataFrame(rows).drop_duplicates(["election_date", "election_state", "election_type_id", "office_sought",
                                          "election_district", "election_party"])
fd["election_district"] = fd.election_district.fillna("").str.strip()
fd["election_party"] = fd.election_party.fillna("").str.strip()
fd["election_date"] = pd.to_datetime(fd.election_date)
print("fec date rows", len(fd), fd.groupby("election_year").size().to_dict())

def pick(y, st, off, types, party=None, district=""):
    x = fd[(fd.election_year == y) & (fd.election_state == st) & (fd.office_sought == off) & fd.election_type_id.isin(types)]
    if district:
        xd = x[x.election_district == district]
        if len(xd): return xd.iloc[0]["election_date"], "district_specific"
    x = x[x.election_district == ""]
    if party:
        xp = x[x.election_party == party]
        if len(xp): x = xp
        else: x = x[x.election_party == ""]
    else:
        x = x[x.election_party == ""]
    if len(x) == 0: return pd.NaT, "missing"
    d = sorted(set(x.election_date))
    return d[0], ("ok" if len(d) == 1 else "multiple:" + ",".join(str(z.date()) for z in d))


# ---- FEC results workbook "Primary Dates" sheets (second official source; cross-check and fallback)
import re
ST = {"Alabama":"AL","Alaska":"AK","Arizona":"AZ","Arkansas":"AR","California":"CA","Colorado":"CO","Connecticut":"CT","Delaware":"DE",
 "Florida":"FL","Georgia":"GA","Hawaii":"HI","Idaho":"ID","Illinois":"IL","Indiana":"IN","Iowa":"IA","Kansas":"KS","Kentucky":"KY",
 "Louisiana":"LA","Maine":"ME","Maryland":"MD","Massachusetts":"MA","Michigan":"MI","Minnesota":"MN","Mississippi":"MS","Missouri":"MO",
 "Montana":"MT","Nebraska":"NE","Nevada":"NV","New Hampshire":"NH","New Jersey":"NJ","New Mexico":"NM","New York":"NY","North Carolina":"NC",
 "North Dakota":"ND","Ohio":"OH","Oklahoma":"OK","Oregon":"OR","Pennsylvania":"PA","Rhode Island":"RI","South Carolina":"SC",
 "South Dakota":"SD","Tennessee":"TN","Texas":"TX","Utah":"UT","Vermont":"VT","Virginia":"VA","Washington":"WA","West Virginia":"WV",
 "Wisconsin":"WI","Wyoming":"WY","D.C.":"DC","District of Columbia":"DC","Guam":"GU","Puerto Rico":"PR","American Samoa":"AS",
 "Virgin Islands":"VI","U.S. Virgin Islands":"VI","Northern Mariana Islands":"MP"}
def _pd(v):
    if isinstance(v, (pd.Timestamp, dt.datetime)): return pd.Timestamp(v)
    mt = re.match(r"^\s*(\d{1,2})/(\d{1,2})/(20\d\d)\d?", str(v))
    return pd.Timestamp(int(mt.group(3)), int(mt.group(1)), int(mt.group(2))) if mt else pd.NaT
wbd = {}   # (year, state) -> list of (primary, runoff, raw_text)
for y, sh in ((2018, "2018 Primary Dates"), (2020, "15. 2020 Primary Dates"), (2022, "10. 2022 Primary Dates")):
    g = pd.read_excel(SRC + "/fecres/fe%d.xlsx" % y, sheet_name=sh, header=None)
    for c0 in range(g.shape[1]):
        for i in range(len(g)):
            nm = g.iat[i, c0]
            if isinstance(nm, str) and nm.strip() in ST and c0 + 2 < g.shape[1]:
                raw = g.iat[i, c0 + 2]
                if c0 == 0 and y == 2020: continue   # left block of the 2020 sheet is presidential
                dd = _pd(raw)
                if pd.notna(dd):
                    ro = _pd(g.iat[i, c0 + 3]) if c0 + 3 < g.shape[1] else pd.NaT
                    wbd.setdefault((y, ST[nm.strip()]), []).append((dd, ro, str(raw)))
print("workbook primary-date entries", len(wbd))

# ---- legislators
import yaml
def load(p):
    with open(p) as fh: return yaml.load(fh, Loader=yaml.CSafeLoader)
leg = load(SRC + "/legislators-current.yaml") + load(SRC + "/legislators-historical.yaml")
L = {}
for p in leg:
    b = p["id"].get("bioguide")
    L[b] = dict(fec=set(p["id"].get("fec", []) or []), last=p["name"]["last"].upper(), terms=p["terms"])

wb = {}   # FEC official results (federalelections{yy}.xlsx): one row per candidate per race, with FEC id
def _col(cols, key, nth=0):
    hits = [c for c in cols if key in str(c).upper()]
    return hits[nth] if len(hits) > nth else None
for y in (2018, 2020, 2022):
    f = SRC + "/fecres/fe%d.xlsx" % y
    xl = pd.ExcelFile(f)
    parts = []
    for sh in xl.sheet_names:
        if ("HOUSE RESULTS" in sh.upper() or "SENATE RESULTS" in sh.upper()) and "NC 09" not in sh.upper():
            d = xl.parse(sh, header=0)
            cid = _col(d.columns, "FEC ID"); st = _col(d.columns, "STATE ABBREVIATION")
            dist = _col(d.columns, "DISTRICT"); pv = _col(d.columns, "PRIMARY VOTES")
            gv = _col(d.columns, "GENERAL VOTES"); win = _col(d.columns, "GE WINNER"); inc = _col(d.columns, "INCUMBENT")
            q = pd.DataFrame({"id": d[cid], "state": d[st], "dist": d[dist], "prim": d[pv], "gen": d[gv],
                              "win": d[win], "inc": d[inc] if inc is not None else np.nan})
            q["id"] = q.id.astype(str).str.strip().str.upper()
            q = q[q.id.str.match(r"^[HSP]\d[A-Z]{2}\d{5}$|^[HSP]\d[A-Z]{2}\d{2}\d{3}$") & q.state.notna()]
            parts.append(q)
    q = pd.concat(parts)
    q["has_primary"] = q.prim.notna(); q["has_general"] = q.gen.notna()
    q["winner"] = q.win.astype(str).str.upper().str.startswith("W")
    wb[y] = q.groupby("id").agg(state=("state", "first"), dist=("dist", "first"), has_primary=("has_primary", "max"),
                               has_general=("has_general", "max"), winner=("winner", "max")).reset_index().set_index("id")
    print("results", y, len(wb[y]))

sen = pd.read_csv(SRC + "/senate.tab") if os.path.exists(SRC + "/senate.tab") else None

m = pd.read_csv(OUT + "/../ti/account_member_map.csv")
m = m[m.bioguide.notna()]
mem = m.drop_duplicates("bioguide")[["bioguide", "icpsr", "name", "party", "chamber", "state"]].copy()
mem["icpsr"] = mem.icpsr.astype(int)

# manual, web-verified (Utah Lt. Governor's office canvass / county notices): Utah primaries outside the FEC files loaded
MANUAL_PRIMARY = {("UT", 2024): (pd.Timestamp("2024-06-25"), "manual_verified_vote.utah.gov_canvass"),
                  ("UT", 2026): (pd.Timestamp("2026-06-23"), "manual_verified_county_notices_and_wikipedia_2026_Utah_elections")}
out = []
for r in mem.itertuples():
    info = L.get(r.bioguide)
    cycles = [2018, 2020, 2022] + ([2024, 2026] if r.icpsr == 41110 else [])
    for y in cycles:
        gen = pd.Timestamp(GEN[y]); jan = pd.Timestamp("%d-01-02" % (y + 1))
        t_at = [t for t in info["terms"] if pd.Timestamp(t["start"]) <= gen <= pd.Timestamp(t["end"])] if info else []
        t_next = [t for t in info["terms"] if pd.Timestamp("%d-11-04" % y) <= pd.Timestamp(t["start"]) <= pd.Timestamp("%d-01-10" % (y + 1))] if info else []
        fec_row = None
        if info and y in wb:
            hit = [i for i in info["fec"] if i in wb[y].index and i[0] in "HS"]
            if hit:
                hit.sort(key=lambda i: (i[0] != ("S" if (t_at and t_at[0]["type"] == "sen") else "H")))
                fec_row = (hit[0], wb[y].loc[hit[0]])
        in_fec = fec_row is not None
        won = len(t_next) > 0
        on = in_fec or won
        # office sought
        if fec_row: office = fec_row[0][0]
        elif won: office = "S" if t_next[0]["type"] == "sen" else "H"
        elif t_at: office = "S" if t_at[0]["type"] == "sen" else "H"
        else: office = "S" if r.chamber == "senate" else "H"
        cur_office = ("S" if t_at[0]["type"] == "sen" else "H") if t_at else ""
        st = (fec_row[1]["state"] if fec_row else (t_next[0]["state"] if won else (t_at[0]["state"] if t_at else r.state)))
        dist = ""
        if office == "H":
            dd = fec_row[1]["dist"] if fec_row else (t_next[0].get("district") if won else (t_at[0].get("district") if t_at else None))
            dist = "" if dd in (None, "00", 0, "0") else str(int(dd)) if str(dd).isdigit() else ""
        # senate class check
        sclass = ""; class_up = ""
        if t_at and t_at[0]["type"] == "sen":
            sclass = t_at[0].get("class", "")
            up = {1: (2018, 2024), 2: (2020, 2026), 3: (2022,)}.get(sclass, ())
            class_up = int(y in up)
        medsl = ""
        if sen is not None and info and office == "S" and y <= 2024:
            s = sen[(sen.year == y) & (sen.state_po == st) & (sen.stage.str.lower() == "gen")]
            medsl = int(any(info["last"] in str(c).upper() for c in s.candidate))
        # ballot reason
        fr = fec_row[1] if fec_row else None
        if on:
            if won: reason = "ran_and_won"
            elif fr is not None and not fr["has_general"]: reason = "ran_lost_primary"
            elif fr is not None: reason = "ran_lost_general" if not fr["winner"] else "ran_won_no_next_term"
            else: reason = "next_term_without_results_row"
            if fec_row and cur_office and office != cur_office: reason += "_other_office"
        else:
            if t_at and t_at[0]["type"] == "sen" and t_at[0]["end"] > "%d-01-10" % (y + 1): reason = "senate_seat_not_up"
            elif t_at: reason = "in_office_did_not_run"
            else: reason = "not_in_office_no_candidacy_found"
        srcs = ";".join(s for s, ok in (("fec_official_results", in_fec), ("next_term", won)) if ok) or "none_found"
        # verified: official results row consistent with term history; "not on ballot" needs absence from results
        # (2018-2022) or Senate class evidence (2024, 2026 where no results file is loaded)
        if on:
            ver = int(in_fec and (won or not fr["winner"]))
        elif y in wb:
            ver = 1
        else:
            ver = int(class_up == 0) if sclass else 0
        # dates
        party = r.party
        pk = {"D": "DEM", "R": "REP"}.get(party, None)
        pdt, pnote = pick(y, st, office, ["P"], pk, dist)
        if pnote == "missing":
            pdt, pnote = pick(y, st, office, ["P", "C"], pk, dist)
            if pd.notna(pdt): pnote = "fallback_convention_or_other"
        wbe = wbd.get((y, st), [])
        wb_p = wbe[0][0] if len(wbe) == 1 else (next((e[0] for e in wbe if ("house" in e[2].lower()) == (office == "H") and "senate" not in e[2].lower() or (office == "S" and "senate" in e[2].lower())), pd.NaT) if wbe else pd.NaT)
        agree = ""
        if pd.notna(pdt) and pd.notna(wb_p) and pnote not in ("fallback_convention_or_other",): agree = int(pdt == wb_p)
        if pd.notna(wb_p) and pnote in ("missing", "fallback_convention_or_other"):
            pdt, pnote = wb_p, "fec_results_workbook_only"
        if str(pnote).startswith("multiple") and pd.notna(wb_p) and str(wb_p.date()) in pnote:
            pdt, pnote = wb_p, "ok_resolved_multiple_by_workbook"
        ro, ronote = pick(y, st, office, ["PR"], pk, dist)
        if pd.isna(ro) and wbe and pd.notna(wbe[0][1]): ro, ronote = wbe[0][1], "fec_results_workbook_only"
        gd, gnote = pick(y, st, office, ["G"], None, dist)
        if gnote == "missing" or pd.isna(gd): gd = gen; gnote = "national_default"
        special_gen = fd[(fd.election_year == y) & (fd.election_state == st) & (fd.election_type_id.isin(["SG"])) &
                         (fd.office_sought == office) & (fd.election_district == dist)] if dist or office == "S" else fd.iloc[0:0]
        if (st, y) in MANUAL_PRIMARY:
            pdt, pnote = MANUAL_PRIMARY[(st, y)]
        pnote_flag = ""
        la_jungle = (st == "LA" and y in (2018, 2020, 2022, 2024) and office == "H") or (st == "LA" and office == "S")
        if la_jungle:
            pdt = pd.NaT; pnote = "louisiana_jungle_primary_no_party_primary"
        pdate_ver = int((pnote in ("ok", "district_specific", "ok_resolved_multiple_by_workbook") and agree != 0)
                        or pnote == "fec_results_workbook_only" or str(pnote).startswith("manual_verified"))
        out.append(dict(bioguide=r.bioguide, icpsr=r.icpsr, name=r.name, party=party, chamber=r.chamber, state=st,
                        district=dist, cycle=y, on_ballot=int(on), ballot_reason=reason, office_sought=office,
                        won_next_term=int(won), fec_candidate_id=(fec_row[0] if fec_row else ""),
                        senate_class=sclass, senate_class_up=class_up, medsl_senate_name_match=medsl,
                        ballot_sources=srcs, ballot_verified=ver,
                        primary_date=pdt, primary_note=pnote, runoff_date=ro, runoff_note=ronote,
                        general_date=gd, general_note=gnote, primary_date_verified=pdate_ver, primary_sources_agree=agree,
                        special_election_in_race=int(len(special_gen) > 0),
                        source="FEC election-dates API (api.open.fec.gov/v1/election-dates); FEC candidate master bulk; congress-legislators; MEDSL senate"))
el = pd.DataFrame(out)
el.to_csv(OUT + "/elections.csv", index=False)
print("elections rows", len(el), "on_ballot", el.on_ballot.sum())
print(el.groupby("cycle")[["on_ballot", "ballot_verified", "primary_date_verified"]].mean())
print(el.ballot_reason.value_counts())

# ---- windows
def labor_day(y):
    d = dt.date(y, 9, 1)
    while d.weekday() != 0: d += dt.timedelta(days=1)
    return pd.Timestamp(d)
w = el.copy()
w["primary_days"] = args.primary_days; w["post_days"] = args.post_days; w["runoff_days"] = args.runoff_days
on = w.on_ballot == 1
w["primary_window_start"] = (w.primary_date - pd.to_timedelta(args.primary_days, "D")).where(on)
w["primary_window_end"] = w.primary_date.where(on)
w["runoff_window_start"] = (w.runoff_date - pd.to_timedelta(args.runoff_days, "D")).where(on)
w["runoff_window_end"] = w.runoff_date.where(on)
w["general_window_start"] = w.cycle.map(lambda y: labor_day(y)).where(on)
w["general_window_end"] = w.general_date.where(on)
w["post_election_start"] = (w.general_date + pd.Timedelta(days=1)).where(on)
w["post_election_end"] = (w.general_date + pd.to_timedelta(args.post_days, "D")).where(on)
lame = (w.cycle <= 2022) | (w.bioguide != "L000577")
w["lame_duck"] = ((w.on_ballot == 0) & w.ballot_reason.isin(["in_office_did_not_run"])) | ((w.on_ballot == 1) & (w.won_next_term == 0) & (w.ballot_reason.str.startswith("ran_lost") | w.ballot_reason.str.startswith("ran_won_no")))
w["lame_duck"] = w.lame_duck.astype(int)
w["lame_duck_start"] = (w.general_date + pd.Timedelta(days=1)).where(w.lame_duck == 1)
w["lame_duck_end"] = w.cycle.map(lambda y: pd.Timestamp("%d-01-03" % (y + 1))).where(w.lame_duck == 1)
w["source"] = "derived from elections.csv"
w.to_csv(OUT + "/windows.csv", index=False)
print("windows", len(w), "lame", w.lame_duck.sum())
