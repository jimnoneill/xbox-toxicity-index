"""Member-day location model (spec B).

For each member and Eastern calendar date D:
  dc_vote     member cast a vote (VoteView cast_code 1..8) on D
  dc_bracket  weekday strictly between two of the member's vote days at most 4 days
              apart, with no weekend day inside the span
  home        two or more days from any vote day, or a weekend day that is not a vote day
  transition  everything else (day after the last vote of a week, day before the first
              vote of a week, chamber voted on D but the member did not)

Home time zone = IANA zone of the member's state. For House members in the 12 split-zone states the zone of
the district is used (data/district_zone/district_zone.csv, built by scripts/v7/temporal_district_zones.py from
Census 2020 tract population centres, TIGER district boundaries and IANA time zone polygons; district zone = zone of
the largest population share). Senators and members whose district is not in the table keep the state zone.
All conversions use IANA zones, so daylight saving time is always correct.

Four scenarios for every location-dependent feature:
  all_dc            every post in America/New_York
  all_home          every post in the home zone
  model_trans_dc    classes dc_vote, dc_bracket, transition -> New_York; home -> home zone (primary)
  model_trans_home  classes dc_vote, dc_bracket -> New_York; transition, home -> home zone

Time inputs are integer UTC epoch seconds throughout.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ET = "America/New_York"
PRESENT_CODES = (1, 2, 3, 4, 5, 6, 7, 8)
SCENARIOS = ("all_dc", "all_home", "model_trans_dc", "model_trans_home")
PRIMARY = "model_trans_dc"

DC_VOTE, DC_BRACKET, HOME, TRANSITION = 0, 1, 2, 3
CLASS_NAMES = {0: "dc_vote", 1: "dc_bracket", 2: "home", 3: "transition"}

# State -> IANA zone. Split states use the zone of the larger share of the population.
STATE_ZONE = {
    "AL": "America/Chicago", "AK": "America/Anchorage", "AZ": "America/Phoenix",
    "AR": "America/Chicago", "CA": "America/Los_Angeles", "CO": "America/Denver",
    "CT": "America/New_York", "DE": "America/New_York", "DC": "America/New_York",
    "FL": "America/New_York", "GA": "America/New_York", "HI": "Pacific/Honolulu",
    "ID": "America/Boise", "IL": "America/Chicago", "IN": "America/Indiana/Indianapolis",
    "IA": "America/Chicago", "KS": "America/Chicago", "KY": "America/New_York",
    "LA": "America/Chicago", "ME": "America/New_York", "MD": "America/New_York",
    "MA": "America/New_York", "MI": "America/Detroit", "MN": "America/Chicago",
    "MS": "America/Chicago", "MO": "America/Chicago", "MT": "America/Denver",
    "NE": "America/Chicago", "NV": "America/Los_Angeles", "NH": "America/New_York",
    "NJ": "America/New_York", "NM": "America/Denver", "NY": "America/New_York",
    "NC": "America/New_York", "ND": "America/Chicago", "OH": "America/New_York",
    "OK": "America/Chicago", "OR": "America/Los_Angeles", "PA": "America/New_York",
    "RI": "America/New_York", "SC": "America/New_York", "SD": "America/Chicago",
    "TN": "America/Chicago", "TX": "America/Chicago", "UT": "America/Denver",
    "VT": "America/New_York", "VA": "America/New_York", "WA": "America/Los_Angeles",
    "WV": "America/New_York", "WI": "America/Chicago", "WY": "America/Denver",
    "PR": "America/Puerto_Rico", "VI": "America/St_Thomas", "GU": "Pacific/Guam",
    "AS": "Pacific/Pago_Pago", "MP": "Pacific/Saipan",
}
SPLIT_STATE_NOTE = {
    "FL": "panhandle west of the Apalachicola is Central; majority zone Eastern used",
    "TN": "east Tennessee is Eastern; majority zone Central used",
    "KY": "western Kentucky is Central; majority zone Eastern used",
    "IN": "NW and SW corners are Central; majority zone Eastern used",
    "MI": "four Upper Peninsula counties are Central; majority zone Eastern used",
    "ND": "SW North Dakota is Mountain; majority zone Central used",
    "SD": "western South Dakota is Mountain; majority zone Central used",
    "NE": "western Nebraska is Mountain; majority zone Central used",
    "KS": "four western counties are Mountain; majority zone Central used",
    "TX": "El Paso and Hudspeth counties are Mountain; majority zone Central used",
    "ID": "northern Idaho is Pacific; majority zone Mountain used",
    "OR": "Malheur County is Mountain; majority zone Pacific used",
}


# ------------------------------------------------------------------ vote loading
def load_vote_days(vdir, chambers=("S", "H"), congresses=(115, 116, 117, 118, 119)):
    """Return (member_days, chamber_days, member_chambers).

    member_days     {icpsr: sorted int64 array of epoch-day ordinals with a cast vote}
    chamber_days    {"S": array, "H": array} of days on which the chamber held any roll call
    member_chambers {icpsr: set of chamber letters in which the member cast a vote}
    """
    vdir = Path(vdir)
    mem, cham = {}, {}
    mchamb = {}
    for ch in chambers:
        parts, cdays = [], []
        for c in congresses:
            vf, rf = vdir / f"{ch}{c}_votes.csv", vdir / f"{ch}{c}_rollcalls.csv"
            if not (vf.exists() and rf.exists()):
                continue
            v = pd.read_csv(vf, usecols=["congress", "rollnumber", "icpsr", "cast_code"])
            r = pd.read_csv(rf, usecols=["congress", "rollnumber", "date"])
            r["day"] = (pd.to_datetime(r["date"]).values.astype("datetime64[D]").astype("int64"))
            cdays.append(r["day"].values)
            v = v[v.cast_code.isin(PRESENT_CODES)]
            v = v.merge(r[["congress", "rollnumber", "day"]], on=["congress", "rollnumber"])
            parts.append(v[["icpsr", "day"]].drop_duplicates())
        if not parts:
            continue
        allv = pd.concat(parts).drop_duplicates()
        cham[ch] = np.unique(np.concatenate(cdays))
        for icpsr, g in allv.groupby("icpsr"):
            mem.setdefault(int(icpsr), []).append(g["day"].values)
            mchamb.setdefault(int(icpsr), set()).add(ch)
    member_days = {k: np.unique(np.concatenate(v)) for k, v in mem.items()}
    return member_days, cham, mchamb


# ------------------------------------------------------------------ day classification
def _dow(day_ord):
    """Monday=0 for epoch-day ordinals (1970-01-01 was a Thursday)."""
    return (np.asarray(day_ord) + 3) % 7


def classify_days(days, vote_days, chamber_days):
    """Vectorised version of the validated Lee model. days, vote_days: epoch-day int arrays."""
    days = np.asarray(days, dtype="int64")
    out = np.full(days.shape, HOME, dtype="int8")
    if len(vote_days) == 0:
        wk = _dow(days) < 5
        return out  # no votes: every day is home (caller decides whether the model is usable)
    vd = np.asarray(vote_days, dtype="int64")
    i = np.searchsorted(vd, days)  # first vote day >= d
    is_vote = (i < len(vd)) & (vd[np.minimum(i, len(vd) - 1)] == days)
    prev_gap = np.where(i > 0, days - vd[np.maximum(i - 1, 0)], 999)
    # for non-vote days vd[i] > d; for vote days we do not use next_gap
    next_gap = np.where(i < len(vd), vd[np.minimum(i, len(vd) - 1)] - days, 999)
    weekday = _dow(days) < 5
    # weekend days in the span [d - prev_gap, d + next_gap]
    lo = int(days.min()) - 1005
    hi = int(days.max()) + 1005
    grid = np.arange(lo, hi + 1)
    cumw = np.concatenate([[0], np.cumsum(_dow(grid) >= 5)])

    def weekends_between(a, b):
        return cumw[b - lo + 1] - cumw[a - lo]

    ok = (prev_gap < 999) & (next_gap < 999)
    a = np.where(ok, days - prev_gap, days)
    b = np.where(ok, days + next_gap, days)
    has_we = weekends_between(a, b) > 0
    bracket = weekday & ok & (prev_gap + next_gap <= 4) & ~has_we
    mingap = np.minimum(prev_gap, next_gap)
    home = (mingap >= 2) | ~weekday
    out[:] = TRANSITION
    out[home] = HOME
    out[bracket] = DC_BRACKET
    if chamber_days is not None and len(chamber_days):
        cd = np.asarray(chamber_days, dtype="int64")
        j = np.searchsorted(cd, days)
        in_ch = (j < len(cd)) & (cd[np.minimum(j, len(cd) - 1)] == days)
        out[(out == HOME) & in_ch & (mingap <= 1)] = TRANSITION
    out[is_vote] = DC_VOTE
    return out


# ------------------------------------------------------------------ localisation
def to_local(t_sec, zone):
    """Return (hour, dow Monday=0, local epoch-day ordinal, year, month) for UTC epoch seconds."""
    idx = pd.to_datetime(np.asarray(t_sec, dtype="int64"), unit="s", utc=True).tz_convert(zone)
    local_day = idx.tz_localize(None).values.astype("datetime64[D]").astype("int64")
    return (idx.hour.values.astype("int16"), idx.dayofweek.values.astype("int16"), local_day,
            idx.year.values.astype("int32"), idx.month.values.astype("int16"))


class MemberLocation:
    """Location model for one member. Construct from vote days and home state."""

    def __init__(self, vote_days, chamber_days, state, name="", home_zone=None):
        self.vote_days = np.asarray(vote_days, dtype="int64")
        self.chamber_days = chamber_days
        self.state = state
        self.state_zone = STATE_ZONE[state]
        self.home_zone = home_zone or STATE_ZONE[state]
        self.name = name
        self.n_vote_days = len(self.vote_days)

    def classes(self, t_sec):
        """Day class per UTC instant (Eastern calendar date)."""
        _, _, et_day, _, _ = to_local(t_sec, ET)
        ud, inv = np.unique(et_day, return_inverse=True)
        cls = classify_days(ud, self.vote_days, self.chamber_days)
        return cls[inv]

    def use_dc(self, cls, scenario):
        if scenario == "all_dc":
            return np.ones(len(cls), dtype=bool)
        if scenario == "all_home":
            return np.zeros(len(cls), dtype=bool)
        if scenario == "model_trans_dc":
            return cls != HOME
        if scenario == "model_trans_home":
            return (cls == DC_VOTE) | (cls == DC_BRACKET)
        raise ValueError(scenario)

    def local_hour_dow(self, t_sec, scenarios=SCENARIOS, cls=None):
        """Dict scenario -> (hour, dow, local_day) arrays."""
        t_sec = np.asarray(t_sec, dtype="int64")
        if cls is None:
            cls = self.classes(t_sec)
        h_dc, d_dc, day_dc, _, _ = to_local(t_sec, ET)
        if self.home_zone == ET:
            h_h, d_h, day_h = h_dc, d_dc, day_dc
        else:
            h_h, d_h, day_h, _, _ = to_local(t_sec, self.home_zone)
        out = {}
        for s in scenarios:
            u = self.use_dc(cls, s)
            out[s] = (np.where(u, h_dc, h_h), np.where(u, d_dc, d_h), np.where(u, day_dc, day_h))
        return out


def model_usable(loc: MemberLocation, min_vote_days=20):
    return loc.n_vote_days >= min_vote_days


# ------------------------------------------------------------------ district-level zones for House members
SPLIT_STATES = ("FL", "TN", "KY", "IN", "MI", "ND", "SD", "NE", "KS", "TX", "ID", "OR")
# congress -> district map in the table (the 2016 Florida map and the 2012-era maps of the other states were in
# force for the 115th, 116th and 117th Congress; the 118th used the 2022 maps)
CONGRESS_MAP = {115: "cd116", 116: "cd116", 117: "cd116", 118: "cd118", 119: "cd118"}
# posting-time boundaries of each Congress (UTC epoch seconds): a seat is assigned the district of the Congress in
# which the member posted most.
CONGRESS_SPAN = {115: ("2017-01-03", "2019-01-03"), 116: ("2019-01-03", "2021-01-03"),
                 117: ("2021-01-03", "2023-01-03"), 118: ("2023-01-03", "2025-01-03")}


def canonical_zone(zone):
    """Collapse an IANA name to the four contiguous-US families by January UTC offset (used to compare zones)."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    off = ZoneInfo(zone).utcoffset(datetime(2020, 1, 15)).total_seconds() / 3600
    return {-5: "America/New_York", -6: "America/Chicago", -7: "America/Denver", -8: "America/Los_Angeles"}.get(int(off), zone)


def load_district_table(path):
    d = pd.read_csv(path)
    return {(r["map"], r["state"], int(r["district"])): (r["zone"], float(r["majority_share"]))
            for _, r in d.iterrows()}


def district_zone_for_member(icpsr, state, all_members, table, post_times=None):
    """Return (zone, detail) for a House member. all_members = voteview_all_members.csv as a DataFrame.

    Looks up the member's district in every Congress 115..118 where the member's voteview state equals `state`, takes
    the district zone from the matching map, and returns the zone of the Congress with the most posts
    (post_times = UTC epoch seconds of the member's posts; equal weights if None). Returns (None, 'no_district')
    when no district row matches or the state is not a split state."""
    if state not in SPLIT_STATES:
        return None, "not_split_state"
    r = all_members[(all_members.icpsr == icpsr) & (all_members.chamber == "House") &
                    (all_members.state_abbrev == state) & all_members.congress.isin(list(CONGRESS_SPAN))]
    zones, weights = {}, {}
    for _, row in r.iterrows():
        c = int(row["congress"])
        dist = int(row["district_code"])
        dist = 1 if dist == 0 else dist
        hit = table.get((CONGRESS_MAP[c], state, dist))
        if hit is None:
            continue
        lo, hi = (int(np.datetime64(x, "s").astype("int64")) for x in CONGRESS_SPAN[c])
        w = 1.0 if post_times is None else float(((post_times >= lo) & (post_times < hi)).sum())
        zones[c] = hit[0]
        weights[c] = w
    if not zones:
        return None, "no_district"
    tot = {}
    for c, z in zones.items():
        tot[z] = tot.get(z, 0.0) + weights[c]
    best = max(tot, key=tot.get)
    return best, ("district_stable" if len(set(zones.values())) == 1 else "district_changes_across_congresses")
