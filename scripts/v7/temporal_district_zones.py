"""District-level IANA zone table for House members in split-time-zone states (task: split-state zones).

Method (documented, reproducible):
  1. Census 2020 tract population centers (CenPop2020_Mean_TR{FIPS}.txt: population and lat/lon of the
     population centre of every tract) for the 12 split states FL TN KY IN MI ND SD NE KS TX ID OR.
  2. Congressional district boundaries: TIGER/Line cartographic files cb_2018_us_cd116_500k (used for the 115th,
     116th and 117th Congress; the 2016 Florida map and the 2012 maps of the other states were in force for all
     three) and cb_2022_us_cd118_500k (118th). Each tract centre is assigned to the district polygon that contains it.
  3. Time zone of each tract centre from the timezonefinder polygon set (IANA tz boundary data), collapsed to the four
     contiguous-US families by standard UTC offset (-5 America/New_York, -6 America/Chicago,
     -7 America/Denver, -8 America/Los_Angeles). All are DST-aware IANA zones.
  4. District zone = zone holding the largest share of the district's tract population; the share is recorded.
     The state majority zone is computed the same way (population over all tracts of the state) as a check.
Output: data/district_zone/district_zone.csv (map, state, district, zone, majority_share, pop, shares by zone)
        data/district_zone/state_zone_check.csv
Usage: PYTHONPATH=~/x_box/pyextra python temporal_district_zones.py
"""
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import shapefile  # noqa: E402  (pyshp)
from shapely.geometry import Point, shape  # noqa: E402
from shapely.strtree import STRtree  # noqa: E402
from timezonefinder import TimezoneFinder  # noqa: E402
from xbox.location import STATE_ZONE  # noqa: E402

D = ROOT / "data/district_zone"
FIPS = {"FL": "12", "TN": "47", "KY": "21", "IN": "18", "MI": "26", "ND": "38", "SD": "46", "NE": "31",
        "KS": "20", "TX": "48", "ID": "16", "OR": "41"}
CANON = {-5: "America/New_York", -6: "America/Chicago", -7: "America/Denver", -8: "America/Los_Angeles"}
_tf = TimezoneFinder()


def canon_zone(lat, lng):
    z = _tf.timezone_at(lat=lat, lng=lng)
    if z is None:
        return None
    off = ZoneInfo(z).utcoffset(datetime(2020, 1, 15)).total_seconds() / 3600
    return CANON.get(int(off))


def canon_name(z):
    off = ZoneInfo(z).utcoffset(datetime(2020, 1, 15)).total_seconds() / 3600
    return CANON.get(int(off), z)


def load_districts(zipname, field):
    ex = D / ("_x_" + zipname.stem)
    if not ex.exists():
        with zipfile.ZipFile(zipname) as z:
            z.extractall(ex)
    shp = next(ex.glob("*.shp"))
    r = shapefile.Reader(str(shp))
    keep = set(FIPS.values())
    geoms, ids = [], []
    for sr in r.iterShapeRecords():
        rec = sr.record.as_dict()
        if rec["STATEFP"] in keep:
            geoms.append(shape(sr.shape.__geo_interface__))
            ids.append((rec["STATEFP"], rec[field]))
    return geoms, ids


def main():
    tracts = []
    for st, f in FIPS.items():
        t = pd.read_csv(D / f"CenPop2020_Mean_TR{f}.txt", encoding="utf-8-sig", dtype={"STATEFP": str})
        t["state"] = st
        tracts.append(t)
    tr = pd.concat(tracts, ignore_index=True)
    tr["zone"] = [canon_zone(a, b) for a, b in zip(tr.LATITUDE, tr.LONGITUDE)]
    print("tracts", len(tr), "no zone", int(tr.zone.isna().sum()))
    # state check
    sc = []
    for st, g in tr.groupby("state"):
        p = g.groupby("zone").POPULATION.sum()
        sc.append(dict(state=st, tz_majority=p.idxmax(), share=float(p.max() / p.sum()),
                       code_STATE_ZONE=canon_name(STATE_ZONE[st]), **{f"pop_{k.split('/')[1]}": int(v) for k, v in p.items()}))
    pd.DataFrame(sc).to_csv(D / "state_zone_check.csv", index=False)
    print(pd.DataFrame(sc).to_string())
    rows = []
    for mapname, zipf, field in (("cd116", D / "cb_2018_us_cd116_500k.zip", "CD116FP"),
                                 ("cd118", D / "cb_2022_us_cd118_500k.zip", "CD118FP")):
        geoms, ids = load_districts(zipf, field)
        tree = STRtree(geoms)
        pts = [Point(lo, la) for la, lo in zip(tr.LATITUDE, tr.LONGITUDE)]
        assign = np.full(len(tr), -1)
        for i, p in enumerate(pts):
            for j in tree.query(p):
                if geoms[j].contains(p):
                    assign[i] = j
                    break
        # tract centres on a boundary or offshore: nearest district polygon
        miss = np.where(assign < 0)[0]
        for i in miss:
            assign[i] = int(tree.nearest(pts[i]))
        print(mapname, "tract centres unmatched by containment:", len(miss))
        tr["_d"] = assign
        for j, (sf, cd) in enumerate(ids):
            g = tr[(tr._d == j) & (tr.STATEFP == sf)]
            if len(g) == 0:
                continue
            p = g.groupby("zone").POPULATION.sum()
            st = [k for k, v in FIPS.items() if v == sf][0]
            dnum = int(cd)
            if dnum in (0, 98):
                dnum = 1 if dnum == 0 else dnum
            row = dict(map=mapname, state=st, district=dnum, zone=p.idxmax(),
                       majority_share=float(p.max() / p.sum()), pop=int(p.sum()))
            for k, v in p.items():
                row["share_" + k.split("/")[1]] = float(v / p.sum())
            rows.append(row)
    out = pd.DataFrame(rows).sort_values(["map", "state", "district"])
    out.to_csv(D / "district_zone.csv", index=False)
    sz = pd.DataFrame(dict(state=list(STATE_ZONE), sz=[canon_name(z) for z in STATE_ZONE.values()]))
    diff = out.merge(sz, on="state")
    diff = diff[diff.zone != diff.sz]
    print("districts whose zone differs from the code's state zone:")
    print(diff[["map", "state", "district", "zone", "sz", "majority_share"]].to_string())


if __name__ == "__main__":
    main()
