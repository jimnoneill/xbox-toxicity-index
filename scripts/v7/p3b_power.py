"""P3b step 1: known-in-advance political power features per member and origin (new file; reads only).

For origin t (features from year t, forecast of year t+1) and a window definition (political year Jul-Jun or calendar
year), three versions of the TARGET-year power status are built from the national calendar and each member's own side:
  i_honest    status on the forecast date (last day of year t) held constant for the whole target year
  ii_postelec status known at the forecast update (political year: 1 December of the target year; calendar year:
              31 December of year t): the realized calendar, except that the Georgia runoffs of 5 January 2021 are
              unresolved at that date, so the Senate majority is held at its pre-runoff value (R) for the target year
              when the knowledge date precedes 5 January 2021
  iii_oracle  realized average status over the target year
Current-year quantities (known at forecast time): realized average over year t and the end-of-year value.
Change features are target-average minus current-year average (the exposure the current-year content reflects), and
target-average minus the end-of-year value (zero under i_honest by construction).
Nothing from outcomes enters: the functions read only the calendar, party and chamber.
"""
import numpy as np
import pandas as pd


def load(root):
    cal = pd.read_parquet(root / "output/v7/t3/calendar_member_day.parquet",
                          columns=["date", "bioguide", "in_office", "party", "term_chamber", "aligned_with_president",
                                   "in_chamber_majority", "president_party", "house_majority", "senate_majority"])
    cal = cal[cal.in_office == 1].copy()
    for c in ("bioguide", "party", "term_chamber", "president_party", "house_majority", "senate_majority"):
        cal[c] = cal[c].astype(str)
    cal["date"] = pd.to_datetime(cal.date)
    nat = pd.read_csv(root / "output/v7/t3/national.csv",
                      usecols=["date", "president_party", "senate_majority", "house_majority"], parse_dates=["date"])
    return cal, nat.set_index("date")


def other(s):
    return np.where(s == "R", "D", "R")


def sides(cal):
    maj = np.where(cal.term_chamber == "house", cal.house_majority, cal.senate_majority)
    sp = np.where(cal.aligned_with_president == 1, cal.president_party, other(cal.president_party))
    sm = np.where(cal.in_chamber_majority == 1, maj, other(maj))
    d = pd.DataFrame(dict(bioguide=cal.bioguide.values, sp=sp, sm=sm))

    def mode(s):
        return s.value_counts().index[0]

    return d.groupby("bioguide").agg(side_pres=("sp", mode), side_maj=("sm", mode))


def flags(rows, pp, hm, sm_, side):
    sp = rows.bioguide.map(side.side_pres).values
    smj = rows.bioguide.map(side.side_maj).values
    maj = np.where(rows.term_chamber.values == "house", hm, sm_)
    a = (pp == sp).astype(float)
    m = (maj == smj).astype(float)
    tri = ((pp == sp) & (hm == smj) & (sm_ == smj)).astype(float)
    return a, m, tri


def window(mode, y):
    if mode == "py":
        return pd.Timestamp(f"{y}-07-01"), pd.Timestamp(f"{y + 1}-07-01")
    return pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y + 1}-01-01")


def knowledge_date(mode, target_y):
    return pd.Timestamp(f"{target_y}-12-01") if mode == "py" else pd.Timestamp(f"{target_y - 1}-12-31")


def build(root, mode, origins):
    cal, nat = load(root)
    side = sides(cal)
    out = []
    for t in origins:
        cs, ce = window(mode, t)
        ts, te = window(mode, t + 1)
        E = ts - pd.Timedelta(days=1)  # last day of year t = forecast date
        cur = cal[(cal.date >= cs) & (cal.date < ce)]
        tgt = cal[(cal.date >= ts) & (cal.date < te)]
        ca, cm, ct = flags(cur, cur.president_party.values, cur.house_majority.values, cur.senate_majority.values, side)
        cur = cur.assign(a=ca, m=cm, tri=ct)
        cg = cur.groupby("bioguide").agg(a_cur_avg=("a", "mean"), m_cur_avg=("m", "mean"), tri_cur_avg=("tri", "mean"))
        last = cur.sort_values("date").groupby("bioguide").tail(1)
        n_E = nat.loc[E]
        la, lm, lt = flags(last, np.full(len(last), n_E.president_party), np.full(len(last), n_E.house_majority),
                           np.full(len(last), n_E.senate_majority), side)
        eg = pd.DataFrame(dict(a_cur_end=la, m_cur_end=lm, tri_cur_end=lt), index=last.bioguide.values)
        n_t = len(tgt)
        ga, gm, gt = flags(tgt, tgt.president_party.values, tgt.house_majority.values, tgt.senate_majority.values, side)
        sm_ii = tgt.senate_majority.values.copy()
        if knowledge_date(mode, t + 1) < pd.Timestamp("2021-01-05"):
            sm_ii = np.where(tgt.date.values >= np.datetime64("2021-01-20"), "R", sm_ii)
        ia, im, it = flags(tgt, tgt.president_party.values, tgt.house_majority.values, sm_ii, side)
        ha, hm_, ht = flags(tgt, np.full(n_t, n_E.president_party), np.full(n_t, n_E.house_majority),
                            np.full(n_t, n_E.senate_majority), side)
        tg = tgt.assign(a_iii=ga, m_iii=gm, tri_iii=gt, a_ii=ia, m_ii=im, tri_ii=it, a_i=ha, m_i=hm_, tri_i=ht)
        cols = [f"{k}_{v}" for v in ("i", "ii", "iii") for k in ("a", "m", "tri")]
        tg = tg.groupby("bioguide")[cols].mean()
        j = cg.join(eg, how="inner").join(tg, how="inner")
        j["origin"] = t
        out.append(j.reset_index().rename(columns={"index": "bioguide"}))
    return pd.concat(out, ignore_index=True), side


POWER = ["a_next", "m_next", "tri_next", "d_a", "d_m", "d_tri", "d_a_end", "d_m_end", "d_a_x_R", "d_m_x_R",
         "a_cur_avg", "m_cur_avg"]


def variant_table(P, variant):
    d = P[["bioguide", "origin", "a_cur_avg", "m_cur_avg", "a_cur_end", "m_cur_end"]].copy()
    d["a_next"], d["m_next"], d["tri_next"] = P[f"a_{variant}"], P[f"m_{variant}"], P[f"tri_{variant}"]
    d["d_a"] = d.a_next - d.a_cur_avg
    d["d_m"] = d.m_next - d.m_cur_avg
    d["d_tri"] = d.tri_next - P.tri_cur_avg
    d["d_a_end"] = d.a_next - d.a_cur_end
    d["d_m_end"] = d.m_next - d.m_cur_end
    return d.drop(columns=["a_cur_end", "m_cur_end"])
