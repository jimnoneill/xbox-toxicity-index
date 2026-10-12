"""Temporal features for every congresstweets member and account (spec C). Deterministic.

Unit: member (all of a member's accounts pooled) as primary; accounts for the account-type
comparison. Post sets: 'nonrt' (every post that is not a retweet, text does not start "RT @"; primary) and 'all'
(secondary). The earlier "original posts" set relied on the unreliable "@"-prefix reply rule and is retired.
Windows (UTC bounds):
full (2017-2023), calendar years 2017..2023, w12 (2022-07-25 to 2023-07-24, the calendar span of
the first 12 months of the @BasedMikeLee archive).
Minimum 200 posts of the post set in the window; below that only n is recorded.
Home zone: district-level IANA zone for House members in the 12 split-zone states (data/district_zone/
district_zone.csv, scripts/v7/temporal_district_zones.py); senators and all other members use the state zone.
Features are also computed with the state zone in every block (columns *_statezone) so the effect of the
district assignment can be reported. Profile-likelihood CIs for eta are written to
hawkes_ci_diagnostics.csv (model-conditional, not for the manuscript), not to member_features.csv.

Location: xbox.location member-day model (VoteView S/H 115-119 cast votes, IANA home zone by state),
four scenarios; primary = model_trans_dc. Hawkes: month x 48-bin hour-of-week baseline in the
primary scenario, with a constant-baseline fit and a week-level-baseline fit as sensitivities.

Account-to-member map: output/v7/ti/account_member_map.csv (built by the TI workstream).

Output: output/v7/temporal/{member_features.csv, account_features.csv, features_run.json}
Usage: python temporal_features.py [--workers 16] [--no-hawkes]
"""
import argparse
import json
import sys
import time
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from xbox import hawkes as hk  # noqa: E402
from xbox import location as lc  # noqa: E402
from xbox import temporal_v2 as tv  # noqa: E402

OUT = ROOT / "output/v7/temporal"
MIN_N = 200
PRIMARY = lc.PRIMARY


def d2s(s):
    return int(np.datetime64(s, "s").astype("int64"))


WINDOWS = {"full": (d2s("2017-01-01"), d2s("2024-01-01")),
           "w12": (d2s("2022-07-25"), d2s("2023-07-25"))}
for _y in range(2017, 2024):
    WINDOWS[f"y{_y}"] = (d2s(f"{_y}-01-01"), d2s(f"{_y + 1}-01-01"))


# ------------------------------------------------------------------ Hawkes on real data
def slot_arrays(ml, t_first, t_last, scenario, week=False, day=False):
    slot0 = (int(t_first) // 3600) * 3600
    ns = int((int(t_last) - slot0) // 3600) + 1
    ts = slot0 + 3600 * np.arange(ns, dtype="int64")
    if ml is None:
        h, dow, _, yr, mo = lc.to_local(ts, lc.ET)
    else:
        cls = ml.classes(ts)
        h, dow, _ = ml.local_hour_dow(ts, (scenario,), cls)[scenario]
        _, _, _, yr, mo = lc.to_local(ts, lc.ET)
    k = (h + 24 * (dow >= 5)).astype("int64")
    if week or day:
        m = ((ts - slot0) // ((24 if day else 168) * 3600)).astype("int64")
        nm = int(m.max()) + 1
    else:
        key = yr.astype("int64") * 12 + mo
        _, m = np.unique(key, return_inverse=True)
        m = m.astype("int64")
        nm = int(m.max()) + 1
    return slot0, m, k, nm


DAY_RE_KAPPA = 2.0


def hawkes_block(t, ml, scenario, seed, ci=False, sens=False):
    out = {}
    slot0, m, k, nm = slot_arrays(ml, t[0], t[-1], scenario)
    d = hk.make_design(t, m, k, slot0, nm, jitter_seed=seed)
    f = hk.fit(d)
    out.update(hawkes_eta=f["eta"], hawkes_inv_beta_min=f["inv_beta_min"], hawkes_n_iter=f["n_iter"],
               hawkes_n_levels=nm)
    ks = hk.time_rescaling(d, f)
    out.update(hawkes_ks_stat=ks["ks_stat"], hawkes_ks_p=ks["ks_p"])
    if ci:
        lo, hi = hk.profile_ci(d, f)
        out.update(hawkes_eta_lo=lo, hawkes_eta_hi=hi)
    if sens:
        out["hawkes_eta_const_baseline"] = hk.fit(hk.constant_baseline_design(d))["eta"]
        slot0w, mw, kw, nmw = slot_arrays(ml, t[0], t[-1], scenario, week=True)
        dw = hk.make_design(t, mw, kw, slot0w, nmw, jitter_seed=seed)
        out["hawkes_eta_week_baseline"] = hk.fit(dw)["eta"]
        slot0d, md, kd, nmd = slot_arrays(ml, t[0], t[-1], scenario, day=True)
        dd = hk.make_design(t, md, kd, slot0d, nmd, jitter_seed=seed)
        out["hawkes_eta_day_baseline"] = hk.fit(dd)["eta"]
        out["hawkes_eta_day_re_baseline"] = hk.fit(dd, prior_kappa=DAY_RE_KAPPA)["eta"]
    return out


# ------------------------------------------------------------------ one block of features
def block_features(t, ml, loc_ok, seed, hawkes=True, ci=False, sens=False, ml_state=None):
    t = np.sort(np.asarray(t, dtype="int64"))
    res = dict(n=len(t))
    if len(t) < MIN_N:
        return res
    b = tv.burstiness_stats(t)
    res.update(r=b["r"], B=b["B"], A_n=b["A_n"], zero_gap_share=b["zero_gap_share"],
               rapid_share_300=tv.rapid_share(t))
    cls = ml.classes(t)
    res["span_days"] = float((t[-1] - t[0]) / 86400)
    for i, nm in lc.CLASS_NAMES.items():
        res[f"cls_{nm}"] = float((cls == i).mean())
    hd = ml.local_hour_dow(t, lc.SCENARIOS, cls)
    for s in lc.SCENARIOS:
        h, dow, _ = hd[s]
        nan = (not loc_ok) and s.startswith("model")
        res[f"entropy_{s}"] = np.nan if nan else tv.entropy_norm(h)
        res[f"night_00_06_{s}"] = np.nan if nan else tv.night_share(h, "00_06")
        res[f"night_22_06_{s}"] = np.nan if nan else tv.night_share(h, "22_06")
        ws, wr = tv.weekend_stats(dow)
        res[f"weekend_share_{s}"] = np.nan if nan else ws
        res[f"weekend_ratio_{s}"] = np.nan if nan else wr
    if ml_state is not None and loc_ok:
        if ml_state.home_zone != ml.home_zone:
            hs = ml_state.local_hour_dow(t, (PRIMARY,), cls)[PRIMARY]
        else:
            hs = hd[PRIMARY]
        res["entropy_statezone"] = tv.entropy_norm(hs[0])
        res["night_00_06_statezone"] = tv.night_share(hs[0], "00_06")
        res["night_22_06_statezone"] = tv.night_share(hs[0], "22_06")
        res["weekend_share_statezone"] = tv.weekend_stats(hs[1])[0]
    if hawkes:
        res.update(hawkes_block(t, ml if loc_ok else None, PRIMARY, seed, ci=ci, sens=sens))
    return res


# ------------------------------------------------------------------ globals for workers
G = {}


def member_task(bg):
    mrow = G["members"][bg]
    df = G["posts"][bg]
    ml = G["loc"][bg]
    ml_state = G["loc_state"][bg]
    loc_ok = lc.model_usable(ml)
    rows = []
    t_all = df["t"].values
    is_orig = df["nonrt"].values
    cc = df["cclass"].values

    def emit(level, ident, acct_type, sub_mask, windows, hawkes_plan):
        for wname in windows:
            lo, hi = WINDOWS[wname]
            in_w = (t_all >= lo) & (t_all < hi) & sub_mask
            for ps in ("nonrt", "all"):
                sel = in_w & (is_orig if ps == "nonrt" else True)
                plan = hawkes_plan.get((wname, ps))
                seed = zlib.crc32(f"{ident}|{wname}|{ps}".encode()) & 0x7FFFFFFF
                if plan is None:
                    f = block_features(t_all[sel], ml, loc_ok, seed, hawkes=False, ml_state=ml_state)
                else:
                    f = block_features(t_all[sel], ml, loc_ok, seed, hawkes=G["hawkes"], ci=plan.get("ci", False),
                                       sens=plan.get("sens", False), ml_state=ml_state)
                row = dict(level=level, id=ident, bioguide=bg, account_type=acct_type, window=wname, postset=ps,
                           party=mrow["party"], chamber=mrow["chamber"], state=mrow["state"], loc_model_ok=loc_ok,
                           n_vote_days=ml.n_vote_days, home_zone=ml.home_zone,
                           state_zone=ml.state_zone, **f)
                if wname == "full" and f["n"] >= 1:
                    for c in ("phone", "tool", "web", "other"):
                        row[f"share_{c}"] = float((cc[sel] == c).mean())
                rows.append(row)

    ywin = [w for w in WINDOWS if w.startswith("y")]
    hplan = {("full", "nonrt"): dict(ci=True, sens=True), ("full", "all"): dict(ci=True),
             ("w12", "nonrt"): dict(ci=True)}
    for y in ywin:
        hplan[(y, "nonrt")] = dict()
    emit("member", bg, "pooled", np.ones(len(t_all), bool), list(WINDOWS), hplan)
    # accounts: full window only
    for acct, atype in df.groupby("screen_name")["account_type"].first().items():
        m = (df["screen_name"].values == acct)
        emit("account", acct, atype, m, ["full"], {("full", "nonrt"): dict(ci=False)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--no-hawkes", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    amap = pd.read_csv(ROOT / "output/v7/ti/account_member_map.csv", dtype={"user_id": str})
    amap = amap[amap.entity_type == "member"].copy()
    tw = pd.read_parquet(ROOT / "data/congresstweets_2017_2023.parquet",
                         columns=["id", "screen_name", "created_at", "text", "source"])
    n0 = len(tw)
    tw = tw.drop_duplicates("id")
    ndup = n0 - len(tw)
    tw = tw.merge(amap[["screen_name", "bioguide", "account_type"]], on="screen_name", how="inner")
    tw["t"] = (tw["created_at"].dt.tz_convert("UTC").dt.tz_localize(None).values.astype("datetime64[s]")
               .astype("int64"))
    tw["nonrt"] = tv.nonretweet_mask_congresstweets(tw["text"].values)
    n_rt = int((~tw["nonrt"]).sum())
    tw["cclass"] = tv.client_class(tw["source"].values)
    tw = tw[["screen_name", "bioguide", "account_type", "t", "nonrt", "cclass"]].sort_values("t")
    print("posts matched", len(tw), "dups dropped", ndup, flush=True)

    member_days, chamber_days, mchamb = lc.load_vote_days(ROOT / "data/voteview")
    print("vote members", len(member_days), "loaded in", round(time.time() - t0), "s", flush=True)
    mm = (amap.sort_values("n_tweets", ascending=False).groupby("bioguide")
          .agg(icpsr=("icpsr", "first"), party=("party", "first"), chamber=("chamber", "first"),
               state=("state", "first")))
    G["members"] = {bg: r for bg, r in mm.iterrows()}
    G["posts"] = {bg: g.reset_index(drop=True) for bg, g in tw.groupby("bioguide")}
    G["loc"], G["loc_state"] = {}, {}
    allm = pd.read_csv(ROOT / "data/voteview_all_members.csv")
    dtab = lc.load_district_table(ROOT / "data/district_zone/district_zone.csv")
    zrows = []
    for bg, r in mm.iterrows():
        icpsr = int(r["icpsr"])
        ch = sorted(mchamb.get(icpsr, {"H"}))
        cd = np.unique(np.concatenate([chamber_days[c] for c in ch if c in chamber_days]))
        vd = member_days.get(icpsr, np.array([], dtype="int64"))
        hz, detail = None, "senate_or_other"
        if r["chamber"] == "house":
            hz, detail = lc.district_zone_for_member(icpsr, r["state"], allm, dtab, G["posts"][bg]["t"].values)
        G["loc_state"][bg] = lc.MemberLocation(vd, cd, r["state"], name=bg)
        G["loc"][bg] = lc.MemberLocation(vd, cd, r["state"], name=bg, home_zone=hz)
        sz, dz = G["loc"][bg].state_zone, G["loc"][bg].home_zone
        zrows.append(dict(bioguide=bg, icpsr=icpsr, chamber=r["chamber"], state=r["state"], party=r["party"],
                          state_zone=sz, home_zone=dz, detail=detail,
                          changed=lc.canonical_zone(sz) != lc.canonical_zone(dz)))
    pd.DataFrame(zrows).to_csv(OUT / "member_zone_assignment.csv", index=False)
    G["hawkes"] = not a.no_hawkes
    bgs = sorted(G["posts"], key=lambda b: -len(G["posts"][b]))
    if a.limit:
        bgs = bgs[:a.limit]
    # warm numba cache before forking
    d0 = hk.make_design(np.arange(0, 20000, 37), np.zeros(10, int), np.zeros(10, int), 0, 1)
    hk.fit(d0)
    hk.time_rescaling(d0, hk.fit(d0))
    with Pool(a.workers) as p:
        results = []
        for i, r in enumerate(p.imap_unordered(member_task, bgs, chunksize=1)):
            results.extend(r)
            if i % 50 == 0:
                print(i, len(bgs), round(time.time() - t0), "s", flush=True)
    df = pd.DataFrame(results)
    cic = ["level", "id", "window", "postset", "n", "hawkes_eta", "hawkes_eta_lo", "hawkes_eta_hi", "hawkes_ks_stat",
           "hawkes_ks_p"]
    dg = df.loc[df.hawkes_eta_lo.notna(), cic].copy()
    dg["note"] = "model-conditional profile-likelihood CI; zero empirical coverage under day-level shocks; diagnostics only"
    dg.to_csv(OUT / "hawkes_ci_diagnostics.csv", index=False)
    df = df.drop(columns=["hawkes_eta_lo", "hawkes_eta_hi"])
    df[df.level == "member"].to_csv(OUT / "member_features.csv", index=False)
    df[df.level == "account"].to_csv(OUT / "account_features.csv", index=False)
    meta = dict(posts_matched=int(len(tw)), retweets_rt_prefix=n_rt, nonretweet_posts=int(len(tw) - n_rt),
                duplicate_ids_dropped=int(ndup), members=len(bgs),
                members_loc_model_ok=int(sum(lc.model_usable(G["loc"][b]) for b in bgs)),
                accounts=int(tw.screen_name.nunique()), min_n=MIN_N, elapsed_s=round(time.time() - t0),
                windows={k: [int(v[0]), int(v[1])] for k, v in WINDOWS.items()})
    (OUT / "features_run.json").write_text(json.dumps(meta, indent=2))
    print(meta)


if __name__ == "__main__":
    main()
