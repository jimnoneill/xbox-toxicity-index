"""FULL-corpus rerun of T1 step 4: model-free descriptors on cleaned, thread-collapsed, account-level event streams (member pooled = secondary).

Variants per stream:
  raw     every non-retweet post, no mask, calendar-span exposure (the earlier convention; reproduces
          output/v7/temporal/account_features.csv, nonrt, full window)
  masked  posts in observed hours only, gaps spanning unobserved hours dropped, observed-hours exposure
  clean   thread-collapsed events (thread start time) on top of the mask
Outputs: output/v7/t1/descriptors_account.csv, descriptors_member.csv, descriptor_comparison.csv,
descriptor_comparison.json, descriptor_old_check.json
Usage: python t1_descriptors.py [--workers 16]
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t1full_common import *  # noqa: E402,F401,F403

G = {}
METRICS = ["A_n", "B", "r", "entropy", "night_00_06", "night_22_06", "weekend_share", "rapid_share_300",
           "sessions_per_day_30", "events_per_session_30", "sessions_per_activeday_30", "sessions_per_day_15",
           "events_per_session_15", "sessions_per_day_60", "events_per_session_60", "zero_gap_share"]


def stream_rows(ident, level, meta, t_raw, t_mask, t_ev, sz_ev, ml, loc_ok):
    out = []
    # raw
    r = describe(t_raw, ml, loc_ok, None, None, G["hour0"], None, naive_days=True)
    out.append(dict(level=level, id=ident, variant="raw", **meta, **r))
    # masked
    v = hy.gap_valid(t_mask, G["cum"], G["hour0"]) if len(t_mask) > 1 else np.zeros(0, bool)
    r = describe(t_mask, ml, loc_ok, v, G["cum"], G["hour0"], None)
    out.append(dict(level=level, id=ident, variant="masked", **meta, **r))
    # clean
    v = hy.gap_valid(t_ev, G["cum"], G["hour0"]) if len(t_ev) > 1 else np.zeros(0, bool)
    r = describe(t_ev, ml, loc_ok, v, G["cum"], G["hour0"], sz_ev)
    out.append(dict(level=level, id=ident, variant="clean", **meta, **r))
    return out


def task(bg):
    ml = G["loc"][bg]
    loc_ok = lc.model_usable(ml)
    mrow = G["mm"].loc[bg]
    raw, msk, ev = G["raw"][bg], G["msk"][bg], G["ev"][bg]
    rows = []
    base = dict(bioguide=bg, party=mrow["party"], chamber=mrow["chamber"], state=mrow["state"], loc_model_ok=loc_ok)
    for acct, atype in raw.groupby("screen_name")["account_type"].first().items():
        a_raw = raw.t.values[raw.screen_name.values == acct]
        a_msk = msk.t.values[msk.screen_name.values == acct]
        e = ev[ev.screen_name.values == acct]
        rows += stream_rows(acct, "account", dict(account_type=atype, **base), a_raw, a_msk, e.t.values,
                            e["size"].values, ml, loc_ok)
    # member pooled (secondary view): streams of all accounts merged in time order
    t_raw = np.sort(raw.t.values)
    t_msk = np.sort(msk.t.values)
    e = ev.sort_values("t")
    rows += stream_rows(bg, "member", dict(account_type="pooled", **base), t_raw, t_msk, e.t.values,
                        e["size"].values, ml, loc_ok)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    t0 = time.time()
    m, hour0, obs, cum = load_mask()
    G.update(hour0=hour0, cum=cum)
    amap = load_amap()
    raw = load_raw()
    raw = raw.merge(amap[["screen_name", "bioguide", "account_type"]], on="screen_name")
    raw = raw[tv.nonretweet_mask_congresstweets(raw.text.values)][["screen_name", "bioguide", "account_type", "t"]]
    raw = raw.sort_values(["t", "screen_name"])
    pt = pd.read_parquet(T1 / "posts_threads.parquet", columns=["screen_name", "bioguide", "t"]).sort_values("t")
    ev = pd.read_parquet(T1 / "events.parquet", columns=["screen_name", "bioguide", "t", "size"]).sort_values("t")
    G["raw"] = {b: g.reset_index(drop=True) for b, g in raw.groupby("bioguide")}
    G["msk"] = {b: g.reset_index(drop=True) for b, g in pt.groupby("bioguide")}
    G["ev"] = {b: g.reset_index(drop=True) for b, g in ev.groupby("bioguide")}
    for b in G["raw"]:
        G["msk"].setdefault(b, pt.iloc[:0])
        G["ev"].setdefault(b, ev.iloc[:0])
    G["loc"], G["mm"] = build_locations(amap, {b: g.t.values for b, g in G["raw"].items()})
    bgs = sorted(G["raw"], key=lambda b: -len(G["raw"][b]))
    print("members", len(bgs), "loaded", round(time.time() - t0), "s", flush=True)
    rows = []
    with Pool(a.workers) as p:
        for i, r in enumerate(p.imap_unordered(task, bgs, chunksize=2)):
            rows.extend(r)
            if i % 100 == 0:
                print(i, len(bgs), round(time.time() - t0), "s", flush=True)
    df = pd.DataFrame(rows)
    df[df.level == "account"].to_csv(T1 / "descriptors_account.csv", index=False)
    df[df.level == "member"].to_csv(T1 / "descriptors_member.csv", index=False)
    print("written", len(df), round(time.time() - t0), "s", flush=True)
    compare(df)


def compare(df):
    out = {}
    rowsc = []
    for level in ("account", "member"):
        d = df[df.level == level]
        w = d.pivot(index="id", columns="variant", values=[c for c in METRICS + ["n", "n_valid_gaps"] if c in d.columns])
        res = {}
        for pair in (("raw", "masked"), ("masked", "clean"), ("raw", "clean")):
            ok = (w[("n", pair[0])] >= MIN_N) & (w[("n", pair[1])] >= MIN_N)
            res[f"{pair[0]}->{pair[1]}"] = dict(n_streams=int(ok.sum()))
            for c in METRICS:
                if (c, pair[0]) not in w:
                    continue
                x, y = w.loc[ok, (c, pair[0])], w.loc[ok, (c, pair[1])]
                good = x.notna() & y.notna()
                x, y = x[good], y[good]
                if len(x) < 10:
                    continue
                dlt = y - x
                row = dict(level=level, step=f"{pair[0]}->{pair[1]}", metric=c, n=int(len(x)), median_from=float(x.median()),
                           median_to=float(y.median()), median_change=float(dlt.median()),
                           mean_change=float(dlt.mean()), p10_change=float(dlt.quantile(0.1)),
                           p90_change=float(dlt.quantile(0.9)), spearman=float(x.corr(y, method="spearman")),
                           pearson=float(x.corr(y)))
                rowsc.append(row)
        sub = {}
        ok = (w[("n", "raw")] >= MIN_N) & (w[("n", "clean")] >= MIN_N)
        sub["streams_qualifying_raw_and_clean"] = int(ok.sum())
        sub["streams_qualifying_raw"] = int((w[("n", "raw")] >= MIN_N).sum())
        sub["streams_qualifying_clean"] = int((w[("n", "clean")] >= MIN_N).sum())
        sub["events_over_posts_median"] = float((w.loc[ok, ("n", "clean")] / w.loc[ok, ("n", "raw")]).median())
        sub["valid_gap_share_clean_median"] = float((w.loc[ok, ("n_valid_gaps", "clean")] / (w.loc[ok, ("n", "clean")] - 1)).median())
        out[level] = sub
    cmp_ = pd.DataFrame(rowsc)
    cmp_.to_csv(T1 / "descriptor_comparison.csv", index=False)
    (T1 / "descriptor_comparison.json").write_text(json.dumps(out, indent=2))
    # comparison with the earlier 1-in-3 sample results (output/v7/t1)
    old = pd.read_csv(T1OLD / "descriptors_account.csv")
    chk = {}
    for var in ("masked", "clean"):
        o = old[old.variant == var].set_index("id")
        n_ = df[(df.level == "account") & (df.variant == var)].set_index("id")
        j = o.join(n_, how="inner", lsuffix="_old", rsuffix="_new")
        j = j[(j.n_old >= MIN_N) & (j.n_new >= MIN_N)]
        res = dict(n_accounts=int(len(j)))
        for c in METRICS:
            if f"{c}_old" in j:
                ok = j[f"{c}_old"].notna() & j[f"{c}_new"].notna()
                if ok.sum() > 10:
                    res[c] = dict(median_old=float(j.loc[ok, f"{c}_old"].median()),
                                  median_new=float(j.loc[ok, f"{c}_new"].median()),
                                  spearman=float(j.loc[ok, f"{c}_old"].corr(j.loc[ok, f"{c}_new"], method="spearman")),
                                  median_abs_diff=float((j.loc[ok, f"{c}_old"] - j.loc[ok, f"{c}_new"]).abs().median()))
        chk[f"full_vs_1in3_{var}"] = res
    (T1 / "descriptor_old_check.json").write_text(json.dumps(chk, indent=2))
    print(json.dumps(chk["full_vs_1in3_clean"], indent=1))


if __name__ == "__main__":
    main()
