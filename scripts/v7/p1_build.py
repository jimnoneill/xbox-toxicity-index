"""P1 step 1: causal event-level feature table for the next-post provocation task.

Inputs : output/v7/t1full/events.parquet + posts_threads.parquet (thread-collapsed events, full archive, outage hours
         removed), classifier probabilities (--source subset | full), output/v7/t2full/congress_hourly.parquet,
         output/v7/t3/calendar_member_day.parquet, output/v7/t1full/scheduler_accounts.csv, location model (spec B).
Output : output/v7/p1[/dev]/features.parquet, build_meta.json
Event TI7 = mean over the event's scored posts of the mean of the seven dimension probabilities.
Target rows: event scored, previous event scored, gap to previous event and the last 24 h inside observed hours,
at least 10 scored events before, event on or after 2017-08-01.  Timing features always use the FULL event
sequence (all days), so days without probabilities (subset mode) do not distort gaps, sessions or counts.
Seed 20261010.  Usage: python p1_build.py --source subset|full
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p1_common import *  # noqa
from t1full_common import ROOT as _R, T1, T2, load_amap, build_locations, load_mask, lc, hy  # noqa

a = get_args()
OUT = outdir(a.source)
t0 = time.time()
ev = pd.read_parquet(T1 / "events.parquet", columns=["thread_id", "first_id", "screen_name", "bioguide", "account_type", "t", "size",
                                                     "tool_client", "cclass"])
pt = pd.read_parquet(T1 / "posts_threads.parquet", columns=["id", "thread_id"])
probs = pd.read_parquet(PROBS[a.source], columns=["id"] + [f"p_{d}" for d in fp.DIMS7]).drop_duplicates("id")
probs["ti7"] = probs[[f"p_{d}" for d in fp.DIMS7]].mean(axis=1).astype("float32")
pt = pt.merge(probs[["id", "ti7"]], on="id", how="left")
ti = pt.groupby("thread_id").agg(ti7=("ti7", "mean"), n_scored_posts=("ti7", "count")).reset_index()
ev = ev.merge(ti, on="thread_id", how="left")
rawtxt = pd.read_parquet(ROOT / "data/congresstweets_full/congresstweets_full.parquet", columns=["id", "text"]).drop_duplicates("id")
rawtxt["is_reply"] = rawtxt.text.fillna("").str.lstrip().str.startswith("@")
ev["is_reply"] = ev.first_id.map(rawtxt.set_index("id").is_reply).fillna(False).astype(float)
ev["is_multi"] = (ev["size"] >= 2).astype(float)
del rawtxt
print("events", len(ev), "scored", int(ev.ti7.notna().sum()), flush=True)

# accounts with >= 200 events overall (spec minimum)
cnt = ev.groupby("screen_name").size()
ev = ev[ev.screen_name.isin(cnt[cnt >= 200].index)].sort_values(["screen_name", "t", "thread_id"]).reset_index(drop=True)
accts = ev.screen_name.unique()
acct_code = {s: i for i, s in enumerate(accts)}
ev["acct"] = ev.screen_name.map(acct_code).astype("int32")

# ----- location model (primary scenario: transition days treated as DC)
amap = load_amap()
t_by_member = {b: g.t.values for b, g in ev.groupby("bioguide")}
loc, _ = build_locations(amap, t_by_member)
m_, hour0, obs, cum = load_mask()
nh = len(obs)

# ----- congress lagged features by hour
ch = pd.read_parquet(T2 / "congress_hourly.parquet")
hmin, hmax = int(ch.hour_idx.min()), int(ch.hour_idx.max())
cc = np.zeros(hmax - hmin + 1)
bad = np.ones(hmax - hmin + 1, dtype=bool)
cc[ch.hour_idx.values - hmin] = ch.total_events.values
bad[ch.hour_idx.values - hmin] = (~ch.observed.values) | ch.outage.values
CG = fp.congress_hour_features(cc, bad)

cols = {k: [] for k in ["cnt", "dt_prev", "sess_pos", "sess_el", "n1h", "n24h", "n_scored_prior", "prev_ti7",
                        "prev_scored", "last_scored", "roll10", "roll50", "sched_share", "hour", "weekend",
                        "gap_ok", "ok24", "loc_model", "prev_reply", "prev_multi", "reply_share10", "own_reply"]}
# faster: group boundaries (ev is sorted by screen_name)
bounds = np.flatnonzero(np.r_[True, ev.screen_name.values[1:] != ev.screen_name.values[:-1], True])
for b0, b1 in zip(bounds[:-1], bounds[1:]):
    g = ev.iloc[b0:b1]
    t = g.t.values
    f = fp.account_features(t, g.ti7.values, g.tool_client.values, g.is_reply.values, g.is_multi.values)
    bg = g.bioguide.iloc[0]
    ml = loc.get(bg)
    if ml is not None and lc.model_usable(ml):
        cls = ml.classes(t)
        h, dow, _ = ml.local_hour_dow(t, (lc.PRIMARY,), cls)[lc.PRIMARY]
        lm = 1
    else:
        h, dow, _, _, _ = lc.to_local(t, lc.ET)
        lm = 0
    hh = t // 3600 - hour0
    pidx = np.maximum(f["cnt"] - 1, 0)
    hp = hh[pidx]
    gap_ok = (f["cnt"] > 0) & ((cum[hh + 1] - cum[hp]) == (hh - hp + 1))
    lo = np.maximum(hh - 24, 0)
    ok24 = (cum[hh + 1] - cum[lo]) == (hh - lo + 1)
    for k in ["cnt", "dt_prev", "sess_pos", "sess_el", "n1h", "n24h", "n_scored_prior", "prev_ti7", "prev_scored",
              "last_scored", "roll10", "roll50", "sched_share", "prev_reply", "prev_multi", "reply_share10", "own_reply"]:
        cols[k].append(f[k])
    cols["hour"].append(h.astype(float))
    cols["weekend"].append((dow >= 5).astype(float))
    cols["gap_ok"].append(gap_ok)
    cols["ok24"].append(ok24)
    cols["loc_model"].append(np.full(len(t), lm))
for k, v in cols.items():
    ev[k] = np.concatenate(v)
print("features done", round(time.time() - t0), "s", flush=True)

# ----- derived columns
ev["log_dt"] = np.log1p(ev.dt_prev)
ev["sess_pos"] = np.log1p(ev.sess_pos)
ev["sess_el"] = np.log1p(ev.sess_el)
ev["n1h"] = np.log1p(ev.n1h)
ev["n24h"] = np.log1p(ev.n24h)
ang = 2 * np.pi * ev.hour / 24
ev["hsin"], ev["hcos"] = np.sin(ang), np.cos(ang)
ev["h2sin"], ev["h2cos"] = np.sin(2 * ang), np.cos(2 * ang)
ev["night"] = (ev.hour < 6).astype(float)
hidx = (ev.t.values // 3600 - hmin).astype(int)
for k, v in CG.items():
    ev[k] = v[hidx]
ev["when"] = pd.to_datetime(ev.t.values, unit="s")

# ----- regime flags known in advance (calendar of sessions' recesses, election dates, SOTU)
cal = pd.read_parquet(ROOT / "output/v7/t3/calendar_member_day.parquet",
                      columns=["date", "bioguide", "primary_window", "general_window", "post_election", "lame_duck",
                               "on_ballot_this_cycle", "aligned_with_president", "in_chamber_majority", "sotu",
                               "own_chamber_recess"])
cal["bioguide"] = cal.bioguide.astype(str)
cal = cal.rename(columns={"on_ballot_this_cycle": "on_ballot", "aligned_with_president": "aligned_pres",
                          "in_chamber_majority": "in_majority", "own_chamber_recess": "recess"})
et = pd.DatetimeIndex(ev.when).tz_localize("UTC").tz_convert("America/New_York").tz_localize(None).normalize()
ev["date"] = et.values
ev = ev.merge(cal, on=["bioguide", "date"], how="left")
reg = ["primary_window", "general_window", "post_election", "lame_duck", "on_ballot", "aligned_pres", "in_majority",
       "sotu", "recess"]
ev["regime_missing"] = ev[reg[0]].isna().astype(int)
ev[reg] = ev[reg].fillna(0).astype(float)

# ----- target rows
row = (ev.ti7.notna() & ev.prev_scored & ev.gap_ok & ev.ok24 & (ev.n_scored_prior >= MIN_PRIOR_SCORED) &
       (ev.when >= FIRST_ROW) & ev.cong_p1.notna())
meta = dict(source=a.source, events_all=int(len(ev)), events_scored=int(ev.ti7.notna().sum()),
            rows_target=int(row.sum()), accounts_with_rows=int(ev[row].screen_name.nunique()),
            drop_prev_unscored=int((ev.ti7.notna() & ~ev.prev_scored).sum()),
            drop_gap_across_outage=int((ev.ti7.notna() & ev.prev_scored & ~ev.gap_ok).sum()),
            drop_outage_24h=int((ev.ti7.notna() & ev.prev_scored & ev.gap_ok & ~ev.ok24).sum()),
            share_loc_model=float(ev.loc_model.mean()), regime_missing_share=float(ev.regime_missing.mean()),
            ti7_mean=float(ev.ti7.mean()), ti7_sd=float(ev.ti7.std()))
# account attributes (used only to stratify results)
att = amap.drop_duplicates("screen_name").set_index("screen_name")[["party", "chamber"]]
sch = pd.read_csv(T1 / "scheduler_accounts.csv").set_index("screen_name")[["flag_minute", "flag_tool", "flag_strong"]]
ev = ev.join(att, on="screen_name").join(sch, on="screen_name")
keep = (["thread_id", "acct", "screen_name", "bioguide", "account_type", "party", "chamber", "flag_minute", "flag_tool",
         "flag_strong", "t", "when", "size", "ti7", "last_scored"] + fp.PERSIST + fp.OWN + fp.TOD + fp.SCHED + fp.CONG + fp.TYPE_PREV + fp.TYPE_OWN +
        reg + ["loc_model", "regime_missing", "n_scored_prior"])
out = ev.loc[row, list(dict.fromkeys(keep))].reset_index(drop=True)
out["acct"] = out.acct.astype("int32")
for c in out.columns:
    if out[c].dtype == "float64":
        out[c] = out[c].astype("float32")
out.to_parquet(OUT / "features.parquet", index=False)
(OUT / "build_meta.json").write_text(json.dumps(meta, indent=1))
print(meta, round(time.time() - t0), "s", flush=True)
