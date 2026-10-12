"""FULL-corpus rerun of T1 steps 1-3 (continuous daily archive, outage detection with design=None): collection-outage mask, thread collapsing (event table), scheduler signature.

Outputs (output/v7/t1/): outage_mask.parquet, outage_runs.csv, outage_summary.json, outage_plot.png,
posts_threads.parquet, events.parquet, thread_stats.json, thread_validation_sample.csv, scheduler_accounts.csv,
scheduler_summary.json
Usage: python t1_build.py
"""
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t1full_common import *  # noqa: E402,F401,F403

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

T1.mkdir(parents=True, exist_ok=True)
t0 = time.time()
rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------- 1. outages
raw = load_raw()
hmin, hmax = int(raw.t.min() // 3600), int(raw.t.max() // 3600)
hour0 = hmin
cnt = np.bincount(raw.t.values // 3600 - hour0, minlength=hmax - hour0 + 1)
mask = hy.detect_outages(cnt, hour0, design=None)
info = mask.attrs["info"]
mask.to_parquet(T1 / "outage_mask.parquet", index=False)
runs = hy.outage_runs(mask)
runs.to_csv(T1 / "outage_runs.csv", index=False)
day_tab = (mask.groupby("et_day").agg(count=("count", "sum"), expected=("expected", "sum"),
                                      out_h=("outage", "sum"), frame_h=("in_frame", "sum")).reset_index())
day_tab["date"] = pd.to_datetime(day_tab.et_day, unit="D").dt.strftime("%Y-%m-%d")
day_tab["dt"] = pd.to_datetime(day_tab.et_day, unit="D")
n_cal_days = len(day_tab)
frame_days = day_tab[day_tab.frame_h > 0]
out_days = day_tab[day_tab.out_h > 0]
summ = dict(
    span_first_utc=str(pd.Timestamp(hmin * 3600, unit="s")), span_last_utc=str(pd.Timestamp(hmax * 3600, unit="s")),
    posts_unique_ids=int(len(raw)), calendar_days_in_span=int(n_cal_days),
    calendar_days_with_posts=int((day_tab["count"] > 0).sum()),
    calendar_days_zero_posts=int((day_tab["count"] == 0).sum()),
    design_rule="none: full daily archive, every Eastern calendar day is a frame day",
    frame_days=int(len(frame_days)), frame_hours=int(mask.in_frame.sum()),
    unobserved_by_design_hours=int((~mask.in_frame).sum()),
    posts_in_non_frame_hours=int(mask.loc[~mask.in_frame, "count"].sum()),
    outage_hours=int(mask.outage.sum()), outage_runs=int(len(runs)), outage_days_touched=int(len(out_days)),
    outage_days_whole=int(info["outage_days_whole"]), zero_post_frame_days=int(info["zero_frame_days"]),
    posts_in_outage_hours=int(mask.loc[mask.outage, "count"].sum()),
    observed_hours=int(mask.observed.sum()),
    share_of_frame_hours_in_outage=float(mask.outage.sum() / mask.in_frame.sum()),
    detection_parameters=dict(windows_hours=[1, 3, 6, 12, "day"], ratio_max=0.08, k_sigma=5.0, pad_hours=1,
                              min_expected_per_hour=25),
    robust_sigma_per_window={str(k): v for k, v in info["detection"].items()},
    weekday_factor=info["weekday_factor"],
    zero_post_frame_days_list=day_tab[(day_tab.frame_h > 0) & (day_tab["count"] == 0)].date.tolist(),
    outage_days_list=out_days.date.tolist())
yr = mask.assign(year=pd.to_datetime(mask.hour * 3600, unit="s").dt.year)
summ["by_year"] = yr.groupby("year").agg(frame_hours=("in_frame", "sum"), outage_hours=("outage", "sum"),
                                         observed_hours=("observed", "sum")).astype(int).to_dict("index")
# plot
fig, ax = plt.subplots(3, 1, figsize=(11, 9.5))
fr = day_tab[day_tab.frame_h > 0]
ax[0].plot(fr.dt, fr["count"], ".", ms=3, color="#444", label="frame day")
ax[0].plot(fr.dt, fr.expected, "-", lw=0.7, color="#2a7", label="expected")
o = fr[fr.out_h > 0]
ax[0].plot(o.dt, o["count"], "o", ms=5, mfc="none", color="#d33", label="day with outage hours")
ax[0].set_ylabel("posts per Eastern day")
ax[0].set_title("Congress-wide daily posts (full archive) and detected outages")
ax[0].legend(loc="upper right", fontsize=8)
m_ = mask.assign(month=pd.to_datetime(mask.hour * 3600, unit="s").dt.to_period("M").dt.to_timestamp())
mm_ = m_.groupby("month").agg(obs=("observed", "sum"), out=("outage", "sum"), frame=("in_frame", "sum"),
                              n=("hour", "size")).reset_index()
mm_["design"] = mm_.n - mm_.frame
ax[1].bar(mm_.month, mm_.obs / 24, width=20, color="#4a7", label="observed")
ax[1].bar(mm_.month, mm_.out / 24, bottom=mm_.obs / 24, width=20, color="#d33", label="outage")
ax[1].bar(mm_.month, mm_.design / 24, bottom=(mm_.obs + mm_.out) / 24, width=20, color="#ccc",
          label="not in collection design")
ax[1].set_ylabel("days per month")
ax[1].legend(fontsize=8, ncol=3, loc="upper right")
rr = fr[fr.expected > 200].copy()
rr["ratio"] = rr["count"] / rr.expected
ax[2].hist(np.log2(rr.ratio.clip(0.01, 8)), bins=80, color="#68a")
ax[2].axvline(np.log2(0.08), color="#d33", ls="--", label="ratio 0.08 (limit)")
ax[2].set_xlabel("log2(observed / expected) of the daily total, frame days")
ax[2].legend(fontsize=8)
for a_ in ax[:2]:
    a_.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(T1 / "outage_plot.png", dpi=130)
plt.close()
(T1 / "outage_summary.json").write_text(json.dumps(summ, indent=2, default=str))
print("outages done", round(time.time() - t0), "s", {k: summ[k] for k in
      ("outage_hours", "outage_runs", "outage_days_touched", "zero_post_frame_days", "posts_in_outage_hours")},
      flush=True)

# ---------------------------------------------------------------- 2. threads
amap = load_amap()
tw = raw.merge(amap[["screen_name", "bioguide", "account_type"]], on="screen_name", how="inner")
n_matched = len(tw)
tw["nonrt"] = tv.nonretweet_mask_congresstweets(tw.text.values)
n_rt = int((~tw.nonrt).sum())
tw = tw[tw.nonrt].copy()
hh = tw.t.values // 3600 - hour0
tw["observed_hour"] = mask.observed.values[hh]
n_unobs = int((~tw.observed_hour).sum())
n_nonframe = int((~mask.in_frame.values[hh]).sum())
n_outage = int((mask.outage.values[hh]).sum())
tw = tw[tw.observed_hour].drop(columns=["nonrt", "observed_hour"])
x = tw.sort_values(["screen_name", "t", "id"]).reset_index(drop=True)
mk = hy.thread_markers(x.text.values)
forbid = np.array(hy.addressee_mismatch(x.text.values), dtype=bool)
forbid[1:] &= (x.screen_name.values[1:] == x.screen_name.values[:-1])
forbid[0] = False
tid, rule = hy.collapse_threads(x.screen_name.values, x.t.values, mk.num.values, mk.tot.values, mk.kw.values,
                                forbid=forbid)
x["thread_id"] = tid
x["rule"] = rule
x["num"] = mk.num.values
x["kw"] = mk.kw.values
x["size"] = x.groupby("thread_id")["thread_id"].transform("size").astype("int32")
x["pos"] = x.groupby("thread_id").cumcount().astype("int32")
x["cclass"] = tv.client_class(x.source.values)
x["tool"] = (x.cclass == "tool")
x["minute_quarter"] = np.isin((x.t.values // 60) % 60, (0, 15, 30, 45))
x[["id", "screen_name", "bioguide", "account_type", "t", "thread_id", "pos", "size", "rule", "num", "kw", "source",
   "cclass"]].to_parquet(T1 / "posts_threads.parquet", index=False)
g = x.groupby("thread_id")
ev = g.agg(screen_name=("screen_name", "first"), bioguide=("bioguide", "first"),
           account_type=("account_type", "first"), t=("t", "first"), t_end=("t", "last"), size=("size", "first"),
           first_id=("id", "first"), source=("source", "first"), cclass=("cclass", "first"),
           n_numbered=("num", lambda s: int((s >= 0).sum())), any_kw=("kw", "max"),
           max_rule=("rule", "max")).reset_index()
ev["duration_s"] = ev.t_end - ev.t
ev["tool_client"] = ev.cclass == "tool"
ev["minute_quarter"] = np.isin((ev.t.values // 60) % 60, (0, 15, 30, 45))
ev["sched_like"] = ev.tool_client & ev.minute_quarter
ev.to_parquet(T1 / "events.parquet", index=False)
n_posts, n_ev = len(x), len(ev)
sz = ev["size"]
rc = pd.Series(x.rule.values).value_counts().to_dict()
dur = ev.duration_s
acc = x.groupby("screen_name").agg(posts=("id", "size"), threads=("thread_id", "nunique"))
acc["collapsed_share"] = 1 - acc.threads / acc.posts
dt_prev = np.concatenate([[10 ** 9], np.diff(x.t.values)])
ts = dict(
    matched_posts=int(n_matched), retweets_excluded=n_rt, nonretweet_posts=int(n_matched - n_rt),
    dropped_unobserved_hour=n_unobs, dropped_non_frame_hours=n_nonframe, dropped_outage_hours=n_outage,
    posts_used=int(n_posts), events=int(n_ev), posts_collapsed_into_other_event=int(n_posts - n_ev),
    share_posts_collapsed=float(1 - n_ev / n_posts), share_events_multi=float((sz >= 2).mean()),
    share_posts_in_multi_threads=float(sz[sz >= 2].sum() / n_posts),
    median_size_all=float(sz.median()), median_size_multi=float(sz[sz >= 2].median()),
    mean_size_multi=float(sz[sz >= 2].mean()), p90_size_multi=float(sz[sz >= 2].quantile(0.9)),
    max_size=int(sz.max()), size_counts={int(k): int(v) for k, v in sz.value_counts().sort_index().head(10).items()},
    link_rules={RN: int(rc.get(k, 0)) for k, RN in hy.RULE_NAMES.items()},
    threads_with_numbered_marker=int((ev.n_numbered >= 2).sum()), posts_with_thread_keyword=int(x.kw.sum()),
    posts_numbered=int((x.num >= 0).sum()),
    multi_thread_duration_s_median=float(dur[sz >= 2].median()),
    multi_thread_duration_s_p90=float(dur[sz >= 2].quantile(0.9)),
    threads_longer_than_1h=int((dur > 3600).sum()), threads_ge_20_posts=int((sz >= 20).sum()),
    accounts=int(len(acc)), account_collapsed_share_median=float(acc.collapsed_share.median()),
    account_collapsed_share_p90=float(acc.collapsed_share.quantile(0.9)),
    account_collapsed_share_max=float(acc.collapsed_share.max()),
    by_account_type={k: dict(posts=int(len(v)), collapsed=float(1 - v.thread_id.nunique() / len(v)))
                     for k, v in x.groupby("account_type")},
    different_addressee_pairs_not_linked=int((forbid & (dt_prev < 60)).sum()))
(T1 / "thread_stats.json").write_text(json.dumps(ts, indent=2))
print("threads done", round(time.time() - t0), "s",
      {k: ts[k] for k in ("posts_used", "events", "share_posts_collapsed", "median_size_multi")}, flush=True)


# validation sample: 200 random linked posts, compared with proxy evidence
def toks(s):
    return set(re.findall(r"\w+", str(s).lower()))


def proxy(p_text, c_text, rule):
    if rule in (2, 3):
        return "marker"
    ph = re.match(r"^@(\w+)", p_text or "")
    ch = re.match(r"^@(\w+)", c_text or "")
    if ph and ch and ph.group(1).lower() == ch.group(1).lower():
        return "same_addressee"
    pu = set(re.findall(r"https?://\S+", p_text or ""))
    cu = set(re.findall(r"https?://\S+", c_text or ""))
    if pu & cu:
        return "same_url"
    a, b = toks(p_text), toks(c_text)
    if a and b and len(a & b) / len(a | b) >= 0.5:
        return "similar_text"
    return "unclear"


linked = np.flatnonzero(x.rule.values > 0)
pick = np.sort(rng.choice(linked, size=200, replace=False))
rows = []
for i in pick:
    rows.append(dict(i=int(i), screen_name=x.screen_name[i], gap_s=int(x.t[i] - x.t[i - 1]),
                     rule=hy.RULE_NAMES[int(x.rule[i])], thread_size=int(x["size"][i]),
                     prev_text=x.text[i - 1][:160].replace("\n", " "), text=x.text[i][:160].replace("\n", " "),
                     proxy=proxy(x.text[i - 1], x.text[i], int(x.rule[i]))))
vs = pd.DataFrame(rows)
vs["manual_label"] = ""
vs.to_csv(T1 / "thread_validation_sample.csv", index=False)
print(vs.proxy.value_counts().to_dict())
same = np.zeros(len(x), bool)
same[1:] = x.screen_name.values[1:] == x.screen_name.values[:-1]
gaps = np.zeros(len(x), dtype=int)
gaps[1:] = x.t.values[1:] - x.t.values[:-1]
rate = {}
for lo, hi in ((0, 60), (60, 120), (120, 300)):
    idx = np.flatnonzero(same & (gaps >= lo) & (gaps < hi) & (x.rule.values != 2) & (x.rule.values != 3)
                         & ~forbid)
    idx = rng.choice(idx, size=min(3000, len(idx)), replace=False)
    pr = [proxy(x.text[i - 1], x.text[i], 1) for i in idx]
    rate[f"{lo}-{hi}s"] = float(np.mean([p != "unclear" for p in pr]))
print("proxy-continuity rate by gap bucket", rate, flush=True)
json.dump(dict(sample_proxy_counts=vs.proxy.value_counts().to_dict(), proxy_rate_by_gap=rate),
          open(T1 / "thread_validation_proxy.json", "w"), indent=2)

# ---------------------------------------------------------------- 3. scheduler
rows = []
posts_by_acct = {a: d.t.values for a, d in x.groupby("screen_name")}
for acct, d in ev.groupby("screen_name"):
    s = hy.scheduler_stats(d.t.values, d.tool_client.values)
    sp = hy.scheduler_stats(posts_by_acct[acct])
    rows.append(dict(screen_name=acct, bioguide=d.bioguide.iloc[0], account_type=d.account_type.iloc[0],
                     n_events=s["n"], quarter_share=s["quarter_share"], quarter_ratio=s["quarter_ratio"],
                     quarter_z=s["quarter_z"], min00_share=s["min00_share"], sec0_share=s["sec0_share"],
                     tool_share=s["tool_share"], quarter_share_posts=sp["quarter_share"],
                     sched_like_share=float(d.sched_like.mean())))
sc = pd.DataFrame(rows)
sc["flag_minute"] = (sc.n_events >= 200) & (sc.quarter_ratio >= 2.0) & (sc.quarter_z >= 6)
sc["flag_tool"] = (sc.n_events >= 200) & (sc.tool_share >= 0.6)
sc["flag_strong"] = sc.flag_minute
sc.to_csv(T1 / "scheduler_accounts.csv", index=False)
q = sc[sc.n_events >= 200]
tool_ev = ev[ev.tool_client]
nt_ev = ev[~ev.tool_client]
ss = dict(accounts_total=int(len(sc)), accounts_n200=int(len(q)),
          flag_minute=int(q.flag_minute.sum()), flag_tool=int(q.flag_tool.sum()),
          flag_both=int((q.flag_minute & q.flag_tool).sum()),
          flag_minute_by_type={k: int(v) for k, v in q.groupby("account_type").flag_minute.sum().items()},
          n200_by_type={k: int(v) for k, v in q.groupby("account_type").size().items()},
          quarter_ratio_median=float(q.quarter_ratio.median()), quarter_ratio_p90=float(q.quarter_ratio.quantile(0.9)),
          quarter_ratio_max=float(q.quarter_ratio.max()),
          quarter_ratio_max_account=q.loc[q.quarter_ratio.idxmax(), "screen_name"],
          overall_quarter_share_all_events=float(ev.minute_quarter.mean()),
          overall_quarter_share_tool_events=float(tool_ev.minute_quarter.mean()),
          overall_quarter_share_nontool_events=float(nt_ev.minute_quarter.mean()),
          overall_tool_share_events=float(ev.tool_client.mean()), sec0_share_overall=float(((ev.t % 60) == 0).mean()),
          sched_like_events=int(ev.sched_like.sum()), sched_like_share=float(ev.sched_like.mean()),
          tool_share_median_over_accounts=float(q.tool_share.median()),
          top_sources={k: int(v) for k, v in ev.source.value_counts().head(8).items()})
(T1 / "scheduler_summary.json").write_text(json.dumps(ss, indent=2, default=str))
print("scheduler", ss, flush=True)
print("done", round(time.time() - t0), "s")
