"""FULL-corpus rerun (output/v7/t2full) of T2 step 7: event detection from Congress-wide volume. Rhythm-adjusted robust z-scores of hourly and daily volume,
top 60 event days with common hashtags, distinctive words and sample texts.

Volume = thread-collapsed non-retweet events of all matched member accounts, observed hours only.
Expected volume E: level x weekday x hour-of-day rhythm of xbox.hygiene.detect_outages (rolling median level over
11 frame days, so single-day spikes do not move it), fit to the events series.
Hourly z: l = log((C+.5)/(E+.5)); z = (l - median_k)/(1.4826 MAD_k) within each of the 168 Eastern hour-of-week bins.
Daily z: the same on the daily total (days with no outage hours), within weekday.

Outputs (output/v7/t2/): congress_z_hourly.parquet, congress_daily.csv, event_days_top60.csv, event_hours_top60.csv,
event_summary.json
Usage: python t2_events.py
"""
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t1full_common import *  # noqa: E402,F401,F403

t0 = time.time()
rng = np.random.default_rng(SEED)
T2.mkdir(parents=True, exist_ok=True)
cong = pd.read_parquet(T2 / "congress_hourly.parquet")
hour0 = int(cong.hour_idx.iloc[0])
tot = cong.total_events.values
# expected from the events series (the mask itself is the canonical one)
fit = hy.detect_outages(tot, hour0, design=None)
E = fit.expected.values
obs = cong.observed.values
lr = np.log((tot + 0.5) / (E + 0.5))
how = (cong.weekday.values * 24 + cong.et_hour.values).astype(int)
z = np.full(len(cong), np.nan)
for k in range(168):
    m = obs & (how == k) & (E > 0)
    if m.sum() < 20:
        continue
    med = np.median(lr[m])
    mad = 1.4826 * np.median(np.abs(lr[m] - med))
    z[m & (how == k)] = (lr[m] - med) / max(mad, 0.05)
cong["expected"] = E
cong["log_ratio"] = lr
cong["z_hour"] = z
cong[["hour_idx", "hour_utc", "observed", "et_day", "et_hour", "weekday", "total_events", "expected", "z_hour"]].to_parquet(
    T2 / "congress_z_hourly.parquet", index=False)

# ---- daily
cong["date"] = pd.to_datetime(cong.et_day, unit="D")
d = cong[cong.in_frame].groupby("et_day").agg(date=("date", "first"), weekday=("weekday", "first"),
                                              events=("total_events", "sum"), expected=("expected", "sum"),
                                              frame_h=("in_frame", "sum"), obs_h=("observed", "sum")).reset_index()
d["complete"] = (d.obs_h == d.frame_h) & (d.frame_h >= 23)
d["log_ratio"] = np.log((d.events + 0.5) / (d.expected + 0.5))
d["z_day"] = np.nan
for k in range(7):
    m = d.complete & (d.weekday == k)
    med = d.log_ratio[m].median()
    mad = 1.4826 * np.median(np.abs(d.log_ratio[m] - med))
    d.loc[m, "z_day"] = (d.log_ratio[m] - med) / max(mad, 0.02)
d["ratio"] = d.events / d.expected
d["date_str"] = d.date.dt.strftime("%Y-%m-%d")
d.drop(columns=["et_day"]).to_csv(T2 / "congress_daily.csv", index=False)
top_h = cong[obs].nlargest(60, "z_hour")[["hour_utc", "et_day", "et_hour", "total_events", "expected", "z_hour"]].copy()
top_h["et_date"] = pd.to_datetime(top_h.et_day, unit="D").dt.strftime("%Y-%m-%d")
top_h.drop(columns=["et_day"]).to_csv(T2 / "event_hours_top60.csv", index=False)
top = d[d.complete].nlargest(60, "z_day").reset_index(drop=True)
print("z done", round(time.time() - t0), "s; days>3:", int((d.z_day > 3).sum()), "of", int(d.complete.sum()), flush=True)

# ---- text mining for the top days
STOP = set("""about above after again against all also and any are because been before being below between both but
can could did does doing down during each few for from further had has have having her here hers him his how into its
itself just more most much must not now off once only other our ours out over own same she should some such than that
the their theirs them then there these they this those through too under until upon very was were what when where which
while who whom why will with would you your yours yourself today this week will thank thanks join make made need want
every year years time must new more many like get got going come back help work working american americans congress
house senate bill act help via read live rep sen amp https http pic twitter com one two three first last great
today's said says say still even take taking part right people country states state national across proud
""".split())
amap = load_amap()
raw = load_raw().merge(amap[["screen_name", "party"]], on="screen_name")
raw = raw[tv.nonretweet_mask_congresstweets(raw.text.values)].copy()
hrs = raw.t.values // 3600 - hour0
raw = raw[obs[hrs]].copy()
raw["et_day"] = hy.et_fields(raw.t.values // 3600)[0]
frame_days = d.et_day.values
top_days = set(top.et_day.values)
need = set()
for ed in top_days:
    need |= set(frame_days[(np.abs(frame_days - ed) <= 45)])
sub = raw[raw.et_day.isin(need)].copy()
HT = re.compile(r"#\w+")
WD = re.compile(r"[a-z][a-z']{3,}")
URL = re.compile(r"https?://\S+|@\w+")


def toks(text):
    s = URL.sub(" ", text.lower())
    tags = set(HT.findall(s))
    words = {w for w in WD.findall(HT.sub(" ", s)) if w not in STOP}
    return tags, words


tg, wd = [], []
for tx in sub.text.values:
    a, b = toks(tx)
    tg.append(a)
    wd.append(b)
sub["tags"], sub["words"] = tg, wd
print("tokenized", len(sub), round(time.time() - t0), "s", flush=True)
by_day = {k: v for k, v in sub.groupby("et_day")}
cnt_w, cnt_t, n_posts = {}, {}, {}
for k, v in by_day.items():
    cw, ct = Counter(), Counter()
    for ws in v.words.values:
        cw.update(ws)
    for ts_ in v.tags.values:
        ct.update(ts_)
    cnt_w[k], cnt_t[k], n_posts[k] = cw, ct, len(v)
rows = []
for r in top.itertuples():
    ed = r.et_day
    v = by_day[ed]
    base_days = [x for x in frame_days if x != ed and abs(x - ed) <= 45 and x in by_day and x not in top_days]
    bw, bn = Counter(), 0
    for x in base_days:
        bw.update(cnt_w[x])
        bn += n_posts[x]
    N = n_posts[ed]
    sc = {}
    for w, c in cnt_w[ed].items():
        if c < 20:
            continue
        b = (bw.get(w, 0) + 0.5) / (bn + 1)
        sc[w] = c * np.log((c / N) / b)
    topw = [w for w, _ in sorted(sc.items(), key=lambda kv: -kv[1])[:5]]
    topt = [f"{t_} ({c})" for t_, c in cnt_t[ed].most_common(5)]
    key = set(topw) | {t_.split(" ")[0] for t_ in topt[:3]}
    sc10 = set([w for w, _ in sorted(sc.items(), key=lambda kv: -kv[1])[:10]]) | key
    match = np.array([len((ws | ts_) & sc10) for ws, ts_ in zip(v.words.values, v.tags.values)])
    longish = v.text.str.len().values >= 80
    cand = np.flatnonzero((match >= 2) & longish)
    if len(cand) < 3:
        cand = np.flatnonzero(match >= 1)
    if len(cand) < 3:
        cand = np.arange(len(v))
    cand = cand[np.argsort(-match[cand], kind="stable")][:60]
    chosen, seen = [], set()
    for i in rng.permutation(cand):
        a_ = v.screen_name.values[i]
        if a_ in seen:
            continue
        seen.add(a_)
        chosen.append(i)
        if len(chosen) == 3:
            break
    samples = [f"{v.screen_name.values[i]} ({v.party.values[i]}): " + v.text.values[i].replace("\n", " ")[:230]
               for i in chosen]
    rows.append(dict(rank=len(rows) + 1, date=r.date_str, weekday=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][r.weekday],
                     events=int(r.events), expected=round(float(r.expected)), ratio=round(float(r.ratio), 2),
                     z_day=round(float(r.z_day), 2), posts_in_text_sample=N,
                     top_hashtags="; ".join(topt), distinctive_words=", ".join(topw),
                     sample_1=samples[0] if len(samples) > 0 else "", sample_2=samples[1] if len(samples) > 1 else "",
                     sample_3=samples[2] if len(samples) > 2 else ""))
out = pd.DataFrame(rows)
out.to_csv(T2 / "event_days_top60.csv", index=False)
summ = dict(frame_days=int(len(d)), complete_days=int(d.complete.sum()), days_z_gt_3=int((d.z_day > 3).sum()),
            days_z_gt_5=int((d.z_day > 5).sum()), days_z_lt_minus3=int((d.z_day < -3).sum()),
            hours_z_gt_5=int((cong.z_hour > 5).sum()), hours_observed=int(obs.sum()),
            top_day=str(out.date.iloc[0]), top_z=float(out.z_day.iloc[0]),
            top60_min_z=float(out.z_day.iloc[-1]), elapsed_s=round(time.time() - t0))
(T2 / "event_summary.json").write_text(json.dumps(summ, indent=2))
print(json.dumps(summ, indent=1))
