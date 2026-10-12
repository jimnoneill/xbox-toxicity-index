"""T1 step 4 for @BasedMikeLee: thread collapse (conversation_id / self-reply chain + the same gap and marker rules)
and descriptors on raw vs collapsed events. The archive is an API pull, not the daily scrape, so no outage mask is
applied; collection holes are reported (longest gaps, zero-post days).
Outputs: output/v7/t1/lee_threads.parquet, lee_descriptors.csv, lee_summary.json
Usage: python t1_lee.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t1_common import *  # noqa: E402,F401,F403

T1.mkdir(parents=True, exist_ok=True)
df = load_lee()
nonrt = tv.nonretweet_mask_lee(df.type.values)
x = df[nonrt].reset_index(drop=True)
n = len(x)
mk = hy.thread_markers(x.text.fillna("").values)
forbid = np.array(hy.addressee_mismatch(x.text.fillna("").values), dtype=bool)
forbid[0] = False
t = x.t.values
ids = x.id.astype(str).values
irt = x.in_reply_to_user_id.astype(str).values
conv = x.conversation_id.astype(str).values
ref = x.referenced_tweet_ids.astype(str).values
typ = x.type.values
# self-reply chain: a reply to Lee himself that points at one of his previous 5 posts (<1 h) or shares the
# conversation_id of the previous post (where the field is present)
extra = np.zeros(n, dtype=bool)
selfreply = (typ == "reply") & (irt == LEE_ID)
for k in range(1, 6):
    prev_id = np.concatenate([[""] * k, ids[:-k]])
    dt = np.concatenate([[10 ** 9] * k, t[k:] - t[:-k]])
    extra |= selfreply & (ref == prev_id) & (dt < 3600)
pc = np.concatenate([[""], conv[:-1]])
dt1 = np.concatenate([[10 ** 9], np.diff(t)])
extra |= selfreply & (conv == pc) & (conv != "nan") & (dt1 < 3600)
g = np.zeros(n, dtype=object)
g[:] = "lee"
tid, rule = hy.collapse_threads(g, t, mk.num.values, mk.tot.values, mk.kw.values, extra_link=extra, forbid=forbid)
x["thread_id"], x["rule"] = tid, rule
x["size"] = x.groupby("thread_id")["thread_id"].transform("size")
ev = x.groupby("thread_id").agg(t=("t", "first"), t_end=("t", "last"), size=("size", "first"),
                                first_id=("id", "first"), max_rule=("rule", "max")).reset_index()
ev.to_parquet(T1 / "lee_threads.parquet", index=False)
rc = pd.Series(rule).value_counts().to_dict()
sz = ev["size"]
by_year = x.assign(year=pd.to_datetime(x.t, unit="s").dt.year).groupby("year").agg(
    posts=("id", "size"), events=("thread_id", "nunique"))
by_year["collapsed_share"] = 1 - by_year.events / by_year.posts
# collection holes
days = pd.to_datetime(df.t, unit="s").dt.floor("D")
allday = pd.date_range(days.min(), days.max(), freq="D")
cnt = days.value_counts().reindex(allday, fill_value=0)
gaps = np.diff(df.t.values)
summ = dict(
    posts_all=int(len(df)), nonretweet_posts=int(n), events=int(len(ev)), share_posts_collapsed=float(1 - len(ev) / n),
    share_events_multi=float((sz >= 2).mean()), median_size_multi=float(sz[sz >= 2].median()),
    max_size=int(sz.max()), link_rules={hy.RULE_NAMES[k]: int(rc.get(k, 0)) for k in range(1, 5)},
    conversation_id_share=float((x.conversation_id.notna()).mean()),
    collapsed_share_by_year={int(k): float(v) for k, v in by_year.collapsed_share.items()},
    zero_post_days=int((cnt == 0).sum()), calendar_days=int(len(cnt)),
    longest_gaps_hours=[float(v) for v in np.sort(gaps)[-8:][::-1] / 3600],
    first=str(pd.Timestamp(df.t.min(), unit="s")), last=str(pd.Timestamp(df.t.max(), unit="s")))
(T1 / "lee_summary.json").write_text(json.dumps(summ, indent=2))
print(json.dumps(summ, indent=1), flush=True)

# descriptors
md, cd, mch = lc.load_vote_days(ROOT / "data/voteview")
ml = lc.MemberLocation(md[LEE_ICPSR], cd["S"], "UT", name="Lee")


def d2s(s):
    return int(np.datetime64(s, "s").astype("int64"))


wins = {"full": (0, int(t.max()) + 1), "first12": (d2s("2022-07-25"), d2s("2023-07-25"))}
for y in range(2022, 2027):
    wins[f"y{y}"] = (d2s(f"{y}-01-01"), d2s(f"{y + 1}-01-01"))
rows = []
for w, (lo, hi) in wins.items():
    m_raw = (t >= lo) & (t < hi)
    m_ev = (ev.t.values >= lo) & (ev.t.values < hi)
    r = describe(t[m_raw], ml, True, None, None, 0, None, naive_days=True)
    rows.append(dict(window=w, variant="raw", **r))
    r = describe(ev.t.values[m_ev], ml, True, None, None, 0, ev["size"].values[m_ev], naive_days=True)
    rows.append(dict(window=w, variant="clean", **r))
res = pd.DataFrame(rows)
res.to_csv(T1 / "lee_descriptors.csv", index=False)
# compare with the earlier numbers
old = pd.read_csv(ROOT / "output/v7/temporal/lee_features_v2_parquet.csv")
old = old[old.postset == "nonrt"].set_index("window")
chk = {}
for w in ("full", "first12"):
    rr = res[(res.window == w) & (res.variant == "raw")].iloc[0]
    chk[w] = dict(A_n_old=float(old.loc[w, "A_n"]), A_n_raw=float(rr.A_n), night_old=float(old.loc[w, "night_00_06_model_trans_dc"]),
                  night_raw=float(rr.night_00_06), n_old=int(old.loc[w, "n"]), n_raw=int(rr.n))
(T1 / "lee_old_check.json").write_text(json.dumps(chk, indent=2))
pd.set_option("display.width", 250)
print(res[["window", "variant", "n", "A_n", "entropy", "night_00_06", "weekend_share", "rapid_share_300",
           "sessions_per_day_30", "events_per_session_30", "posts_per_session_30"]].round(3).to_string())
print(json.dumps(chk, indent=1))
