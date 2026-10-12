"""T6 tasks 4 and 5: personal vs official contrast (volume, post-type mix, local-time profiles, thread-collapsed descriptors)
and the contemporaneous count-based reference (36 office accounts, a NON-REPRESENTATIVE convenience sample in list order).
Local time: the Lee location model (VoteView vote days, Utah home zone) is applied to both accounts for comparability;
the official account's operators may post from elsewhere (stated limitation). Scenarios all_dc / all_home also reported.
Reference: hourly counts (all post types, counts endpoint, from:handle) 2023-07-01 to 2026-10-05 UTC, local = home state zone
(state-majority IANA zone; district zones not used for House members). Lee archives are put on the same footing
(all post types, same window, all_home = Mountain).
Outputs: contrast_descriptors.csv, contrast_type_mix.csv, hour_profiles.csv, reference_features.csv, contrast_numbers.json, fig3_hour_profiles.png
"""
import sys
import glob
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from t6_common import *  # noqa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

num = {}
P = load_posts("personal")
O = load_posts("official")
# official author id from raw pages
from collections import Counter
cn = Counter()
for f in glob.glob(str(ROOT / "data/senmikelee_raw_20261009/page_*.json")):
    for tw in json.load(open(f)).get("data", []):
        cn[tw["author_id"]] += 1
OFF_ID = cn.most_common(1)[0][0]
num["official_author_id_share"] = cn.most_common(1)[0][1] / sum(cn.values())

# ---- location model (Lee)
md, cd, mch = lc.load_vote_days(ROOT / "data/voteview")
ml = lc.MemberLocation(md[LEE_ICPSR], cd["S"], "UT", name="Lee")


def collapse(df, self_id):
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
    extra = np.zeros(n, dtype=bool)
    selfreply = (typ == "reply") & (irt == self_id)
    for k in range(1, 6):
        prev_id = np.concatenate([[""] * k, ids[:-k]])
        dt = np.concatenate([[10 ** 9] * k, t[k:] - t[:-k]])
        extra |= selfreply & (ref == prev_id) & (dt < 3600)
    pc = np.concatenate([[""], conv[:-1]])
    dt1 = np.concatenate([[10 ** 9], np.diff(t)])
    extra |= selfreply & (conv == pc) & (conv != "nan") & (dt1 < 3600)
    g = np.zeros(n, dtype=object)
    g[:] = "a"
    tid, rule = hy.collapse_threads(g, t, mk.num.values, mk.tot.values, mk.kw.values, extra_link=extra, forbid=forbid)
    x["thread_id"] = tid
    x["size"] = x.groupby("thread_id")["thread_id"].transform("size")
    ev = x.groupby("thread_id").agg(t=("t", "first"), size=("size", "first")).reset_index()
    return x, ev


def d2s(s):
    return int(np.datetime64(s, "s").astype("int64"))


wins = {"full": (d2s("2022-07-25"), d2s("2026-10-06"))}
for y in range(2023, 2027):
    wins[f"y{y}"] = (d2s(f"{y}-01-01"), d2s(f"{y + 1}-01-01"))
wins["jan_jul_2025"] = (d2s("2025-01-01"), d2s("2025-08-01"))
drows = []
EV = {}
for acct, D, sid in [("personal", P, LEE_ID), ("official", O, OFF_ID)]:
    x, ev = collapse(D, sid)
    EV[acct] = (x, ev)
    num[f"{acct}_nonrt_posts"] = int(len(x))
    num[f"{acct}_events"] = int(len(ev))
    num[f"{acct}_share_posts_collapsed"] = float(1 - len(ev) / len(x))
    for w, (lo, hi) in wins.items():
        m = (ev.t.values >= lo) & (ev.t.values < hi)
        r = describe(ev.t.values[m], ml, True, None, None, 0, ev["size"].values[m], naive_days=True)
        drows.append(dict(account=acct, window=w, variant="clean_events", **r))
        mr = (x.t.values >= lo) & (x.t.values < hi)
        r = describe(x.t.values[mr], ml, True, None, None, 0, None, naive_days=True)
        drows.append(dict(account=acct, window=w, variant="raw_nonrt_posts", **r))
dd = pd.DataFrame(drows)
dd.to_csv(T6 / "contrast_descriptors.csv", index=False)
sel = dd[(dd.window == "full")][["account", "variant", "n", "A_n", "B", "entropy", "night_00_06", "night_22_06", "weekend_share",
                                  "weekend_ratio", "rapid_share_300", "sessions_per_day_30", "events_per_session_30", "posts_per_session_30"]]
num["descriptors_full"] = {f"{r.account}|{r.variant}": {k: (None if pd.isna(v) else round(float(v), 4)) for k, v in r._asdict().items()
                                                         if k not in ("Index", "account", "variant")} for r in sel.itertuples()}

# ---- type mix
tm = []
for acct, D in [("personal", P), ("official", O)]:
    D = D[D.et_date.between(D0, D1)]
    ct = pd.crosstab(D.et_date.dt.year, D.type)
    sh = ct.div(ct.sum(1), axis=0)
    for y in ct.index:
        for ty in ct.columns:
            tm.append(dict(account=acct, year=int(y), type=ty, n=int(ct.loc[y, ty]), share=float(sh.loc[y, ty])))
    tot = D.type.value_counts()
    num[f"{acct}_type_share_all"] = {k: round(float(v / tot.sum()), 4) for k, v in tot.items()}
pd.DataFrame(tm).to_csv(T6 / "contrast_type_mix.csv", index=False)

# ---- hour profiles (non-retweet posts, three scenarios)
hp = []
prof = {}
for acct, D in [("personal", P), ("official", O)]:
    t = D[(D.type != "retweet") & D.et_date.between(D0, D1)].t.values
    cls = ml.classes(t)
    sc = ml.local_hour_dow(t, ("model_trans_dc", "all_dc", "all_home"), cls)
    for s, (h, dow, day) in sc.items():
        p = np.bincount(h, minlength=24) / len(h)
        for hh in range(24):
            hp.append(dict(account=acct, scenario=s, hour=hh, share=float(p[hh])))
        wk_share = float((dow >= 5).mean())
        num[f"{acct}_{s}"] = dict(night_00_06=float(np.isin(h, range(0, 6)).mean()), weekend_share=wk_share,
                                  weekend_ratio=wk_share / (2 / 7), n=int(len(h)),
                                  entropy_norm=float(tv.entropy_norm(h)), peak_hour=int(p.argmax()))
        if s == "model_trans_dc":
            prof[acct] = (p, np.bincount(dow, minlength=7) / len(dow))
pd.DataFrame(hp).to_csv(T6 / "hour_profiles.csv", index=False)

# ---- contemporaneous reference
acc = pd.read_csv(ROOT / "output/v7/counts/accounts.csv").set_index("handle")
C = pd.concat(pd.read_parquet(f) for f in glob.glob(str(ROOT / "output/v7/counts/hourly/*/*.parquet")))
C["hour_start_utc"] = pd.to_datetime(C.hour_start_utc, utc=True)
C = C[(C.hour_start_utc >= "2023-07-01") & (C.hour_start_utc < "2026-10-06")]
full = C.groupby("account").size()
full = full[full == full.max()].index
ndays = (pd.Timestamp("2026-10-06") - pd.Timestamp("2023-07-01")).days


def feats_from_hourly(hours_utc, counts, zone):
    loc = hours_utc.dt.tz_convert(zone)
    h = loc.dt.hour.values
    dow = loc.dt.dayofweek.values
    tot = counts.sum()
    ph = np.bincount(h, weights=counts, minlength=24) / tot
    nz = ph[ph > 0]
    return dict(total_posts=int(tot), posts_per_day=tot / ndays, entropy_norm=float(-(nz * np.log2(nz)).sum() / np.log2(24)),
                night_00_06=float(ph[:6].sum()), weekend_share=float(counts[dow >= 5].sum() / tot),
                weekend_ratio=float(counts[dow >= 5].sum() / tot / (2 / 7)))


rows = []
for a in full:
    g = C[C.account == a].sort_values("hour_start_utc")
    z = lc.STATE_ZONE[acc.loc[a, "state"]]
    rows.append(dict(account=a, kind="reference_office", chamber=acc.loc[a, "chamber"], state=acc.loc[a, "state"],
                     **feats_from_hourly(g.hour_start_utc, g["count"].values.astype(float), z)))
for nm, D in [("@BasedMikeLee (personal)", P), ("@SenMikeLee (official)", O)]:
    t = D[(D.t >= d2s("2023-07-01")) & (D.t < d2s("2026-10-06"))].t.values
    hrs = pd.Series(pd.to_datetime(t // 3600 * 3600, unit="s", utc=True))
    cnt = hrs.value_counts().sort_index()
    for zname, z in [("all_home_Mountain", "America/Denver"), ("all_dc", "America/New_York")]:
        rows.append(dict(account=nm, kind="lee_" + zname, chamber="senate", state="UT",
                         **feats_from_hourly(pd.Series(cnt.index), cnt.values.astype(float), z)))
RF = pd.DataFrame(rows)
RF.to_csv(T6 / "reference_features.csv", index=False)
ref = RF[RF.kind == "reference_office"]
ref_sen = ref[ref.chamber == "senate"]
num["reference"] = dict(n_accounts=int(len(ref)), n_senate=int(len(ref_sen)), window="2023-07-01 to 2026-10-05 UTC",
                        label="non-representative convenience sample: first 36 office accounts in list order")
for c in ["posts_per_day", "entropy_norm", "night_00_06", "weekend_share"]:
    num["reference"][c] = dict(median=float(ref[c].median()), q25=float(ref[c].quantile(.25)), q75=float(ref[c].quantile(.75)),
                               min=float(ref[c].min()), max=float(ref[c].max()))
    for nm in ["@BasedMikeLee (personal)", "@SenMikeLee (official)"]:
        r = RF[(RF.account == nm) & (RF.kind == "lee_all_home_Mountain")].iloc[0]
        num["reference"].setdefault("lee_" + nm.split("@")[1].split(" ")[0], {})[c] = dict(
            value=float(r[c]), rank_of_37=int((ref[c] < r[c]).sum() + 1), n_reference_below=int((ref[c] < r[c]).sum()))
wjson("contrast_numbers.json", num)

# ---- Figure 3
fig, axs = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw=dict(width_ratios=[1.5, 1, 1.5]))
hrs_ = np.arange(24)
axs[0].plot(hrs_, prof["personal"][0], color=PAL["personal"], lw=2, marker="o", ms=3, label="@BasedMikeLee (personal)")
axs[0].plot(hrs_, prof["official"][0], color=PAL["official"], lw=2, marker="s", ms=3, label="@SenMikeLee (official)")
axs[0].set_xlabel("Local hour (location model, primary scenario)")
axs[0].set_ylabel("Share of non-retweet posts")
axs[0].legend(fontsize=8)
axs[0].set_title("Hour of day, 2022-07 to 2026-10", fontsize=9)
dn = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
w = 0.38
axs[1].bar(np.arange(7) - w / 2, prof["personal"][1], w, color=PAL["personal"], label="personal")
axs[1].bar(np.arange(7) + w / 2, prof["official"][1], w, color=PAL["official"], label="official")
axs[1].axhline(1 / 7, color="k", ls=":", lw=1)
axs[1].set_xticks(range(7))
axs[1].set_xticklabels(dn, fontsize=8)
axs[1].set_title("Day of week (dotted: uniform 1/7)", fontsize=9)
# reference profile
ref_prof = {}
Cg = C[C.account.isin(full)].copy()
for a, g in Cg.groupby("account"):
    z = lc.STATE_ZONE[acc.loc[a, "state"]]
    h = g.hour_start_utc.dt.tz_convert(z).dt.hour.values
    p = np.bincount(h, weights=g["count"].values.astype(float), minlength=24)
    ref_prof[a] = p / p.sum()
R_ = np.array(list(ref_prof.values()))
axs[2].fill_between(hrs_, np.percentile(R_, 10, axis=0), np.percentile(R_, 90, axis=0), color="#bbbbbb", alpha=0.6,
                    label="36 office accounts, 10th-90th pct (convenience sample)")
axs[2].plot(hrs_, np.median(R_, axis=0), color="#555555", lw=1.5, label="median of 36")
for nm, col, lab in [("@BasedMikeLee (personal)", PAL["personal"], "personal"), ("@SenMikeLee (official)", PAL["official"], "official")]:
    D = P if "Based" in nm else O
    t = D[(D.t >= d2s("2023-07-01")) & (D.t < d2s("2026-10-06"))].t.values
    h = pd.to_datetime(t, unit="s", utc=True).tz_convert("America/Denver").hour.values
    axs[2].plot(hrs_, np.bincount(h, minlength=24) / len(h), color=col, lw=2, label=f"{lab}, all posts, Mountain time")
axs[2].set_xlabel("Hour in home-state time zone (all post types)")
axs[2].set_title("Contemporaneous reference, 2023-07 to 2026-10", fontsize=9)
axs[2].legend(fontsize=7)
for ax in axs:
    ax.grid(alpha=0.25)
fig.savefig(T6 / "fig3_hour_profiles.png", dpi=160, bbox_inches="tight")
print(json.dumps(num, indent=1, default=float)[:9000])
