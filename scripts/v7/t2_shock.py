"""T2 step 6: shared-shock share. How much of an account's hourly activity moves with Congress-wide activity.

Per account (>= 200 events), on its observed hours between first and last event:
 (a) corr_hourly: Pearson correlation of the account's hourly count y_t with the leave-one-out Congress count C_{-a,t},
     after removing the account-local hour-of-week profile (168 bins) and calendar-month level from both series
     (sequential demeaning). corr_daily: same on Eastern-day sums (weekday and month removed, days with >= 20 hours).
     corr_within_day: Eastern-day means removed as well, so only hour-to-hour co-movement inside a day remains.
     r2_*: share of the rhythm-adjusted hourly variance explained by C (corr^2), by own previous hour (same day), by both.
 (b) spike share: hours in the top 1% of the rhythm-adjusted LOO Congress count (robust z by Eastern hour-of-week bin, from
     t2_events). spike_share = share of the account's events in those hours; expected shares: naive (spike hours / hours)
     and rhythm-matched (account's own fitted profile x month level). ratio = observed / expected.
 (c) summaries by party, account type, party x type, chamber, scheduler flag with bootstrap CIs over accounts.
Outputs (output/v7/t2/): shared_shock_accounts.csv, shared_shock_groups.csv, shared_shock_summary.json
Usage: python t2_shock.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t1_common import *  # noqa: E402,F401,F403

t0 = time.time()
rng = np.random.default_rng(SEED)
cong = pd.read_parquet(T2 / "congress_z_hourly.parquet")
hour0 = int(cong.hour_idx.iloc[0])
Ecg = cong.expected.values
tot = cong.total_events.values
obs = cong.observed.values
how_c = (cong.weekday.values * 24 + cong.et_hour.values).astype(int)
lr_c = np.log((tot + 0.5) / (Ecg + 0.5))
med_k, sig_k = np.zeros(168), np.ones(168)
for k in range(168):
    m = obs & (how_c == k) & (Ecg > 0)
    if m.sum() >= 20:
        med_k[k] = np.median(lr_c[m])
        sig_k[k] = max(1.4826 * np.median(np.abs(lr_c[m] - med_k[k])), 0.05)
dts = pd.to_datetime(cong.hour_utc.values).tz_localize("UTC").tz_convert(hy.ET)
month_c = (dts.year * 12 + dts.month).values.astype(int)

df = pd.read_parquet(T2 / "hourly_counts.parquet",
                     columns=["account", "bioguide", "account_type", "party", "chamber", "hour_idx", "local_hour",
                              "local_dow", "et_day", "count", "loo_congress_count"])
print("loaded", len(df), round(time.time() - t0), "s", flush=True)
sched = pd.read_csv(T1 / "scheduler_accounts.csv")[["screen_name", "flag_minute", "flag_tool"]].rename(
    columns={"screen_name": "account"})


def demean_by(x, key):
    s = np.bincount(key, weights=x)
    n = np.bincount(key)
    m = np.where(n > 0, s / np.maximum(n, 1), 0.0)
    return x - m[key], m


def rhythm_resid(x, how, month):
    r, _ = demean_by(x.astype(float), how)
    r, _ = demean_by(r, month)
    return r


def corr(a, b):
    if len(a) < 30 or a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


rows = []
accts = df.account.values
bounds = np.flatnonzero(np.r_[True, accts[1:] != accts[:-1], True])
for i0, i1 in zip(bounds[:-1], bounds[1:]):
    g = df.iloc[i0:i1]
    h = g.hour_idx.values.astype("int64")
    y = g["count"].values.astype(float)
    c = g.loo_congress_count.values.astype(float)
    how = (g.local_dow.values.astype(int) * 24 + g.local_hour.values.astype(int))
    ci = h - hour0
    mon = month_c[ci]
    _, mon = np.unique(mon, return_inverse=True)
    yr, cr = rhythm_resid(y, how, mon), rhythm_resid(c, how, mon)
    row = dict(account=g.account.iloc[0], bioguide=g.bioguide.iloc[0], account_type=g.account_type.iloc[0],
               party=g.party.iloc[0], chamber=g.chamber.iloc[0], n_events=int(y.sum()), hours=len(y))
    row["corr_hourly"] = corr(yr, cr)
    # daily
    ed = g.et_day.values.astype(int)
    _, di = np.unique(ed, return_inverse=True)
    nh = np.bincount(di)
    Yd, Cd = np.bincount(di, weights=y), np.bincount(di, weights=c)
    ok = nh >= 20
    if ok.sum() >= 30:
        dday = pd.to_datetime(np.unique(ed), unit="D")
        wk = dday.dayofweek.values
        mk = (dday.year * 12 + dday.month).values
        _, mk = np.unique(mk, return_inverse=True)
        yd, cd = rhythm_resid(Yd[ok], wk[ok], mk[ok]), rhythm_resid(Cd[ok], wk[ok], mk[ok])
        row["corr_daily"] = corr(yd, cd)
        row["days"] = int(ok.sum())
    # within day
    yw, _ = demean_by(yr, di)
    cw, _ = demean_by(cr, di)
    row["corr_within_day"] = corr(yw, cw)
    # lag-1 own momentum (previous hour present, same day)
    prev = np.r_[False, (h[1:] - h[:-1] == 1) & (ed[1:] == ed[:-1])]
    if prev.sum() > 100:
        yl, yc, cc = yr[:-1][prev[1:]], yr[1:][prev[1:]], cr[1:][prev[1:]]
        row["r2_c"] = corr(yc, cc) ** 2
        row["r2_lag1"] = corr(yc, yl) ** 2
        X = np.c_[np.ones(len(yc)), yl, cc]
        beta, *_ = np.linalg.lstsq(X, yc, rcond=None)
        res = yc - X @ beta
        row["r2_both"] = float(1 - res.var() / yc.var()) if yc.var() > 0 else np.nan
    # spike share
    kk = (cong.weekday.values[ci] * 24 + cong.et_hour.values[ci]).astype(int)
    zc = (np.log((c + 0.5) / (Ecg[ci] + 0.5)) - med_k[kk]) / sig_k[kk]
    thr = np.quantile(zc, 0.99)
    sp = zc >= thr
    _, bm = demean_by(y, how)
    yhat0 = bm[how]
    sm = np.bincount(mon, weights=y)
    sh = np.bincount(mon, weights=yhat0)
    fac = np.where(sh > 0, sm / np.maximum(sh, 1e-12), 1.0)
    yhat = yhat0 * fac[mon]
    obs_sp = y[sp].sum()
    row.update(spike_hours=int(sp.sum()), spike_events=int(obs_sp), spike_share=float(obs_sp / y.sum()),
               spike_expected_naive=float(sp.mean()), spike_expected_rhythm=float(yhat[sp].sum() / yhat.sum()),
               spike_expected_events=float(yhat[sp].sum()))
    row["spike_ratio_naive"] = row["spike_share"] / row["spike_expected_naive"]
    row["spike_ratio_rhythm"] = row["spike_share"] / row["spike_expected_rhythm"]
    row["spike_z"] = float((obs_sp - yhat[sp].sum()) / np.sqrt(max(yhat[sp].sum(), 1e-9)))
    rows.append(row)
acc = pd.DataFrame(rows).merge(sched, on="account", how="left")
acc.to_csv(T2 / "shared_shock_accounts.csv", index=False)
print("accounts", len(acc), round(time.time() - t0), "s", flush=True)


# ---- group summaries
def boot_ci(v, stat=np.median, n=1000):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 5:
        return np.nan, np.nan
    idx = rng.integers(0, len(v), size=(n, len(v)))
    s = stat(v[idx], axis=1)
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def summarize(d, label, gtype):
    out = dict(group_type=gtype, group=label, accounts=len(d), events=int(d.n_events.sum()))
    for col in ("corr_hourly", "corr_daily", "corr_within_day", "r2_c", "r2_lag1", "r2_both", "spike_ratio_rhythm",
                "spike_ratio_naive", "spike_share"):
        v = d[col].dropna()
        out[f"{col}_median"] = float(v.median()) if len(v) else np.nan
        lo, hi = boot_ci(v)
        out[f"{col}_lo"], out[f"{col}_hi"] = lo, hi
    out["spike_pooled_share"] = float(d.spike_events.sum() / d.n_events.sum())
    out["spike_pooled_expected_rhythm"] = float(d.spike_expected_events.sum() / d.n_events.sum())
    out["spike_pooled_ratio"] = out["spike_pooled_share"] / out["spike_pooled_expected_rhythm"]
    out["spike_pooled_expected_naive"] = float((d.spike_expected_naive * d.n_events).sum() / d.n_events.sum())
    out["share_corr_hourly_gt_0"] = float((d.corr_hourly > 0).mean())
    return out


acc["party3"] = acc.party.where(acc.party.isin(["D", "R"]), "other")
gl = [summarize(acc, "all", "all")]
for p_, d in acc.groupby("party3"):
    gl.append(summarize(d, p_, "party"))
for p_, d in acc.groupby("account_type"):
    gl.append(summarize(d, p_, "account_type"))
for (p_, t_), d in acc.groupby(["party3", "account_type"]):
    if len(d) >= 10:
        gl.append(summarize(d, f"{p_}|{t_}", "party_x_type"))
for p_, d in acc.groupby("chamber"):
    gl.append(summarize(d, p_, "chamber"))
for p_, d in acc.groupby(acc.flag_minute.fillna(False)):
    gl.append(summarize(d, f"scheduler_minute_flag={bool(p_)}", "scheduler"))
grp = pd.DataFrame(gl)
grp.to_csv(T2 / "shared_shock_groups.csv", index=False)
# volume-adjusted: partial Spearman of corr_hourly with log events
from scipy.stats import spearmanr  # noqa: E402

rho, pv = spearmanr(acc.n_events, acc.corr_hourly, nan_policy="omit")
summ = dict(accounts=int(len(acc)), spearman_corr_hourly_vs_events=float(rho),
            corr_hourly_quartiles=[float(x) for x in acc.corr_hourly.quantile([.1, .25, .5, .75, .9])],
            spike_ratio_rhythm_quartiles=[float(x) for x in acc.spike_ratio_rhythm.quantile([.1, .25, .5, .75, .9])],
            accounts_spike_ratio_gt_1=float((acc.spike_ratio_rhythm > 1).mean()),
            accounts_spike_z_gt_3=float((acc.spike_z > 3).mean()), elapsed_s=round(time.time() - t0))
(T2 / "shared_shock_summary.json").write_text(json.dumps(summ, indent=2))
pd.set_option("display.width", 250)
print(grp[["group_type", "group", "accounts", "corr_hourly_median", "corr_hourly_lo", "corr_hourly_hi",
           "corr_daily_median", "corr_within_day_median", "r2_c_median", "r2_lag1_median", "r2_both_median",
           "spike_ratio_rhythm_median", "spike_ratio_rhythm_lo", "spike_ratio_rhythm_hi", "spike_pooled_share",
           "spike_pooled_expected_rhythm", "spike_pooled_ratio"]].round(3).to_string())
print(json.dumps(summ, indent=1))
