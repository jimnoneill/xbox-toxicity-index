"""T6 tasks 1 and 2: collection completeness, weekly/daily volume with regimes, NB regressions, PELT changepoints,
event responses, Figure 1.
Outputs (output/v7/t6/): completeness.json, zero_days_personal.csv, gaps_personal.csv, weekly_volume.csv, daily_volume.csv,
  nb_rate_ratios.csv, pelt.json, event_response.csv, fig1_weekly_volume.png, volume_numbers.json
NB model: GLM NegativeBinomial (NB2), alpha estimated by ML (discrete NB), HAC (Newey-West, 7 lags) covariance.
  log mean = flags + day-of-week + natural cubic spline of time (df=5). Flags are associations, not causal effects.
PELT: mean-shift cost on log(1 + weekly count), Gaussian, sigma^2 from the MAD of first differences / sqrt(2),
  penalty = 2 ln(n) (standardised units, BIC-type for a change plus a mean), min segment 4 weeks; sensitivity at 0.5x and 2x.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from t6_common import *  # noqa
import warnings
import statsmodels.api as sm
from patsy import dmatrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
warnings.filterwarnings("ignore")

out = {}
reg, ev = regime_frame()
idx = reg.index
P = load_posts("personal")
O = load_posts("official")
NR_P = P.type != "retweet"

# ------------------------------------------------------------------ task 1: completeness
cnt = pd.read_csv(ROOT / "data/lee_counts_by_day_20261007.csv")
cnt["d"] = pd.to_datetime(cnt.start).dt.tz_localize(None).dt.normalize()
P["utc_date"] = P.created_at.dt.tz_localize(None).dt.normalize()
arch = P.groupby("utc_date").size().reindex(cnt.d, fill_value=0)
m = pd.DataFrame({"d": cnt.d.values, "api": cnt.tweet_count.values, "arch": arch.values})
m["deficit"] = m.api - m.arch
zero = m[m.arch == 0]
comp = dict(
    utc_days=int(len(m)), api_total=int(m.api.sum()), archive_total=int(m.arch.sum()),
    archive_zero_days=int((m.arch == 0).sum()), counts_zero_days=int((m.api == 0).sum()),
    both_zero_days=int(((m.arch == 0) & (m.api == 0)).sum()),
    archive_zero_but_counts_positive=int(((m.arch == 0) & (m.api > 0)).sum()),
    days_archive_below_counts=int((m.deficit > 0).sum()), max_deficit_posts_one_day=int(m.deficit.max()),
    total_deficit_posts=int(m.deficit.clip(lower=0).sum()),
    days_archive_above_counts=int((m.deficit < 0).sum()), total_excess_posts=int((-m.deficit).clip(lower=0).sum()),
    note_excess="Archive exceeds the counts endpoint on 63 days, concentrated in 2024-11 and 2025-03; consistent with posts deleted after collection (the counts endpoint reflects the present).")
zd = zero.copy()
zd["year"] = zd.d.dt.year
zd.to_csv(T6 / "zero_days_personal.csv", index=False)
comp["zero_days_by_year"] = {int(k): int(v) for k, v in zd.groupby("year").size().items()}
zr = reg.reindex(zd.d)
comp["zero_days_in_own_race_2022"] = int(zr.own_race_2022.sum())
comp["zero_days_in_session"] = int(zr.in_session.sum())
comp["zero_days_in_recess"] = int(zr.recess.sum())
tt = P.t.values
gap_h = np.diff(tt) / 3600
order = np.argsort(gap_h)[::-1][:10]
rows = []
for i in order:
    a, b = pd.Timestamp(tt[i], unit="s"), pd.Timestamp(tt[i + 1], unit="s")
    inside = m[(m.d > a.normalize()) & (m.d < b.normalize())]
    edge = m[(m.d == a.normalize()) | (m.d == b.normalize())]
    rows.append(dict(start_utc=a, end_utc=b, gap_hours=gap_h[i], full_utc_days_inside=len(inside),
                     counts_posts_inside_full_days=int(inside.api.sum()), archive_posts_inside=int(inside.arch.sum()),
                     counts_on_two_edge_days=int(edge.api.sum()), archive_on_edge_days=int(edge.arch.sum())))
gp = pd.DataFrame(rows)
gp.to_csv(T6 / "gaps_personal.csv", index=False)
comp["longest_gap"] = {k: (str(v) if isinstance(v, pd.Timestamp) else v) for k, v in gp.iloc[0].to_dict().items()}
comp["gaps_over_72h"] = int((gap_h > 72).sum())
comp["top10_gaps_with_counts_posts_inside_full_days"] = int((gp.counts_posts_inside_full_days > 0).sum())
od = daily_counts(O)
og = np.diff(O.t.values) / 3600
comp["official_zero_days_ET_window"] = int((od == 0).sum())
comp["official_longest_gap_hours"] = float(og.max())
comp["official_verification"] = "monthly counts only (earlier check: 47 of 54 months exact, 7 short by one post); daily silences not verifiable"
out["completeness"] = comp
wjson("completeness.json", comp)

# ------------------------------------------------------------------ task 2: volume
d_p = daily_counts(P)
d_pn = daily_counts(P, mask=NR_P)
d_o = daily_counts(O)
d_on = daily_counts(O, mask=(O.type != "retweet"))
dv = pd.DataFrame({"personal_all": d_p, "personal_nonrt": d_pn, "official_all": d_o, "official_nonrt": d_on})
dv = dv.join(reg)
dv.to_csv(T6 / "daily_volume.csv", index_label="date")
wk = pd.DataFrame({c: weekly_counts(dv[c]) for c in ["personal_all", "personal_nonrt", "official_all", "official_nonrt"]})
wk.to_csv(T6 / "weekly_volume.csv")
out["volume"] = dict(
    personal_posts_window=int(d_p.sum()), official_posts_window=int(d_o.sum()),
    personal_per_day_by_year={int(y): float(g.mean()) for y, g in d_p.groupby(d_p.index.year)},
    official_per_day_by_year={int(y): float(g.mean()) for y, g in d_o.groupby(d_o.index.year)},
    ratio_personal_to_official=float(d_p.sum() / d_o.sum()),
    weekly_corr_personal_official_log=float(np.corrcoef(np.log1p(wk.personal_all), np.log1p(wk.official_all))[0, 1]),
    weeks=int(len(wk)))

FLAGS = ["in_session", "lee_voted", "recess", "shutdown", "own_race_2022", "post_2022", "natl_general_2024", "post_2024",
         "inaug_2wk", "r_president", "event_day"]
LABEL = {"in_session": "Senate roll-call day", "lee_voted": "Lee cast a vote (extra)", "recess": "Senate recess block",
         "shutdown": "Federal shutdown", "own_race_2022": "Own 2022 general window (Sep 5-Nov 8)",
         "post_2022": "2 weeks after own 2022 election", "natl_general_2024": "2024 general window (Labor Day-Nov 5)",
         "post_2024": "2 weeks after 2024 election", "inaug_2wk": "Inauguration 2 weeks (from 2025-01-20)",
         "r_president": "Republican president (2025-01-20 on)", "event_day": "Curated event day or next day"}


def design(spline_df=5, flags=FLAGS):
    X = reg[flags].astype(float).copy()
    dow = pd.get_dummies(reg.dow, prefix="dow", drop_first=True).astype(float)
    sp = dmatrix(f"cr(tday, df={spline_df}) - 1", {"tday": reg.tday.values}, return_type="dataframe")
    sp.index = reg.index
    sp.columns = [f"s{i}" for i in range(sp.shape[1])]
    return pd.concat([X, dow, sp], axis=1)


def fit_nb(y, X):
    """NB2 GLM; alpha by profile maximum likelihood (bounded search over log alpha), HAC covariance at the optimum."""
    from scipy.optimize import minimize_scalar
    Xv = np.asarray(X, dtype=float)

    def nll(la):
        return -sm.GLM(y, Xv, family=sm.families.NegativeBinomial(alpha=np.exp(la))).fit(maxiter=200).llf
    la = minimize_scalar(nll, bounds=(-6, 3), method="bounded", options=dict(xatol=1e-3)).x
    alpha = float(np.exp(la))
    glm = sm.GLM(y, Xv, family=sm.families.NegativeBinomial(alpha=alpha)).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
    return glm, alpha


rr_rows = []
for acct, y in [("personal", d_p), ("official", d_o), ("personal_nonrt", d_pn), ("official_nonrt", d_on)]:
    for tag, kw in [("main", dict(spline_df=5)), ("spline3", dict(spline_df=3)), ("spline8", dict(spline_df=8)),
                    ("no_r_president", dict(spline_df=5, flags=[f for f in FLAGS if f != "r_president"]))]:
        if tag != "main" and acct.endswith("nonrt"):
            continue
        X = design(**kw)
        g, alpha = fit_nb(y.values, X)
        ci = g.conf_int()
        ci = pd.DataFrame(np.asarray(ci), index=X.columns)
        par = pd.Series(np.asarray(g.params), index=X.columns)
        pv = pd.Series(np.asarray(g.pvalues), index=X.columns)
        for f in X.columns:
            if f in (kw.get("flags") or FLAGS) or f.startswith("dow_"):
                rr_rows.append(dict(account=acct, spec=tag, term=f, rate_ratio=float(np.exp(par[f])),
                                    lo=float(np.exp(ci.loc[f, 0])), hi=float(np.exp(ci.loc[f, 1])),
                                    p=float(pv[f]), alpha=alpha, n_days=len(y)))
rr = pd.DataFrame(rr_rows)
rr["label"] = rr.term.map(LABEL).fillna(rr.term)
rr.to_csv(T6 / "nb_rate_ratios.csv", index=False)
out["nb_main"] = {a: {r.term: [round(r.rate_ratio, 3), round(r.lo, 3), round(r.hi, 3), round(r.p, 4)]
                      for r in rr[(rr.account == a) & (rr.spec == "main") & rr.term.isin(FLAGS)].itertuples()}
                  for a in ["personal", "official"]}
out["nb_alpha"] = {a: float(rr[(rr.account == a) & (rr.spec == "main")].alpha.iloc[0]) for a in ["personal", "official"]}


# ------------------------------------------------------------------ PELT
def pelt_mean(x, pen, minsize=4):
    n = len(x)
    c1 = np.concatenate([[0], np.cumsum(x)])
    c2 = np.concatenate([[0], np.cumsum(x * x)])

    def cost(a, b):
        s = c1[b] - c1[a]
        return c2[b] - c2[a] - s * s / (b - a)
    F = np.full(n + 1, np.inf)
    F[0] = -pen
    last = np.zeros(n + 1, dtype=int)
    R = [0]
    for t in range(minsize, n + 1):
        cand = [(F[s] + cost(s, t) + pen, s) for s in R if t - s >= minsize and np.isfinite(F[s])]
        if not cand:
            R.append(t - minsize + 1)
            continue
        F[t], last[t] = min(cand)
        R = [s for s in R if (not np.isfinite(F[s])) or F[s] + cost(s, t) <= F[t]]
        R.append(t - minsize + 1)
    cps, t = [], n
    while t > 0:
        s = last[t]
        if s > 0:
            cps.append(s)
        t = s
    return sorted(cps)


pel = {}
for acct, col in [("personal", "personal_all"), ("official", "official_all")]:
    x = np.log1p(wk[col].values.astype(float))
    sig = np.median(np.abs(np.diff(x) - np.median(np.diff(x)))) * 1.4826 / np.sqrt(2)
    z = x / sig
    n = len(z)
    res = {}
    for mult in (0.5, 1.0, 2.0):
        pen = mult * 2 * np.log(n)
        cps = pelt_mean(z, pen)
        res[f"pen_x{mult}"] = dict(penalty=float(pen), n_changepoints=len(cps),
                                   weeks_start=[str(wk.index[c].date()) for c in cps])
    pel[acct] = dict(sigma_hat_log1p=float(sig), n_weeks=n, **res)
    cps = pelt_mean(z, 2 * np.log(n))
    b = [0] + cps + [n]
    pel[acct]["segments_primary"] = [dict(start=str(wk.index[b[i]].date()), end=str(wk.index[b[i + 1] - 1].date()),
                                          mean_weekly_posts=float(wk[col].values[b[i]:b[i + 1]].mean())) for i in range(len(b) - 1)]
    al = []
    for c in cps:
        d = wk.index[c]
        e2 = ev.assign(off=(ev.date - d).dt.days)
        e2 = e2[(e2.off.abs() <= 14)]
        al.append(dict(week=str(d.date()), events_within_14d=[f"{r.date.date()} {r.label}" for r in e2.itertuples()]))
    pel[acct]["alignment_primary"] = al
wjson("pelt.json", pel)
out["pelt"] = {a: {"n_primary": pel[a]["pen_x1.0"]["n_changepoints"], "n_half": pel[a]["pen_x0.5"]["n_changepoints"],
                   "n_double": pel[a]["pen_x2.0"]["n_changepoints"], "weeks_primary": pel[a]["pen_x1.0"]["weeks_start"]} for a in pel}


# ------------------------------------------------------------------ event response
def event_resp(d, name):
    rows = []
    v = d.values.astype(float)

    def ratio(i):
        pre = v[max(0, i - 28):i]
        if len(pre) < 14 or i + 1 >= len(v):
            return np.nan
        return (v[i] + v[i + 1]) / 2 / max(pre.mean(), 1e-9)
    allr = np.array([ratio(i) for i in range(len(v))])
    allr = allr[~np.isnan(allr)]
    for e in ev.itertuples():
        i = int((e.date - d.index[0]).days)
        r = ratio(i)
        rows.append(dict(account=name, date=e.date.date(), label=e.label, category=e.category, ratio_2day_vs_prior28=r,
                         pctile_vs_all_days=float((allr < r).mean()) if r == r else np.nan,
                         in_session_that_day=int(reg.in_session.iloc[i])))
    return rows


er = pd.DataFrame(event_resp(d_p, "personal") + event_resp(d_o, "official"))
er.to_csv(T6 / "event_response.csv", index=False)
out["events"] = {a: dict(n=int(len(g)), median_ratio=float(g.ratio_2day_vs_prior28.median()),
                         share_above_1=float((g.ratio_2day_vs_prior28 > 1).mean()),
                         median_pctile=float(g.pctile_vs_all_days.median()),
                         n_pctile_ge_90=int((g.pctile_vs_all_days >= 0.9).sum()),
                         top5=g.sort_values("ratio_2day_vs_prior28", ascending=False).head(5)[["date", "label", "ratio_2day_vs_prior28"]].astype(str).values.tolist())
                 for a, g in er.groupby("account")}

# ------------------------------------------------------------------ Figure 1
ev_short = ev.reset_index(drop=True)
fig = plt.figure(figsize=(17, 10.5))
gs = fig.add_gridspec(2, 2, width_ratios=[3.2, 1], hspace=0.12, wspace=0.02)
axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])]
axk = fig.add_subplot(gs[:, 1])
axk.axis("off")
rec = reg.recess.values
blocks, st = [], None
for i, v in enumerate(rec):
    if v and st is None:
        st = i
    if (not v) and st is not None:
        blocks.append((idx[st], idx[i - 1]))
        st = None
if st is not None:
    blocks.append((idx[st], idx[-1]))
r22 = reg.index[reg.own_race_2022 == 1]
for ax, col, lab, c in [(axs[0], "personal_all", "@BasedMikeLee (personal), posts per week", PAL["personal"]),
                        (axs[1], "official_all", "@SenMikeLee (official), posts per week", PAL["official"])]:
    for a, b in blocks:
        if (b - a).days >= 6:
            ax.axvspan(a, b + pd.Timedelta(days=1), color="#d9d9d9", alpha=0.55, lw=0)
    ax.axvspan(r22[0], r22[-1], color=PAL["yellow"], alpha=0.30, lw=0)
    ax.axvspan(pd.Timestamp("2024-09-02"), pd.Timestamp("2024-11-05"), color=PAL["sky"], alpha=0.30, lw=0)
    ax.axvline(pd.Timestamp("2025-01-20"), color="k", lw=1.2, ls="-")
    ax.plot(wk.index, wk[col], color=c, lw=1.4)
    ax.set_ylabel(lab, fontsize=9)
    ax.grid(alpha=0.25)
    acct = "personal" if col.startswith("personal") else "official"
    for k in pel[acct]["pen_x1.0"]["weeks_start"]:
        ax.axvline(pd.Timestamp(k), color=PAL["purple"], ls="--", lw=1.0)
ymax = axs[0].get_ylim()[1]
for i, e in ev_short.iterrows():
    axs[0].annotate(str(i + 1), (e.date, ymax * (0.97 - 0.07 * (i % 3))), fontsize=6.5, ha="center", color="#333",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.8))
    axs[1].axvline(e.date, color="#888", lw=0.35, alpha=0.6)
axs[0].set_xticklabels([])
axs[1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
axs[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
axs[0].set_xlim(wk.index[0], wk.index[-1] + pd.Timedelta(days=7))
axs[1].set_xlim(axs[0].get_xlim())
axs[0].set_title("Sen. Mike Lee, weekly posts 2022-2026 (all post types). Grey: Senate recess blocks; amber: own 2022 general window; "
                 "blue: 2024 general window;\nblack line: inauguration 2025-01-20; dashed purple: PELT changepoints (penalty 2 ln n); "
                 "numbers: curated events (key at right)", fontsize=9)
key = "\n".join(f"{i + 1:>2} {e.date.date()} {e.label[:44]}" for i, e in ev_short.iterrows())
axk.text(0, 1.0, key, fontsize=6.3, va="top", family="monospace")
fig.savefig(T6 / "fig1_weekly_volume.png", dpi=170, bbox_inches="tight")
plt.close(fig)
wjson("volume_numbers.json", out)
print(json.dumps(out, indent=1, default=float)[:7000])
