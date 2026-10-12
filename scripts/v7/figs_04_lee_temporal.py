"""Fig 8 (Lee case study) and Fig 9 (point-process model comparison)."""
from figs_common import *
style()

DIMS = ["offensive", "toxic", "neg_sentiment", "irony", "ragebait", "tribal", "performative"]
# ------------------------------------------------------------------ Fig 8
wk = pd.read_csv(V7 + "/t6/weekly_volume.csv", parse_dates=["week"])
wk = wk[wk.week <= pd.Timestamp("2026-09-28")]  # last week is partial (archive ends 2026-10-05)
pp = pd.read_parquet(V7 + "/ti/post_probs_lee.parquet")
pp["id"] = pp.id.astype(str)
arch = pd.read_parquet(DATA + "/BasedMikeLee_archive_v2.parquet", columns=["id", "created_at", "type"])
arch["id"] = arch.id.astype(str)
P = arch.merge(pp[["id"] + [f"p_{d}" for d in DIMS]], on="id", how="left")
print("personal types", P.type.value_counts().to_dict(), "scored", P.p_offensive.notna().mean())
off = pd.read_parquet(DATA + "/SenMikeLee_2022_2026.parquet", columns=["id", "created_at", "type"])
for d in DIMS:
    off[f"p_{d}"] = np.load(V7 + f"/t6/_probs_official/{d}.npy")
print("official types", off.type.value_counts().to_dict())


def weekly_ti7(D):
    D = D[~D.type.isin(["retweeted", "retweet"])].copy()
    t = pd.to_datetime(D.created_at, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    D["week"] = (t.dt.normalize() - pd.to_timedelta(t.dt.dayofweek, unit="D"))
    D["ti7"] = D[[f"p_{d}" for d in DIMS]].mean(axis=1)
    return D.dropna(subset=["ti7"]).groupby("week").ti7.agg(["mean", "size", "std"]).reset_index()


wp, wo = weekly_ti7(P), weekly_ti7(off)
mo = pd.read_csv(V7 + "/t6/content_monthly.csv")
mp = mo[mo.account == "personal"].set_index("month")
print("personal monthly TI7 (file) vs recomputed overall", mp.TI7.mean(), wp["mean"].mean())
win = pd.read_csv(V7 + "/t3/windows.csv")
lw = win[(win.name == "Mike Lee") & (win.cycle == 2022)].iloc[0]
g0, g1 = pd.Timestamp(lw.general_window_start), pd.Timestamp(lw.general_window_end)
print("lee general window", g0, g1)
MIN_N = 5
fig = plt.figure(figsize=(W2, 5.0))
gsp = fig.add_gridspec(2, 1, height_ratios=[2.3, 1], hspace=0.08)
ax = fig.add_subplot(gsp[0]); ax2 = fig.add_subplot(gsp[1], sharex=ax)
ax.plot(wk.week, wk.official_all, color=ORANGE, lw=1.0, label="Official account")
ax.plot(wk.week, wk.personal_all, color=BLUE, lw=1.0, label="Personal account")
ax.set_yscale("symlog", linthresh=1)
ax.set_ylim(0, 6000); ax.set_yticks([0, 1, 10, 100, 1000]); ax.set_yticklabels(["0", "1", "10", "100", "1000"])
ax.set_ylabel("Posts per week")
ev = [(g0, "2022 general-election window"), (pd.Timestamp("2024-11-05"), "2024 election"), (pd.Timestamp("2025-01-20"), "Inauguration")]
for a in (ax, ax2):
    a.axvspan(g0, g1, color=GREY, alpha=0.18, lw=0)
    for d in ev[1:]:
        a.axvline(d[0], color=BLACK, lw=0.8, ls="--")
ax.text(g0 + (g1 - g0) / 2, 4000, "2022 general\nelection window", ha="center", va="top", fontsize=7.5)
ax.text(ev[1][0] - pd.Timedelta(days=10), 4000, "Nov 2024\nelection", ha="right", va="top", fontsize=7.5)
ax.text(ev[2][0] + pd.Timedelta(days=10), 4000, "Jan 2025\ninauguration", ha="left", va="top", fontsize=7.5)
ax.legend(loc="lower left", ncol=2, fontsize=8)
import matplotlib.dates as mdates
plt.setp(ax.get_xticklabels(), visible=False)
for w_, c in [(wo, ORANGE), (wp, BLUE)]:
    k = w_[w_["size"] >= MIN_N]
    ax2.plot(k.week, k["mean"], "o-", color=c, ms=2.2, lw=0.6, alpha=0.9)
ax2.xaxis.set_major_locator(mdates.YearLocator()); ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax2.set_ylabel("Weekly mean TI7")
ax2.set_xlim(wk.week.min() - pd.Timedelta(days=7), wk.week.max() + pd.Timedelta(days=14))
ax2.text(0.99, 0.03, f"weeks with at least {MIN_N} non-retweet posts", transform=ax2.transAxes, ha="right", va="bottom", fontsize=7, color=GREY)
save(fig, "fig8_lee_case_study")

# ------------------------------------------------------------------ Fig 9
sm = pd.read_csv(V7 + "/temporal/model_compare/summary_by_model.csv")
mm = pd.read_csv(V7 + "/temporal/model_compare/members_models.csv")
print(sm[["model", "n", "bic_best_share", "aic_best_share", "med_k"]].to_string())
lab = {"casc3": "Cascade (3 levels)", "hawkes_soe": "Hawkes, sum of exponentials", "hawkes_exp": "Hawkes, exponential", "mmpp3c": "MMPP-3, clock", "mmpp3": "MMPP-3",
       "renc_he3": "Renewal, hyperexp.-3, clock", "casc2": "Cascade (2 levels)", "renc_he2": "Renewal, hyperexp.-2, clock", "ren_he2": "Renewal, hyperexp.-2", "ren_gm": "Renewal, gamma", "ren_wb": "Renewal, Weibull", "renc_wb": "Renewal, Weibull, clock", "mmpp2": "MMPP-2", "mmpp2c": "MMPP-2, clock", "nhpp_hw": "Inhomogeneous Poisson", "ren_he3": "Renewal, hyperexp.-3", "renc_gm": "Renewal, gamma, clock", "renc_ln": "Renewal, lognormal, clock", "ren_ln": "Renewal, lognormal"}
s2 = sm.sort_values("bic_best_share", ascending=True)
fig, axs = plt.subplots(1, 2, figsize=(W2, 3.6), gridspec_kw={"width_ratios": [1.1, 1], "wspace": 0.55})
ax = axs[0]
cols = [BLUE if m == "casc3" else ORANGE if m.startswith("hawkes") else GREY for m in s2.model]
ax.barh(range(len(s2)), 100 * s2.bic_best_share, color=cols)
ax.set_yticks(range(len(s2))); ax.set_yticklabels([lab.get(m, m) for m in s2.model], fontsize=7)
ax.set_xlabel(f"Members best fit by BIC (%), n = {int(s2.n.max())}")
plab(ax, "a", dx=-0.8)
ax = axs[1]
piv = mm.pivot_table(index="bg", columns="model", values="aic")
for other, c, l in [("hawkes_exp", ORANGE, "vs Hawkes, exponential"), ("hawkes_soe", GREEN, "vs Hawkes, sum of exponentials")]:
    d = (piv["casc3"] - piv[other]).dropna().sort_values().values
    ax.plot(d, np.arange(1, len(d) + 1) / len(d), color=c, label=l)
    print("fig9", other, "n", len(d), "share casc3 lower AIC", (d < 0).mean(), "median", np.median(d))
ax.axvline(0, color="black", lw=0.7)
ax.set_xscale("symlog", linthresh=10)
ax.set_xlabel("AIC(cascade 3) minus AIC(Hawkes)")
ax.set_ylabel("Cumulative share of members")
ax.set_xticks([-1000, -10, 0, 10, 1000]); ax.set_xticklabels(["-1000", "-10", "0", "10", "1000"])
ax.legend(loc="lower right", fontsize=7.5)
plab(ax, "b", dx=-0.2)
save(fig, "fig9_point_process_comparison")
