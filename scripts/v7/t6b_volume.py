"""T6b: Lee case study, review fixes (M7 and M8). New file; T6 outputs are untouched.

1. January 2025 transition. The Senate majority change (2025-01-03) and the inauguration (2025-01-20) are 17 days apart and the
   personal account trends upward. The model now has ONE step indicator `jan2025_transition` (1 from 2025-01-03 on) and no
   inaug_2wk flag. The presidency and the Senate majority change cannot be separated in these data: the step is an association
   with the whole January 2025 transition. Sensitivity: spline df 3, 5, 8, 12, 16; step at 2025-01-20 instead; both steps in one
   model (collinearity shown); no transition term; window restricted to 2024-01-01 on.
2. Own 2022 general window. The account started 2022-07-24 and was nearly silent for its first 15 weeks, so the window cannot be
   separated from start-up. Reported descriptively (daily counts before, during, after; confirmed zero-count stretches). Models
   restarted on 2022-11-01 and 2023-01-01 (the window flag no longer applies and is dropped). The full-series rate ratio is shown
   across spline df together with a split of the window into the confirmed silence and the rest of the window; it is not
   presented as a stable effect.
Model: GLM NegativeBinomial (NB2), alpha by profile ML, HAC (7 lags); log mean = flags + day-of-week + natural cubic spline of time.
Outputs (output/v7/t6b/): nb_rate_ratios.csv, descriptive_2022.csv, zero_runs_2022.csv, weekly_2022.csv, transition_context.csv,
numbers_t6b.json
"""
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t6_common import *  # noqa
import statsmodels.api as sm
from patsy import dmatrix
from scipy.optimize import minimize_scalar

warnings.filterwarnings("ignore")
T6B = ROOT / "output/v7/t6b"
T6B.mkdir(parents=True, exist_ok=True)

reg, ev = regime_frame()
idx = reg.index
P = load_posts("personal")
O = load_posts("official")
d_p = daily_counts(P)
d_o = daily_counts(O)

TRANS = pd.Timestamp("2025-01-03")   # Senate majority changes to R
INAUG = pd.Timestamp("2025-01-20")   # inauguration
reg["jan2025_transition"] = (idx >= TRANS).astype(int)
reg["step_0120"] = (idx >= INAUG).astype(int)
reg["step_0103"] = reg["jan2025_transition"]
ra, rb = reg.step_0103.values, reg.step_0120.values
corr_steps = float(np.corrcoef(ra, rb)[0, 1])
# silence = the longest zero run of ET days inside the own 2022 window
BASE = ["in_session", "lee_voted", "recess", "shutdown", "own_race_2022", "post_2022", "natl_general_2024", "post_2024",
        "event_day"]
MAIN = BASE + ["jan2025_transition"]
LABEL = {"in_session": "Senate roll-call day", "lee_voted": "Lee cast a vote (extra)", "recess": "Senate recess block",
         "shutdown": "Federal shutdown", "own_race_2022": "Own 2022 general window (Sep 5-Nov 8)",
         "post_2022": "2 weeks after own 2022 election", "natl_general_2024": "2024 general window (Labor Day-Nov 5)",
         "post_2024": "2 weeks after 2024 election", "jan2025_transition": "January 2025 transition (step from 2025-01-03)",
         "step_0120": "Step from 2025-01-20 (inauguration)", "step_0103": "Step from 2025-01-03 (Senate majority change)",
         "event_day": "Curated event day or next day", "own_race_silence": "Own 2022 window: confirmed zero-count stretch",
         "own_race_rest": "Own 2022 window: rest of window"}

# zero runs (ET days) of the personal account
z = (d_p == 0).values
runs, st = [], None
for i, v in enumerate(z):
    if v and st is None:
        st = i
    if (not v) and st is not None:
        runs.append((st, i - 1))
        st = None
if st is not None:
    runs.append((st, len(z) - 1))
cnt = pd.read_csv(ROOT / "data/lee_counts_by_day_20261007.csv")
cnt["d"] = pd.to_datetime(cnt.start).dt.tz_localize(None).dt.normalize()
zr = []
for a, b in runs:
    if b - a + 1 < 5:
        continue
    da, db = idx[a], idx[b]
    inside = cnt[(cnt.d > da) & (cnt.d < db)]  # UTC days strictly inside the ET run
    zr.append(dict(start=str(da.date()), end=str(db.date()), et_days=int(b - a + 1),
                   in_own_window=bool(reg.own_race_2022.iloc[a] == 1 and reg.own_race_2022.iloc[b] == 1),
                   counts_endpoint_posts_in_full_utc_days_inside=int(inside.tweet_count.sum()),
                   full_utc_days_inside=int(len(inside))))
zrd = pd.DataFrame(zr)
zrd.to_csv(T6B / "zero_runs_2022.csv", index=False)
win = reg.own_race_2022 == 1
longest_in_win = zrd[zrd.in_own_window].sort_values("et_days", ascending=False).iloc[0]
s0, s1 = pd.Timestamp(longest_in_win.start), pd.Timestamp(longest_in_win.end)
reg["own_race_silence"] = ((idx >= s0) & (idx <= s1)).astype(int)
reg["own_race_rest"] = ((reg.own_race_2022 == 1) & (reg.own_race_silence == 0)).astype(int)

# ---------------------------------------------------------------- descriptive 2022 (item 8)
periods = [("before_window (2022-07-25..2022-09-04)", "2022-07-25", "2022-09-04"),
           ("window (2022-09-05..2022-11-08)", "2022-09-05", "2022-11-08"),
           ("after_window_2022 (2022-11-09..2022-12-31)", "2022-11-09", "2022-12-31"),
           ("2023-H1 (2023-01-01..2023-06-30)", "2023-01-01", "2023-06-30")]
rows = []
for nm, a, b in periods:
    for acct, d in (("personal", d_p), ("official", d_o)):
        x = d[a:b]
        rows.append(dict(account=acct, period=nm, days=int(len(x)), posts=int(x.sum()), mean_per_day=float(x.mean()),
                         median_per_day=float(x.median()), max_per_day=int(x.max()), zero_days=int((x == 0).sum()),
                         share_zero_days=float((x == 0).mean())))
dd = pd.DataFrame(rows)
dd.to_csv(T6B / "descriptive_2022.csv", index=False)
wk22 = d_p["2022-07-25":"2023-01-01"].resample("W-SUN").sum()
wk22o = d_o["2022-07-25":"2023-01-01"].resample("W-SUN").sum()
pd.DataFrame({"week_ending": wk22.index.date, "personal_posts": wk22.values, "official_posts": wk22o.values}).to_csv(
    T6B / "weekly_2022.csv", index=False)
first_post = pd.Timestamp(P.created_at.min()).tz_localize(None) if P.created_at.min().tzinfo is None else P.created_at.min().tz_convert(None)
desc22 = dict(
    first_personal_post_utc=str(first_post), personal_weeks_before_window_posts=[int(v) for v in wk22[:6].values],
    window_days=int(win.sum()), window_zero_days=int((d_p[win] == 0).sum()),
    window_share_zero_days=float((d_p[win] == 0).mean()),
    longest_zero_run_in_window=dict(start=longest_in_win.start, end=longest_in_win.end, et_days=int(longest_in_win.et_days),
                                    counts_endpoint_posts_in_full_utc_days_inside=int(longest_in_win.counts_endpoint_posts_in_full_utc_days_inside)),
    longest_run_share_of_window_days=float(longest_in_win.et_days / win.sum()),
    longest_run_share_of_window_zero_days=float(longest_in_win.et_days / max(int((d_p[win] == 0).sum()), 1)),
    all_zero_runs_5plus_days=zrd.to_dict("records"),
    periods=dd.round(4).to_dict("records"),
    note=("The account posted 1-8 times per week for its first 15 weeks (apart from one 32-post week in August) and its volume "
          "rose from mid-November 2022, so the own-window association cannot be separated from account start-up."))

# ---------------------------------------------------------------- NB models


def design(flags, spline_df, start=None, extra=None):
    X = reg[flags].astype(float).copy()
    if extra is not None:
        for k, v in extra.items():
            X[k] = v
    dow = pd.get_dummies(reg.dow, prefix="dow", drop_first=True).astype(float)
    sp = dmatrix(f"cr(tday, df={spline_df}) - 1", {"tday": reg.tday.values}, return_type="dataframe")
    sp.index = reg.index
    sp.columns = [f"s{i}" for i in range(sp.shape[1])]
    D = pd.concat([X, dow, sp], axis=1)
    if start is not None:
        D = D.loc[start:]
    keep = [c for c in D.columns if D[c].std() > 0 or c.startswith("s")]
    return D[keep]


def fit_nb(y, X):
    Xv = np.asarray(X, dtype=float)

    def nll(la):
        return -sm.GLM(y, Xv, family=sm.families.NegativeBinomial(alpha=np.exp(la))).fit(maxiter=200).llf
    la = minimize_scalar(nll, bounds=(-6, 3), method="bounded", options=dict(xatol=1e-3)).x
    alpha = float(np.exp(la))
    glm = sm.GLM(y, Xv, family=sm.families.NegativeBinomial(alpha=alpha)).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
    return glm, alpha


def spec_list():
    s = []
    for df in (3, 5, 8, 12, 16):
        s.append(dict(spec=f"main_df{df}", flags=MAIN, df=df))
    s.append(dict(spec="step_0120_df5", flags=BASE + ["step_0120"], df=5))
    s.append(dict(spec="step_0120_df8", flags=BASE + ["step_0120"], df=8))
    s.append(dict(spec="both_steps_df5", flags=BASE + ["step_0103", "step_0120"], df=5))
    s.append(dict(spec="both_steps_df8", flags=BASE + ["step_0103", "step_0120"], df=8))
    s.append(dict(spec="no_transition_df5", flags=BASE, df=5))
    s.append(dict(spec="window_2024on_df5", flags=["in_session", "lee_voted", "recess", "shutdown", "natl_general_2024",
                                                   "post_2024", "event_day", "jan2025_transition"], df=5, start="2024-01-01"))
    s.append(dict(spec="window_2024on_df3", flags=["in_session", "lee_voted", "recess", "shutdown", "natl_general_2024",
                                                   "post_2024", "event_day", "jan2025_transition"], df=3, start="2024-01-01"))
    s.append(dict(spec="start_20221101_df5", flags=[f for f in MAIN if f != "own_race_2022"], df=5, start="2022-11-01"))
    s.append(dict(spec="start_20230101_df5", flags=[f for f in MAIN if f not in ("own_race_2022", "post_2022")], df=5,
                  start="2023-01-01"))
    for df in (3, 5, 8):
        s.append(dict(spec=f"silence_split_df{df}", flags=[f for f in MAIN if f != "own_race_2022"] +
                      ["own_race_silence", "own_race_rest"], df=df))
    return s


rows = []
for acct, y in (("personal", d_p), ("official", d_o)):
    for sp in spec_list():
        X = design(sp["flags"], sp["df"], sp.get("start"))
        yy = y.loc[X.index].values
        try:
            g, alpha = fit_nb(yy, X)
        except Exception as e:
            print("fail", acct, sp["spec"], e)
            continue
        ci = pd.DataFrame(np.asarray(g.conf_int()), index=X.columns)
        par = pd.Series(np.asarray(g.params), index=X.columns)
        pv = pd.Series(np.asarray(g.pvalues), index=X.columns)
        for f in X.columns:
            if f in sp["flags"] or f.startswith("dow_"):
                rows.append(dict(account=acct, spec=sp["spec"], spline_df=sp["df"], start=str(X.index[0].date()), term=f,
                                 rate_ratio=float(np.exp(par[f])), lo=float(np.exp(ci.loc[f, 0])),
                                 hi=float(np.exp(ci.loc[f, 1])), p=float(pv[f]), alpha=alpha, n_days=len(yy),
                                 llf=float(g.llf), aic=float(-2 * g.llf + 2 * len(par))))
rr = pd.DataFrame(rows)
rr["label"] = rr.term.map(LABEL).fillna(rr.term)
rr.to_csv(T6B / "nb_rate_ratios.csv", index=False)


def tab(acct, term, specs):
    out = {}
    for s in specs:
        r = rr[(rr.account == acct) & (rr.spec == s) & (rr.term == term)]
        if len(r):
            r = r.iloc[0]
            out[s] = [round(r.rate_ratio, 3), round(r.lo, 3), round(r.hi, 3), round(r.p, 4)]
    return out


# ---------------------------------------------------------------- transition context (descriptive)
wk = d_p["2024-09-30":"2025-06-01"].resample("W-SUN").sum()
wko = d_o["2024-09-30":"2025-06-01"].resample("W-SUN").sum()
pd.DataFrame({"week_ending": wk.index.date, "personal_posts": wk.values, "official_posts": wko.values}).to_csv(
    T6B / "transition_context.csv", index=False)


def mean_per_day(d, a, b):
    return float(d[a:b].mean())


ctx = {}
for acct, d in (("personal", d_p), ("official", d_o)):
    ctx[acct] = dict(
        mean_per_day_12wk_before_0103=mean_per_day(d, "2024-10-11", "2025-01-02"),
        mean_per_day_0103_to_0119=mean_per_day(d, "2025-01-03", "2025-01-19"),
        mean_per_day_0120_to_0202=mean_per_day(d, "2025-01-20", "2025-02-02"),
        mean_per_day_12wk_after_0203=mean_per_day(d, "2025-02-03", "2025-04-27"),
        mean_per_day_2024_h2=mean_per_day(d, "2024-07-01", "2024-12-31"),
        mean_per_day_2025_h1=mean_per_day(d, "2025-01-01", "2025-06-30"))
AC = "personal"
trans_main = tab("personal", "jan2025_transition", ["main_df3", "main_df5", "main_df8", "main_df12", "main_df16",
                                                   "window_2024on_df3", "window_2024on_df5", "start_20221101_df5",
                                                   "start_20230101_df5"])
trans_off = tab("official", "jan2025_transition", ["main_df3", "main_df5", "main_df8", "main_df12", "main_df16",
                                                  "window_2024on_df5"])
aic = {s: round(float(rr[(rr.account == "personal") & (rr.spec == s)].aic.iloc[0]), 1)
       for s in ("main_df3", "main_df5", "main_df8", "main_df12", "main_df16")}
aic_off = {s: round(float(rr[(rr.account == "official") & (rr.spec == s)].aic.iloc[0]), 1)
           for s in ("main_df3", "main_df5", "main_df8", "main_df12", "main_df16")}
step_cmp = {a: {"step_0103": tab(a, "step_0103", ["both_steps_df5", "both_steps_df8"]),
                "step_0120": tab(a, "step_0120", ["both_steps_df5", "both_steps_df8", "step_0120_df5", "step_0120_df8"])}
            for a in ("personal", "official")}

old = json.load(open(ROOT / "output/v7/t6/numbers_t6.json"))
num = dict(
    phase="T6b",
    january_2025_transition=dict(
        definition="single step indicator, 1 from 2025-01-03 (Senate majority changes to R) on; the inauguration (2025-01-20) is "
                   "17 days later and is not separately modelled; inaug_2wk dropped",
        statement=("The presidency and the Senate majority change cannot be separated in these data (17 days apart, both "
                   "persistent steps, and the account trends upward). The estimate is an association with the January 2025 "
                   "transition as a whole, not with either event alone."),
        correlation_between_step_0103_and_step_0120=corr_steps,
        personal_rate_ratio_by_spec=trans_main, official_rate_ratio_by_spec=trans_off,
        personal_aic_by_spline_df=aic, official_aic_by_spline_df=aic_off,
        both_steps_in_one_model=step_cmp, step_at_0120_only={"personal": tab("personal", "step_0120", ["step_0120_df5", "step_0120_df8"]),
                                                             "official": tab("official", "step_0120", ["step_0120_df5", "step_0120_df8"])},
        no_transition_term_personal_AIC=float(rr[(rr.account == "personal") & (rr.spec == "no_transition_df5")].aic.iloc[0]),
        descriptive_posts_per_day=ctx,
        reading=("Rate ratio format [RR, lo, hi, p]. The estimate depends strongly on how flexibly the time trend is modelled; "
                 "see by-spec values.")),
    own_race_2022=dict(
        descriptive=desc22,
        rate_ratio_full_series_by_spec=tab("personal", "own_race_2022", ["main_df3", "main_df5", "main_df8", "main_df12", "main_df16"]),
        official_rate_ratio_full_series_by_spec=tab("official", "own_race_2022", ["main_df3", "main_df5", "main_df8", "main_df12", "main_df16"]),
        window_split_personal=dict(
            own_race_silence=f"degenerate: zero posts on all {int(reg.own_race_silence.sum())} days of the stretch "
                             f"{s0.date()} to {s1.date()} (perfect separation; the rate ratio is not estimable and the fitted 0 is not a finding)",
            own_race_rest=tab("personal", "own_race_rest", ["silence_split_df3", "silence_split_df5", "silence_split_df8"]),
            own_race_rest_posts_per_day=float(d_p[(reg.own_race_rest == 1)].mean()),
            own_race_rest_days=int(reg.own_race_rest.sum())),
        series_restarted={"2022-11-01_flag_dropped": "own_race_2022 omitted (only 8 window days remain)",
                          "2023-01-01_flag_dropped": "own_race_2022 and post_2022 omitted"},
        other_terms_when_restarted={t: tab("personal", t, ["main_df5", "start_20221101_df5", "start_20230101_df5"])
                                    for t in ("in_session", "recess", "natl_general_2024", "post_2024", "event_day", "jan2025_transition")},
        caution=("The full-series rate ratio is driven by an account that posted 1-8 times a week for its first 15 weeks and is "
                 "not a stable effect; use the descriptive counts. 54 of the 65 window days have zero posts, 28 of them in one "
                 "confirmed stretch (counts endpoint also zero).")),
    nb_main_personal={t: tab("personal", t, ["main_df5"]).get("main_df5") for t in MAIN},
    nb_main_official={t: tab("official", t, ["main_df5"]).get("main_df5") for t in MAIN},
    nb_alpha={a: float(rr[(rr.account == a) & (rr.spec == "main_df5")].alpha.iloc[0]) for a in ("personal", "official")},
    files=dict(rate_ratios="nb_rate_ratios.csv", descriptive="descriptive_2022.csv", zero_runs="zero_runs_2022.csv",
               weekly_2022="weekly_2022.csv", transition_context="transition_context.csv"),
    guardrail="associations only; descriptive and not causal",
    unchanged_from_t6={k: old[k] for k in ("completeness", "pelt", "content", "contrast") if k in old},
    superseded=dict(note="T6 model with inaug_2wk plus r_president (step from 2025-01-20) and no restart/descriptive 2022 treatment",
                    volume_models=old.get("volume_models"), notes=old.get("notes")))
(T6B / "numbers_t6b.json").write_text(json.dumps(num, indent=1, default=float))
pd.set_option("display.width", 250)
print(json.dumps({k: num[k] for k in ("january_2025_transition", "own_race_2022")}, indent=1, default=float)[:9000])
