"""P3 step 3: one illustrative out-of-sample case. Train on congress members (all transitions, origins 2017 to 2021,
Lee's own official accounts excluded), then predict the personal account @BasedMikeLee's next-political-year changes:
  origin PY2023 (profile PY2022 and PY2023)  -> PY2024 (1 Jul 2024 to 30 Jun 2025)
  origin PY2024 (profile PY2023 and PY2024)  -> PY2025 (1 Jul 2025 to 30 Jun 2026)
This is a different period (2022 to 2026 vs 2017 to 2023) and a different account type (personal, ~16x the volume of
the members' official accounts). Only the core feature set is used (no account-mix and no co-movement features:
the archive has no client field and Congress-wide counts end in July 2023). One account, one case, no inference.
Usage: python p3_lee.py --ti sample|full
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p3_panel as pp  # noqa: E402
from p3_panel import c1, lc, tv, ROOT, py_bounds, et_day, DIMS, DIMS7  # noqa: E402

sys.path.insert(0, str(ROOT))
from xbox import forecast_p3 as fp  # noqa: E402

LEE = "L000577"
FEATS = fp.FEATURES_CORE
YEARS = [2022, 2023, 2024, 2025]


def lee_panel():
    ev = pd.read_parquet(ROOT / "output/v7/t1/lee_threads.parquet")
    arch = c1.load_lee()
    first_t, last_t = int(arch.t.min()), int(arch.t.max())
    md, cd, _ = lc.load_vote_days(ROOT / "data/voteview")
    ml = lc.MemberLocation(md[c1.LEE_ICPSR], cd["S"], "UT", name="Lee")
    pr = pd.read_parquet(ROOT / "output/v7/ti/post_probs_lee.parquet")
    pr = pr[pr.post_type == "original"].copy()
    t = pr.created_at.dt.tz_convert("UTC").dt.tz_localize(None).values.astype("datetime64[s]").astype("int64")
    pr["t"] = t
    et = pd.to_datetime(t, unit="s", utc=True).tz_convert("America/New_York")
    pr["py"] = np.where(et.month >= 7, et.year, et.year - 1)
    pr["day"] = et.tz_localize(None).values.astype("datetime64[D]").astype("int64")
    pr["ti7"] = pr[DIMS7].mean(axis=1)
    lo = {y: py_bounds(y)[0] for y in YEARS}
    hi = {y: py_bounds(y)[1] for y in YEARS}
    pr = pr[pr.py.isin(YEARS)]
    pr["frac"] = [(a - lo[y]) / (hi[y] - lo[y]) - 0.5 for a, y in zip(pr.t, pr.py)]
    pr["bioguide"] = LEE
    ti = pp.ti_agg(pr)
    rows = []
    for y in YEARS:
        m = (ev.t.values >= lo[y]) & (ev.t.values < hi[y])
        e = ev[m].sort_values("t")
        tt = e.t.values
        r = c1.describe(tt, ml, True, None, None, 0, e["size"].values, naive_days=True)
        days = (min(hi[y], last_t + 1) - max(lo[y], first_t)) / 86400.0
        mid = lo[y] + (hi[y] - lo[y]) // 2
        row = dict(bioguide=LEE, py=y, n_events=int(m.sum()), exposure_cal_days=days,
                   log_rate=float(np.log(m.sum() / days)),
                   vol_trend=float(np.log((np.sum(tt >= mid) + 1.0) / (np.sum(tt < mid) + 1.0))))
        for k in ("A_n", "rapid_share_300", "sessions_per_day_30", "events_per_session_30", "entropy", "night_00_06",
                  "weekend_share"):
            row[k] = r.get(k, np.nan)
        rows.append(row)
    p = pd.DataFrame(rows).merge(ti, on=["bioguide", "py"], how="left")
    return p


def lee_transitions(p, su):
    """Same construction as fp.make_transitions for one member; targets filled where the next year exists."""
    cal = pd.read_parquet(ROOT / "output/v7/t3/calendar_member_day.parquet",
                          columns=["date", "bioguide", "aligned_with_president", "in_chamber_majority"])
    cal = cal[cal.bioguide == LEE].set_index("date")
    vv = pd.read_csv(ROOT / "data/voteview_all_members.csv", usecols=["congress", "icpsr", "nokken_poole_dim1"])
    first = pd.read_csv(ROOT / "data/voteview_all_members.csv", usecols=["congress", "icpsr"])
    first = first[first.icpsr == c1.LEE_ICPSR].congress.min()
    npd = vv[vv.icpsr == c1.LEE_ICPSR].groupby("congress").nokken_poole_dim1.mean()
    rows = []
    for t in (2023, 2024):
        r = p[p.py == t].iloc[0]
        prev = p[p.py == t - 1].iloc[0]
        end = pd.Timestamp(f"{t + 1}-06-30")
        cong = (t + 1 - 1789) // 2 + 1
        row = {c: r[c] for c in fp.LEVELS + fp.DESC + fp.TRENDS}
        np1 = float(npd.get(cong, npd.iloc[-1]))
        row.update(party_R=1.0, chamber_senate=1.0, np_dim1=np1, abs_np_dim1=abs(np1),
                   seniority_congresses=float(max(cong - first, 0)),
                   aligned_with_president=float(cal.loc[end, "aligned_with_president"]),
                   in_chamber_majority=float(cal.loc[end, "in_chamber_majority"]))
        row.update(fp.seat_flags(su, LEE, t))
        row.update(has_lag=1.0, origin=t, bioguide=LEE, party="R", chamber="senate")
        for v, nm in (("log_rate", "lag_d_log_rate"), ("ti7", "lag_d_ti7"), ("tribal", "lag_d_tribal"),
                      ("performative", "lag_d_performative"), ("ragebait", "lag_d_ragebait")):
            row[nm] = float(r[v] - prev[v])
        hist = p[p.py <= t]
        for v in ("log_rate", "ti7"):
            row[f"dev_hist_{v}"] = float(hist[v].mean() - r[v])
        n1 = p[p.py == t + 1].iloc[0]
        row["d_logvol"] = float(n1.log_rate - r.log_rate)
        for tg in ("ti7", "tribal", "performative", "ragebait"):
            row[f"d_{tg}"] = float(n1[tg] - r[tg])
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ti", default="sample", choices=["sample", "full"])
    a = ap.parse_args()
    out = ROOT / "output/v7/p3" / a.ti
    T = pd.read_csv(out / "transitions.csv")
    T = T[T.bioguide != LEE].reset_index(drop=True)  # leave Lee's official accounts out of training
    p = lee_panel()
    p.to_csv(out / "lee_panel.csv", index=False)
    su = {(LEE, 2022): 1, (LEE, 2024): 0, (LEE, 2026): 0}
    L = lee_transitions(p, su)
    L.to_csv(out / "lee_transitions.csv", index=False)
    # out-of-sample residual quantiles from the member rolling-origin predictions (dev + final)
    pr = pd.concat([pd.read_csv(out / f) for f in ("preds_dev.csv", "preds_final.csv") if (out / f).exists()])
    # absolute-threshold escalation labels (top quintile of training changes), because the members' within-year
    # quintile cannot be formed for Lee's years
    thr = {"esc_ti7": T.d_ti7.quantile(0.8), "esc_vol": T.d_logvol.quantile(0.8)}
    T["esc_ti7"] = (T.d_ti7 > thr["esc_ti7"]).astype(int)
    T["esc_vol"] = (T.d_logvol > thr["esc_vol"]).astype(int)
    L["esc_ti7"] = (L.d_ti7 > thr["esc_ti7"]).astype(int)
    L["esc_vol"] = (L.d_logvol > thr["esc_vol"]).astype(int)
    recs = []
    for tg, (lvl, _) in fp.TARGETS.items():
        base = fp.baselines_reg(T, L.assign(party="R", chamber="senate"), tg, lvl)
        preds = {k: v for k, v in base.items() if k != "party_year_ORACLE"}
        for kind, tag in (("ridge", "ridge_core"), ("gbt", "gbt_core")):
            preds[tag], _, _ = fp.fit_predict(kind, T, L, FEATS, tg)
        for m, pv in preds.items():
            # interval from member out-of-sample residuals of the same model
            rr = pr[(pr.target == tg) & (pr.model == m)]
            res = (rr.y - rr.pred).values if len(rr) else np.array([0.0])
            qlo, qhi = np.percentile(res, [5, 95])
            for i, t in enumerate(L.origin):
                recs.append(dict(target=tg, origin=int(t), target_year=int(t) + 1, model=m, pred=float(pv[i]),
                                 lo90=float(pv[i] + qlo), hi90=float(pv[i] + qhi), actual=float(L[tg].iloc[i]),
                                 interval_from="member out-of-sample residuals" if len(rr) else "none"))
    for e in fp.ESC:
        for kind, tag in (("logit", "logit_core"), ("gbt", "gbt_core")):
            pv, _, _ = fp.fit_predict(kind, T, L, FEATS, e, binary=True)
            for i, t in enumerate(L.origin):
                recs.append(dict(target=e, origin=int(t), target_year=int(t) + 1, model=tag, pred=float(pv[i]),
                                 lo90=np.nan, hi90=np.nan, actual=float(L[e].iloc[i]), interval_from="threshold %.3f" % thr[e]))
    R = pd.DataFrame(recs)
    R["abs_error"] = (R.pred - R.actual).abs()
    R["in_interval"] = (R.actual >= R.lo90) & (R.actual <= R.hi90)
    R.to_csv(out / "lee_predictions.csv", index=False)
    # where Lee's origin profile falls among members' same-origin features (out-of-distribution check)
    Tm = pd.read_csv(out / "transitions.csv")
    ood = []
    for t in (2023, 2024):
        Lr = L[L.origin == t].iloc[0]
        ref = Tm[Tm.origin == 2021]
        for f in ["log_rate", "ti7", "tribal", "performative", "ragebait", "A_n", "rapid_share_300", "sessions_per_day_30",
                  "entropy", "night_00_06", "weekend_share"]:
            ood.append(dict(origin=t, feature=f, lee=float(Lr[f]), percentile_vs_members_2021=float((ref[f] < Lr[f]).mean()),
                            member_median=float(ref[f].median())))
    pd.DataFrame(ood).to_csv(out / "lee_ood_percentiles.csv", index=False)
    # figure
    fig, axs = plt.subplots(1, 5, figsize=(14, 3.6))
    for ax, tg in zip(axs, fp.TARGETS):
        d = R[(R.target == tg) & R.model.isin(["gbt_core", "ridge_core", "persistence", "regression_to_mean"])]
        for k, (m, c) in enumerate((("persistence", "#7f7f7f"), ("regression_to_mean", "#009E73"),
                                    ("ridge_core", "#0072B2"), ("gbt_core", "#D55E00"))):
            for j, t in enumerate((2023, 2024)):
                x = j + (k - 1.5) * 0.12
                r = d[(d.model == m) & (d.origin == t)].iloc[0]
                ax.errorbar(x, r.pred, yerr=[[r.pred - r.lo90], [r.hi90 - r.pred]], fmt="o", color=c, ms=4,
                            label=m if j == 0 else None)
        for j, t in enumerate((2023, 2024)):
            ax.plot([j - 0.3, j + 0.3], [d[d.origin == t].actual.iloc[0]] * 2, color="k", lw=2,
                    label="actual" if j == 0 else None)
        ax.axhline(0, color="k", lw=0.5, ls=":")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["to PY2024", "to PY2025"])
        ax.set_title(fp.TARGETS[tg][1])
    axs[0].legend(frameon=False, fontsize=7)
    fig.suptitle("@BasedMikeLee, predicted vs actual next-year change (90% intervals from member residuals); one case", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig_lee_forecast.png", dpi=160)
    (out / "numbers_lee.json").write_text(json.dumps(dict(
        panel=p.round(5).to_dict("records"), predictions=R.round(5).to_dict("records"),
        thresholds={k: float(v) for k, v in thr.items()}, train_rows=int(len(T)),
        ood=pd.DataFrame(ood).round(4).to_dict("records")), indent=1, default=float))
    print(R[R.target.isin(list(fp.TARGETS))].round(3).to_string())


if __name__ == "__main__":
    main()
