"""Shared data preparation for the T4 binned point-process regression (new file).

Reads output/v7/t2full/hourly_counts.parquet (60-minute bins of thread-collapsed events per account in observed hours,
local hour and weekday from the location model, leave-one-out Congress count), the T3 regime calendar and the detected
event days (top 60 from the full corpus, output/v7/t2full/event_days_top60.csv).

account_design(i, variant) returns everything needed to fit or simulate one account:
  X (fit rows), y, names, cluster, plus dense-grid pieces for simulation/predictive checks.
Variants: full (Congress term + lags + regimes), nolag, noC, base (neither), csplit (Congress term split in daily mean and
within-day deviation), noreg (no regime flags).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t1full_common import *  # noqa: E402,F401,F403
from xbox import binned_model as bm  # noqa: E402

SEED = 20261010
T4 = ROOT / "output/v7/t4"
MIN_EVENTS_T4 = 500
G = {}


def load_all():
    cols = ["account", "bioguide", "account_type", "party", "chamber", "hour_idx", "local_hour", "local_dow", "count",
            "loo_congress_count"]
    hc = pd.read_parquet(T2 / "hourly_counts.parquet", columns=cols)
    hc = hc.sort_values(["account", "hour_idx"], kind="stable").reset_index(drop=True)
    acc = hc["account"].values
    starts = np.flatnonzero(np.r_[True, acc[1:] != acc[:-1]])
    ends = np.r_[starts[1:], len(acc)]
    G["accounts"] = acc[starts]
    G["starts"], G["ends"] = starts, ends
    G["bioguide"] = hc["bioguide"].astype(str).values[starts]
    G["party"] = hc["party"].astype(str).values[starts]
    G["chamber"] = hc["chamber"].astype(str).values[starts]
    G["atype"] = hc["account_type"].astype(str).values[starts]
    G["hour_idx"] = hc["hour_idx"].values.astype("int64")
    G["local_hour"] = hc["local_hour"].values.astype("int16")
    G["local_dow"] = hc["local_dow"].values.astype("int16")
    G["count"] = hc["count"].values.astype("int32")
    G["loo"] = hc["loo_congress_count"].values.astype("int32")
    cal = pd.read_parquet(ROOT / "output/v7/t3/calendar_member_day.parquet",
                          columns=["date", "bioguide"] + bm.CAL_COLS)
    cal["bioguide"] = cal["bioguide"].astype(str)
    G["cal"] = bm.calendar_arrays(cal)
    ev = pd.read_csv(T2 / "event_days_top60.csv")
    days = pd.to_datetime(ev["date"]).values.astype("datetime64[D]").astype("int64") - bm.CAL_DAY0
    det = np.zeros(4000, dtype=np.int8)
    det[days] = 1
    G["detected"] = det
    return len(G["accounts"])


def account_arrays(i):
    a, b = G["starts"][i], G["ends"][i]
    return dict(hour_idx=G["hour_idx"][a:b], local_hour=G["local_hour"][a:b], local_dow=G["local_dow"][a:b],
                count=G["count"][a:b], loo=G["loo"][a:b])


def account_design(i, variant="full", y_override=None):
    """Design for account index i. y_override: dense-grid counts (for simulation studies), else observed counts."""
    d = account_arrays(i)
    bg = G["bioguide"][i]
    cal_arr = G["cal"].get(bg)
    hrs = d["hour_idx"]
    ld = bm.local_day_from_hours(hrs, d["local_hour"])
    cell = (d["local_dow"] >= 5).astype(int) * 24 + d["local_hour"].astype(int)
    logc = np.log1p(d["loo"].astype(float))
    congress = {"full": "plain", "nolag": "plain", "noC": "none", "base": "none", "csplit": "split",
                "noreg": "plain"}[variant]
    Xs, names, inoff = bm.build_static(cell, ld, logc, cal_arr, G["detected"], 1, congress,
                                       exclude_regimes=(variant == "noreg"))
    lo = int(hrs.min())
    T = int(hrs.max()) - lo + 1
    pos = (hrs - lo).astype(int)
    ydense = np.zeros(T)
    ydense[pos] = d["count"]
    if y_override is not None:
        ydense = y_override
    present = np.zeros(T, dtype=bool)
    present[pos[inoff]] = True
    lag1, lagL, valid = bm.lag_features(ydense, present, 24)
    fit_rows = valid[pos] & inoff
    ncore = 1 + 47 + (0 if congress == "none" else (1 if congress == "plain" else 2))
    use_lags = variant not in ("nolag", "base")
    parts, nms = [Xs[:, :ncore]], names[:ncore]
    lag_cols = np.column_stack([lag1[pos], lagL[pos]])
    if use_lags:
        parts.append(lag_cols)
        nms = nms + ["log1p_lag1", "log1p_lag2_24"]
    parts.append(Xs[:, ncore:])
    nms = nms + names[ncore:]
    Xall = np.hstack(parts)
    out = dict(X=Xall[fit_rows], names=nms, y=ydense[pos][fit_rows], cluster=(hrs[fit_rows] // 168),
               fit_rows=fit_rows, pos=pos, lo=lo, T=T, present=present, ld=ld, Xall=Xall, ydense=ydense,
               lag_cols=lag_cols, local_hour=d["local_hour"], inoff=inoff, n_rows=int(len(hrs)),
               cal_ok=cal_arr is not None, ncore=ncore, use_lags=use_lags)
    return out
