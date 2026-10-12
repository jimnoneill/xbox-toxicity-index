"""Numbers for the full-corpus T1 rerun: outages, thread collapsing, scheduler flags, model-free descriptors (account level
primary, member level secondary), and the comparison with the 1-in-3 sample results in output/v7/t1.
Output: output/v7/t1full/numbers_t1full.json, descriptor_summary.csv
Usage: python t1full_numbers.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t1full_common import *  # noqa: E402,F401,F403

M = ["A_n", "B", "r", "entropy", "night_00_06", "night_22_06", "weekend_share", "rapid_share_300",
     "sessions_per_day_15", "sessions_per_day_30", "sessions_per_day_60", "events_per_session_15",
     "events_per_session_30", "events_per_session_60", "zero_gap_share"]


def q(s):
    s = s.dropna()
    return dict(n=int(len(s)), median=float(s.median()), q25=float(s.quantile(.25)), q75=float(s.quantile(.75)),
                mean=float(s.mean()))


out = {}
o = json.load(open(T1 / "outage_summary.json"))
oo = json.load(open(T1OLD / "outage_summary.json"))
runs = pd.read_csv(T1 / "outage_runs.csv")
out["outages"] = dict(
    posts_unique_ids=o["posts_unique_ids"], calendar_days=o["calendar_days_in_span"],
    zero_post_days=o["zero_post_frame_days"], zero_post_days_list=o["zero_post_frame_days_list"],
    outage_hours=o["outage_hours"], outage_runs=o["outage_runs"], outage_days_touched=o["outage_days_touched"],
    outage_days_whole=o["outage_days_whole"], share_of_hours=o["share_of_frame_hours_in_outage"],
    observed_hours=o["observed_hours"], posts_in_outage_hours=o["posts_in_outage_hours"], by_year=o["by_year"],
    longest_run_hours=int(runs.hours.max()), longest_run_start=str(runs.loc[runs.hours.idxmax(), "et_start"]),
    runs_ge_12h=runs[runs.hours >= 12][["et_start", "hours", "count", "expected"]].round(0).astype(str).to_dict("records"),
    previous_1in3_sample=dict(outage_hours=oo["outage_hours"], runs=oo["outage_runs"],
                              share=oo["share_of_frame_hours_in_outage"]))
out["threads"] = json.load(open(T1 / "thread_stats.json"))
out["threads"]["validation_proxy"] = json.load(open(T1 / "thread_validation_proxy.json"))
out["scheduler"] = json.load(open(T1 / "scheduler_summary.json"))
out["scheduler"]["previous_1in3"] = {k: v for k, v in json.load(open(T1OLD / "scheduler_summary.json")).items()
                                     if k in ("accounts_n200", "flag_minute", "flag_tool", "flag_both")}

rows = []
for level in ("account", "member"):
    d = pd.read_csv(T1 / f"descriptors_{level}.csv")
    d = d[d.n >= MIN_N]
    res = {}
    for var, g in d.groupby("variant"):
        res[var] = {"n_streams": int(g.id.nunique()), **{m: q(g[m]) for m in M if m in g}}
        for m in M:
            if m in g:
                rows.append(dict(level=level, variant=var, metric=m, **q(g[m])))
    out[f"descriptors_{level}"] = res
    c = d[d.variant == "clean"].copy()
    c["party3"] = c.party.where(c.party.isin(["D", "R"]), "other")
    for key in ("party3", "account_type", "chamber"):
        out[f"descriptors_{level}_clean_by_{key}"] = {
            str(k): {m: float(g[m].median()) for m in ("A_n", "entropy", "night_00_06", "weekend_share",
                                                      "rapid_share_300", "sessions_per_day_30", "events_per_session_30")}
            | {"n": int(len(g))} for k, g in c.groupby(key)}
pd.DataFrame(rows).to_csv(T1 / "descriptor_summary.csv", index=False)
# comparison with the 1-in-3 sample
chk = json.load(open(T1 / "descriptor_old_check.json"))
cmp_ = {}
for var in ("masked", "clean"):
    r = chk[f"full_vs_1in3_{var}"]
    cmp_[var] = {"n_accounts": r["n_accounts"]} | {m: r[m] for m in ("A_n", "rapid_share_300", "sessions_per_day_30",
                                                                      "events_per_session_30", "entropy",
                                                                      "night_00_06", "weekend_share") if m in r}
out["comparison_with_1in3"] = cmp_
# secondary-clean comparison: share of accounts qualifying, A_n distribution percentiles
a = pd.read_csv(T1 / "descriptors_account.csv")
a = a[(a.variant == "clean") & (a.n >= MIN_N)]
out["A_n_clean_percentiles"] = {str(p): float(a.A_n.quantile(p)) for p in (.05, .25, .5, .75, .95)}
out["note"] = ("Account level primary (streams with >=200 events); member level pools all of a member's accounts. "
               "Variants: raw (no mask, calendar-span exposure), masked (outage hours removed), clean (thread-collapsed on top). "
               "Associations and descriptors only.")
(T1 / "numbers_t1full.json").write_text(json.dumps(out, indent=1, default=float))
print(json.dumps({k: out[k] for k in ("outages",)}, indent=1, default=float)[:1500])
print(json.dumps(out["comparison_with_1in3"], indent=1, default=float)[:3000])
