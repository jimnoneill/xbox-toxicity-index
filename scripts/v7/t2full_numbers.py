"""Numbers for the full-corpus T2 rerun: shared-shock diagnostics, event-day detection and the T3 cross-check.
Output: output/v7/t2full/numbers_t2full.json
Usage: python t2full_numbers.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t1full_common import *  # noqa: E402,F401,F403

out = {}
g = pd.read_csv(T2 / "shared_shock_groups.csv")
go = pd.read_csv(T2OLD / "shared_shock_groups.csv")
keep = ["group_type", "group", "accounts", "events", "corr_hourly_median", "corr_hourly_lo", "corr_hourly_hi",
        "corr_daily_median", "corr_within_day_median", "r2_c_median", "r2_lag1_median", "r2_both_median",
        "spike_ratio_rhythm_median", "spike_ratio_rhythm_lo", "spike_ratio_rhythm_hi", "spike_pooled_share",
        "spike_pooled_expected_rhythm", "spike_pooled_ratio", "share_corr_hourly_gt_0"]
out["groups_full"] = g[keep].round(4).to_dict("records")
out["groups_1in3"] = go[["group_type", "group", "accounts", "corr_hourly_median", "corr_daily_median",
                         "spike_ratio_rhythm_median", "spike_pooled_ratio"]].round(4).to_dict("records")
out["summary_full"] = json.load(open(T2 / "shared_shock_summary.json"))
out["summary_1in3"] = json.load(open(T2OLD / "shared_shock_summary.json"))
out["hourly_table"] = json.load(open(T2 / "hourly_counts_meta.json"))
es = json.load(open(T2 / "event_summary.json"))
out["event_summary_full"] = es
out["event_summary_1in3"] = json.load(open(T2OLD / "event_summary.json"))
top = pd.read_csv(T2 / "event_days_top60.csv")
out["top15_event_days"] = top.head(15)[["rank", "date", "events", "expected", "ratio", "z_day", "top_hashtags"]].to_dict("records")
out["event_day_count_by_type"] = {
    "state_of_union_or_joint_address": int(top.top_hashtags.str.contains("sotu|jointaddress|stateoftheunion", case=False).sum()),
    "holiday_or_observance": int(top.distinctive_words.str.contains("happy|merry|independence|veterans|halloween|hanukkah|christmas",
                                                                      case=False).sum())}
out["crosscheck_t3"] = json.load(open(T2 / "crosscheck_summary.json"))
sh = pd.read_csv(T2 / "shared_shock_accounts.csv")
out["accounts_spike_ratio_gt_1"] = float((sh.spike_ratio_rhythm > 1).mean())
out["note"] = ("Shared-shock metrics are associations between an account's hourly counts and the leave-one-out Congress-wide "
               "count after removing the account's own weekly rhythm and monthly level; not causal.")
(T2 / "numbers_t2full.json").write_text(json.dumps(out, indent=1, default=float))
print(json.dumps(out["groups_full"][:6], indent=1, default=float)[:2500])
