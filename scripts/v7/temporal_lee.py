"""Temporal features for @BasedMikeLee (Sen. Mike Lee, VoteView icpsr 41110, home state UT).

Archive path is a parameter: the original xlsx (49,415 posts to 2026-02-24) or the merged archive parquet
(data/BasedMikeLee_archive_v2.parquet, 57,589 posts to 2026-10-06).
Post sets: nonrt = type in {tweet, reply, quote} (primary; matches "text does not start RT @" for congresstweets);
all = every post (secondary). Diagnostic sets: tweet_quote (the retired "original" set), tweet (tweets only),
reply, retweet.
Windows (UTC): full, first12 (2022-07-25 to 2023-07-24), calendar years.
Percentiles are empirical ranks among congresstweets members (output/v7/temporal/member_features.csv):
  first12 and calendar 2022/2023 against the same calendar span (same period);
  full window against members' 2017-2023 full window (cross-period; stated in the output).
Usage: python temporal_lee.py --archive PATH --tag xlsx
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from xbox import location as lc  # noqa: E402
from xbox import temporal_v2 as tv  # noqa: E402
import temporal_features as tf  # noqa: E402

OUT = ROOT / "output/v7/temporal"
LEE = 41110
PCT_COLS = ["A_n", "B", "entropy_model_trans_dc", "night_00_06_model_trans_dc", "night_22_06_model_trans_dc",
            "night_00_06_all_dc", "night_00_06_all_home", "night_00_06_model_trans_home",
            "weekend_ratio_model_trans_dc", "rapid_share_300", "hawkes_eta", "hawkes_inv_beta_min"]


def load_archive(path):
    p = Path(path)
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_excel(p, dtype={"id": str})
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["created_at"])
    df["t"] = df["created_at"].dt.tz_localize(None).values.astype("datetime64[s]").astype("int64")
    return df.sort_values("t").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=str(Path.home() / "x_box/BasedMikeLee_full_archive.xlsx"))
    ap.add_argument("--tag", default="xlsx")
    ap.add_argument("--voteview", default=str(ROOT / "data/voteview"))
    ap.add_argument("--no-hawkes", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_archive(a.archive)
    md, cd, mch = lc.load_vote_days(a.voteview)
    ml = lc.MemberLocation(md[LEE], cd["S"], "UT", name="Lee")
    t_last = int(df.t.max())
    win = dict(tf.WINDOWS)
    win["first12"] = win.pop("w12")
    win["full"] = (0, t_last + 1)
    for y in range(2022, 2027):
        win[f"y{y}"] = (tf.d2s(f"{y}-01-01"), tf.d2s(f"{y + 1}-01-01"))
    for y in range(2017, 2022):
        win.pop(f"y{y}", None)
    sets = {"nonrt": tv.nonretweet_mask_lee(df.type.values),
            "all": np.ones(len(df), bool),
            "tweet_quote": df.type.isin(["tweet", "quote"]).values, "tweet": (df.type == "tweet").values,
            "reply": (df.type == "reply").values, "retweet": (df.type == "retweet").values}
    rows = []
    for w, (lo, hi) in win.items():
        inw = (df.t.values >= lo) & (df.t.values < hi)
        for ps, m in sets.items():
            sel = inw & m
            big = ps in ("nonrt", "all")
            f = tf.block_features(df.t.values[sel], ml, True, 12345, hawkes=big and not a.no_hawkes,
                                  ci=big and w in ("full", "first12"), sens=(ps == "nonrt" and w in ("full", "first12")))
            rows.append(dict(archive=a.tag, window=w, postset=ps, **f))
    lf = pd.DataFrame(rows)
    ci_cols = [c for c in ("hawkes_eta_lo", "hawkes_eta_hi") if c in lf]
    if ci_cols:
        d = lf.loc[lf[ci_cols[0]].notna(), ["window", "postset", "n", "hawkes_eta", *ci_cols, "hawkes_ks_stat",
                                            "hawkes_ks_p"]].copy()
        d["archive"] = a.tag
        d["note"] = "model-conditional profile-likelihood CI; diagnostics only"
        d.to_csv(OUT / f"hawkes_ci_diagnostics_lee_{a.tag}.csv", index=False)
    lf_pub = lf.drop(columns=ci_cols)
    lf_pub.to_csv(OUT / f"lee_features_{a.tag}.csv", index=False)

    # location-model summary on all posts (checks against lee_location_model.py)
    cls = ml.classes(df.t.values)
    hd = ml.local_hour_dow(df.t.values, lc.SCENARIOS, cls)
    summ = dict(archive=a.tag, n_posts=int(len(df)), first=str(df.created_at.min()), last=str(df.created_at.max()),
                lee_vote_days=int(ml.n_vote_days),
                day_class_posts={lc.CLASS_NAMES[i]: int((cls == i).sum()) for i in range(4)},
                scenarios={s: dict(night_00_06=round(tv.night_share(hd[s][0]), 4),
                                   night_22_06=round(tv.night_share(hd[s][0], "22_06"), 4),
                                   weekend=round(float((hd[s][1] >= 5).mean()), 4)) for s in lc.SCENARIOS},
                type_counts=df.type.value_counts().to_dict())
    (OUT / f"lee_location_summary_{a.tag}.json").write_text(json.dumps(summ, indent=2, default=str))

    # percentiles
    mf = pd.read_csv(OUT / "member_features.csv")
    mf = mf[(mf.postset == "nonrt") & (mf.n >= tf.MIN_N)]
    refmap = {"first12": ("w12", "same calendar span 2022-07-25..2023-07-24"),
              "y2022": ("y2022", "same calendar year"), "y2023": ("y2023", "same calendar year"),
              "full": ("full", "cross-period: members 2017-2023 vs Lee archive window"),
              "y2024": ("full", "cross-period: members 2017-2023"), "y2025": ("full", "cross-period: members 2017-2023"),
              "y2026": ("full", "cross-period: members 2017-2023")}
    prow = []
    for _, r in lf[lf.postset == "nonrt"].iterrows():
        if r.window not in refmap or r.n < tf.MIN_N:
            continue
        rw, note = refmap[r.window]
        ref = mf[mf.window == rw]
        for c in PCT_COLS:
            if c not in ref or pd.isna(r.get(c)):
                continue
            ev = ref[c].values
            vm, nvm = tv.volume_matched_percentile(r[c], r.n, ev, ref.n.values)
            row = dict(archive=a.tag, lee_window=r.window, ref_window=rw, ref_note=note, feature=c,
                       lee_value=r[c], lee_n=r.n, n_ref=int((~np.isnan(ev)).sum()),
                       pct_all=tv.emp_percentile(r[c], ev), pct_volume_matched=vm, n_ref_vm=nvm)
            if c == "hawkes_eta":
                for thr, nm in ((0.05, "wellfit05"), (0.10, "wellfit10")):
                    rr = ref[ref.hawkes_ks_p >= thr]
                    row[f"pct_all_{nm}"] = tv.emp_percentile(r[c], rr[c].values)
                    row[f"n_ref_{nm}"] = int(len(rr))
                    row[f"pct_vm_{nm}"] = (tv.volume_matched_percentile(r[c], r.n, rr[c].values, rr.n.values)[0]
                                           if len(rr) >= 30 else np.nan)
            prow.append(row)
    pd.DataFrame(prow).to_csv(OUT / f"lee_percentiles_{a.tag}.csv", index=False)
    pd.set_option("display.width", 250)
    print(json.dumps(summ, indent=1, default=str))
    keep = ["window", "postset", "n", "A_n", "B", "entropy_model_trans_dc", "night_00_06_model_trans_dc",
            "weekend_ratio_model_trans_dc", "rapid_share_300", "hawkes_eta", "hawkes_eta_lo", "hawkes_eta_hi",
            "hawkes_inv_beta_min", "hawkes_ks_stat", "hawkes_ks_p"]
    print(lf[keep].round(3).to_string())


if __name__ == "__main__":
    main()
