"""Staff signal (spec C7): phone-dominant (phone share >= 0.6) vs tool-dominant (tool share >= 0.6)
members and accounts, non-retweet posts, full window. Hedges g (phone minus tool), 10,000-resample
bootstrap 95% CI (fixed seed), plus the client composition.
Output: output/v7/temporal/staff_signal.csv, staff_counts.json
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from xbox import temporal_v2 as tv  # noqa: E402

OUT = ROOT / "output/v7/temporal"
FEATS = ["A_n", "entropy_model_trans_dc", "night_00_06_model_trans_dc", "weekend_ratio_model_trans_dc",
         "rapid_share_300", "hawkes_eta", "hawkes_inv_beta_min"]


def main():
    res, counts = [], {}
    for level, fn in (("member", "member_features.csv"), ("account", "account_features.csv")):
        d = pd.read_csv(OUT / fn)
        d = d[(d.window == "full") & (d.postset == "nonrt") & (d.n >= 200)]
        ph, tl = d[d.share_phone >= 0.6], d[d.share_tool >= 0.6]
        counts[level] = dict(qualifying=len(d), phone_dominant=len(ph), tool_dominant=len(tl),
                             neither=len(d) - len(ph) - len(tl),
                             mean_share_phone=float(d.share_phone.mean()), mean_share_tool=float(d.share_tool.mean()),
                             mean_share_web=float(d.share_web.mean()))
        for f in FEATS:
            b = tv.hedges_g_boot(ph[f].values, tl[f].values, n_boot=10000, seed=7)
            res.append(dict(level=level, feature=f, g_phone_minus_tool=b["g"], ci_lo=b["lo"], ci_hi=b["hi"],
                            n_phone=b["n_x"], n_tool=b["n_y"], mean_phone=ph[f].mean(), mean_tool=tl[f].mean(),
                            median_phone=ph[f].median(), median_tool=tl[f].median()))
    r = pd.DataFrame(res)
    r.to_csv(OUT / "staff_signal.csv", index=False)
    (OUT / "staff_counts.json").write_text(json.dumps(counts, indent=2))
    print(json.dumps(counts, indent=1)); print(r.round(3).to_string())


if __name__ == "__main__":
    main()
