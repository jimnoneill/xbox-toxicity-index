"""Leakage perturbation test: perturb everything in panel year Y, rebuild; features of target-year-Y rows must not change.
Also confirms that next-year (Y+1) features DO change (so the test is sensitive)."""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE.parents[1] / "xbox"))
import numpy as np, pandas as pd
import kistner_ext as K
OUT = Path.home() / "x_box/pipeline/output/v7/kistner"
rng = np.random.default_rng(1)
pan = K.build_panel(("Twitter",)); fb = K.build_panel(("Facebook",))
base = K.build_transitions(pan, fb)
cols = K.PROFILE + K.STATIC + K.POWER


def perturb(df, Y):
    df = df.copy(); df["n"] = df["n"].astype(float); m = (df.Year == Y).values
    df.loc[m, "n"] = (df.loc[m, "n"] * rng.uniform(0.5, 2, m.sum())).clip(lower=K.MIN_N)  # keep membership in the panel
    df.loc[m, "share"] = rng.uniform(0, 1, m.sum())
    for c in ["pos", "adv", "cc", "cs", "bip"]:
        df.loc[m, c + "_sh"] = rng.uniform(0, 1, m.sum())
    return df


res = {}; ok = True
for Y in range(2011, 2023):
    t2 = K.build_transitions(perturb(pan, Y), perturb(fb, Y))
    a = base[base.Y == Y].set_index("MemberICPSR"); b = t2[t2.Y == Y].set_index("MemberICPSR")
    common = a.index.intersection(b.index)
    same = bool(np.allclose(a.loc[common, cols].values.astype(float), b.loc[common, cols].values.astype(float), equal_nan=True))
    tgt_changed = not np.allclose(a.loc[common].target.values, b.loc[common].target.values)
    na = base[base.Y == Y + 1].set_index("MemberICPSR"); nb = t2[t2.Y == Y + 1].set_index("MemberICPSR")
    cn = na.index.intersection(nb.index)
    nxt = True if len(cn) == 0 else (not np.allclose(na.loc[cn, cols].values.astype(float), nb.loc[cn, cols].values.astype(float), equal_nan=True))
    res[Y] = {"features_unchanged_on_target_year_rows": same, "target_changed": bool(tgt_changed),
              "next_year_features_change": bool(nxt), "n_common": int(len(common))}
    ok = ok and same and tgt_changed
print(json.dumps(res)); print("LEAK_TEST_PASS", ok)
json.dump({"pass": bool(ok), "by_year": res}, open(OUT / "leak_test.json", "w"), indent=1)
