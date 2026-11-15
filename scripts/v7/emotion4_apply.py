"""emotion4_apply.py: write the emotion model's four class probabilities into the stored score tables.

Before: p_emotion_anger and argmax_emotion_anger held the third output of the model (optimism), because the class was
looked up by name in the model's config.json. After: p_emotion_anger, p_emotion_joy, p_emotion_optimism and
p_emotion_sadness hold the four outputs in the order of the benchmark mapping file, and argmax_emotion_anger is 1 where
anger is the largest. Each table is checked first (same ids in the same order, and the earlier column equal to the new
optimism column), copied to <name>.before_emotion_order_fix.parquet, and replaced in one step. Run once.
"""
import glob
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.expanduser("~/x_box/pipeline")
V7 = f"{ROOT}/output/v7"
E4 = f"{V7}/emotion4"
EM = ["anger", "joy", "optimism", "sadness"]
LOG = {}


def apply(path, e4, key):
    bak = path.replace(".parquet", ".before_emotion_order_fix.parquet")
    if os.path.exists(bak):
        print("already done, skip", path)
        return
    t = pd.read_parquet(path)
    assert len(t) == len(e4), (path, len(t), len(e4))
    assert (t.id.astype(str).values == e4.id.astype(str).values).all(), f"{path}: ids differ"
    d = np.abs(t.p_emotion_anger.values.astype(np.float64) - e4.p_emotion_optimism.values.astype(np.float64))
    assert d.max() < 1e-3, f"{path}: earlier column is not the optimism output (max diff {d.max()})"
    agree = float((t.argmax_emotion_anger.values.astype(int) == (e4.emotion_class.astype(str).values == "optimism").astype(int)).mean())
    assert agree > 0.9999, f"{path}: earlier flag is not the optimism class ({agree})"
    cols = list(t.columns)
    for c in EM:
        t["p_emotion_" + c] = e4["p_emotion_" + c].values.astype(t.p_emotion_anger.dtype if c == "anger" else np.float32)
    t["argmax_emotion_anger"] = (e4.emotion_class.astype(str).values == "anger").astype(t.argmax_emotion_anger.dtype)
    i = cols.index("p_emotion_anger")
    order = cols[:i + 1] + ["p_emotion_joy", "p_emotion_optimism", "p_emotion_sadness"] + cols[i + 1:]
    tmp = path + ".tmp"
    t[order].to_parquet(tmp, index=False)
    os.replace(path, bak)
    os.replace(tmp, path)
    LOG[key] = dict(path=path.replace(ROOT + "/", ""), rows=int(len(t)), max_abs_diff_old_vs_optimism=float(d.max()), old_flag_is_optimism_class=agree,
                    anger_class_share=float(t.argmax_emotion_anger.mean()), backup=os.path.basename(bak))
    print(key, LOG[key], flush=True)


full = pd.read_parquet(f"{E4}/full.parquet")
n_old = len(pd.read_parquet(f"{V7}/ti/post_probs.parquet", columns=["id"]))
apply(f"{V7}/full/post_probs_full.parquet", full, "corpus")
apply(f"{V7}/ti/post_probs.parquet", full.iloc[:n_old].reset_index(drop=True), "sample_2017_2023")
apply(f"{V7}/full/post_probs_new.parquet", full.iloc[n_old:].reset_index(drop=True), "added_posts")
apply(f"{V7}/ti/post_probs_lee.parquet", pd.read_parquet(f"{E4}/lee.parquet"), "single_account")

# labelled 1,000-post file: the class label column
lp = glob.glob(f"{ROOT}/data/*_advocacy_anger_1k.parquet")[0]
bak = lp.replace(".parquet", ".before_emotion_order_fix.parquet")
if not os.path.exists(bak):
    L = pd.read_parquet(lp)
    e = pd.read_parquet(f"{E4}/labels1k.parquet")
    assert len(L) == len(e)
    ren = {"joy": "anger", "optimism": "joy", "anger": "optimism", "sadness": "sadness"}
    # The file holds post text cut at 400 characters, so a few borderline posts rescore differently from the full text.
    # The stored classes are therefore renamed, not replaced; the rescored classes serve as a check.
    new = L.cardiff_emotion.map(ren).values
    agree = float((new == e.emotion_class.astype(str).values).mean())
    assert agree >= 0.99, agree
    L2 = L.copy()
    L2["cardiff_emotion"] = new
    L2.to_parquet(lp + ".tmp", index=False)
    os.replace(lp, bak)
    os.replace(lp + ".tmp", lp)
    LOG["labels_1k"] = dict(rows=int(len(L)), renamed_old_label_agrees_with_rescored=agree,
                            before={k: int(v) for k, v in L.cardiff_emotion.value_counts().items()},
                            after={k: int(v) for k, v in L2.cardiff_emotion.value_counts().items()})
    print("labels_1k", LOG["labels_1k"], flush=True)
json.dump(LOG, open(f"{E4}/apply_log.json", "w"), indent=1)
open(f"{V7}/full/README_emotion_order_fix.txt", "w").write(
    "2026-11-15. p_emotion_anger and argmax_emotion_anger in post_probs_full.parquet, post_probs_new.parquet, ti/post_probs.parquet and\n"
    "ti/post_probs_lee.parquet now hold the anger class of the emotion model. Until this date they held its optimism class.\n"
    "The earlier files are kept beside them as *.before_emotion_order_fix.parquet. Scores: output/v7/emotion4 (scripts/v7/emotion4_score.py).\n")
print("DONE")
