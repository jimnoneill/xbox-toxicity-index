"""Assemble post_probs_new.parquet / post_probs_full.parquet from output/v7/full/_probs_new/*.npy and check the
2,000-post overlap sample against stored probabilities."""
import os, json, numpy as np, pandas as pd
R = os.path.expanduser("~/x_box/pipeline"); O = f"{R}/output/v7/full"; P = f"{O}/_probs_new"
old = pd.read_parquet(f"{R}/output/v7/ti/post_probs.parquet")
t = pd.read_parquet(f"{O}/score_input.parquet").drop(columns="text")
heads = [c[2:] for c in old.columns if c.startswith("p_")]
for h in heads:
    t[f"p_{h}"] = np.load(f"{P}/{h}.npy")
for h in ["neg_sentiment", "emotion_anger"]:
    t[f"argmax_{h}"] = np.load(f"{P}/{h}_argmax.npy").astype(np.uint8)
smp = t[t.is_sample].drop(columns="is_sample"); new = t[~t.is_sample].drop(columns="is_sample").reset_index(drop=True)
m = smp[["id"] + [c for c in old.columns if c.startswith(("p_", "argmax_"))]].merge(
    old.drop_duplicates("id", keep=False), on="id", suffixes=("_n", "_o"))
rows = []
for c in [c for c in old.columns if c.startswith(("p_", "argmax_"))]:
    d = (m[c + "_n"].astype(float) - m[c + "_o"].astype(float)).abs()
    rows.append(dict(col=c, n=len(d), max_abs_diff=d.max(), mean_abs_diff=d.mean()))
rep = pd.DataFrame(rows); rep.to_csv(f"{O}/overlap_check.csv", index=False); print(rep.to_string())
new = new[old.columns]
new.to_parquet(f"{O}/post_probs_new.parquet", index=False)
full = pd.concat([old, new], ignore_index=True)
for c in old.columns:
    if str(old[c].dtype) != str(full[c].dtype): full[c] = full[c].astype(old[c].dtype)
full.to_parquet(f"{O}/post_probs_full.parquet", index=False)
print("new", new.shape, "full", full.shape, "dup ids in full", int(full.id.duplicated().sum()))
