"""Build output/v7/full/score_input.parquet: new posts (ids not in post_probs.parquet, first occurrence of each id)
followed by a 2,000-post overlap sample of already-scored posts (is_sample=True) for reproducibility checking."""
import os, numpy as np, pandas as pd
R = os.path.expanduser("~/x_box/pipeline"); O = f"{R}/output/v7/full"
n = pd.read_parquet(f"{R}/data/congresstweets_full/congresstweets_full.parquet",
                    columns=["id", "screen_name", "user_id", "created_at", "text"])
old = pd.read_parquet(f"{R}/output/v7/ti/post_probs.parquet", columns=["id"])
oldids = set(old.id.astype(str))
n["id"] = n["id"].astype(str)
isnew = ~n.id.isin(oldids)
new = n[isnew].drop_duplicates("id", keep="first").copy()
print("full rows", len(n), "new rows", int(isnew.sum()), "new unique ids", len(new), "dup-id rows dropped", int(isnew.sum()) - len(new))
cand = n[~isnew].drop_duplicates("id", keep=False)
smp = cand.sample(2000, random_state=0).copy()
new["is_sample"] = False; smp["is_sample"] = True
t = pd.concat([new, smp], ignore_index=True)
s = t.text.astype(str)
t["post_type"] = np.where(s.str.startswith("RT @"), "retweet", np.where(s.str.startswith("@"), "reply", "original"))
t.to_parquet(f"{O}/score_input.parquet", index=False)
print("score_input", len(t), "post types", t[~t.is_sample].post_type.value_counts().to_dict())
