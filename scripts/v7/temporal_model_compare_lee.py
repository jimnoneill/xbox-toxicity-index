"""Lee archive design for the model comparison (same construction as temporal_lee.py, full window, nonrt)."""
import sys, pickle
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts/v7"))
from xbox import location as lc, temporal_v2 as tv, hawkes as hk
import temporal_features as tf, temporal_lee as tl
df = tl.load_archive(ROOT / "data/BasedMikeLee_archive_v2.parquet")
md, cd, mch = lc.load_vote_days(ROOT / "data/voteview")
ml = lc.MemberLocation(md[tl.LEE], cd["S"], "UT", name="Lee")
sel = tv.nonretweet_mask_lee(df.type.values)
t = np.sort(df.t.values[sel].astype("int64"))
slot0, m, k, nm = tf.slot_arrays(ml, t[0], t[-1], tf.PRIMARY)
d = hk.make_design(t, m, k, slot0, nm, jitter_seed=12345)
out = {"LEE": dict(t=d["t"], slot_month=d["slot_month"], slot_bin=d["slot_bin"], nm=int(d["occ"].shape[0]), seed=12345, n=len(t))}
pickle.dump(out, open(ROOT / "output/v7/temporal/model_compare/cache_lee.pkl", "wb"))
print("lee n", len(t), nm)
