"""Build the per-member design cache for the model comparison (read only on existing outputs)."""
import sys, pickle, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts/v7"))
import temporal_hawkes_diag as dg
t0 = time.time()
pick = dg._load(100000)
print("members", len(pick), round(time.time()-t0), flush=True)
out = {}
for bg in pick:
    t, d, seed = dg.member_design(bg)
    out[bg] = dict(t=d["t"], slot_month=d["slot_month"], slot_bin=d["slot_bin"], nm=int(d["occ"].shape[0]), seed=int(seed), n=len(t))
pickle.dump(out, open(ROOT / "output/v7/temporal/model_compare/cache.pkl", "wb"))
print("done", round(time.time()-t0), flush=True)
