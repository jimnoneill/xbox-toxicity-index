"""Where does the best descriptive model (casc3) miss? Observed vs model-predicted share of gaps by length class."""
import sys, json, pickle
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path.home() / "x_box/pipeline"
sys.path.insert(0, str(ROOT))
from xbox import pointprocess_models as pm
OUT = ROOT / "output/v7" / sys.argv[1] / "model_compare"
mm = pd.read_csv(OUT / "members_models.csv")
mm = mm[(mm.model == "casc3") & (mm.bg != "LEE")]
cache = pickle.load(open(OUT / "cache.pkl", "rb"))
edges_s = np.array([0, 1, 5, 30, 120, 600, 3600, 6 * 3600, 24 * 3600, np.inf])
names = ["<1s", "1-5s", "5-30s", "30s-2m", "2-10m", "10-60m", "1-6h", "6-24h", ">24h"]
edges = edges_s / 60.0
obs = np.zeros(len(names)); pred = np.zeros(len(names)); rows = []
for _, r in mm.iterrows():
    d = pm.calendar_from_cache(cache[r.bg]); h = d.hw()
    P = json.loads(r.params)
    s = P["session_rate_per_day"] / 1440.0; w0 = P["w_idle"]
    fast = [(P["w1"], P["rate1"]), (P["w2"], P["rate2"])]
    t0 = d.t[:-1]; g = np.diff(d.t)
    def lam(x):
        x = np.minimum(x, d.T - 1e-9); sl = np.minimum((x // 60).astype(int), d.ns - 1)
        return h["cum"][sl] + h["rate"][sl] * (x - sl * 60.0)
    L0 = lam(t0)
    def F(x):
        if np.isinf(x): return np.ones_like(t0)
        S = w0 * np.exp(-s * (lam(t0 + x) - L0)) + sum(w * np.exp(-q * x) for w, q in fast)
        return 1 - S
    cdfs = np.array([F(e) for e in edges])        # (edges, ngaps)
    predm = (cdfs[1:] - cdfs[:-1]).sum(axis=1)
    ob = np.histogram(g, bins=edges)[0]
    obs += ob; pred += predm
    rows.append(dict(bg=r.bg, n=len(g), **{f"obs_{k}": o / len(g) for k, o in zip(names, ob)}, **{f"pred_{k}": p / len(g) for k, p in zip(names, predm)}))
pd.DataFrame(rows).to_csv(OUT / "casc3_scale_misfit.csv", index=False)
out = pd.DataFrame(dict(cls=names, observed_share=obs / obs.sum(), predicted_share=pred / pred.sum()))
out["obs_over_pred"] = out.observed_share / out.predicted_share
print(out.round(4).to_string(index=False))
df = pd.DataFrame(rows)
print("median per-member observed/predicted share by class")
for k in names:
    print(k, round(float((df[f'obs_{k}'] / df[f'pred_{k}'].clip(lower=1e-9)).median()), 3))
