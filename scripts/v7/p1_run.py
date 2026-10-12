"""P1 step 2: rolling-origin evaluation (expanding window, monthly origins) of baselines, mixed models and boosted trees.

Stages (run in this order):
  tune : hyperparameters (leaves, min leaf, rounds) for every GBM task/feature set on train < 2019-01-01, validation
         2019-01..2019-06 (entirely before every forecast origin).  Writes hp.json.
  dev  : monthly origins 2019-07-01 .. 2022-06-01.  Rows on or after 2022-07-01 are removed from memory.
  test : the untouched final test, monthly origins 2022-07-01 .. 2023-07-01 (expanding window includes earlier test
         months), same frozen hyperparameters.  Run once, after dev is final.
Output: <out>/hp.json, <out>/preds_<stage>/origin_YYYY-MM.parquet, <out>/fit_info_<stage>.json
Usage : python p1_run.py --source subset|full --stage tune|dev|test
Seed 20261010.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p1_models import *  # noqa

a = get_args(lambda ap: ap.add_argument("--stage", choices=["tune", "dev", "test"], required=True))
OUT = outdir(a.source)
t0 = time.time()
F = pd.read_parquet(OUT / "features.parquet")
F["when"] = pd.to_datetime(F["when"])
if a.stage != "test":
    F = F[F.when < fp.TEST_START].reset_index(drop=True)      # the final test period is not even loaded
print(a.stage, "rows", len(F), flush=True)

if a.stage == "tune":
    tr, va, J, _ = prepare_origin(F, TUNE_TRAIN_END, TUNE_VAL_END)
    hp = tune(tr, va)
    (OUT / "hp.json").write_text(json.dumps(hp, indent=1))
    print(json.dumps(hp), round(time.time() - t0), "s")
    sys.exit(0)

hp = json.loads((OUT / "hp.json").read_text())
pdir = OUT / f"preds_{a.stage}"
pdir.mkdir(exist_ok=True)
infos = json.loads((OUT / f"fit_info_{a.stage}.json").read_text()) if (OUT / f"fit_info_{a.stage}.json").exists() else {}
meta_cols = ["thread_id", "acct", "screen_name", "when", "ti7", "y_bina", "y_binb", "thr_a", "prev_ti7", "roll50", "origin",
             "horizon_days"]
for o in origins("dev" if a.stage == "dev" else "test"):
    f = pdir / f"origin_{o:%Y-%m}.parquet"
    end = window_end(o)
    if f.exists():
        continue
    tr, te, J, thr_g = prepare_origin(F, o, end)
    if len(te) == 0:
        continue
    assert tr.when.max() < o <= te.when.min() and te.when.max() < end      # strict temporal separation
    pred, info = run_split(tr, te, hp, J)
    te["origin"] = o
    te["horizon_days"] = ((te.when - o).dt.total_seconds() / 86400).astype("float32")
    res = te[meta_cols].copy()
    res["thr_g"] = thr_g
    for k, v in pred.items():
        res[k] = np.asarray(v, dtype="float32")
    res.to_parquet(f, index=False)
    infos[f"{o:%Y-%m}"] = dict(n_train=len(tr), n_test=len(te), accounts=J, **{k: v for k, v in info.items()})
    (OUT / f"fit_info_{a.stage}.json").write_text(json.dumps(infos, indent=0, default=float))
    print(f"{o:%Y-%m} train {len(tr)} test {len(te)} J {J} {round(time.time() - t0)} s", flush=True)
print("done", round(time.time() - t0), "s")
