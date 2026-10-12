"""P1 step 6: assemble numbers_p1.json from the stage outputs.  Usage: python p1_numbers.py --source subset|full"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from p1_common import *  # noqa
a = get_args()
O = outdir(a.source)
rd = lambda f: json.loads((O / f).read_text())
num = dict(source=a.source, build=rd("build_meta.json"), hyperparameters=rd("hp.json"),
           dev=rd("metrics_dev.json"), test=rd("metrics_test.json"), final=rd("final.json"), lee=rd("lee.json"))
for st in ("dev", "test"):
    s = pd.read_csv(O / f"skill_{st}.csv")
    num[f"skill_table_{st}"] = s.to_dict("records")
    num[f"horizon_{st}"] = pd.read_csv(O / f"horizon_{st}.csv").to_dict("records")
    num[f"heterogeneity_{st}"] = pd.read_csv(O / f"hetero_{st}.csv").to_dict("records")
num["pdp"] = pd.read_csv(O / "pdp.csv").to_dict("records")
num["empirical_bins"] = pd.read_csv(O / "empirical_bins.csv").to_dict("records")
(O / "numbers_p1.json").write_text(json.dumps(num, indent=1, default=float))
print("ok", len(json.dumps(num, default=float)))
