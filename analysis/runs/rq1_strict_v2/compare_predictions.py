"""Element comparison of all 91 regenerated v2 origin forecasts against v1."""
import json
from pathlib import Path
import pandas as pd
from compare_features import prediction_differences
from run_strict import ARMS
HERE=Path(__file__).resolve().parent
old=HERE.parent/"rq1_strict/predictions"
report={}
for arm in ARMS:
    rows=[]
    old_files=sorted((old/arm).glob("origin_*.parquet"))
    new_files=sorted((HERE/"predictions"/arm).glob("origin_*.parquet"))
    if len(old_files)!=len(new_files) or len(new_files)!=13:raise AssertionError("missing origin forecasts "+arm)
    for a,b in zip(old_files,new_files):
        if a.name!=b.name:raise AssertionError("origin file mismatch")
        d=prediction_differences(pd.read_parquet(a),pd.read_parquet(b))
        rows.append({"origin_file":a.name,**d})
    report[arm]={"files":len(rows),"rows":sum(x["rows"] for x in rows),
                 "months_changed":sum(bool(x["content_changed"] or x["rhythm_changed"]) for x in rows),
                 "content_changed":sum(x["content_changed"] for x in rows),
                 "rhythm_changed":sum(x["rhythm_changed"] for x in rows),"per_origin":rows}
(HERE/"prediction_comparison.json").write_text(json.dumps(report,indent=2))
print(json.dumps({arm:{k:v for k,v in d.items() if k!="per_origin"} for arm,d in report.items()},indent=2))
