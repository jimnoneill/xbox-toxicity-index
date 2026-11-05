"""Descriptive contiguous monthly holdout-block sensitivity from saved predictions.
No independent-month assumption, resampling, or confidence interval is claimed.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent


def metrics(n, sa, sb, oa, ob):
    n=int(n)
    return {"n":n,"strict_mse_content":float(sa/n),"strict_mse_rhythm":float(sb/n),
            "strict_delta_mse":float((sa-sb)/n),
            "strict_relative_mse_reduction":float((sa-sb)/sa),
            "legacy_delta_mse":float((oa-ob)/n),
            "strict_minus_legacy_delta_mse":float(((sa-sb)-(oa-ob))/n)}


def block_sensitivity(strict,legacy):
    """Pair held-out rows, aggregate squared errors by monthly origin, then omit blocks."""
    names=["thread_id","acct","ti7","content","rhythm","origin"]
    if strict.thread_id.duplicated().any() or legacy.thread_id.duplicated().any():
        raise ValueError("duplicate event ID")
    pair=strict[names].merge(legacy[names],on=["thread_id","acct","ti7","origin"],
                             suffixes=("_strict","_old"),validate="one_to_one")
    if pair.empty or len(pair)!=len(strict):raise ValueError("strict rows not fully paired with legacy")
    y=pair.ti7.to_numpy(dtype=float)
    losses=pd.DataFrame({"origin":pair.origin,
         "sa":(y-pair.content_strict.to_numpy(dtype=float))**2,
         "sb":(y-pair.rhythm_strict.to_numpy(dtype=float))**2,
         "oa":(y-pair.content_old.to_numpy(dtype=float))**2,
         "ob":(y-pair.rhythm_old.to_numpy(dtype=float))**2})
    grouped=losses.groupby("origin",sort=True).agg(n=("sa","size"),sa=("sa","sum"),
               sb=("sb","sum"),oa=("oa","sum"),ob=("ob","sum"))
    if len(grouped)<2:raise ValueError("need at least two contiguous time blocks")
    idx=pd.to_datetime(grouped.index)
    expected=pd.date_range(idx.min(),idx.max(),freq="MS")
    if not idx.equals(expected):raise ValueError("monthly blocks must be contiguous")
    totals=grouped.sum(axis=0)
    overall=metrics(*totals[["n","sa","sb","oa","ob"]].to_numpy())
    per=[]; leave=[]
    for origin,row in grouped.iterrows():
        vals=row[["n","sa","sb","oa","ob"]].to_numpy()
        per.append({"origin":origin,**metrics(*vals)})
        leave.append({"omitted_origin":origin,**metrics(*(totals[["n","sa","sb","oa","ob"]].to_numpy()-vals))})
    return {"n_common":len(pair),"legacy_only_rows":len(legacy)-len(pair),
            "n_months":len(per),"positive_gain_months":sum(x["strict_delta_mse"]>0 for x in per),
            "positive_paired_difference_months":sum(x["strict_minus_legacy_delta_mse"]>0 for x in per),
            "overall":overall,"per_month":per,"leave_one_month_out":leave,
            "interpretation":"Descriptive monthly sensitivity only: 13 contiguous/serially dependent test blocks (last month partial), no calibrated block bootstrap or CI."}


def main():
    def load_arm(name):
        paths=sorted((HERE/"predictions"/name).glob("origin_*.parquet"))
        if len(paths)!=13:raise AssertionError(f"expected 13 files, got {len(paths)} for {name}")
        return pd.concat((pd.read_parquet(p) for p in paths),ignore_index=True)
    result=block_sensitivity(load_arm("combined_conditional_strict"),load_arm("legacy_reproduction"))
    (HERE/"temporal_block_sensitivity.json").write_text(json.dumps(result,indent=2))
    print(json.dumps({k:result[k] for k in ("n_common","n_months","positive_gain_months","positive_paired_difference_months","overall")},indent=2))

if __name__=="__main__":main()
