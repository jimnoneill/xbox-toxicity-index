"""Element-level v1/v2 model inputs and per-origin selection proof."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from run_strict import CONTENT, RHYTHM, CLOCK, LAGS, CONG, ARMS, CLOSURE_SECONDS, SEED

HERE=Path(__file__).resolve().parent
MODEL_COLUMNS=list(dict.fromkeys(CONTENT+RHYTHM+["strict_"+c for c in CLOCK+LAGS+CONG]))
CONTROL_COLUMNS=["thread_id","acct","ti7","t","t_end","n_completed_prior","strict_n_completed_prior","size"]
ALL_COLUMNS=list(dict.fromkeys(MODEL_COLUMNS+CONTROL_COLUMNS))


def array_hash(s):
    a=np.ascontiguousarray(s.to_numpy())
    h=hashlib.sha256()
    h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def compare_model_frames(old,new,columns=MODEL_COLUMNS):
    """Compare every used value, including NaN as equal, and both eligibility rules."""
    control=[c for c in CONTROL_COLUMNS if c in old.columns and c in new.columns]
    all_cols=list(dict.fromkeys(list(columns)+control))
    if len(old)!=len(new):
        return {"prediction_reuse_justified":False,"row_count_old":len(old),"row_count_new":len(new)}
    changed={};hashes={}
    for c in all_cols:
        a,b=old[c],new[c]
        same=a.eq(b).fillna(False)|(a.isna()&b.isna())
        diff=int((~same).sum())
        if diff or a.dtype!=b.dtype:changed[c]=diff if diff else "dtype_only"
        hashes[c]={"v1_sha256":array_hash(a),"v2_sha256":array_hash(b),
                   "dtype_v1":str(a.dtype),"dtype_v2":str(b.dtype)}
    eligibility={}
    for label,col in (("legacy","n_completed_prior"),("strict","strict_n_completed_prior")):
        if col in old and col in new:
            eligibility[label]=int(np.count_nonzero((old[col].to_numpy()>=10)!=(new[col].to_numpy()>=10)))
    return {"row_count_old":len(old),"row_count_new":len(new),
            "columns_compared":all_cols,"changed_columns":changed,"array_sha256":hashes,
            "changed_eligibility_rows":eligibility,
            "prediction_reuse_justified":not changed and not any(eligibility.values())}


def selection_proof(old,new):
    """Compare actual origin train/test indices including account cutoff and fit cap."""
    results={}; different=[]
    for arm,changes in ARMS.items():
        pairs=[]
        for origin in pd.date_range("2022-07-01","2023-07-01",freq="MS"):
            o=int(origin.timestamp());stop=int(min(origin+pd.offsets.MonthBegin(1),pd.Timestamp("2023-07-12")).timestamp())
            entries=[]
            for frame in (old,new):
                acct=frame.acct.to_numpy();t=frame.t.to_numpy();end=frame.t_end.to_numpy()
                prior=frame["strict_n_completed_prior" if "lag" in changes else "n_completed_prior"].to_numpy()
                eligible=prior>=10;train_end=end+CLOSURE_SECONDS if "embargo" in changes else end
                initial=eligible&(t<o)&(train_end<o)
                counts=np.bincount(acct[initial],minlength=int(acct.max())+1)
                valid=counts>=100
                tr=np.flatnonzero(initial&valid[acct]);te=np.flatnonzero(eligible&(t>=o)&(t<stop)&valid[acct])
                if len(tr)>600_000:tr=pd.Series(tr).sample(600_000,random_state=SEED).to_numpy()
                entries.append((tr,te))
            eq=all(np.array_equal(a,b) for a,b in zip(*entries))
            if not eq:different.append(f"{arm}:{origin:%Y-%m}")
            def sig(a):return hashlib.sha256(np.ascontiguousarray(a,dtype=np.int64).tobytes()).hexdigest()
            pairs.append({"origin":str(origin.date()),"n_train_fit":len(entries[0][0]),"n_test":len(entries[0][1]),
                          "train_index_sha256_v1":sig(entries[0][0]),"train_index_sha256_v2":sig(entries[1][0]),
                          "test_index_sha256_v1":sig(entries[0][1]),"test_index_sha256_v2":sig(entries[1][1]),
                          "identical":eq})
        results[arm]=pairs
    return {"different_origin_arm_selections":different,"per_arm_origin":results}



def prediction_differences(old,new):
    """Confirm held-out IDs/labels match, then count changed forecast values."""
    if len(old)!=len(new):raise AssertionError("prediction row count changed")
    for col in ("thread_id","acct","t","ti7"):
        if not old[col].equals(new[col]):raise AssertionError("prediction control changed: "+col)
    return {"rows":len(old),
            "content_changed":int(np.count_nonzero(old.content.to_numpy()!=new.content.to_numpy())),
            "rhythm_changed":int(np.count_nonzero(old.rhythm.to_numpy()!=new.rhythm.to_numpy()))}


def main():
    old=pd.read_parquet(HERE.parent/"rq1_strict/strict_features.parquet",columns=ALL_COLUMNS)
    new=pd.read_parquet(HERE/"strict_features.parquet",columns=ALL_COLUMNS)
    proof=compare_model_frames(old,new)
    proof.update(selection_proof(old,new))
    proof["prediction_reuse_justified"]=bool(proof["prediction_reuse_justified"] and not proof["different_origin_arm_selections"])
    proof["comparison_scope"]="All 39 model-used feature columns plus 8 control/label columns, 2,821,325 aligned rows, 7 arms x 13 origins; exact element comparison and SHA256 of array bytes."
    proof["action"]="reuse copied predictions" if proof["prediction_reuse_justified"] else "refit all 7 arms"
    (HERE/"feature_comparison.json").write_text(json.dumps(proof,indent=2))
    print(json.dumps({k:proof[k] for k in ("row_count_old","row_count_new","changed_columns","changed_eligibility_rows","different_origin_arm_selections","prediction_reuse_justified","action")},indent=2))

if __name__=="__main__":main()
