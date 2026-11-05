"""Isolated RQ1 reconstruction. No writes outside this script directory.
Usage: ~/x_box/venv/bin/python -u run_strict.py --stage build|eval|manifest
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
assert ROOT.name=="pipeline" and HERE.name=="rq1_strict"
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(HERE.parent/"rq1")) # read-only baseline implementation, hashed in manifest
from rq1_arrival import fit_pair, metric_summary, score_events
from strict_features import (clocks_for_frame, repair_lags_strict, causal_congress_features,
                             CLOSURE_SECONDS)
from xbox import location as lc

T1=ROOT/"output/v7/t1full"
P1=ROOT/"output/v7/p1"
OLD=HERE.parent/"rq1"
DIMS=["offensive","toxic","neg_sentiment","irony","ragebait","tribal","performative"]
PROB=["p_"+x for x in DIMS]
CONTENT=["prev_ti7","roll10","roll50","prev_reply","prev_multi","reply_share10","own_reply"]
RHYTHM=["log_dt","sess_pos","sess_el","n1h","n24h","hour","hsin","hcos","h2sin","h2cos",
        "night","weekend","sched_share","cong_p1","cong_p24","cong_spike_lr","cong_spike"]
CLOCK=["hour","hsin","hcos","h2sin","h2cos","night","weekend"]
LAGS=["prev_ti7","roll10","roll50","prev_multi"]
CONG=["cong_p1","cong_p24","cong_spike_lr","cong_spike"]
SEED=20261010
ARMS={"legacy_reproduction":(), "local_clock_only":("clock",),
      "training_embargo_only":("embargo",), "feature_closure_only":("lag",),
      "congress_no_hindsight_imputation_only":("cong",),
      "clock_plus_training_embargo":("clock","embargo"),
      "combined_conditional_strict":("clock","embargo","lag","cong")}


def build():
    frame=pd.read_parquet(OLD/"arrival_features.parquet")
    events=pd.read_parquet(T1/"events.parquet",columns=["thread_id","screen_name","t","t_end","size"])
    posts=pd.read_parquet(T1/"posts_threads.parquet",columns=["id","thread_id"])
    probs=pd.read_parquet(ROOT/"output/v7/full/post_probs_full.parquet",columns=["id"]+PROB)
    events=score_events(events,posts,probs,PROB)
    del posts,probs
    match=events.set_index("thread_id").ti7.reindex(frame.thread_id).to_numpy()
    if not (np.isfinite(match).all() and np.max(np.abs(match-frame.ti7.to_numpy()))<2e-6):
        raise AssertionError("target score mismatch")
    strict=repair_lags_strict(events).set_index("thread_id")
    for c in LAGS+["n_completed_prior","prior_inflight"]:
        frame["strict_"+c]=strict[c].reindex(frame.thread_id).to_numpy()
    del strict
    print("scored events and closure lags",len(events),flush=True)
    amap=pd.read_csv(ROOT/"output/v7/ti/account_member_map.csv")
    vote,_,_=lc.load_vote_days(ROOT/"data/voteview")
    members=pd.read_csv(ROOT/"data/voteview_all_members.csv")
    table=lc.load_district_table(ROOT/"data/district_zone/district_zone.csv")
    clock,clock_audit=clocks_for_frame(frame,amap,vote,members,table)
    for c in CLOCK:
        frame["strict_"+c]=clock[c].to_numpy()
    frame["strict_loc_class"]=clock.strict_loc_class.to_numpy()
    del clock,amap,vote,members,table
    print("local clocks",clock_audit,flush=True)
    hours=pd.read_parquet(ROOT/"output/v7/t2full/congress_hourly.parquet",columns=["hour_idx","total_events"])
    hmin=int(hours.hour_idx.min())
    counts=np.bincount(events.t.to_numpy(dtype=np.int64)//3600-hmin,minlength=len(hours))
    if not np.array_equal(counts,hours.total_events.to_numpy()) or not np.array_equal(hours.hour_idx.to_numpy(),np.arange(hmin,hmin+len(hours))):
        raise AssertionError("T2 Congress event count mismatch; refuse approximate reconstruction")
    cg=causal_congress_features(counts)
    hidx=frame.t.to_numpy(dtype=np.int64)//3600-hmin
    for c in CONG:
        frame["strict_"+c]=cg[c][hidx].astype(np.float32)
    diffs={c:int((~np.isclose(frame[c].to_numpy(),frame["strict_"+c].to_numpy(),equal_nan=True,rtol=0,atol=1e-6)).sum()) for c in CONG}
    print("Congress lags without hindsight outage imputation",diffs,flush=True)
    # Distinguish the required non-split-state test subset from changed split-state seats.
    st=pd.read_csv(ROOT/"output/v7/ti/account_member_map.csv")
    st=st[st.entity_type=="member"].drop_duplicates("bioguide").set_index("bioguide").state
    state=frame.bioguide.map(st)
    test=(frame.t >= int(pd.Timestamp("2022-07-01").timestamp())) & (frame.t < int(pd.Timestamp("2023-07-12").timestamp()))
    non_split=~state.isin(lc.SPLIT_STATES)
    change=(frame.hour!=frame.strict_hour)
    audit={"rows":int(len(frame)),"events_all":int(len(events)),"clock":clock_audit,
           "test_rows_changed_hour_non_split_state":int((test&non_split&change).sum()),
           "test_rows_changed_hour_all":int((test&change).sum()),
           "rows_changed_score_lag":int((~np.isclose(frame.prev_ti7,frame.strict_prev_ti7,equal_nan=True)).sum()),
           "rows_under_10_closed_prior":int((frame.strict_n_completed_prior<10).sum()),
           "congress_changed_columns":diffs,
           "closure_seconds":CLOSURE_SECONDS,
           "origin_label_incidence_before_account_filter":{}}
    for origin in pd.date_range("2022-07-01","2023-07-01",freq="MS"):
        o=int(origin.timestamp())
        mask=(frame.t<o)&(frame.t_end<o)&(frame.t_end+CLOSURE_SECONDS>=o)&(frame.n_completed_prior>=10)
        audit["origin_label_incidence_before_account_filter"][str(origin.date())]=int(mask.sum())
    audit["origin_label_incidence_total_before_account_filter"]=sum(audit["origin_label_incidence_before_account_filter"].values())
    frame.to_parquet(HERE/"strict_features.parquet",index=False)
    (HERE/"audit.json").write_text(json.dumps(audit,indent=2))
    print("BUILT",{k:audit[k] for k in ("rows","test_rows_changed_hour_non_split_state","rows_under_10_closed_prior","origin_label_incidence_total_before_account_filter")},flush=True)


def extended_metric(d,boot=1000):
    m=metric_summary(d,boot=boot,seed=SEED)
    if not len(d): return m
    a=(d.ti7.to_numpy(dtype=float)-d.content.to_numpy(dtype=float))**2
    b=(d.ti7.to_numpy(dtype=float)-d.rhythm.to_numpy(dtype=float))**2
    agg=pd.DataFrame({"acct":d.acct.to_numpy(),"a":a,"b":b}).groupby("acct").agg(sa=("a","sum"),sb=("b","sum"))
    v=agg.to_numpy()
    rng=np.random.default_rng(SEED)
    draws=rng.integers(0,len(v),size=(boot,len(v)))
    aa=v[draws,0].sum(axis=1); bb=v[draws,1].sum(axis=1)
    m["ci95_relative_mse_reduction"]=np.quantile((aa-bb)/aa,[.025,.975]).tolist()
    return m


def evaluate():
    f=pd.read_parquet(HERE/"strict_features.parquet")
    n=len(f); acct=f.acct.to_numpy(); t=f.t.to_numpy(); end=f.t_end.to_numpy()
    prev=f.n_completed_prior.to_numpy(); closed=f.strict_n_completed_prior.to_numpy()
    pred_dir=HERE/"predictions";pred_dir.mkdir(exist_ok=True)
    summary={}; monthly={}
    for arm, changes in ARMS.items():
        sub=pred_dir/arm;sub.mkdir(exist_ok=True)
        columns={c:("strict_"+c if ("clock" in changes and c in CLOCK) or ("lag" in changes and c in LAGS) or ("cong" in changes and c in CONG) else c)
                 for c in CONTENT+RHYTHM}
        eligible=(closed if "lag" in changes else prev)>=10
        train_end=end+CLOSURE_SECONDS if "embargo" in changes else end
        out=[];month=[]
        for origin in pd.date_range("2022-07-01","2023-07-01",freq="MS"):
            o=int(origin.timestamp());stop=int(min(origin+pd.offsets.MonthBegin(1),pd.Timestamp("2023-07-12")).timestamp())
            initial=eligible & (t<o) & (train_end<o)
            account_counts=np.bincount(acct[initial],minlength=int(acct.max())+1)
            valid=account_counts>=100
            trids=np.flatnonzero(initial & valid[acct]); teids=np.flatnonzero(eligible & (t>=o)&(t<stop)&valid[acct])
            if not len(teids): continue
            if len(trids)>600_000:
                trids=pd.Series(trids).sample(600_000,random_state=SEED).to_numpy()
            def subset(ids):
                z=f.iloc[ids][["acct","ti7"]+list(dict.fromkeys(columns.values()))].copy()
                return z.rename(columns={v:k for k,v in columns.items() if v!=k})
            tr,te=subset(trids),subset(teids)
            if not (end[trids]+(CLOSURE_SECONDS if "embargo" in changes else 0)<o).all():
                raise AssertionError("training label not closed by origin")
            p=fit_pair(tr,te,CONTENT,RHYTHM)
            rec=f.iloc[teids][["thread_id","acct","t","t_end","size","ti7"]].copy()
            rec["content"]=p["content"];rec["rhythm"]=p["rhythm"]
            rec["origin"]=str(origin.date())
            rec.to_parquet(sub/f"origin_{origin:%Y-%m}.parquet",index=False)
            out.append(rec)
            month.append({"origin":str(origin.date()),"n_train_eligible":int(initial.sum()),"n_train_after_account_filter":int((initial & valid[acct]).sum()),
                          "n_train_fit":int(len(trids)),"n_test":int(len(teids)),"accounts_train":int(valid.sum()),
                          **extended_metric(rec,boot=300)})
            print(arm,origin.date(),"train",int(initial.sum()),"test",len(teids),flush=True)
        pred=pd.concat(out,ignore_index=True)
        if pred.thread_id.duplicated().any(): raise AssertionError("duplicate prediction event")
        summary[arm]={"changes":list(changes),"primary":extended_metric(pred),
                      "single_post":extended_metric(pred[pred["size"]==1]),
                      "n_train_fit_months":len(month)}
        monthly[arm]=month
        print("RESULT",arm,summary[arm]["primary"],flush=True)
    (HERE/"evaluation.json").write_text(json.dumps(summary,indent=2))
    (HERE/"monthly_metrics.json").write_text(json.dumps(monthly,indent=2))
    # Legacy identity check against the read-only archived predictions and metrics.
    orig=json.loads((OLD/"evaluation.json").read_text())["primary"]
    got=summary["legacy_reproduction"]["primary"]
    if got["n"]!=orig["n"] or abs(got["delta_mse"]-orig["delta_mse"])>2e-7:
        raise AssertionError("baseline did not reproduce archived 0.44% result")
    # Paired difference in MSE gains on the common held-out rows, account clustered.
    def load_arm(arm):
        return pd.concat((pd.read_parquet(x) for x in sorted((pred_dir/arm).glob("origin_*.parquet"))),ignore_index=True)
    a=load_arm("legacy_reproduction");z=load_arm("combined_conditional_strict")
    merged=a[["thread_id","acct","ti7","content","rhythm"]].merge(z[["thread_id","acct","ti7","content","rhythm"]],on=["thread_id","acct","ti7"],suffixes=("_old","_new"),validate="one_to_one")
    y=merged.ti7.to_numpy(dtype=float)
    d_old=(y-merged.content_old.to_numpy())**2-(y-merged.rhythm_old.to_numpy())**2
    d_new=(y-merged.content_new.to_numpy())**2-(y-merged.rhythm_new.to_numpy())**2
    gain=d_new-d_old
    agg=pd.DataFrame({"acct":merged.acct,"gain":gain}).groupby("acct").agg(n=("gain","size"),g=("gain","sum")).to_numpy()
    rng=np.random.default_rng(SEED);draws=rng.integers(0,len(agg),size=(1000,len(agg)))
    bs=agg[draws,1].sum(axis=1)/agg[draws,0].sum(axis=1)
    paired={"n_common":int(len(merged)),"accounts":int(len(agg)),"strict_minus_legacy_delta_mse":float(gain.mean()),
            "ci95":np.quantile(bs,[.025,.975]).tolist(),
            "legacy_only_test_rows":int(len(a)-len(merged)),"strict_only_test_rows":int(len(z)-len(merged))}
    (HERE/"paired_comparison.json").write_text(json.dumps(paired,indent=2))
    print("PAIRED",paired,flush=True)


def manifest():
    files={}
    for p in sorted(HERE.rglob("*")):
        if not p.is_file() or p.name=="manifest.json" or "__pycache__" in p.parts or ".pytest_cache" in p.parts:
            continue
        sha=hashlib.sha256()
        with p.open("rb") as f:
            for block in iter(lambda:f.read(4*1024*1024),b""): sha.update(block)
        files[str(p.relative_to(HERE))]={"bytes":p.stat().st_size,"sha256":sha.hexdigest()}
    source_hash={}
    inputs=[OLD/"rq1_arrival.py",OLD/"arrival_features.parquet",OLD/"evaluation.json",
            P1/"features.parquet",T1/"events.parquet",T1/"posts_threads.parquet",
            ROOT/"output/v7/full/post_probs_full.parquet",ROOT/"output/v7/t2full/congress_hourly.parquet",
            ROOT/"data/voteview_all_members.csv",ROOT/"data/district_zone/district_zone.csv",
            ROOT/"output/v7/ti/account_member_map.csv",ROOT/"xbox/location.py",
            ROOT/"xbox/forecast_p1.py",ROOT/"scripts/v7/p1_build.py",ROOT/"scripts/v7/t1_common.py"]
    inputs+=sorted((ROOT/"data/voteview").glob("*_votes.csv"))
    inputs+=sorted((ROOT/"data/voteview").glob("*_rollcalls.csv"))
    for p in inputs:
        sha=hashlib.sha256()
        with p.open("rb") as f:
            for block in iter(lambda:f.read(4*1024*1024),b""):sha.update(block)
        source_hash[str(p)]={"sha256":sha.hexdigest(),"bytes":p.stat().st_size}
    (HERE/"manifest.json").write_text(json.dumps({"files":files,"read_only_inputs":source_hash,
      "seed":SEED,"design":"Monthly expanding 2022-07..2023-07; 600000 deterministic cap; alpha10 LSQR; 1000 account bootstrap"},indent=2))
    print("MANIFEST",len(files),"local files",flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--stage",choices=["build","eval","manifest"],required=True)
    args=parser.parse_args()
    {"build":build,"eval":evaluate,"manifest":manifest}[args.stage]()
