# logged deviation from rvt_c_DESIGN.md: C5 repeated on the losing-minus-gaining party difference, because early January is a seasonal low-to-high point for both parties
import numpy as np, pandas as pd, statsmodels.api as sm, os, json
exec(open("rvt_c_01_state.py").read().split("# ---------------------------------------------------------------- responses to the general-election window")[0])
D["lev"]=np.log1p(D.events); pm_=D.groupby(["party","date"]).lev.mean().unstack(0)
LOSER={"2019-01-03":"R","2021-01-20":"R","2023-01-03":"D"}; W=120
def fit(s,c):
    t=(s.index-c).days.values.astype(float); post=(t>=0).astype(float)
    dow=pd.get_dummies(s.index.dayofweek,drop_first=True).astype(float).values
    base=np.column_stack([np.ones(len(t)),t,dow])
    r1=sm.OLS(s.values,np.column_stack([base,post])).fit(); r2=sm.OLS(s.values,np.column_stack([base,t*post])).fit()
    rh=sm.OLS(s.values,np.column_stack([base,post])).fit(cov_type="HAC",cov_kwds={"maxlags":7}); k=base.shape[1]
    return float(r1.params[k]),float(rh.bse[k]),float(r2.bic-r1.bic)
out={}
for tr,p in LOSER.items():
    c=pd.Timestamp(tr); g="D" if p=="R" else "R"
    for nm,s in (("loser",pm_[p]),("gainer",pm_[g]),("loser_minus_gainer",pm_[p]-pm_[g])):
        s=s[(s.index>=c-pd.Timedelta(days=W))&(s.index<=c+pd.Timedelta(days=W))&(s.index!=c)].dropna()
        out[tr+" "+nm]=[round(x,3) for x in fit(s,c)]
    # same day one year earlier, loser party
    c2=c-pd.Timedelta(days=365); s=pm_[p]; s=s[(s.index>=c2-pd.Timedelta(days=W))&(s.index<=c2+pd.Timedelta(days=W))&(s.index!=c2)].dropna()
    out[tr+" loser_same_day_year_before"]=[round(x,3) for x in fit(s,c2)]
json.dump(out,open(O+"/c5_deviation.json","w"),indent=1); print(json.dumps(out,indent=1))
