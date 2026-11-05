import numpy as np, pandas as pd, statsmodels.api as sm, json
O='/home/joneill/x_box/pipeline/output/v7/rvt_b'
rng=np.random.default_rng(20261013)
md=pd.read_parquet(O+'/member_day.parquet',columns=['date','bioguide','ti7','party'])
EV={'2019-01-03':'R','2020-11-03':'R','2021-01-20':'R','2023-01-03':'D'}  # loser party
EVD=[pd.Timestamp(k) for k in EV]; W=120
dmin,dmax=md.date.min(),md.date.max()
def series(party,center,stayers):
    lo,hi=center-pd.Timedelta(days=W),center+pd.Timedelta(days=W)
    x=md[(md.party==party)&(md.date>=lo)&(md.date<=hi)]
    if stayers:
        days=(hi-lo).days+1
        # in office every day: require presence in calendar-based panel; proxy = member posts on first and last 14 days of window and is in md (in_office rows only exist for posting days) -> use calendar below
        x=x[x.bioguide.isin(STAY(party,lo,hi))]
    s=x.groupby('date').ti7.mean()
    s=s[s.index!=center]
    return s
cal=pd.read_parquet('/home/joneill/x_box/pipeline/output/v7/t3/calendar_member_day.parquet',columns=['date','bioguide','in_office','party'])
cal['bioguide']=cal.bioguide.astype(str); cal=cal[cal.in_office==1]
cal=cal[(cal.date>=dmin-pd.Timedelta(days=130))&(cal.date<=dmax)]
cnt=cal.groupby(['bioguide']).date.agg(['min','max'])
def STAY(party,lo,hi):
    # in office every day lo..hi: count in_office days == window length
    c=cal[(cal.date>=lo)&(cal.date<=hi)&(cal.party==party)].groupby('bioguide').size()
    return set(c[c>=(hi-lo).days+1].index)
def design(s,center,model):
    t=(s.index-center).days.values.astype(float)
    X=pd.DataFrame({'c':1.0,'t':t},index=s.index)
    dow=pd.get_dummies(s.index.dayofweek,prefix='d',drop_first=True).astype(float); dow.index=s.index
    post=(t>=0).astype(float)
    if model in('ii','iv'): X['step']=post
    if model in('iii','iv'): X['ramp']=t*post
    return pd.concat([X,dow],axis=1)
def fit_all(s,center,hac=True):
    out={}
    for m in('i','ii','iii','iv'):
        X=design(s,center,m); r=sm.OLS(s.values,X.values).fit()
        d={'aic':r.aic,'bic':r.bic}
        if hac:
            rh=sm.OLS(s.values,X.values).fit(cov_type='HAC',cov_kwds={'maxlags':7})
            for nm in('step','ramp'):
                if nm in X.columns:
                    j=list(X.columns).index(nm); d[nm]=rh.params[j]; d[nm+'_se']=rh.bse[j]
        else:
            for nm in('step','ramp'):
                if nm in X.columns: d[nm]=r.params[list(X.columns).index(nm)]
        out[m]=d
    return out
rows=[]; plc=[]
allowed=[d for d in pd.date_range(dmin+pd.Timedelta(days=W),dmax-pd.Timedelta(days=W)) if min(abs((d-e).days) for e in EVD)>=21]
for party in('D','R'):
    pl=rng.choice(len(allowed),50,replace=False)
    pls=[allowed[i] for i in pl]
    for stay in(False,True):
        P=[]
        for d in pls:
            s=series(party,d,stay)
            if len(s)<150: continue
            f=fit_all(s,d,hac=False)
            P.append({'party':party,'stay':stay,'date':d,'step':f['ii']['step'],'ramp':f['iii']['ramp'],'dbic':f['iii']['bic']-f['ii']['bic'],'dbic_null_step':f['i']['bic']-f['ii']['bic'],'dbic_null_ramp':f['i']['bic']-f['iii']['bic']})
        plc+=P
plc=pd.DataFrame(plc); plc.to_csv(O+'/h4a_placebo.csv',index=False)
for e,loser in EV.items():
    c=pd.Timestamp(e)
    for party in('D','R'):
        sgn=1 if party==loser else -1
        for stay in(False,True):
            s=series(party,c,stay); f=fit_all(s,c)
            P=plc[(plc.party==party)&(plc.stay==stay)]
            st=f['ii']['step']; rp=f['iii']['ramp']*W
            r={'event':e,'party':party,'stay':stay,'n_days':len(s),'n_members_note':'', 'sign':sgn,
               'step':st,'step_se':f['ii']['step_se'],'step_lo':st-1.96*f['ii']['step_se'],'step_hi':st+1.96*f['ii']['step_se'],
               'ramp120':rp,'ramp_se120':f['iii']['ramp_se']*W,
               'step_iv':f['iv']['step'],'ramp120_iv':f['iv']['ramp']*W,
               **{f'bic_{m}':f[m]['bic'] for m in f},**{f'aic_{m}':f[m]['aic'] for m in f}}
            r['dbic_step_vs_ramp']=f['iii']['bic']-f['ii']['bic']
            r['dbic_chg_vs_null']=f['i']['bic']-min(f['ii']['bic'],f['iii']['bic'],f['iv']['bic'])
            r['placebo_n']=len(P)
            r['pct_abs_step']=(np.abs(P.step)<abs(st)).mean()
            r['p_signed_step']=(1+(sgn*P.step>=sgn*st).sum())/(1+len(P))
            r['pct_abs_ramp']=(np.abs(P.ramp*W)<abs(rp)).mean()
            r['placebo_step_pref_share']=(P.dbic>2).mean()
            r['p95_abs_step']=np.quantile(np.abs(P.step),.95); r['p95_abs_ramp120']=np.quantile(np.abs(P.ramp*W),.95)
            ch=(r['dbic_chg_vs_null']>6) and (abs(st)>r['p95_abs_step'] or abs(rp)>r['p95_abs_ramp120'])
            r['changes']=bool(ch)
            r['class']=('step' if r['dbic_step_vs_ramp']>2 else 'ramp' if r['dbic_step_vs_ramp']<-2 else 'ambig') if ch else 'nochange'
            r['signed_in_pred_dir']=bool(sgn*st>0)
            rows.append(r)
R=pd.DataFrame(rows); R.to_csv(O+'/h4a_results.csv',index=False)
pd.set_option('display.width',250); pd.set_option('display.max_columns',40)
print(R[['event','party','stay','n_days','step','step_lo','step_hi','ramp120','bic_i','bic_ii','bic_iii','bic_iv','dbic_step_vs_ramp','dbic_chg_vs_null','pct_abs_step','p_signed_step','changes','class','signed_in_pred_dir']].round(4).to_string())
for stay in(False,True):
    X=R[R.stay==stay]; print('stay',stay,'changing',X.changes.sum(),X['class'].value_counts().to_dict(),'dir_ok among changing',X[X.changes].signed_in_pred_dir.sum())
    pp_=plc[plc.stay==stay]; print(' placebo step-pref share (dbic>2)',(pp_.dbic>2).mean().round(3),' ramp-pref',(pp_.dbic<-2).mean().round(3),'real step-pref',(X.dbic_step_vs_ramp>2).mean().round(3),'real ramp-pref',(X.dbic_step_vs_ramp<-2).mean().round(3),'n placebo',len(pp_))
