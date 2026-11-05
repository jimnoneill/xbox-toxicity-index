import numpy as np, pandas as pd
V7='/home/joneill/x_box/pipeline/output/v7'; O=V7+'/rvt_b'
D7=['offensive','toxic','neg_sentiment','irony','ragebait','tribal','performative']
pp=pd.read_parquet(f'{V7}/full/post_probs_full.parquet',columns=['screen_name','created_at','post_type']+[f'p_{d}' for d in D7])
print('raw',len(pp)); pp=pp[pp.post_type!='retweet']
pp['ti7']=pp[[f'p_{d}' for d in D7]].mean(axis=1).astype('float32')
pp=pp[['screen_name','created_at','ti7']]
am=pd.read_csv(f'{V7}/ti/account_member_map.csv')[['screen_name','bioguide','account_type']].dropna(subset=['bioguide'])
pp=pp.merge(am,on='screen_name'); print('mapped nonRT',len(pp), pp.bioguide.nunique())
pp['date']=pd.to_datetime(pp.created_at,utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None).dt.normalize()
pp['camp']=(pp.account_type=='campaign').astype(int)
g=pp.groupby(['bioguide','date']).agg(ti_sum=('ti7','sum'),n=('ti7','size'),n_camp=('camp','sum')).reset_index()
g['ti7']=g.ti_sum/g.n
cal=pd.read_parquet(f'{V7}/t3/calendar_member_day.parquet',columns=['date','bioguide','in_office','party','chamber','aligned_with_president','in_chamber_majority'])
cal['bioguide']=cal.bioguide.astype(str); cal['party']=cal.party.astype(str); cal['chamber']=cal.chamber.astype(str)
cal=cal[(cal.in_office==1)&cal.party.isin(['D','R'])]
g['bioguide']=g.bioguide.astype(str)
md=g.merge(cal,on=['date','bioguide'])
print('member-day',len(md),md.bioguide.nunique(),md.date.min(),md.date.max())
md.to_parquet(O+'/member_day.parquet')
# calendar of in-office member-days (for weekly volume zeros)
cal[['date','bioguide','party','chamber']].assign(week=lambda d:d.date-pd.to_timedelta(d.date.dt.dayofweek,unit='D')).groupby(['bioguide','week']).size().rename('days_in_office').reset_index().to_parquet(O+'/member_week_office.parquet')
md['week']=md.date-pd.to_timedelta(md.date.dt.dayofweek,unit='D')
mw=md.groupby(['bioguide','week']).agg(ti_sum=('ti_sum','sum'),n=('n','sum')).reset_index(); mw['ti7']=mw.ti_sum/mw.n
mw.to_parquet(O+'/member_week.parquet'); print('member-week',len(mw))
