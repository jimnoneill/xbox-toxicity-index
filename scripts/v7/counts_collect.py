#!/usr/bin/env python
"""Resumable hourly counts collector (X counts/all). Usage: counts_collect.py [--max-calls N]"""
import json, os, sys, time, datetime as dt, requests, pandas as pd
B=os.path.expanduser('~/x_box/pipeline/output/v7/counts/')
LEDGER=B+'ledger.jsonl'; LOG=B+'collect.log'; STOP=B+'STOP'; USAGE=B+'usage_checks.jsonl'
START=dt.datetime(2023,7,1,tzinfo=dt.timezone.utc); END=dt.datetime(2026,10,8,tzinfo=dt.timezone.utc)
STEP=dt.timedelta(days=31)
tok=json.load(open(os.path.expanduser('~/x_box/.secrets/x_api.json')))['bearer_token']
S=requests.Session(); S.headers['Authorization']='Bearer '+tok
def log(m):
    with open(LOG,'a') as f: f.write(dt.datetime.utcnow().strftime('%F %T ')+m+'\n')
def halt(m):
    log('HALT '+m); open(STOP,'w').write(m); print('HALT',m); sys.exit(3)
maxc=int(sys.argv[sys.argv.index('--max-calls')+1]) if '--max-calls' in sys.argv else 10**9
if os.path.exists(STOP): sys.exit('STOP file present: '+open(STOP).read())
acc=pd.read_csv(B+'accounts.csv',dtype=str).fillna('')
acc['o']=acc.account_type.map({'office':0,'personal':0,'campaign':1}); acc=acc.sort_values('o',kind='stable')
done=set(); dead=set()
if os.path.exists(LEDGER):
    for l in open(LEDGER):
        r=json.loads(l)
        if r['st']=='ok': done.add((r['h'],r['w']))
        elif r['st']=='skip400': dead.add(r['h'])
wins=[]; s=START
while s<END: wins.append((s,min(s+STEP,END))); s+=STEP
iso=lambda d:d.strftime('%Y-%m-%dT%H:%M:%SZ')
calls=0; buf=[]; led=[]
def flush():
    global buf,led
    if buf:
        df=pd.DataFrame(buf,columns=['account','hour_start_utc','count'])
        df['hour_start_utc']=pd.to_datetime(df.hour_start_utc,utc=True)
        for y,g in df.groupby(df.hour_start_utc.dt.year):
            p=B+f'hourly/year={y}/'; os.makedirs(p,exist_ok=True)
            g.to_parquet(p+f'part-{int(time.time()*1000)}.parquet',index=False)
    with open(LEDGER,'a') as f:
        for r in led: f.write(json.dumps(r)+'\n')
    buf=[];led=[]
def usage():
    r=S.get('https://api.x.com/2/usage/tweets',timeout=30)
    d=r.json().get('data',{}) if r.status_code==200 else {}
    rec={'t':iso(dt.datetime.utcnow()),'calls':calls,'status':r.status_code,'project_usage':d.get('project_usage'),'cap':d.get('project_cap')}
    open(USAGE,'a').write(json.dumps(rec)+'\n'); return rec
base=usage(); log(f'start baseline usage {base}')
def call(h,ws,we):
    global calls
    out=[]; nt=None
    while True:
        p={'query':f'from:{h}','granularity':'hour','start_time':iso(ws),'end_time':iso(we)}
        if nt: p['next_token']=nt
        for att in range(6):
            r=S.get('https://api.x.com/2/tweets/counts/all',params=p,timeout=60); calls+=1
            rem=r.headers.get('x-rate-limit-remaining'); rs=r.headers.get('x-rate-limit-reset')
            if r.status_code==429 or (rem is not None and int(rem)<=1 and r.status_code==200):
                w=max(1,int(rs)-int(time.time())+2) if rs else 900
                log(f'rate limit, sleeping {w}s')
                if r.status_code==200: break
                time.sleep(w); continue
            if r.status_code>=500: time.sleep(5*(att+1)); continue
            break
        if r.status_code==200 and rem is not None and int(rem)<=1:
            time.sleep(max(1,int(rs)-int(time.time())+2))
        txt=r.text[:300]
        if r.status_code==402 or any(k in r.text.lower() for k in ('credit','usagecap','billing','usage cap')) and r.status_code!=200:
            halt(f'{r.status_code} {txt}')
        if r.status_code==400: return 'skip400',txt
        if r.status_code!=200: return 'err',f'{r.status_code} {txt}'
        j=r.json(); out+=[(h,b['start'],b['tweet_count']) for b in j.get('data',[])]
        nt=j.get('meta',{}).get('next_token')
        if not nt: return 'ok',out
for _,a in acc.iterrows():
    h=a.handle
    for ws,we in wins:
        if h in dead: break
        w=iso(ws)
        if (h,w) in done: continue
        if calls>=maxc: flush(); log(f'max-calls reached {calls}'); sys.exit(0)
        st,res=call(h,ws,we)
        if st=='ok':
            buf+=res; led.append({'h':h,'w':w,'st':'ok','n':len(res),'sum':sum(x[2] for x in res)})
        elif st=='skip400':
            log(f'400 skip {h}: {res}'); led.append({'h':h,'w':w,'st':'skip400','msg':res}); dead.add(h)
        else:
            log(f'err {h} {w}: {res}'); led.append({'h':h,'w':w,'st':'err','msg':res})
        if len(led)>=100: flush()
        if calls and calls%500<3 and calls//500!=getattr(sys.modules[__name__],'_lu',0):
            _lu=calls//500; sys.modules[__name__]._lu=_lu
            u=usage(); log(f'usage check at {calls}: {u}')
            if u['project_usage'] is not None and base['project_usage'] is not None and int(u['project_usage'])>int(base['project_usage']):
                flush(); halt(f'post-read usage rose {base["project_usage"]} -> {u["project_usage"]}')
        if calls%50<2: log(f'progress calls={calls} at {h} {w}')
flush(); log(f'ALL DONE calls={calls}')
