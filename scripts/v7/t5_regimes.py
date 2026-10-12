"""
t5_regimes.py  T5 content regimes on the full corpus (associations only).

Outcome: member-day and member-week mean of TI7 (mean of 7 probabilities, genuine anger excluded), and of tribal, performative, ragebait.
Posts: non-retweet posts of member-mapped accounts (office and campaign pooled; also office-only and campaign-only), Eastern-time calendar day.
Only days (weeks) with at least one post are observed, so estimates describe posting days. Weights = number of posts in the unit (WLS);
unweighted TI7 shown as a sensitivity.
Regime flags (T3 calendar, in-office days only): own primary window (primary_window), general-election window, post-election window,
own chamber in session, president of own party (aligned_with_president), own party in chamber majority, curated event day, detected
event day (T2full top-60 volume days, whole Congress). Week level: share of the member's in-office days in the week with the flag.
Model: member fixed effects (within transformation with weighted member means), day-of-week dummies (day level only), natural cubic
spline of calendar time (df 8, about one per year), WLS, SE clustered by member. Holm across the 8 regime terms within each fit.
Sensitivity: spline df 4 and 16, and unweighted, for TI7 overall (member-day).
Groups: all, D, R, office-only, campaign-only. D vs R difference in each coefficient: z test on independent fits.
Seed 20261010 (no randomness used). Output: output/v7/t5/.
"""
import json, os, sys, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrix
from scipy import stats

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats_common import holm

ROOT = os.path.expanduser('~/x_box/pipeline'); V7 = f'{ROOT}/output/v7'; OUT = f'{V7}/t5'
os.makedirs(OUT, exist_ok=True)
D7 = ['offensive', 'toxic', 'neg_sentiment', 'irony', 'ragebait', 'tribal', 'performative']
REG = ['own_primary_window', 'general_window', 'post_election', 'in_session', 'aligned_president', 'party_majority', 'event_curated', 'event_detected']
OUTC = ['ti7', 'p_tribal', 'p_performative', 'p_ragebait']
NUM = {}


def put(k, v, d):
    if isinstance(v, (np.floating,)): v = float(v)
    if isinstance(v, (np.integer,)): v = int(v)
    NUM[k] = {'value': v, 'description': d}


amap = pd.read_csv(f'{V7}/ti/account_member_map.csv')
am = amap[amap.entity_type == 'member'][['screen_name', 'bioguide', 'account_type', 'party']]
pp = pd.read_parquet(f'{V7}/full/post_probs_full.parquet', columns=['id', 'screen_name', 'created_at', 'post_type'] + [f'p_{d}' for d in D7])
pp = pp.drop_duplicates('id'); pp = pp[pp.post_type != 'retweet']
pp = pp.merge(am, on='screen_name', how='inner')
pp['ti7'] = pp[[f'p_{d}' for d in D7]].mean(axis=1).astype('float32')
pp['date'] = pp.created_at.dt.tz_convert('America/New_York').dt.tz_localize(None).dt.normalize()
put('n_posts_member_nonrt', len(pp), 'Non-retweet posts of member-mapped accounts')

cal = pd.read_parquet(f'{V7}/t3/calendar_member_day.parquet', columns=['date', 'bioguide', 'in_office', 'chamber_in_session', 'primary_window', 'general_window',
                                                                      'post_election', 'aligned_with_president', 'in_chamber_majority', 'event_day'])
cal['date'] = pd.to_datetime(cal.date).dt.normalize()
cal = cal[(cal.date >= '2017-06-21') & (cal.date <= '2023-07-11') & (cal.in_office == 1)].copy()
ev = pd.read_csv(f'{V7}/t2full/event_days_top60.csv'); det = set(pd.to_datetime(ev.date).dt.normalize())
cal['own_primary_window'] = cal.primary_window.fillna(0).astype(float)
cal['general_window'] = cal.general_window.fillna(0).astype(float)
cal['post_election'] = cal.post_election.fillna(0).astype(float)
cal['in_session'] = cal.chamber_in_session.fillna(0).astype(float)
cal['aligned_president'] = cal.aligned_with_president.fillna(0).astype(float)
cal['party_majority'] = cal.in_chamber_majority.fillna(0).astype(float)
cal['event_curated'] = cal.event_day.fillna(0).astype(float)
cal['event_detected'] = cal.date.isin(det).astype(float)
cal = cal[['date', 'bioguide'] + REG]
put('n_calendar_member_days', len(cal), 'In-office member-days in the corpus window')
for r in REG: put(f'flag_share_{r}', cal[r].mean(), f'Share of in-office member-days with {r}')


def build(posts, unit):
    p = posts.copy()
    if unit == 'week':
        p['unit'] = p.date.dt.to_period('W-SUN').dt.start_time
    else:
        p['unit'] = p.date
    g = p.groupby(['bioguide', 'unit']).agg(n=('ti7', 'size'), ti7=('ti7', 'mean'), p_tribal=('p_tribal', 'mean'), p_performative=('p_performative', 'mean'),
                                           p_ragebait=('p_ragebait', 'mean'), party=('party', 'first')).reset_index()
    c = cal.copy()
    if unit == 'week':
        c['unit'] = c.date.dt.to_period('W-SUN').dt.start_time
        c = c.groupby(['bioguide', 'unit'])[REG].mean().reset_index()
    else:
        c = c.rename(columns={'date': 'unit'})
    g = g.merge(c, on=['bioguide', 'unit'], how='inner')
    g['t'] = (g.unit - pd.Timestamp('2017-06-21')).dt.days.astype(float)
    if unit == 'day':
        g['dow'] = g.unit.dt.dayofweek
    return g


def spline(t, df):
    B = np.asarray(dmatrix(f'cr(t, df={df})', {'t': t}, return_type='dataframe'))
    return B[:, 1:]


def fit(g, y, unit, df=8, weighted=True):
    X = g[REG].values.astype(float)
    parts = [X]
    names = list(REG)
    if unit == 'day':
        D = pd.get_dummies(g.dow, drop_first=True).values.astype(float); parts.append(D); names += [f'dow{i}' for i in range(D.shape[1])]
    S = spline(g.t.values, df); parts.append(S); names += [f'sp{i}' for i in range(S.shape[1])]
    X = np.hstack(parts)
    w = g.n.values.astype(float) if weighted else np.ones(len(g))
    yv = g[y].values.astype(float)
    code = pd.factorize(g.bioguide)[0]
    sw = np.bincount(code, weights=w)
    ym = np.bincount(code, weights=w * yv) / sw
    Xm = np.column_stack([np.bincount(code, weights=w * X[:, j]) / sw for j in range(X.shape[1])])
    Xd = X - Xm[code]; yd = yv - ym[code]
    keep = Xd.std(0) > 1e-9
    res = sm.WLS(yd, Xd[:, keep], weights=w).fit(cov_type='cluster', cov_kwds={'groups': code})
    nm = [n for n, k in zip(names, keep) if k]
    out = []
    for r in REG:
        if r in nm:
            j = nm.index(r); out.append(dict(term=r, coef=res.params[j], se=res.bse[j], p=res.pvalues[j]))
        else:
            out.append(dict(term=r, coef=np.nan, se=np.nan, p=np.nan))
    o = pd.DataFrame(out); ok = o.p.notna()
    o['p_holm'] = np.nan; o.loc[ok, 'p_holm'] = holm(o.loc[ok, 'p'].values)
    o['lo'] = o.coef - 1.96 * o.se; o['hi'] = o.coef + 1.96 * o.se
    o['n_units'] = len(g); o['n_members'] = int(len(sw)); o['outcome_mean'] = float(np.average(yv, weights=w))
    return o


GROUPS = {'all': lambda d: d, 'D': lambda d: d[d.party == 'D'], 'R': lambda d: d[d.party == 'R']}
rows = []
for unit in ['day', 'week']:
    base = {'all_types': pp, 'office': pp[pp.account_type == 'office'], 'campaign': pp[pp.account_type == 'campaign']}
    tabs = {k: build(v, unit) for k, v in base.items()}
    put(f'n_{unit}_units_all_types', len(tabs['all_types']), f'Member-{unit}s with >=1 post and calendar flags'); put(f'n_{unit}_members', int(tabs['all_types'].bioguide.nunique()), 'members')
    specs = [('all', tabs['all_types'], 'all_types'), ('D', GROUPS['D'](tabs['all_types']), 'all_types'), ('R', GROUPS['R'](tabs['all_types']), 'all_types'),
             ('office_only', tabs['office'], 'office'), ('campaign_only', tabs['campaign'], 'campaign')]
    for gname, g, _ in specs:
        g = g[g.party.isin(['D', 'R'])] if gname in ('all', 'office_only', 'campaign_only') else g
        for y in OUTC:
            o = fit(g, y, unit); o.insert(0, 'unit', unit); o.insert(1, 'group', gname); o.insert(2, 'outcome', y); rows.append(o)
    # sensitivities (day only, ti7, all)
    if unit == 'day':
        g = tabs['all_types'][tabs['all_types'].party.isin(['D', 'R'])]
        for lab, kw in [('spline_df4', dict(df=4)), ('spline_df16', dict(df=16)), ('unweighted', dict(weighted=False))]:
            o = fit(g, 'ti7', unit, **kw); o.insert(0, 'unit', unit); o.insert(1, 'group', 'all'); o.insert(2, 'outcome', f'ti7_{lab}'); rows.append(o)
res = pd.concat(rows, ignore_index=True)
res.to_csv(f'{OUT}/regime_coefficients.csv', index=False)
# D vs R differences
dif = []
for (unit, y, term), d in res[res.group.isin(['D', 'R'])].groupby(['unit', 'outcome', 'term']):
    if set(d.group) == {'D', 'R'}:
        a, b = d[d.group == 'R'].iloc[0], d[d.group == 'D'].iloc[0]
        z = (a.coef - b.coef) / np.sqrt(a.se ** 2 + b.se ** 2)
        dif.append(dict(unit=unit, outcome=y, term=term, coef_R=a.coef, coef_D=b.coef, diff=a.coef - b.coef, z=z, p=2 * stats.norm.sf(abs(z))))
dif = pd.DataFrame(dif)
dif['p_holm'] = np.nan
for (unit, y), d in dif.groupby(['unit', 'outcome']):
    dif.loc[d.index, 'p_holm'] = holm(d.p.values)
dif.to_csv(f'{OUT}/regime_party_difference.csv', index=False)
for _, r in res.iterrows():
    put(f'{r.unit}_{r.group}_{r.outcome}_{r.term}', r.coef, f'coef {r.coef:.5f}, SE {r.se:.5f}, p={r.p:.3g}, Holm p={r.p_holm:.3g}, CI [{r.lo:.5f},{r.hi:.5f}]; n_units={r.n_units}, n_members={r.n_members}, outcome mean {r.outcome_mean:.4f}')
for _, r in dif.iterrows():
    put(f'diffRD_{r.unit}_{r.outcome}_{r.term}', r['diff'], f'R minus D coefficient difference; z={r.z:.2f}, p={r.p:.3g}, Holm p={r.p_holm:.3g}')
json.dump(NUM, open(f'{OUT}/numbers_t5.json', 'w'), indent=1, default=str)
pd.set_option('display.width', 250)
print(res[(res.group == 'all') & (res.outcome == 'ti7')].round(5).to_string())
print('DONE')
