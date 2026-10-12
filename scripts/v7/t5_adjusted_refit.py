"""Review check: is aligned_president an artifact of party-specific time trends?
Member-day TI7, member FE via demeaning, OLS cluster-robust by member.
A: dow + spline(t,8) + regimes.  B: A + party x spline(t,8)."""
import numpy as np, pandas as pd, sys
sys.path.insert(0, '/home/joneill/x_box/pylib_v7')
import statsmodels.api as sm
V7 = '/home/joneill/x_box/pipeline/output/v7'
D7 = ['offensive', 'toxic', 'neg_sentiment', 'irony', 'ragebait', 'tribal', 'performative']
pp = pd.read_parquet(f'{V7}/full/post_probs_full.parquet', columns=['screen_name', 'created_at', 'post_type'] + [f'p_{d}' for d in D7])
pp = pp[pp.post_type != 'retweet']
pp['ti7'] = pp[[f'p_{d}' for d in D7]].mean(axis=1)
amap = pd.read_csv(f'{V7}/ti/account_member_map.csv')[['screen_name', 'bioguide']].dropna()
pp = pp.merge(amap, on='screen_name')
pp['date'] = pd.to_datetime(pp.created_at, utc=True).dt.tz_convert('America/New_York').dt.date
day = pp.groupby(['bioguide', 'date']).agg(ti7=('ti7', 'mean'), n=('ti7', 'size')).reset_index()
REG = ['primary_window', 'general_window', 'post_election', 'chamber_in_session', 'aligned_with_president', 'in_chamber_majority', 'event_day']
cal = pd.read_parquet(f'{V7}/t3/calendar_member_day.parquet', columns=['date', 'bioguide', 'in_office', 'party'] + REG)
cal['date'] = pd.to_datetime(cal.date).dt.date
cal = cal[cal.in_office & cal.party.isin(['Democrat', 'Republican', 'D', 'R'])]
df = day.merge(cal, on=['bioguide', 'date'])
df['t'] = (pd.to_datetime(df.date) - pd.Timestamp('2017-06-21')).dt.days / 2211.0
df['dow'] = pd.to_datetime(df.date).dt.dayofweek
df['rep'] = df.party.str.startswith('R').astype(float)
print('rows', len(df), 'members', df.bioguide.nunique())
def nat_spline(t, df_):
    knots = np.quantile(t, np.linspace(0, 1, df_ + 1))
    X = [t]
    for k in knots[1:-1]:
        X.append(np.clip(t - k, 0, None) ** 3)
    return np.column_stack(X)
S = nat_spline(df.t.values, 8)
dow = pd.get_dummies(df.dow, prefix='dow', drop_first=True).values.astype(float)
Xreg = df[REG].astype(float).values
base = np.column_stack([Xreg, dow, S])
names_base = REG + [f'dow{i}' for i in range(dow.shape[1])] + [f'sp{i}' for i in range(S.shape[1])]
PS = S * df.rep.values[:, None]
Xb = np.column_stack([base, PS])
names_b = names_base + [f'rep_sp{i}' for i in range(PS.shape[1])]
y = df.ti7.values
gidx = df.bioguide.values
def fe_fit(X, names):
    D = pd.DataFrame(X, columns=names); D['y'] = y; D['g'] = gidx
    dm = D.groupby('g').transform('mean')
    Z = D[names].values - dm[names].values
    yy = D['y'].values - dm['y'].values
    m = sm.OLS(yy, Z).fit(cov_type='cluster', cov_kwds={'groups': gidx})
    return {n: (m.params[i], m.bse[i]) for i, n in enumerate(names) if n in REG}
print('--- A: common trend only')
for k, (c, s) in fe_fit(base, names_base).items(): print(f'{k:24s} {c:+.5f} (se {s:.5f})')
print('--- B: + party x trend')
for k, (c, s) in fe_fit(Xb, names_b).items(): print(f'{k:24s} {c:+.5f} (se {s:.5f})')

# Persist both specifications for figures and supplement.
rows = []
for spec, (X, nm) in {"common_trend": (base, names_base), "party_trend": (Xb, names_b)}.items():
    for k, (c, s) in fe_fit(X, nm).items():
        rows.append({"spec": spec, "term": k, "coef": c, "se": s,
                     "ci_lo": c - 1.959964 * s, "ci_hi": c + 1.959964 * s})
out = pd.DataFrame(rows)
out.to_csv(f"{V7}/t5/regime_coefficients_adjusted.csv", index=False)
print("wrote", f"{V7}/t5/regime_coefficients_adjusted.csv")
