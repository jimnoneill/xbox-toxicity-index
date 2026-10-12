"""
ANOVA + party comparison with TI v2 numbers.
"""
import pandas as pd
import numpy as np
from scipy import stats
import json

ti_dir = '/home/joneill/x_box/pipeline/output/toxicity_index_v2'
data_dir = '/home/joneill/x_box/pipeline/data'

members = pd.read_csv(f'{ti_dir}/all_members_ti_v2.csv')
ct = pd.read_parquet(f'{ti_dir}/congress_ti_v2.parquet')
print(f'Members: {len(members)}, Tweets: {len(ct):,}')

# Build party+chamber map from handles files
sh = pd.read_parquet(f'{data_dir}/senator_handles.parquet')[['twitter_handle', 'party']].copy()
sh['chamber'] = 'Senate'
hh = pd.read_parquet(f'{data_dir}/house_handles.parquet')[['twitter_handle', 'party', 'chamber']].copy()
handles = pd.concat([sh, hh]).drop_duplicates('twitter_handle')
party_map = handles.set_index('twitter_handle')['party'].to_dict()
chamber_map = handles.set_index('twitter_handle')['chamber'].to_dict()

members['party'] = members['screen_name'].map(party_map)
members['chamber'] = members['screen_name'].map(chamber_map)
ct['party'] = ct['screen_name'].map(party_map)

print(f"\nParty coverage: {members['party'].notna().sum()}/{len(members)}")
print(members['party'].value_counts())

r = members[members['party'] == 'Republican']
d = members[members['party'] == 'Democrat']
print(f'\nR={len(r)}, D={len(d)}')

# ANOVA (per-tweet)
tweet_groups = [g['ti_v2'].values for sn, g in ct.groupby('screen_name') if len(g) >= 50]
f_stat, anova_p = stats.f_oneway(*tweet_groups[:500])
print(f'\n{"="*70}')
print(f'ANOVA (per-tweet, top 500 members): F={f_stat:.1f}, p={anova_p:.2e}')

# Party TI comparison
print(f'\n{"="*70}')
print(f'PARTY COMPARISON: TI v2')
print(f'Republican mean TI: {r["ti_v2_mean"].mean():.4f} (SD={r["ti_v2_mean"].std():.4f}, n={len(r)})')
print(f'Democrat mean TI:   {d["ti_v2_mean"].mean():.4f} (SD={d["ti_v2_mean"].std():.4f}, n={len(d)})')
t_ti, p_ti = stats.ttest_ind(r['ti_v2_mean'], d['ti_v2_mean'])
pooled_ti = np.sqrt(((len(r)-1)*r['ti_v2_mean'].std()**2 + (len(d)-1)*d['ti_v2_mean'].std()**2) / (len(r)+len(d)-2))
d_ti = (r['ti_v2_mean'].mean() - d['ti_v2_mean'].mean()) / pooled_ti if pooled_ti > 0 else 0
print(f't={t_ti:.2f}, p={p_ti:.2e}, Cohen d={d_ti:+.3f}')

# Per-flag
print(f'\n{"="*70}')
print(f'PER-FLAG PARTY COMPARISON')
print(f'{"Flag":>25s} {"R_mean":>8s} {"D_mean":>8s} {"d":>8s} {"p":>10s} {"sig":>4s}')
flag_cols = [c for c in members.columns if c.endswith('_rate')]
results = {}
for col in sorted(flag_cols):
    rv = r[col].dropna().values
    dv = d[col].dropna().values
    if len(rv) < 2 or len(dv) < 2:
        continue
    t, p = stats.ttest_ind(rv, dv)
    pooled = np.sqrt(((len(rv)-1)*rv.std()**2 + (len(dv)-1)*dv.std()**2) / (len(rv)+len(dv)-2))
    cd = (rv.mean() - dv.mean()) / pooled if pooled > 0 else 0
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    label = col.replace('flag_', '').replace('_rate', '')
    print(f'{label:>25s} {rv.mean():8.4f} {dv.mean():8.4f} {cd:+8.3f} {p:10.2e} {sig:>4s}')
    results[label] = {'r_mean': round(float(rv.mean()), 4), 'd_mean': round(float(dv.mean()), 4), 'd': round(float(cd), 3), 'p': float(p)}

# Temporal trend
print(f'\n{"="*70}')
print(f'TEMPORAL TREND')
ct['year'] = pd.to_datetime(ct['created_at']).dt.year
yearly = ct.groupby('year')['ti_v2'].mean()
print(yearly.to_string())
years = yearly.index.values.astype(float)
vals = yearly.values
slope, intercept, r_val, p_val_trend, se = stats.linregress(years, vals)
print(f'Slope: {slope:+.5f}/year, r={r_val:.3f}, p={p_val_trend:.4f}')

# Party x year
print(f'\nBy party x year:')
for party in ['Republican', 'Democrat']:
    sub = ct[ct['party'] == party]
    yp = sub.groupby('year')['ti_v2'].mean()
    s, i, rv, pv, se = stats.linregress(yp.index.values.astype(float), yp.values)
    print(f'  {party}: slope={s:+.5f}/yr, r={rv:.3f}')

# Chamber comparison
print(f'\n{"="*70}')
print(f'CHAMBER COMPARISON')
senate = members[members['chamber'] == 'Senate'].dropna(subset=['ti_v2_mean'])
house = members[members['chamber'].str.lower() == 'house'].dropna(subset=['ti_v2_mean'])
if len(senate) > 5 and len(house) > 5:
    t, p = stats.ttest_ind(senate['ti_v2_mean'], house['ti_v2_mean'])
    pooled = np.sqrt(((len(senate)-1)*senate['ti_v2_mean'].std()**2 + (len(house)-1)*house['ti_v2_mean'].std()**2) / (len(senate)+len(house)-2))
    cd = (senate['ti_v2_mean'].mean() - house['ti_v2_mean'].mean()) / pooled if pooled > 0 else 0
    print(f'Senate: {senate["ti_v2_mean"].mean():.4f} (n={len(senate)})')
    print(f'House:  {house["ti_v2_mean"].mean():.4f} (n={len(house)})')
    print(f't={t:.2f}, p={p:.2e}, d={cd:+.3f}')
else:
    print(f'Senate: n={len(senate)}, House: n={len(house)} - insufficient for comparison')

# Save
all_results = {
    'anova': {'f': float(f_stat), 'p': float(anova_p)},
    'party_ti': {'r_mean': round(float(r['ti_v2_mean'].mean()), 4), 'r_sd': round(float(r['ti_v2_mean'].std()), 4),
                 'd_mean': round(float(d['ti_v2_mean'].mean()), 4), 'd_sd': round(float(d['ti_v2_mean'].std()), 4),
                 'cohen_d': round(float(d_ti), 3), 't': round(float(t_ti), 2), 'p': float(p_ti)},
    'per_flag': results,
    'temporal': {'slope': round(float(slope), 5), 'r': round(float(r_val), 3), 'p': float(p_val_trend)},
}
with open(f'{ti_dir}/anova_party_v2.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print(f'\nSaved: {ti_dir}/anova_party_v2.json')
print('DONE')
