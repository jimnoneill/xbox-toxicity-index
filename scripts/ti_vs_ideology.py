"""
Cross-reference TI v2 with DW-NOMINATE / Nokken-Poole ideology scores.
Addresses reviewer comment: "Need to add granularity from VoteView"
"""
import pandas as pd
import numpy as np
from scipy import stats
import json

ti_dir = '/home/joneill/x_box/pipeline/output/toxicity_index_v2'
data_dir = '/home/joneill/x_box/pipeline/data'

# Load TI v2 per-member
members = pd.read_csv(f'{ti_dir}/all_members_ti_v2.csv')

# Load VoteView - filter to 115th-118th Congress (2017-2023, matching our tweet data)
vv = pd.read_csv(f'{data_dir}/voteview_all_members.csv')
vv_recent = vv[vv['congress'].isin([115, 116, 117, 118])].copy()
print(f'VoteView records (115th-118th): {len(vv_recent)}')

# Load handles to get twitter_handle <-> bioguide mapping
sh = pd.read_parquet(f'{data_dir}/senator_handles.parquet')[['twitter_handle', 'bioguide_id']]
hh = pd.read_parquet(f'{data_dir}/house_handles.parquet')[['twitter_handle', 'bioguide_id']]
handles = pd.concat([sh, hh]).drop_duplicates('twitter_handle')
bio_map = handles.set_index('twitter_handle')['bioguide_id'].to_dict()

members['bioguide_id'] = members['screen_name'].map(bio_map)

# Average Nokken-Poole across congresses for each member
vv_avg = vv_recent.groupby('bioguide_id').agg(
    nokken_dim1=('nokken_poole_dim1', 'mean'),
    nokken_dim2=('nokken_poole_dim2', 'mean'),
    dw_dim1=('nominate_dim1', 'mean'),
    n_congresses=('congress', 'nunique'),
).reset_index()

merged = members.merge(vv_avg, on='bioguide_id', how='inner')
print(f'Matched members (TI + VoteView): {len(merged)}')

# Correlation: TI vs ideology
print(f'\n{"="*70}')
print('TI v2 vs DW-NOMINATE / Nokken-Poole Correlations')
print(f'{"="*70}')

for score_col, label in [('nokken_dim1', 'Nokken-Poole dim1'), ('dw_dim1', 'DW-NOMINATE dim1')]:
    valid = merged[[score_col, 'ti_v2_mean']].dropna()
    r, p = stats.pearsonr(valid[score_col], valid['ti_v2_mean'])
    rho, p_rho = stats.spearmanr(valid[score_col], valid['ti_v2_mean'])
    print(f'\n{label} vs TI:')
    print(f'  Pearson r={r:.3f}, p={p:.2e}')
    print(f'  Spearman rho={rho:.3f}, p={p_rho:.2e}')
    print(f'  n={len(valid)}')

# Per-flag correlations with ideology
print(f'\n{"="*70}')
print('Per-flag correlation with Nokken-Poole dim1 (conservative=positive)')
print(f'{"="*70}')
flag_cols = [c for c in members.columns if c.endswith('_rate')]
for col in sorted(flag_cols):
    valid = merged[[col, 'nokken_dim1']].dropna()
    if len(valid) < 10:
        continue
    r, p = stats.pearsonr(valid['nokken_dim1'], valid[col])
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    label = col.replace('flag_', '').replace('_rate', '')
    print(f'  {label:>25s}  r={r:+.3f}  p={p:.2e} {sig}')

# Mike Lee specifically
print(f'\n{"="*70}')
print('Mike Lee in context')
print(f'{"="*70}')
ml_vv = vv_recent[vv_recent['bioguide_id'] == 'L000577']
print(f'\nMike Lee VoteView (per Congress):')
for _, row in ml_vv.iterrows():
    print(f'  {int(row.congress)}th: Nokken-Poole={row.nokken_poole_dim1:.3f}/{row.nokken_poole_dim2:.3f}  DW-NOM={row.nominate_dim1:.3f}/{row.nominate_dim2:.3f}')

# Where does Mike Lee fall in the TI distribution?
ml_ti = members[members['screen_name'].isin(['BasedMikeLee', 'SenMikeLee'])]
if len(ml_ti) > 0:
    for _, r in ml_ti.iterrows():
        pctl = (members['ti_v2_mean'] < r['ti_v2_mean']).mean() * 100
        print(f'\n  {r.screen_name}: TI={r.ti_v2_mean:.3f} (percentile: {pctl:.0f}th)')

# Quartile analysis: TI by ideology quartile
print(f'\nTI by ideology quartile:')
merged['ideo_quartile'] = pd.qcut(merged['nokken_dim1'], 4, labels=['Most Liberal', 'Center-Left', 'Center-Right', 'Most Conservative'])
for q in ['Most Liberal', 'Center-Left', 'Center-Right', 'Most Conservative']:
    sub = merged[merged['ideo_quartile'] == q]
    print(f'  {q:20s}: TI={sub["ti_v2_mean"].mean():.4f} (n={len(sub)})')

# Save
results = {
    'n_matched': len(merged),
    'ti_vs_nokken_dim1': {'r': round(float(stats.pearsonr(merged['nokken_dim1'].dropna(), merged.loc[merged['nokken_dim1'].notna(), 'ti_v2_mean'])[0]), 3)},
}
with open(f'{ti_dir}/ti_vs_ideology.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f'\nSaved: {ti_dir}/ti_vs_ideology.json')
print('DONE')
