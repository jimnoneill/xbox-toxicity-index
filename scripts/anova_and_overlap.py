"""ANOVA, overlap analysis, and party comparisons on 1.68M classified congressional tweets."""
import os, sys, json
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out_dir = '/home/joneill/x_box/pipeline/output/toxicity_index'
fig_dir = '/home/joneill/x_box/pipeline/figures/toxicity_index'
os.makedirs(fig_dir, exist_ok=True)

plt.rcParams.update({'font.size': 11, 'font.family': 'serif'})

# Load member TI scores
members = pd.read_csv(f'{out_dir}/all_members_toxicity_index.csv')
print(f'Members: {len(members)}')

# Load handles to get party
senate = pd.read_parquet('/home/joneill/x_box/pipeline/data/senator_handles.parquet')
house = pd.read_parquet('/home/joneill/x_box/pipeline/data/house_handles.parquet')
all_handles = pd.concat([senate, house], ignore_index=True)

# Map party by handle
party_map = all_handles.set_index('twitter_handle')['party'].to_dict()
# Also try lowercase
party_map_lower = {k.lower(): v for k, v in party_map.items()}

members['party'] = members['screen_name'].map(party_map)
# Fill missing with lowercase lookup
missing = members['party'].isna()
members.loc[missing, 'party'] = members.loc[missing, 'screen_name'].str.lower().map(party_map_lower)

# Also get chamber
chamber_map = {}
for _, r in senate.iterrows():
    chamber_map[r['twitter_handle']] = 'Senate'
for _, r in house.iterrows():
    chamber_map[r['twitter_handle']] = 'House'
members['chamber'] = members['screen_name'].map(chamber_map)

n_party = members['party'].notna().sum()
n_chamber = members['chamber'].notna().sum()
print(f'Party mapped: {n_party}/{len(members)}')
print(f'Chamber mapped: {n_chamber}/{len(members)}')

# Filter to members with party + enough tweets
m = members[(members['party'].isin(['Republican', 'Democrat'])) & (members['n_tweets'] >= 50)].copy()
print(f'Analysis sample: {len(m)} members (>= 50 tweets, R or D)')

flag_cols = [c for c in m.columns if c.endswith('_rate')]
print(f'Flag columns: {flag_cols}')

# ============================================================
# 1. ANOVA: Does TI differ across members?
# ============================================================
print('\n' + '=' * 70)
print('1. ONE-WAY ANOVA: TI across members')
print('=' * 70)

# Load per-tweet data for ANOVA
ct = pd.read_parquet(f'{out_dir}/congress_classified_2017_2023.parquet',
                     columns=['screen_name', 'toxicity_index'])
# Filter to our analysis sample
ct = ct[ct['screen_name'].isin(m['screen_name'].values)]

groups = [g['toxicity_index'].values for _, g in ct.groupby('screen_name') if len(g) >= 50]
f_stat, p_val = stats.f_oneway(*groups[:500])  # cap at 500 groups for memory
print(f'F({len(groups)-1}, {len(ct)-len(groups)}) = {f_stat:.1f}, p = {p_val:.2e}')
eta_sq = (f_stat * (len(groups) - 1)) / (f_stat * (len(groups) - 1) + (len(ct) - len(groups)))
print(f'Eta-squared = {eta_sq:.4f}')
del ct  # free memory

# ============================================================
# 2. PARTY COMPARISONS
# ============================================================
print('\n' + '=' * 70)
print('2. PARTY COMPARISONS')
print('=' * 70)

rep = m[m.party == 'Republican']
dem = m[m.party == 'Democrat']
print(f'Republicans: n={len(rep)}, Democrats: n={len(dem)}')

print(f'\n{"Metric":30s} {"R mean":>8s} {"R SD":>7s} {"D mean":>8s} {"D SD":>7s} {"t":>7s} {"p":>8s} {"d":>7s}')
print('-' * 85)

metrics = [('ti_mean', 'Toxicity Index'), ('pct_clean', '% Clean'),
           ('pct_high', '% High TI')] + \
          [(c, c.replace('flag_', '').replace('_rate', '')) for c in flag_cols]

party_results = []
for col, label in metrics:
    rv = rep[col].dropna()
    dv = dem[col].dropna()
    t, p = stats.ttest_ind(rv, dv)
    pooled = np.sqrt(((len(rv)-1)*rv.std()**2 + (len(dv)-1)*dv.std()**2) / (len(rv)+len(dv)-2))
    d = (rv.mean() - dv.mean()) / pooled if pooled > 0 else 0
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    print(f'{label:30s} {rv.mean():8.3f} {rv.std():7.3f} {dv.mean():8.3f} {dv.std():7.3f} {t:+7.2f} {p:8.4f} {d:+7.3f} {sig}')
    party_results.append({
        'metric': label, 'r_mean': round(rv.mean(), 4), 'r_sd': round(rv.std(), 4),
        'd_mean': round(dv.mean(), 4), 'd_sd': round(dv.std(), 4),
        't': round(t, 3), 'p': round(p, 5), 'cohens_d': round(d, 3), 'sig': sig,
    })

# ============================================================
# 3. CHAMBER COMPARISONS
# ============================================================
print('\n' + '=' * 70)
print('3. CHAMBER COMPARISONS')
print('=' * 70)

mc = m[m['chamber'].notna()]
sen = mc[mc.chamber == 'Senate']
hou = mc[mc.chamber == 'House']
print(f'Senate: n={len(sen)}, House: n={len(hou)}')

for col, label in [('ti_mean', 'Toxicity Index'), ('pct_clean', '% Clean')]:
    sv = sen[col].dropna()
    hv = hou[col].dropna()
    t, p = stats.ttest_ind(sv, hv)
    print(f'  {label}: Senate={sv.mean():.3f} House={hv.mean():.3f} t={t:+.2f} p={p:.4f}')

# ============================================================
# 4. TEMPORAL TRENDS (by year)
# ============================================================
print('\n' + '=' * 70)
print('4. TEMPORAL TRENDS')
print('=' * 70)

ct = pd.read_parquet(f'{out_dir}/congress_classified_2017_2023.parquet',
                     columns=['screen_name', 'created_at', 'toxicity_index'])
ct['created_at'] = pd.to_datetime(ct['created_at'], utc=True, errors='coerce')
ct['year'] = ct['created_at'].dt.year

yearly = ct.groupby('year')['toxicity_index'].agg(['mean', 'median', 'std', 'count'])
print(yearly.round(4))

# Trend test
slope, intercept, r, p, se = stats.linregress(yearly.index, yearly['mean'])
print(f'\nLinear trend: slope={slope:.4f}/year, r={r:.3f}, p={p:.4f}')
del ct

# ============================================================
# 5. OVERLAP ANALYSIS
# ============================================================
print('\n' + '=' * 70)
print('5. OVERLAP ANALYSIS (from member-level rates)')
print('=' * 70)

# Average flag rates across all members
print('\nOverall flag prevalence (member-averaged):')
for c in flag_cols:
    label = c.replace('flag_', '').replace('_rate', '')
    print(f'  {label:25s} {m[c].mean():.1%}')

# Correlation matrix of flag rates across members
flag_corr = m[flag_cols].corr()
print('\nFlag correlation matrix (across members):')
print(flag_corr.round(3).to_string())

# ============================================================
# 6. FIGURES
# ============================================================
print('\n' + '=' * 70)
print('6. GENERATING FIGURES')
print('=' * 70)

# Fig: TI distribution by party
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
bins = np.linspace(0, 0.55, 30)
ax.hist(rep['ti_mean'], bins=bins, alpha=0.6, color='red', label=f'Republican (n={len(rep)})', edgecolor='black', linewidth=0.5)
ax.hist(dem['ti_mean'], bins=bins, alpha=0.6, color='blue', label=f'Democrat (n={len(dem)})', edgecolor='black', linewidth=0.5)
ax.axvline(x=rep['ti_mean'].mean(), color='darkred', linestyle='--', linewidth=2)
ax.axvline(x=dem['ti_mean'].mean(), color='darkblue', linestyle='--', linewidth=2)
ax.set_xlabel('Toxicity Index (Member Mean)')
ax.set_ylabel('Number of Members')
ax.set_title('Toxicity Index Distribution by Party (n=1,659 members, 2017-2023)')
ax.legend()
plt.tight_layout()
plt.savefig(f'{fig_dir}/ti_by_party.png', dpi=300)
plt.close()
print('  ti_by_party.png')

# Fig: Flag rates by party
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
flag_labels = [c.replace('flag_', '').replace('_rate', '') for c in flag_cols]
r_means = [rep[c].mean() for c in flag_cols]
d_means = [dem[c].mean() for c in flag_cols]
x = np.arange(len(flag_cols))
w = 0.35
ax.bar(x - w/2, r_means, w, label='Republican', color='red', alpha=0.7, edgecolor='black', linewidth=0.5)
ax.bar(x + w/2, d_means, w, label='Democrat', color='blue', alpha=0.7, edgecolor='black', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(flag_labels, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Mean Flag Rate')
ax.set_title('Negative Behavior Flag Rates by Party')
ax.legend()
plt.tight_layout()
plt.savefig(f'{fig_dir}/flags_by_party.png', dpi=300)
plt.close()
print('  flags_by_party.png')

# Fig: Temporal trend
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.plot(yearly.index, yearly['mean'], 'ro-', markersize=10, linewidth=2.5)
ax.fill_between(yearly.index, yearly['mean'] - yearly['std']/np.sqrt(yearly['count']),
                yearly['mean'] + yearly['std']/np.sqrt(yearly['count']), alpha=0.2, color='red')
ax.set_xlabel('Year')
ax.set_ylabel('Mean Toxicity Index')
ax.set_title('Congressional Toxicity Index Over Time (2017-2023)')
plt.tight_layout()
plt.savefig(f'{fig_dir}/ti_temporal_trend.png', dpi=300)
plt.close()
print('  ti_temporal_trend.png')

# Fig: Flag correlation heatmap
fig, ax = plt.subplots(1, 1, figsize=(8, 7))
im = ax.imshow(flag_corr.values, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(flag_cols)))
ax.set_yticks(range(len(flag_cols)))
ax.set_xticklabels(flag_labels, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(flag_labels, fontsize=9)
for i in range(len(flag_cols)):
    for j in range(len(flag_cols)):
        ax.text(j, i, f'{flag_corr.values[i, j]:.2f}', ha='center', va='center', fontsize=8)
plt.colorbar(im, ax=ax, shrink=0.8)
ax.set_title('Flag Co-occurrence Correlation Matrix')
plt.tight_layout()
plt.savefig(f'{fig_dir}/flag_correlation.png', dpi=300)
plt.close()
print('  flag_correlation.png')

# Save all results
pd.DataFrame(party_results).to_csv(f'{out_dir}/party_comparison_anova.csv', index=False)
yearly.to_csv(f'{out_dir}/temporal_trend.csv')

print(f'\nAll results saved to {out_dir}/')
print(f'Figures saved to {fig_dir}/')
