"""
stats_decomp.py  (review fix M3 / M2-ladder, task 2)  Specification curve for the party effect.

Why is the party effect smaller than the draft's d = +0.53, and how much does the answer depend on the TI definition?
Every specification step is computed for BOTH TI definitions:
  ti_v2   = old binary TI (8 flags, sentiment flag = argmax, 0.3 advocacy dampener), row-aligned from output/toxicity_index_v2
  ti_prob = PRIMARY TI, mean of the 8 head probabilities (no flags, no dampener)
plus two single-dimension flag rates (genuine-anger head, advocacy head) for continuity with the first-pass ladder.
Effect = Hedges g (R minus D), 95% bootstrap CI (10,000 resamples stratified by party, seed 20261008), Welch p.

Post bases: "all posts"; "non-retweet" (PRIMARY: text does not start "RT @"); "strict original" (neither "RT @" nor "@" prefix; first-pass basis).
Units: account (screen_name) or member (bioguide, accounts pooled).
Outputs (output/v7/stats/): party_effect_ladder.csv (long, every step x TI definition), spec_curve.csv (tidy: spec, ti_definition, n_R, n_D, g, ci_lo, ci_hi, ...),
  post_type_share_by_party.csv, numbers_decomp.json.
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats_common import Numbers, contrast_table

ROOT = os.path.expanduser('~/x_box/pipeline')
TI = f'{ROOT}/output/v7/ti'
OUT = f'{ROOT}/output/v7/stats'
N = Numbers()
DIMS = ['offensive', 'toxic', 'neg_sentiment', 'irony', 'ragebait', 'tribal', 'performative', 'genuine_anger']
DEFN = {'ti_v2': 'ti_v2: old binary TI (8 flags at 0.5, 0.3 advocacy dampener)',
        'ti_prob': 'ti_prob: mean of 8 head probabilities (primary)',
        'flag_genuine_anger': 'genuine-anger head flag rate (p>0.5), single dimension',
        'flag_advocacy': 'advocacy head flag rate (p>0.5), single dimension'}
MEAS = list(DEFN)

v2 = pd.read_parquet(f'{ROOT}/output/toxicity_index_v2/congress_ti_v2.parquet', columns=['screen_name', 'ti_v2', 'flag_genuine_anger', 'flag_advocacy'])
pp = pd.read_parquet(f'{TI}/post_probs.parquet', columns=['screen_name', 'post_type'] + [f'p_{d}' for d in DIMS])
assert (pp.screen_name.values == v2.screen_name.values).all()
v2['post_type'] = pp.post_type.values
v2['ti_prob'] = pp[[f'p_{d}' for d in DIMS]].mean(axis=1).astype(np.float32).values
del pp
amap = pd.read_csv(f'{TI}/account_member_map.csv')
m = amap[amap.entity_type == 'member'][['screen_name', 'bioguide', 'party', 'account_type', 'party_switch']]
v2 = v2.merge(m, on='screen_name', how='left')       # unmatched accounts keep NaN bioguide
sh = pd.read_parquet(f'{ROOT}/data/senator_handles.parquet')[['twitter_handle', 'party']]
hh = pd.read_parquet(f'{ROOT}/data/house_handles.parquet')[['twitter_handle', 'party']]
hp = pd.concat([sh, hh]).drop_duplicates('twitter_handle').set_index('twitter_handle')['party'].map({'Republican': 'R', 'Democrat': 'D'})
v2['hp'] = v2.screen_name.map(hp)
switchers = set(m[m.party_switch == 1].bioguide.dropna())

BASIS = {'all posts': lambda d: d,
         'non-retweet': lambda d: d[d.post_type != 'retweet'],
         'strict original': lambda d: d[d.post_type == 'original']}


def agg(d, key):
    g = d.groupby(key)[MEAS].mean()
    g['size'] = d.groupby(key).size()
    return g.reset_index()


rows = []


def run(step, desc, unit, basis, minp, tab, party_col, filt=''):
    if party_col != 'party':
        tab = tab.drop(columns=['party'], errors='ignore')
    t = tab[tab[party_col].isin(['R', 'D']) & (tab['size'] >= minp)].rename(columns={party_col: 'party'})
    for r in contrast_table(t, MEAS):
        rows.append(dict(step=step, spec=desc, unit=unit, post_basis=basis, min_posts=minp, account_filter=filt, measure=r['measure'],
                         ti_definition=DEFN[r['measure']], nR=r['n1'], nD=r['n2'], mean_R=r['mean1'], mean_D=r['mean2'],
                         hedges_g=r['hedges_g'], g_lo=r['g_lo'], g_hi=r['g_hi'], p=r['p']))


# ---- account-level steps ----
for basis in ['all posts', 'non-retweet']:
    d = BASIS[basis](v2)
    d = d[d.screen_name.notna()]
    acc = agg(d, 'screen_name')
    meta = v2.drop_duplicates('screen_name').set_index('screen_name')[['hp', 'party', 'account_type', 'bioguide']]
    acc = acc.join(meta, on='screen_name')
    tag = 'A-all' if basis == 'all posts' else 'A-nonrt'
    run(f'{tag}1', f'accounts in the 119th-Congress handle table (draft sample), {basis}', 'account', basis, 50, acc, 'hp', 'draft handle table')
    if basis == 'all posts':
        run(f'{tag}2', f'same accounts, party from legislators file, {basis}', 'account', basis, 50, acc[acc.hp.notna()], 'party', 'draft handle table, map party')
    run(f'{tag}3', f'all member-mapped accounts, {basis}', 'account', basis, 50, acc[acc.bioguide.notna()], 'party', 'all mapped')
    run(f'{tag}4', f'office accounts only, {basis}', 'account', basis, 50, acc[acc.account_type == 'office'], 'party', 'office only')
    run(f'{tag}5', f'campaign accounts only, {basis}', 'account', basis, 50, acc[acc.account_type == 'campaign'], 'party', 'campaign only')
d = BASIS['strict original'](v2)
acc = agg(d[d.screen_name.notna()], 'screen_name').join(v2.drop_duplicates('screen_name').set_index('screen_name')[['hp', 'party', 'account_type', 'bioguide']], on='screen_name')
run('A-orig3', 'all member-mapped accounts, strict original posts (first-pass basis)', 'account', 'strict original', 50, acc[acc.bioguide.notna()], 'party', 'all mapped')
run('A-orig1', 'accounts in the 119th-Congress handle table, strict original posts', 'account', 'strict original', 50, acc[acc.hp.notna()], 'party', 'draft handle table')

# ---- member-level steps ----
mem = v2[v2.bioguide.notna()]


def member_tab(d):
    g = agg(d, 'bioguide')
    return g.merge(m.drop_duplicates('bioguide')[['bioguide', 'party']], on='bioguide')


run('M1', 'members, all posts, >= 50 posts', 'member', 'all posts', 50, member_tab(mem), 'party')
mn = BASIS['non-retweet'](mem)
mt = member_tab(mn)
run('M2', 'members, non-retweet posts, >= 50 posts (PRIMARY)', 'member', 'non-retweet', 50, mt, 'party')
run('M3', 'members, non-retweet posts, >= 200 posts', 'member', 'non-retweet', 200, mt, 'party')
run('M3a', 'members, all posts, >= 200 posts', 'member', 'all posts', 200, member_tab(mem), 'party')
run('M4', 'members, non-retweet posts, >= 50 posts, excluding party switchers (already excluded: they carry party I/L in the map)', 'member', 'non-retweet', 50, mt[~mt.bioguide.isin(switchers)], 'party', 'excl. switchers')
run('M4a', 'members, all posts, >= 50 posts, excluding party switchers', 'member', 'all posts', 50, member_tab(mem)[lambda t: ~t.bioguide.isin(switchers)], 'party', 'excl. switchers')
# The four members flagged party_switch (Manchin, Sinema, Mitchell, Amash) carry their LAST party (I or L) in the map, so M2 already
# excludes them from the R/D contrast and M4 is identical to M2 by construction. M4b puts them back under their original party.
pre = {'M001183': 'D', 'S001191': 'D', 'M001201': 'R', 'A000367': 'R'}
mt_b = mt.copy()
for bg, pty in pre.items():
    mt_b.loc[mt_b.bioguide == bg, 'party'] = pty
run('M4b', 'members, non-retweet posts, >= 50 posts, party switchers counted under their ORIGINAL party', 'member', 'non-retweet', 50, mt_b, 'party', 'switchers under original party')
run('M5', 'members, office accounts only (pooled), non-retweet posts, >= 50 posts', 'member', 'non-retweet', 50, member_tab(mn[mn.account_type == 'office']), 'party', 'office only')
run('M6', 'members, campaign accounts only (pooled), non-retweet posts, >= 50 posts', 'member', 'non-retweet', 50, member_tab(mn[mn.account_type == 'campaign']), 'party', 'campaign only')
run('M7', 'members, strict original posts, >= 50 posts (first-pass basis)', 'member', 'strict original', 50, member_tab(BASIS['strict original'](mem)), 'party')

res = pd.DataFrame(rows)
res.to_csv(f'{OUT}/party_effect_ladder.csv', index=False)
sc = res[res.measure.isin(['ti_v2', 'ti_prob'])].copy()
sc['is_primary'] = (sc.step == 'M2') & (sc.measure == 'ti_prob')
sc = sc.rename(columns={'nR': 'n_R', 'nD': 'n_D', 'hedges_g': 'g', 'g_lo': 'ci_lo', 'g_hi': 'ci_hi'})
sc[['spec', 'ti_definition', 'n_R', 'n_D', 'g', 'ci_lo', 'ci_hi', 'step', 'unit', 'post_basis', 'min_posts', 'account_filter', 'p', 'is_primary']].to_csv(
    f'{OUT}/spec_curve.csv', index=False)
sh_ = mem.groupby('party').post_type.value_counts(normalize=True).unstack()
sh_.to_csv(f'{OUT}/post_type_share_by_party.csv')
for _, r in res.iterrows():
    N.add(f'ladder2_{r.step}_{r.measure}_g', r.hedges_g, f'Hedges g R-D, {r.ti_definition}; step {r.step}: {r.spec}; nR={int(r.nR)}, nD={int(r.nD)}; CI [{r.g_lo:.3f}, {r.g_hi:.3f}]')
N.add('spec_curve_n_specs', int(sc.spec.nunique()), 'Specification steps in spec_curve.csv (each for ti_v2 and ti_prob)')
for mdef in ['ti_v2', 'ti_prob']:
    s_ = sc[sc.measure == mdef]
    N.add(f'spec_curve_g_min_{mdef}', float(s_.g.min()), f'Smallest g across specification steps, {mdef}')
    N.add(f'spec_curve_g_max_{mdef}', float(s_.g.max()), f'Largest g across specification steps, {mdef}')
    N.add(f'spec_curve_g_median_{mdef}', float(s_.g.median()), f'Median g across specification steps, {mdef}')
    mem_s = s_[s_.unit == 'member']
    N.add(f'spec_curve_member_g_min_{mdef}', float(mem_s.g.min()), f'Smallest g across member-level steps, {mdef}')
    N.add(f'spec_curve_member_g_max_{mdef}', float(mem_s.g.max()), f'Largest g across member-level steps, {mdef}')
N.save(f'{OUT}/numbers_decomp.json', superseded_from=f'{OUT}/superseded_v1/numbers_decomp.json')
pd.set_option('display.width', 250)
print(sc[['step', 'spec', 'measure', 'n_R', 'n_D', 'g', 'ci_lo', 'ci_hi', 'p']].round(3).to_string())
print(sh_.round(3))
