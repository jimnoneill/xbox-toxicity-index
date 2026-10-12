"""
ti_build.py  (spec D)  Toxicity Index rebuild from probabilities.

Inputs : output/v7/ti/post_probs.parquet, output/v7/ti/account_member_map.csv,
         output/toxicity_index/congress_classified_2017_2023.parquet (v1 TI column, for the Table 3 reconciliation only)
Outputs (output/v7/ti/): member_ti_original.csv, member_ti_all.csv, member_year_ti.csv, account_ti.csv,
         prevalence.csv, factor_analysis.json, variant_rank_corr.csv, variant_party_g.csv, table3_yearly.csv,
         numbers_ti.json

REVISION 2026-10-08 (review fixes): PRIMARY SAMPLE = all non-retweet posts (retweet = text starts "RT @"; the "@"-prefix
  reply category is dropped because it is unreliable). SECONDARY = all posts. Every member-level result is computed on both;
  secondary keys carry the suffix _allposts. First-pass (original-only) outputs are in output/v7/ti/superseded_v1/ and embedded in
  numbers_ti.json under "superseded".

Definitions
  Dimensions (8): offensive, toxic, neg_sentiment, irony, ragebait, tribal, performative, genuine_anger.
  Primary post TI  = mean of the 8 probabilities (equal weights, no dampener).
  Variant 1 (old TI, binary)      = mean of 8 binary flags (flag = p>0.5; neg_sentiment flag = argmax class, as in v2).
  Variant 2 (old TI, dampened)    = variant 1 * (1 - 0.3 * advocacy_flag).
  Variant 3 (one factor)          = one-factor score (sklearn FactorAnalysis on z-scored member-level dimension means).
  Variant 4 (leave one dimension out) = mean of the other 7 dimension probabilities, one per dropped dimension.
  Member TI = mean over the member's ORIGINAL posts (primary) or ALL posts (secondary), all accounts pooled.
  Member sample for inference: entity_type member, >= MIN_ORIG original posts (50), party D or R for contrasts.
  Majority share: fraction of the member's posts made while the member's party held the chamber majority
  (House: R to 2019-01-03, D to 2023-01-03, R after. Senate: R to 2021-01-20, D after; independents counted as D).
Seeds fixed (stats_common.SEED).
"""
import json, os, sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import FactorAnalysis

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats_common import Numbers, contrast_table, hedges_g, SEED

ROOT = os.path.expanduser('~/x_box/pipeline')
SUF = os.environ.get('V7_SUFFIX', '')
MAPDIR = f'{ROOT}/output/v7/ti'
OUT = f'{ROOT}/output/v7/ti{SUF}'
os.makedirs(OUT, exist_ok=True)
MIN_ORIG = 50
MIN_MY = 20
DIMS = ['offensive', 'toxic', 'neg_sentiment', 'irony', 'ragebait', 'tribal', 'performative', 'genuine_anger']
N = Numbers()
NA = Numbers(suffix='_allposts')

pp = pd.read_parquet(f'{OUT}/post_probs.parquet')
amap = pd.read_csv(f'{MAPDIR}/account_member_map.csv', dtype={'user_id': str})
keep = ['screen_name', 'bioguide', 'entity_type', 'account_type', 'party', 'chamber', 'state', 'np_dim1', 'icpsr', 'name']
pp = pp.merge(amap[keep], on='screen_name', how='left')
pp['entity_type'] = pp['entity_type'].fillna('unmatched')

# per-post quantities
P = pp[[f'p_{d}' for d in DIMS]].values.astype(np.float32)
F = P > 0.5
F[:, DIMS.index('neg_sentiment')] = pp['argmax_neg_sentiment'].values.astype(bool)
adv_flag = (pp['p_advocacy'].values > 0.5)
pp['ti_prob'] = P.mean(1)
pp['ti_bin'] = F.mean(1)
pp['ti_damp'] = pp['ti_bin'] * (1 - 0.3 * adv_flag)
for i, d in enumerate(DIMS):
    pp[f'f_{d}'] = F[:, i].astype(np.float32)
pp['f_advocacy'] = adv_flag.astype(np.float32)
pp['zero_flags'] = (F.sum(1) == 0).astype(np.float32)
pp['clean'] = (pp['ti_damp'] < 0.1).astype(np.float32)
pp['high'] = (pp['ti_damp'] >= 0.5).astype(np.float32)
for i, d in enumerate(DIMS):  # leave-one-out post TI
    pp[f'loo_{d}'] = (P.sum(1) - P[:, i]) / 7.0
pp['year'] = pp['created_at'].dt.year
pp['primary'] = (pp['post_type'] != 'retweet')  # non-retweet: original plus '@'-prefixed
v1 = pd.read_parquet(f'{ROOT}/output/toxicity_index/congress_classified_2017_2023.parquet', columns=['toxicity_index'])
assert len(v1) == len(pp)
pp['ti_v1'] = v1['toxicity_index'].values.astype(np.float32)

# majority status per post
d = pp['created_at'].dt.tz_localize(None) if pp['created_at'].dt.tz is not None else pp['created_at']
is_house = pp['chamber'].eq('house').values
is_sen = pp['chamber'].eq('senate').values
rep_party = pp['party'].eq('R').values
house_rmaj = ((d < '2019-01-03') | (d >= '2023-01-03')).values
sen_rmaj = (d < '2021-01-20').values
rmaj = np.where(is_house, house_rmaj, np.where(is_sen, sen_rmaj, np.nan))
pp['in_majority'] = np.where(np.isnan(rmaj), np.nan, np.where(rep_party, rmaj, 1 - rmaj)).astype(np.float32)

# ---------- corpus-level descriptives ----------
N.add('n_posts_total', len(pp), 'Posts scored (congresstweets 2017-06 to 2023-07 with parseable timestamp)')
vc = pp['post_type'].value_counts()
for k in ['original', 'reply', 'retweet']:
    N.add(f'n_posts_{k}', int(vc.get(k, 0)), f'Posts of type {k} (congresstweets approximation: RT @ prefix = retweet, @ prefix = reply)')
et = pp['entity_type'].value_counts()
for k, v in et.items():
    N.add(f'n_posts_entity_{k}', int(v), f'Posts from accounts with entity type {k}')
mem = pp[pp.entity_type == 'member']
N.add('n_posts_member', len(mem), 'Posts from accounts mapped to a member')
N.add('n_posts_member_original', int((mem.post_type == 'original').sum()), 'Original posts from member-mapped accounts')

# ---------- member tables ----------
agg_cols = ([f'p_{d}' for d in DIMS] + ['p_advocacy', 'p_emotion_anger'] + [f'f_{d}' for d in DIMS] + ['f_advocacy', 'argmax_emotion_anger',
            'ti_prob', 'ti_bin', 'ti_damp', 'zero_flags', 'clean', 'high', 'in_majority'] + [f'loo_{d}' for d in DIMS])
meta = (mem.groupby('bioguide').agg(name=('name', 'first'), party=('party', 'first'), chamber=('chamber', 'first'),
                                    state=('state', 'first'), np_dim1=('np_dim1', 'first'), icpsr=('icpsr', 'first'),
                                    n_accounts=('screen_name', 'nunique'), n_posts_all=('ti_prob', 'size'))
        .reset_index())
sw = amap[amap.entity_type == 'member'].groupby('bioguide').party_switch.max().rename('party_switch').reset_index()
meta = meta.merge(sw, on='bioguide', how='left')


def member_table(sub):
    g = sub.groupby('bioguide')[agg_cols].mean()
    g['n_posts'] = sub.groupby('bioguide').size()
    g['n_orig_total'] = sub[sub.post_type == 'original'].groupby('bioguide').size()
    g['n_reply_total'] = sub[sub.post_type == 'reply'].groupby('bioguide').size()
    g['n_rt_total'] = mem[mem.post_type == 'retweet'].groupby('bioguide').size()
    g = g.reset_index().merge(meta.drop(columns='n_posts_all'), on='bioguide', how='left')
    g[['n_orig_total', 'n_reply_total', 'n_rt_total']] = g[['n_orig_total', 'n_reply_total', 'n_rt_total']].fillna(0)
    g['n_primary'] = g['n_posts']
    g['n_orig'] = g['n_posts']  # kept for old consumers: count of posts in the basis (non-retweet for the primary table)
    return g


mp = mem[mem.primary]
mo = member_table(mp)
ma = member_table(mem)
mo.to_csv(f'{OUT}/member_ti_primary.csv', index=False)
ma.to_csv(f'{OUT}/member_ti_all.csv', index=False)

N.add('n_posts_nonretweet', int(pp.primary.sum()), 'Non-retweet posts (primary basis; original plus "@"-prefixed)')
N.add('n_posts_member_nonretweet', int(mp.shape[0]), 'Non-retweet posts from member-mapped accounts')
N.add('n_members_mapped', int(meta.shape[0]), 'Members with at least one post in the corpus')
N.add('n_members_party_switch', int((meta.party_switch == 1).sum()), 'Members flagged party_switch in the account map (excluded in the spec curve)')


def fit_fa(tab):
    X = tab[[f'p_{d}' for d in DIMS]].copy()
    Z = (X - X.mean()) / X.std(ddof=0)
    fa = FactorAnalysis(n_components=1, random_state=0).fit(Z.values)
    lam = fa.components_[0].copy()
    sign = 1 if lam.sum() >= 0 else -1
    lam = sign * lam
    psi = fa.noise_variance_
    R = np.corrcoef(Z.values.T)
    return fa, lam, psi, R, sign * fa.transform(Z.values)[:, 0]


def analyze(Mtab, sub_posts, NN, lab, fname_suffix):
    """Mtab: all-member table for the basis. sub_posts: member posts of the basis."""
    elig = Mtab[Mtab.n_posts >= MIN_ORIG].copy()
    NN.add('n_members_ge50', len(elig), f'Members with >= {MIN_ORIG} posts in basis {lab} (inference sample)')
    for p in ['D', 'R', 'I']:
        NN.add(f'n_members_ge50_{p}', int((elig.party == p).sum()), f'Inference-sample members with party {p}, basis {lab}')
    NN.add('n_members_ge50_senate', int((elig.chamber == 'senate').sum()), f'Inference-sample senators, basis {lab}')
    NN.add('n_members_ge50_house', int((elig.chamber == 'house').sum()), f'Inference-sample House members, basis {lab}')
    fa, lam, psi, R, score = fit_fa(elig)
    omega = lam.sum() ** 2 / (lam.sum() ** 2 + psi.sum())
    eig = np.sort(np.linalg.eigvalsh(R))[::-1]
    k = len(DIMS)
    rbar = (R.sum() - k) / (k * (k - 1))
    alpha_std = k * rbar / (1 + (k - 1) * rbar)
    elig['ti_factor'] = score
    fa_out = dict(basis=lab, n_members=len(elig), loadings=dict(zip(DIMS, map(float, lam))), uniqueness=dict(zip(DIMS, map(float, psi))),
                  omega_total=float(omega), cronbach_alpha_standardized=float(alpha_std),
                  share_variance_explained=float((lam ** 2).sum() / k), correlation_eigenvalues=list(map(float, eig)),
                  note='One-factor model on z-scored member-level mean probabilities. omega_total = (sum lambda)^2 / ((sum lambda)^2 + sum psi).')
    json.dump(fa_out, open(f'{OUT}/factor_analysis{fname_suffix}.json', 'w'), indent=1)
    NN.add('fa_omega_total', omega, f'McDonald omega (total), one-factor model over the 8 member-level dimension means, basis {lab}')
    NN.add('fa_alpha_standardized', alpha_std, f'Standardized Cronbach alpha over the 8 dimension means, basis {lab}')
    NN.add('fa_first_eigenvalue', eig[0], f'First eigenvalue of the 8x8 correlation matrix of member-level dimension means, basis {lab}')
    NN.add('fa_second_eigenvalue', eig[1], f'Second eigenvalue, basis {lab}')
    NN.add('fa_loadings', fa_out['loadings'], f'One-factor loadings by dimension, basis {lab}')
    NN.add('fa_uniqueness', fa_out['uniqueness'], f'One-factor uniquenesses by dimension, basis {lab}')
    variants = {'V0_primary_prob_mean': 'ti_prob', 'V1_binary_flags': 'ti_bin', 'V2_binary_dampened_old': 'ti_damp', 'V3_one_factor': 'ti_factor'}
    for d_ in DIMS:
        variants[f'V4_without_{d_}'] = f'loo_{d_}'
    vcols = list(variants.values())
    rho = elig[vcols].corr(method='spearman')
    rho.index = rho.columns = list(variants.keys())
    rho.to_csv(f'{OUT}/variant_rank_corr{fname_suffix}.csv')
    dr = elig[elig.party.isin(['D', 'R'])].copy()
    vg = pd.DataFrame(contrast_table(dr, vcols, group_col='party', g1='R', g2='D'))
    vg.insert(0, 'variant', list(variants.keys()))
    vg.to_csv(f'{OUT}/variant_party_g{fname_suffix}.csv', index=False)
    for name in variants:
        r = vg[vg.variant == name].iloc[0]
        NN.add(f'g_party_{name}', r.hedges_g, f'Hedges g (R minus D), member-level, basis {lab}, variant {name}; CI [{r.g_lo:.3f}, {r.g_hi:.3f}]')
    NN.add('rankcorr_V0_vs_V1', rho.loc['V0_primary_prob_mean', 'V1_binary_flags'], f'Spearman between member rankings, primary vs binary flags, basis {lab}')
    NN.add('rankcorr_V0_vs_V2', rho.loc['V0_primary_prob_mean', 'V2_binary_dampened_old'], f'Spearman, primary vs old dampened TI, basis {lab}')
    NN.add('rankcorr_V0_vs_V3', rho.loc['V0_primary_prob_mean', 'V3_one_factor'], f'Spearman, primary vs one-factor score, basis {lab}')
    loo_cols = [v for v in variants if v.startswith('V4')]
    NN.add('rankcorr_V0_vs_V4_min', rho.loc['V0_primary_prob_mean', loo_cols].min(), f'Smallest Spearman between primary and any leave-one-dimension-out TI, basis {lab}')
    for col in ['ti_prob', 'ti_damp']:
        s_ = elig[col]
        NN.add(f'member_{col}_mean', s_.mean(), f'Mean over members of member {col}, basis {lab}')
        NN.add(f'member_{col}_sd', s_.std(ddof=1), f'SD over members of member {col}, basis {lab}')
        for lo, hi, nm in [(None, 0.15, 'lt015'), (0.15, 0.30, '015_030'), (0.30, None, 'gt030')]:
            m = (s_ < hi) if lo is None else ((s_ > hi) if hi is None else ((s_ >= lo) & (s_ <= hi)))
            NN.add(f'share_members_{col}_{nm}', m.mean(), f'Share of members with {col} in band {nm} (draft thresholds 0.15 / 0.30), basis {lab}')
            NN.add(f'n_members_{col}_{nm}', int(m.sum()), f'Count of members with {col} in band {nm}, basis {lab}')
    NN.add('post_ti_prob_mean_member', sub_posts.ti_prob.mean(), f'Mean post-level primary TI over member posts, basis {lab}')
    po = sub_posts[sub_posts.bioguide.isin(elig.bioguide)]
    for col in ['ti_prob', 'ti_damp']:
        x = po[col].values.astype(np.float64)
        g = po.groupby('bioguide')[col].agg(['size', 'mean'])
        nk = g['size'].values; mk = g['mean'].values
        Nn, K = len(x), len(g)
        grand = x.mean()
        ssb = (nk * (mk - grand) ** 2).sum()
        ssw = (x ** 2).sum() - (nk * mk ** 2).sum()
        msb, msw = ssb / (K - 1), ssw / (Nn - K)
        Fv = msb / msw
        n0 = (Nn - (nk ** 2).sum() / Nn) / (K - 1)
        icc = (msb - msw) / (msb + (n0 - 1) * msw)
        NN.add(f'anova_F_{col}', Fv, f'One-way ANOVA across the {K} inference-sample members on post-level {col}, basis {lab}, no truncation')
        NN.add(f'anova_df_{col}', [K - 1, Nn - K], f'ANOVA df (between, within) for {col}, basis {lab}')
        NN.add(f'anova_p_{col}', stats.f.sf(Fv, K - 1, Nn - K), f'ANOVA p for {col}, basis {lab} (0.0 means below float precision)')
        NN.add(f'anova_eta2_{col}', ssb / (ssb + ssw), f'eta squared for {col}, basis {lab}')
        NN.add(f'icc1_{col}', icc, f'ICC(1): share of post-level variance in {col} attributable to the member, basis {lab}')
    return elig, vg, rho


elig, vg, rho = analyze(mo, mp, N, 'nonrt', '')
elig_a, vg_a, rho_a = analyze(ma, mem, NA, 'allposts', '_allposts')
elig.to_csv(f'{OUT}/member_ti_inference_sample.csv', index=False)
elig_a.to_csv(f'{OUT}/member_ti_inference_sample_allposts.csv', index=False)
N.add('share_reply_prefix_at', float((pp.post_type == 'reply').mean()), 'Share of posts starting with "@" (reply category dropped as unreliable; these posts stay in the primary basis)')
N.add('share_retweet', float((pp.post_type == 'retweet').mean()), 'Share of posts starting with "RT @" (excluded from primary)')

# ---------- account level (office vs campaign), non-retweet posts ----------
acc = mp.groupby(['screen_name', 'bioguide', 'account_type'])[['ti_prob', 'ti_damp'] + [f'p_{d}' for d in DIMS]].mean()
acc['n_orig'] = mp.groupby(['screen_name', 'bioguide', 'account_type']).size()
acc = acc.reset_index()
acc.to_csv(f'{OUT}/account_ti.csv', index=False)

# ---------- prevalence ----------
prev = []
for d_ in DIMS + ['advocacy']:
    fc = f'f_{d_}'
    r = dict(dimension=d_, prev_flag_all_posts=pp[fc].mean(), prev_flag_nonrt_member=mp[fc].mean(),
             mean_prob_all_posts=pp[f'p_{d_}'].mean(), mean_prob_nonrt_member=mp[f'p_{d_}'].mean(),
             member_mean_prob_median=elig[f'p_{d_}'].median(), member_flag_rate_median=elig[fc].median())
    r['near_zero_prevalence'] = bool(r['prev_flag_all_posts'] < 0.02)
    prev.append(r)
prev = pd.DataFrame(prev)
prev.to_csv(f'{OUT}/prevalence.csv', index=False)
for _, r in prev.iterrows():
    N.add(f'prev_flag_{r.dimension}', r.prev_flag_all_posts, f'Share of all posts flagged on {r.dimension} (p>0.5; sentiment = argmax negative)')
    N.add(f'meanprob_{r.dimension}', r.mean_prob_all_posts, f'Mean probability on {r.dimension} over all posts')


# ---------- member-year tables and Table 3 ----------
def member_year(sub):
    t = sub.groupby(['bioguide', 'year']).agg(ti_prob=('ti_prob', 'mean'), ti_damp=('ti_damp', 'mean'),
                                              ti_bin=('ti_bin', 'mean'), in_majority=('in_majority', 'mean'), n=('ti_prob', 'size')).reset_index()
    return t.merge(meta[['bioguide', 'party', 'chamber']], on='bioguide', how='left')


my = member_year(mp)
my.to_csv(f'{OUT}/member_year_ti.csv', index=False)
my_a = member_year(mem)
my_a.to_csv(f'{OUT}/member_year_ti_allposts.csv', index=False)
myq = my[my.n >= MIN_MY]
myq_a = my_a[my_a.n >= MIN_MY]
cov = pp.groupby('year').created_at.agg(['min', 'max'])
rows = []
for y, g in pp.groupby('year'):
    gm = mem[(mem.year == y)]
    go = gm[gm.primary]
    first, last = cov.loc[y, 'min'], cov.loc[y, 'max']
    days = (last.normalize() - first.normalize()).days + 1
    ndays = 366 if (y % 4 == 0) else 365
    rows.append(dict(year=y, n_posts_all=len(g), n_posts_member_nonrt=len(go), first_post=first.date(), last_post=last.date(),
                     days_covered=days, share_of_year_covered=days / ndays, partial_year=bool(days < 0.97 * ndays),
                     draft_table3_definition_v1_ti_all_posts=g.ti_v1.mean(),
                     v2_dampened_binary_all_posts=g.ti_damp.mean(),
                     v1_binary_nodamp_all_posts=g.ti_bin.mean(),
                     primary_prob_all_posts=g.ti_prob.mean(),
                     primary_prob_member_nonrt_posts=go.ti_prob.mean(),
                     primary_prob_member_all_posts=gm.ti_prob.mean(),
                     member_year_mean_primary_nonrt=myq[myq.year == y].ti_prob.mean(),
                     member_year_mean_primary_allposts=myq_a[myq_a.year == y].ti_prob.mean(),
                     n_members_ge20_nonrt=int((myq.year == y).sum()), n_members_ge20_allposts=int((myq_a.year == y).sum())))
t3 = pd.DataFrame(rows)
t3.to_csv(f'{OUT}/table3_yearly.csv', index=False)
v2acc = pd.read_csv(f'{ROOT}/output/toxicity_index_v2/all_members_ti_v2.csv')
N.add('t3_overall_v1_ti_post_mean', pp.ti_v1.mean(), 'All-post mean of v1 TI (8 binary flags incl. off-the-shelf anger, no dampener). Draft Table 3 yearly values 0.244-0.279 are this definition')
N.add('t3_overall_v2_damp_post_mean', pp.ti_damp.mean(), 'All-post mean of v2 TI (genuine anger, 0.3 advocacy dampener), tweet level')
N.add('t3_overall_v2_damp_account_mean', float(v2acc.ti_v2_mean.mean()), 'Mean over the 1,659 accounts (>=20 tweets) of account-mean v2 TI. This is the draft 0.156 (SD 0.075) in Section 4.1 / Table 1')
N.add('t3_overall_v2_damp_account_sd', float(v2acc.ti_v2_mean.std()), 'SD of the same account means')
for _, r in t3.iterrows():
    y = int(r.year)
    N.add(f'yearly_{y}_primary_prob_member_nonrt', r.primary_prob_member_nonrt_posts, f'Mean primary TI of member non-retweet posts in {y}; partial year = {r.partial_year} ({r.first_post} to {r.last_post})')
    N.add(f'yearly_{y}_primary_prob_member_allposts', r.primary_prob_member_all_posts, f'Mean primary TI of all member posts in {y}')
    N.add(f'yearly_{y}_partial_year', bool(r.partial_year), f'Partial-year flag for {y}')
    N.add(f'yearly_{y}_v1_ti', r.draft_table3_definition_v1_ti_all_posts, f'v1 TI mean in {y} (the draft Table 3 column)')
    N.add(f'yearly_{y}_v2_damp', r.v2_dampened_binary_all_posts, f'v2 dampened TI mean over all posts in {y}')
    N.add(f'yearly_{y}_n_posts', int(r.n_posts_all), f'Posts in {y}')

N.d.update(NA.d)
N.save(f'{OUT}/numbers_ti.json', superseded_from=f'{OUT}/superseded_v1/numbers_ti.json')
print('done', len(elig), 'members primary;', len(elig_a), 'members all posts')
print(prev.round(4).to_string())
print(vg[['variant', 'hedges_g', 'g_lo', 'g_hi', 'p']].round(3).to_string())
print(vg_a[['variant', 'hedges_g', 'g_lo', 'g_hi', 'p']].round(3).to_string())
print(t3.round(4).to_string())
