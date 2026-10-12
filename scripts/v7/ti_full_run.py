"""
ti_full_run.py  Full-corpus rerun of the Toxicity Index analyses with the DECIDED primary definition TI7.

TI7 = mean of the probabilities of 7 dimensions: offensive, toxic, neg_sentiment, irony, ragebait, tribal, performative.
(genuine anger is excluded; TI8 = the old 8-dimension mean is carried only as a comparison column).
Primary sample = non-retweet posts (text does not start "RT @"); secondary = all posts (suffix _allposts).
Member level pools office and campaign accounts. Account level = per account.
Seed 20261010 (bootstrap, ideology CI).

Usage: python ti_full_run.py --source full|sample
  full   -> output/v7/ti_full/ and output/v7/stats_full/   (post_probs_full.parquet)
  sample -> output/v7/ti_full/sample_ti7/ and output/v7/stats_full/sample_ti7/  (every-third-day sample, same TI7 code, for comparison)
Spec: MATH_SPEC D, E.  Reuses stats_common (hedges_g, contrast_table, holm).
"""
import argparse, json, os, sys, warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from sklearn.decomposition import FactorAnalysis

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats_common import contrast_table, hedges_g, holm

SEED = 20261010
ap = argparse.ArgumentParser()
ap.add_argument('--source', default='full', choices=['full', 'sample'])
A = ap.parse_args()
ROOT = os.path.expanduser('~/x_box/pipeline')
V7 = f'{ROOT}/output/v7'
sub = '' if A.source == 'full' else '/sample_ti7'
TIO = f'{V7}/ti_full{sub}'
STO = f'{V7}/stats_full{sub}'
os.makedirs(TIO, exist_ok=True); os.makedirs(STO, exist_ok=True)
PP = f'{V7}/full/post_probs_full.parquet' if A.source == 'full' else f'{V7}/ti/post_probs.parquet'

D7 = ['offensive', 'toxic', 'neg_sentiment', 'irony', 'ragebait', 'tribal', 'performative']
D8 = D7 + ['genuine_anger']
MIN_P, MIN_MY = 50, 20
NT, NS = {}, {}


def put(d, k, v, desc):
    if isinstance(v, (np.floating,)): v = float(v)
    if isinstance(v, (np.integer,)): v = int(v)
    if isinstance(v, (np.bool_,)): v = bool(v)
    d[k] = {'value': v, 'description': desc}


# ------------------------------------------------------------------ load and per-post quantities
cols = ['id', 'screen_name', 'created_at', 'post_type'] + [f'p_{d}' for d in D8] + ['p_advocacy', 'p_emotion_anger', 'argmax_neg_sentiment', 'argmax_emotion_anger']
pp = pd.read_parquet(PP, columns=cols)
n_raw = len(pp)
pp = pp.drop_duplicates('id', keep='first').reset_index(drop=True)
put(NT, 'n_rows_raw', n_raw, 'Rows in the probability file'); put(NT, 'n_posts_total', len(pp), 'Posts after dropping duplicate ids')
amap = pd.read_csv(f'{V7}/ti/account_member_map.csv', dtype={'user_id': str})
keep = ['screen_name', 'bioguide', 'entity_type', 'account_type', 'party', 'chamber', 'state', 'np_dim1', 'name']
pp = pp.merge(amap[keep], on='screen_name', how='left')
pp['entity_type'] = pp['entity_type'].fillna('unmatched')
P8 = pp[[f'p_{d}' for d in D8]].values.astype(np.float32)
P7 = P8[:, :7]
pp['ti7'] = P7.mean(1)
pp['ti8'] = P8.mean(1)
F8 = P8 > 0.5
F8[:, 2] = pp['argmax_neg_sentiment'].values.astype(bool)
adv = pp['p_advocacy'].values > 0.5
pp['ti_bin8'] = F8.mean(1)                      # old binary TI (8 flags)
pp['ti_damp8'] = pp['ti_bin8'] * (1 - 0.3 * adv)  # old dampened TI
pp['ti_bin7'] = F8[:, :7].mean(1)
for i, d in enumerate(D8): pp[f'f_{d}'] = F8[:, i].astype(np.float32)
pp['f_advocacy'] = adv.astype(np.float32)
for i, d in enumerate(D7): pp[f'loo_{d}'] = (P7.sum(1) - P7[:, i]) / 6.0
pp['year'] = pp['created_at'].dt.year
pp['primary'] = pp['post_type'] != 'retweet'
dd = pp['created_at'].dt.tz_localize(None)
is_h = pp['chamber'].eq('house').values; is_s = pp['chamber'].eq('senate').values; isR = pp['party'].eq('R').values
hr = ((dd < '2019-01-03') | (dd >= '2023-01-03')).values; sr = (dd < '2021-01-20').values
rm = np.where(is_h, hr, np.where(is_s, sr, np.nan))
pp['in_majority'] = np.where(np.isnan(rm), np.nan, np.where(isR, rm, 1 - rm)).astype(np.float32)
del P8, P7, F8
mem = pp[pp.entity_type == 'member']
mp = mem[mem.primary]
vc = pp.post_type.value_counts()
for k in ['original', 'reply', 'retweet']: put(NT, f'n_posts_{k}', int(vc.get(k, 0)), f'Posts by type (RT @ prefix = retweet, @ prefix = reply)')
put(NT, 'n_posts_nonretweet', int(pp.primary.sum()), 'Non-retweet posts (primary basis)')
put(NT, 'n_posts_member', len(mem), 'Posts from member-mapped accounts'); put(NT, 'n_posts_member_nonretweet', len(mp), 'Non-retweet member posts')
put(NT, 'n_accounts', int(pp.screen_name.nunique()), 'Accounts with posts')
put(NT, 'n_posts_unmatched', int((pp.entity_type == 'unmatched').sum()), 'Posts from accounts not in the account map')

AGG = ([f'p_{d}' for d in D8] + ['p_advocacy', 'p_emotion_anger', 'argmax_emotion_anger'] + [f'f_{d}' for d in D8] + ['f_advocacy'] +
       ['ti7', 'ti8', 'ti_bin8', 'ti_damp8', 'ti_bin7', 'in_majority'] + [f'loo_{d}' for d in D7])
meta = mem.groupby('bioguide').agg(name=('name', 'first'), party=('party', 'first'), chamber=('chamber', 'first'), state=('state', 'first'),
                                   np_dim1=('np_dim1', 'first'), n_accounts=('screen_name', 'nunique')).reset_index()


def mtab(s):
    g = s.groupby('bioguide')[AGG].mean()
    g['n_posts'] = s.groupby('bioguide').size()
    g['n_primary'] = g['n_posts']
    return g.reset_index().merge(meta, on='bioguide', how='left')


MO, MA = mtab(mp), mtab(mem)
MO.to_csv(f'{TIO}/member_ti7_primary.csv', index=False); MA.to_csv(f'{TIO}/member_ti7_all.csv', index=False)
BASES = {'nonrt': (MO, mp, ''), 'allposts': (MA, mem, '_allposts')}


def fit_fa(tab, dims):
    X = tab[[f'p_{d}' for d in dims]]
    Z = (X - X.mean()) / X.std(ddof=0)
    fa = FactorAnalysis(n_components=1, random_state=0).fit(Z.values)
    lam = fa.components_[0].copy(); sg = 1 if lam.sum() >= 0 else -1; lam *= sg
    return lam, fa.noise_variance_, np.corrcoef(Z.values.T), sg * fa.transform(Z.values)[:, 0], Z.values


# ------------------------------------------------------------------ per-basis analyses
ELIG = {}
for bname, (M, S, sfx) in BASES.items():
    el = M[M.n_posts >= MIN_P].copy()
    lam, psi, R, score, Z = fit_fa(el, D7)
    el['ti_factor7'] = score
    ELIG[bname] = el
    k = lambda s: s + sfx
    put(NT, k('n_members_ge50'), len(el), f'Members with >=50 posts in basis {bname}')
    for p in 'DRI': put(NT, k(f'n_members_ge50_{p}'), int((el.party == p).sum()), f'Members party {p}, basis {bname}')
    put(NT, k('n_members_ge50_senate'), int((el.chamber == 'senate').sum()), 'senators'); put(NT, k('n_members_ge50_house'), int((el.chamber == 'house').sum()), 'house')
    # factor analysis
    omega = lam.sum() ** 2 / (lam.sum() ** 2 + psi.sum()); eig = np.sort(np.linalg.eigvalsh(R))[::-1]
    kk = len(D7); rbar = (R.sum() - kk) / (kk * (kk - 1)); alpha = kk * rbar / (1 + (kk - 1) * rbar)
    fa_out = dict(basis=bname, n_members=len(el), loadings=dict(zip(D7, map(float, lam))), uniqueness=dict(zip(D7, map(float, psi))),
                  omega_total=float(omega), alpha_std=float(alpha), eigenvalues=list(map(float, eig)))
    json.dump(fa_out, open(f'{TIO}/factor_analysis_ti7{sfx}.json', 'w'), indent=1)
    put(NT, k('fa7_omega'), omega, 'McDonald omega, 1-factor, 7 dims'); put(NT, k('fa7_alpha'), alpha, 'standardized alpha 7 dims')
    put(NT, k('fa7_eig1'), eig[0], 'first eigenvalue'); put(NT, k('fa7_eig2'), eig[1], 'second eigenvalue')
    put(NT, k('fa7_loadings'), fa_out['loadings'], 'one-factor loadings'); 
    # variants
    var = {'V0_ti7': 'ti7', 'V1_binary7': 'ti_bin7', 'V1b_binary8_old': 'ti_bin8', 'V2_binary8_dampened_old': 'ti_damp8', 'V3_one_factor7': 'ti_factor7', 'V0b_ti8': 'ti8'}
    for d in D7: var[f'V4_without_{d}'] = f'loo_{d}'
    rho = el[list(var.values())].corr(method='spearman'); rho.index = rho.columns = list(var)
    rho.to_csv(f'{TIO}/variant_rank_corr{sfx}.csv')
    dr = el[el.party.isin(['D', 'R'])]
    vg = pd.DataFrame(contrast_table(dr, list(var.values()), seed=SEED)); vg.insert(0, 'variant', list(var))
    vg.to_csv(f'{TIO}/variant_party_g{sfx}.csv', index=False)
    for _, r in vg.iterrows():
        put(NT, k(f'g_party_{r.variant}'), r.hedges_g, f'Hedges g R-D, variant {r.variant}; CI [{r.g_lo:.3f},{r.g_hi:.3f}]')
    for v in ['V1_binary7', 'V2_binary8_dampened_old', 'V3_one_factor7', 'V0b_ti8']:
        put(NT, k(f'rankcorr_V0_vs_{v}'), rho.loc['V0_ti7', v], f'Spearman member ranks TI7 vs {v}')
    loo = [v for v in var if v.startswith('V4')]
    put(NT, k('rankcorr_V0_vs_V4_min'), rho.loc['V0_ti7', loo].min(), 'min Spearman TI7 vs leave-one-out')
    put(NT, k('member_ti7_mean'), el.ti7.mean(), 'mean over members of member TI7'); put(NT, k('member_ti7_sd'), el.ti7.std(ddof=1), 'SD over members')
    put(NT, k('member_ti7_median'), el.ti7.median(), 'median over members')
    put(NT, k('post_ti7_mean_member_posts'), S.ti7.mean(), 'post-level mean TI7 over member posts in basis')
    # ANOVA / ICC at post level on inference members
    po = S[S.bioguide.isin(el.bioguide)]
    x = po.ti7.values.astype(np.float64); g = po.groupby('bioguide').ti7.agg(['size', 'mean'])
    nk, mk = g['size'].values, g['mean'].values; Nn, K = len(x), len(g); grand = x.mean()
    ssb = (nk * (mk - grand) ** 2).sum(); ssw = (x ** 2).sum() - (nk * mk ** 2).sum()
    msb, msw = ssb / (K - 1), ssw / (Nn - K); Fv = msb / msw; n0 = (Nn - (nk ** 2).sum() / Nn) / (K - 1)
    icc = (msb - msw) / (msb + (n0 - 1) * msw)
    put(NT, k('anova_F_ti7'), Fv, f'One-way ANOVA across {K} members, post-level TI7'); put(NT, k('anova_df_ti7'), [K - 1, Nn - K], 'df')
    put(NT, k('anova_p_ti7'), stats.f.sf(Fv, K - 1, Nn - K), 'p (0 = below float precision)'); put(NT, k('anova_eta2_ti7'), ssb / (ssb + ssw), 'eta squared')
    put(NT, k('icc1_ti7'), icc, 'ICC(1) member share of post-level TI7 variance')
    # prevalence
    prev = []
    for d in D8 + ['advocacy']:
        prev.append(dict(dimension=d, prev_flag_all_posts=pp[f'f_{d}'].mean(), prev_flag_basis_member=S[f'f_{d}'].mean(),
                         mean_prob_all_posts=pp[f'p_{d}'].mean(), mean_prob_basis_member=S[f'p_{d}'].mean(), member_mean_prob_median=el[f'p_{d}'].median()))
    prev = pd.DataFrame(prev); prev.to_csv(f'{TIO}/prevalence{sfx}.csv', index=False)
    for _, r in prev.iterrows():
        put(NT, k(f'prev_flag_{r.dimension}'), r.prev_flag_all_posts, f'Flag share (p>0.5) all posts, {r.dimension}')
        put(NT, k(f'meanprob_{r.dimension}'), r.mean_prob_all_posts, f'Mean probability all posts, {r.dimension}')
    # member-year
    my = S.groupby(['bioguide', 'year']).agg(ti7=('ti7', 'mean'), ti8=('ti8', 'mean'), in_majority=('in_majority', 'mean'), n=('ti7', 'size')).reset_index()
    my = my.merge(meta[['bioguide', 'party', 'chamber']], on='bioguide', how='left')
    my.to_csv(f'{TIO}/member_year_ti7{sfx}.csv', index=False)
    myq = my[(my.n >= MIN_MY) & my.party.isin(['D', 'R'])].copy()
    ym = []
    for y in sorted(pp.year.unique()):
        gy = myq[myq.year == y]; sy = S[S.year == y]
        r = dict(year=int(y), post_mean_ti7_basis_member=sy.ti7.mean(), post_mean_ti7_all_posts=pp[pp.year == y].ti7.mean(),
                 member_year_mean=gy.ti7.mean(), n_members=gy.bioguide.nunique(), n_posts_member=len(sy), first_post=str(pp[pp.year == y].created_at.min().date()),
                 last_post=str(pp[pp.year == y].created_at.max().date()))
        for p in 'DR': r[f'member_year_mean_{p}'] = gy[gy.party == p].ti7.mean(); r[f'n_members_{p}'] = gy[gy.party == p].bioguide.nunique()
        ym.append(r)
        put(NT, k(f'yearly_{int(y)}_member_year_mean_ti7'), r['member_year_mean'], f'Mean TI7 over qualifying member-years {int(y)} (partial years 2017, 2023)')
        put(NT, k(f'yearly_{int(y)}_post_mean_ti7_basis_member'), r['post_mean_ti7_basis_member'], 'post-level mean TI7, member posts')
    pd.DataFrame(ym).to_csv(f'{TIO}/yearly_means_ti7{sfx}.csv', index=False)

# account level
acc = mp.groupby(['screen_name', 'bioguide', 'account_type']).agg(ti7=('ti7', 'mean'), ti8=('ti8', 'mean'), n=('ti7', 'size'),
                                                                   **{f'p_{d}': (f'p_{d}', 'mean') for d in D8}, p_advocacy=('p_advocacy', 'mean')).reset_index()
acc = acc.merge(meta[['bioguide', 'party', 'chamber']], on='bioguide', how='left')
acc.to_csv(f'{TIO}/account_ti7.csv', index=False)
aq = acc[acc.n >= MIN_P]
put(NT, 'n_accounts_member_ge50', len(aq), 'Member-mapped accounts with >=50 non-retweet posts')
for t_ in ['office', 'campaign', 'unknown']:
    s_ = aq[aq.account_type == t_]
    put(NT, f'account_mean_ti7_{t_}', s_.ti7.mean(), f'Account-level mean TI7, {t_} accounts (n={len(s_)})'); put(NT, f'account_n_{t_}', len(s_), f'accounts {t_}')
ad = aq[aq.party.isin(['D', 'R']) & aq.account_type.isin(['office', 'campaign'])]
for t_ in ['office', 'campaign', 'all']:
    s_ = ad if t_ == 'all' else ad[ad.account_type == t_]
    r = contrast_table(s_, ['ti7'], seed=SEED)[0]
    put(NT, f'account_party_g_ti7_{t_}', r['hedges_g'], f'Account-level party g (R-D) TI7, {t_}; CI [{r["g_lo"]:.3f},{r["g_hi"]:.3f}], nR={r["n1"]}, nD={r["n2"]}')

# ------------------------------------------------------------------ stats: contrasts, OLS, trend, ideology, acct type
rng = np.random.default_rng(SEED)
for bname, (M, S, sfx) in BASES.items():
    el = ELIG[bname]; dr = el[el.party.isin(['D', 'R'])].copy(); k = lambda s: s + sfx
    nR, nD = int((dr.party == 'R').sum()), int((dr.party == 'D').sum())
    put(NS, k('n_R'), nR, 'R members'); put(NS, k('n_D'), nD, 'D members')
    # contrasts
    ti_cols = ['ti7', 'ti8', 'ti_bin7', 'ti_damp8', 'ti_factor7']
    pc = [f'p_{d}' for d in D7] + ['p_advocacy']; fc = [f'f_{d}' for d in D7] + ['f_advocacy']
    r1 = contrast_table(dr, ti_cols, seed=SEED)
    for r in r1: r['family'] = 'TI'; r['p_holm'] = np.nan
    r2 = contrast_table(dr, pc + ['p_genuine_anger'], seed=SEED, holm_family=list(range(8)))
    for r in r2: r['family'] = 'prob_mean_dimensions_7plus_advocacy' if r['measure'] != 'p_genuine_anger' else 'genuine_anger_outside_family'
    r3 = contrast_table(dr, fc + ['f_genuine_anger'], seed=SEED, holm_family=list(range(8)))
    for r in r3: r['family'] = 'binary_rate_dimensions_7plus_advocacy' if r['measure'] != 'f_genuine_anger' else 'genuine_anger_outside_family'
    ct = pd.DataFrame(r1 + r2 + r3); ct.to_csv(f'{STO}/party_contrasts_ti7_{bname}.csv', index=False)
    for _, r in ct.iterrows():
        put(NS, k(f'party_{r.measure}_g'), r.hedges_g, f'Hedges g R-D {r.measure}; CI [{r.g_lo:.3f},{r.g_hi:.3f}]; Welch p={r.p:.3g}; Holm p={r.p_holm:.3g}; mean R {r.mean1:.4f}, D {r.mean2:.4f}')
        put(NS, k(f'party_{r.measure}_p'), r.p, 'Welch p'); put(NS, k(f'party_{r.measure}_pholm'), r.p_holm, 'Holm p within family')
        put(NS, k(f'party_{r.measure}_welch_t'), r.welch_t, f'Welch t (df {r.welch_df:.1f})')
    # OLS HC3 incl. office/campaign split
    ols = []
    d2 = dr.dropna(subset=['in_majority']).copy()
    d2['logposts'] = np.log(d2.n_primary); d2['R'] = (d2.party == 'R').astype(int); d2['senate'] = (d2.chamber == 'senate').astype(int)
    for dep in ['ti7', 'ti_damp8']:
        m = smf.ols(f'{dep} ~ R + senate + in_majority + logposts', data=d2).fit(cov_type='HC3'); ci = m.conf_int()
        for t in m.params.index:
            ols.append(dict(dep=dep, term=t, coef=m.params[t], se=m.bse[t], p=m.pvalues[t], lo=ci.loc[t, 0], hi=ci.loc[t, 1], n=int(m.nobs), r2=m.rsquared))
        put(NS, k(f'ols_{dep}_R_coef'), m.params['R'], f'OLS HC3 R coef {dep}; p={m.pvalues["R"]:.3g}; CI [{ci.loc["R",0]:.4f},{ci.loc["R",1]:.4f}]; n={int(m.nobs)}')
        put(NS, k(f'ols_{dep}_R_p'), m.pvalues['R'], 'HC3 p'); put(NS, k(f'ols_{dep}_R_in_sd'), m.params['R'] / d2[dep].std(ddof=1), 'R coef in member SD')
        put(NS, k(f'ols_{dep}_majority_coef'), m.params['in_majority'], 'majority coef'); put(NS, k(f'ols_{dep}_r2'), m.rsquared, 'R2')
        if dep == 'ti7':
            for t in ['senate', 'in_majority', 'logposts']: put(NS, k(f'ols_ti7_{t}_coef'), m.params[t], f'p={m.pvalues[t]:.3g}')
    pd.DataFrame(ols).to_csv(f'{STO}/ols_party_ti7{sfx}.csv', index=False)
    # office vs campaign split (member-level, accounts pooled within type)
    for t_ in ['office', 'campaign']:
        mt = mtab(S[S.account_type == t_])
        mt = mt[(mt.n_posts >= MIN_P) & mt.party.isin(['D', 'R'])]
        r = contrast_table(mt, ['ti7'], seed=SEED)[0]
        put(NS, k(f'split_{t_}_party_g_ti7'), r['hedges_g'], f'Member-level party g, {t_} accounts only; CI [{r["g_lo"]:.3f},{r["g_hi"]:.3f}], nR={r["n1"]}, nD={r["n2"]}')
        mt = mt.dropna(subset=['in_majority']).copy(); mt['R'] = (mt.party == 'R').astype(int); mt['senate'] = (mt.chamber == 'senate').astype(int); mt['logposts'] = np.log(mt.n_posts)
        m = smf.ols('ti7 ~ R + senate + in_majority + logposts', data=mt).fit(cov_type='HC3')
        put(NS, k(f'split_{t_}_ols_R_coef'), m.params['R'], f'OLS HC3 R coef {t_} only; p={m.pvalues["R"]:.3g}; n={int(m.nobs)}')
    # paired office vs campaign
    if bname == 'nonrt':
        w = aq[aq.account_type.isin(['office', 'campaign'])].groupby(['bioguide', 'account_type']).ti7.mean().unstack().dropna()
        dl = (w['campaign'] - w['office']).values; t, p = stats.ttest_1samp(dl, 0.0)
        bm = dl[rng.integers(0, len(dl), size=(10000, len(dl)))].mean(1)
        put(NS, 'accttype_n_pairs', len(w), 'members with office and campaign account'); put(NS, 'accttype_mean_office', w['office'].mean(), 'mean TI7 office')
        put(NS, 'accttype_mean_campaign', w['campaign'].mean(), 'mean TI7 campaign')
        put(NS, 'accttype_diff_campaign_minus_office', dl.mean(), f'paired diff; CI [{np.percentile(bm,2.5):.4f},{np.percentile(bm,97.5):.4f}], t={t:.2f}, p={p:.3g}')
        put(NS, 'accttype_dz', dl.mean() / dl.std(ddof=1), 'paired d_z')
    # trends
    my = pd.read_csv(f'{TIO}/member_year_ti7{sfx}.csv')
    my = my[(my.n >= MIN_MY) & my.party.isin(['D', 'R'])].copy(); my['yc'] = my.year - 2017; my['R'] = (my.party == 'R').astype(int)
    cnt = my.groupby('bioguide').year.apply(set); full5 = {b for b, s in cnt.items() if set(range(2018, 2023)) <= s}
    panels = {'full_2017_2023': my, 'balanced_2018_2022': my[my.bioguide.isin(full5) & my.year.between(2018, 2022)]}
    tr = []

    def mixed(f, d):
        last = None
        for meth in (['bfgs', 'lbfgs'], ['powell'], ['nm']):
            try: return smf.mixedlm(f, d, groups=d['bioguide']).fit(reml=True, method=meth)
            except Exception as e: last = e
        raise last
    for pn, pdf in panels.items():
        for party in 'DR':
            d = pdf[pdf.party == party]
            for model in ['mixed_RI', 'mixed_RI_majority', 'member_FE']:
                dd_ = d.dropna(subset=['in_majority']) if model == 'mixed_RI_majority' else d
                try:
                    if model == 'member_FE': m = smf.ols('ti7 ~ yc + C(bioguide)', data=dd_).fit(cov_type='cluster', cov_kwds={'groups': dd_['bioguide']})
                    else: m = mixed('ti7 ~ yc' + (' + in_majority' if model == 'mixed_RI_majority' else ''), dd_)
                    b, se = float(m.params['yc']), float(m.bse['yc'])
                    tr.append(dict(panel=pn, model=model, party=party, slope=b, se=se, lo=b - 1.96 * se, hi=b + 1.96 * se, p=2 * stats.norm.sf(abs(b / se)), n_obs=len(dd_), n_members=dd_.bioguide.nunique()))
                    put(NS, k(f'trend_{pn}_{model}_{party}_slope'), b, f'TI7 slope/yr party {party}, {pn}, {model}; SE {se:.5f}; p={2*stats.norm.sf(abs(b/se)):.3g}; n_obs={len(dd_)}, n_members={dd_.bioguide.nunique()}')
                except Exception as e:
                    tr.append(dict(panel=pn, model=model, party=party, note=str(e)[:60]))
        try:
            m = mixed('ti7 ~ yc * R', pdf); b3, se3 = float(m.params['yc:R']), float(m.bse['yc:R'])
            tr.append(dict(panel=pn, model='mixed_interaction', party='R-D', slope=b3, se=se3, p=2 * stats.norm.sf(abs(b3 / se3)), n_obs=len(pdf), n_members=pdf.bioguide.nunique()))
            put(NS, k(f'trend_{pn}_slope_diff_R_minus_D'), b3, f'party x year interaction; SE {se3:.5f}; p={2*stats.norm.sf(abs(b3/se3)):.3g}')
        except Exception as e: pass
        put(NS, k(f'trend_{pn}_n_members_total'), pdf.bioguide.nunique(), 'members in panel')
    pd.DataFrame(tr).to_csv(f'{STO}/trend_ti7{sfx}.csv', index=False)
    # ideology
    idr = []
    for grp in ['R', 'D', 'pooled']:
        s = dr if grp == 'pooled' else dr[dr.party == grp]; s = s.dropna(subset=['np_dim1'])
        for xn, xv in [('np_dim1', s.np_dim1.values), ('abs_np_dim1', np.abs(s.np_dim1.values))]:
            y = s.ti7.values; r_, p_ = stats.pearsonr(xv, y); rs, ps = stats.spearmanr(xv, y)
            idx = rng.integers(0, len(y), size=(10000, len(y))); yy, xx = y[idx], xv[idx]
            ym_, xm_ = yy - yy.mean(1, keepdims=True), xx - xx.mean(1, keepdims=True)
            rb = (ym_ * xm_).sum(1) / np.sqrt((ym_ ** 2).sum(1) * (xm_ ** 2).sum(1)); lo, hi = np.percentile(rb, [2.5, 97.5])
            idr.append(dict(group=grp, ideology=xn, n=len(y), r=r_, p=p_, lo=lo, hi=hi, rho=rs, rho_p=ps))
            put(NS, k(f'ideology_{grp}_{xn}_r'), r_, f'Pearson r member TI7 with {xn} ({grp}); n={len(y)}; CI [{lo:.3f},{hi:.3f}]; p={p_:.3g}; Spearman {rs:.3f}')
    pd.DataFrame(idr).to_csv(f'{STO}/ideology_ti7{sfx}.csv', index=False)
    # factor discrepancy: weights vs party difference
    lam, psi, R_, score, Z = fit_fa(dr, D7)
    isr = (dr.party == 'R').values; sd = Z.std(0)
    ctf = pd.DataFrame(contrast_table(dr, [f'p_{d}' for d in D7], seed=SEED))
    fdf = pd.DataFrame(dict(dimension=D7, loading=lam, uniqueness=psi, party_g=ctf.hedges_g.values, g_lo=ctf.g_lo.values, g_hi=ctf.g_hi.values))
    fdf.to_csv(f'{STO}/factor7_loadings_party_g{sfx}.csv', index=False)
    put(NS, k('factor7_party_g_one_factor_score'), hedges_g(score[isr], score[~isr]), 'party g of the one-factor score (7 dims)')
    put(NS, k('factor7_corr_mean_vs_factor'), np.corrcoef(dr.ti7.values, score)[0, 1], 'corr of mean TI7 with factor score')
    put(NS, k('factor7_table'), fdf.round(4).to_dict(orient='records'), 'loadings and per-dimension party g')

# ------------------------------------------------------------------ spec curve (probabilities only; old dampened flags computed from probs)
sh = pd.read_parquet(f'{ROOT}/data/senator_handles.parquet')[['twitter_handle', 'party']]
hh = pd.read_parquet(f'{ROOT}/data/house_handles.parquet')[['twitter_handle', 'party']]
hp = pd.concat([sh, hh]).drop_duplicates('twitter_handle').set_index('twitter_handle')['party'].map({'Republican': 'R', 'Democrat': 'D'})
MEAS = ['ti7', 'ti_damp8', 'f_advocacy']
DEF = {'ti7': 'TI7 (mean of 7 probabilities, primary)', 'ti_damp8': 'old binary TI, 8 flags, 0.3 advocacy dampener (computed from probabilities)', 'f_advocacy': 'advocacy flag rate'}
pm = pp[pp.entity_type == 'member'].copy()
pm['hp'] = pm.screen_name.map(hp)
sw = set(amap[(amap.entity_type == 'member') & (amap.party_switch == 1)].bioguide.dropna())
BS = {'all posts': lambda d: d, 'non-retweet': lambda d: d[d.post_type != 'retweet'], 'strict original': lambda d: d[d.post_type == 'original']}
rows = []


def agg(d, key):
    g = d.groupby(key)[MEAS].mean(); g['size'] = d.groupby(key).size(); return g.reset_index()


def run(step, desc, unit, basis, minp, tab, pcol):
    t = tab[tab[pcol].isin(['R', 'D']) & (tab['size'] >= minp)]
    t = t.drop(columns=['party'], errors='ignore').rename(columns={pcol: 'party'}) if pcol != 'party' else t
    for r in contrast_table(t, MEAS, seed=SEED):
        rows.append(dict(step=step, spec=desc, unit=unit, post_basis=basis, min_posts=minp, measure=r['measure'], ti_definition=DEF[r['measure']],
                         n_R=r['n1'], n_D=r['n2'], g=r['hedges_g'], ci_lo=r['g_lo'], ci_hi=r['g_hi'], p=r['p']))


meta_a = pm.drop_duplicates('screen_name').set_index('screen_name')[['hp', 'party', 'account_type', 'bioguide']]
for basis, tag in [('all posts', 'A-all'), ('non-retweet', 'A-nonrt'), ('strict original', 'A-orig')]:
    a_ = agg(BS[basis](pm), 'screen_name').join(meta_a, on='screen_name')
    run(f'{tag}1', f'accounts in the 119th-Congress handle table, {basis}', 'account', basis, 50, a_[a_.hp.notna()], 'hp')
    run(f'{tag}3', f'all member-mapped accounts, {basis}', 'account', basis, 50, a_, 'party')
    if basis != 'strict original':
        run(f'{tag}4', f'office accounts only, {basis}', 'account', basis, 50, a_[a_.account_type == 'office'], 'party')
        run(f'{tag}5', f'campaign accounts only, {basis}', 'account', basis, 50, a_[a_.account_type == 'campaign'], 'party')
mmeta = meta[['bioguide', 'party']]


def mt_(d): return agg(d, 'bioguide').merge(mmeta, on='bioguide')


mn = BS['non-retweet'](pm); mtn = mt_(mn)
run('M1', 'members, all posts, >=50', 'member', 'all posts', 50, mt_(pm), 'party')
run('M2', 'members, non-retweet posts, >=50 (PRIMARY)', 'member', 'non-retweet', 50, mtn, 'party')
run('M3', 'members, non-retweet, >=200', 'member', 'non-retweet', 200, mtn, 'party')
run('M3a', 'members, all posts, >=200', 'member', 'all posts', 200, mt_(pm), 'party')
pre = {'M001183': 'D', 'S001191': 'D', 'M001201': 'R', 'A000367': 'R'}
mb = mtn.copy()
for bg, pt_ in pre.items(): mb.loc[mb.bioguide == bg, 'party'] = pt_
run('M4b', 'members, non-retweet, party switchers under original party', 'member', 'non-retweet', 50, mb, 'party')
run('M5', 'members, office accounts only, non-retweet', 'member', 'non-retweet', 50, mt_(mn[mn.account_type == 'office']), 'party')
run('M6', 'members, campaign accounts only, non-retweet', 'member', 'non-retweet', 50, mt_(mn[mn.account_type == 'campaign']), 'party')
run('M7', 'members, strict original posts', 'member', 'strict original', 50, mt_(BS['strict original'](pm)), 'party')
sc = pd.DataFrame(rows); sc['is_primary'] = (sc.step == 'M2') & (sc.measure == 'ti7')
sc.to_csv(f'{STO}/spec_curve_full.csv' if A.source == 'full' else f'{STO}/spec_curve_sample_ti7.csv', index=False)
for m_ in ['ti7', 'ti_damp8']:
    s_ = sc[sc.measure == m_]
    put(NS, f'spec_g_min_{m_}', s_.g.min(), 'min g across steps'); put(NS, f'spec_g_max_{m_}', s_.g.max(), 'max g'); put(NS, f'spec_g_median_{m_}', s_.g.median(), 'median g')
    sm = s_[s_.unit == 'member']; put(NS, f'spec_member_g_min_{m_}', sm.g.min(), 'min g member steps'); put(NS, f'spec_member_g_max_{m_}', sm.g.max(), 'max g member steps')
for _, r in sc[sc.measure == 'ti7'].iterrows():
    put(NS, f'spec_{r.step}_ti7_g', r.g, f'{r.spec}; nR={r.n_R}, nD={r.n_D}; CI [{r.ci_lo:.3f},{r.ci_hi:.3f}]')
put(NS, 'spec_n_steps', int(sc.step.nunique()), 'steps')

# ------------------------------------------------------------------ off-the-shelf anger
an = {}
for bname, (M, S, sfx) in BASES.items():
    k = lambda s: s + sfx
    el = ELIG[bname]; dr = el[el.party.isin(['D', 'R'])].copy()
    po = S[S.bioguide.isin(dr.bioguide)].copy()
    po['anger'] = po.argmax_emotion_anger.astype(float); po['adv'] = po.p_advocacy > 0.5
    po['anger_adv'] = ((po.anger == 1) & po.adv).astype(float); po['anger_nonadv'] = ((po.anger == 1) & ~po.adv).astype(float)
    po['anger_among_nonadv'] = np.where(~po.adv, po.anger, np.nan); po['anger_among_adv'] = np.where(po.adv, po.anger, np.nan)
    po['adv_among_anger'] = np.where(po.anger == 1, po.adv.astype(float), np.nan); po['adv_among_nonanger'] = np.where(po.anger == 0, po.adv.astype(float), np.nan)
    dec = po.groupby('bioguide').agg(anger_and_advocacy=('anger_adv', 'mean'), anger_not_advocacy=('anger_nonadv', 'mean'), anger_rate_among_nonadv=('anger_among_nonadv', 'mean'),
                                     anger_rate_among_adv=('anger_among_adv', 'mean'), adv_share_among_anger_posts=('adv_among_anger', 'mean'),
                                     adv_share_among_other_posts=('adv_among_nonanger', 'mean')).reset_index()
    dr = dr.merge(dec, on='bioguide', how='left'); dr['anger_ots'] = dr.argmax_emotion_anger
    ms = ['anger_ots', 'anger_rate_among_nonadv', 'anger_rate_among_adv', 'p_emotion_anger', 'p_advocacy', 'f_advocacy', 'anger_and_advocacy', 'anger_not_advocacy', 'adv_share_among_anger_posts']
    ct = pd.DataFrame(contrast_table(dr, ms, seed=SEED)); ct.to_csv(f'{STO}/anger_ots_party_{bname}.csv', index=False)
    for _, r in ct.iterrows():
        put(NS, k(f'anger_{r.measure}_g'), r.hedges_g, f'Hedges g R-D {r.measure}; CI [{r.g_lo:.3f},{r.g_hi:.3f}]; p={r.p:.3g}; mean R {r.mean1:.4f}, D {r.mean2:.4f}')
    a = po[po.anger == 1]
    put(NS, k('anger_ots_n_posts'), len(a), 'off-the-shelf anger (argmax) posts among D/R inference members'); put(NS, k('anger_ots_post_share'), po.anger.mean(), 'share of posts called anger')
    put(NS, k('anger_ots_share_advocacy_flag'), a.adv.mean(), 'share of off-the-shelf anger posts flagged advocacy (p>0.5), post level')
    put(NS, k('anger_nonanger_share_advocacy_flag'), po[po.anger == 0].adv.mean(), 'share of non-anger posts flagged advocacy')
    put(NS, k('anger_ots_member_mean_share_advocacy'), dr.adv_share_among_anger_posts.mean(), 'member-mean share of anger posts flagged advocacy')
    for p in 'DR':
        put(NS, k(f'anger_ots_share_advocacy_flag_{p}'), a[a.party == p].adv.mean(), f'party {p}')
    put(NS, k('anger_ots_share_genuine_head'), (a.p_genuine_anger > 0.5).mean(), 'share of anger posts flagged by purpose-trained genuine-anger head')
    # advocacy vs anger relation at member level
    put(NS, k('anger_member_corr_anger_rate_vs_advocacy_rate'), np.corrcoef(dr.anger_ots, dr.f_advocacy)[0, 1], 'member-level Pearson r between off-the-shelf anger rate and advocacy flag rate')
    # post-level odds ratio
    a11 = ((po.anger == 1) & po.adv).sum(); a10 = ((po.anger == 1) & ~po.adv).sum(); a01 = ((po.anger == 0) & po.adv).sum(); a00 = ((po.anger == 0) & ~po.adv).sum()
    put(NS, k('anger_advocacy_odds_ratio'), (a11 * a00) / (a10 * a01), 'post-level odds ratio advocacy flag given anger vs not')
    # party-level anger share
    put(NS, k('anger_ots_member_mean_R'), dr[dr.party == 'R'].anger_ots.mean(), 'R'); put(NS, k('anger_ots_member_mean_D'), dr[dr.party == 'D'].anger_ots.mean(), 'D')

json.dump(NT, open(f'{TIO}/numbers_ti_full.json' if A.source == 'full' else f'{TIO}/numbers_ti_sample_ti7.json', 'w'), indent=1, default=str)
json.dump(NS, open(f'{STO}/numbers_stats_full.json' if A.source == 'full' else f'{STO}/numbers_stats_sample_ti7.json', 'w'), indent=1, default=str)
print('DONE', A.source, len(NT), len(NS))
