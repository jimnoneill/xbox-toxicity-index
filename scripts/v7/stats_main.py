"""
stats_main.py  (spec E)  Member-level party contrasts, OLS (HC3), mixed-model trend, ideology, account type.

REVISION 2026-10-08: primary basis = all non-retweet posts (BASIS=nonrt, default); secondary = all posts (BASIS=allposts, keys and files
carry the suffix _allposts). The first-pass original-only outputs are in output/v7/stats/superseded_v1/ and under "superseded" in numbers_stats.json.

Run with PYTHONPATH=~/x_box/pylib_v7 (statsmodels).  Inputs come from ti_build.py (output/v7/ti/).
Outputs (output/v7/stats/): party_contrasts_nonrt.csv, party_contrasts_allposts.csv, ols_party{,_allposts}.csv,
  trend_mixed.csv, trend_descriptive_7pt.csv, ideology.csv, account_type.csv, numbers_stats.json

Conventions
  Group 1 = Republican, group 2 = Democrat: positive g means Republicans higher.
  Hedges g = J * (m1 - m2) / sp, J = 1 - 3/(4 df - 1). Bootstrap CI: 10,000 member resamples stratified by party, seed 20261008.
  Welch t with Welch-Satterthwaite df. Holm across the dimension family (8 dimensions + advocacy), separately for the
  probability-mean family and the binary-flag-rate family. TI rows are reported outside the family.
  Independents (n small) are excluded from party contrasts and kept in the OLS only if party D or R (they are dropped).
"""
import json, os, sys
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats_common import Numbers, contrast_table, hedges_g, boot_g, SEED, B_DEFAULT

ROOT = os.path.expanduser('~/x_box/pipeline')
SUF = os.environ.get('V7_SUFFIX', '')
TI = f'{ROOT}/output/v7/ti{SUF}'
MAPDIR = f'{ROOT}/output/v7/ti'
OUT = f'{ROOT}/output/v7/stats{SUF}'
os.makedirs(OUT, exist_ok=True)
DIMS = ['offensive', 'toxic', 'neg_sentiment', 'irony', 'ragebait', 'tribal', 'performative', 'genuine_anger']
BASIS = os.environ.get('BASIS', 'nonrt')
SFX = '' if BASIS == 'nonrt' else '_allposts'
N = Numbers(suffix=SFX)

elig = pd.read_csv(f'{TI}/member_ti_inference_sample{SFX}.csv')
dr = elig[elig.party.isin(['D', 'R'])].copy()
nR, nD = int((dr.party == 'R').sum()), int((dr.party == 'D').sum())
N.add('n_R', nR, 'Republican members with >= 50 posts in the basis')
N.add('n_D', nD, 'Democratic members with >= 50 posts in the basis')
N.add('n_I_excluded', int((elig.party == 'I').sum()), 'Independents excluded from party contrasts')
N.add('J_factor', float(1 - 3 / (4 * (nR + nD - 2) - 1)), 'Hedges small-sample correction J for the party contrast')


def run_contrasts(tab, label):
    rows = []
    # TI measures (outside the Holm family)
    ti_cols = ['ti_prob', 'ti_damp', 'ti_bin', 'ti_factor', 'clean', 'high', 'zero_flags']
    ti_cols = [c for c in ti_cols if c in tab.columns]
    r1 = contrast_table(tab, ti_cols)
    for r in r1:
        r['family'] = 'TI'; r['p_holm'] = np.nan
    # probability-mean family
    pc = [f'p_{d}' for d in DIMS] + ['p_advocacy']
    r2 = contrast_table(tab, pc)
    for r in r2:
        r['family'] = 'prob_mean_dimensions'
    fc = [f'f_{d}' for d in DIMS] + ['f_advocacy']
    r3 = contrast_table(tab, fc)
    for r in r3:
        r['family'] = 'binary_rate_dimensions'
    df = pd.DataFrame(r1 + r2 + r3)
    df.to_csv(f'{OUT}/party_contrasts_{label}.csv', index=False)
    return df


pc_o = run_contrasts(dr, BASIS)
for _, r in pc_o.iterrows():
    N.add(f'party_{r.measure}_R', r.mean1, f'Member-level mean of {r.measure}, Republicans (posts in the basis)')
    N.add(f'party_{r.measure}_D', r.mean2, f'Member-level mean of {r.measure}, Democrats (posts in the basis)')
    N.add(f'party_{r.measure}_g', r.hedges_g, f'Hedges g R minus D for {r.measure}; 95% bootstrap CI [{r.g_lo:.3f}, {r.g_hi:.3f}]')
    N.add(f'party_{r.measure}_t', r.welch_t, f'Welch t for {r.measure} (df {r.welch_df:.1f})')
    N.add(f'party_{r.measure}_p', r.p, f'Welch p for {r.measure} (unadjusted)')
    N.add(f'party_{r.measure}_pholm', r.p_holm, f'Holm-adjusted p within family {r.family}')

# ---------------- OLS with HC3 ----------------
ols_rows = []
for dep in ['ti_prob', 'ti_damp']:
    d2 = dr.dropna(subset=['in_majority']).copy()
    d2['logposts'] = np.log(d2['n_primary'])
    d2['R'] = (d2.party == 'R').astype(int)
    d2['senate'] = (d2.chamber == 'senate').astype(int)
    m = smf.ols(f'{dep} ~ R + senate + in_majority + logposts', data=d2).fit(cov_type='HC3')
    ci = m.conf_int()
    for term in m.params.index:
        ols_rows.append(dict(dep=dep, term=term, coef=m.params[term], se_hc3=m.bse[term], t=m.tvalues[term], p=m.pvalues[term],
                             lo=ci.loc[term, 0], hi=ci.loc[term, 1], n=int(m.nobs), r2=m.rsquared))
    N.add(f'ols_{dep}_R_coef', m.params['R'], f'OLS HC3 coefficient on Republican (vs Democrat) for member {dep}, controls chamber, majority share, log(posts in basis); p={m.pvalues["R"]:.2g}')
    N.add(f'ols_{dep}_R_p', m.pvalues['R'], f'HC3 p for the Republican coefficient, {dep}')
    N.add(f'ols_{dep}_R_ci', [ci.loc['R', 0], ci.loc['R', 1]], f'95% CI for the Republican coefficient, {dep}')
    N.add(f'ols_{dep}_n', int(m.nobs), f'N members in OLS for {dep}')
    N.add(f'ols_{dep}_r2', m.rsquared, f'R squared, {dep}')
    sd = d2[dep].std(ddof=1)
    N.add(f'ols_{dep}_R_coef_in_sd', m.params['R'] / sd, f'Republican coefficient in SD units of member {dep}')
pd.DataFrame(ols_rows).to_csv(f'{OUT}/ols_party{SFX}.csv', index=False)

# ---------------- Trend: member-year mixed model ----------------
def fit_mixed(formula, data):
    last = None
    for meth in (['bfgs', 'lbfgs'], ['powell'], ['nm']):
        try:
            return smf.mixedlm(formula, data, groups=data['bioguide']).fit(reml=True, method=meth)
        except Exception as e:  # singular fit on degenerate (synthetic) data
            last = e
    raise last

my = pd.read_csv(f'{TI}/member_year_ti{SFX}.csv')
my = my[(my.n >= 20) & my.party.isin(['D', 'R'])].copy()
my['yc'] = my['year'] - 2017
my['R'] = (my.party == 'R').astype(int)
trend_rows = []
for dep in ['ti_prob', 'ti_damp']:
    m = fit_mixed(f'{dep} ~ yc * R', my)
    cov = m.cov_params()
    b1 = m.params['yc']; se1 = m.bse['yc']
    b3 = m.params['yc:R']
    sR = b1 + b3
    seR = float(np.sqrt(cov.loc['yc', 'yc'] + cov.loc['yc:R', 'yc:R'] + 2 * cov.loc['yc', 'yc:R']))
    var_u = float(m.cov_re.iloc[0, 0]); var_e = float(m.scale)
    for nm, est, se in [('slope_D_per_year', b1, se1), ('slope_R_per_year', sR, seR), ('party_x_year_interaction', b3, m.bse['yc:R']),
                        ('party_gap_at_2017_R_minus_D', m.params['R'], m.bse['R']), ('intercept_D_2017', m.params['Intercept'], m.bse['Intercept'])]:
        z = est / se
        trend_rows.append(dict(dep=dep, term=nm, est=est, se=se, z=z, p=2 * stats.norm.sf(abs(z)), lo=est - 1.96 * se, hi=est + 1.96 * se,
                               n_obs=len(my), n_members=my.bioguide.nunique(), var_member=var_u, var_resid=var_e))
        N.add(f'trend_{dep}_{nm}', est, f'Mixed model ({dep}, random intercept per member): {nm}; SE {se:.5f}, p={2*stats.norm.sf(abs(z)):.3g}')
    N.add(f'trend_{dep}_n_obs', len(my), 'Member-year observations (>= 20 posts in the basis in the year)')
    N.add(f'trend_{dep}_n_members', int(my.bioguide.nunique()), 'Members in the trend panel')
    N.add(f'trend_{dep}_icc_member', var_u / (var_u + var_e), f'Model-based ICC (member share of member-year variance), {dep}')
    # majority-status control (opposition posting is more negative; parties switch roles during 2017-2023)
    mj = my.dropna(subset=['in_majority'])
    mm2 = fit_mixed(f'{dep} ~ yc * R + in_majority', mj)
    cv = mm2.cov_params()
    seR2 = float(np.sqrt(cv.loc['yc', 'yc'] + cv.loc['yc:R', 'yc:R'] + 2 * cv.loc['yc', 'yc:R']))
    N.add(f'trend_{dep}_majctrl_slope_D', mm2.params['yc'], f'Mixed model with member-year majority share as covariate: slope D per year ({dep}); SE {mm2.bse["yc"]:.5f}')
    N.add(f'trend_{dep}_majctrl_slope_R', mm2.params['yc'] + mm2.params['yc:R'], f'Same, slope R per year; SE {seR2:.5f}')
    N.add(f'trend_{dep}_majctrl_majority_coef', mm2.params['in_majority'], f'Same, coefficient on majority share ({dep})')
    # member fixed effects, clustered SE (robustness)
    mf = smf.ols(f'{dep} ~ yc + yc:R + C(bioguide)', data=my).fit(cov_type='cluster', cov_kwds={'groups': my['bioguide']})
    N.add(f'trend_{dep}_FE_slope_D', mf.params['yc'], f'Member fixed-effects slope per year, Democrats, {dep}; cluster SE {mf.bse["yc"]:.5f}')
    N.add(f'trend_{dep}_FE_slope_R', mf.params['yc'] + mf.params['yc:R'], f'Member fixed-effects slope per year, Republicans, {dep}')
    # balanced panel
    ok = my[(my.year >= 2018) & (my.year <= 2022)].groupby('bioguide').year.nunique()
    keep = ok[ok == 5].index
    mb = my[my.bioguide.isin(keep) & my.year.between(2018, 2022)]
    if len(keep) > 30:
        mm = fit_mixed(f'{dep} ~ yc * R', mb)
        N.add(f'trend_{dep}_balanced_slope_D', mm.params['yc'], f'Balanced panel (members with >=20 posts in each of 2018-2022, n={len(keep)}), slope D, {dep}')
        N.add(f'trend_{dep}_balanced_slope_R', mm.params['yc'] + mm.params['yc:R'], f'Balanced panel slope R, {dep}')
        N.add(f'trend_{dep}_balanced_n_members', int(len(keep)), 'Members in the balanced panel')
pd.DataFrame(trend_rows).to_csv(f'{OUT}/trend_mixed{SFX}.csv', index=False)

# descriptive 7-point regression (not inferential)
t3 = pd.read_csv(f'{TI}/table3_yearly.csv')
desc = []
for col in ['draft_table3_definition_v1_ti_all_posts', 'v2_dampened_binary_all_posts', 'primary_prob_all_posts', 'primary_prob_member_nonrt_posts', 'primary_prob_member_all_posts']:
    lr = stats.linregress(t3.year.values.astype(float), t3[col].values)
    desc.append(dict(series=col, slope=lr.slope, r=lr.rvalue, p=lr.pvalue, n_points=len(t3)))
    N.add(f'descr7_{col}_slope', lr.slope, f'DESCRIPTIVE 7-point regression slope per year, {col}; r={lr.rvalue:.3f}, p={lr.pvalue:.4f}; not inferential (serial dependence, 7 points)')
    N.add(f'descr7_{col}_r', lr.rvalue, f'r of the same 7-point regression')
pd.DataFrame(desc).to_csv(f'{OUT}/trend_descriptive_7pt{SFX}.csv', index=False)
YCOL = 'primary_prob_member_nonrt_posts' if BASIS == 'nonrt' else 'primary_prob_member_all_posts'
first = t3.iloc[1][YCOL]; last = t3.iloc[-1][YCOL]
N.add('yearly_change_2018_to_2023_primary_pct', 100 * (last / first - 1), 'Percent change in the yearly mean of primary TI (member posts in the basis) from 2018 to 2023 (2023 = Jan-Jul)')
f1 = t3.iloc[1]['draft_table3_definition_v1_ti_all_posts']; l1 = t3.iloc[-1]['draft_table3_definition_v1_ti_all_posts']
N.add('yearly_change_2018_to_2023_v1_pct', 100 * (l1 / f1 - 1), 'Percent change 2018 to 2023 in the v1 TI (the draft claimed 14%)')

# ---------------- Ideology ----------------
idr = []
rng = np.random.default_rng(SEED)
for dep in ['ti_prob', 'ti_damp']:
    for grp in ['R', 'D', 'pooled']:
        s = dr if grp == 'pooled' else dr[dr.party == grp]
        s = s.dropna(subset=['np_dim1'])
        for xn, xv in [('np_dim1', s.np_dim1.values), ('abs_np_dim1', np.abs(s.np_dim1.values))]:
            y = s[dep].values
            r, p = stats.pearsonr(xv, y)
            rs, ps = stats.spearmanr(xv, y)
            idx = rng.integers(0, len(y), size=(10000, len(y)))
            yy, xx = y[idx], xv[idx]
            ym, xm = yy - yy.mean(1, keepdims=True), xx - xx.mean(1, keepdims=True)
            rb = (ym * xm).sum(1) / np.sqrt((ym ** 2).sum(1) * (xm ** 2).sum(1))
            lo, hi = np.percentile(rb, [2.5, 97.5])
            idr.append(dict(dep=dep, group=grp, ideology=xn, n=len(y), pearson_r=r, p=p, r_lo=lo, r_hi=hi, spearman_rho=rs, spearman_p=ps))
            if dep == 'ti_prob':
                N.add(f'ideology_{grp}_{xn}_r', r, f'Pearson r of member ti_prob with {xn} ({grp}); n={len(y)}, 95% bootstrap CI [{lo:.3f}, {hi:.3f}], p={p:.3g}')
pd.DataFrame(idr).to_csv(f'{OUT}/ideology{SFX}.csv', index=False)

# ---------------- Account type (office vs campaign), paired within member ----------------
if BASIS == 'nonrt':  # account-type analysis uses the non-retweet account table
    acc = pd.read_csv(f'{TI}/account_ti.csv')
    acc = acc[acc.account_type.isin(['office', 'campaign']) & (acc.n_orig >= 50)]
    w = acc.groupby(['bioguide', 'account_type']).agg(ti=('ti_prob', 'mean')).unstack()
    w.columns = [c[1] for c in w.columns]
    pair = w.dropna()
    if len(pair) > 10:
        dlt = (pair['campaign'] - pair['office']).values
        t, p = stats.ttest_1samp(dlt, 0.0)
        dz = dlt.mean() / dlt.std(ddof=1)
        idx = rng.integers(0, len(dlt), size=(10000, len(dlt)))
        bm = dlt[idx].mean(1)
        lo, hi = np.percentile(bm, [2.5, 97.5])
        gav = hedges_g(pair['campaign'].values, pair['office'].values)
        pd.DataFrame([dict(n_pairs=len(pair), mean_office=pair['office'].mean(), mean_campaign=pair['campaign'].mean(), mean_diff=dlt.mean(),
                           diff_lo=lo, diff_hi=hi, paired_t=t, p=p, dz=dz, hedges_g_between=gav)]).to_csv(f'{OUT}/account_type.csv', index=False)
        N.add('accttype_n_pairs', len(pair), 'Members with both an office and a campaign account, each with >= 50 non-retweet posts')
        N.add('accttype_mean_office', pair['office'].mean(), 'Mean primary TI, office accounts (paired members)')
        N.add('accttype_mean_campaign', pair['campaign'].mean(), 'Mean primary TI, campaign accounts (paired members)')
        N.add('accttype_diff_campaign_minus_office', dlt.mean(), f'Paired mean difference; 95% bootstrap CI [{lo:.4f}, {hi:.4f}], paired t={t:.2f}, p={p:.3g}')
        N.add('accttype_dz', dz, 'Paired standardized difference (d_z)')

    # account counts by type (from the map)
    am = pd.read_csv(f'{MAPDIR}/account_member_map.csv')
    for k, v in am[am.entity_type == 'member'].account_type.value_counts().items():
        N.add(f'accounts_member_type_{k}', int(v), f'Member-mapped accounts of type {k}')

N.save(f'{OUT}/numbers_stats{SFX}.json', superseded_from=(f'{OUT}/superseded_v1/numbers_stats.json' if BASIS == 'nonrt' else None))
print(pc_o[['measure', 'mean1', 'mean2', 'hedges_g', 'g_lo', 'g_hi', 'p', 'p_holm', 'family']].round(4).to_string())
print(pd.DataFrame(ols_rows).round(4).to_string())
print(pd.DataFrame(trend_rows).round(5).to_string())
print(pd.DataFrame(desc).round(4).to_string())
print(pd.DataFrame(idr).round(3).to_string())
