"""
stats_trend.py  (review fixes M5/M6, task 5)  Trend robustness, by party, on ONE definition: ti_prob (mean of 8 probabilities).

Basis: non-retweet posts (primary) and all posts (secondary, column basis = allposts). Member-year cell = member with >= 20 posts in the basis
in that calendar year. year coded yc = year - 2017. Corpus covers 2017-06-22 to 2023-07-11, so 2017 and 2023 are PARTIAL years (flagged).

Panels
  full_2017_2023        every qualifying member-year (unbalanced; includes both partial years)
  full_years_2018_2022  unbalanced, 2018-2022 only (partial years dropped)
  balanced_2018_2022    members qualifying in ALL five years 2018-2022 (the review's balanced panel)
  balanced_2017_2023    members qualifying in all seven years (includes partial endpoints)
Models, fitted separately WITHIN each party
  mixed_RI              ti ~ yc + (1 | member), REML random intercept; slope per year, model SE
  mixed_RI_majority     same plus member-year share of posts made while the member's party held the chamber majority
  member_FE             ti ~ yc + member fixed effects, member-clustered SE
and a pooled interaction model (ti ~ yc * R + (1 | member)) giving the party difference in slopes.
Outputs (output/v7/stats/): trend_robustness.csv, trend_party_difference.csv, yearly_means_consistent.csv, numbers_trend.json
"""
import os, sys, warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats_common import Numbers

ROOT = os.path.expanduser('~/x_box/pipeline')
TI = f'{ROOT}/output/v7/ti'
OUT = f'{ROOT}/output/v7/stats'
N = Numbers()
MIN_MY = 20
PARTIAL = {2017, 2023}


def mixed(formula, data):
    last = None
    for meth in (['bfgs', 'lbfgs'], ['powell'], ['nm']):
        try:
            return smf.mixedlm(formula, data, groups=data['bioguide']).fit(reml=True, method=meth)
        except Exception as e:
            last = e
    raise last


rows, diff_rows, ym_rows = [], [], []
t3 = pd.read_csv(f'{TI}/table3_yearly.csv')
for basis, sfx in [('nonrt', ''), ('allposts', '_allposts')]:
    my = pd.read_csv(f'{TI}/member_year_ti{sfx}.csv')
    my = my[(my.n >= MIN_MY) & my.party.isin(['D', 'R'])].copy()
    my['yc'] = my.year - 2017
    my['R'] = (my.party == 'R').astype(int)
    cnt = my.groupby('bioguide').year.apply(set)
    full5 = {b for b, s in cnt.items() if set(range(2018, 2023)) <= s}
    full7 = {b for b, s in cnt.items() if set(range(2017, 2024)) <= s}
    panels = {'full_2017_2023': my, 'full_years_2018_2022': my[my.year.between(2018, 2022)],
              'balanced_2018_2022': my[my.bioguide.isin(full5) & my.year.between(2018, 2022)],
              'balanced_2017_2023': my[my.bioguide.isin(full7)]}
    for pname, pd_ in panels.items():
        for party in ['D', 'R']:
            d = pd_[pd_.party == party].copy()
            for model in ['mixed_RI', 'mixed_RI_majority', 'member_FE']:
                dd = d.dropna(subset=['in_majority']) if model == 'mixed_RI_majority' else d
                try:
                    if model == 'member_FE':
                        m = smf.ols('ti_prob ~ yc + C(bioguide)', data=dd).fit(cov_type='cluster', cov_kwds={'groups': dd['bioguide']})
                    else:
                        m = mixed('ti_prob ~ yc' + (' + in_majority' if model == 'mixed_RI_majority' else ''), dd)
                    b, se = float(m.params['yc']), float(m.bse['yc'])
                    z = b / se
                    rows.append(dict(basis=basis, panel=pname, model=model, party=party, slope_per_year=b, se=se, ci_lo=b - 1.96 * se, ci_hi=b + 1.96 * se,
                                     p=2 * stats.norm.sf(abs(z)), n_obs=len(dd), n_members=dd.bioguide.nunique(),
                                     includes_partial_years=bool(set(dd.year) & PARTIAL)))
                except Exception as e:
                    rows.append(dict(basis=basis, panel=pname, model=model, party=party, slope_per_year=np.nan, n_obs=len(dd), note=str(e)[:80]))
        # pooled interaction: party difference in slopes
        try:
            m = mixed('ti_prob ~ yc * R', pd_)
            cov = m.cov_params()
            b3, se3 = float(m.params['yc:R']), float(m.bse['yc:R'])
            diff_rows.append(dict(basis=basis, panel=pname, slope_D=float(m.params['yc']), slope_R=float(m.params['yc'] + b3), slope_diff_R_minus_D=b3,
                                  se_diff=se3, ci_lo=b3 - 1.96 * se3, ci_hi=b3 + 1.96 * se3, p=2 * stats.norm.sf(abs(b3 / se3)),
                                  n_obs=len(pd_), n_members=pd_.bioguide.nunique()))
        except Exception as e:
            diff_rows.append(dict(basis=basis, panel=pname, note=str(e)[:80]))
    # yearly means on one definition (ti_prob), partial-year flags
    for y in range(2017, 2024):
        g = my[my.year == y]
        pr = t3[t3.year == y].iloc[0]
        r = dict(basis=basis, year=y, partial_year=bool(pr.partial_year), first_post=pr.first_post, last_post=pr.last_post, share_of_year_covered=pr.share_of_year_covered,
                 n_members=g.bioguide.nunique(), mean_posts_per_qualifying_member=g.n.mean(),
                 post_weighted_mean_all=np.average(g.ti_prob, weights=g.n), member_year_mean_all=g.ti_prob.mean())
        for party in ['D', 'R']:
            h = g[g.party == party]
            r[f'n_members_{party}'] = h.bioguide.nunique()
            r[f'member_year_mean_{party}'] = h.ti_prob.mean()
            r[f'member_year_se_{party}'] = h.ti_prob.std(ddof=1) / np.sqrt(len(h))
            r[f'post_weighted_mean_{party}'] = np.average(h.ti_prob, weights=h.n)
        # same cohort mean: members present in all of 2018-2022 (composition held fixed)
        hb = my[my.bioguide.isin(full5) & (my.year == y)]
        r['balanced_cohort_n'] = hb.bioguide.nunique()
        for party in ['D', 'R']:
            r[f'balanced_cohort_mean_{party}'] = hb[hb.party == party].ti_prob.mean()
        ym_rows.append(r)
tr = pd.DataFrame(rows)
tr.to_csv(f'{OUT}/trend_robustness.csv', index=False)
pdiff = pd.DataFrame(diff_rows)
pdiff.to_csv(f'{OUT}/trend_party_difference.csv', index=False)
ym = pd.DataFrame(ym_rows)
ym.to_csv(f'{OUT}/yearly_means_consistent.csv', index=False)
for _, r in tr.iterrows():
    if pd.isna(r.slope_per_year):
        continue
    N.add(f'trend2_{r.basis}_{r.panel}_{r.model}_{r.party}_slope', r.slope_per_year,
          f'ti_prob slope per year, party {r.party}, panel {r.panel}, model {r.model}, basis {r.basis}; SE {r.se:.5f}, 95% CI [{r.ci_lo:.4f}, {r.ci_hi:.4f}], p={r.p:.3g}; n_obs={int(r.n_obs)}, n_members={int(r.n_members)}')
for _, r in pdiff.iterrows():
    if 'slope_diff_R_minus_D' in r and not pd.isna(r.get('slope_diff_R_minus_D', np.nan)):
        N.add(f'trend2_{r.basis}_{r.panel}_slope_diff_R_minus_D', r.slope_diff_R_minus_D,
              f'Party difference in yearly slope (R minus D), mixed model with interaction; SE {r.se_diff:.5f}, p={r.p:.3g}')
for _, r in ym.iterrows():
    N.add(f'yearly2_{r.basis}_{int(r.year)}_member_year_mean_all', r.member_year_mean_all, f'Mean ti_prob over qualifying member-years, {int(r.year)}, basis {r.basis}; partial_year={r.partial_year}; n_members={int(r.n_members)}')
N.add('trend2_definition', 'ti_prob (mean of 8 probabilities), non-retweet posts primary; member-year cell needs >= 20 posts; 2017 (from 2017-06-22) and 2023 (to 2023-07-11) are partial years',
      'One consistent definition for all trend and yearly-mean results in this file')
N.save(f'{OUT}/numbers_trend.json')
pd.set_option('display.width', 250)
print(tr[tr.basis == 'nonrt'][['panel', 'model', 'party', 'slope_per_year', 'se', 'ci_lo', 'ci_hi', 'p', 'n_obs', 'n_members']].round(4).to_string())
print(pdiff.round(4).to_string())
print(ym[ym.basis == 'nonrt'].round(4).T.to_string())
