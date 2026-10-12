"""
stats_factor.py  (review fix M1, task 3)  Why does the one-factor TI give a party g of about 0.05 while the mean gives about 0.16-0.18?

Member-level, inference sample (>= 50 non-retweet posts; the all-posts basis is repeated with the suffix _allposts).
Dimension means are z-scored across members. One-factor model (sklearn FactorAnalysis, regression/Thomson factor scores).

Outputs (output/v7/stats/):
  factor_loadings_party_g.csv   one row per dimension: loading, uniqueness, communality, factor-score weight (z units), weight in
                                raw-probability units, member SD, party mean difference (R-D) in raw probability and in member SD units,
                                Hedges g with bootstrap CI and Holm p, contribution of the dimension to the party g of (a) the equal-weight
                                mean and (b) the factor score (contribution = weight * standardized party difference / SD of the composite).
  factor_discrepancy.json       summary quantities that explain the gap (cosines, composite party g, two-factor solution, residual g).
  numbers_factor.json           numbers keyed for the manuscript.

Logic of the decomposition. For a composite S = sum_k w_k z_k, party difference in S = sum_k w_k * dz_k where dz_k is the R-D mean
difference of z_k, and SD(S) = sqrt(w' R w). Dividing gives the part of the composite's standardized party difference carried by
dimension k. The first factor is the direction of greatest SHARED variance; the party difference lies in a different direction of the
8-dimensional space (irony, tribal signaling, performative outrage up for Republicans; ragebait, negative sentiment, offensive, toxic slightly
down), so a score built to capture shared variance gives those dimensions little or negative weight net.
"""
import json, os, sys
import numpy as np
import pandas as pd
from sklearn.decomposition import FactorAnalysis

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats_common import Numbers, contrast_table, hedges_g, holm

ROOT = os.path.expanduser('~/x_box/pipeline')
TI = f'{ROOT}/output/v7/ti'
OUT = f'{ROOT}/output/v7/stats'
DIMS = ['offensive', 'toxic', 'neg_sentiment', 'irony', 'ragebait', 'tribal', 'performative', 'genuine_anger']
N = Numbers()
summary = {}
for basis, sfx in [('nonrt', ''), ('allposts', '_allposts')]:
    NN = Numbers(suffix=sfx)
    el = pd.read_csv(f'{TI}/member_ti_inference_sample{sfx}.csv')
    el = el[el.party.isin(['D', 'R'])].copy()
    X = el[[f'p_{d}' for d in DIMS]].values
    sd = X.std(0, ddof=0)
    Z = (X - X.mean(0)) / sd
    is_r = (el.party == 'R').values
    fa = FactorAnalysis(n_components=1, random_state=0).fit(Z)
    lam = fa.components_[0].copy(); sign = 1 if lam.sum() >= 0 else -1; lam = sign * lam
    psi = fa.noise_variance_
    w = (lam / psi) / (1 + (lam ** 2 / psi).sum())                # regression-method factor score weights on z
    score = Z @ w
    assert np.allclose(score, sign * fa.transform(Z)[:, 0], atol=1e-6)
    R = np.corrcoef(Z.T)
    dz = Z[is_r].mean(0) - Z[~is_r].mean(0)                       # R-D difference in global-SD units
    # per-dimension party contrasts (probability means), with bootstrap CI and Holm across the 8
    ct = pd.DataFrame(contrast_table(el, [f'p_{d}' for d in DIMS]))
    ct['p_holm8'] = holm(ct.p.values)
    w_eq = sd / 8.0                                               # equal-weight mean of raw probabilities, expressed in z units
    sdS_eq, sdS_f = np.sqrt(w_eq @ R @ w_eq), np.sqrt(w @ R @ w)
    contrib_eq = w_eq * dz / sdS_eq
    contrib_f = w * dz / sdS_f
    g_eq = hedges_g(el.loc[is_r, 'ti_prob'], el.loc[~is_r, 'ti_prob'])
    g_f = hedges_g(el.loc[is_r, 'ti_factor'], el.loc[~is_r, 'ti_factor']) if 'ti_factor' in el else hedges_g(score[is_r], score[~is_r])
    tab = pd.DataFrame(dict(
        dimension=DIMS, loading=lam, uniqueness=psi, communality=lam ** 2,
        factor_weight_z=w, factor_weight_share_abs=np.abs(w) / np.abs(w).sum(),
        factor_weight_raw=w / sd, mean_weight_raw=1 / 8.0, mean_weight_share=sd / sd.sum(),
        member_sd=sd, party_diff_raw=ct['diff'].values, party_diff_in_member_sd=dz,
        party_g=ct.hedges_g.values, party_g_lo=ct.g_lo.values, party_g_hi=ct.g_hi.values, party_p=ct.p.values, party_p_holm=ct.p_holm8.values,
        contrib_to_mean_g=contrib_eq, contrib_to_factor_g=contrib_f))
    tab.insert(0, 'basis', basis)
    tab.to_csv(f'{OUT}/factor_loadings_party_g{sfx}.csv', index=False)
    # discriminant direction (party) vs factor direction
    Sw = (np.cov(Z[is_r].T) * (is_r.sum() - 1) + np.cov(Z[~is_r].T) * ((~is_r).sum() - 1)) / (len(Z) - 2)
    lda = np.linalg.solve(Sw, dz)
    cos_f_lda = float(w @ lda / np.linalg.norm(w) / np.linalg.norm(lda))
    cos_f_dz = float(w @ dz / np.linalg.norm(w) / np.linalg.norm(dz))
    cos_eq_dz = float(w_eq @ dz / np.linalg.norm(w_eq) / np.linalg.norm(dz))
    # residual of the mean TI after removing the factor score (what the factor leaves out)
    b = np.polyfit(score, el.ti_prob.values, 1)
    resid = el.ti_prob.values - np.polyval(b, score)
    g_resid = hedges_g(resid[is_r], resid[~is_r])
    r_party_factor = float(np.corrcoef(is_r.astype(float), score)[0, 1])
    r_party_mean = float(np.corrcoef(is_r.astype(float), el.ti_prob.values)[0, 1])
    r_mean_factor = float(np.corrcoef(el.ti_prob.values, score)[0, 1])
    # two-factor varimax solution
    fa2 = FactorAnalysis(n_components=2, rotation='varimax', random_state=0).fit(Z)
    L2 = fa2.components_.T
    S2 = fa2.transform(Z)
    g2 = [float(hedges_g(S2[is_r, j], S2[~is_r, j])) for j in range(2)]
    # orient each factor so its largest loading is positive
    for j in range(2):
        if L2[np.argmax(np.abs(L2[:, j])), j] < 0:
            L2[:, j] *= -1; S2[:, j] *= -1; g2[j] = -g2[j]
    two = pd.DataFrame(L2, columns=['loading_f1', 'loading_f2']); two.insert(0, 'dimension', DIMS)
    two.to_csv(f'{OUT}/factor_two_factor_loadings{sfx}.csv', index=False)
    d = dict(basis=basis, n_members=len(el), n_R=int(is_r.sum()), n_D=int((~is_r).sum()),
             party_g_mean_ti=float(g_eq), party_g_one_factor_score=float(g_f), sum_contrib_mean=float(contrib_eq.sum()), sum_contrib_factor=float(contrib_f.sum()),
             cosine_factor_weights_vs_party_difference=cos_f_dz, cosine_equal_weights_vs_party_difference=cos_eq_dz,
             cosine_factor_weights_vs_party_discriminant=cos_f_lda,
             corr_party_with_factor_score=r_party_factor, corr_party_with_mean_ti=r_party_mean, corr_mean_ti_with_factor_score=r_mean_factor,
             party_g_of_mean_ti_residual_after_factor=float(g_resid),
             two_factor_party_g=g2, two_factor_loadings=two.set_index('dimension').round(3).to_dict(),
             dims_with_party_g_gt_0_2=[d_ for d_, g_ in zip(DIMS, ct.hedges_g) if g_ > 0.2],
             dims_with_negative_party_g=[d_ for d_, g_ in zip(DIMS, ct.hedges_g) if g_ < 0],
             highest_loading_dims=[DIMS[i] for i in np.argsort(-lam)[:3]],
             mean_loading_of_dims_with_g_gt_0_2=float(np.mean([l for l, g_ in zip(lam, ct.hedges_g) if g_ > 0.2])),
             mean_loading_of_other_dims=float(np.mean([l for l, g_ in zip(lam, ct.hedges_g) if g_ <= 0.2])))
    summary[basis] = d
    for k_, v_ in d.items():
        if isinstance(v_, (int, float)):
            NN.add(f'factor_{k_}', v_, f'One-factor discrepancy diagnostic: {k_} (basis {basis})')
    NN.add('factor_table', tab.round(4).to_dict(orient='records'), f'Per-dimension loading, uniqueness, factor weight, party g, contributions (basis {basis}); same as factor_loadings_party_g{sfx}.csv')
    NN.add('factor_two_factor_party_g', g2, f'Party g of the two varimax factor scores (basis {basis})')
    N.d.update(NN.d)
    pd.set_option('display.width', 250)
    print(basis, 'n', len(el))
    print(tab.drop(columns=['basis']).round(3).to_string())
    print(json.dumps({k: v for k, v in d.items() if k != 'two_factor_loadings'}, indent=1, default=float))
    print(two.round(3).to_string())
json.dump(summary, open(f'{OUT}/factor_discrepancy.json', 'w'), indent=1, default=float)
N.save(f'{OUT}/numbers_factor.json')
