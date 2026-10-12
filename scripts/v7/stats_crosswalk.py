"""
stats_crosswalk.py  (task 4/6)  Merges numbers_ti.json, numbers_stats.json (+ numbers_anger.json) and writes
output/v7/stats/crosswalk.csv: old value in the draft, new value, source script, note.
Also rewrites output/v7/stats/numbers_stats.json to include the anger numbers (run after stats_main.py and stats_anger.py).
Draft = wk_final.txt (Wayne's tracked draft, accepted changes). Old values are transcribed from the draft text and tables.
"""
import json, os
import pandas as pd

ROOT = os.path.expanduser('~/x_box/pipeline')
TI = json.load(open(f'{ROOT}/output/v7/ti/numbers_ti.json'))
ST = json.load(open(f'{ROOT}/output/v7/stats/numbers_stats.json'))
SUP_ST = ST.pop('superseded', {})
for fn in ['numbers_stats_allposts', 'numbers_anger', 'numbers_decomp', 'numbers_factor', 'numbers_trend']:
    d = json.load(open(f'{ROOT}/output/v7/stats/{fn}.json'))
    d.pop('superseded', None)
    ST.update(d)
# first-pass v7 values (before the 2026-10-08 review fixes), keyed by the OLD key names
SUP_TI = TI.pop('superseded', {})
OLD = {**{k: v['value'] for k, v in SUP_TI.items() if isinstance(v, dict) and 'value' in v},
       **{k: v['value'] for k, v in SUP_ST.items() if isinstance(v, dict) and 'value' in v}}
ST['superseded'] = SUP_ST
json.dump(ST, open(f'{ROOT}/output/v7/stats/numbers_stats.json', 'w'), indent=1)
TI_OUT = dict(TI); TI_OUT['superseded'] = SUP_TI
json.dump(TI_OUT, open(f'{ROOT}/output/v7/ti/numbers_ti.json', 'w'), indent=1)
ALL = {**{k: v['value'] for k, v in TI.items()}, **{k: v['value'] for k, v in ST.items() if k != 'superseded'}}
rows = []


def add(loc, qty, old, key, src, note='', scale=1.0, fp=None):
    new = ALL.get(key, None) if isinstance(key, str) else key
    fpv = OLD.get(fp if fp else key, None) if isinstance(key, str) else None
    if isinstance(new, (int, float)) and scale != 1.0:
        new = new * scale
    rows.append(dict(manuscript_location=loc, quantity=qty, old_value_in_draft=old, first_pass_v7_value=fpv, new_value=new, new_key=key if isinstance(key, str) else '',
                     source_script=src, note=note))


# ---- Section 4.1 / Table 1 / abstract counts ----
add('Abstract, 3.1, 4.1, Table 1', 'Tweets analysed', '1,677,042', 'n_posts_total', 'ti_build.py', 'Unchanged. Includes committee, caucus and party accounts.')
add('Abstract, 3.1, 4.1', '"Members" in corpus', '1,806', 'n_members_mapped', 'ti_build.py',
    '1,806 is the number of ACCOUNTS (screen names). Accounts map to this many distinct members (bioguide) via congresstweets metadata; the rest are committee, caucus, party or unmatched accounts. See account_member_coverage.csv.')
add('Table 1', 'N scored ("members")', '1,659', 'n_members_ge50', 'ti_build.py', '1,659 was accounts with >= 20 tweets of any type. New: members with >= 50 original posts, accounts pooled.', fp='n_members_ge50_original')
add('4.1, Table 1', 'Mean TI', '0.156', 'member_ti_damp_mean', 'ti_build.py', '0.156 was the mean over 1,659 ACCOUNTS of the v2 dampened binary TI. New value is the same definition at member level on original posts. The primary probability-mean TI is a different scale (member_ti_prob_mean_orig).', fp='member_ti_damp_mean_orig')
add('4.1, Table 1', 'Mean TI (primary, probability mean)', 'n/a', 'member_ti_prob_mean', 'ti_build.py', 'New primary definition.', fp='member_ti_prob_mean_orig')
add('4.1, Table 1', 'SD of TI', '0.069', 'member_ti_damp_sd', 'ti_build.py', 'Draft 0.069 does not match the 0.075 in the v2 account table either; recomputed.', fp='member_ti_damp_sd_orig')
add('4.1, Table 1', 'Primary TI SD', 'n/a', 'member_ti_prob_sd', 'ti_build.py', fp='member_ti_prob_sd_orig')
add('4.1, Table 1', 'Share of members TI < 0.15', '51.2%', 'share_members_ti_damp_lt015', 'ti_build.py', 'Dampened binary TI, thresholds from the draft.', fp='share_members_ti_damp_lt015_orig')
add('4.1, Table 1', 'Share 0.15 to 0.30', '45.4%', 'share_members_ti_damp_015_030', 'ti_build.py', fp='share_members_ti_damp_015_030_orig')
add('4.1, Table 1', 'Share > 0.30', '3.4%', 'share_members_ti_damp_gt030', 'ti_build.py', fp='share_members_ti_damp_gt030_orig')
add('4.1, Table 1', 'Share of members, primary TI < 0.15', 'n/a', 'share_members_ti_prob_lt015', 'ti_build.py', 'Thresholds are not meaningful for the probability scale; shown for completeness.', fp='share_members_ti_prob_lt015_orig')
add('4.1, Table 1', 'ANOVA F across members', '183.4', 'anova_F_ti_damp', 'ti_build.py',
    'Draft F came from scripts/anova_party_v2.py, which passes only the FIRST 500 account groups (groups[:500]) to f_oneway. New F uses every inference-sample member. ICC(1) is the better statement of "toxicity is a member property".')
add('4.1', 'ANOVA F (primary TI)', 'n/a', 'anova_F_ti_prob', 'ti_build.py')
add('4.1', 'ICC(1), primary TI', 'n/a', 'icc1_ti_prob', 'ti_build.py', 'Share of post-level variance attributable to the member.')
add('4.1', 'ICC(1), old dampened TI', 'n/a', 'icc1_ti_damp', 'ti_build.py')

# ---- party sample ----
add('4.2, Table 2', 'n Republicans', '195', 'n_R', 'stats_main.py', 'Members (accounts pooled) with >= 50 original posts. Draft used 119th-Congress handle table matched to accounts.')
add('4.2, Table 2', 'n Democrats', '182', 'n_D', 'stats_main.py')

# ---- composite TI ----
add('Abstract, 4.2, Table 2', 'TI, Republicans (old definition)', '0.173', 'party_ti_damp_R', 'stats_main.py', 'Old dampened binary TI at member level, original posts.')
add('Abstract, 4.2, Table 2', 'TI, Democrats (old definition)', '0.138', 'party_ti_damp_D', 'stats_main.py')
add('Abstract, 4.2, Table 2', 'TI effect size', '+0.53 (Cohen d)', 'party_ti_damp_g', 'stats_main.py', 'Hedges g; CI in numbers_stats.json description.')
add('4.2', 'TI Welch t (old definition)', '+5.3', 'party_ti_damp_t', 'stats_main.py', 'v2 json had t=5.1 (pooled-variance t); draft 5.3 matched neither. Welch t now.')
add('4.2', 'TI p', '<0.001', 'party_ti_damp_p', 'stats_main.py')
add('Table 2 (primary)', 'TI primary, Republicans', 'n/a', 'party_ti_prob_R', 'stats_main.py')
add('Table 2 (primary)', 'TI primary, Democrats', 'n/a', 'party_ti_prob_D', 'stats_main.py')
add('Table 2 (primary)', 'TI primary effect size g', 'n/a', 'party_ti_prob_g', 'stats_main.py')
add('Table 2 (primary)', 'TI primary Welch p', 'n/a', 'party_ti_prob_p', 'stats_main.py')

# ---- clean / high ----
add('4.2, Table 2', '% clean tweets, R', '52.7%', 'party_clean_R', 'stats_main.py', 'Clean = dampened binary TI < 0.1 (v2 definition).')
add('4.2, Table 2', '% clean tweets, D', '61.1%', 'party_clean_D', 'stats_main.py')
add('4.2, Table 2', '% clean effect', '-0.54', 'party_clean_g', 'stats_main.py')
add('Table 2', '% high TI tweets, R', '10.3%', 'party_high_R', 'stats_main.py', 'High = dampened binary TI >= 0.5.')
add('Table 2', '% high TI tweets, D', '6.7%', 'party_high_D', 'stats_main.py')
add('Table 2', '% high TI effect', '+0.59', 'party_high_g', 'stats_main.py')

# ---- dimensions: binary flag rates (old presentation) and probability means ----
dims = [('performative', 'Performative outrage', '33.7%', '22.9%', '+0.74', '<0.001'), ('genuine_anger', 'Genuine anger', '29.9%', '22.4%', '+0.56', '<0.001'),
        ('tribal', 'Tribal signaling', '29.6%', '21.6%', '+0.54', '<0.001'), ('irony', 'Irony', '32.5%', '28.5%', '+0.47', '<0.001'),
        ('neg_sentiment', 'Negative sentiment', '25.5%', '21.0%', '+0.43', '<0.001'), ('advocacy', 'Advocacy', '61.8%', '65.8%', '-0.39', '<0.001'),
        ('ragebait', 'Ragebait', '37.8%', '32.8%', '+0.34', '0.001'), ('offensive', 'Offensive', '1.5%', '1.4%', '+0.01', '0.932'),
        ('toxic', 'Toxic', '0.3%', '0.3%', '-0.07', '0.487')]
for k, nm, oR, oD, og, op in dims:
    loc = 'Abstract, 4.2, Table 2'
    add(loc, f'{nm}: R flag rate', oR, f'party_f_{k}_R', 'stats_main.py', 'Member-level mean of the binary flag rate (p>0.5), original posts.')
    add(loc, f'{nm}: D flag rate', oD, f'party_f_{k}_D', 'stats_main.py')
    add(loc, f'{nm}: effect', og, f'party_f_{k}_g', 'stats_main.py', 'Hedges g (R minus D) on flag rates.')
    add(loc, f'{nm}: p (unadjusted)', op, f'party_f_{k}_p', 'stats_main.py')
    add(loc, f'{nm}: p (Holm within family)', 'n/a', f'party_f_{k}_pholm', 'stats_main.py')
    add('Table 2 (primary)', f'{nm}: R mean probability', 'n/a', f'party_p_{k}_R', 'stats_main.py', 'Primary presentation: mean probability.')
    add('Table 2 (primary)', f'{nm}: D mean probability', 'n/a', f'party_p_{k}_D', 'stats_main.py')
    add('Table 2 (primary)', f'{nm}: g on probabilities', 'n/a', f'party_p_{k}_g', 'stats_main.py')
    add('Table 2 (primary)', f'{nm}: Holm p on probabilities', 'n/a', f'party_p_{k}_pholm', 'stats_main.py')

# ---- trend and Table 3 ----
add('Abstract, 4.4, Table 3', 'Trend slope per year', '+0.005/yr', 'descr7_draft_table3_definition_v1_ti_all_posts_slope', 'stats_main.py',
    'Reproduces the draft 7-point regression on the v1 yearly means. Descriptive only.')
add('Abstract, 4.4, Table 3', 'Trend r', '0.882', 'descr7_draft_table3_definition_v1_ti_all_posts_r', 'stats_main.py', 'v2 json: 0.873; the draft mixes v1 and v2.')
add('4.4', 'Trend slope, v2 dampened definition, 7 points', '+0.00473 (v2 json)', 'descr7_v2_dampened_binary_all_posts_slope', 'stats_main.py', 'Descriptive.')
add('4.4', 'Trend slope, primary TI, member non-retweet posts, 7 points', 'n/a', 'descr7_primary_prob_member_nonrt_posts_slope', 'stats_main.py', 'Descriptive.', fp='descr7_primary_prob_member_original_posts_slope')
add('4.4', 'Mixed-model slope per year, Democrats (primary TI)', 'n/a', 'trend_ti_prob_slope_D_per_year', 'stats_main.py', 'Inferential replacement. Random intercept per member.')
add('4.4', 'Mixed-model slope per year, Republicans (primary TI)', 'n/a', 'trend_ti_prob_slope_R_per_year', 'stats_main.py')
add('4.4', 'Party x year interaction (primary TI)', 'n/a', 'trend_ti_prob_party_x_year_interaction', 'stats_main.py')
add('4.4', 'Mixed-model slope per year, Democrats (old definition)', 'n/a', 'trend_ti_damp_slope_D_per_year', 'stats_main.py')
add('4.4', 'Mixed-model slope per year, Republicans (old definition)', 'n/a', 'trend_ti_damp_slope_R_per_year', 'stats_main.py')
add('4.4', '"14% increase 2018 to 2023" (v1 definition)', '14%', 'yearly_change_2018_to_2023_v1_pct', 'stats_main.py', 'Percent change of the v1 yearly mean. 2023 covers Jan to Jul only.')
add('4.4', 'Percent change 2018 to 2023, primary TI', 'n/a', 'yearly_change_2018_to_2023_primary_pct', 'stats_main.py')
yr = {2017: ('0.257', '121,572'), 2018: ('0.244', '235,996'), 2019: ('0.256', '278,668'), 2020: ('0.257', '333,332'), 2021: ('0.269', '284,918'),
      2022: ('0.279', '290,030'), 2023: ('0.279', '132,526')}
for y, (m, n) in yr.items():
    add('Table 3', f'{y} mean TI', m, f'yearly_{y}_v1_ti', 'ti_build.py', 'Reproduced exactly: Table 3 is the v1 TI (8 binary flags including off-the-shelf anger, no dampener), all posts, UTC year.')
    add('Table 3', f'{y} mean TI, v2 dampened definition (what Table 1 used)', 'n/a', f'yearly_{y}_v2_damp', 'ti_build.py')
    add('Table 3', f'{y} mean TI, primary (member original posts)', 'n/a', f'yearly_{y}_primary_prob_member_nonrt', 'ti_build.py', 'Use this column in the regenerated Table 3. Non-retweet posts of member accounts; partial years 2017 and 2023 are flagged in yearly_YYYY_partial_year.', fp=f'yearly_{y}_primary_prob_member_original')
    add('Table 3', f'{y} N tweets', n, f'yearly_{y}_n_posts', 'ti_build.py')
add('Table 3 / 4.1', 'Overall mean, v1 definition (tweet level)', '(not stated)', 't3_overall_v1_ti_post_mean', 'ti_build.py', 'Explains 0.244-0.279 yearly values.')
add('Table 3 / 4.1', 'Overall mean, v2 dampened (account mean, the "0.156")', '0.156', 't3_overall_v2_damp_account_mean', 'ti_build.py', 'Mean of 1,659 account means; the tweet-level v2 mean is 0.183.')
add('Table 3 / 4.1', 'Overall mean, v2 dampened (tweet level)', '(not stated)', 't3_overall_v2_damp_post_mean', 'ti_build.py')

# ---- anger ----
add('3.3, 4.3, Table 4', 'D off-the-shelf anger rate', '44.9%', 'repro_v1_anger_D_rate', 'stats_anger.py', 'Reproduced from the v1 account table with the 119th-Congress handle party map.')
add('3.3, 4.3, Table 4', 'R off-the-shelf anger rate', '36.3%', 'repro_v1_anger_R_rate', 'stats_anger.py')
add('3.3, 4.3, Table 4', 'Off-the-shelf anger effect', '-1.20', 'repro_v1_anger_cohen_d', 'stats_anger.py', 'Reproduction of the draft number at account level.')
add('4.3, Table 4', 'Off-the-shelf anger effect, member level, original posts (Hedges g)', 'n/a', 'anger_anger_ots_g', 'stats_anger.py', 'The headline reversal must be checked against this value.')
add('4.3, Table 4', 'D off-the-shelf anger rate, member level', '44.9%', 'anger_anger_ots_D', 'stats_anger.py')
add('4.3, Table 4', 'R off-the-shelf anger rate, member level', '36.3%', 'anger_anger_ots_R', 'stats_anger.py')
add('Table 4', 'Genuine anger D', '22.4%', 'party_f_genuine_anger_D', 'stats_main.py')
add('Table 4', 'Genuine anger R', '29.9%', 'party_f_genuine_anger_R', 'stats_main.py')
add('Table 4', 'Genuine anger effect', '+0.56', 'party_f_genuine_anger_g', 'stats_main.py')
add('Abstract, 3.3, Table 4', 'Share of 500 off-the-shelf anger tweets that were advocacy', '77.4%', 'prov_share_advocacy_ge065', 'stats_anger.py',
    'Rests on a continuous model-generated advocacy label thresholded at >= 0.60 to 0.70 (77.4% at 0.60 to 0.70), not on the SetFit head. Sample = 500 random anger tweets (seed 42).')
add('3.3, Table 4', 'Share of the same 500 that were genuinely angry', '0.2%', 'prov_share_genuine_anger_ge04', 'stats_anger.py', 'One tweet of 500 with model-generated genuine anger label >= 0.40.')
add('4.2, Table 2', 'Party effect on old TI in the draft sample (119th-Congress handle accounts, all posts)', '+0.53', 'ladder2_A-all1_ti_v2_g', 'stats_decomp.py', 'Reproduced exactly (Hedges g 0.530, n 195/182).')
add('4.2, Table 2', 'Same, all member-mapped accounts, all posts', 'n/a', 'ladder2_A-all3_ti_v2_g', 'stats_decomp.py', 'Effect shrinks when the sample is not restricted to current-Congress official handles.')
add('4.2, Table 2', 'Same (old TI), member level, strict original posts (first-pass main analysis)', 'n/a', 'ladder2_M7_ti_v2_g', 'stats_decomp.py', 'Main analysis.')
add('4.2, 4.3', 'Genuine anger effect in the draft sample', '+0.56', 'ladder2_A-all1_flag_genuine_anger_g', 'stats_decomp.py', 'Reproduced (0.556).')
add('4.2, 4.3', 'Genuine anger effect, member level, original posts', 'n/a', 'ladder2_M2_flag_genuine_anger_g', 'stats_decomp.py', 'Does not replicate outside the draft sample.')
add('3.3', 'Released advocacy head on off-the-shelf anger posts, corpus', 'n/a', 'corpus_anger_posts_share_advocacy_head', 'stats_anger.py', 'Recomputed on 1.68M posts with the released head.')
add('3.3', 'Released genuine anger head on off-the-shelf anger posts, corpus', '0.2% (claimed)', 'corpus_anger_posts_share_genuine_anger_head', 'stats_anger.py', 'The released head gives a few percent, not 0.2%: the 0.2% is a statement about the continuous model-generated label (1 tweet of 500), not about the released head.')


# ---- review-fix rows (2026-10-08) ----
add('Table 2 (primary)', 'TI primary Hedges g, non-retweet posts, R minus D', 'n/a', 'party_ti_prob_g', 'stats_main.py', 'PRIMARY. Replaces the first-pass original-only 0.158.')
add('Table 2 (secondary)', 'TI primary Hedges g, all posts', 'n/a', 'party_ti_prob_g_allposts', 'stats_main.py', 'Secondary basis.')
add('Table 2 (primary)', 'OLS HC3 Republican coefficient, ti_prob', 'n/a', 'ols_ti_prob_R_coef', 'stats_main.py', 'Controls: chamber, majority share, log(non-retweet posts).')
add('Table 2 (secondary)', 'OLS HC3 Republican coefficient, ti_prob, all posts', 'n/a', 'ols_ti_prob_R_coef_allposts', 'stats_main.py')
add('4.4', 'Mixed-model slope per year, D, all posts', 'n/a', 'trend_ti_prob_slope_D_per_year_allposts', 'stats_main.py')
add('4.4', 'Mixed-model slope per year, R, all posts', 'n/a', 'trend_ti_prob_slope_R_per_year_allposts', 'stats_main.py')
add('4.5', 'Ideology r (pooled, |dim1|), non-retweet', 'n/a', 'ideology_pooled_abs_np_dim1_r', 'stats_main.py')
add('4.5', 'Ideology r (pooled, |dim1|), all posts', 'n/a', 'ideology_pooled_abs_np_dim1_r_allposts', 'stats_main.py')
add('4.1', 'ANOVA F, primary TI, all posts', 'n/a', 'anova_F_ti_prob_allposts', 'ti_build.py')
add('4.1', 'ICC(1), primary TI, all posts', 'n/a', 'icc1_ti_prob_allposts', 'ti_build.py')
add('Supplement S1 spec curve', 'Primary party g (member, non-retweet, >=50), ti_prob', 'n/a', 'ladder2_M2_ti_prob_g', 'stats_decomp.py', 'Point M2 on spec_curve.csv; every step is computed for ti_v2 and ti_prob.')
add('Supplement S1 spec curve', 'Same, ti_v2 (old binary TI)', 'n/a', 'ladder2_M2_ti_v2_g', 'stats_decomp.py')
add('Supplement S1 spec curve', 'Draft sample (119th handles, all posts), ti_prob', '+0.53', 'ladder2_A-all1_ti_prob_g', 'stats_decomp.py', 'ti_v2 reproduces 0.530; ti_prob 0.524.')
add('Supplement S1 spec curve', 'All mapped accounts, all posts, ti_prob', 'n/a', 'ladder2_A-all3_ti_prob_g', 'stats_decomp.py')
add('Supplement S1 spec curve', 'Office accounts only, all posts, ti_prob', 'n/a', 'ladder2_A-all4_ti_prob_g', 'stats_decomp.py')
add('Supplement S1 spec curve', 'Campaign accounts only, all posts, ti_prob', 'n/a', 'ladder2_A-all5_ti_prob_g', 'stats_decomp.py')
add('Supplement S1 spec curve', 'Members, min 200 non-retweet posts, ti_prob', 'n/a', 'ladder2_M3_ti_prob_g', 'stats_decomp.py')
add('Supplement S1 spec curve', 'Members, switchers under original party, ti_prob', 'n/a', 'ladder2_M4b_ti_prob_g', 'stats_decomp.py', 'M4 (exclude switchers) equals M2 because switchers carry party I/L in the map.')
add('Supplement S1 spec curve', 'Range over member-level steps, ti_prob (min)', 'n/a', 'spec_curve_member_g_min_ti_prob', 'stats_decomp.py')
add('Supplement S1 spec curve', 'Range over member-level steps, ti_prob (max)', 'n/a', 'spec_curve_member_g_max_ti_prob', 'stats_decomp.py')
add('Supplement S2 one-factor', 'One-factor party g (non-retweet)', 'n/a', 'g_party_V3_one_factor', 'ti_build.py', 'Explained in factor_loadings_party_g.csv and factor_discrepancy.json.')
add('Supplement S2 one-factor', 'Mean-TI party g (non-retweet)', 'n/a', 'factor_party_g_mean_ti', 'stats_factor.py')
add('Supplement S2 one-factor', 'Cosine, factor weights vs party-difference vector', 'n/a', 'factor_cosine_factor_weights_vs_party_difference', 'stats_factor.py')
add('Supplement S2 one-factor', 'Cosine, equal weights vs party-difference vector', 'n/a', 'factor_cosine_equal_weights_vs_party_difference', 'stats_factor.py')
add('Supplement S2 one-factor', 'Corr(party, factor score)', 'n/a', 'factor_corr_party_with_factor_score', 'stats_factor.py')
add('3.3, 4.3', 'Advocacy-head flag rate, anger-class posts (CardiffNLP four-class model)', '77.4% (label file)', 'corpus_advocacy_head_rate_anger', 'stats_anger.py', 'Full corpus, released head.')
add('3.3, 4.3', 'Advocacy-head flag rate, joy-class posts', 'n/a', 'corpus_advocacy_head_rate_joy', 'stats_anger.py')
add('3.3, 4.3', 'Advocacy-head flag rate, optimism-class posts', 'n/a', 'corpus_advocacy_head_rate_optimism', 'stats_anger.py')
add('3.3, 4.3', 'Advocacy-head flag rate, sadness-class posts', 'n/a', 'corpus_advocacy_head_rate_sadness', 'stats_anger.py')
add('3.3, 4.3', 'Genuine-anger-head flag rate, anger-class posts', '0.2% (label file)', 'corpus_genuine_anger_head_rate_anger', 'stats_anger.py')
add('3.3, 4.3', 'Genuine-anger-head flag rate, joy-class posts', 'n/a', 'corpus_genuine_anger_head_rate_joy', 'stats_anger.py', 'Large: the head flags most joy-class posts, a validity concern.')
add('3.3, 4.3', 'Genuine-anger-head flag rate, optimism-class posts', 'n/a', 'corpus_genuine_anger_head_rate_optimism', 'stats_anger.py')
add('3.3, 4.3', 'Genuine-anger-head flag rate, sadness-class posts', 'n/a', 'corpus_genuine_anger_head_rate_sadness', 'stats_anger.py')
add('3.3', '500-post validation: advocacy label >= 0.65, all 500', '77.4%', 'prov_share_advocacy_ge065', 'stats_anger.py')
add('3.3', '500-post validation: same excluding the 185 advocacy-pool posts', 'n/a', 'prov_share_advocacy_ge065_excl_trainpool', 'stats_anger.py', 'Rises, because the pool removes most low-advocacy rows; not an unbiased held-out estimate.')
add('3.3', '500-post validation: genuine-anger label >= 0.40, excluding the 120 pool posts', '0.2%', 'prov_share_genuine_anger_ge04_excl_trainpool', 'stats_anger.py')
add('4.3', 'Member-level g, EM anger rate, non-retweet', '-1.20 (account level)', 'anger_anger_ots_g', 'stats_anger.py')
add('4.3', 'Member-level g, EM anger rate among NON-advocacy posts', 'n/a', 'anger_anger_rate_among_nonadv_posts_g', 'stats_anger.py', 'The first-pass claim that non-advocacy anger is equal across parties used the share of ALL posts that are anger and not advocacy (g -0.15). Conditional on being non-advocacy, the gap persists.')
add('4.3', 'Member-level g, genuine-anger head probability', 'n/a', 'anger_p_genuine_anger_g', 'stats_anger.py')
add('4.3', 'Member-level g, genuine-anger head flag rate', '+0.56', 'anger_f_genuine_anger_g', 'stats_anger.py')
for pn in ['full_2017_2023', 'full_years_2018_2022', 'balanced_2018_2022']:
    for pty in ['D', 'R']:
        add('4.4, Supplement S3 trend', f'Slope per year {pty}, mixed random intercept, {pn}', 'n/a', f'trend2_nonrt_{pn}_mixed_RI_{pty}_slope', 'stats_trend.py')
        add('4.4, Supplement S3 trend', f'Slope per year {pty}, member fixed effects, {pn}', 'n/a', f'trend2_nonrt_{pn}_member_FE_{pty}_slope', 'stats_trend.py')
add('4.4, Supplement S3 trend', 'Party difference in slope (R minus D), full 2017-2023', 'n/a', 'trend2_nonrt_full_2017_2023_slope_diff_R_minus_D', 'stats_trend.py')
add('4.4', 'Yearly member-year mean ti_prob, 2017 (partial)', 'n/a', 'yearly2_nonrt_2017_member_year_mean_all', 'stats_trend.py', 'Partial year: from 2017-06-22.')
add('4.4', 'Yearly member-year mean ti_prob, 2023 (partial)', 'n/a', 'yearly2_nonrt_2023_member_year_mean_all', 'stats_trend.py', 'Partial year: to 2023-07-11.')

cw = pd.DataFrame(rows)
cw.to_csv(f'{ROOT}/output/v7/stats/crosswalk.csv', index=False)
missing = cw[cw.new_key.ne('') & cw.new_value.isna()]
print(len(cw), 'rows;', len(missing), 'rows with missing new value')
print(missing[['quantity', 'new_key']].to_string())
