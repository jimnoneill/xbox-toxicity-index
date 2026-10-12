"""
stats_anger.py  (review fixes M2/M3, task 4)  The anger correction claim (draft Sections 3.3 and 4.3, Table 4).

EM = "a widely used four-class emotion model (CardiffNLP)": anger / joy / optimism / sadness, argmax class per post
(stored in output/toxicity_index/congress_classified_2017_2023.parquet, column emotion_label, row-aligned with post_probs).

Draft claims: (a) of 500 tweets EM called "anger", 77.4% were advocacy and 0.2% genuinely angry;
(b) the EM anger party effect was d = -1.20 (Democrats higher) and reversed to d = +0.56 for genuine anger.

Computed here
  1. Provenance of (a): the labelled 1,000-tweet file (500 EM-anger, 500 other, seed 42). Its political_advocacy and genuine_anger
     columns are CONTINUOUS MODEL-GENERATED LABELS in [0,1], not outputs of the released SetFit heads. The 77.4% and 0.2% are
     recomputed at several thresholds, and again EXCLUDING the posts that sat in the pools used to train the released advocacy
     head (185 of the 500) and the genuine-anger head (120 of the 500). Training pools are reconstructed from
     retrain_and_rebuild_ti.py (first 200 positives and first 200 negatives in file order).
  2. Full corpus (all 1,677,042 posts) with the released heads: advocacy-head and genuine-anger-head base rates by EM class
     (anger, joy, optimism, sadness), as flag rates (p>0.5) and mean probabilities.
  3. Member-level party contrasts (R minus D, Hedges g, bootstrap CI), on the primary basis (non-retweet posts) and, with the
     suffix _allposts, on all posts: EM anger rate; EM anger rate among non-advocacy posts; genuine-anger head (probability, rate);
     advocacy; anger-and-advocacy and anger-and-not-advocacy shares of posts.
  4. Reproduction of the draft's d = -1.20 from the v1 account table with the 119th Congress handle table.
"""
import glob, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats_common import Numbers, contrast_table, hedges_g

ROOT = os.path.expanduser('~/x_box/pipeline')
SUF = os.environ.get('V7_SUFFIX', '')
TI = f'{ROOT}/output/v7/ti{SUF}'
MAPDIR = f'{ROOT}/output/v7/ti'
OUT = f'{ROOT}/output/v7/stats{SUF}'
N = Numbers()
NA = Numbers(suffix='_allposts')
EM = 'a widely used four-class emotion model (CardiffNLP)'

# ---- 1. provenance of the 77.4% ----
lab = pd.read_parquet(glob.glob(f'{ROOT}/data/*_advocacy_anger_1k.parquet')[0]).reset_index(drop=True)
ang = lab[lab.cardiff_emotion == 'anger']
N.add('prov_n_labelled_total', len(lab), 'Rows in the labelled file (500 EM-anger + 500 non-anger, random sample, seed 42)')
N.add('prov_n_anger_sample', len(ang), f'Rows whose class under {EM} is anger')
rows = []
for th in [0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.8, 1.0]:
    rows.append(dict(threshold=th, share_adv_ge=(ang.political_advocacy >= th).mean(), share_genuine_anger_ge=(ang.genuine_anger >= th).mean()))
th_tab = pd.DataFrame(rows)
th_tab.to_csv(f'{OUT}/anger_label_thresholds.csv', index=False)

# training pools (reconstructed, file order)
tr_adv = pd.concat([lab[lab.political_advocacy >= 0.65].head(200), lab[lab.political_advocacy <= 0.30].head(200)])
tr_ga = pd.concat([lab[lab.genuine_anger >= 0.40].head(200), lab[lab.genuine_anger <= 0.10].head(200)])
lab['in_adv_pool'] = lab.index.isin(tr_adv.index)
lab['in_ga_pool'] = lab.index.isin(tr_ga.index)
ang = lab[lab.cardiff_emotion == 'anger']
n_in_adv, n_in_ga = int(ang.in_adv_pool.sum()), int(ang.in_ga_pool.sum())
N.add('prov_anger_sample_in_advocacy_training_pool', n_in_adv, 'Anger-sample rows inside the pool used to train the advocacy head (80/20 split inside the pool, so an unknown subset is held out)')
N.add('prov_anger_sample_in_genuine_anger_training_pool', n_in_ga, 'Anger-sample rows inside the genuine anger training pool')

val_rows = []
for th in [0.60, 0.65, 0.70]:
    for excl, sub in [('all 500', ang), (f'excluding advocacy-pool rows (n={len(ang) - n_in_adv})', ang[~ang.in_adv_pool])]:
        val_rows.append(dict(statistic='share advocacy label >= threshold', threshold=th, sample=excl, n=len(sub), value=(sub.political_advocacy >= th).mean()))
for th in [0.3, 0.4, 0.5]:
    for excl, sub in [('all 500', ang), (f'excluding genuine-anger-pool rows (n={len(ang) - n_in_ga})', ang[~ang.in_ga_pool])]:
        val_rows.append(dict(statistic='share genuine-anger label >= threshold', threshold=th, sample=excl, n=len(sub), value=(sub.genuine_anger >= th).mean()))
pd.DataFrame(val_rows).to_csv(f'{OUT}/anger_validation_excl_training.csv', index=False)
N.add('prov_share_advocacy_ge065', (ang.political_advocacy >= 0.65).mean(), f'Share of the 500 anger-class tweets ({EM}) with model-generated advocacy label >= 0.65 (draft: 77.4%)')
N.add('prov_share_advocacy_ge065_excl_trainpool', (ang[~ang.in_adv_pool].political_advocacy >= 0.65).mean(),
      f'Same share excluding the {n_in_adv} anger-class tweets that were in the advocacy-head training pool (n={len(ang) - n_in_adv})')
N.add('prov_share_advocacy_ge06', (ang.political_advocacy >= 0.6).mean(), 'Same at >= 0.60')
N.add('prov_share_advocacy_ge06_excl_trainpool', (ang[~ang.in_adv_pool].political_advocacy >= 0.6).mean(), 'Same at >= 0.60, excluding the advocacy training pool rows')
N.add('prov_share_advocacy_ge05', (ang.political_advocacy >= 0.5).mean(), 'Same at >= 0.50')
N.add('prov_share_advocacy_ge05_excl_trainpool', (ang[~ang.in_adv_pool].political_advocacy >= 0.5).mean(), 'Same at >= 0.50, excluding the advocacy training pool rows')
N.add('prov_share_genuine_anger_ge04', (ang.genuine_anger >= 0.4).mean(), 'Share with model-generated genuine anger label >= 0.40 (draft: 0.2%)')
N.add('prov_share_genuine_anger_ge04_excl_trainpool', (ang[~ang.in_ga_pool].genuine_anger >= 0.4).mean(), f'Same, excluding the {n_in_ga} anger-class tweets in the genuine-anger training pool (n={len(ang) - n_in_ga})')
N.add('prov_mean_advocacy_label', ang.political_advocacy.mean(), 'Mean advocacy label in the anger sample')
N.add('prov_mean_advocacy_label_excl_trainpool', ang[~ang.in_adv_pool].political_advocacy.mean(), 'Mean advocacy label in the anger sample excluding the advocacy training pool rows')
N.add('prov_mean_genuine_anger_label', ang.genuine_anger.mean(), 'Mean genuine anger label in the anger sample')
N.add('prov_advocacy_all1k_ge065', (lab.political_advocacy >= 0.65).mean(), 'Share of all 1,000 labelled tweets with advocacy >= 0.65 (the base rate)')
N.add('prov_advocacy_by_emotion', lab.groupby('cardiff_emotion').political_advocacy.apply(lambda s: float((s >= 0.65).mean())).to_dict(),
      f'1,000-tweet file: share with advocacy label >= 0.65 by class of {EM}. Anger is not special: joy is higher')
N.add('prov_advocacy_by_emotion_excl_trainpool', lab[~lab.in_adv_pool].groupby('cardiff_emotion').political_advocacy.apply(lambda s: float((s >= 0.65).mean())).to_dict(),
      'Same, excluding rows in the advocacy-head training pool')
N.add('prov_label_source', 'model-generated labels (continuous 0-1), not human labels; released SetFit heads were trained on a subset of the same file',
      'What the 77.4% and 0.2% rest on')

# ---- 2. full corpus with the released heads ----
cols = ['screen_name', 'post_type', 'p_advocacy', 'p_genuine_anger', 'p_emotion_anger', 'argmax_emotion_anger']
pp = pd.read_parquet(f'{TI}/post_probs.parquet', columns=cols)
v1 = pd.read_parquet(f'{ROOT}/output/toxicity_index/congress_classified_2017_2023.parquet', columns=['emotion_label'])
assert len(v1) == len(pp)
pp['emotion'] = v1['emotion_label'].astype(str).values
agree = float(((pp.emotion == 'anger').values.astype(int) == pp.argmax_emotion_anger.values).mean())
N.add('corpus_emotion_label_agreement_with_argmax_anger', agree, 'Agreement between the stored four-class label (anger vs rest) and the argmax_emotion_anger flag used elsewhere (row alignment check)')
pp['adv'] = pp.p_advocacy > 0.5
pp['ga'] = pp.p_genuine_anger > 0.5
byem = pp.groupby('emotion').agg(n_posts=('adv', 'size'), advocacy_head_flag_rate=('adv', 'mean'), advocacy_head_mean_prob=('p_advocacy', 'mean'),
                                 genuine_anger_head_flag_rate=('ga', 'mean'), genuine_anger_head_mean_prob=('p_genuine_anger', 'mean'))
byem['share_of_corpus'] = byem.n_posts / len(pp)
byem['advocacy_head_flag_rate_gap_vs_all_other_classes'] = [
    pp[pp.emotion == e].adv.mean() - pp[pp.emotion != e].adv.mean() for e in byem.index]
byem['genuine_anger_head_flag_rate_gap_vs_all_other_classes'] = [
    pp[pp.emotion == e].ga.mean() - pp[pp.emotion != e].ga.mean() for e in byem.index]
byem.to_csv(f'{OUT}/anger_head_rates_by_emotion_class.csv')
for e, r in byem.iterrows():
    N.add(f'corpus_advocacy_head_rate_{e}', r.advocacy_head_flag_rate, f'Full corpus: share of {e}-class posts ({EM}) flagged advocacy by the released head (n={int(r.n_posts)})')
    N.add(f'corpus_genuine_anger_head_rate_{e}', r.genuine_anger_head_flag_rate, f'Full corpus: share of {e}-class posts ({EM}) flagged genuine anger by the released head')
a = pp[pp.argmax_emotion_anger == 1]
N.add('corpus_n_offtheshelf_anger_posts', len(a), f'Posts {EM} calls anger (argmax), all accounts')
N.add('corpus_share_offtheshelf_anger', pp.argmax_emotion_anger.mean(), f'Share of all posts called anger by {EM}')
N.add('corpus_anger_posts_share_advocacy_head', a.adv.mean(), 'Among EM-anger posts, share flagged advocacy by the released SetFit head')
N.add('corpus_anger_posts_share_genuine_anger_head', a.ga.mean(), 'Among EM-anger posts, share flagged genuine anger by the released SetFit head')
N.add('corpus_anger_posts_share_adv_and_not_ga', (a.adv & ~a.ga).mean(), 'Among EM-anger posts, advocacy and not genuine anger')
N.add('corpus_anger_posts_share_neither', (~a.adv & ~a.ga).mean(), 'Among EM-anger posts, neither advocacy nor genuine anger')
N.add('corpus_nonanger_posts_share_advocacy_head', pp[pp.argmax_emotion_anger == 0].adv.mean(), 'Among non-anger posts, share flagged advocacy (base-rate comparison)')
N.add('corpus_nonanger_posts_share_genuine_anger_head', pp[pp.argmax_emotion_anger == 0].ga.mean(), 'Among non-anger posts, share flagged genuine anger')

# ---- 3. member-level contrasts, primary (non-retweet) and all posts ----
amap = pd.read_csv(f'{MAPDIR}/account_member_map.csv')
pp = pp.merge(amap[['screen_name', 'bioguide', 'entity_type']], on='screen_name', how='left')
pp['anger'] = pp.argmax_emotion_anger.astype(float)
pp['anger_and_advocacy'] = ((pp.anger == 1) & pp.adv).astype(float)
pp['anger_not_advocacy'] = ((pp.anger == 1) & ~pp.adv).astype(float)
pp['anger_nonadv_only'] = np.where(~pp.adv, pp.anger, np.nan)   # anger rate among non-advocacy posts
pp['anger_adv_only'] = np.where(pp.adv, pp.anger, np.nan)       # anger rate among advocacy posts
measures = ['anger_ots', 'anger_rate_among_nonadv_posts', 'anger_rate_among_adv_posts', 'p_emotion_anger', 'p_genuine_anger', 'f_genuine_anger',
            'p_advocacy', 'f_advocacy', 'anger_and_advocacy', 'anger_not_advocacy']
allct = []
for basis, NN, sfx in [('nonrt', N, ''), ('allposts', NA, '_allposts')]:
    mo = pd.read_csv(f'{TI}/member_ti_inference_sample{sfx}.csv')
    dr = mo[mo.party.isin(['D', 'R'])].copy()
    dr['anger_ots'] = dr['argmax_emotion_anger']
    po = pp[pp.entity_type.eq('member') & pp.bioguide.isin(dr.bioguide)]
    if basis == 'nonrt':
        po = po[po.post_type != 'retweet']
    dec = po.groupby('bioguide').agg(anger_and_advocacy=('anger_and_advocacy', 'mean'), anger_not_advocacy=('anger_not_advocacy', 'mean'),
                                     anger_rate_among_nonadv_posts=('anger_nonadv_only', 'mean'), anger_rate_among_adv_posts=('anger_adv_only', 'mean')).reset_index()
    dr = dr.merge(dec, on='bioguide', how='left')
    rows = contrast_table(dr, measures)
    ct = pd.DataFrame(rows)
    ct.insert(0, 'basis', basis)
    allct.append(ct)
    for r in rows:
        NN.add(f'anger_{r["measure"]}_R', r['mean1'], f'Member-level mean, Republicans: {r["measure"]} (basis {basis}; {EM} for anger_*)')
        NN.add(f'anger_{r["measure"]}_D', r['mean2'], f'Member-level mean, Democrats: {r["measure"]} (basis {basis})')
        NN.add(f'anger_{r["measure"]}_g', r['hedges_g'], f'Hedges g (R minus D): {r["measure"]}, basis {basis}; CI [{r["g_lo"]:.3f}, {r["g_hi"]:.3f}], p={r["p"]:.3g}')
pd.concat(allct).to_csv(f'{OUT}/anger_party_contrasts.csv', index=False)

# ---- 4. reproduce the draft numbers from the v1 table ----
v1t = pd.read_csv(f'{ROOT}/output/toxicity_index/all_members_toxicity_index.csv')
sh = pd.read_parquet(f'{ROOT}/data/senator_handles.parquet')[['twitter_handle', 'party']]
hh = pd.read_parquet(f'{ROOT}/data/house_handles.parquet')[['twitter_handle', 'party']]
pm = pd.concat([sh, hh]).drop_duplicates('twitter_handle').set_index('twitter_handle')['party']
v1t['party'] = v1t.screen_name.map(pm)
vv = v1t[v1t.party.isin(['Republican', 'Democrat'])]
r_, d_ = vv[vv.party == 'Republican'], vv[vv.party == 'Democrat']
col = 'flag_anger_rate'
x1, x2 = r_[col].dropna().values, d_[col].dropna().values
pooled = np.sqrt(((len(x1) - 1) * x1.var(ddof=1) + (len(x2) - 1) * x2.var(ddof=1)) / (len(x1) + len(x2) - 2))
N.add('repro_v1_anger_R_rate', x1.mean(), f'v1 account table (119th-Congress handle party map), EM anger rate, R (n={len(x1)}); draft Table 4: 36.3%')
N.add('repro_v1_anger_D_rate', x2.mean(), f'Same, D (n={len(x2)}); draft: 44.9%')
N.add('repro_v1_anger_cohen_d', (x1.mean() - x2.mean()) / pooled, 'Cohen d (R minus D) from the v1 account table; draft: -1.20')
N.d.update(NA.d)
N.save(f'{OUT}/numbers_anger.json', superseded_from=f'{OUT}/superseded_v1/numbers_anger.json')
print(th_tab.round(3).to_string())
print(pd.DataFrame(val_rows).round(3).to_string())
print(byem.round(4).to_string())
print(pd.concat(allct)[['basis', 'measure', 'mean1', 'mean2', 'hedges_g', 'g_lo', 'g_hi', 'p']].round(4).to_string())
for k in ['corpus_anger_posts_share_advocacy_head', 'corpus_anger_posts_share_genuine_anger_head', 'repro_v1_anger_cohen_d',
          'prov_share_advocacy_ge065', 'prov_share_advocacy_ge065_excl_trainpool', 'prov_share_genuine_anger_ge04',
          'prov_anger_sample_in_advocacy_training_pool', 'prov_anger_sample_in_genuine_anger_training_pool', 'corpus_emotion_label_agreement_with_argmax_anger']:
    print(k, N.d[k]['value'])
