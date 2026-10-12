"""
ti_assemble_post_probs.py  (task 2)

Assembles output/v7/ti/post_probs.parquet from output/v7/ti/_probs/*.npy and checks that binarized
probabilities reproduce the v2 flags in output/toxicity_index_v2/congress_ti_v2.parquet (row-aligned;
alignment is verified on screen_name and created_at).

Columns: id (string), screen_name, user_id (string), created_at (UTC), post_type (original|reply|retweet),
         p_<head> for the 8 TI dimensions + p_advocacy + p_emotion_anger (off-the-shelf), and
         argmax_neg_sentiment, argmax_emotion_anger (uint8; the indicator the v2 flags actually used).
post_type: congresstweets has no type field. retweet if text starts with "RT @", reply if text starts with "@",
         else original. This is an approximation (quote tweets count as original, replies that do not start
         with @ are missed).
Output: agreement_v2.csv (per head: n, agreement at p>0.5, agreement of the exact v2 rule, kappa, n_mismatch, mismatch rate).
"""
import json, os
import numpy as np
import pandas as pd

ROOT = os.path.expanduser('~/x_box/pipeline')
OUT = f'{ROOT}/output/v7/ti'
P = f'{OUT}/_probs'
HEADS = ['offensive', 'toxic', 'neg_sentiment', 'irony', 'ragebait', 'tribal', 'performative', 'genuine_anger',
         'advocacy', 'emotion_anger']

df = pd.read_parquet(f'{ROOT}/data/congresstweets_2017_2023.parquet',
                     columns=['id', 'screen_name', 'user_id', 'created_at', 'text'])
df['created_at'] = pd.to_datetime(df['created_at'], utc=True, errors='coerce')
df = df.dropna(subset=['created_at']).reset_index(drop=True)
t = df['text'].astype(str)
df['post_type'] = np.where(t.str.startswith('RT @'), 'retweet', np.where(t.str.startswith('@'), 'reply', 'original'))
df = df.drop(columns='text')
df['id'] = df['id'].astype(str)
EXTRA = ['hate', 'epistemic', 'engbait', 'agency']
HEADS_ALL = HEADS + [h for h in EXTRA if os.path.exists(f'{P}/{h}.npy')]
for h in HEADS_ALL:
    a = np.load(f'{P}/{h}.npy')
    assert len(a) == len(df), (h, len(a), len(df))
    df[f'p_{h}'] = a
for h in ['neg_sentiment', 'emotion_anger']:
    df[f'argmax_{h}'] = np.load(f'{P}/{h}_argmax.npy').astype(np.uint8)
df.to_parquet(f'{OUT}/post_probs.parquet', index=False)
print('saved', df.shape, df.post_type.value_counts().to_dict())

v2 = pd.read_parquet(f'{ROOT}/output/toxicity_index_v2/congress_ti_v2.parquet')
assert len(v2) == len(df)
v2['created_at'] = pd.to_datetime(v2['created_at'], utc=True)
assert (v2['screen_name'].values == df['screen_name'].values).all()
assert (v2['created_at'].values == df['created_at'].values).all()
print('row alignment with v2 verified on screen_name and created_at')

rows = []
mapping = {'offensive': 'flag_offensive', 'toxic': 'flag_toxic', 'neg_sentiment': 'flag_neg_sentiment',
           'irony': 'flag_irony', 'ragebait': 'flag_ragebait', 'tribal': 'flag_tribal',
           'performative': 'flag_performative', 'genuine_anger': 'flag_genuine_anger', 'advocacy': 'flag_advocacy'}
for h, fc in mapping.items():
    v = v2[fc].values.astype(int)
    b05 = (df[f'p_{h}'].values > 0.5).astype(int)
    exact = df[f'argmax_{h}'].values.astype(int) if f'argmax_{h}' in df else b05
    for rule, b in [('p>0.5', b05), ('v2 rule', exact)]:
        agree = (b == v).mean()
        po = agree
        pe = v.mean() * b.mean() + (1 - v.mean()) * (1 - b.mean())
        rows.append(dict(head=h, rule=rule, n=len(v), v2_prevalence=v.mean(), new_prevalence=b.mean(),
                         agreement=agree, mismatches=int((b != v).sum()), mismatch_rate=1 - agree,
                         cohen_kappa=(po - pe) / (1 - pe) if pe < 1 else np.nan))
# off-the-shelf anger vs the v1 classified parquet
v1 = pd.read_parquet(f'{ROOT}/output/toxicity_index/congress_classified_2017_2023.parquet', columns=['emotion_label'])
assert len(v1) == len(df)
ang = (v1['emotion_label'].isin(['anger', 'LABEL_0'])).values.astype(int)
b = df['argmax_emotion_anger'].values.astype(int)
rows.append(dict(head='emotion_anger (v1 flag_anger)', rule='v1 rule', n=len(b), v2_prevalence=ang.mean(),
                 new_prevalence=b.mean(), agreement=(b == ang).mean(), mismatches=int((b != ang).sum()),
                 mismatch_rate=(b != ang).mean(), cohen_kappa=np.nan))
ag = pd.DataFrame(rows)
ag.to_csv(f'{OUT}/agreement_v2.csv', index=False)
print(ag.to_string())


# ---- Lee archive (same checkpoints, same TI scale) ----
PL = f'{OUT}/_probs_lee'
if os.path.exists(f'{PL}/advocacy.npy'):
    lee = pd.read_parquet(f'{ROOT}/data/BasedMikeLee_archive_v2.parquet', columns=['id', 'created_at', 'type'])
    lee['id'] = lee['id'].astype(str)
    lee['created_at'] = pd.to_datetime(lee['created_at'], utc=True, errors='coerce')
    lee['post_type'] = lee['type'].replace({'tweet': 'original'})
    for h in HEADS + [x for x in EXTRA if os.path.exists(f'{PL}/{x}.npy')]:
        lee[f'p_{h}'] = np.load(f'{PL}/{h}.npy')
    for h in ['neg_sentiment', 'emotion_anger']:
        lee[f'argmax_{h}'] = np.load(f'{PL}/{h}_argmax.npy').astype(np.uint8)
    DIMS8 = ['offensive', 'toxic', 'neg_sentiment', 'irony', 'ragebait', 'tribal', 'performative', 'genuine_anger']
    lee['ti_prob'] = lee[[f'p_{d}' for d in DIMS8]].mean(axis=1)
    lee.to_parquet(f'{OUT}/post_probs_lee.parquet', index=False)
    print('saved Lee', lee.shape, lee.groupby('post_type').ti_prob.agg(['size', 'mean']).to_dict())
