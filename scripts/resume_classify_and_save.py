"""
Resume from where retrain_and_rebuild_ti.py crashed:
- Load pre-trained SetFit models (advocacy, genuine_anger)
- Classify all 1.68M tweets 
- Rebuild TI v2
- Save per-tweet parquet (with duplicate column fix)
- Run overlap analysis
"""
import os, sys, time, gc
import pandas as pd
import numpy as np
import torch
from scipy import stats

sys.path.insert(0, '/home/joneill/x_box/pipeline')

DEVICE = 'cuda:0'
out_dir = '/home/joneill/x_box/pipeline/output'
ti_dir = f'{out_dir}/toxicity_index_v2'
os.makedirs(ti_dir, exist_ok=True)

# Load pre-classified tweets (6 CardiffNLP + 4 old SetFit heads)
print('Loading classified tweets...')
ct = pd.read_parquet(f'{out_dir}/toxicity_index/congress_classified_2017_2023.parquet')
texts = ct['text'].astype(str).tolist()
n = len(texts)
print(f'Loaded {n:,} tweets')

# Run new SetFit classifiers (pre-trained, skip training)
from setfit import SetFitModel

for dim in ['political_advocacy', 'genuine_anger']:
    path = f'/home/joneill/x_box/pipeline/models_v3/setfit_{dim}'
    if not os.path.exists(path):
        print(f'  {dim}: model not found, SKIPPING')
        continue
    t0 = time.time()
    print(f'Classifying with {dim}...')
    model = SetFitModel.from_pretrained(path).to(DEVICE)
    preds = []
    for i in range(0, n, 512):
        batch = texts[i:i+512]
        p = model.predict(batch)
        preds.extend(p.tolist() if hasattr(p, 'tolist') else list(p))
        if i > 0 and i % 100000 == 0:
            elapsed = time.time() - t0
            rate = i / elapsed
            eta = (n - i) / rate
            print(f'  {dim}: {i:,}/{n:,} ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)')
    ct[f'setfit_{dim}'] = preds
    rate = sum(preds) / len(preds)
    print(f'  {dim}: {rate:.1%} positive ({time.time()-t0:.0f}s)')
    del model
    torch.cuda.empty_cache()
    gc.collect()

# Build flags
print('\nBuilding TI v2 flags...')
ct['flag_offensive'] = (ct['offensive_label'].isin(['offensive', 'LABEL_1', '1'])).astype(float)
ct['flag_toxic'] = (ct['toxicity_label'].isin(['toxic', 'LABEL_1', '1'])).astype(float)
ct['flag_neg_sentiment'] = (ct['sentiment_label'].isin(['negative', 'LABEL_0'])).astype(float)
ct['flag_irony'] = (ct['irony_label'].isin(['irony', 'LABEL_1', '1'])).astype(float)
ct['flag_ragebait'] = ct['setfit_ragebait'].astype(float)
ct['flag_tribal'] = ct['setfit_tribal'].astype(float)
ct['flag_performative'] = ct['setfit_performative'].astype(float)
ct['flag_genuine_anger'] = ct['setfit_genuine_anger'].astype(float) if 'setfit_genuine_anger' in ct.columns else 0.0
ct['flag_advocacy'] = ct['setfit_political_advocacy'].astype(float) if 'setfit_political_advocacy' in ct.columns else 0.0

positive_flags = ['flag_offensive', 'flag_toxic', 'flag_neg_sentiment', 'flag_irony',
                  'flag_ragebait', 'flag_tribal', 'flag_performative', 'flag_genuine_anger']
ct['ti_v2'] = ct[positive_flags].mean(axis=1) * (1 - 0.3 * ct['flag_advocacy'])
ct['ti_v2'] = ct['ti_v2'].clip(0, 1)

ti = ct['ti_v2']
print(f'TI v2 - Mean: {ti.mean():.4f}, Median: {ti.median():.4f}')
print(f'  Clean (TI<0.1): {(ti<0.1).mean():.1%}')
print(f'  High (TI>=0.5): {(ti>=0.5).mean():.1%}')

# Per-member summary
print('\nComputing per-member TI v2...')
member_results = []
for sn in ct['screen_name'].unique():
    sub = ct[ct['screen_name'] == sn]
    if len(sub) < 20:
        continue
    member_results.append({
        'screen_name': sn,
        'n_tweets': len(sub),
        'ti_v2_mean': round(sub['ti_v2'].mean(), 4),
        'pct_clean': round((sub['ti_v2'] < 0.1).mean(), 4),
        'pct_high': round((sub['ti_v2'] >= 0.5).mean(), 4),
        'advocacy_rate': round(sub['flag_advocacy'].mean(), 4),
        'genuine_anger_rate': round(sub['flag_genuine_anger'].mean(), 4),
        **{f'{c}_rate': round(sub[c].mean(), 4) for c in positive_flags},
    })
members = pd.DataFrame(member_results).sort_values('ti_v2_mean', ascending=False)
members.to_csv(f'{ti_dir}/all_members_ti_v2.csv', index=False)
print(f'Scored {len(members):,} members')

print(f'\nTop 10:')
for _, r in members.head(10).iterrows():
    print(f'  {r.screen_name:25s} TI={r.ti_v2_mean:.3f} clean={r.pct_clean:.0%} adv={r.advocacy_rate:.0%} ({r.n_tweets:,} tweets)')

# Save per-tweet parquet (THE FIX: no duplicate columns)
print('\nSaving per-tweet parquet...')
save_cols = ['screen_name', 'created_at', 'text', 'ti_v2', 'flag_advocacy'] + positive_flags
ct[save_cols].to_parquet(f'{ti_dir}/congress_ti_v2.parquet', index=False)
print(f'Saved: {ti_dir}/congress_ti_v2.parquet')

# Overlap analysis
print('\n' + '=' * 70)
print('OVERLAP ANALYSIS')
print('=' * 70)

flag_matrix = ct[positive_flags].values
n_flags_per_tweet = flag_matrix.sum(axis=1)
print(f'\nFlags per tweet distribution:')
for k in range(9):
    pct = (n_flags_per_tweet == k).mean()
    if pct > 0.001:
        print(f'  {k} flags: {pct:.1%}')

print(f'\n  Mean flags/tweet: {n_flags_per_tweet.mean():.2f}')
print(f'  Zero flags (clean): {(n_flags_per_tweet == 0).mean():.1%}')
print(f'  3+ flags: {(n_flags_per_tweet >= 3).mean():.1%}')

# Co-occurrence matrix
print('\nCo-occurrence matrix (P(col | row)):')
flag_names_short = [f.replace('flag_', '') for f in positive_flags]
header = f'{"":>16s} ' + ' '.join(f'{f:>12s}' for f in flag_names_short)
print(header)
for i, fi in enumerate(positive_flags):
    row_mask = ct[fi] == 1
    if row_mask.sum() == 0:
        continue
    vals = []
    for j, fj in enumerate(positive_flags):
        if i == j:
            vals.append(f'{"---":>12s}')
        else:
            p = ct.loc[row_mask, fj].mean()
            vals.append(f'{p:>12.3f}')
    print(f'{flag_names_short[i]:>16s} ' + ' '.join(vals))

# Save overlap stats
overlap_stats = {
    'mean_flags_per_tweet': float(n_flags_per_tweet.mean()),
    'pct_clean': float((n_flags_per_tweet == 0).mean()),
    'pct_3plus': float((n_flags_per_tweet >= 3).mean()),
    'distribution': {str(k): float((n_flags_per_tweet == k).mean()) for k in range(9) if (n_flags_per_tweet == k).mean() > 0.0001},
}
import json
with open(f'{ti_dir}/overlap_stats.json', 'w') as f:
    json.dump(overlap_stats, f, indent=2)

print(f'\nSaved overlap stats: {ti_dir}/overlap_stats.json')
print('\nDONE')
