"""
Run all 12 classification heads on 1.68M congressional tweets using GPU 0 (RTX 4090).
Then compute per-tweet Toxicity Index and per-member TI scores.
"""
import os, sys, time, json, gc
import pandas as pd
import numpy as np
import torch

sys.path.insert(0, '/home/joneill/x_box/pipeline')

DEVICE = 'cuda:0'  # RTX 4090
out_dir = '/home/joneill/x_box/pipeline/output/toxicity_index'
os.makedirs(out_dir, exist_ok=True)

print(f'Device: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB')

# ── Load data ──
t0 = time.time()
df = pd.read_parquet('/home/joneill/x_box/pipeline/data/congresstweets_2017_2023.parquet')
df['created_at'] = pd.to_datetime(df['created_at'], utc=True, errors='coerce')
df = df.dropna(subset=['created_at'])

# Need text column
text_col = None
for c in ['text', 'full_text', 'content']:
    if c in df.columns:
        text_col = c
        break
if text_col is None:
    print(f'Columns: {list(df.columns)}')
    raise ValueError('No text column found')

df['text'] = df[text_col].astype(str)
texts = df['text'].tolist()
n = len(texts)
print(f'Loaded {n:,} tweets in {time.time()-t0:.0f}s')

# ── Preprocess for CardiffNLP ──
def preprocess(text):
    tokens = text.split()
    out = []
    for t in tokens:
        if t.startswith('@') and len(t) > 1:
            out.append('@user')
        elif t.startswith('http'):
            out.append('http')
        else:
            out.append(t)
    return ' '.join(out)

texts_processed = [preprocess(t) for t in texts]
print('Preprocessed texts')

# ============================================================
# PART 1: CardiffNLP (6 heads)
# ============================================================
from transformers import AutoModelForSequenceClassification, AutoTokenizer

cardiff_models = {
    'sentiment': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
    'emotion': 'cardiffnlp/twitter-roberta-base-emotion',
    'offensive': 'cardiffnlp/twitter-roberta-base-offensive',
    'irony': 'cardiffnlp/twitter-roberta-base-irony',
    'hate': 'cardiffnlp/twitter-roberta-base-hate-multiclass-latest',
    'toxicity': 's-nlp/roberta_toxicity_classifier',
}

BATCH_SIZE = 128

for name, model_id in cardiff_models.items():
    t1 = time.time()
    print(f'\n=== {name}: {model_id} ===')

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id).to(DEVICE).eval()
    id2label = model.config.id2label if hasattr(model.config, 'id2label') else {}

    all_labels = []
    all_scores = []

    for i in range(0, n, BATCH_SIZE):
        batch = texts_processed[i:i+BATCH_SIZE]
        encoded = tokenizer(batch, padding=True, truncation=True, max_length=128,
                           return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        for row in probs:
            label_idx = row.argmax()
            label = id2label.get(label_idx, str(label_idx))
            all_labels.append(label)
            all_scores.append(float(row.max()))

        if (i // BATCH_SIZE) % 100 == 0 and i > 0:
            rate = i / (time.time() - t1)
            eta = (n - i) / rate / 60
            print(f'  {i:>9,}/{n:,} ({i/n:.0%}) {rate:.0f} tweets/s ETA {eta:.1f}m')

    df[f'{name}_label'] = all_labels
    df[f'{name}_score'] = all_scores

    elapsed = time.time() - t1
    print(f'  Done: {n:,} tweets in {elapsed:.0f}s ({n/elapsed:.0f}/s)')

    # Free GPU memory
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

# Extract specific scores needed for TI
# Offensive
df['offensive_prob'] = 0.0
off_mask = df['offensive_label'].isin(['offensive', 'LABEL_1', '1'])
df.loc[off_mask, 'offensive_prob'] = df.loc[off_mask, 'offensive_score']

# Toxicity
df['toxicity_prob'] = 0.0
tox_mask = df['toxicity_label'].isin(['toxic', 'LABEL_1', '1'])
df.loc[tox_mask, 'toxicity_prob'] = df.loc[tox_mask, 'toxicity_score']

# Irony
df['irony_prob'] = 0.0
iro_mask = df['irony_label'].isin(['irony', 'LABEL_1', '1'])
df.loc[iro_mask, 'irony_prob'] = df.loc[iro_mask, 'irony_score']

print('\nCardiffNLP complete. Starting SetFit...')

# ============================================================
# PART 2: SetFit (6 heads)
# ============================================================
from setfit import SetFitModel

setfit_models = {
    'ragebait': '/home/joneill/x_box/pipeline/models/setfit_ragebait',
    'epistemic': '/home/joneill/x_box/pipeline/models/setfit_epistemic_manipulation',
    'tribal': '/home/joneill/x_box/pipeline/models_v2/setfit_tribal_signal',
    'performative': '/home/joneill/x_box/pipeline/models_v2/setfit_performative_outrage',
    'engbait': '/home/joneill/x_box/pipeline/models_v2/setfit_engagement_bait',
    'agency': '/home/joneill/x_box/pipeline/models_v2/setfit_agency_language',
}

for name, path in setfit_models.items():
    t1 = time.time()
    print(f'\n=== SetFit {name} ===')

    model = SetFitModel.from_pretrained(path).to(DEVICE)

    # SetFit predict in batches
    SETFIT_BATCH = 512
    all_preds = []
    for i in range(0, n, SETFIT_BATCH):
        batch = texts[i:i+SETFIT_BATCH]
        preds = model.predict(batch)
        all_preds.extend(preds.tolist() if hasattr(preds, 'tolist') else list(preds))

        if (i // SETFIT_BATCH) % 50 == 0 and i > 0:
            rate = i / (time.time() - t1)
            eta = (n - i) / rate / 60
            print(f'  {i:>9,}/{n:,} ({i/n:.0%}) {rate:.0f}/s ETA {eta:.1f}m')

    df[f'setfit_{name}'] = all_preds
    elapsed = time.time() - t1
    rate = sum(all_preds) / len(all_preds)
    print(f'  Done: {rate:.1%} positive, {elapsed:.0f}s ({n/elapsed:.0f}/s)')

    del model
    torch.cuda.empty_cache()
    gc.collect()

# ============================================================
# PART 3: Compute Toxicity Index
# ============================================================
print('\n=== Computing Toxicity Index ===')

# Build flags
df['flag_offensive'] = (df['offensive_prob'] > 0.5).astype(float)
df['flag_toxic'] = (df['toxicity_prob'] > 0.5).astype(float)
df['flag_neg_sentiment'] = (df['sentiment_label'].isin(['negative', 'LABEL_0'])).astype(float)
df['flag_anger'] = (df['emotion_label'].isin(['anger', 'LABEL_0'])).astype(float)
df['flag_irony'] = (df['irony_prob'] > 0.5).astype(float)
df['flag_ragebait'] = df['setfit_ragebait'].astype(float)
df['flag_tribal'] = df['setfit_tribal'].astype(float)
df['flag_performative'] = df['setfit_performative'].astype(float)

flag_cols = [c for c in df.columns if c.startswith('flag_')]
df['toxicity_index'] = df[flag_cols].mean(axis=1)

print(f'TI distribution (all Congress):')
ti = df['toxicity_index']
print(f'  Mean:   {ti.mean():.4f}')
print(f'  Median: {ti.median():.4f}')
print(f'  Std:    {ti.std():.4f}')
print(f'  Clean (TI<0.1): {(ti<0.1).mean():.1%}')
print(f'  High (TI>=0.5): {(ti>=0.5).mean():.1%}')

# ============================================================
# PART 4: Per-member TI scores
# ============================================================
print('\n=== Per-member Toxicity Index ===')

member_results = []
for sn in df['screen_name'].unique():
    sub = df[df['screen_name'] == sn]
    if len(sub) < 20:
        continue
    ti_vals = sub['toxicity_index']
    flags = sub[flag_cols]

    member_results.append({
        'screen_name': sn,
        'n_tweets': len(sub),
        'ti_mean': round(ti_vals.mean(), 4),
        'ti_median': round(ti_vals.median(), 4),
        'ti_std': round(ti_vals.std(), 4),
        'pct_clean': round((ti_vals < 0.1).mean(), 4),
        'pct_high': round((ti_vals >= 0.5).mean(), 4),
        **{f'{c}_rate': round(flags[c].mean(), 4) for c in flag_cols},
    })

members = pd.DataFrame(member_results).sort_values('ti_mean', ascending=False)
members.to_csv(f'{out_dir}/all_members_toxicity_index.csv', index=False)

print(f'Scored {len(members):,} members')
print(f'\nTop 15 by TI:')
for _, r in members.head(15).iterrows():
    print(f'  {r.screen_name:25s} TI={r.ti_mean:.3f} clean={r.pct_clean:.0%} ({r.n_tweets:,} tweets)')

print(f'\nBottom 15 by TI:')
for _, r in members.tail(15).iterrows():
    print(f'  {r.screen_name:25s} TI={r.ti_mean:.3f} clean={r.pct_clean:.0%} ({r.n_tweets:,} tweets)')

# ============================================================
# PART 5: Save classified data
# ============================================================
print('\n=== Saving ===')
# Save per-tweet (large file)
save_cols = ['screen_name', 'created_at', 'text', 'toxicity_index'] + flag_cols + \
            [c for c in df.columns if c.endswith('_label') or c.startswith('setfit_')]
for col in save_cols:
    if col in df.columns and df[col].dtype == object:
        df[col] = df[col].astype(str)

df[save_cols].to_parquet(f'{out_dir}/congress_classified_2017_2023.parquet', index=False)
sz = os.path.getsize(f'{out_dir}/congress_classified_2017_2023.parquet') / 1e6
print(f'Saved classified tweets: {sz:.0f} MB')

total_time = time.time() - t0
print(f'\nTotal time: {total_time:.0f}s ({total_time/60:.1f}m)')
