"""
t6_score_official.py: T6 scoring of @SenMikeLee (3,578 posts) with the same checkpoints and code path as ti_classify_probs.py, on CPU.
Outputs to output/v7/t6/_probs_official/. (derived from ti_classify_probs.py)

Re-runs the classification heads used for Toxicity Index v2 and stores PROBABILITIES per post.
Same code path and same checkpoints as scripts/classify_all_congress.py and scripts/retrain_and_rebuild_ti.py:
  CardiffNLP / s-nlp heads: AutoModelForSequenceClassification, softmax, tweet preprocessing (@user, http),
    max_length=128, fp32 (default), dynamic padding.
  SetFit heads: SetFitModel.from_pretrained(path); probability = predict_proba[:, 1] of the stored logistic head.
Row order = congresstweets parquet after dropping rows with unparseable created_at (identical to the v2 run).
Rows are processed sorted by text length (padding does not change outputs) and written back in original order.

Outputs (output/v7/ti/_probs/):  <head>.npy (float32, original order), <head>.json (checkpoint record).
Corpus: congress (default) or lee (data/BasedMikeLee_archive_v2.parquet, same checkpoints, written to _probs_lee/).
Heads: offensive, toxic, neg_sentiment, irony, ragebait, tribal, performative, genuine_anger, advocacy,
       plus emotion_anger (off-the-shelf, needed only to recompute the Table 4 off-the-shelf anger column).
For the 3/4-class heads the argmax indicator used by the v2 flags is stored as <head>_argmax.npy.

Usage: python ti_classify_probs.py --gpu 0 [--heads a,b] [--limit N]
Never stops or touches other processes. Keep VRAM under ~8 GB (batch sizes are conservative).
"""
import argparse, gc, hashlib, json, os, sys, time
import numpy as np
import pandas as pd
import torch

ROOT = os.path.expanduser('~/x_box/pipeline')
ap = argparse.ArgumentParser()
ap.add_argument('--gpu', type=int, default=0)
ap.add_argument('--heads', default='')
ap.add_argument('--limit', type=int, default=0)
ap.add_argument('--bs', type=int, default=128)
ap.add_argument('--corpus', default='congress', choices=['congress', 'lee', 'official'])
args = ap.parse_args()
DEVICE = 'cpu'
torch.set_num_threads(16)
OUT = f'{ROOT}/output/v7/t6/_probs_official'
os.makedirs(OUT, exist_ok=True)

CARDIFF = {  # head -> (hf id, positive label names used by v2 flag logic)
    'offensive': ('cardiffnlp/twitter-roberta-base-offensive', ['offensive', 'LABEL_1', '1']),
    'toxic': ('s-nlp/roberta_toxicity_classifier', ['toxic', 'LABEL_1', '1']),
    'neg_sentiment': ('cardiffnlp/twitter-roberta-base-sentiment-latest', ['negative', 'LABEL_0']),
    'irony': ('cardiffnlp/twitter-roberta-base-irony', ['irony', 'LABEL_1', '1']),
    'emotion_anger': ('cardiffnlp/twitter-roberta-base-emotion', ['anger', 'LABEL_0']),
    'hate': ('cardiffnlp/twitter-roberta-base-hate-multiclass-latest', None),  # prob = 1 - P(not-hate)
}
SETFIT = {
    'ragebait': f'{ROOT}/models/setfit_ragebait',
    'tribal': f'{ROOT}/models_v2/setfit_tribal_signal',
    'performative': f'{ROOT}/models_v2/setfit_performative_outrage',
    'genuine_anger': f'{ROOT}/models_v3/setfit_genuine_anger',
    'advocacy': f'{ROOT}/models_v3/setfit_political_advocacy',
    'epistemic': f'{ROOT}/models/setfit_epistemic_manipulation',
    'engbait': f'{ROOT}/models_v2/setfit_engagement_bait',
    'agency': f'{ROOT}/models_v2/setfit_agency_language',
}
heads = [h for h in args.heads.split(',') if h] or list(CARDIFF) + list(SETFIT)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def preprocess(text):
    out = []
    for t in text.split():
        if t.startswith('@') and len(t) > 1:
            out.append('@user')
        elif t.startswith('http'):
            out.append('http')
        else:
            out.append(t)
    return ' '.join(out)


if args.corpus == 'official':
    df = pd.read_parquet(f'{ROOT}/data/SenMikeLee_2022_2026.parquet', columns=['text', 'created_at'])
    df['created_at'] = pd.to_datetime(df['created_at'], utc=True, errors='coerce')
elif args.corpus == 'lee':
    df = pd.read_parquet(f'{ROOT}/data/BasedMikeLee_archive_v2.parquet', columns=['text', 'created_at'])
    df['created_at'] = pd.to_datetime(df['created_at'], utc=True, errors='coerce')
else:
    df = pd.read_parquet(f'{ROOT}/data/congresstweets_2017_2023.parquet', columns=['text', 'created_at'])
    df['created_at'] = pd.to_datetime(df['created_at'], utc=True, errors='coerce')
    df = df.dropna(subset=['created_at'])
texts = df['text'].astype(str).tolist()
del df
if args.limit:
    texts = texts[:args.limit]
n = len(texts)
order = np.argsort([len(t) for t in texts], kind='stable')
print(f'rows {n:,} device {DEVICE} ', flush=True)
sfx = f'_lim{args.limit}' if args.limit else ''

for head in heads:
    if os.path.exists(f'{OUT}/{head}{sfx}.npy'):
        print(head, 'exists, skip', flush=True)
        continue
    t0 = time.time()
    prob = np.zeros(n, dtype=np.float32)
    rec = {'head': head, 'n': n, 'device': DEVICE, 'torch': torch.__version__}
    if head in CARDIFF:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        mid, posnames = CARDIFF[head]
        tok = AutoTokenizer.from_pretrained(mid)
        model = AutoModelForSequenceClassification.from_pretrained(mid).to(DEVICE).eval()
        id2 = model.config.id2label
        if posnames is None:  # hate: positive = anything but not-hate
            nh = [i for i, l in id2.items() if l.lower().replace('_', '-') == 'not-hate']
            assert len(nh) == 1, id2
            pos_idx = nh[0]
        else:
            pos_idx = [i for i, l in id2.items() if l in posnames]
            assert len(pos_idx) == 1, (head, id2)
            pos_idx = pos_idx[0]
        rec.update(model_id=mid, commit=getattr(model.config, '_commit_hash', None), id2label={str(k): v for k, v in id2.items()},
                   positive_index=pos_idx, max_length=128, batch=args.bs)
        amax = np.zeros(n, dtype=bool)
        for k in range(0, n, args.bs):
            idx = order[k:k + args.bs]
            enc = tok([preprocess(texts[i]) for i in idx], padding=True, truncation=True, max_length=128,
                      return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                p = torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()
            if posnames is None:
                prob[idx] = 1.0 - p[:, pos_idx]
                amax[idx] = p.argmax(1) != pos_idx
            else:
                prob[idx] = p[:, pos_idx]
                amax[idx] = p.argmax(1) == pos_idx
            if (k // args.bs) % 1000 == 0 and k:
                print(f'  {head} {k:,}/{n:,} {k/(time.time()-t0):.0f}/s', flush=True)
        np.save(f'{OUT}/{head}_argmax{sfx}.npy', amax)
        del model, tok
    else:
        from setfit import SetFitModel
        path = SETFIT[head]
        model = SetFitModel.from_pretrained(path).to(DEVICE)
        rec.update(model_path=path.replace(ROOT + '/', ''), safetensors_sha256=sha(f'{path}/model.safetensors'),
                   head_sha256=sha(f'{path}/model_head.pkl'), head_class=type(model.model_head).__name__,
                   classes=[int(c) for c in model.model_head.classes_])
        assert list(model.model_head.classes_) == [0, 1]
        CH = 20000
        for k in range(0, n, CH):
            idx = order[k:k + CH]
            p = model.predict_proba([texts[i] for i in idx], batch_size=128, as_numpy=True, show_progress_bar=False)
            prob[idx] = p[:, 1]
            if (k // CH) % 10 == 0 and k:
                print(f'  {head} {k:,}/{n:,} {k/(time.time()-t0):.0f}/s', flush=True)
        del model
    np.save(f'{OUT}/{head}{sfx}.npy', prob)
    rec['seconds'] = round(time.time() - t0)
    rec['mean_prob'] = float(prob.mean())
    json.dump(rec, open(f'{OUT}/{head}{sfx}.json', 'w'), indent=1)
    print(f'DONE {head} mean_p={prob.mean():.4f} {rec["seconds"]}s', flush=True)
    gc.collect()
print('ALL DONE', flush=True)
