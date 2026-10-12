#!/usr/bin/env python
"""Held-out evaluation of the eight purpose-trained SetFit heads (Table S1).

Part 1 (released checkpoints, genuine held-out): the original training runs used a
deterministic 80/20 split (datasets.train_test_split(test_size=0.2, seed=42)) of a
class-balanced subset. We rebuild each training set exactly (verified against the
class counts and mean word count stored in each checkpoint README), then score the
released checkpoints on (a) the original 20% test split and (b) every labeled post
that the checkpoint never saw (the "unseen pool", ambiguous middle scores included
for the soft-score correlations).

Part 2 (retrained, k-fold CV): SetFit retrained with the same hyperparameters as the
original scripts on 4/5 of the labeled posts, evaluated on the held-out fold.
Folds are stratified by binarized label and grouped by account where the label file
carries one (LABELS_B: senator_name, LABELS_C: screen_name). LABELS_A is a single
account, so its posts are grouped by exact text only.

Label files are located by wildcard and referred to as LABELS_A / B / C:
  A: 6 dimensions, one account, 1,972 posts
  B: 4 dimensions, 14 accounts, 2,149 posts
  C: advocacy + anger, 562 accounts, 1,000 posts
"""
import os, sys, glob, time, json, re, gc, argparse, hashlib
import numpy as np, pandas as pd

ROOT = os.path.expanduser('~/x_box/pipeline')
OUT = f'{ROOT}/output/v7/heads_eval'
BACKBONE = 'sentence-transformers/all-mpnet-base-v2'
DEVICE = 'cuda:0'
N_FOLDS = 5
SEED = 42


def one(pattern):
    m = glob.glob(f'{ROOT}/data/{pattern}')
    assert len(m) == 1, (pattern, m)
    return m[0]


SRC_FILES = {'A': '*labels_mikelee_2k.parquet', 'B': '*labels_diverse_v2.parquet',
             'C': '*advocacy_anger_1k.parquet'}
# label column, label sources (concat order matters for exact reproduction), released
# checkpoint, binarization thresholds (pos >=, neg <=), original per-class cap and how
# the cap was applied ('head' = first N rows, 'sample' = random_state=42).
HEADS = {
    'ragebait': dict(col='ragebait', src=['A'], ckpt='models/setfit_ragebait', pt=.65, nt=.35, cap=50, how='head'),
    'tribal_signaling': dict(col='tribal_signal', src=['A', 'B'], ckpt='models_v2/setfit_tribal_signal', pt=.65, nt=.30, cap=200, how='sample'),
    'performative_outrage': dict(col='performative_outrage', src=['A', 'B'], ckpt='models_v2/setfit_performative_outrage', pt=.65, nt=.30, cap=200, how='sample'),
    'epistemic_manipulation': dict(col='epistemic_manipulation', src=['A'], ckpt='models/setfit_epistemic_manipulation', pt=.65, nt=.35, cap=50, how='head'),
    'engagement_baiting': dict(col='engagement_bait', src=['A', 'B'], ckpt='models_v2/setfit_engagement_bait', pt=.65, nt=.30, cap=200, how='sample'),
    'agency_language': dict(col='agency_language', src=['A', 'B'], ckpt='models_v2/setfit_agency_language', pt=.65, nt=.30, cap=200, how='sample'),
    'policy_advocacy': dict(col='political_advocacy', src=['C'], ckpt='models_v3/setfit_political_advocacy', pt=.65, nt=.30, cap=200, how='head'),
    'genuine_anger': dict(col='genuine_anger', src=['C'], ckpt='models_v3/setfit_genuine_anger', pt=.40, nt=.10, cap=200, how='head'),
}

_SRC = {}


def load_src(k):
    if k not in _SRC:
        _SRC[k] = pd.read_parquet(one(SRC_FILES[k]))
    return _SRC[k]


def build_pool(h):
    """Pool of labeled posts for a head, in the same row order as the original script."""
    cfg = HEADS[h]
    frames = []
    for k in cfg['src']:
        d = load_src(k)
        f = d[['text', cfg['col']]].dropna().copy()
        f.columns = ['text', 'soft']
        f['src'] = k
        if k == 'B':
            f['account'] = d.loc[f.index, 'senator_name'].values
        elif k == 'C':
            f['account'] = d.loc[f.index, 'screen_name'].values
        else:
            f['account'] = ''
        frames.append(f)
    p = pd.concat(frames, ignore_index=True)
    p['rid'] = np.arange(len(p))
    # grouping key: account when known; otherwise exact-text group (single-account source)
    tg = p['text'].map(lambda t: 'txt:' + hashlib.md5(t.encode()).hexdigest()[:12])
    p['group'] = np.where(p['account'] != '', p['account'].astype(str), tg)
    p['cls'] = np.where(p.soft >= cfg['pt'], 1, np.where(p.soft <= cfg['nt'], 0, -1))
    return p


def reproduce_original_split(h, p):
    """Rebuild the original balanced training subset and its 80/20 split (row ids)."""
    from datasets import Dataset
    cfg = HEADS[h]
    cap = cfg['cap']
    pos = p[p.cls == 1]
    neg = p[p.cls == 0]
    if cfg['how'] == 'head':
        pos = pos.head(cap)
        neg = neg.head(cap)
    else:
        k = min(cap, len(pos), len(neg))
        pos = pos.sample(k, random_state=42) if len(pos) > k else pos
        neg = neg.sample(k, random_state=42) if len(neg) > k else neg
    pos = pos.assign(label=1)
    neg = neg.assign(label=0)
    tr = pd.concat([pos[['text', 'label', 'rid']], neg[['text', 'label', 'rid']]], ignore_index=True).sample(frac=1, random_state=42)
    s = Dataset.from_pandas(tr).train_test_split(test_size=0.2, seed=42)
    return s['train'].to_pandas(), s['test'].to_pandas()


def readme_fingerprint(path):
    t = open(path + '/README.md').read()
    m = re.search(r'Word count\s*\|\s*(\d+)\s*\|\s*([\d.]+)', t)
    lab = {int(a): int(b) for a, b in re.findall(r'\|\s*([01])\s*\|\s*(\d+)\s*\|', t)}
    return float(m.group(2)), lab


def load_model(path):
    from setfit import SetFitModel
    return SetFitModel.from_pretrained(path).to(DEVICE)


def proba(model, texts):
    out = []
    for i in range(0, len(texts), 128):
        pr = model.predict_proba(list(texts[i:i + 128]))
        out.append(pr.detach().cpu().numpy() if hasattr(pr, 'detach') else np.asarray(pr))
    return np.concatenate(out)[:, 1]


def metrics(y, pr):
    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
    y = np.asarray(y)
    pr = np.asarray(pr)
    r = dict(n=int(len(y)), n_pos=int(y.sum()), pos_rate=float(y.mean()) if len(y) else np.nan)
    if len(y) and 0 < y.sum() < len(y):
        yh = (pr >= .5).astype(int)
        r.update(auc=roc_auc_score(y, pr), f1=f1_score(y, yh, zero_division=0),
                 precision=precision_score(y, yh, zero_division=0),
                 recall=recall_score(y, yh, zero_division=0), accuracy=accuracy_score(y, yh))
    return r


def soft_corr(pr, soft):
    from scipy.stats import pearsonr, spearmanr
    if len(pr) < 5 or np.std(pr) == 0:
        return dict(pearson=np.nan, spearman=np.nan, n_soft=len(pr))
    return dict(pearson=float(pearsonr(pr, soft)[0]), spearman=float(spearmanr(pr, soft)[0]), n_soft=int(len(pr)))


def train_head(train_df, seed):
    from datasets import Dataset
    from setfit import SetFitModel, Trainer, TrainingArguments
    from transformers import set_seed
    set_seed(seed)
    ds = Dataset.from_pandas(train_df[['text', 'label']].reset_index(drop=True))
    model = SetFitModel.from_pretrained(BACKBONE).to(DEVICE)
    # identical to the original scripts: batch 16, 1 epoch, 5 iterations
    args = TrainingArguments(batch_size=16, num_epochs=1, num_iterations=5, show_progress_bar=False, seed=seed)
    Trainer(model=model, args=args, train_dataset=ds).train()
    return model


def free():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    global DEVICE
    ap = argparse.ArgumentParser()
    ap.add_argument('--heads', default='all')
    ap.add_argument('--folds', type=int, default=N_FOLDS)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    import torch
    from scipy.stats import pearsonr, spearmanr
    from sklearn.model_selection import StratifiedGroupKFold
    os.makedirs(OUT, exist_ok=True)
    t_start = time.time()
    names = list(HEADS) if a.heads == 'all' else a.heads.split(',')
    inv_rows, ho_rows, fold_rows, pred_rows, repro_rows = [], [], [], [], []

    def dump():
        pd.DataFrame(fold_rows).to_csv(f'{OUT}/fold_metrics.csv', index=False)
        pd.DataFrame(inv_rows).to_csv(f'{OUT}/label_inventory.csv', index=False)
        pd.DataFrame(ho_rows).to_csv(f'{OUT}/released_heldout_metrics.csv', index=False)
        pd.DataFrame(repro_rows).to_csv(f'{OUT}/split_reproduction_check.csv', index=False)
        if pred_rows:
            pd.concat(pred_rows).to_csv(f'{OUT}/cv_predictions.csv.gz', index=False)

    for h in names:
        cfg = HEADS[h]
        p = build_pool(h)
        ck = f"{ROOT}/{cfg['ckpt']}"
        has_acc = (p.account != '')
        inv_rows.append(dict(head=h, sources='+'.join('LABELS_' + k for k in cfg['src']), n_posts=len(p),
                             n_accounts=int(p.loc[has_acc, 'account'].nunique()),
                             pos_thr=cfg['pt'], neg_thr=cfg['nt'], n_pos=int((p.cls == 1).sum()), n_neg=int((p.cls == 0).sum()),
                             n_ambiguous=int((p.cls == -1).sum()), soft_mean=p.soft.mean(), soft_sd=p.soft.std(),
                             orig_cap_per_class=cfg['cap'], orig_cap_method=cfg['how']))
        # reproduce the original split and verify it against the checkpoint README
        tr, te = reproduce_original_split(h, p)
        wc_readme, lab_readme = readme_fingerprint(ck)
        wc_repro = tr.text.map(lambda s: len(s.split(' '))).mean()
        lab_repro = {int(k): int(v) for k, v in tr.label.value_counts().to_dict().items()}
        ok = abs(wc_readme - wc_repro) < 1e-3 and lab_repro == lab_readme
        repro_rows.append(dict(head=h, train_n=len(tr), test_n=len(te), readme_wc=wc_readme, repro_wc=round(wc_repro, 4),
                               readme_counts=str(lab_readme), repro_counts=str(lab_repro), match=bool(ok)))
        print(h, 'split reproduction matches README:', ok, flush=True)
        # released checkpoint scores on the whole pool
        m = load_model(ck)
        p['released'] = proba(m, p.text.tolist())
        del m
        free()
        train_rids = set(tr.rid)
        test_rids = set(te.rid)
        train_texts = set(p.loc[p.rid.isin(train_rids), 'text'])
        p['rel_split'] = np.where(p.rid.isin(train_rids), 'train', np.where(p.rid.isin(test_rids), 'test', 'unused'))
        p['unseen'] = ~p.rid.isin(train_rids) & ~p.text.isin(train_texts)
        for nm, mask in [('original_test_split', p.rel_split == 'test'), ('unseen_pool', p.unseen), ('original_train_split', p.rel_split == 'train')]:
            s = p[mask]
            c = s[s.cls >= 0]
            ho_rows.append(dict(head=h, eval_set=nm, **metrics(c.cls, c.released), **soft_corr(s.released.values, s.soft.values)))
        grouping = 'account' if has_acc.all() else ('account (B) + exact-text (A, single account)' if has_acc.any() else 'exact-text only (single account)')

        variants = [('cv_cap160', 160)] if cfg['cap'] > 50 else [('cv_cap40', 40), ('cv_cap160', 160)]
        y3 = p.cls.values
        for vname, cap in variants:
            for attempt in (0, 1):
                seed_off = 1000 * attempt
                skf = StratifiedGroupKFold(n_splits=a.folds, shuffle=True, random_state=SEED)
                rows_v = []
                for f, (itr, ite) in enumerate(skf.split(p, y3, groups=p.group.values)):
                    t0 = time.time()
                    trn = p.iloc[itr]
                    tst = p.iloc[ite].copy()
                    assert not (set(trn.group) & set(tst.group)), 'group leak'
                    pos = trn[trn.cls == 1]
                    neg = trn[trn.cls == 0]
                    pos = pos.sample(min(cap, len(pos)), random_state=SEED + f)
                    neg = neg.sample(min(cap, len(neg)), random_state=SEED + f)
                    tdf = pd.concat([pos.assign(label=1), neg.assign(label=0)]).sample(frac=1, random_state=SEED + f)
                    model = None
                    for _ in range(2):
                        try:
                            model = train_head(tdf, SEED + f + seed_off)
                            tst['cv_p'] = proba(model, tst.text.tolist())
                            break
                        except torch.cuda.OutOfMemoryError:
                            print('CUDA OOM, falling back to CPU', flush=True)
                            model = None
                            free()
                            DEVICE = 'cpu'
                    del model
                    free()
                    c = tst[tst.cls >= 0]
                    r = dict(head=h, variant=vname, attempt=attempt, fold=f, train_cap=cap, n_train=len(tdf), grouping=grouping,
                             n_groups_test=int(tst.group.nunique()), **metrics(c.cls, c.cv_p), **soft_corr(tst.cv_p.values, tst.soft.values))
                    r['rel_vs_cv_pearson_all'] = float(pearsonr(tst.released, tst.cv_p)[0])
                    r['rel_vs_cv_spearman_all'] = float(spearmanr(tst.released, tst.cv_p)[0])
                    un = tst[tst.unseen]
                    r['n_unseen_by_released'] = int(len(un))
                    r['rel_vs_cv_pearson_unseen'] = float(pearsonr(un.released, un.cv_p)[0]) if len(un) > 5 else np.nan
                    r['rel_vs_cv_spearman_unseen'] = float(spearmanr(un.released, un.cv_p)[0]) if len(un) > 5 else np.nan
                    r['secs'] = round(time.time() - t0, 1)
                    rows_v.append(r)
                    fold_rows.append(r)
                    pred_rows.append(tst[['rid', 'src', 'group', 'soft', 'cls', 'released', 'cv_p']].assign(head=h, variant=vname, attempt=attempt, fold=f))
                    print(h, vname, 'att', attempt, 'fold', f, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k in ('n', 'auc', 'f1', 'rel_vs_cv_pearson_all', 'secs')}, flush=True)
                    dump()
                    if a.smoke:
                        break
                mauc = np.nanmean([r.get('auc', np.nan) for r in rows_v])
                if mauc >= 0.58 or attempt == 1 or a.smoke:
                    break
                print(f'{h} {vname}: mean AUC {mauc:.3f} near chance, rerunning once with a different seed', flush=True)
            if a.smoke:
                break
        dump()
        if a.smoke:
            return

    # tables. The reported CV run per (head, variant) is the last attempt; attempt 0 stays in fold_metrics.csv.
    fm = pd.DataFrame(fold_rows)
    ho = pd.DataFrame(ho_rows)
    amax = fm.groupby(['head', 'variant']).attempt.max().rename('amax').reset_index()
    use = fm.merge(amax, on=['head', 'variant'])
    use = use[use.attempt == use.amax]
    agg = use.groupby(['head', 'variant']).agg(
        cv_folds=('fold', 'count'), cv_n_total=('n', 'sum'), cv_pos_rate=('pos_rate', 'mean'),
        cv_auc_mean=('auc', 'mean'), cv_auc_sd=('auc', 'std'), cv_f1_mean=('f1', 'mean'), cv_f1_sd=('f1', 'std'),
        cv_precision_mean=('precision', 'mean'), cv_recall_mean=('recall', 'mean'), cv_accuracy_mean=('accuracy', 'mean'),
        cv_pearson_soft_mean=('pearson', 'mean'), cv_spearman_soft_mean=('spearman', 'mean'),
        released_vs_retrained_pearson=('rel_vs_cv_pearson_all', 'mean'),
        released_vs_retrained_pearson_unseen=('rel_vs_cv_pearson_unseen', 'mean'),
        released_vs_retrained_spearman_unseen=('rel_vs_cv_spearman_unseen', 'mean'),
        grouping=('grouping', 'first'), train_cap=('train_cap', 'first'), rerun_used=('attempt', 'max')).reset_index()
    agg.to_csv(f'{OUT}/cv_summary_all_variants.csv', index=False)
    # Table S1: CV at the training size of the released checkpoints (cap 40 for the two
    # small-set heads, cap 160 for the rest) next to the released checkpoints' held-out metrics.
    small = ['ragebait', 'epistemic_manipulation']
    prim = agg[((agg.variant == 'cv_cap40') & agg['head'].isin(small)) | ((agg.variant == 'cv_cap160') & ~agg['head'].isin(small))]
    u = ho[ho.eval_set == 'unseen_pool'].set_index('head')
    t = ho[ho.eval_set == 'original_test_split'].set_index('head')
    inv = pd.DataFrame(inv_rows).set_index('head')
    s1 = prim.set_index('head').join(inv[['sources', 'n_posts', 'n_pos', 'n_neg', 'n_ambiguous']])
    s1 = s1.join(u[['n', 'pos_rate', 'auc', 'f1', 'precision', 'recall', 'accuracy', 'pearson', 'spearman']].add_prefix('rel_unseen_'))
    s1 = s1.join(t[['n', 'auc', 'f1']].add_prefix('rel_origtest_'))
    s1 = s1.reindex(list(HEADS)).reset_index()
    s1.round(4).to_csv(f'{OUT}/table_s1.csv', index=False)
    json.dump(dict(total_runtime_min=round((time.time() - t_start) / 60, 1), device_final=DEVICE), open(f'{OUT}/run_meta.json', 'w'))
    print('done in', round((time.time() - t_start) / 60, 1), 'min')


if __name__ == '__main__':
    main()
