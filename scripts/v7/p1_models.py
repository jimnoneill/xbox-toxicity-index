"""P1 model definitions shared by p1_run.py, p1_final.py and p1_lee.py: feature sets, GBM tuning/fitting, mixed models,
baselines for one (train, test) split.  All fitting uses training rows only; labels and thresholds of the test rows are
never touched during fitting."""
import numpy as np
import pandas as pd
import lightgbm as lgb

from p1_common import *  # noqa

SETS = {
    "P": fp.PERSIST + fp.TE,
    "R": fp.PERSIST + fp.RHYTHM + fp.TE,
    "F": fp.PERSIST + fp.RHYTHM + fp.REGIME + fp.TE,
    # post-type variants: T = previous event's type, O = also the target's own first-post type
    "PT": fp.PERSIST + fp.TYPE_PREV + fp.TE,
    "RT": fp.PERSIST + fp.TYPE_PREV + fp.RHYTHM + fp.TE,
    "PTO": fp.PERSIST + fp.TYPE_PREV + fp.TYPE_OWN + fp.TE,
    "RTO": fp.PERSIST + fp.TYPE_PREV + fp.TYPE_OWN + fp.RHYTHM + fp.TE,
}
SETS_NOCONG = {
    "P": fp.PERSIST + fp.TE,
    "R": fp.PERSIST + fp.RHYTHM_NOCONG + fp.TE,
    "PT": fp.PERSIST + fp.TYPE_PREV + fp.TE,
    "RT": fp.PERSIST + fp.TYPE_PREV + fp.RHYTHM_NOCONG + fp.TE,
    "PTO": fp.PERSIST + fp.TYPE_PREV + fp.TYPE_OWN + fp.TE,
    "RTO": fp.PERSIST + fp.TYPE_PREV + fp.TYPE_OWN + fp.RHYTHM_NOCONG + fp.TE,
}
GBM_TASKS = [("reg", s) for s in ("P", "R", "F", "PT", "RT", "PTO", "RTO")] + \
            [("bina", s) for s in ("P", "R", "PT", "RT", "PTO", "RTO")] + [("binb", s) for s in ("P", "R", "PT", "RT")]
LMM_TASKS = [("reg", s) for s in ("P", "R", "PT", "RT", "PTO", "RTO")] + [("bina", "P"), ("bina", "R"), ("binb", "P"), ("binb", "R")]
GRID = [(15, 200), (15, 1000), (63, 200), (63, 1000)]
MAX_GBM_ROWS = 600_000


def lmm_cols(kind, nocong=False):
    rh = fp.RHYTHM_NOCONG if nocong else fp.RHYTHM
    cols = list(fp.PERSIST)
    if "T" in kind:
        cols += fp.TYPE_PREV
    if "O" in kind:
        cols += fp.TYPE_OWN
    if kind.startswith("R"):
        cols += [c for c in rh if c != "hour"]
    return cols


def add_labels(tr, te):
    """Binary labels from TRAINING-window thresholds: account 80th percentile (bina) and global 80th percentile (binb)."""
    thr_a = tr.groupby("acct").ti7.quantile(0.8)
    thr_g = float(tr.ti7.quantile(0.8))
    for d in (tr, te):
        d["thr_a"] = d.acct.map(thr_a).astype("float32")
        d["y_bina"] = (d.ti7 > d.thr_a).astype("float32")
        d["y_binb"] = (d.ti7 > thr_g).astype("float32")
    return thr_g


def add_te(tr, te, J):
    m, s = fp.target_encode(tr.acct.values, tr.ti7.values, J)
    tr["te_mean"], tr["te_sd"] = m.astype("float32"), s.astype("float32")
    m2, s2 = fp.target_encode(tr.acct.values, tr.ti7.values, J, test_g=te.acct.values)
    te["te_mean"], te["te_sd"] = m2.astype("float32"), s2.astype("float32")


def gbm_params(obj, leaves, min_leaf, seed=fp.SEED):
    return dict(objective="regression" if obj == "reg" else "binary", learning_rate=0.08, num_leaves=leaves,
                min_data_in_leaf=min_leaf, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0,
                num_threads=THREADS, seed=seed, verbose=-1, deterministic=True, force_row_wise=True,
                metric="l2" if obj == "reg" else "binary_logloss")


def ylab(task):
    return {"reg": "ti7", "bina": "y_bina", "binb": "y_binb"}[task]


def tune(tr, va, tasks=GBM_TASKS, sets=SETS):
    """Grid search on a validation block that lies entirely before every forecast origin."""
    best = {}
    for task, s in tasks:
        cols = sets[s]
        cur = None
        for leaves, ml in GRID:
            ds = lgb.Dataset(tr[cols], tr[ylab(task)], free_raw_data=False)
            dv = lgb.Dataset(va[cols], va[ylab(task)], reference=ds)
            ev = {}
            bst = lgb.train(gbm_params(task, leaves, ml), ds, 800, valid_sets=[dv],
                            callbacks=[lgb.early_stopping(40, verbose=False), lgb.record_evaluation(ev)])
            sc = min(next(iter(ev["valid_0"].values())))
            if cur is None or sc < cur[0]:
                cur = (sc, leaves, ml, int(bst.best_iteration))
        best[f"{task}_{s}"] = dict(score=float(cur[0]), leaves=cur[1], min_leaf=cur[2], rounds=cur[3])
    return best


def fit_gbm(tr, task, s, hp, sets=SETS, seed=fp.SEED):
    cols = sets[s]
    h = hp[f"{task}_{s}"]
    d = tr if len(tr) <= MAX_GBM_ROWS else tr.sample(MAX_GBM_ROWS, random_state=seed)
    ds = lgb.Dataset(d[cols], d[ylab(task)])
    return lgb.train(gbm_params(task, h["leaves"], h["min_leaf"], seed), ds, h["rounds"])


class LMM:
    """Standardized design + MixedLM for one (task, feature kind)."""

    def __init__(self, task, kind, nocong=False):
        self.task, self.kind = task, kind
        self.cols = lmm_cols(kind, nocong)
        sl = [1 + self.cols.index(c) for c in fp.LMM_SLOPES if c in self.cols] if kind.startswith("R") else []
        self.zidx = [0] + sl

    def design(self, d):
        Z = (d[self.cols].to_numpy("float64") - self.mu) / self.sd
        Z = np.nan_to_num(Z)
        return np.hstack([np.ones((len(d), 1)), Z])

    def fit(self, tr, code, J):
        A = tr[self.cols].to_numpy("float64")
        self.mu = np.nanmean(A, 0)
        self.sd = np.where(np.nanstd(A, 0) > 0, np.nanstd(A, 0), 1.0)
        fam = "gaussian" if self.task == "reg" else "binomial"
        self.m = fp.MixedLM(self.zidx, family=fam).fit(self.design(tr), tr[ylab(self.task)].to_numpy("float64"), code, J)
        return self

    def predict(self, te, code):
        return self.m.predict(self.design(te), code)


def run_split(tr, te, hp, J, sets=SETS, lmm_tasks=LMM_TASKS, gbm_tasks=GBM_TASKS, nocong=False, with_lmm=True):
    """Fit everything on tr, predict te.  tr/te must already carry labels and target encodings.
    Returns (dict name -> prediction array, info dict)."""
    out, info = {}, {}
    # baselines
    amean = tr.groupby("acct").ti7.mean()
    out["b_mean"] = te.acct.map(amean).fillna(tr.ti7.mean()).to_numpy()
    out["b_persist"] = te.prev_ti7.to_numpy()
    out["b_roll"] = te.roll50.to_numpy()
    out["b_global"] = np.full(len(te), tr.ti7.mean())
    for task in ("bina", "binb"):
        yl = ylab(task)
        out[f"b_rate_{task}"] = te.acct.map(tr.groupby("acct")[yl].mean()).fillna(tr[yl].mean()).to_numpy()
        thr = tr.thr_a if task == "bina" else pd.Series(tr.ti7.quantile(0.8), index=tr.index)
        prev_hi = (tr.prev_ti7 > thr).to_numpy()
        r1, r0 = tr[yl].to_numpy()[prev_hi].mean(), tr[yl].to_numpy()[~prev_hi].mean()
        thr_te = te.thr_a if task == "bina" else tr.ti7.quantile(0.8)
        out[f"b_persist_{task}"] = np.where((te.prev_ti7 > thr_te).to_numpy(), r1, r0)
    for task, s in gbm_tasks:
        bst = fit_gbm(tr, task, s, hp, sets)
        out[f"gbm_{task}_{s}"] = bst.predict(te[sets[s]])
    if with_lmm:
        code_tr, code_te = tr.ccode.to_numpy(), te.ccode.to_numpy()
        for task, s in lmm_tasks:
            l = LMM(task, s, nocong).fit(tr, code_tr, J)
            out[f"lmm_{task}_{s}"] = l.predict(te, code_te)
            info[f"lmm_{task}_{s}"] = dict(s2=float(l.m.s2), G_diag=np.diag(l.m.G).tolist(), beta=l.m.beta.tolist(),
                                          cols=l.cols)
    return out, info


def prepare_origin(F, o, end):
    """Training rows (< o) and test rows [o, end) for eligible accounts; labels, TE and compact account codes."""
    tr = F[F.when < o]
    te = F[(F.when >= o) & (F.when < end)]
    n = tr.groupby("acct").size()
    ok = n[n >= MIN_TRAIN_ROWS].index
    tr = tr[tr.acct.isin(ok)].copy()
    te = te[te.acct.isin(ok)].copy()
    cmap = {a: i for i, a in enumerate(sorted(ok))}
    tr["ccode"] = tr.acct.map(cmap).astype("int32")
    te["ccode"] = te.acct.map(cmap).astype("int32")
    J = len(cmap)
    thr_g = add_labels(tr, te)
    # target encoding uses compact codes
    a_tr, a_te = tr.acct.values, te.acct.values
    tr["acct"], te["acct"] = tr.ccode.values, te.ccode.values
    add_te(tr, te, J)
    tr["acct"], te["acct"] = a_tr, a_te
    return tr, te, J, thr_g
