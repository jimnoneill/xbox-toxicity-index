"""
stats_common.py  (spec E)  Shared, deterministic helpers.

hedges_g:  g = J * (m1 - m2) / sp,  sp = pooled SD (ddof=1),  J = 1 - 3 / (4 df - 1),  df = n1 + n2 - 2.
           Group 1 is the first argument; positive g means group 1 is higher.
           Check (spec): n1=195, n2=182 gives J = 0.9980.
boot_g:    percentile 95% CI by 10,000 bootstrap resamples of members, stratified by group (each group resampled
           within itself), seeded.
welch:     scipy ttest_ind(equal_var=False) with Welch-Satterthwaite df.
holm:      Holm step-down adjusted p-values (monotone).
Numbers:   collects key -> {value, description} and writes JSON.
"""
import json
import os
import numpy as np
from scipy import stats

SEED = 20261008
B_DEFAULT = 10000


def J_factor(n1, n2):
    df = n1 + n2 - 2
    return 1.0 - 3.0 / (4.0 * df - 1.0)


def hedges_g(x1, x2):
    x1 = np.asarray(x1, float); x2 = np.asarray(x2, float)
    n1, n2 = len(x1), len(x2)
    sp = np.sqrt(((n1 - 1) * x1.var(ddof=1) + (n2 - 1) * x2.var(ddof=1)) / (n1 + n2 - 2))
    return J_factor(n1, n2) * (x1.mean() - x2.mean()) / sp if sp > 0 else np.nan


def boot_g(X1, X2, B=B_DEFAULT, seed=SEED, chunk=500):
    """X1 (n1 x k), X2 (n2 x k). Returns (k,2) array of 2.5/97.5 percentiles of Hedges g, and (k,2) for mean diff."""
    X1 = np.atleast_2d(np.asarray(X1, float).T).T if np.ndim(X1) == 1 else np.asarray(X1, float)
    X2 = np.atleast_2d(np.asarray(X2, float).T).T if np.ndim(X2) == 1 else np.asarray(X2, float)
    rng = np.random.default_rng(seed)
    n1, n2, k = X1.shape[0], X2.shape[0], X1.shape[1]
    J = J_factor(n1, n2)
    gs = np.empty((B, k)); ds = np.empty((B, k))
    for s in range(0, B, chunk):
        b = min(chunk, B - s)
        i1 = rng.integers(0, n1, size=(b, n1)); i2 = rng.integers(0, n2, size=(b, n2))
        a = X1[i1]; c = X2[i2]  # (b, n, k)
        m1, m2 = a.mean(1), c.mean(1)
        v1, v2 = a.var(1, ddof=1), c.var(1, ddof=1)
        sp = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
        with np.errstate(divide='ignore', invalid='ignore'):
            gs[s:s + b] = J * (m1 - m2) / sp
        ds[s:s + b] = m1 - m2
    return np.nanpercentile(gs, [2.5, 97.5], axis=0).T, np.percentile(ds, [2.5, 97.5], axis=0).T


def welch(x1, x2):
    t, p = stats.ttest_ind(x1, x2, equal_var=False)
    v1, v2 = np.var(x1, ddof=1) / len(x1), np.var(x2, ddof=1) / len(x2)
    df = (v1 + v2) ** 2 / (v1 ** 2 / (len(x1) - 1) + v2 ** 2 / (len(x2) - 1))
    return float(t), float(df), float(p)


def holm(pvals):
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def contrast_table(df, cols, group_col='party', g1='R', g2='D', B=B_DEFAULT, seed=SEED, holm_family=None):
    """Member-level contrast of columns between two groups. Returns list of dict rows."""
    a = df[df[group_col] == g1]; b = df[df[group_col] == g2]
    ci_g, ci_d = boot_g(a[cols].values, b[cols].values, B=B, seed=seed)
    rows = []
    for j, c in enumerate(cols):
        x1, x2 = a[c].values, b[c].values
        t, dfw, p = welch(x1, x2)
        rows.append(dict(measure=c, n1=len(x1), n2=len(x2), mean1=x1.mean(), mean2=x2.mean(), sd1=x1.std(ddof=1),
                         sd2=x2.std(ddof=1), diff=x1.mean() - x2.mean(), diff_lo=ci_d[j, 0], diff_hi=ci_d[j, 1],
                         welch_t=t, welch_df=dfw, p=p, hedges_g=hedges_g(x1, x2), g_lo=ci_g[j, 0], g_hi=ci_g[j, 1]))
    fam = holm_family if holm_family is not None else list(range(len(rows)))
    ph = holm([rows[i]['p'] for i in fam])
    for r in rows:
        r['p_holm'] = np.nan
    for k, i in enumerate(fam):
        rows[i]['p_holm'] = ph[k]
    return rows


class Numbers:
    """Collects key -> {value, description}. suffix is appended to every key (used for the all-posts basis).
    save(path, superseded_from=file) embeds the earlier version of the same file under the key "superseded"
    (only once: an existing superseded block in that file is carried over, never nested)."""
    def __init__(self, suffix=''):
        self.d = {}
        self.suffix = suffix

    def add(self, key, value, desc):
        key = key + self.suffix
        if isinstance(value, (np.floating,)):
            value = float(value)
        elif isinstance(value, (np.integer,)):
            value = int(value)
        self.d[key] = {'value': value, 'description': desc}

    def save(self, path, superseded_from=None):
        out = dict(self.d)
        if superseded_from and os.path.exists(superseded_from):
            old = json.load(open(superseded_from))
            out['superseded'] = old.get('superseded', old)
        json.dump(out, open(path, 'w'), indent=1, default=lambda o: float(o) if hasattr(o, '__float__') else str(o))
