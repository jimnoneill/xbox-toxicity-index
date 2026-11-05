"""As-of event-arrival analysis. Event labels are eventual completed-thread TI7."""
import numpy as np
import pandas as pd


def repair_lags(events):
    """Rebuild score lags from completed events, with strict end < arrival.

    For simultaneous timestamps, no event ending at that timestamp is visible.
    The current event's eventual TI7 is a label, never an input.
    """
    chunks = []
    for _, g in events.groupby("screen_name", sort=False):
        starts = g.sort_values(["t", "thread_id"], kind="stable").copy()
        done = g.sort_values(["t_end", "thread_id"], kind="stable")
        ends = done.t_end.to_numpy(dtype=np.int64)
        t = starts.t.to_numpy(dtype=np.int64)
        count = np.searchsorted(ends, t, side="left")
        first = np.searchsorted(t, t, side="left")
        starts["prior_inflight"] = first - count
        starts["n_completed_prior"] = count
        scores = done.ti7.to_numpy(dtype=float)
        valid = np.isfinite(scores)
        n_scored = np.r_[0, valid.cumsum()][count]
        sv = scores[valid]
        cs = np.r_[0., sv.cumsum()]
        last = np.full(len(t), np.nan)
        last[n_scored > 0] = sv[n_scored[n_scored > 0] - 1]
        starts["prev_ti7"] = last
        for window in (10, 50):
            lo = np.maximum(n_scored - window, 0)
            den = n_scored - lo
            starts[f"roll{window}"] = np.where(den > 0, (cs[n_scored] - cs[lo]) / np.maximum(den, 1), np.nan)
        multi = (done["size"].to_numpy() > 1).astype(float)
        starts["prev_multi"] = np.where(count > 0, multi[np.maximum(count - 1, 0)], np.nan)
        chunks.append(starts[["thread_id", "prior_inflight", "n_completed_prior", "prev_ti7", "roll10", "roll50", "prev_multi"]])
    return pd.concat(chunks, ignore_index=True)


def previous_start_inflight(events):
    """Whether the most recent strictly earlier event start is still incomplete."""
    chunks = []
    for _, g in events.groupby("screen_name", sort=False):
        ordered = g.sort_values(["t", "thread_id"], kind="stable")
        starts = ordered.t.to_numpy(dtype=np.int64)
        ends = ordered.t_end.to_numpy(dtype=np.int64)
        pos = np.searchsorted(starts, starts, side="left") - 1
        flag = (pos >= 0) & (ends[np.maximum(pos, 0)] >= starts)
        prior_id = ordered.thread_id.to_numpy()[np.maximum(pos, 0)].copy()
        prior_id[pos < 0] = -1
        prior_end = ends[np.maximum(pos, 0)].astype(float)
        prior_end[pos < 0] = np.nan
        chunks.append(pd.DataFrame({"thread_id": ordered.thread_id.to_numpy(),
                                    "prev_start_thread_id": prior_id, "prev_start_end": prior_end,
                                    "naive_prev_unavailable": flag.astype(np.int8)}))
    return pd.concat(chunks, ignore_index=True)


def select_origin(rows, origin, end, min_train=100):
    """Same accounts and eligible arrival rows for matched train/test models."""
    eligible = rows[rows.n_completed_prior >= 10]
    tr = eligible[(eligible.t < origin) & (eligible.t_end < origin)]
    te = eligible[(eligible.t >= origin) & (eligible.t < end)]
    ok = tr.acct.value_counts()
    keep = ok.index[ok >= min_train]
    return tr[tr.acct.isin(keep)].copy(), te[te.acct.isin(keep)].copy()


def fit_pair(tr, te, base_cols, rhythm_cols, max_train=600_000):
    """Matched-sample ridge with account fixed effects; only training labels fit.

    Standardization and missing-value replacement are estimated on training rows.
    The two models use the same rows and regularization; rhythm nests content.
    """
    from scipy import sparse
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.linear_model import Ridge

    if len(tr) > max_train:
        tr = tr.sample(max_train, random_state=20261010)
    enc = OneHotEncoder(handle_unknown="ignore", dtype=np.float32)
    ctr = enc.fit_transform(tr[["acct"]])
    cte = enc.transform(te[["acct"]])
    y = tr.ti7.to_numpy(dtype=np.float64)
    out = {}
    for key, cols in (("content", list(base_cols)), ("rhythm", list(base_cols) + list(rhythm_cols))):
        a = tr[cols].to_numpy(dtype=np.float32)
        b = te[cols].to_numpy(dtype=np.float32)
        mu = np.nanmean(np.where(np.isfinite(a), a, np.nan), axis=0)
        mu = np.nan_to_num(mu)
        a = np.where(np.isfinite(a), a, mu)
        b = np.where(np.isfinite(b), b, mu)
        sd = np.std(a, axis=0)
        sd = np.where(sd > 1e-8, sd, 1.)
        xtr = sparse.hstack([ctr, sparse.csr_matrix((a - mu) / sd)], format="csr")
        xte = sparse.hstack([cte, sparse.csr_matrix((b - mu) / sd)], format="csr")
        model = Ridge(alpha=10., solver="lsqr", tol=1e-5)
        model.fit(xtr, y)
        out[key] = model.predict(xte)
    return out


def assemble_frame(events, original):
    """Join observed event completion and replace only unsafe target-score lags.

    Event's own first-post reply and timing/rhythm features remain original:
    the original table calculated those from first arrival and earlier starts.
    """
    if events.thread_id.duplicated().any() or original.thread_id.duplicated().any():
        raise ValueError("thread_id must be unique")
    safe = repair_lags(events)
    out = original.merge(events[["thread_id", "t_end"]], on="thread_id", how="left", validate="one_to_one")
    if out.t_end.isna().any():
        raise ValueError("original row without matching event")
    for col in ("prev_ti7", "roll10", "roll50", "prev_multi"):
        out = out.rename(columns={col: "naive_" + col})
    out = out.merge(safe, on="thread_id", how="left", validate="one_to_one")
    if out.n_completed_prior.isna().any():
        raise ValueError("missing completed-event history")
    return out


def score_events(events, posts, probs, probability_columns):
    """Eventual completed-thread label: mean of scored posts' mean dimensions."""
    if posts.id.duplicated().any():
        raise ValueError("duplicate post ID in thread assignment")
    p = probs[["id"] + list(probability_columns)].drop_duplicates("id").copy()
    p["post_ti7"] = p[probability_columns].mean(axis=1)
    joined = posts[["id", "thread_id"]].merge(p[["id", "post_ti7"]], on="id", how="left", validate="one_to_one")
    labels = joined.groupby("thread_id", sort=False).post_ti7.mean().rename("ti7")
    return events.join(labels, on="thread_id")


def metric_summary(d, boot=1000, seed=20261010):
    """Paired MSE gain with row-weighted account-cluster bootstrap CI."""
    if d.empty:
        return {"n": 0, "accounts": 0}
    y = d.ti7.to_numpy()
    a = (y - d.content.to_numpy()) ** 2
    b = (y - d.rhythm.to_numpy()) ** 2
    group = pd.DataFrame({"acct": d.acct.to_numpy(), "a": a, "b": b}).groupby("acct").agg(n=("a", "size"), sa=("a", "sum"), sb=("b", "sum"))
    arr = group[["n", "sa", "sb"]].to_numpy()
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(arr), size=(boot, len(arr)))
    counts = arr[samples, 0].sum(axis=1)
    delta = (arr[samples, 1].sum(axis=1) - arr[samples, 2].sum(axis=1)) / counts
    ma, mb = float(a.mean()), float(b.mean())
    return {"n": int(len(d)), "accounts": int(len(group)), "mse_content": ma,
            "mse_rhythm": mb, "delta_mse": ma - mb,
            "relative_mse_reduction": (ma - mb) / ma,
            "ci95_delta_mse": np.quantile(delta, [.025, .975]).tolist()}
