"""Temporal posting features, corrected version (spec C, E).

All functions take plain numpy arrays. Times are integer UTC epoch seconds unless stated.

Features
  burstiness_stats   r = sigma/mu (ddof=0), Goh-Barabasi B, Kim-Jo finite-size A_n
  entropy_norm       normalized hour-of-day entropy, log2(24) denominator
  night_share        share of posts with local hour in [0,6) or [22,6)
  weekend_stats      share of posts on local Sat/Sun and ratio to 2/7
  rapid_share        share of posts within 300 s of the previous post (volume sensitive)
  hedges_g           small-sample corrected standardized mean difference, bootstrap CI
  emp_percentile     empirical mid-rank percentile (no Gaussian z)
"""
from __future__ import annotations

import numpy as np

LOG2_24 = float(np.log2(24.0))
UNIFORM_WEEKEND = 2.0 / 7.0

PHONE_CLIENTS = {"Twitter for iPhone", "Twitter for Android", "Twitter for iPad"}
TOOL_CLIENTS = {
    "TweetDeck", "TweetDeck Web App", "Sprout Social", "Hootsuite", "Hootsuite Inc.",
    "Buffer", "Twitter Media Studio", "Media Studio", "Gain Platform",
    "Twitter Ads Composer", "Twitter for Advertisers", "Twitter for Advertisers (legacy)",
    "Twitter Ads", "Spredfast", "Sprinklr", "Khoros", "Falcon", "Salesforce", "Dynamic Signal",
    "IFTTT", "Zapier", "dlvr.it", "SocialFlow", "CoSchedule", "Sprout Social Inc.",
    "Publer", "Later", "Oktopost", "Conversocial", "Echobox", "Brandwatch", "Crowdfire",
    "Social Studio", "Twitter Studio", "Twitter Business Experience", "Agora Pulse",
    "Arena", "Meltwater", "NGP VAN", "ActionKit", "Wordpress.com", "WordPress.com",
    "Facebook", "Instagram", "Periscope", "Vine - Make a Scene",
}
WEB_CLIENTS = {"Twitter Web App", "Twitter Web Client", "Twitter for Websites", "Twitter Lite", "Mobile Web (M5)"}


# ------------------------------------------------------------------ burstiness
def a_n(r, n):
    """Kim and Jo (2016) finite-size burstiness. n = number of events."""
    r = np.asarray(r, dtype=float)
    sn1, sm1 = np.sqrt(n + 1.0), np.sqrt(n - 1.0)
    return (sn1 * r - sm1) / ((sn1 - 2.0) * r + sm1)


def goh_barabasi(r):
    r = np.asarray(r, dtype=float)
    return (r - 1.0) / (r + 1.0)


def burstiness_stats(t_sec):
    """Inter-event statistics from event times in seconds (any order)."""
    t = np.sort(np.asarray(t_sec, dtype="int64"))
    n = len(t)
    if n < 3:
        return dict(n=n, r=np.nan, B=np.nan, A_n=np.nan, zero_gap_share=np.nan, mean_gap_s=np.nan)
    tau = np.diff(t).astype(float)
    mu = tau.mean()
    if mu == 0:
        return dict(n=n, r=np.nan, B=np.nan, A_n=np.nan, zero_gap_share=1.0, mean_gap_s=0.0)
    r = tau.std(ddof=0) / mu
    return dict(n=n, r=float(r), B=float(goh_barabasi(r)), A_n=float(a_n(r, n)),
                zero_gap_share=float((tau == 0).mean()), mean_gap_s=float(mu))


# ------------------------------------------------------------------ circadian features
def entropy_norm(hours):
    """Normalized Shannon entropy of the 24-bin hour-of-day histogram, in [0,1]."""
    h = np.asarray(hours, dtype=int)
    if len(h) == 0:
        return np.nan
    p = np.bincount(h, minlength=24) / len(h)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum() / LOG2_24)


def night_share(hours, window="00_06"):
    h = np.asarray(hours)
    if len(h) == 0:
        return np.nan
    if window == "00_06":
        return float((h < 6).mean())
    if window == "22_06":
        return float(((h >= 22) | (h < 6)).mean())
    raise ValueError(window)


def weekend_stats(dow):
    """Share of posts on Saturday/Sunday (dow Monday=0) and ratio to the uniform 2/7.
    The denominator is posts, not active days."""
    d = np.asarray(dow)
    if len(d) == 0:
        return np.nan, np.nan
    s = float((d >= 5).mean())
    return s, s / UNIFORM_WEEKEND


def rapid_share(t_sec, window_s=300):
    """Share of posts (those with a predecessor) posted at most window_s after the previous post."""
    t = np.sort(np.asarray(t_sec, dtype="int64"))
    if len(t) < 2:
        return np.nan
    return float((np.diff(t) <= window_s).mean())


# ------------------------------------------------------------------ post types, clients
def post_type_congresstweets(text):
    """'retweet' if text starts 'RT @', 'reply' if it starts '@', else 'original' (approximation)."""
    import pandas as pd
    s = pd.Series(text).fillna("").astype(str)
    out = np.full(len(s), "original", dtype=object)
    out[s.str.startswith("@").values] = "reply"
    out[s.str.startswith("RT @").values] = "retweet"
    return out


def is_retweet_congresstweets(text):
    """True where the post text starts with 'RT @' (the only retweet rule used for congresstweets)."""
    import pandas as pd
    return pd.Series(text).fillna("").astype(str).str.startswith("RT @").values


def nonretweet_mask_congresstweets(text):
    """Primary post set: every post that is not a retweet. Replies are kept (the '@'-prefix reply rule is unreliable,
    so replies are not separated)."""
    return ~is_retweet_congresstweets(text)


LEE_NONRETWEET_TYPES = ("tweet", "reply", "quote")


def nonretweet_mask_lee(types):
    """Lee archive: non-retweet = type in {tweet, reply, quote}."""
    return np.isin(np.asarray(types), LEE_NONRETWEET_TYPES)


def client_class(source):
    """Map a client name to 'phone', 'tool', 'web' or 'other'."""
    import pandas as pd
    s = pd.Series(source).fillna("").astype(str)
    out = np.full(len(s), "other", dtype=object)
    out[s.isin(WEB_CLIENTS).values] = "web"
    out[s.isin(TOOL_CLIENTS).values] = "tool"
    out[s.isin(PHONE_CLIENTS).values] = "phone"
    return out


# ------------------------------------------------------------------ statistics
def emp_percentile(x, ref):
    """Empirical mid-rank percentile (0..100) of x within reference values ref (NaN ignored)."""
    ref = np.asarray(ref, dtype=float)
    ref = ref[~np.isnan(ref)]
    if len(ref) == 0 or np.isnan(x):
        return np.nan
    less = (ref < x).sum()
    eq = (ref == x).sum()
    return float(100.0 * (less + 0.5 * eq) / len(ref))


def volume_matched_percentile(x, n_x, ref_values, ref_n, n_bins=10):
    """Percentile of x among reference units in the same decile of post count as n_x.
    Deciles are cut on the reference counts; n_x outside the range is clamped to the end deciles."""
    ref_values = np.asarray(ref_values, dtype=float)
    ref_n = np.asarray(ref_n, dtype=float)
    ok = ~np.isnan(ref_values)
    ref_values, ref_n = ref_values[ok], ref_n[ok]
    edges = np.quantile(ref_n, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    b = int(np.clip(np.searchsorted(edges, n_x, side="right") - 1, 0, n_bins - 1))
    in_bin = (ref_n >= edges[b]) & ((ref_n < edges[b + 1]) if b < n_bins - 1 else True)
    return emp_percentile(x, ref_values[in_bin]), int(in_bin.sum())


def hedges_g(x, y):
    """Hedges g (x minus y) with J = 1 - 3/(4 df - 1), df = n1 + n2 - 2."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x[~np.isnan(x)], y[~np.isnan(y)]
    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2:
        return np.nan
    df = n1 + n2 - 2
    sp = np.sqrt(((n1 - 1) * x.var(ddof=1) + (n2 - 1) * y.var(ddof=1)) / df)
    if sp == 0:
        return np.nan
    return float((x.mean() - y.mean()) / sp * (1.0 - 3.0 / (4.0 * df - 1.0)))


def hedges_g_boot(x, y, n_boot=10000, seed=20261008):
    """Hedges g with 95% percentile bootstrap CI (resampling each group, fixed seed)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x[~np.isnan(x)], y[~np.isnan(y)]
    rng = np.random.default_rng(seed)
    g = hedges_g(x, y)
    gs = np.empty(n_boot)
    for b in range(n_boot):
        gs[b] = hedges_g(rng.choice(x, len(x)), rng.choice(y, len(y)))
    lo, hi = np.nanpercentile(gs, [2.5, 97.5])
    return dict(g=g, lo=float(lo), hi=float(hi), n_x=len(x), n_y=len(y))


def welch_p(x, y):
    from scipy.stats import ttest_ind
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x[~np.isnan(x)], y[~np.isnan(y)]
    if len(x) < 2 or len(y) < 2:
        return np.nan
    return float(ttest_ind(x, y, equal_var=False).pvalue)


def holm_adjust(p):
    """Holm step-down adjusted p-values (NaN entries are left out of the family and returned as NaN)."""
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    ok = np.where(~np.isnan(p))[0]
    m = len(ok)
    order = ok[np.argsort(p[ok])]
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        out[i] = min(1.0, run)
    return out
