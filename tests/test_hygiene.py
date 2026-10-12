import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from xbox import hygiene as hy  # noqa: E402


# ---------------------------------------------------------------- threads
def _collapse(rows, **kw):
    """rows: list of (account, t, text)"""
    rows = sorted(rows, key=lambda r: (r[0], r[1]))
    g = [r[0] for r in rows]
    t = [r[1] for r in rows]
    m = hy.thread_markers([r[2] for r in rows])
    return hy.collapse_threads(g, t, m.num.values, m.tot.values, m.kw.values, **kw)


def test_gap_rule_strict_60s():
    tid, rule = _collapse([("a", 0, "x"), ("a", 59, "y"), ("a", 118, "z"), ("a", 178, "w")])
    assert tid.tolist() == [0, 0, 0, 1]          # 59 s links, 60 s does not
    assert rule.tolist() == [0, 1, 1, 0]


def test_threads_never_cross_accounts():
    tid, _ = _collapse([("a", 0, "x"), ("b", 1, "y"), ("a", 2, "z")])
    # sorted: a0, a2, b1 -> a0,a2 linked; b separate
    assert tid.tolist() == [0, 0, 1]


def test_numbered_markers_link_beyond_gap():
    rows = [("a", 0, "1/4 The bill does X"), ("a", 600, "2/4 It also does Y"), ("a", 1500, "3/4 and Z"),
            ("a", 2000, "(4/4) done https://t.co/abc")]
    tid, rule = _collapse(rows)
    assert len(set(tid)) == 1
    assert rule.tolist() == [0, 2, 2, 2]


def test_numbered_requires_sequence_and_window():
    # out of sequence 1/ then 3/ ; and a 2/ after 3 hours
    tid, _ = _collapse([("a", 0, "1/ start"), ("a", 600, "3/ skipped")])
    assert tid.tolist() == [0, 1]
    tid, _ = _collapse([("a", 0, "1/ start"), ("a", 3 * 3600, "2/ late")])
    assert tid.tolist() == [0, 1]


def test_dates_and_fractions_do_not_link():
    tid, _ = _collapse([("a", 0, "Join us on 10/12/2020"), ("a", 600, "3/4 of voters agree"), ("a", 1200, "see 3/5")])
    assert tid.tolist() == [0, 1, 2]


def test_keyword_links_next_post_within_5_min():
    tid, rule = _collapse([("a", 0, "A thread on the budget \U0001F9F5"), ("a", 200, "First, the numbers"),
                           ("a", 700, "unrelated")])
    assert tid.tolist() == [0, 0, 1]
    assert rule.tolist() == [0, 3, 0]


def test_marker_parsing():
    m = hy.thread_markers(["1/ hello", "(3/n) more", "end 2/5", "plain", "THREAD: why", "x \U0001F9F5", "10/12/2020"])
    assert m.num.tolist() == [1, 3, 2, -1, -1, -1, -1]
    assert m.tot.tolist() == [-1, 0, 5, -1, -1, -1, -1]
    assert m.kw.tolist() == [False, False, False, False, True, True, False]


def test_block_prevents_link():
    g, t = ["a", "a"], [0, 10]
    m = hy.thread_markers(["x", "y"])
    tid, _ = hy.collapse_threads(g, t, m.num.values, m.tot.values, m.kw.values, block=[0, 1])
    assert tid.tolist() == [0, 1]


# ---------------------------------------------------------------- outages
def _synthetic(seed=1, n_days=240, outage_day=None, outage_hours=None):
    """Hourly Poisson counts on a 'design' of every third day (day of month 1,4,..,28), weekday x hour rhythm."""
    rng = np.random.default_rng(seed)
    d0 = np.datetime64("2019-01-01T05:00")          # 00:00 ET (EST)
    h0 = int(d0.astype("datetime64[h]").astype("int64"))
    n = n_days * 24
    hours = h0 + np.arange(n)
    day, eth, dow, dom = hy.et_fields(hours)
    base = np.exp(-0.5 * ((np.arange(24) - 14) / 4.0) ** 2) + 0.05
    wk = np.where(dow >= 5, 0.5, 1.0)
    lam = 3000 * wk * base[eth] / base.sum()
    keep = hy.design_days_default(dom)
    c = rng.poisson(np.where(keep, lam, 0.0))
    return h0, hours, c, keep, day


def test_outage_zero_design_day_and_partial_hours():
    h0, hours, c, keep, day = _synthetic()
    days = np.unique(day[keep])
    zero_day = days[20]
    c = c.copy()
    c[day == zero_day] = 0                                     # whole day missing
    part_day = days[40]
    sel = np.flatnonzero((day == part_day))[8:16]
    c[sel] = 0                                                 # 8 hours missing, daytime
    df = hy.detect_outages(c, h0)
    assert df.outage[(day == zero_day)].all()
    assert df.outage.values[sel].all()
    # no flagged hour on days that were untouched (padding stays within touched days or adjacent hours)
    touched = np.isin(day, [zero_day, part_day])
    assert df.outage.values[~touched].sum() <= 2
    # design-absent days are not outages and are not in the frame
    assert not df.outage.values[~keep].any()
    assert not df.in_frame.values[~keep & ~np.isin(day, [zero_day])].any()
    runs = hy.outage_runs(df)
    assert len(runs) == 2


def test_no_outage_on_clean_data():
    h0, hours, c, keep, day = _synthetic(seed=5)
    df = hy.detect_outages(c, h0)
    assert df.outage.sum() == 0
    assert df.observed.values[keep].all()


def test_gap_valid_drops_gaps_spanning_unobserved():
    obs = np.ones(100, dtype=bool)
    obs[40:50] = False
    cum = np.concatenate([[0], np.cumsum(obs)])
    t = np.array([5, 20, 39, 55, 70]) * 3600 + 100       # hours 5,20,39,55,70
    v = hy.gap_valid(t, cum, 0)
    assert v.tolist() == [True, True, False, True]
    t2 = np.array([30, 45, 60]) * 3600                    # middle event inside the hole
    assert hy.gap_valid(t2, cum, 0).tolist() == [False, False]


def test_sessionize_breaks_on_gap_and_invalid():
    t = np.array([0, 100, 1000, 3000, 3100])
    valid = np.array([True, True, True, False])
    s = hy.sessionize(t, valid, 1800)
    assert s.tolist() == [0, 0, 0, 1, 2]        # 2000 s gap >= 1800 break; invalid gap breaks


# ---------------------------------------------------------------- LOO
def test_loo_counts_sum_to_total_and_mask():
    rng = np.random.default_rng(0)
    own = rng.integers(0, 5, size=(6, 50))
    total = own.sum(0)
    obs = np.ones(50, dtype=bool)
    obs[10:15] = False
    loo = np.stack([hy.loo_counts(total, own[i], obs) for i in range(6)])
    assert np.isnan(loo[:, 10:15]).all()
    assert not np.isnan(loo[:, obs]).any()
    # sum over accounts of own + loo = total for every account
    for i in range(6):
        assert np.allclose(loo[i, obs] + own[i, obs], total[obs])
    assert (loo[:, obs] >= 0).all()
    # the focal account's own posts never enter its LOO count
    a = np.zeros(50, dtype=int)
    a[3] = 7
    assert hy.loo_counts(total + a, total + a, None)[3] == 0


def test_hourly_counts():
    t = np.array([3600 * 5 + 1, 3600 * 5 + 99, 3600 * 7, 3600 * 100])
    c = hy.hourly_counts(t, 4, 10)
    assert c.tolist() == [0, 2, 0, 1, 0, 0, 0, 0, 0, 0]


def test_scheduler_stats():
    t = np.arange(0, 60 * 60 * 10, 60 * 15)              # exactly on quarter minutes
    s = hy.scheduler_stats(t)
    assert s["quarter_share"] == 1.0 and s["quarter_ratio"] == pytest.approx(15.0)
    rng = np.random.default_rng(0)
    s2 = hy.scheduler_stats(rng.integers(0, 10 ** 7, 20000))
    assert abs(s2["quarter_ratio"] - 1) < 0.1


def test_different_addressee_not_linked():
    texts = ["@alice thanks", "@bob thanks", "@bob and you too", "plain follow up"]
    forbid = hy.addressee_mismatch(texts)
    assert forbid.tolist() == [False, True, False, False]
    m = hy.thread_markers(texts)
    tid, rule = hy.collapse_threads(["a"] * 4, [0, 5, 10, 15], m.num.values, m.tot.values, m.kw.values, forbid=forbid)
    assert tid.tolist() == [0, 1, 1, 1]


def test_low_but_nonzero_hours_flagged_and_holiday_like_day_not():
    h0, hours, c, keep, day = _synthetic(seed=3)
    days = np.unique(day[keep])
    c = c.copy()
    sel = np.flatnonzero(day == days[30])[8:18]
    c[sel] = np.random.default_rng(0).poisson(c[sel] * 0.03)        # 3 percent of normal: collection failure
    hol = day == days[60]
    c[hol] = np.random.default_rng(1).poisson(c[hol] * 0.25)         # 25 percent all day: a holiday, not an outage
    df = hy.detect_outages(c, h0)
    assert df.outage.values[sel].all()
    assert not df.outage.values[hol].any()


def test_design_days_and_out_of_range_events():
    assert hy.design_days_default(np.array([1, 4, 28, 31, 29, 2])).tolist() == [True, True, True, False, False, False]
    cum = np.concatenate([[0], np.cumsum(np.ones(10, dtype=int))])
    t = np.array([3, 20]) * 3600
    assert hy.gap_valid(t, cum, 0).tolist() == [False]                # second event is outside the observed range
