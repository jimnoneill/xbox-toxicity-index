"""Tests for phase P1: leakage (every feature uses only data strictly before the target event), mixed model recovery,
target encoding, congress lag features, rolling-origin split."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from xbox import forecast_p1 as fp  # noqa: E402


def synth(n=400, seed=1):
    rng = np.random.default_rng(seed)
    gaps = np.round(rng.exponential(900, n)).astype("int64")
    gaps[rng.random(n) < 0.15] = 0          # ties on purpose
    gaps[rng.random(n) < 0.05] += 7200      # new sessions
    t = 1_600_000_000 + np.cumsum(gaps)
    ti7 = rng.random(n)
    ti7[rng.random(n) < 0.3] = np.nan       # unscored events
    tool = rng.random(n) < 0.3
    return t, ti7, tool


KEYS = ["dt_prev", "sess_pos", "sess_el", "n1h", "n24h", "n_scored_prior", "prev_ti7", "prev_scored", "last_scored",
        "roll10", "roll50", "sched_share"]


def _same(a, b):
    return np.array_equal(np.asarray(a, dtype=float), np.asarray(b, dtype=float), equal_nan=True)


def test_hand_checked_values():
    t = np.array([0, 100, 100, 5000, 5100, 5100, 5200], dtype="int64")
    ti7 = np.array([0.1, 0.2, np.nan, 0.4, 0.5, 0.6, 0.7])
    tool = np.array([1, 0, 0, 1, 0, 0, 0], dtype=bool)
    f = fp.account_features(t, ti7, tool)
    # event 2 ties with event 1: only event 0 is strictly earlier
    assert f["cnt"].tolist() == [0, 1, 1, 3, 4, 4, 6]
    assert f["dt_prev"][2] == 100 and f["dt_prev"][1] == 100
    assert np.isnan(f["dt_prev"][0])
    # event 3 (t=5000): previous strictly earlier event is index 2 (unscored) -> prev_ti7 NaN, prev_scored False
    assert not f["prev_scored"][3] and np.isnan(f["prev_ti7"][3])
    assert f["last_scored"][3] == pytest.approx(0.2)     # last scored among earlier events is index 1
    # sessions: break before event 3 (gap 4900 > 1800)
    assert f["sess_pos"].tolist() == [0, 1, 1, 0, 1, 1, 3]
    assert f["sess_el"][4] == 100 and f["sess_el"][3] == 0
    # events 4 and 5 tie: the tied twin never counts as earlier
    assert f["n1h"][5] == 1 and f["n1h"][6] == 3
    assert f["roll50"][6] == pytest.approx(np.mean([0.1, 0.2, 0.4, 0.5, 0.6]))
    assert f["sched_share"][3] == pytest.approx(1 / 3)


@pytest.mark.parametrize("seed", range(5))
def test_no_future_information_perturbation(seed):
    """Replace labels, client flags and times of every event at or after t_i (including ties with i and i itself):
    the features of event i must not change."""
    t, ti7, tool = synth(300, seed)
    base = fp.account_features(t, ti7, tool)
    rng = np.random.default_rng(100 + seed)
    for i in rng.choice(len(t), 40, replace=False):
        fut = np.flatnonzero(t >= t[i])       # same time or later
        t2, y2, k2 = t.copy(), ti7.copy(), tool.copy()
        y2[fut] = rng.random(len(fut))
        k2[fut] = rng.random(len(fut)) < 0.5
        # shift later events further (monotone, so order is kept); event i keeps its own time
        later = np.flatnonzero(t > t[i])
        t2[later] = t[later] + np.cumsum(rng.integers(0, 4000, len(later)))
        pert = fp.account_features(t2, y2, k2)
        for k in KEYS:
            assert _same(base[k][i], pert[k][i]), (k, i)


@pytest.mark.parametrize("seed", range(3))
def test_prefix_equivalence(seed):
    """Features of event i computed from the prefix of strictly earlier events plus t_i equal the full-sequence value."""
    t, ti7, tool = synth(150, seed)
    base = fp.account_features(t, ti7, tool)
    for i in range(1, len(t), 7):
        keep = np.flatnonzero(t < t[i])
        if len(keep) == 0:
            continue
        tp = np.concatenate([t[keep], [t[i]]])
        yp = np.concatenate([ti7[keep], [np.nan]])
        kp = np.concatenate([tool[keep], [False]])
        pre = fp.account_features(tp, yp, kp)
        for k in KEYS:
            assert _same(base[k][i], pre[k][-1]), (k, i)


def test_congress_features_causal():
    rng = np.random.default_rng(3)
    n = 24 * 7 * 8
    c = rng.poisson(20, n).astype(float)
    bad = np.zeros(n, dtype=bool)
    bad[300:310] = True
    base = fp.congress_hour_features(c, bad)
    for h in (200, 400, 700, 1000, 1300):
        c2 = c.copy()
        c2[h:] = rng.poisson(500, n - h)
        b2 = bad.copy()
        b2[h:] = rng.random(n - h) < 0.3
        pert = fp.congress_hour_features(c2, b2)
        for k in base:
            assert _same(base[k][h], pert[k][h]), (k, h)


def test_congress_spike_detects_burst():
    n = 24 * 7 * 8
    c = np.full(n, 10.0)
    c[800:803] = 80.0
    f = fp.congress_hour_features(c, np.zeros(n, dtype=bool))
    assert f["cong_spike"][803] == 1 and f["cong_spike"][804] == 1 and f["cong_spike"][807] == 0
    assert f["cong_spike"][800] == 0       # the burst itself is not yet known at hour 600


def test_target_encoding_excludes_own_label():
    rng = np.random.default_rng(0)
    g = rng.integers(0, 5, 2000)
    y = rng.random(2000)
    m0, _ = fp.target_encode(g, y, 5, folds=5)
    y2 = y.copy()
    # change one row's label: its own OOF encoding must not move
    i = 17
    y2[i] += 10
    m1, _ = fp.target_encode(g, y2, 5, folds=5)
    assert m0[i] == pytest.approx(m1[i])
    # test encoding uses training labels only
    mt, _ = fp.target_encode(g, y, 5, test_g=np.array([0, 1, 4]))
    assert np.isfinite(mt).all()


def test_mixed_lm_recovers_parameters():
    rng = np.random.default_rng(5)
    J, n = 150, 60
    g = np.repeat(np.arange(J), n)
    x1 = rng.normal(size=J * n)
    x2 = rng.normal(size=J * n)
    X = np.column_stack([np.ones(J * n), x1, x2])
    u0 = rng.normal(0, 0.5, J)
    u1 = rng.normal(0, 0.3, J)
    y = 1.0 + 0.7 * x1 - 0.4 * x2 + u0[g] + u1[g] * x1 + rng.normal(0, 0.8, J * n)
    m = fp.MixedLM([0, 1]).fit(X, y, g, J)
    assert np.allclose(m.beta, [1.0, 0.7, -0.4], atol=0.08)
    assert abs(np.sqrt(m.G[0, 0]) - 0.5) < 0.08 and abs(np.sqrt(m.G[1, 1]) - 0.3) < 0.06
    assert abs(np.sqrt(m.s2) - 0.8) < 0.03
    # shrinkage: predictions with random effects beat fixed-only on the training accounts
    e_re = np.mean((y - m.predict(X, g)) ** 2)
    e_fx = np.mean((y - m.predict(X, g, use_re=False)) ** 2)
    assert e_re < e_fx


def test_mixed_logit_recovers_signs_and_calibrates():
    rng = np.random.default_rng(6)
    J, n = 200, 80
    g = np.repeat(np.arange(J), n)
    x1 = rng.normal(size=J * n)
    X = np.column_stack([np.ones(J * n), x1])
    u0 = rng.normal(0, 0.6, J)
    p = 1 / (1 + np.exp(-(-1.0 + 0.8 * x1 + u0[g])))
    y = (rng.random(J * n) < p).astype(float)
    m = fp.MixedLM([0], family="binomial").fit(X, y, g, J)
    assert abs(m.beta[1] - 0.8) < 0.1 and abs(m.beta[0] + 1.0) < 0.2
    ph = m.predict(X, g)
    assert abs(ph.mean() - y.mean()) < 0.01


def test_skill_boot_ci_contains_point():
    rng = np.random.default_rng(1)
    b = rng.random(100) + 1
    a = b * 0.9
    pt, lo, hi = fp.skill_boot(a, b, B=500)
    assert lo <= pt <= hi and abs(pt - 0.1) < 1e-9


def test_rolling_origin_windows_disjoint_and_final_untouched():
    from xbox.forecast_p1 import TEST_START
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/v7"))
    import p1_common as pc
    dev = pc.origins("dev")
    fin = pc.origins("test")
    assert all(b > a for a, b in zip(dev, dev[1:]))
    assert max(dev) < TEST_START <= min(fin)
    # development test windows end before the final test starts
    assert pc.window_end(max(dev)) <= TEST_START


def test_post_type_features_hand_checked():
    t = np.array([0, 100, 100, 5000, 5100], dtype="int64")
    reply = np.array([1, 0, 1, 1, 0], dtype=float)
    multi = np.array([0, 1, 0, 1, 0], dtype=float)
    f = fp.account_features(t, np.full(5, 0.3), np.zeros(5, bool), reply, multi)
    assert np.isnan(f["prev_reply"][0])
    assert f["prev_reply"][1] == 1 and f["prev_reply"][2] == 1          # tie: previous strictly earlier event is event 0
    assert f["prev_multi"][3] == 0                                      # event 3's previous strictly earlier event is index 2
    assert f["prev_reply"][3] == 1 and f["prev_multi"][4] == 1
    assert f["reply_share10"][4] == pytest.approx(3 / 4)                # events 0..3 : replies 1,0,1,1
    assert f["own_reply"].tolist() == reply.tolist()


@pytest.mark.parametrize("seed", range(4))
def test_post_type_features_use_only_earlier_events(seed):
    t, ti7, tool = synth(250, seed)
    rng = np.random.default_rng(seed + 50)
    reply = (rng.random(len(t)) < 0.3).astype(float)
    multi = (rng.random(len(t)) < 0.2).astype(float)
    base = fp.account_features(t, ti7, tool, reply, multi)
    for i in rng.choice(len(t), 40, replace=False):
        fut = np.flatnonzero(t >= t[i])
        r2, m2 = reply.copy(), multi.copy()
        r2[fut] = rng.random(len(fut)) < 0.5
        m2[fut] = rng.random(len(fut)) < 0.5
        pert = fp.account_features(t, ti7, tool, r2, m2)
        for k in ("prev_reply", "prev_multi", "reply_share10"):
            assert _same(base[k][i], pert[k][i]), (k, i)
