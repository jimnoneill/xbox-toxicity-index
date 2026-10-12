import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from xbox import hawkes as hk  # noqa: E402
from xbox import location as loc  # noqa: E402
from xbox import temporal_v2 as tv  # noqa: E402


def day(s):
    return int(np.datetime64(s, "D").astype("int64"))


def epoch(s):
    return int(np.datetime64(s, "s").astype("int64"))


# ---------------------------------------------------------------- burstiness
def test_a_n_limits():
    for n in (10, 100, 5000):
        assert tv.a_n(0.0, n) == pytest.approx(-1.0)
        r0 = np.sqrt((n - 1) / (n + 1))
        assert tv.a_n(r0, n) == pytest.approx(0.0, abs=1e-12)
    assert tv.a_n(1.7, 10 ** 8) == pytest.approx(tv.goh_barabasi(1.7), abs=1e-3)


def test_regular_process_gives_minus_one():
    t = np.arange(0, 1000 * 60, 60)
    s = tv.burstiness_stats(t)
    assert s["r"] == pytest.approx(0.0) and s["B"] == pytest.approx(-1.0) and s["A_n"] == pytest.approx(-1.0)


def test_shift_invariance_burstiness_and_entropy():
    rng = np.random.default_rng(1)
    t = np.sort(rng.integers(0, 10 ** 7, 3000))
    a = tv.burstiness_stats(t)
    b = tv.burstiness_stats(t + 123456789)
    assert a["A_n"] == pytest.approx(b["A_n"]) and a["B"] == pytest.approx(b["B"])
    h = ((t // 3600) % 24)
    for k in range(24):
        assert tv.entropy_norm((h + k) % 24) == pytest.approx(tv.entropy_norm(h))


def test_entropy_bounds_and_example():
    assert tv.entropy_norm(np.arange(24).repeat(5)) == pytest.approx(1.0)
    assert tv.entropy_norm(np.full(50, 7)) == pytest.approx(0.0)
    assert 4.21 / np.log2(24) == pytest.approx(0.918, abs=1e-3)


def test_zero_gap_share_reported():
    s = tv.burstiness_stats(np.array([0, 0, 10, 10, 20, 50]))
    assert s["zero_gap_share"] == pytest.approx(2 / 5)


# ---------------------------------------------------------------- DST
def test_dst_denver_and_new_york():
    jan = np.array([epoch("2024-01-15T17:00:00")])
    jul = np.array([epoch("2024-07-15T17:00:00")])
    assert loc.to_local(jan, "America/Denver")[0][0] == 10
    assert loc.to_local(jan, "America/New_York")[0][0] == 12
    assert loc.to_local(jul, "America/Denver")[0][0] == 11
    assert loc.to_local(jul, "America/New_York")[0][0] == 13
    # the spring-forward day 2024-03-10: 06:30Z is 23:30 on the 9th in Denver (MST), 08:30Z is 02:30 MDT -> 02:30
    pre = np.array([epoch("2024-03-10T08:30:00")])
    post = np.array([epoch("2024-03-10T09:30:00")])
    assert loc.to_local(pre, "America/Denver")[0][0] == 1
    assert loc.to_local(post, "America/Denver")[0][0] == 3
    # local calendar date crosses UTC midnight
    h, dow, d, _, _ = loc.to_local(np.array([epoch("2024-01-16T03:00:00")]), "America/New_York")
    assert h[0] == 22 and d[0] == day("2024-01-15")


def test_scenario_local_hours():
    # a member with a vote on Tue 2024-01-16 only; a post that day uses DC time, 5 days later uses home
    ml = loc.MemberLocation(np.array([day("2024-01-16")]), np.array([day("2024-01-16")]), "UT")
    t = np.array([epoch("2024-01-16T17:00:00"), epoch("2024-01-23T17:00:00")])
    out = ml.local_hour_dow(t, scenarios=loc.SCENARIOS)
    assert list(out["all_dc"][0]) == [12, 12]
    assert list(out["all_home"][0]) == [10, 10]
    assert list(out["model_trans_dc"][0]) == [12, 10]


# ---------------------------------------------------------------- day classes
def test_day_classes():
    mon = day("2024-01-01")
    votes = np.array([mon + 1, mon + 3, mon + 8, mon + 9, mon + 22])  # Tue, Thu, Tue, Wed, Tue(2 wks later)
    cd = votes.copy()
    days = mon + np.arange(0, 24)
    c = loc.classify_days(days, votes, cd)
    nm = lambda i: loc.CLASS_NAMES[int(c[i])]
    assert nm(1) == "dc_vote" and nm(3) == "dc_vote"
    assert nm(2) == "dc_bracket"                      # Wed between Tue and Thu
    assert nm(0) == "transition"                      # Monday before first vote
    assert nm(4) == "transition"                      # Friday after last vote
    assert nm(5) == "home" and nm(6) == "home"        # weekend never bracketed
    assert nm(7) == "transition"                      # Monday before Tuesday vote
    assert nm(10) == "transition"                     # Thursday after Wednesday vote
    assert nm(12) == "home"                           # Saturday
    assert nm(14) == "home"                           # Monday, 2+ days from votes both sides
    assert nm(21) == "transition"                     # Monday before Tuesday vote


def test_bracket_not_across_weekend():
    mon = day("2024-01-01")
    votes = np.array([mon + 4, mon + 8])              # Fri and Tue
    c = loc.classify_days(np.array([mon + 7]), votes, votes)  # Monday between
    assert loc.CLASS_NAMES[int(c[0])] == "transition"


def test_chamber_vote_without_member_vote_is_transition():
    mon = day("2024-01-01")
    votes = np.array([mon + 1])
    cd = np.array([mon + 1, mon + 6])                 # chamber also voted on Sunday
    c = loc.classify_days(np.array([mon + 6]), votes, cd)
    assert loc.CLASS_NAMES[int(c[0])] == "home"       # gap 5 days: not within one day of a member vote
    votes2 = np.array([mon + 4])
    cd2 = np.array([mon + 4, mon + 5])                # member voted Friday, chamber Saturday
    c2 = loc.classify_days(np.array([mon + 5]), votes2, cd2)
    assert loc.CLASS_NAMES[int(c2[0])] == "transition"


# ---------------------------------------------------------------- weekend
def test_weekend_denominator_is_posts_not_days():
    dow = np.array([5] * 10)                          # ten posts, all on Saturdays
    s, r = tv.weekend_stats(dow)
    assert s == 1.0 and r == pytest.approx(3.5)
    dow2 = np.array([0] * 5 + [5] * 5)
    s2, r2 = tv.weekend_stats(dow2)
    assert s2 == 0.5 and r2 == pytest.approx(0.5 / (2 / 7))
    uniform = np.arange(7).repeat(10)
    assert tv.weekend_stats(uniform)[1] == pytest.approx(1.0)


# ---------------------------------------------------------------- statistics
def test_hedges_j_value():
    df = 195 + 182 - 2
    assert 1 - 3 / (4 * df - 1) == pytest.approx(0.9980, abs=1e-4)
    x = np.random.default_rng(0).normal(0.5, 1, 195)
    y = np.random.default_rng(1).normal(0.0, 1, 182)
    g = tv.hedges_g(x, y)
    d = (x.mean() - y.mean()) / np.sqrt(((194) * x.var(ddof=1) + 181 * y.var(ddof=1)) / 375)
    assert g == pytest.approx(d * (1 - 3 / (4 * 375 - 1)))


def test_empirical_percentile_midrank():
    ref = np.arange(1, 101)
    assert tv.emp_percentile(50, ref) == pytest.approx(49.5)
    assert tv.emp_percentile(1000, ref) == 100.0
    assert tv.emp_percentile(5, [5, 5, 5, 5]) == 50.0


def test_rapid_share():
    t = np.array([0, 100, 500, 800, 5000])
    assert tv.rapid_share(t) == pytest.approx(2 / 4)


# ---------------------------------------------------------------- Hawkes
def _sim_design(eta, n=4000, seed=3):
    N = 8 * 720
    s = np.arange(N)
    m, k = s // 720, (s % 24) + 24 * (((s // 24) % 7) >= 5)
    h = np.arange(24)
    prof = np.tile(0.2 + np.exp(1.5 * np.cos(2 * np.pi * (h - 15) / 24)), 2)
    prof = prof / prof.mean()
    raw = prof[k]
    mu = raw * n * (1 - eta) / (raw.sum() * 60)
    t = hk.simulate_hawkes(mu, eta, 1 / 15.0, seed)
    return hk.make_design(np.floor(t * 60).astype("int64"), m, k, 0, 8, jitter_seed=seed)


def test_hawkes_recovers_eta():
    for eta in (0.2, 0.6):
        d = _sim_design(eta)
        f = hk.fit(d)
        assert abs(f["eta"] - eta) < 0.1
        assert 8 < f["inv_beta_min"] < 30


def test_hawkes_constant_baseline_inflates_on_poisson():
    d = _sim_design(0.0, n=6000)
    assert hk.fit(d)["eta"] < 0.08
    assert hk.fit(hk.constant_baseline_design(d))["eta"] > 0.2


def test_hawkes_ci_contains_truth_and_ks_ok():
    d = _sim_design(0.5, n=6000, seed=11)
    f = hk.fit(d)
    lo, hi = hk.profile_ci(d, f)
    assert lo < 0.5 < hi
    assert hk.time_rescaling(d, f)["ks_stat"] < 0.05


# ---------------------------------------------------------------- primary sample: non-retweet posts
def test_nonretweet_filter_matches_definition():
    texts = ["RT @someone: hello", "@a thanks for coming", "plain post", "quote tweet with comment https://t.co/x",
             "rt @lower case is not the rule", "RT@nospace", " RT @leading space", "", None, "My view: RT @x inside",
             "RT @a RT @b", "@RT_fan reply"]
    got = tv.nonretweet_mask_congresstweets(np.array(texts, dtype=object))
    want = np.array([not (t or "").startswith("RT @") for t in texts])
    assert (got == want).all()
    # spot checks of the definition itself
    assert not got[0] and not got[10]
    assert got[1] and got[2] and got[3] and got[4] and got[5] and got[6] and got[7] and got[8] and got[9] and got[11]
    assert tv.is_retweet_congresstweets(np.array(texts, dtype=object)).sum() == 2


def test_nonretweet_mask_lee_types():
    types = np.array(["tweet", "reply", "quote", "retweet", "tweet", "retweet"])
    assert list(tv.nonretweet_mask_lee(types)) == [True, True, True, False, True, False]


def test_nonretweet_set_is_superset_of_old_original_set():
    texts = np.array(["RT @a x", "@b y", "z", "w"], dtype=object)
    old_orig = tv.post_type_congresstweets(texts) == "original"
    new = tv.nonretweet_mask_congresstweets(texts)
    assert (new | ~old_orig).all() and new.sum() == 3 and old_orig.sum() == 2


# ---------------------------------------------------------------- Holm, zones, random-effect baseline
def test_holm_adjust():
    p = np.array([0.01, 0.04, 0.03, np.nan, 0.5])
    a = tv.holm_adjust(p)
    # sorted ascending among 4 valid: 0.01*4, 0.03*3, 0.04*2, 0.5*1 with running max
    assert a[0] == pytest.approx(0.04) and a[2] == pytest.approx(0.09)
    assert a[1] == pytest.approx(0.09) and a[4] == pytest.approx(0.5) and np.isnan(a[3])


def test_canonical_zone_and_home_zone_override():
    assert loc.canonical_zone("America/Indiana/Indianapolis") == "America/New_York"
    assert loc.canonical_zone("America/Boise") == "America/Denver"
    assert loc.canonical_zone("America/Indiana/Tell_City") == "America/Chicago"
    ml = loc.MemberLocation(np.array([day("2024-01-16")]), np.array([day("2024-01-16")]), "TN",
                            home_zone="America/New_York")
    assert ml.state_zone == "America/Chicago" and ml.home_zone == "America/New_York"
    t = np.array([epoch("2024-01-23T17:00:00")])
    assert ml.local_hour_dow(t, ("all_home",))["all_home"][0][0] == 12


def test_district_zone_lookup():
    import pandas as pd
    allm = pd.DataFrame(dict(icpsr=[1, 1, 2], chamber=["House"] * 3, state_abbrev=["TN", "TN", "TN"],
                             congress=[117, 118, 118], district_code=[3, 3, 7]))
    table = {("cd116", "TN", 3): ("America/New_York", 1.0), ("cd118", "TN", 3): ("America/New_York", 1.0),
             ("cd118", "TN", 7): ("America/Chicago", 1.0)}
    assert loc.district_zone_for_member(1, "TN", allm, table)[0] == "America/New_York"
    assert loc.district_zone_for_member(2, "TN", allm, table)[0] == "America/Chicago"
    assert loc.district_zone_for_member(3, "TN", allm, table) == (None, "no_district")
    assert loc.district_zone_for_member(1, "CA", allm, table) == (None, "not_split_state")


def test_day_random_effect_variant_runs_and_shrinks():
    d = _sim_design(0.0, n=3000, seed=5)
    N = 8 * 720
    k = d["slot_bin"]
    dd = hk.make_design(np.floor(d["t"] * 60).astype("int64"), np.arange(N) // 24, k, 0, N // 24, jitter_seed=5,
                        jitter=False)
    e_fixed = hk.fit(dd)["eta"]
    e_re = hk.fit(dd, prior_kappa=2.0)["eta"]
    assert 0.0 <= e_re <= 0.999 and 0.0 <= e_fixed <= 0.999
