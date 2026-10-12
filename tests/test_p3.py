"""Tests for P3 (xbox/forecast_p3.py): leakage, metric correctness, placebo. Run: python -m pytest tests/test_p3.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from xbox import forecast_p3 as fp  # noqa: E402


def synth(n=120, years=range(2017, 2023), seed=3):
    rng = np.random.default_rng(seed)
    rows = []
    base = rng.normal(0.25, 0.05, n)
    vol = rng.normal(0.5, 1.0, n)
    for y in years:
        for i in range(n):
            r = dict(bioguide=f"M{i:03d}", py=y, n_events=500, in_office_days=365, n_scored=100,
                     party="R" if i % 2 else "D", chamber="senate" if i % 5 == 0 else "house",
                     ti7_se=0.01, aligned_with_president=float(i % 2), in_chamber_majority=float(i % 3 == 0))
            for c in fp.LEVELS + fp.DESC + fp.MIX + fp.COMOVE + fp.TRENDS:
                r[c] = rng.normal()
            r["ti7"] = base[i] + 0.3 * (y - 2017) * 0.01 + rng.normal(0, 0.01)
            r["log_rate"] = vol[i] + rng.normal(0, 0.2)
            r["np_dim1"] = r["abs_np_dim1"] = 0.1
            r["seniority_congresses"] = 3
            rows.append(r)
    return pd.DataFrame(rows)


def build(panel):
    return fp.make_transitions(panel, {}, min_scored=25)


def test_no_name_collision_between_features_and_targets():
    banned = set(fp.TARGETS) | set(fp.ESC)
    assert banned.isdisjoint(fp.FEATURES_FULL)
    assert set(fp.FEATURES_CORE) <= set(fp.FEATURES_FULL)


def test_features_do_not_use_target_year():
    p = synth()
    T0, _ = build(p)
    q = p.copy()
    m = q.py == 2020  # perturb a whole year; origin 2018 targets are year 2019, so origins <= 2018 must not change
    for c in ["ti7", "log_rate", "tribal", "ragebait", "A_n"]:
        q.loc[m, c] = q.loc[m, c] + 100.0
    T1, _ = build(q)
    feats = fp.FEATURES_FULL
    a = T0[T0.origin <= 2018].sort_values(["origin", "bioguide"])
    b = T1[T1.origin <= 2018].sort_values(["origin", "bioguide"])
    pd.testing.assert_frame_equal(a[feats].reset_index(drop=True), b[feats].reset_index(drop=True))
    # origin 2019 targets are year 2020: features unchanged, targets changed
    a = T0[T0.origin == 2019].sort_values("bioguide").reset_index(drop=True)
    b = T1[T1.origin == 2019].sort_values("bioguide").reset_index(drop=True)
    pd.testing.assert_frame_equal(a[feats], b[feats])
    assert (a.d_ti7 != b.d_ti7).all()


def test_rolling_prediction_ignores_future():
    p = synth(n=90)
    T, _ = build(p)
    base, _ = fp.rolling(T, [2019], targets={"d_ti7": ("ti7", "TI7")})
    T2 = T.copy()
    fut = T2.origin >= 2020  # transitions whose targets lie after the origin's own target year
    T2.loc[fut, ["d_ti7", "d_logvol", "esc_ti7"]] = 99.0
    # also destroy the origin's own targets: predictions must not depend on them
    own = T2.origin == 2019
    T2.loc[own, ["d_ti7"]] = -50.0
    alt, _ = fp.rolling(T2, [2019], targets={"d_ti7": ("ti7", "TI7")})
    a = base[base.target == "d_ti7"].set_index(["model", "bioguide"]).pred
    b = alt[alt.target == "d_ti7"].set_index(["model", "bioguide"]).pred
    models = ["persistence", "regression_to_mean", "ridge", "enet", "gbt", "ridge_core", "gbt_core", "party_mean_change"]
    for m in models:
        np.testing.assert_allclose(a.loc[m].sort_index().values, b.loc[m].sort_index().values, atol=1e-10)


def test_placebo_shuffled_targets_have_no_skill():
    p = synth(n=150)
    T, _ = build(p)
    rng = np.random.default_rng(1)
    for t, g in T.groupby("origin"):
        T.loc[g.index, "d_ti7"] = rng.permutation(g.d_ti7.values)
    preds, _ = fp.rolling(T, [2020], targets={"d_ti7": ("ti7", "TI7")})
    r = fp.skill_ci(preds, "d_ti7", "gbt", "persistence")
    assert r["skill"] < 0.03, r  # no better than predicting no change, within noise
    r = fp.skill_ci(preds, "d_ti7", "ridge", "persistence")
    assert r["skill"] < 0.03, r


def test_skill_ci_identity_and_perfect():
    rng = np.random.default_rng(0)
    d = pd.DataFrame(dict(bioguide=np.repeat(np.arange(50), 2), target="t", y=rng.normal(size=100)))
    a = d.assign(model="m", pred=d.y)
    b = d.assign(model="b", pred=0.0)
    both = pd.concat([a, b])
    r = fp.skill_ci(both, "t", "m", "b")
    assert abs(r["skill"] - 1) < 1e-9
    r = fp.skill_ci(both, "t", "b", "b")
    assert abs(r["skill"]) < 1e-9 and r["lo"] <= 0 <= r["hi"]


def test_reliability_calibrated():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 20000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    _, ece = fp.reliability(y, p)
    assert ece < 0.03


def test_seat_flags_schedule():
    su = {("A", 2022): 1, ("A", 2024): 0}
    f = fp.seat_flags(su, "A", 2021)  # origin PY2021, forecasting PY2022 which holds the November 2022 election
    assert f["seat_up_general_next"] == 1 and f["election_year_next"] == 1
    assert f["seat_up_general_cur"] == 0 and f["seat_up_primary_next"] == 0
    f = fp.seat_flags(su, "A", 2022)
    assert f["seat_up_general_cur"] == 1 and f["election_year_next"] == 0


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v()
            print("ok", k)
