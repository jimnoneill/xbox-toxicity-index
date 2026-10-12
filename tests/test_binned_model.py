"""Tests for xbox/binned_model.py (T4). Run: python tests/test_binned_model.py  (or pytest)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from xbox import binned_model as bm  # noqa: E402


def _nb_data(n=20000, seed=1):
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n), rng.normal(size=n), (rng.random(n) < 0.3).astype(float)])
    beta = np.array([-1.0, 0.4, -0.6])
    mu = np.exp(X @ beta)
    alpha = 0.7
    y = rng.poisson(rng.gamma(1 / alpha, alpha * mu))
    return X, y, beta, alpha


def test_fit_recovers_parameters():
    X, y, beta, alpha = _nb_data()
    f = bm.fit_nb2(X, y)
    assert f["converged"]
    assert np.all(np.abs(f["beta"] - beta) < 4 * f["se_model"])
    assert abs(np.log(f["alpha"] / alpha)) < 0.2  # matches statsmodels NegativeBinomial MLE (0.7824)
    assert np.allclose(f["se_model"], f["se_robust"], rtol=0.25)


def test_fit_matches_statsmodels_glm():
    import statsmodels.api as sm
    X, y, beta, alpha = _nb_data(5000, 2)
    f = bm.fit_nb2(X, y)
    g = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=f["alpha"])).fit()
    assert np.allclose(f["beta"], g.params, atol=1e-5)
    assert np.allclose(f["se_model"], g.bse, rtol=1e-3)


def test_collinear_column_dropped():
    X, y, beta, alpha = _nb_data(4000, 3)
    X2 = np.column_stack([X, X[:, 1] * 2.0, np.zeros(len(y))])
    f = bm.fit_nb2(X2, y)
    assert np.isnan(f["beta"][3]) and np.isnan(f["beta"][4])
    assert np.allclose(f["beta"][:3], bm.fit_nb2(X, y)["beta"], atol=1e-6)


def test_lag_features_and_validity():
    y = np.array([0, 2, 0, 1, 3, 0, 4, 1, 0, 2, 5, 1], dtype=float)
    present = np.ones(12, dtype=bool)
    present[5] = False
    lag1, lagL, valid = bm.lag_features(y, present, 4)
    # t = 8: lag1 = log1p(y7 = 1); lagL = log1p(y4 + y5(absent=0) + y6) = log1p(3 + 0 + 4)
    assert np.isclose(lag1[8], np.log1p(1.0)) and np.isclose(lagL[8], np.log1p(7.0))
    assert not valid[5] and not valid[6] and not valid[8] and not valid[9]  # window [t-4, t-1] contains t=5
    assert valid[10] and valid[11] and not valid[3]


def test_simulator_mean_and_lags():
    T = 200000
    eta0 = np.full(T, np.log(0.5))
    y = bm.simulate_nb(eta0, 0.0, 0.0, 0.5, seed=5)
    assert abs(y.mean() - 0.5) < 0.02
    assert abs(y.var() - (0.5 + 0.5 * 0.25)) < 0.05
    y2 = bm.simulate_nb(eta0, 0.4, 0.0, 0.2, seed=6)
    ac = np.corrcoef(y2[1:], y2[:-1])[0, 1]
    assert ac > 0.05  # positive own-lag creates serial dependence
    # recover the lag coefficient with the estimator
    eta0 = np.full(T, np.log(0.3))
    y3 = bm.simulate_nb(eta0, 0.4, 0.2, 0.3, seed=7)
    present = np.ones(T, dtype=bool)
    lag1, lagL, valid = bm.lag_features(y3, present, 24)
    X = np.column_stack([np.ones(T), lag1, lagL])[valid]
    f = bm.fit_nb2(X, y3[valid])
    assert np.all(np.abs(f["beta"][1:] - np.array([0.4, 0.2])) < 4 * f["se_model"][1:])
    assert abs(f["beta"][0] - np.log(0.3)) < 4 * f["se_model"][0]


def test_meta_known_values():
    y = np.array([0.1, 0.3, 0.2, 0.5, -0.1])
    se = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
    r = bm.meta_dl(y, se)
    Q = np.sum((y - y.mean()) ** 2) / 0.01
    tau2 = max(0, (Q - 4) / (5 * 100 - 5 * 100 ** 2 / (5 * 100)))
    assert np.isclose(r["Q"], Q) and np.isclose(r["tau2"], tau2)
    assert np.isclose(r["mu"], y.mean())
    # REML with equal variances: tau2 = var(y, ddof=1) - se^2
    rr = bm.meta_reg(y, se)
    assert np.isclose(rr["tau2"], max(0, y.var(ddof=1) - 0.01), atol=1e-6)
    assert np.isclose(rr["coef"][0], y.mean())
    # homogeneous: tau2 = 0
    assert bm.meta_reg(np.full(6, 0.2), np.full(6, 0.1))["tau2"] == 0.0


def test_meta_moderator_contrast():
    rng = np.random.default_rng(3)
    k = 400
    g = (rng.random(k) < 0.5).astype(float)
    se = rng.uniform(0.03, 0.1, k)
    y = 0.2 + 0.1 * g + rng.normal(0, 0.05, k) + rng.normal(0, se)
    r = bm.meta_reg(y, se, g[:, None])
    assert abs(r["coef"][1] - 0.1) < 4 * r["se"][1]
    assert abs(r["tau"] - 0.05) < 0.02


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
