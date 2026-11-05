"""Contract tests for an isolated, matched P1 LightGBM sensitivity."""
import numpy as np
import pandas as pd
import pytest

from run_gbt import (select_with_cutoff, attach_arrival_lags, paired_summary,
                     prepare_matched, fit_p_r, verify_archive, original_month, matched_month,
                     summarize_predictions, compare_with_ridge, embargo_lags, prior_vote_dc,
                     local_hour_for_member, strict_features, strict_month, strict_summary,
                     aligned_repaired_frames, select_aligned_origin, aligned_month, aligned_summary,
                     month_block_skill_change)


def test_month_block_resampling_keeps_whole_origin_windows_paired():
    d = pd.DataFrame({"origin": ["a", "a", "b", "b"], "ti7": [0., 0., 0., 0.],
                      "orig_p": [1., 1., 2., 2.], "orig_r": [.5, .5, 1., 1.],
                      "aligned_p": [1., 1., 2., 2.], "aligned_r": [.4, .4, .8, .8]})
    z = month_block_skill_change(d, boot=100)
    assert z["n_origins"] == 2
    assert z["paired_skill_change"] == pytest.approx(.09)
    assert len(z["ci95_paired_skill_change"]) == 2


def test_full_aligned_report_uses_repaired_row_cohort_and_strict_ridge():
    from run_gbt import HERE
    import json
    e = json.loads((HERE / "aligned_evaluation.json").read_text())
    assert e["n"] == 422529
    assert e["n_excluded_from_legacy_test"] == 58
    assert e["strict_ridge_paired_comparison"]["n"] == 422529
    assert e["models"]["repaired"]["skill"] > 0
    assert len(e["ci95_paired_skill_change"]) == 2
    assert e["month_block_sensitivity"]["n_origins"] == 13


def test_aligned_summary_pairs_original_and_repaired_on_same_cohort():
    d = pd.DataFrame({"thread_id": [1, 2, 3], "acct": [1, 1, 2], "ti7": [0., 0., 0.],
                      "orig_p": [1., 1., 2.], "orig_r": [.6, .6, 1.2],
                      "aligned_p": [1., 1., 2.], "aligned_r": [.5, .5, 1.]})
    x = aligned_summary(d, boot=100)
    assert x["n"] == 3 and x["accounts"] == 2
    assert x["models"]["legacy_matched"]["skill"] == pytest.approx(.64)
    assert x["models"]["repaired"]["skill"] == pytest.approx(.75)
    assert x["paired_skill_change_repaired_minus_legacy"] == pytest.approx(.11)
    assert len(x["ci95_paired_skill_change"]) == 2
    assert len(x["ci95_paired_delta_mse_change"]) == 2


def test_aligned_month_fits_frozen_p_r_on_exact_closed_cohort():
    from run_gbt import original_models
    m = original_models()
    rng = np.random.default_rng(3)
    o = pd.Timestamp("2022-07-01")
    cols = list(dict.fromkeys(m.SETS["R"]))
    f = pd.DataFrame(rng.normal(size=(130, len(cols))), columns=cols)
    f["thread_id"] = np.arange(130)
    f["acct"] = 1
    f["ti7"] = .5 + .1 * f.prev_ti7
    f["when"] = [o - pd.Timedelta(days=2)] * 125 + [o] * 5
    f["t_end"] = [int(o.timestamp()) - 9000] * 125 + [int(o.timestamp()) + 100] * 5
    f["strict_n_completed_prior"] = 10
    f.loc[0, "strict_n_completed_prior"] = 9
    repaired = f.copy()
    repaired["hour"] = 13.
    prior = f.iloc[125:][["thread_id", "acct", "ti7"]].copy()
    prior["orig_p"] = .1
    prior["orig_r"] = .2
    hp = {f"reg_{s}": {"leaves": 3, "min_leaf": 2, "rounds": 7} for s in ("P", "R")}
    rows, info = aligned_month(f, repaired, o, hp, prior)
    assert info["n_train"] == 124 and info["n_test"] == 5
    assert rows.thread_id.tolist() == list(range(125, 130))
    assert np.isfinite(rows[["aligned_p", "aligned_r"]]).all().all()


def test_aligned_first_origin_exactly_matches_strict_ridge_ids_and_train_count():
    import json
    from run_gbt import P1, RQ1_STRICT, ALIGNED_COLS
    frame = pd.read_parquet(P1 / "features.parquet")
    strict = pd.read_parquet(RQ1_STRICT / "strict_features.parquet",
                             columns=["thread_id", "ti7", "t_end", "strict_n_completed_prior"] +
                                     ["strict_" + c for c in ALIGNED_COLS])
    legacy, repaired = aligned_repaired_frames(frame, strict)
    o = pd.Timestamp("2022-07-01")
    tr, te, trs, tes = select_aligned_origin(legacy, repaired, o)
    ref = pd.read_parquet(RQ1_STRICT / "predictions/combined_conditional_strict/origin_2022-07.parquet",
                          columns=["thread_id"])
    assert te.thread_id.tolist() == ref.thread_id.tolist()
    assert tr.thread_id.equals(trs.thread_id)
    assert te.thread_id.equals(tes.thread_id)
    info = json.loads((RQ1_STRICT / "monthly_metrics.json").read_text())["combined_conditional_strict"][0]
    assert len(tr) == info["n_train_after_account_filter"]
    assert tr.t_end.max() + 7200 < int(o.timestamp())


def test_aligned_repaired_frames_use_all_strict_ridge_feature_blocks():
    from run_gbt import LAGS
    original = pd.DataFrame({"thread_id": [1, 2], "ti7": [.2, .3],
                             "log_dt": [2., 4.], "prev_ti7": [.1, .2],
                             "roll10": [.1, .2], "roll50": [.1, .2],
                             "hour": [16., 12.], "hsin": [0., 0.], "hcos": [0., 0.],
                             "h2sin": [0., 0.], "h2cos": [0., 0.], "night": [0., 0.],
                             "weekend": [0., 0.], "cong_p1": [1., 2.],
                             "cong_p24": [3., 4.], "cong_spike_lr": [.1, .2],
                             "cong_spike": [0., 1.]})
    cols = list(LAGS) + ["hour", "hsin", "hcos", "h2sin", "h2cos", "night", "weekend",
                         "cong_p1", "cong_p24", "cong_spike_lr", "cong_spike"]
    strict = pd.DataFrame({"thread_id": [1, 2], "ti7": [.2, .3], "t_end": [10, 20],
                           "strict_n_completed_prior": [9, 10],
                           **{"strict_" + c: [9., 8.] for c in cols}})
    naive, repaired = aligned_repaired_frames(original, strict)
    assert naive.t_end.tolist() == [10, 20]
    assert repaired.strict_n_completed_prior.tolist() == [9, 10]
    assert repaired.ti7.equals(original.ti7) and repaired.log_dt.equals(original.log_dt)
    for c in cols:
        assert repaired[c].tolist() == [9., 8.]
    with pytest.raises(AssertionError, match="aligned"):
        aligned_repaired_frames(original, strict.iloc[::-1].reset_index(drop=True))


def test_manifest_output_hashes_remain_valid_after_command_exits():
    from run_gbt import HERE
    import hashlib
    import json
    manifest = json.loads((HERE / "manifest.json").read_text())
    assert len(manifest["outputs"]) >= 55
    for rel, spec in manifest["outputs"].items():
        data = (HERE / rel).read_bytes()
        assert len(data) == spec["bytes"]
        assert hashlib.sha256(data).hexdigest() == spec["sha256"], rel


def test_full_strict_report_has_five_ablations_and_paired_intervals():
    from run_gbt import HERE
    import json
    e = json.loads((HERE / "strict_evaluation.json").read_text())
    assert e["n"] == 422587
    assert set(e["models"]) == {"original", "cutoff", "asof", "embargo", "strict"}
    assert "vote_minus_embargo" in e["ci95_paired_skill_change"]
    assert "strict_minus_original" in e["ci95_paired_skill_change"]
    assert len(e["ci95_paired_skill_change"]["strict_minus_original"]) == 2
    assert "1,502" in e["count_discrepancy_note"]


def test_strict_summary_separates_embargo_and_vote_with_paired_account_ci():
    d = pd.DataFrame({"acct": [1, 1, 2], "ti7": [0., 0., 0.],
                      "orig_p": [1., 1., 2.], "orig_r": [.6, .6, 1.2],
                      "cut_p": [1., 1., 2.], "cut_r": [.6, .6, 1.2],
                      "safe_p": [1., 1., 2.], "safe_r": [.6, .6, 1.2],
                      "emb_p": [1., 1., 2.], "emb_r": [.5, .5, 1.],
                      "strict_p": [1., 1., 2.], "strict_r": [.4, .4, .8]})
    x = strict_summary(d, boot=100)
    assert x["n"] == 3 and x["accounts"] == 2
    assert x["models"]["original"]["skill"] == pytest.approx(.64)
    assert x["models"]["strict"]["skill"] == pytest.approx(.84)
    assert x["paired_skill_change"]["strict_minus_original"] == pytest.approx(.20)
    assert x["paired_skill_change"]["vote_minus_embargo"] == pytest.approx(.09)
    assert x["paired_skill_change"]["embargo_minus_asof"] == pytest.approx(.11)
    assert len(x["ci95_paired_skill_change"]["strict_minus_original"]) == 2
    assert len(x["models"]["strict"]["ci95_delta_mse"]) == 2
    assert len(x["ci95_paired_r_mse_difference"]["strict_minus_original"]) == 2


def test_strict_month_real_first_origin_embargo_and_vote_nested():
    import json
    from run_gbt import HERE, P1, RQ1
    frame = pd.read_parquet(P1 / "features.parquet")
    arrivals = pd.read_parquet(RQ1 / "arrival_features.parquet", columns=["thread_id", "t_end"])
    assert frame.thread_id.equals(arrivals.thread_id)
    frame["t_end"] = arrivals.t_end.to_numpy()
    corrected = pd.read_parquet(HERE / "strict_features.parquet")
    lag, vote = strict_features(frame, corrected)
    original = pd.read_parquet(HERE / "predictions_matched/origin_2022-07.parquet")
    r, diag = strict_month(frame, lag, vote, pd.Timestamp("2022-07-01"),
                           json.loads((P1 / "hp.json").read_text()), original)
    assert len(r) == diag["n_test"] <= len(original)
    assert diag["n_train"] < 2392594
    np.testing.assert_array_equal(r.emb_p, r.strict_p)
    assert r.thread_id.isin(original.thread_id).all()
    assert np.isfinite(r[["emb_p", "emb_r", "strict_r"]]).all().all()


def test_full_strict_feature_artifact_repairs_documented_july_vote_case():
    from run_gbt import HERE
    p = HERE / "strict_features.parquet"
    d = pd.read_parquet(p)
    assert len(d) == 2821325 and d.thread_id.is_unique
    row = d.set_index("thread_id").loc[1986702]
    assert row.hour == 15.0  # original 16: anticipates July 12 vote from July 11
    assert d[["prev_ti7", "roll10", "roll50"]].notna().all().all()


def test_strict_features_change_only_lags_and_local_predictors():
    f = pd.DataFrame({"thread_id": [1, 2], "ti7": [.2, .4], "prev_ti7": [.1, .3],
                      "roll10": [.1, .2], "roll50": [.2, .3], "hour": [14., 15.],
                      "hsin": [0., 0.], "hcos": [0., 0.], "h2sin": [0., 0.],
                      "h2cos": [0., 0.], "night": [0., 0.], "weekend": [0., 0.],
                      "log_dt": [3., 4.], "t_end": [30, 40]})
    c = pd.DataFrame({"thread_id": [1, 2], "prev_ti7": [.05, .2],
                      "roll10": [.08, .1], "roll50": [.15, .2],
                      "hour": [13., 16.], "weekend": [0., 1.]})
    lag, vote = strict_features(f, c)
    assert lag.hour.equals(f.hour) and lag.log_dt.equals(f.log_dt)
    assert vote.ti7.equals(f.ti7) and vote.log_dt.equals(f.log_dt)
    assert vote.hour.tolist() == [13., 16.]
    np.testing.assert_allclose(vote.prev_ti7, [.05, .2], atol=1e-7)
    assert vote.weekend.tolist() == [0., 1.]
    np.testing.assert_allclose(vote.hsin, np.sin(2 * np.pi * vote.hour / 24))
    with pytest.raises(AssertionError, match="alignment"):
        strict_features(f, c.iloc[::-1].reset_index(drop=True))


def test_local_hour_ignores_next_and_same_day_vote_and_uses_known_prior():
    from run_gbt import ROOT
    import sys
    sys.path.insert(0, str(ROOT))
    from xbox import location as lc
    t = np.array([int(pd.Timestamp("2022-07-12T20:00:00Z").timestamp())])
    y = np.datetime64("2022-07-11", "D").astype(int)
    nxt = np.datetime64("2022-07-13", "D").astype(int)
    same = np.datetime64("2022-07-12", "D").astype(int)
    old = np.arange(np.datetime64("2022-06-01", "D").astype(int),
                    np.datetime64("2022-06-21", "D").astype(int))
    assert local_hour_for_member(t, np.r_[old, y], "CA", lc)[0].tolist() == [16]
    assert local_hour_for_member(t, np.r_[old, nxt], "CA", lc)[0].tolist() == [13]
    assert local_hour_for_member(t, np.r_[old, same], "CA", lc)[0].tolist() == [13]
    assert local_hour_for_member(t, np.array([same]), "CA", lc)[0].tolist() == [16]


def test_embargo_lags_exclude_equal_second_and_uncertified_closure():
    d = pd.DataFrame({"thread_id": [1, 2, 3, 4], "screen_name": ["a"] * 4,
                      "t": [1000, 8200, 8200, 18000], "t_end": [1000, 8200, 8200, 18000],
                      "ti7": [.2, .4, .6, .8]})
    x = embargo_lags(d, seconds=7200).set_index("thread_id")
    assert np.isnan(x.loc[2, "prev_ti7"])  # exact closure boundary excluded
    assert np.isnan(x.loc[3, "prev_ti7"])  # tied arrival excluded
    assert x.loc[4, "prev_ti7"] == pytest.approx(.6)
    z = embargo_lags(d, seconds=0).set_index("thread_id")
    assert z.loc[2, "prev_ti7"] == pytest.approx(.2)
    assert z.loc[3, "prev_ti7"] == pytest.approx(.2)


def test_prior_day_vote_rule_never_uses_future_or_same_day_vote():
    day = np.array([np.datetime64("2022-07-12", "D").astype(int)])
    yesterday = np.datetime64("2022-07-11", "D").astype(int)
    tomorrow = np.datetime64("2022-07-13", "D").astype(int)
    assert prior_vote_dc(day, np.array([yesterday]), min_votes=1).tolist() == [True]
    assert prior_vote_dc(day, np.array([tomorrow]), min_votes=1).tolist() == [False]
    assert prior_vote_dc(day, np.array([day[0]]), min_votes=1).tolist() == [False]
    assert prior_vote_dc(day, np.array([yesterday]), min_votes=20).tolist() == [False]


def test_training_embargo_excludes_exact_7200_second_boundary():
    o = pd.Timestamp("2022-07-01")
    base = int(o.timestamp())
    d = pd.DataFrame({"thread_id": [1, 2, 3], "acct": [1, 1, 1],
                      "when": [o - pd.Timedelta(days=1)] * 2 + [o],
                      "t_end": [base - 7201, base - 7200, base + 10]})
    tr, te = select_with_cutoff(d, o, o + pd.offsets.MonthBegin(1), min_train=1, embargo=7200)
    assert tr.thread_id.tolist() == [1] and te.thread_id.tolist() == [3]


def test_ridge_comparison_is_id_aligned_and_account_paired():
    g = pd.DataFrame({"thread_id": [10, 11, 12], "acct": [1, 1, 2],
                      "ti7": [0., 0., 0.], "safe_p": [1., 1., 2.],
                      "safe_r": [.5, .5, 1.]})
    ridge = pd.DataFrame({"thread_id": [12, 10, 11], "acct": [2, 1, 1],
                          "ti7": [0., 0., 0.], "content": [2., 1., 1.],
                          "rhythm": [1.5, .75, .75]})
    x = compare_with_ridge(g, ridge, boot=100)
    assert x["n"] == 3 and x["accounts"] == 2
    assert x["gbm_skill"] == pytest.approx(.75)
    assert x["ridge_skill"] == pytest.approx(1 - (2 * .75**2 + 1.5**2) / 6)
    assert x["gap_gbm_minus_ridge"] == pytest.approx(x["gbm_skill"] - x["ridge_skill"])
    assert len(x["ci95_gap_gbm_minus_ridge"]) == 2
    with pytest.raises(AssertionError, match="row IDs"):
        compare_with_ridge(g, ridge.iloc[:2])
    with pytest.raises(AssertionError, match="label"):
        compare_with_ridge(g, ridge.assign(ti7=[1., 1., 1.]))


def test_cli_rejects_unrecognized_stage():
    import subprocess
    import sys
    from pathlib import Path
    p = subprocess.run([sys.executable, str(Path(__file__).with_name("run_gbt.py")),
                        "--stage", "not-a-stage"], capture_output=True, text=True)
    assert p.returncode != 0
    assert "invalid choice" in p.stderr


def test_report_distinguishes_full_original_from_matched_asof():
    full = pd.DataFrame({"thread_id": [1, 2, 3], "acct": [1, 1, 2],
                         "ti7": [0., 0., 0.], "orig_p": [1., 1., 2.],
                         "orig_r": [.5, .5, 1.]})
    matched = full.iloc[:2].copy()
    matched["cut_p"] = [1., 1.]
    matched["cut_r"] = [.5, .5]
    matched["safe_p"] = [.8, .8]
    matched["safe_r"] = [.4, .4]
    result = summarize_predictions(full, matched, boot=50)
    assert result["original_full"]["n"] == 3
    assert result["original_full"]["skill"] == pytest.approx(.75)
    assert result["matched"]["n"] == 2
    assert result["matched"]["asof"]["skill"] == pytest.approx(.75)
    assert result["matched"]["paired_skill_change"]["asof_minus_original"] == pytest.approx(0.)


def test_matched_first_month_uses_exact_original_test_rows_and_completed_train():
    import json
    root = __import__("run_gbt").ROOT
    p1 = root / "output/v7/p1"
    rq1 = root / "manuscript/epjds_submission_work/rq1"
    f = pd.read_parquet(p1 / "features.parquet")
    a = pd.read_parquet(rq1 / "arrival_features.parquet",
                        columns=["thread_id", "t_end", "prev_ti7", "roll10", "roll50",
                                 "naive_prev_ti7", "naive_roll10", "naive_roll50"])
    naive, safe = attach_arrival_lags(f, a)
    o = pd.Timestamp("2022-07-01")
    ref = pd.read_parquet(p1 / "preds_test/origin_2022-07.parquet",
                          columns=["thread_id", "acct", "when", "ti7", "gbm_reg_P", "gbm_reg_R"])
    ref = ref.rename(columns={"gbm_reg_P": "orig_p", "gbm_reg_R": "orig_r"})
    rows, diag = matched_month(naive, safe, o, json.loads((p1 / "hp.json").read_text()), ref)
    assert len(rows) == diag["n_test"] <= len(ref)
    assert rows.thread_id.isin(ref.thread_id).all()
    assert rows.thread_id.is_unique
    assert diag["n_train"] <= 2392594
    assert diag["n_train"] > 2300000
    assert diag["n_changed_test_lags"] >= 0
    assert np.isfinite(rows[["orig_p", "orig_r", "cut_p", "cut_r", "safe_p", "safe_r"]]).all().all()


def test_original_first_month_refit_matches_persisted_full_corpus_predictions():
    from pathlib import Path
    import json
    root = __import__("run_gbt").ROOT
    out = root / "output/v7/p1"
    frame = pd.read_parquet(out / "features.parquet")
    frame["when"] = pd.to_datetime(frame["when"])
    hp = json.loads((out / "hp.json").read_text())
    archived = pd.read_parquet(out / "preds_test/origin_2022-07.parquet",
                               columns=["thread_id", "ti7", "gbm_reg_P", "gbm_reg_R"])
    rows, diag = original_month(frame, pd.Timestamp("2022-07-01"), hp, archived)
    assert diag["n_train"] == 2392594
    assert diag["archive"]["n"] == len(rows) == 40945
    assert diag["archive"]["max_abs_prediction_error"] < 2e-6


def test_archive_validation_requires_identical_rows_labels_and_near_exact_predictions():
    archived = pd.DataFrame({"thread_id": [3, 5], "ti7": [.2, .3],
                             "gbm_reg_P": [.1, .2], "gbm_reg_R": [.2, .3]})
    current = pd.DataFrame({"thread_id": [3, 5], "ti7": [.2, .3],
                            "orig_p": [.1, .2], "orig_r": [.2, .3]})
    assert verify_archive(current, archived)["max_abs_prediction_error"] == 0
    with pytest.raises(AssertionError, match="original rows"):
        verify_archive(current.iloc[::-1], archived)
    with pytest.raises(AssertionError, match="prediction"):
        verify_archive(current.assign(orig_r=[.8, .9]), archived)
    with pytest.raises(AssertionError, match="labels"):
        verify_archive(current.assign(ti7=[.7, .8]), archived)


def test_fit_p_r_uses_exact_original_gbm_recipe_and_frozen_hyperparameters():
    m = __import__("run_gbt").original_models()
    rng = np.random.default_rng(11)
    cols = list(dict.fromkeys(m.SETS["R"]))
    tr = pd.DataFrame(rng.normal(size=(120, len(cols))), columns=cols)
    te = pd.DataFrame(rng.normal(size=(8, len(cols))), columns=cols)
    tr["ti7"] = .5 + .1 * tr["prev_ti7"] + .2 * tr["log_dt"]
    hp = {f"reg_{s}": {"leaves": 3, "min_leaf": 2, "rounds": 7} for s in ("P", "R")}
    actual = fit_p_r(tr, te, hp)
    for s in ("P", "R"):
        expected = m.fit_gbm(tr, "reg", s, hp).predict(te[m.SETS[s]])
        np.testing.assert_allclose(actual[s], expected, rtol=0, atol=1e-12)
    assert np.isfinite(np.column_stack(list(actual.values()))).all()


def test_matched_origin_shares_training_labels_encoding_and_test_ids():
    o = pd.Timestamp("2022-07-01")
    naive = pd.DataFrame({"thread_id": list(range(11)), "acct": [4] * 11,
                          "when": [o - pd.Timedelta(days=12 - i) for i in range(10)] + [o],
                          "t_end": [int((o - pd.Timedelta(days=11 - i)).timestamp()) for i in range(9)]
                                   + [int(o.timestamp()), int((o + pd.Timedelta(days=2)).timestamp())],
                          "ti7": np.linspace(.2, .7, 11),
                          "prev_ti7": [.1] * 11, "roll10": [.2] * 11, "roll50": [.3] * 11})
    safe = naive.copy()
    safe[["prev_ti7", "roll10", "roll50"]] += .1
    tr, te, tr_safe, te_safe = prepare_matched(naive, safe, o, o + pd.offsets.MonthBegin(1), min_train=5)
    assert tr.thread_id.tolist() == list(range(9))
    assert te.thread_id.tolist() == [10]
    assert tr.t_end.max() < int(o.timestamp())
    for col in ("thread_id", "ti7", "te_mean", "te_sd", "ccode", "thr_a"):
        assert tr[col].equals(tr_safe[col]) and te[col].equals(te_safe[col])
    assert tr_safe.prev_ti7.iloc[0] == pytest.approx(tr.prev_ti7.iloc[0] + .1)
    assert te_safe.prev_ti7.iloc[0] == pytest.approx(te.prev_ti7.iloc[0] + .1)
    naive.loc[8, "t_end"] = int(o.timestamp()) - 3600
    safe.loc[8, "t_end"] = int(o.timestamp()) - 3600
    embargo_tr, _, _, _ = prepare_matched(naive, safe, o, o + pd.offsets.MonthBegin(1),
                                          min_train=5, embargo=7200)
    assert 8 not in embargo_tr.thread_id.tolist()


def test_paired_summary_uses_row_weighted_mse_and_account_block_ci():
    d = pd.DataFrame({"acct": [1, 1, 2], "ti7": [0., 0., 0.],
                      "orig_p": [1., 1., 2.], "orig_r": [.5, .5, 1.],
                      "cut_p": [1., 1., 2.], "cut_r": [0., 0., 1.],
                      "safe_p": [0., 0., 1.], "safe_r": [0., 0., .5]})
    result = paired_summary(d, boot=300, seed=7)
    assert result["n"] == 3 and result["accounts"] == 2
    assert result["original"]["mse_p"] == pytest.approx(2.)
    assert result["original"]["skill"] == pytest.approx(.75)
    assert result["asof"]["mse_r"] == pytest.approx(1. / 12.)
    assert result["paired_skill_change"]["asof_minus_original"] == pytest.approx(0.)
    assert result["paired_mse_difference"]["asof_r_minus_original_r"] == pytest.approx(-5. / 12.)
    for key in ("asof_minus_original", "asof_minus_cutoff"):
        assert len(result["ci95_paired_skill_change"][key]) == 2
        assert np.isfinite(result["ci95_paired_skill_change"][key]).all()
    assert paired_summary(d, boot=300, seed=7) == result


def test_arrival_lags_change_only_three_persistence_columns_and_refuse_misalignment():
    original = pd.DataFrame({"thread_id": [1, 2], "ti7": [.3, .4],
                             "prev_ti7": [.2, .5], "roll10": [.3, .6], "roll50": [.4, .7],
                             "log_dt": [2., 3.]})
    arrival = pd.DataFrame({"thread_id": [1, 2], "t_end": [100, 200],
                            "prev_ti7": [.1, .2], "roll10": [.2, .3], "roll50": [.3, .4],
                            "naive_prev_ti7": [.2, .5], "naive_roll10": [.3, .6],
                            "naive_roll50": [.4, .7]})
    a, b = attach_arrival_lags(original, arrival)
    assert a.t_end.tolist() == [100, 200]
    assert b[["prev_ti7", "roll10", "roll50"]].iloc[0].tolist() == [.1, .2, .3]
    assert b.ti7.equals(original.ti7) and b.log_dt.equals(original.log_dt)
    with pytest.raises(AssertionError, match="alignment"):
        attach_arrival_lags(original, arrival.iloc[::-1].reset_index(drop=True))


def test_completed_cutoff_is_strict_but_keeps_arrival_test_rows():
    o = pd.Timestamp("2022-07-01")
    d = pd.DataFrame({
        "thread_id": [1, 2, 3, 4, 5, 6, 7],
        "acct": [10, 10, 10, 10, 10, 20, 20],
        "when": [o - pd.Timedelta(days=4), o - pd.Timedelta(days=3),
                 o - pd.Timedelta(days=2), o - pd.Timedelta(days=1),
                 o, o - pd.Timedelta(days=2), o],
        "t_end": [int((o - pd.Timedelta(days=3)).timestamp()),
                  int((o - pd.Timedelta(days=2)).timestamp()),
                  int(o.timestamp()), int((o + pd.Timedelta(days=1)).timestamp()),
                  int((o + pd.Timedelta(days=2)).timestamp()),
                  int((o - pd.Timedelta(days=1)).timestamp()),
                  int((o + pd.Timedelta(days=1)).timestamp())],
    })
    tr, te = select_with_cutoff(d, o, o + pd.offsets.MonthBegin(1), min_train=2)
    assert tr.thread_id.tolist() == [1, 2]
    assert te.thread_id.tolist() == [5]
    assert tr.t_end.max() < int(o.timestamp())
