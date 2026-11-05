import numpy as np
import pandas as pd
import pytest

from rq1_arrival import repair_lags, select_origin, fit_pair, assemble_frame, score_events, metric_summary, previous_start_inflight


def test_inflight_thread_is_excluded_from_completed_persistence_at_arrival():
    events = pd.DataFrame({
        "thread_id": [1, 2, 3, 4, 5],
        "screen_name": ["alice"] * 5,
        "t": [10, 20, 30, 50, 50],
        "t_end": [40, 20, 30, 50, 50],
        "ti7": [0.9, 0.2, 0.4, 0.6, 0.8],
        "size": [3, 1, 1, 1, 1],
    })
    got = repair_lags(events)
    by_id = got.set_index("thread_id")
    assert by_id.loc[2, "n_completed_prior"] == 0
    assert np.isnan(by_id.loc[2, "prev_ti7"])
    assert by_id.loc[2, "prior_inflight"] == 1
    assert by_id.loc[3, "n_completed_prior"] == 1
    assert by_id.loc[3, "prev_ti7"] == pytest.approx(0.2)
    assert by_id.loc[3, "roll10"] == pytest.approx(0.2)
    assert by_id.loc[4, "n_completed_prior"] == 3
    assert by_id.loc[4, "prev_ti7"] == pytest.approx(0.9)
    assert by_id.loc[4, "roll10"] == pytest.approx(0.5)
    assert by_id.loc[5, "n_completed_prior"] == 3  # equal timestamp is not observed
    assert by_id.loc[5, "prev_ti7"] == pytest.approx(0.9)


def test_origin_training_labels_must_have_completed_before_origin():
    rows = pd.DataFrame({"thread_id": [1, 2, 3, 4, 5], "acct": [7] * 5,
                         "t": [5, 6, 9, 10, 11], "t_end": [5, 20, 9, 10, 11],
                         "n_completed_prior": [10] * 5})
    tr, te = select_origin(rows, origin=10, end=12, min_train=2)
    assert tr.thread_id.tolist() == [1, 3]
    assert te.thread_id.tolist() == [4, 5]
    assert not set(tr.thread_id).intersection(te.thread_id)
    assert tr.t_end.max() < 10


def test_origin_excludes_accounts_without_enough_completed_training_labels():
    rows = pd.DataFrame({"thread_id": [1, 2, 3, 4], "acct": [7, 7, 8, 8],
                         "t": [5, 10, 5, 10], "t_end": [5, 10, 5, 10],
                         "n_completed_prior": [10] * 4})
    tr, te = select_origin(rows, origin=10, end=12, min_train=2)
    assert tr.empty and te.empty


def test_pair_uses_identical_test_rows_and_never_reads_test_labels():
    tr = pd.DataFrame({"acct": [1] * 12 + [2] * 12,
                       "prev_ti7": np.arange(24) / 24,
                       "log_dt": np.arange(24) % 3,
                       "ti7": np.linspace(0.1, 0.9, 24)})
    te = pd.DataFrame({"acct": [1, 2, 1], "prev_ti7": [.1, .3, .8],
                       "log_dt": [1, 2, 0], "ti7": [.2, .4, .6]})
    a = fit_pair(tr, te, ["prev_ti7"], ["log_dt"])
    b = fit_pair(tr, te.assign(ti7=[1e10, -1e10, 3e10]), ["prev_ti7"], ["log_dt"])
    assert set(a) == {"content", "rhythm"}
    assert all(len(p) == 3 and np.isfinite(p).all() for p in a.values())
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])


def test_assemble_preserves_naive_lags_for_audit_and_repairs_inputs():
    events = pd.DataFrame({"thread_id": [1, 2, 3], "screen_name": ["x"] * 3,
                           "t": [10, 20, 30], "t_end": [40, 20, 30],
                           "ti7": [.9, .2, .4], "size": [2, 1, 1]})
    original = pd.DataFrame({"thread_id": [2, 3], "screen_name": ["x"] * 2,
                             "t": [20, 30], "ti7": [.2, .4], "size": [1, 1],
                             "prev_ti7": [.9, .2], "roll10": [.9, .55],
                             "roll50": [.9, .55], "prev_multi": [1., 0.]})
    got = assemble_frame(events, original).set_index("thread_id")
    assert got.loc[2, "naive_prev_ti7"] == pytest.approx(.9)
    assert np.isnan(got.loc[2, "prev_ti7"])
    assert got.loc[3, "prev_ti7"] == pytest.approx(.2)
    assert got.loc[3, "naive_roll10"] == pytest.approx(.55)
    assert got.loc[3, "roll10"] == pytest.approx(.2)
    assert np.isnan(got.loc[2, "prev_multi"])
    assert got.loc[2, "prior_inflight"] == 1


def test_score_events_is_eventual_mean_over_all_scored_thread_posts():
    events = pd.DataFrame({"thread_id": [1, 2], "t": [10, 20]})
    posts = pd.DataFrame({"id": [11, 12, 21], "thread_id": [1, 1, 2]})
    probs = pd.DataFrame({"id": [11, 12, 21], "p_a": [.2, .8, .6], "p_b": [.4, .6, 1.]})
    got = score_events(events, posts, probs, ["p_a", "p_b"]).set_index("thread_id")
    assert got.loc[1, "ti7"] == pytest.approx(.5)
    assert got.loc[2, "ti7"] == pytest.approx(.8)
    assert got.loc[1, "t"] == 10


def test_duplicate_probability_ids_follow_p1_first_occurrence_policy():
    events = pd.DataFrame({"thread_id": [1]})
    posts = pd.DataFrame({"id": [11], "thread_id": [1]})
    probs = pd.DataFrame({"id": [11, 11], "p_a": [.2, .9], "p_b": [.4, .9]})
    got = score_events(events, posts, probs, ["p_a", "p_b"])
    assert got.ti7.iloc[0] == pytest.approx(.3)


def test_metric_summary_clusters_by_account_and_uses_identical_rows():
    d = pd.DataFrame({"acct": [1, 1, 2, 2], "ti7": [1., 0., 1., 0.],
                      "content": [.5] * 4, "rhythm": [1., 0., 1., 0.]})
    s = metric_summary(d, boot=100, seed=42)
    assert s["n"] == 4 and s["accounts"] == 2
    assert s["mse_content"] == pytest.approx(.25)
    assert s["mse_rhythm"] == pytest.approx(0.)
    assert s["delta_mse"] == pytest.approx(.25)
    assert s["ci95_delta_mse"] == pytest.approx([.25, .25])


def test_previous_start_unavailable_excludes_same_time_arrivals():
    events = pd.DataFrame({"thread_id": [1, 2, 3, 4], "screen_name": ["x"] * 4,
                           "t": [10, 20, 20, 30], "t_end": [25, 21, 20, 30]})
    got = previous_start_inflight(events).set_index("thread_id")
    assert got.loc[2, "naive_prev_unavailable"] == 1
    assert got.loc[3, "naive_prev_unavailable"] == 1  # last strictly earlier start is 10
    assert got.loc[4, "naive_prev_unavailable"] == 0
    assert got.loc[2, "prev_start_thread_id"] == 1
    assert got.loc[3, "prev_start_thread_id"] == 1
    assert got.loc[3, "prev_start_end"] == 25


def test_manifest_lists_reproducibility_readme_and_execution_logs(tmp_path, monkeypatch):
    import json
    import run_rq1
    monkeypatch.setattr(run_rq1, "HERE", tmp_path)
    (tmp_path / "README.md").write_text("methods")
    (tmp_path / "audit.log").write_text("real run")
    (tmp_path / "eval.log").write_text("real evaluation")
    run_rq1.manifest()
    files = json.loads((tmp_path / "manifest.json").read_text())["files"]
    assert {"README.md", "audit.log", "eval.log"} <= files.keys()
