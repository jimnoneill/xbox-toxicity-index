import pandas as pd
import pytest
from rq3_forecast import slice_posts


def test_update_target_excludes_every_preorigin_post_and_preserves_boundary():
    posts = pd.DataFrame({"bioguide": ["X"] * 4,
                          "created_at": pd.to_datetime(["2022-07-01 04:00Z", "2022-12-08 04:59Z",
                                                        "2022-12-08 05:00Z", "2023-07-01 03:59Z"], utc=True),
                          "ti7": [0.1, 0.2, 0.8, 0.6]})
    s = slice_posts(posts, 2022)
    row = s.iloc[0]
    assert row.n_pre == 2 and row.n_future == 2
    assert row.mean_pre == pytest.approx(0.15)
    assert row.mean_future == pytest.approx(0.7)
    assert row.mean_full == pytest.approx(0.425)


def test_status_projection_is_asof_not_realized_georgia_runoff():
    from rq3_forecast import status_flags
    # 2020 Georgia Senate runoff was still undecided at the Dec 8 origin.
    assert status_flags(2020, "D", "senate", "D", "D", "R", "update")["majority_after_jan"] == 0
    assert status_flags(2022, "R", "house", "D", "D", "D", "update")["majority_after_jan"] == 1
    assert status_flags(2022, "R", "house", "D", "D", "D", "july")["majority_after_jan"] == 0


def test_assembly_uses_previous_py_and_future_only_target_on_matched_cohort():
    from rq3_forecast import assemble
    panel = pd.DataFrame({"bioguide": ["X", "X", "Y", "Y"], "py": [2021, 2022, 2021, 2022],
        "party": ["D"] * 4, "chamber": ["house"] * 4, "ti7": [.3, .9, .4, .8],
        "n_events": [100] * 4, "in_office_days": [300] * 4, "n_scored": [100] * 4})
    slices = pd.DataFrame({"bioguide": ["X", "Y"], "py": [2022] * 2,
        "n_full": [110, 110], "n_pre": [50, 50], "n_future": [60, 60],
        "mean_full": [.6, .7], "mean_pre": [.2, .5], "mean_future": [.7, .8]})
    national = pd.DataFrame({"py": [2022], "president_party": ["D"],
                             "house_majority": ["D"], "senate_majority": ["D"]})
    rows = assemble(panel, slices, national)
    assert len(rows) == 4
    assert set(rows.phase) == {"july", "update"}
    xj = rows[(rows.bioguide == "X") & (rows.phase == "july")].iloc[0]
    xu = rows[(rows.bioguide == "X") & (rows.phase == "update")].iloc[0]
    assert xj.level == pytest.approx(.3) and xj.y == pytest.approx(.6)
    assert xu.level == pytest.approx(.2) and xu.y == pytest.approx(.7)
    assert xu.n_target == 60 and xj.n_target == 110
    assert xu.origin_date == "2022-12-08" and xj.origin_date == "2022-07-01"


def test_forecast_is_rolling_and_matched_even_if_test_labels_are_replaced():
    from rq3_forecast import forecast
    r = pd.DataFrame([dict(py=y, phase=p, bioguide=str(i), party="D" if i % 2 else "R",
            chamber="house", party_R=i % 2, chamber_senate=0,
            level=.1 * i, prior_ti7=.1 * i, early_change=.02 * i,
            aligned_now=i % 2, majority_now=0, aligned_after_jan=0,
            majority_after_jan=i % 2, y=.1 * i + .01 * (y - 2017),
            n_target=100, origin_date=f"{y}-12-08" if p == "update" else f"{y}-07-01")
            for y in [2018, 2019, 2020] for p in ["july", "update"] for i in range(1, 9)])
    original = forecast(r)
    r.loc[r.py == 2020, "y"] = -1234
    tampered = forecast(r)
    a = original[original.py == 2020].sort_values(["phase", "bioguide", "model"]).reset_index(drop=True)
    b = tampered[tampered.py == 2020].sort_values(["phase", "bioguide", "model"]).reset_index(drop=True)
    pd.testing.assert_series_equal(a.pred, b.pred)
    assert set(a.model) == {"persistence", "party_chamber_change", "ridge_free", "ridge_status"}
    assert a.groupby(["phase", "model"]).size().nunique() == 1
    assert set(a.train_years) == {"2018;2019"}


def test_score_paired_models_and_member_cluster_exact_zero_for_identical_forecasts():
    from rq3_forecast import score
    p = pd.DataFrame([dict(py=year, phase="update", bioguide=i, y=float(i), pred=float(i) + .1,
                            model=m, n_target=50) for year in [2019, 2020] for i in [1, 2, 3]
                      for m in ["persistence", "ridge_free", "ridge_status", "party_chamber_change"]])
    s = score(p, B=100)
    t = s[(s.model == "ridge_status") & (s.baseline == "ridge_free") & (s.set == "development")].iloc[0]
    assert t.n == 6 and t.members == 3
    assert t.skill == pytest.approx(0) and t.lo == pytest.approx(0) and t.hi == pytest.approx(0)


def test_preparation_only_original_mapped_posts_and_seven_components():
    from rq3_forecast import prepare_posts
    d = pd.DataFrame({"screen_name": ["a", "a", "unknown"], "post_type": ["original", "retweet", "original"],
                      "created_at": pd.to_datetime(["2022-12-08 05:00Z"] * 3, utc=True),
                      **{c: [1., 0., 0.] for c in ["p_offensive", "p_toxic", "p_neg_sentiment",
                          "p_irony", "p_ragebait", "p_tribal", "p_performative"]}})
    amap = pd.DataFrame({"screen_name": ["a"], "bioguide": ["X"]})
    r = prepare_posts(d, amap)
    assert r[["bioguide", "ti7"]].to_dict("records") == [{"bioguide": "X", "ti7": 1.}]


def test_panel_crosscheck_rejects_inconsistent_full_year_target():
    from rq3_forecast import check_panel
    panel = pd.DataFrame({"bioguide": ["X"], "py": [2022], "ti7": [.3], "n_scored": [100]})
    slices = pd.DataFrame({"bioguide": ["X"], "py": [2022], "mean_full": [.4], "n_full": [100]})
    with pytest.raises(AssertionError, match="panel"):
        check_panel(panel, slices)
    slices["mean_full"] = .3
    check_panel(panel, slices)


def test_real_data_loader_crosschecks_panel_with_original_posts():
    from pathlib import Path
    from rq3_forecast import load_real, check_panel
    root = Path.home() / "x_box/pipeline"
    panel, slices, national, audit = load_real(root)
    assert check_panel(panel, slices) > 1000
    assert set(slices.py) == set(range(2018, 2023))
    assert audit["original_posts_mapped"] > 1000000
    assert len(national) == 5


def test_carried_july_forecast_is_scored_only_against_future_remainder():
    from rq3_forecast import carry_july
    p = pd.DataFrame([dict(py=2022, phase=phase, bioguide="X", model=model, pred=value,
                            y=.8 if phase == "update" else .5, n_target=80)
                      for phase, model, value in [("july", "ridge_free", .4),
                                                  ("update", "ridge_free", .6),
                                                  ("july", "ridge_status", .3),
                                                  ("update", "ridge_status", .7)]])
    c = carry_july(p)
    assert len(c) == 2 and (c.y == .8).all()
    assert dict(zip(c.model, c.pred)) == {"july_carry_free": .4, "july_carry_status": .3}


def test_real_pipeline_outputs_labeled_reused_year_and_future_only_remainder():
    from pathlib import Path
    import json
    from rq3_forecast import run
    out = Path(__file__).parent
    run(Path.home() / "x_box/pipeline", out)
    p = pd.read_csv(out / "predictions.csv")
    s = pd.read_csv(out / "skill.csv")
    m = json.loads((out / "manifest.json").read_text())
    carry = pd.read_csv(out / "carry_skill.csv")
    results = json.loads((out / "results.json").read_text())
    assert len(carry) == 4 and len(results["primary_skill"]) == 16
    assert set(p.py) == {2019, 2020, 2021, 2022}
    assert set(p.phase) == {"july", "update"}
    assert set(s.set) == {"development", "reused_final_exploratory"}
    assert m["final_year_status"] == "reused_final_exploratory_not_untouched"
    assert m["update_target_start"] == "December 8, 00:00 America/New_York"
    assert (p.groupby(["phase", "py", "model"]).size().groupby(level=[0, 1]).nunique() == 1).all()


def test_update_vs_july_carry_uses_same_future_targets_and_paired_members():
    from rq3_forecast import carry_score
    p = pd.DataFrame([dict(py=2022, phase=phase, bioguide=i, model=model, pred=val,
                            y=.8 if phase == "update" else .3, n_target=80)
                      for i in ["X", "Y"] for phase, model, val in [
                        ("july", "ridge_free", .3), ("update", "ridge_free", .7),
                        ("july", "ridge_status", .3), ("update", "ridge_status", .7)]])
    s = carry_score(p, B=100)
    assert len(s) == 2 and (s.n == 2).all()
    assert s.skill.tolist() == pytest.approx([.8, .8])
    assert set(s.set) == {"reused_final_exploratory"}
