import sys

import pandas as pd
import pytest

sys.path.insert(0, "/home/joneill/x_box/pipeline/xbox")
import kistner_ext as K
import asof_audit as A


def example():
    rows = [(2019, "D", "House"), (2020, "D", "Senate"),
            (2021, "D", "House"), (2021, "D", "Senate"),
            (2021, "R", "House"), (2022, "D", "Senate")]
    t = pd.DataFrame(rows, columns=["Y", "party", "chamber"])
    t["target"] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    for y, g in t.groupby("Y"):
        t.loc[g.index, K.POWER] = K.power_features(y, g.party.values, g.chamber.values)[K.POWER].values
    return t


def test_december_2020_uses_republican_2021_senate_and_preserves_targets():
    original = example()
    before = A.with_asof_power(original, "2020-12-31")
    after = A.with_asof_power(original, "2021-01-20")
    assert before.loc[2, "p_oth"] == 0
    assert after.loc[2, "p_oth"] == 1
    assert before.loc[3, "p_ch"] == 0
    assert after.loc[3, "p_ch"] == 1
    assert before.loc[4, "p_oth"] == 1
    assert after.loc[4, "p_oth"] == 0
    assert before.loc[3, "p_dch"] == 0
    assert after.loc[3, "p_dch"] == 1
    pd.testing.assert_frame_equal(before[["target"]], after[["target"]])
    pd.testing.assert_frame_equal(after, original)
    pd.testing.assert_frame_equal(before.loc[original.Y != 2021], original.loc[original.Y != 2021])


def test_unresolved_runoff_date_is_rejected():
    with pytest.raises(ValueError, match="2021"):
        A.with_asof_power(example(), "2021-01-10")


def test_overlap_counts_distinct_members_and_future_selected_rows():
    t = pd.DataFrame({"MemberICPSR": [1, 2, 2, 2, 3, 3],
                      "Y": [2017, 2018, 2019, 2020, 2019, 2020]})
    panel = pd.DataFrame({"MemberICPSR": [1, 2, 3, 4, 2, 3],
                          "Year": [2018, 2018, 2018, 2018, 2019, 2019],
                          "n": [50, 50, 50, 49, 50, 49]})
    out = A.overlap_diagnostics(t, panel, train_end=2018)
    assert out["train_members"] == 2
    assert out["test_members"] == 2
    assert out["members_seen_in_both"] == 1
    assert out["test_rows_seen_member"] == 2
    assert out["test_rows_total"] == 4
    assert out["by_year"]["2019"]["prior_year_eligible_members"] == 3
    assert out["by_year"]["2019"]["scored_members"] == 2
    assert out["by_year"]["2019"]["unscored_prior_eligible"] == 1


def test_frozen_replay_reproduces_final_predictions_and_recode_only_affects_2021():
    import json
    import numpy as np

    base = "/home/joneill/x_box/pipeline"
    t = pd.read_csv(f"{base}/data/kistner/transitions_tw.csv")
    frozen = json.load(open(f"{base}/output/v7/kistner/frozen.json"))["tw"]
    original = pd.read_csv(f"{base}/output/v7/kistner/preds_final_tw.csv")
    scenarios = A.fit_scenarios(t, frozen)
    old = scenarios["2021-01-20"]
    pre = scenarios["2020-12-31"]
    for name in ["RTM", *K.MODELS]:
        np.testing.assert_allclose(old[name], original[name].values, rtol=1e-8, atol=1e-9)
    for name in ("RTM", "R_ctrl", "G_ctrl"):
        np.testing.assert_allclose(pre[name], old[name], rtol=0, atol=1e-12)
    mask = original.Y.ne(2021).to_numpy()
    for name in ("R_power", "R_both", "G_both"):
        np.testing.assert_allclose(pre[name][mask], old[name][mask], atol=1e-12)
        assert np.max(np.abs(pre[name][~mask] - old[name][~mask])) > 1e-8


def test_metrics_use_identical_rows_targets_and_zero_change_elsewhere():
    import numpy as np
    t = pd.DataFrame({"MemberICPSR": [1, 2, 1, 3], "Y": [2018, 2019, 2021, 2021],
                      "target": [0.0, 0.1, 0.2, -0.2]})
    names = ["RTM", *K.MODELS]
    post = {name: np.array([0.05, 0.0, 0.0]) for name in names}
    pre = {name: np.array([0.05, 0.1, -0.1]) for name in names}
    s = A.summarize(t, {"2020-12-31": pre, "2021-01-20": post}, B=50)
    assert s["scenarios"]["2020-12-31"]["by_year"]["2019"]["R_both"]["rmse"] == pytest.approx(0.05)
    assert s["scenarios"]["2021-01-20"]["by_year"]["2019"]["R_both"]["rmse"] == pytest.approx(0.05)
    assert s["scenarios"]["2020-12-31"]["by_year"]["2021"]["R_both"]["rmse"] < s["scenarios"]["2021-01-20"]["by_year"]["2021"]["R_both"]["rmse"]
    assert s["sensitivity_2021"]["R_both"]["skill_pre_vs_post"] > 0
    assert s["scenarios"]["2020-12-31"]["by_train_identity"]["seen"]["n"] == 1
    assert s["scenarios"]["2020-12-31"]["by_train_identity"]["unseen"]["n"] == 2


def test_cli_writes_complete_isolated_outputs_for_both_platforms():
    import json
    import subprocess
    import hashlib
    from pathlib import Path

    root = Path(__file__).resolve().parent
    run = subprocess.run([sys.executable, str(root / "asof_audit.py")],
                         cwd=root, text=True, capture_output=True, timeout=120)
    assert run.returncode == 0, run.stdout + run.stderr
    result = json.loads((root / "results.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    assert set(result) == {"tw", "twfb"}
    for name in result:
        assert result[name]["replay_max_abs_error"] < 1e-10
        assert result[name]["overlap"]["test_members"] > 0
        assert (root / f"predictions_{name}.csv").exists()
    report = (root / "report.md").read_text()
    assert "not independent" in report
    assert "January 5, 2021" in report
    assert "conditional" in report
    assert "2020-12-31" in report
    for name, record in manifest["outputs"].items():
        p = root / name
        assert record["sha256"] == hashlib.sha256(p.read_bytes()).hexdigest()
        assert str(p) == record["path"]


def test_identity_table_keeps_both_measures_together():
    from pathlib import Path
    report = (Path(__file__).resolve().parent / "report.md").read_text()
    lines = report.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("| tw | 642 |"))
    assert lines[start + 1].startswith("| twfb |")


def test_post_runoff_annual_target_contains_prior_messages():
    import json
    from pathlib import Path
    result = json.loads((Path(__file__).resolve().parent / "results.json").read_text())
    tw = result["tw"]["pre_jan20_2021"]
    assert tw["early_messages"] == 22716
    assert tw["early_members"] == 447
    assert tw["total_messages"] == 468178
    assert result["twfb"]["pre_jan20_2021"]["early_messages"] > tw["early_messages"]


def test_report_cites_direct_official_senate_party_history():
    from pathlib import Path
    report = (Path(__file__).resolve().parent / "report.md").read_text()
    assert "https://www.senate.gov/about/parties-leadership/party-division.htm" in report
