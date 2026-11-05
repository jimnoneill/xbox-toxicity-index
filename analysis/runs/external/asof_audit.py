"""Isolated, retrospective as-of sensitivity for the frozen Kistner panel."""
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, "/home/joneill/x_box/pipeline/xbox")
import kistner_ext as K


def with_asof_power(t, cutoff):
    """Recode only 2021 power columns; Dec 31 is pre-runoff, Jan 20 post-swearing."""
    if cutoff not in ("2020-12-31", "2021-01-20"):
        raise ValueError("2021 cutoff must be 2020-12-31 or 2021-01-20")
    out = t.copy()
    g = out.loc[out.Y == 2021]
    if g.empty:
        return out
    party = g.party.to_numpy()
    is_house = (g.chamber.to_numpy() == "House")
    is_r = (party == "R")
    senate_party = "R" if cutoff == "2020-12-31" else "D"
    pres = (party == K.PRES[2021]).astype(float)
    house = (party == K.HOUSE[2021]).astype(float)
    senate = (party == senate_party).astype(float)
    pres0 = (party == K.PRES[2020]).astype(float)
    house0 = (party == K.HOUSE[2020]).astype(float)
    senate0 = (party == K.SENATE[2020]).astype(float)
    ch = np.where(is_house, house, senate)
    oth = np.where(is_house, senate, house)
    ch0 = np.where(is_house, house0, senate0)
    oth0 = np.where(is_house, senate0, house0)
    dpres, dch, doth = pres - pres0, ch - ch0, oth - oth0
    values = pd.DataFrame({
        "p_pres": pres, "p_ch": ch, "p_oth": oth, "p_tri": pres + house + senate,
        "p_dpres": dpres, "p_dch": dch, "p_doth": doth,
        "p_dtri": dpres + (house - house0) + (senate - senate0),
        "p_dpres_xR": dpres * is_r, "p_dch_xR": dch * is_r,
        "p_lost_any": ((dpres < 0) | (dch < 0) | (doth < 0)).astype(float),
        "p_gain_any": ((dpres > 0) | (dch > 0) | (doth > 0)).astype(float),
    }, index=g.index)
    out.loc[g.index, K.POWER] = values[K.POWER].to_numpy()
    return out


def overlap_diagnostics(t, panel, train_end=2018):
    """Member identity overlap and retrospective inclusion among prior-year eligible IDs."""
    train = t[t.Y <= train_end]
    test = t[t.Y > train_end]
    train_ids = set(train.MemberICPSR)
    test_ids = set(test.MemberICPSR)
    by_year = {}
    for year, rows in test.groupby("Y"):
        eligible_rows = panel[panel.Year.eq(year - 1).mul(panel.n.ge(K.MIN_N)).astype(bool)]
        eligible = set(eligible_rows.MemberICPSR)
        scored = set(rows.MemberICPSR)
        by_year[str(int(year))] = {
            "prior_year_eligible_members": len(eligible),
            "scored_members": len(scored),
            "unscored_prior_eligible": len(eligible.difference(scored)),
            "scored_seen_in_training": len(scored.intersection(train_ids)),
            "scored_new_to_training": len(scored.difference(train_ids)),
        }
    return {
        "train_rows_total": len(train), "test_rows_total": len(test),
        "train_members": len(train_ids), "test_members": len(test_ids),
        "members_seen_in_both": len(train_ids.intersection(test_ids)),
        "test_members_new_to_training": len(test_ids.difference(train_ids)),
        "test_rows_seen_member": int(test.MemberICPSR.isin(train_ids).sum()),
        "test_rows_new_member": int((~test.MemberICPSR.isin(train_ids)).sum()),
        "by_year": by_year,
    }


def fit_scenarios(t, frozen):
    """Frozen 2011-18 fit; predict the identical 2019-22 target rows under two as-of states."""
    train = t[t.Y <= 2018].reset_index(drop=True)
    test = t[t.Y >= 2019].reset_index(drop=True)
    post = {"RTM": K.rtm_predict(train, test)}
    for model, (kind, set_name) in K.MODELS.items():
        post[model] = K.fit_predict(kind, K.SETS[set_name], train, test, frozen[model])
    pretest = with_asof_power(test, "2020-12-31")
    pre = {name: post[name].copy() for name in ("RTM", "R_ctrl", "G_ctrl")}
    for model in ("R_power", "R_both", "G_both"):
        kind, set_name = K.MODELS[model]
        pre[model] = K.fit_predict(kind, K.SETS[set_name], train, pretest, frozen[model])
    return {"2020-12-31": pre, "2021-01-20": post}


def summarize(t, scenarios, B=1000):
    """Same-row RMSE/skill, member-cluster intervals and paired pre/post contrast."""
    te = t[t.Y >= 2019].reset_index(drop=True)
    train_ids = set(t.loc[t.Y <= 2018, "MemberICPSR"])
    seen = te.MemberICPSR.isin(train_ids).to_numpy()
    masks = {"all": np.ones(len(te), dtype=bool),
             "seen": seen, "unseen": ~seen,
             "transitions_2019_2021": te.Y.isin([2019, 2021]).to_numpy()}
    by_year = {str(int(year)): te.Y.eq(year).to_numpy() for year in sorted(te.Y.unique())}
    names = ["RTM", *K.MODELS]

    def metrics(preds, mask):
        y = te.target.to_numpy()[mask]
        ids = te.MemberICPSR.to_numpy()[mask]
        base = K.rmse(y, np.zeros(len(y)))
        result = {}
        for name in names:
            pred = np.asarray(preds[name])[mask]
            skill, lo, hi, _ = K.boot_skill(y, pred, np.zeros(len(y)), ids, B=B)
            result[name] = {"rmse": K.rmse(y, pred), "skill_vs_persist": skill,
                            "skill_ci95_member": [lo, hi]}
        return {"n": int(len(y)), "n_members": int(len(set(ids))),
                "rmse_persist": base, "models": result}

    result = {"scenarios": {}, "sensitivity_2021": {}}
    for date, preds in scenarios.items():
        overall = metrics(preds, masks["all"])
        years = {year: metrics(preds, mask)["models"] for year, mask in by_year.items()}
        identity = {s: metrics(preds, masks[s]) for s in ("seen", "unseen")}
        result["scenarios"][date] = {
            "overall": overall, "by_year": years, "by_train_identity": identity,
            "transition_years": metrics(preds, masks["transitions_2019_2021"])}
    mask = by_year["2021"]
    y = te.target.to_numpy()[mask]
    ids = te.MemberICPSR.to_numpy()[mask]
    for name in names:
        p_pre = np.asarray(scenarios["2020-12-31"][name])[mask]
        p_post = np.asarray(scenarios["2021-01-20"][name])[mask]
        skill, lo, hi, _ = K.boot_skill(y, p_pre, p_post, ids, B=B)
        result["sensitivity_2021"][name] = {
            "n_prediction_changed": int(np.count_nonzero(np.abs(p_pre - p_post) > 1e-12)),
            "skill_pre_vs_post": skill, "ci95_member": [lo, hi],
            "delta_rmse_pre_minus_post": K.rmse(y, p_pre) - K.rmse(y, p_post)}
    return result


def _sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _render_report(results):
    rows = ["# Congressional panel timing and overlap audit", "",
            "This is a retrospective sensitivity, **not independent untouched replication**.",
            "The 2019–2022 test block was partly exposed during design through party-year descriptive rates; model scores were subsequently frozen. Both conditions here are post hoc analyses of that block.",
            "", "## Timing and estimand",
            "- The original code gives 2021 Democratic Senate control to every 2021 prediction and says that all target-year power was known after November 2020. That is false for the Senate: Georgia runoffs were **January 5, 2021**; Democratic control began **January 20, 2021** after swearing-in and the vice president tie-break. Source: https://www.senate.gov/about/parties-leadership/party-division.htm",
            "- `2020-12-31`: 2021 Senate coded Republican (carry-forward assumption before runoff, NOT a known 2021 outcome). `2021-01-20`: Democratic control (original post-runoff coding). All other features, splits, 2011–2018 training rows, frozen hyperparameters, annual targets and test membership are identical. The year-end scenario is the earlier-information **conditional** prediction, not a probability-weighted election forecast.",
            "- For 2019, 2020, and 2022, calendar-year status is known by previous December 31. For 2021 the Senate result was not. Even the post-runoff prediction occurs after some 2021 messages: the annual 2021 target already began January 1, so it is **not** a clean post-election out-of-sample annual forecast. A Jan-20-to-Dec-31 target would require a newly defined outcome and retraining, not the same-target sensitivity requested here.",
            "- Lag share/volume and message features are from year Y-1 or earlier; target-year values enter the scored cohort definition (requires >=50 target-year messages) and outcome. Scored membership is selected using future activity, so this cannot validate a deployable all-members prediction issued at the cutoff. The archive lacks a historical availability snapshot. Metrics are conditional on retrospectively scorable member-years.",
            "", "## Frozen-model performance",
            "Skill = 1 - model RMSE / persistence RMSE. CI = 95% member-cluster percentile bootstrap (1,000, seed 20261012); paired pre/post skill = 1 - pre RMSE / post RMSE (positive favors pre).",
            "", "| measure | model | all pre skill | all post skill | 2021 pre skill | 2021 post skill | 2021 paired pre/post [95% CI] |",
            "|---|---|---:|---:|---:|---:|---:|"]
    for measure, d in results.items():
        a = d["metrics"]["scenarios"]
        comp = d["metrics"]["sensitivity_2021"]
        for model in ("RTM", "R_ctrl", "R_power", "R_both", "G_ctrl", "G_both"):
            pre = a["2020-12-31"]
            post = a["2021-01-20"]
            pair = comp[model]
            ci = pair["ci95_member"]
            rows.append(f"| {measure} | {model} | {pre['overall']['models'][model]['skill_vs_persist']:.4f} | {post['overall']['models'][model]['skill_vs_persist']:.4f} | {pre['by_year']['2021'][model]['skill_vs_persist']:.4f} | {post['by_year']['2021'][model]['skill_vs_persist']:.4f} | {pair['skill_pre_vs_post']:.4f} [{ci[0]:.4f}, {ci[1]:.4f}] |")
    rows.append("\n2021 messages already dated before the January 20 post-runoff cutoff (among the scored cohorts):")
    for measure, d in results.items():
        w = d["pre_jan20_2021"]
        rows.append(f"- {measure}: {w['early_messages']:,}/{w['total_messages']:,} ({100 * w['early_messages']/w['total_messages']:.2f}%) messages; {w['early_members']}/{w['scored_members']} scored members had at least one. This is target-period exposure, not a known-feature input to the fit.")
    rows.extend(["", "## Identity overlap and future selection", "",
                 "| measure | train members | test members | both | test rows with seen member | test rows |",
                 "|---|---:|---:|---:|---:|---:|"])
    for measure, d in results.items():
        o = d["overlap"]
        rows.append(f"| {measure} | {o['train_members']} | {o['test_members']} | {o['members_seen_in_both']} | {o['test_rows_seen_member']} | {o['test_rows_total']} |")
    for measure, d in results.items():
        o = d["overlap"]
        rows.append(f"\n{measure}: prior-year >=50-message candidate pool and scored future-year members (the difference includes departures, activity falling below threshold, and other non-observation):")
        for year, y in o["by_year"].items():
            rows.append(f"- {year}: prior eligible {y['prior_year_eligible_members']}; scored {y['scored_members']}; unscored {y['unscored_prior_eligible']}; scored seen in training {y['scored_seen_in_training']}.")
        for cohort in ("seen", "unseen"):
            x = d["metrics"]["scenarios"]["2020-12-31"]["by_train_identity"][cohort]
            rows.append(f"- {measure} {cohort} identity, year-end conditional: n={x['n']}, R_both skill={x['models']['R_both']['skill_vs_persist']:.4f}, R_power skill={x['models']['R_power']['skill_vs_persist']:.4f}.")
    rows.extend(["", "## Reproduction and scope", "",
                 "Post-runoff replay predictions were compared row-by-row against the original final CSV for every model. See replay_max_abs_error in results.json. Only the isolated external directory was written. Inputs, executable environment, output SHA-256 hashes and commands are recorded in manifest.json. No model-selection rerun or inference of an untouched holdout is asserted.", ""])
    return "\n".join(rows)


def run():
    import json
    import platform
    from pathlib import Path
    import sklearn

    root = Path(__file__).resolve().parent
    base = Path("/home/joneill/x_box/pipeline")
    original_out = base / "output/v7/kistner"
    frozen = json.loads((original_out / "frozen.json").read_text())
    results = {}
    windows = pd.read_csv(root / "jan_window_2021.csv").set_index("measure")
    for name in ("tw", "twfb"):
        t = pd.read_csv(base / f"data/kistner/transitions_{name}.csv")
        panel = pd.read_csv(base / f"data/kistner/panel_{name}.csv")
        original = pd.read_csv(original_out / f"preds_final_{name}.csv")
        te = t[t.Y >= 2019].reset_index(drop=True)
        keycols = ["MemberICPSR", "Y", "party", "chamber"]
        if not te[keycols].equals(original[keycols]) or not np.allclose(te.target, original.target, atol=1e-15, rtol=0):
            raise AssertionError(f"{name}: target rows/order differ from original")
        scenarios = fit_scenarios(t, frozen[name])
        errors = {model: float(np.max(np.abs(scenarios["2021-01-20"][model] - original[model].to_numpy())))
                  for model in ("RTM", *K.MODELS)}
        if max(errors.values()) >= 1e-10:
            raise AssertionError(f"{name}: frozen replay diverges: {errors}")
        predictions = te[["MemberICPSR", "Y", "party", "chamber", "target"]].copy()
        for date, label in (("2020-12-31", "pre"), ("2021-01-20", "post")):
            for model, values in scenarios[date].items():
                predictions[f"{label}_{model}"] = values
        predictions.to_csv(root / f"predictions_{name}.csv", index=False)
        power_pre = with_asof_power(te, "2020-12-31")
        changes = {col: int(np.count_nonzero(np.abs(power_pre[col] - te[col]) > 1e-12)) for col in K.POWER}
        counts = windows.loc[name]
        results[name] = {"pre_jan20_2021": {key: int(counts[key]) for key in ("total_messages", "early_messages", "early_members", "scored_members")},
                         "replay_max_abs_error": max(errors.values()), "replay_by_model": errors,
                         "overlap": overlap_diagnostics(t, panel),
                         "power_feature_rows_changed": changes,
                         "metrics": summarize(t, scenarios)}
    (root / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    (root / "report.md").write_text(_render_report(results))
    input_files = [
        base / "xbox/kistner_ext.py", base / "scripts/v7/kistner_01_panel.py",
        base / "scripts/v7/kistner_03_dev.py", base / "scripts/v7/kistner_04_final.py",
        base / "output/v7/kistner/DESIGN.md", original_out / "frozen.json",
        base / "data/kistner/member_year_mode.csv", base / "data/kistner/raw/Scaled.Rda",
        base / "data/kistner/raw/Voteview.tab", root / "asof_audit.py", root / "test_asof_audit.py",
        root / "jan_window.R", root / "jan_window_2021.csv",
    ]
    for name in ("tw", "twfb"):
        input_files.extend([base / f"data/kistner/transitions_{name}.csv",
                            base / f"data/kistner/panel_{name}.csv",
                            original_out / f"preds_final_{name}.csv"])
    output_files = [root / "results.json", root / "report.md",
                    root / "predictions_tw.csv", root / "predictions_twfb.csv"]
    manifest = {
        "method": "same train 2011-2018, test 2019-2022, frozen.json, no test refit; 2021 senate carry-forward versus Jan20 control",
        "asof_dates": ["2020-12-31", "2021-01-20"],
        "command": f"cd {root} && PYTHONDONTWRITEBYTECODE=1 /home/joneill/x_box/venv/bin/python asof_audit.py",
        "test_command": f"cd {root} && PYTHONDONTWRITEBYTECODE=1 /home/joneill/x_box/venv/bin/python -m pytest -p no:cacheprovider -q test_asof_audit.py",
        "raw_date_aggregation_command": f"Rscript {root / 'jan_window.R'}",
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "sklearn": sklearn.__version__},
        "inputs": {str(p): {"path": str(p), "sha256": _sha256(p)} for p in input_files},
        "outputs": {p.name: {"path": str(p), "sha256": _sha256(p)} for p in output_files},
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({name: {"overlap": d["overlap"], "replay_max_abs_error": d["replay_max_abs_error"],
                             "2021_R_both_pre": d["metrics"]["scenarios"]["2020-12-31"]["by_year"]["2021"]["R_both"],
                             "2021_R_both_post": d["metrics"]["scenarios"]["2021-01-20"]["by_year"]["2021"]["R_both"]}
                      for name, d in results.items()}, indent=2))


if __name__ == "__main__":
    run()
