"""As-of political-year RQ3 forecasts. Run with the x_box virtualenv on HPCF."""
import pandas as pd

TZ = "America/New_York"


def bounds(year):
    return (pd.Timestamp(year, 7, 1, tz=TZ),
            pd.Timestamp(year, 12, 8, tz=TZ),
            pd.Timestamp(year + 1, 7, 1, tz=TZ))


def slice_posts(posts, year):
    start, update, end = bounds(year)
    t = pd.to_datetime(posts.created_at, utc=True).dt.tz_convert(TZ)
    d = posts.loc[(t >= start) & (t < end)].copy()
    d["pre"] = t.loc[d.index] < update
    pre = d[d.pre].groupby("bioguide").ti7.agg(n_pre="size", mean_pre="mean")
    fut = d[~d.pre].groupby("bioguide").ti7.agg(n_future="size", mean_future="mean")
    full = d.groupby("bioguide").ti7.agg(n_full="size", mean_full="mean")
    return full.join(pre).join(fut).reset_index()

# Elections resolved by December 8, not actual future Senate control when unresolved.
# 2018: House flips D; 2020: Biden/House D, Georgia runoffs unresolved (S R);
# 2022: House R, Senate D following the December 6 runoff.
KNOWN_JAN = {2018: ("R", "D", "R"), 2019: ("R", "D", "R"),
             2020: ("D", "D", "R"), 2021: ("D", "D", "D"),
             2022: ("D", "R", "D")}


def status_flags(year, party, chamber, president, house, senate, phase):
    if phase not in ("july", "update") or party not in ("D", "R") or chamber not in ("house", "senate"):
        raise ValueError("Unknown phase/party/chamber")
    pp, hm, sm = (president, house, senate) if phase == "july" else KNOWN_JAN[year]
    return dict(aligned_now=int(party == president),
                majority_now=int(party == (house if chamber == "house" else senate)),
                aligned_after_jan=int(party == pp),
                majority_after_jan=int(party == (hm if chamber == "house" else sm)))


def assemble(panel, slices, national):
    """Paired member/PY rows. Outcome-based quality gate is evaluation selection, never a feature."""
    prev = panel.copy()
    prev["py"] = prev.py + 1
    prior = prev.rename(columns={"ti7": "prior_ti7", "n_scored": "prior_scored",
                                 "n_events": "prior_events", "in_office_days": "prior_office"})
    target = panel[["bioguide", "py", "n_scored", "n_events", "in_office_days"]].rename(
        columns={"n_scored": "target_scored", "n_events": "target_events",
                 "in_office_days": "target_office"})
    d = prior.merge(slices, on=["bioguide", "py"], validate="one_to_one")
    d = d.merge(target, on=["bioguide", "py"], validate="one_to_one")
    d = d.merge(national, on="py", validate="many_to_one")
    d = d[(d.prior_scored >= 75) & (d.prior_events >= 50) & (d.prior_office >= 150)
          & (d.target_scored >= 75) & (d.target_events >= 50) & (d.target_office >= 150)
          & (d.n_pre >= 25) & (d.n_future >= 40) & d.party.isin(["D", "R"])
          & d.chamber.isin(["house", "senate"])].copy()
    result = []
    for phase in ("july", "update"):
        x = d.copy()
        x["phase"] = phase
        x["origin_date"] = x.py.map(lambda y: str(bounds(int(y))[0 if phase == "july" else 1].date()))
        x["level"] = x.prior_ti7 if phase == "july" else x.mean_pre
        x["y"] = x.mean_full if phase == "july" else x.mean_future
        x["n_target"] = x.n_full if phase == "july" else x.n_future
        x["party_R"] = (x.party == "R").astype(int)
        x["chamber_senate"] = (x.chamber == "senate").astype(int)
        x["early_change"] = 0.0 if phase == "july" else x.mean_pre - x.prior_ti7
        status = [status_flags(int(r.py), r.party, r.chamber, r.president_party,
                               r.house_majority, r.senate_majority, phase) for r in x.itertuples()]
        x = pd.concat([x.reset_index(drop=True), pd.DataFrame(status)], axis=1)
        result.append(x)
    return pd.concat(result, ignore_index=True)

FREE = ["level", "prior_ti7", "early_change", "party_R", "chamber_senate"]
STATUS = FREE + ["aligned_now", "majority_now", "aligned_after_jan", "majority_after_jan"]


def forecast(rows):
    """Fixed-spec ridge, fit strictly on earlier fully resolved political years."""
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    recs = []
    for phase in ("july", "update"):
        d = rows[rows.phase == phase]
        for year in sorted(d.py.unique()):
            tr, te = d[d.py < year], d[d.py == year]
            if tr.empty or te.empty:
                continue
            assert tr.py.max() < year
            shift = tr.y.to_numpy() - tr.level.to_numpy()
            group = tr.assign(shift=shift).groupby(["party", "chamber"])["shift"].mean()
            party = np.array([group.get((p, c), shift.mean()) for p, c in zip(te.party, te.chamber)])
            preds = {"persistence": te.level.to_numpy(),
                     "party_chamber_change": te.level.to_numpy() + party}
            for name, features in (("ridge_free", FREE), ("ridge_status", STATUS)):
                a, b = tr[features].astype(float), te[features].astype(float)
                med = a.median().fillna(0)
                scaler = StandardScaler().fit(a.fillna(med))
                model = Ridge(alpha=50.0).fit(scaler.transform(a.fillna(med)), shift)
                preds[name] = te.level.to_numpy() + model.predict(scaler.transform(b.fillna(med)))
            for name, p in preds.items():
                out = te[["py", "phase", "origin_date", "bioguide", "party", "chamber", "y", "n_target"]].copy()
                out["model"] = name
                out["pred"] = p
                out["train_years"] = ";".join(str(int(y)) for y in sorted(tr.py.unique()))
                recs.append(out)
    return pd.concat(recs, ignore_index=True)


def score(pred, B=1000, pairs=None, phases=("july", "update")):
    """Paired equal-row RMSE skill, member-cluster percentile bootstrap."""
    import numpy as np
    pairs = pairs or [("ridge_free", "persistence"), ("ridge_status", "persistence"),
             ("ridge_status", "ridge_free"), ("party_chamber_change", "persistence")]
    out = []
    for name, years in (("development", [2019, 2020, 2021]),
                        ("reused_final_exploratory", [2022])):
        for phase in phases:
            p = pred[pred.py.isin(years) & (pred.phase == phase)]
            if p.empty:
                continue
            wide = p.pivot(index=["py", "bioguide"], columns="model", values="pred")
            assert p.groupby(["py", "bioguide"]).y.nunique().eq(1).all()
            y = p.drop_duplicates(["py", "bioguide"]).set_index(["py", "bioguide"]).y
            assert wide.notna().all().all() and len(wide) == len(y)
            for model, baseline in pairs:
                a = ((wide[model] - y) ** 2).rename("a")
                b = ((wide[baseline] - y) ** 2).rename("b")
                sums = pd.concat([a, b], axis=1).reset_index().groupby("bioguide").agg(
                    a=("a", "sum"), b=("b", "sum"), n=("a", "size"))
                rm, rb = np.sqrt(a.mean()), np.sqrt(b.mean())
                rng = np.random.default_rng(20261031)
                weights = rng.multinomial(len(sums), [1 / len(sums)] * len(sums), size=B)
                aa, bb = weights @ sums.a.to_numpy(), weights @ sums.b.to_numpy()
                ratios = 1 - np.sqrt(aa / bb)
                out.append(dict(set=name, phase=phase, model=model, baseline=baseline,
                                n=len(wide), members=len(sums), rmse=float(rm), baseline_rmse=float(rb),
                                skill=float(1 - rm / rb), lo=float(np.percentile(ratios, 2.5)),
                                hi=float(np.percentile(ratios, 97.5)), bootstrap=B))
    return pd.DataFrame(out)

DIMS = ["p_offensive", "p_toxic", "p_neg_sentiment", "p_irony",
        "p_ragebait", "p_tribal", "p_performative"]


def prepare_posts(posts, amap):
    """Reproduce the published panel convention for original-post TI7/member map."""
    a = amap[["screen_name", "bioguide"]].drop_duplicates("screen_name")
    p = posts.loc[posts.post_type == "original", ["screen_name", "created_at"] + DIMS].copy()
    p = p.merge(a, on="screen_name", how="inner", validate="many_to_one")
    p["ti7"] = p[DIMS].mean(axis=1)
    return p[["bioguide", "created_at", "ti7"]].dropna()


def check_panel(panel, slices):
    import numpy as np
    j = panel[["bioguide", "py", "ti7", "n_scored"]].merge(
        slices[["bioguide", "py", "mean_full", "n_full"]], on=["bioguide", "py"])
    j = j[j.n_scored.notna()]
    assert len(j) > 0, "panel crosscheck has no matching rows"
    assert np.allclose(j.ti7, j.mean_full, atol=1e-7, rtol=0), "panel TI7 differs from full-year slices"
    assert (j.n_scored == j.n_full).all(), "panel scored counts differ from full-year slices"
    return int(len(j))


def load_real(root):
    """Read-only source data; every new output is written only by the caller."""
    panel = pd.read_csv(root / "output/v7/p3/panel_member_py_full.csv")
    amap = pd.read_csv(root / "output/v7/ti/account_member_map.csv",
                       usecols=["screen_name", "bioguide", "entity_type"])
    amap = amap[amap.entity_type == "member"]
    assert (amap.groupby("screen_name").bioguide.nunique() <= 1).all()
    path = root / "output/v7/full/post_probs_full.parquet"
    raw = pd.read_parquet(path, columns=["screen_name", "created_at", "post_type"] + DIMS)
    posts = prepare_posts(raw, amap)
    del raw
    slices = pd.concat([slice_posts(posts, y).assign(py=y) for y in range(2018, 2023)], ignore_index=True)
    national_source = pd.read_csv(root / "output/v7/t3/national.csv")
    national = []
    for y in range(2018, 2023):
        vals = national_source[national_source.date.isin([f"{y}-07-01", f"{y}-12-08"])]
        assert len(vals) == 2 and vals[["president_party", "house_majority", "senate_majority"]].nunique().eq(1).all()
        national.append(dict(py=y, **vals.iloc[0][["president_party", "house_majority", "senate_majority"]].to_dict()))
    audit = {"original_posts_mapped": len(posts), "panel_rows": len(panel),
             "sliced_member_years": len(slices), "panel_crosschecked_rows": check_panel(panel, slices)}
    return panel, slices, pd.DataFrame(national), audit


def carry_july(pred):
    """Secondary diagnostic: July annual-level prediction vs post-Dec-8 remainder, not a treatment effect."""
    july = pred[(pred.phase == "july") & pred.model.isin(["ridge_free", "ridge_status"])].copy()
    targets = pred[(pred.phase == "update") & (pred.model == "ridge_free")][
        ["py", "bioguide", "y", "n_target"]]
    july = july.drop(columns=["y", "n_target"]).merge(
        targets, on=["py", "bioguide"], validate="many_to_one")
    july["model"] = july.model.map({"ridge_free": "july_carry_free", "ridge_status": "july_carry_status"})
    july["phase"] = "update_remainder_carry"
    return july


def run(root, out):
    """Real-data rerun in the isolated manuscript work directory."""
    import hashlib
    import json
    from pathlib import Path
    out = Path(out).resolve()
    assert out == (root / "manuscript/epjds_submission_work/rq3").resolve(), "writes restricted to isolated rq3 directory"
    panel, slices, national, audit = load_real(root)
    rows = assemble(panel, slices, national)
    assert rows.groupby(["py", "phase"]).size().ge(50).all(), "insufficient matched cohort"
    predictions = forecast(rows)
    scores = score(predictions)
    carry = carry_july(predictions)
    carry_metrics = carry_score(predictions)
    assert len(carry) == len(predictions[predictions.model == "ridge_free"])
    for name, frame in (("slices.csv", slices), ("cohort.csv", rows),
                        ("predictions.csv", predictions), ("skill.csv", scores),
                        ("july_carry_to_remainder.csv", carry),
                        ("carry_skill.csv", carry_metrics)):
        frame.to_csv(out / name, index=False)
    source = [root / "output/v7/full/post_probs_full.parquet",
              root / "output/v7/p3/panel_member_py_full.csv",
              root / "output/v7/ti/account_member_map.csv",
              root / "output/v7/t3/national.csv",
              Path(__file__), Path(__file__).with_name("test_rq3.py")]
    hashes = {}
    for path in source:
        with path.open("rb") as f:
            hashes[str(path)] = dict(sha256=hashlib.file_digest(f, "sha256").hexdigest(), bytes=path.stat().st_size)
    manifest = dict(
        design="observational predictive association, not an election treatment effect",
        final_year_status="reused_final_exploratory_not_untouched",
        year_definition="political year July 1 through June 30, America/New_York; right-open intervals",
        july_origin="July 1, 00:00 America/New_York of PY y",
        july_target="entire PY y [July 1 y, July 1 y+1); annual TI7 original-post mean",
        update_origin="December 8, 00:00 America/New_York of PY y",
        update_target_start="December 8, 00:00 America/New_York",
        update_target="only [December 8 y, July 1 y+1) original-post TI7 mean; earlier posts excluded",
        update_observation="[July 1 y, December 8 y) posts only, never the remainder",
        training_rule="each held-out PY y trained only on PY < y, separately by phase; fixed Ridge alpha=50; training-only scaling; no hyperparameter search",
        training_origin_years=[2018, 2019, 2020, 2021],
        development_evaluation_years=[2019, 2020, 2021], reused_final_evaluation_year=2022,
        matched_models=["persistence", "party_chamber_change", "ridge_free", "ridge_status"],
        free_features=FREE, status_extra_features=STATUS[len(FREE):],
        status_projection="July freezes current national control; Dec 8 uses publicly resolved incoming Jan national control; 2020 Georgia runoff unresolved, Senate R projection, not realized D",
        known_january_projection=KNOWN_JAN,
        member_party_caveat="previous-PY party/chamber static metadata; no future membership identity features; retrospective complete-case cohort",
        inclusion="prior PY and full target PY >=75 scored original posts, >=50 events, >=150 in-office days; target first segment >=25 and remainder >=40 posts; same member-PY cohort both phases",
        sample_selection_caveat="target-year quality/retention gate is retrospective evaluation selection, not available at forecast origin; limits prospective generalization",
        score_definition="equal-member-year RMSE; skill=1-RMSE(model)/RMSE(matched baseline); 1000 member-cluster percentile-bootstrap replicates; same target/origin/cohort",
        carry_caveat="July annual-level forecasts carried to remainder as secondary same-remainder diagnostic; not a separately trained remainder model or treatment effect",
        prior_results_caveat="previous +10.5% post-election and -9.1% July came from different models/targets and are not a treatment-effect contrast",
        audit={**audit, "cohort_by_origin": {f"{y}:{phase}": int(n) for (y, phase), n in rows.groupby(["py", "phase"]).size().items()},
               "predictions": len(predictions)},
        source_hashes=hashes,
        generated_outputs=["slices.csv", "cohort.csv", "predictions.csv", "skill.csv", "july_carry_to_remainder.csv", "carry_skill.csv", "results.json", "manifest.json"])
    (out / "results.json").write_text(json.dumps(
        dict(primary_skill=scores.to_dict("records"), secondary_carry_skill=carry_metrics.to_dict("records"),
             caveat=manifest["carry_caveat"]), indent=2) + "\n")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return scores


def carry_score(pred, B=1000):
    """Secondary same-remainder comparison, annual July forecasts carried unchanged."""
    carried = carry_july(pred).assign(phase="update")
    d = pd.concat([pred[pred.phase == "update"], carried], ignore_index=True)
    return score(d, B=B, pairs=[("ridge_free", "july_carry_free"),
                               ("ridge_status", "july_carry_status")], phases=("update",))


if __name__ == "__main__":
    from pathlib import Path
    result = run(Path.home() / "x_box/pipeline", Path(__file__).resolve().parent)
    print(result.to_string(index=False))


