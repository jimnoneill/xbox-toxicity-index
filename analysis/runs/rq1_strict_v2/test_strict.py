import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path.home() / "x_box/pipeline"
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "manuscript/epjds_submission_work/rq1")]
from xbox import location as lc
from strict_features import local_clock, select_origin_strict, repair_lags_strict, rebuild_local_columns, zone_for_congress, clocks_for_frame, causal_congress_features


def test_real_1986702_does_not_see_later_vote():
    original = pd.read_parquet(ROOT / "output/v7/p1/features.parquet", columns=["thread_id", "bioguide", "t", "hour"])
    row = original.loc[original.thread_id == 1986702].iloc[0]
    assert row.bioguide == "A000055" and row.hour == 16
    member_days, _, _ = lc.load_vote_days(ROOT / "data/voteview")
    days = member_days[29701]
    hour, weekend, cls = local_clock(np.array([row.t]), days, "America/Chicago")
    assert hour.tolist() == [15]
    assert weekend.tolist() == [0]
    assert cls.tolist() == [lc.HOME]
    # Replacing every vote on/after the Eastern date cannot affect the answer.
    et_day = lc.to_local([row.t], lc.ET)[2][0]
    altered = np.r_[days[days < et_day], et_day, et_day + 1]
    np.testing.assert_array_equal(hour, local_clock(np.array([row.t]), altered, "America/Chicago")[0])


def test_strict_clock_uses_prior_date_only_and_dst_local_calendar():
    t = np.array([int(pd.Timestamp("2022-07-11T02:00:00Z").timestamp()),
                  int(pd.Timestamp("2022-07-11T20:55:32Z").timestamp())])
    day = lc.to_local(t, lc.ET)[2]
    vote = np.array([day[1], day[1] + 1])
    h, weekend, cls = local_clock(t, vote, "America/Chicago")
    assert h.tolist() == [21, 15]  # Sunday local and Monday local
    assert weekend.tolist() == [1, 0]
    assert cls.tolist() == [lc.HOME, lc.HOME]
    h2, _, cls2 = local_clock(t[1:], np.array([day[1] - 1]), "America/Chicago")
    assert cls2.tolist() == [lc.TRANSITION] and h2.tolist() == [16]


def test_real_origin_closure_embargo_and_exact_boundary():
    origin = int(pd.Timestamp("2022-07-01").timestamp())
    data = pd.read_parquet(ROOT / "manuscript/epjds_submission_work/rq1/arrival_features.parquet",
                           columns=["thread_id", "acct", "t", "t_end", "n_completed_prior"])
    near = data[(data.t < origin) & (data.t_end >= origin - 7200) & (data.t_end < origin)]
    assert len(near) > 0
    # Use a real near-boundary account, and a qualifying real early row of it.
    acct = near.acct.iloc[0]
    rows = data[data.acct == acct]
    tr, _ = select_origin_strict(rows, origin, origin + 86400, min_train=1)
    assert not set(near.thread_id).intersection(tr.thread_id)
    assert (tr.t_end + 7200 < origin).all()
    fake = pd.DataFrame({"thread_id": [1, 2, 3, 4], "acct": [1]*4,
                         "t": [origin-9000, origin-9000, origin-9000, origin],
                         "t_end": [origin-7201, origin-7200, origin-7199, origin],
                         "n_completed_prior": [10]*4})
    a, b = select_origin_strict(fake, origin, origin+86400, min_train=1)
    assert a.thread_id.tolist() == [1] and b.thread_id.tolist() == [4]


def test_label_and_size_lags_require_closed_thread():
    events = pd.DataFrame({"thread_id": [1, 2, 3], "screen_name": ["x"]*3,
                           "t": [100, 10000, 10001], "t_end": [2800, 10000, 10001],
                           "ti7": [.9, .2, .4], "size": [2, 1, 1]})
    got = repair_lags_strict(events).set_index("thread_id")
    assert got.loc[2, "n_completed_prior"] == 0  # closure exactly at arrival
    assert np.isnan(got.loc[2, "prev_ti7"])
    assert got.loc[3, "n_completed_prior"] == 1
    assert got.loc[3, "prev_ti7"] == pytest.approx(.9)
    assert got.loc[3, "prev_multi"] == 1


def test_all_derived_clock_columns_are_rebuilt():
    src = pd.DataFrame({"hour": [16.], "hsin": [0.], "hcos": [0.], "h2sin": [0.],
                        "h2cos": [0.], "night": [1.], "weekend": [1.]})
    dst = rebuild_local_columns(src, np.array([15]), np.array([0]))
    assert dst.hour.iloc[0] == 15 and dst.weekend.iloc[0] == 0 and dst.night.iloc[0] == 0
    assert dst.hsin.iloc[0] == pytest.approx(np.sin(2*np.pi*15/24))
    assert dst.h2cos.iloc[0] == pytest.approx(np.cos(4*np.pi*15/24))


def test_split_state_district_zone_follows_active_congress_not_future_post_weights():
    members = pd.DataFrame({"icpsr": [42,42], "chamber": ["House","House"],
                            "state_abbrev": ["TX","TX"], "congress": [117,118],
                            "district_code": [1,2]})
    zones = {("cd116","TX",1): ("America/Chicago",1.),
             ("cd118","TX",2): ("America/Denver",1.)}
    assert zone_for_congress(42,"TX","house",117,members,zones)=="America/Chicago"
    assert zone_for_congress(42,"TX","house",118,members,zones)=="America/Denver"
    # An amended future Congress record cannot move the 117th's zone.
    changed = members.copy(); changed.loc[changed.congress==118,"district_code"] = 99
    assert zone_for_congress(42,"TX","house",117,changed,zones)=="America/Chicago"


def test_full_frame_clock_rebuild_uses_real_member_metadata():
    p = pd.read_parquet(ROOT / "output/v7/p1/features.parquet",
                        columns=["thread_id","bioguide","t","hour","weekend"])
    row = p[p.thread_id==1986702].copy()
    amap = pd.read_csv(ROOT / "output/v7/ti/account_member_map.csv")
    vote, _, _ = lc.load_vote_days(ROOT / "data/voteview")
    members = pd.read_csv(ROOT / "data/voteview_all_members.csv")
    zones = lc.load_district_table(ROOT / "data/district_zone/district_zone.csv")
    result, audit = clocks_for_frame(row, amap, vote, members, zones)
    assert result.hour.iloc[0] == 15 and result.weekend.iloc[0] == 0
    assert audit["split_state_rows_using_state_zone"] == 0


def test_congress_lag_does_not_consult_future_hours_or_retrospective_bad_flag():
    counts=np.arange(1000,dtype=float)%17
    first=causal_congress_features(counts)
    change=counts.copy(); change[801:]=1e9
    second=causal_congress_features(change)
    for c in first:
        np.testing.assert_array_equal(first[c][:801],second[c][:801])


def test_clock_uses_117th_seat_for_a_2022_event_not_future_118th_seat():
    t=int(pd.Timestamp("2022-07-11T20:55:32Z").timestamp())
    frame=pd.DataFrame({"bioguide":["T000001"],"t":[t],"hour":[0.],"weekend":[0.]})
    amap=pd.DataFrame({"bioguide":["T000001"],"entity_type":["member"],
                       "icpsr":[42],"state":["TX"],"chamber":["house"]})
    members=pd.DataFrame({"icpsr":[42,42],"chamber":["House","House"],
                          "state_abbrev":["TX","TX"],"congress":[117,118],
                          "district_code":[1,2]})
    zones={("cd116","TX",1):("America/Chicago",1.),
           ("cd118","TX",2):("America/Denver",1.)}
    out,_=clocks_for_frame(frame,amap,{42:np.array([],dtype=np.int64)},members,zones)
    assert out.hour.tolist()==[15]


def test_manifest_hashes_vote_and_congress_inputs():
    import json
    import run_strict
    run_strict.manifest()
    data=json.loads((HERE/"manifest.json").read_text())
    inputs=data["read_only_inputs"]
    for path in (ROOT/"data/voteview/H117_votes.csv",
                 ROOT/"data/voteview/H117_rollcalls.csv",
                 ROOT/"data/voteview_all_members.csv",
                 ROOT/"data/district_zone/district_zone.csv",
                 ROOT/"output/v7/t2full/congress_hourly.parquet",
                 ROOT/"output/v7/full/post_probs_full.parquet",
                 ROOT/"output/v7/t1full/posts_threads.parquet",
                 ROOT/"output/v7/t1full/outage_mask.parquet",
                 ROOT/"xbox/hygiene.py",
                 ROOT/"scripts/v7/t1full_build.py",
                 ROOT/"scripts/v7/t1full_common.py",
                 ROOT/"scripts/v7/full_corpus_collect.py",
                 ROOT/"scripts/v7/full_corpus_assemble.py",
                 ROOT/"data/congresstweets_full/congresstweets_full.parquet",
                 ROOT/"data/congresstweets_full/expected_days.json",
                 ROOT/"data/congresstweets_full/raw/2017-06-21.json.gz",
                 ROOT/"manuscript/epjds_submission_work/rq1/predictions/origin_2022-07.parquet"):
        assert str(path) in inputs
    for pattern in (ROOT/"data/congresstweets_full/raw", ROOT/"manuscript/epjds_submission_work/rq1/predictions"):
        for file in pattern.glob("*.json.gz" if pattern.name=="raw" else "origin_*.parquet"):
            assert str(file) in inputs


def test_full_artifact_retains_real_repaired_clock_and_all_harmonics():
    path=HERE/"strict_features.parquet"
    f=pd.read_parquet(path,columns=["thread_id","hour","strict_hour","strict_hsin","strict_hcos","strict_h2sin","strict_h2cos","strict_night","strict_weekend"])
    row=f.loc[f.thread_id==1986702].iloc[0]
    assert row.hour==16 and row.strict_hour==15
    assert row.strict_weekend==0 and row.strict_night==0
    angle=2*np.pi*15/24
    assert row.strict_hsin==pytest.approx(np.sin(angle),abs=1e-6)
    assert row.strict_hcos==pytest.approx(np.cos(angle),abs=1e-6)
    assert row.strict_h2sin==pytest.approx(np.sin(2*angle),abs=1e-6)
    assert row.strict_h2cos==pytest.approx(np.cos(2*angle),abs=1e-6)


def test_legacy_reproduction_matches_archived_monthly_predictions():
    old=ROOT/"manuscript/epjds_submission_work/rq1/predictions"
    new=HERE/"predictions/legacy_reproduction"
    archived=sorted(old.glob("origin_*.parquet")); rebuilt=sorted(new.glob("origin_*.parquet"))
    assert len(archived)==len(rebuilt)==13
    for x,y in zip(archived,rebuilt):
        a=pd.read_parquet(x,columns=["thread_id","content","rhythm"])
        b=pd.read_parquet(y,columns=["thread_id","content","rhythm"])
        np.testing.assert_array_equal(a.thread_id,b.thread_id)
        np.testing.assert_array_equal(a.content,b.content)
        np.testing.assert_array_equal(a.rhythm,b.rhythm)


def test_house_seat_changes_at_noon_eastern_not_midnight_2023():
    # 117th remains active through 2023-01-03 16:59:59Z (11:59:59 ET).
    # At 17:00:00Z (noon ET) the 118th seat becomes active.
    t=[int(pd.Timestamp(x).timestamp()) for x in
       ("2023-01-03T16:59:59Z", "2023-01-03T17:00:00Z")]
    frame=pd.DataFrame({"bioguide":["T000001"]*2,"t":t,"hour":[0.]*2,"weekend":[0.]*2})
    amap=pd.DataFrame({"bioguide":["T000001"],"entity_type":["member"],
                       "icpsr":[42],"state":["TX"],"chamber":["house"]})
    members=pd.DataFrame({"icpsr":[42,42],"chamber":["House","House"],
                          "state_abbrev":["TX","TX"],"congress":[117,118],
                          "district_code":[1,2]})
    zones={(lc.CONGRESS_MAP[117],"TX",1):("America/Chicago",1.),
           (lc.CONGRESS_MAP[118],"TX",2):("America/Denver",1.)}
    out,_=clocks_for_frame(frame,amap,{42:np.array([],dtype=np.int64)},members,zones)
    assert out.hour.tolist()==[10,10]  # old midnight switch gives [9,10]
    assert out.strict_loc_class.tolist()==[lc.HOME,lc.HOME]
