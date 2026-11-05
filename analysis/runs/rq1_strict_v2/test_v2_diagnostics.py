import numpy as np
import pandas as pd
import pytest
from compare_features import compare_model_frames


def test_model_input_comparison_rejects_changed_array_and_eligibility():
    old=pd.DataFrame({"thread_id":[1,2],"acct":[0,0],"t":[1,2],"t_end":[1,2],
                      "n_completed_prior":[10,10],"strict_n_completed_prior":[10,10],
                      "strict_hour":[10.,np.nan]})
    assert compare_model_frames(old,old.copy(),["strict_hour"])["prediction_reuse_justified"]
    changed=old.copy();changed.loc[0,"strict_hour"]=11
    res=compare_model_frames(old,changed,["strict_hour"])
    assert res["changed_columns"]["strict_hour"]==1
    assert not res["prediction_reuse_justified"]
    changed=old.copy();changed.loc[0,"strict_n_completed_prior"]=9
    res=compare_model_frames(old,changed,["strict_hour"])
    assert not res["prediction_reuse_justified"]
    assert res["changed_eligibility_rows"]["strict"]==1


from time_blocks import block_sensitivity


def test_month_blocks_are_row_weighted_and_leave_one_out_not_ci():
    strict=pd.DataFrame({"thread_id":[1,2,3],"acct":[0,1,0],
                         "ti7":[0.,0.,0.],"content":[1.,1.,2.],
                         "rhythm":[0.,2.,0.],
                         "origin":["2022-07-01","2022-07-01","2022-08-01"]})
    old=strict.copy();old["rhythm"]=old["content"]
    result=block_sensitivity(strict,old)
    assert result["n_common"]==3
    assert result["overall"]["strict_delta_mse"]==pytest.approx(2/3)
    assert result["overall"]["strict_relative_mse_reduction"]==pytest.approx(1/3)
    assert result["overall"]["strict_minus_legacy_delta_mse"]==pytest.approx(2/3)
    assert [x["n"] for x in result["per_month"]]==[2,1]
    assert [x["strict_delta_mse"] for x in result["per_month"]]==[-1.,4.]
    assert [x["strict_delta_mse"] for x in result["leave_one_month_out"]]==[4.,-1.]
    assert "ci95" not in result


from compare_features import prediction_differences


def test_prediction_comparison_counts_changed_forecasts_not_merely_files():
    old=pd.DataFrame({"thread_id":[1,2],"acct":[0,0],"t":[1,2],"ti7":[0.,0.],
                      "content":[1.,1.],"rhythm":[2.,2.]})
    new=old.copy();new.loc[1,"rhythm"]=3.
    assert prediction_differences(old,new)=={"rows":2,"content_changed":0,"rhythm_changed":1}
