from run_gbt import HERE, RQ1_STRICT

def test_v2_source_and_isolation():
    assert HERE.name == "rq1_gbt_v2"
    assert RQ1_STRICT.name == "rq1_strict_v2"
    assert (RQ1_STRICT / "strict_features.parquet").exists()
