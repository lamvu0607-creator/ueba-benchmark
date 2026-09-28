"""
Unit tests for feature extraction components and preprocessor.
"""

import polars as pl
import pytest
from src.features.extractor import entity_type_expr, DEFAULT_FEATURE_CFG, RAW_ORDERED_COLS
from src.features.preprocessor import normalize_features
from src.features.schema import FeatureSchema


def test_entity_type_classification():
    data = pl.DataFrame({
        "UserName": [
            "administrator",
            "SYSTEM",
            "comp1234$",
            "User9988",
            "appservice",
            "unknown_stranger",
        ]
    })

    result = data.with_columns(entity_type_expr(DEFAULT_FEATURE_CFG))
    types = result["entity_type"].to_list()
    assert types == ["Admin", "System", "Machine", "User", "Service", "Other"]


def test_normalize_features():
    df_raw = pl.DataFrame({
        "DomainName": ["dom1"],
        "UserName": ["User1"],
        "day": [1],
        "entity_type": ["User"],
        "total_logons": [10],
        "distinct_hosts": [2],
        "rare_logon_type_count": [0],
        "failure_ratio": [0.1],
        "failure_locked_out_share": [None],
        "interarrival_dt_mean": [None],
        "delta_t_cv": [None],
    })

    df_norm = normalize_features(df_raw, fill_null_strategy="indicator")

    assert "log_total_logons" in df_norm.columns
    assert "log_distinct_hosts" in df_norm.columns
    assert "rare_logon_type_count_log" in df_norm.columns

    # log1p(10) ~ 2.397895
    assert abs(df_norm["log_total_logons"][0] - 2.397895) < 1e-4
    # NULL handling: fill 0 khi dùng indicator
    assert df_norm["failure_locked_out_share"][0] == 0.0
    assert df_norm["interarrival_dt_mean"][0] == 0.0


def test_feature_schema_validation_success():
    schema = FeatureSchema()
    df = pl.DataFrame({
        "DomainName": ["dom1"],
        "UserName": ["User1"],
        "day": [1],
        "entity_type": ["User"],
        "failure_ratio": [0.5],
        "off_hours_ratio": [0.2],
    })

    res = schema.validate_features(df)
    assert res["is_valid"] is True
    assert len(res["missing_keys"]) == 0
    assert len(res["invalid_ratios"]) == 0


def test_feature_schema_validation_invalid_ratio():
    schema = FeatureSchema()
    df = pl.DataFrame({
        "DomainName": ["dom1"],
        "UserName": ["User1"],
        "day": [1],
        "entity_type": ["User"],
        "failure_ratio": [1.5],  # Sai: ratio > 1
    })

    res = schema.validate_features(df)
    assert res["is_valid"] is False
    assert "failure_ratio" in res["invalid_ratios"]
