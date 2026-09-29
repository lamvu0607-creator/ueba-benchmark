"""
Unit tests for feature extraction components and preprocessor.
"""

from pathlib import Path

import polars as pl
import pytest
from src.features.extractor import (
    DEFAULT_FEATURE_CFG,
    RAW_ORDERED_COLS,
    entity_type_expr,
    resolve_feature_config,
)
from src.features.preprocessor import normalize_features
from src.features.schema import FeatureSchema
from src.models.detectors import IsolationForestDetector
from src.models.pipeline import AnomalyPipeline

PROCESSED_MATRIX = Path(__file__).resolve().parents[1] / "data" / "processed" / "feature_matrix_processed.parquet"


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


def test_feature_extraction_columns_and_counts():
    """
    Hợp đồng cột của extractor: ``RAW_ORDERED_COLS`` không trùng lặp, phủ đủ 16 đặc trưng core
    (kể cả 3 biến thể sinh bằng log1p), và cấu hình cửa sổ giờ/entity_type đúng giá trị đã chốt.
    """
    assert len(RAW_ORDERED_COLS) == len(set(RAW_ORDERED_COLS)) == 20
    assert RAW_ORDERED_COLS[:4] == ["DomainName", "UserName", "day", "entity_type"]

    # 3 đặc trưng core được sinh từ cột thô trong RAW_ORDERED_COLS bằng log1p (xem schema v2).
    derived = {
        "log_total_logons": "total_logons",
        "log_distinct_hosts": "distinct_hosts",
        "rare_logon_type_count_log": "rare_logon_type_count",
    }
    extracted = set(RAW_ORDERED_COLS)
    core = list(FeatureSchema().core_features)
    missing = [name for name in core if name not in extracted and derived.get(name) not in extracted]
    assert missing == [], f"Extractor không sinh được các đặc trưng core: {missing}"
    assert len(core) == 16

    # total_logons là cột HIỂN THỊ (trùng hạng rho = 1.0000 với log_total_logons) -> không vào model.
    assert "total_logons" not in core

    cfg = resolve_feature_config({})
    assert (cfg["off_hours_start"], cfg["off_hours_end"]) == (18, 7)
    assert (cfg["work_hours_start"], cfg["work_hours_end"]) == (8, 17)
    assert cfg["entity_type"]["machine_suffix"] == "$"
    assert cfg["entity_type"]["admin_accounts"] == ["administrator"]
    assert cfg["filter"] == {"drop_machine_accounts": False, "drop_system_accounts": False}
    assert DEFAULT_FEATURE_CFG["off_hours_start"] == 18


def test_model_interfaces():
    """
    Hợp đồng giữa tầng feature và tầng model: parquet processed phải có đủ 16 đặc trưng core dạng số,
    và ``AnomalyPipeline`` phải chọn ĐÚNG 16 cột đó (không dùng biến thể thô trùng lặp).

    Test này là bản phục hồi của ``test_features.py::test_model_interfaces`` trong bản benchmark
    đã mất (dấu vết nằm ở ``.pytest_cache/v/cache/nodeids``).
    """
    core = list(FeatureSchema().core_features)
    pipeline = AnomalyPipeline(IsolationForestDetector(contamination=0.05))
    assert pipeline.feature_names == core
    assert len(core) == 16

    if not PROCESSED_MATRIX.is_file():
        pytest.skip("Chưa có data/processed/feature_matrix_processed.parquet")

    all_columns = set(pl.scan_parquet(PROCESSED_MATRIX).collect_schema().names())
    assert set(core) <= all_columns

    # Các cột còn lại (ngoài danh tính) chỉ được là biến thể thô/hiển thị -> model không dùng.
    non_core = all_columns - set(core) - {"DomainName", "UserName", "day", "entity_type"}
    assert non_core <= {"total_logons", "distinct_hosts", "rare_logon_type_count"}

    sample = pl.read_parquet(PROCESSED_MATRIX, columns=core).head(1000)
    assert sample.width == 16
    for name in core:
        assert sample[name].cast(pl.Float64, strict=False).null_count() < sample.height
