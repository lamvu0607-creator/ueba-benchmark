"""
Unit tests for feature extraction components and preprocessor.
"""

from pathlib import Path

import math

import polars as pl
import pytest
from src.features.extractor import (
    DEFAULT_FEATURE_CFG,
    INTRADAY_V3_FEATURES,
    INTRADAY_V3_REJECTED,
    RAW_ORDERED_COLS,
    entity_type_expr,
    intraday_behavior_features,
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
    Hợp đồng cột của extractor: ``RAW_ORDERED_COLS`` không trùng lặp, phủ đủ **22 đặc trưng core**
    (kể cả 3 biến thể sinh bằng log1p và 6 đặc trưng nhóm 9 của v3.0), và cấu hình cửa sổ giờ /
    entity_type đúng giá trị đã chốt.
    """
    assert len(RAW_ORDERED_COLS) == len(set(RAW_ORDERED_COLS)) == 24
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
    assert len(core) == 20

    # 4 đặc trưng v3.0 phải nằm trong hợp đồng cột của extractor; 2 biến đã bác bỏ thì KHÔNG.
    for name in INTRADAY_V3_FEATURES:
        assert name in extracted, f"Thiếu đặc trưng v3.0 '{name}' trong RAW_ORDERED_COLS"
    assert len(INTRADAY_V3_FEATURES) == 4
    for name in INTRADAY_V3_REJECTED:
        assert name not in extracted, f"'{name}' đã bị bác bỏ, không được sinh vào ma trận"

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
    Hợp đồng giữa tầng feature và tầng model: parquet processed phải có đủ 20 đặc trưng core dạng số,
    và ``AnomalyPipeline`` phải chọn ĐÚNG 20 cột đó (không dùng biến thể thô trùng lặp).

    Test này là bản phục hồi của ``test_features.py::test_model_interfaces`` trong bản benchmark
    đã mất (dấu vết nằm ở ``.pytest_cache/v/cache/nodeids``).
    """
    core = list(FeatureSchema().core_features)
    pipeline = AnomalyPipeline(IsolationForestDetector(contamination=0.05))
    assert pipeline.feature_names == core
    assert len(core) == 20

    if not PROCESSED_MATRIX.is_file():
        pytest.skip("Chưa có data/processed/feature_matrix_processed.parquet")

    all_columns = set(pl.scan_parquet(PROCESSED_MATRIX).collect_schema().names())
    stale = [name for name in core if name not in all_columns]
    if stale:
        pytest.skip(
            "data/processed/feature_matrix_processed.parquet là artifact của schema CŨ (thiếu "
            f"{len(stale)} đặc trưng: {stale}). Hãy dựng lại: python main.py --stage features. "
            "Kiểm định artifact ở mức script: scripts/feature_engineering/check_feature_matrix.py"
        )
    assert set(core) <= all_columns

    # Các cột còn lại (ngoài danh tính) chỉ được là biến thể thô/hiển thị -> model không dùng.
    non_core = all_columns - set(core) - {"DomainName", "UserName", "day", "entity_type"}
    assert non_core <= {"total_logons", "distinct_hosts", "rare_logon_type_count"}

    sample = pl.read_parquet(PROCESSED_MATRIX, columns=core).head(1000)
    assert sample.width == 20
    for name in core:
        assert sample[name].cast(pl.Float64, strict=False).null_count() < sample.height


def _synthetic_events() -> pl.DataFrame:
    """3 tài khoản bao đủ các case biên của nhóm 9 (1 sự kiện, trải đều nhiều giờ, gai hoà nhau)."""
    rows = [
        # UserA: 09h có 3 lần 4625 liên tiếp + 1 lần 4624, sau đó 10h có 1 lần 4624
        ("dom1", "UserA", 9 * 3600 + 10, 4625, "host1"),
        ("dom1", "UserA", 9 * 3600 + 20, 4625, "host1"),
        ("dom1", "UserA", 9 * 3600 + 30, 4625, "host2"),
        ("dom1", "UserA", 9 * 3600 + 40, 4624, "host2"),
        ("dom1", "UserA", 10 * 3600 + 5, 4624, "host2"),
        # UserB: đúng 1 sự kiện 4625 lúc 23h -> case biên của streak/NULL/entropy
        ("dom1", "UserB", 23 * 3600, 4625, "host1"),
        # UserC: 4 sự kiện trải 4 giờ, đều nhau, cùng 1 host -> hour_entropy = 1.0, dst = 0.0
        ("dom1", "UserC", 1 * 3600, 4624, "host9"),
        ("dom1", "UserC", 2 * 3600, 4624, "host9"),
        ("dom1", "UserC", 3 * 3600, 4624, "host9"),
        ("dom1", "UserC", 4 * 3600, 4624, "host9"),
    ]
    return pl.DataFrame(
        rows, schema=["DomainName", "UserName", "Time", "EventID", "LogHost"], orient="row"
    )


def test_intraday_behavior_features():
    """
    4 đặc trưng nhóm 9 (schema v3.0) đúng công thức trên dữ liệu tổng hợp, gồm cả case biên.

    Bằng chứng giá trị kỳ vọng (tự tính tay):
      * UserA: host distribution 3/2 -> evenness = -(0,6·ln0,6 + 0,4·ln0,4)/ln2 = 0,9709506;
        hour distribution 4/1 -> H/ln2 = 0,7219281.
      * UserC: 4 khung giờ đều nhau -> H/ln4 = 1,0; 1 host -> 0,0.
      * UserB (1 sự kiện): mọi entropy = 0 (quy ước S <= 1).
    """
    feats = intraday_behavior_features(_synthetic_events()).sort("UserName")
    by_user = {row["UserName"]: row for row in feats.to_dicts()}

    assert feats.height == 3
    assert list(feats.columns[2:]) == list(INTRADAY_V3_FEATURES)

    # UserA — giờ cao điểm 9h + hai evenness
    user_a = by_user["UserA"]
    assert user_a["activity_peak_hour_sin"] == pytest.approx(math.sin(2 * math.pi * 9 / 24), abs=1e-12)
    assert user_a["activity_peak_hour_cos"] == pytest.approx(math.cos(2 * math.pi * 9 / 24), abs=1e-12)
    assert user_a["hour_entropy"] == pytest.approx(0.721928094887, abs=1e-9)
    assert user_a["dst_host_entropy"] == pytest.approx(0.970950594455, abs=1e-9)

    # UserB — ngày 1 sự kiện: entropy = 0 (mọi sự kiện cùng 1 giờ / 1 host)
    user_b = by_user["UserB"]
    assert user_b["hour_entropy"] == 0.0
    assert user_b["dst_host_entropy"] == 0.0

    # UserC — trải đều 4 giờ, 1 host
    user_c = by_user["UserC"]
    assert user_c["hour_entropy"] == pytest.approx(1.0, abs=1e-12)
    assert user_c["dst_host_entropy"] == 0.0

    # Bất biến: vào khung CHƯA sắp xếp vẫn cho cùng kết quả (hàm tự sort)
    shuffled = intraday_behavior_features(
        _synthetic_events().sample(fraction=1.0, seed=7)
    ).sort("UserName")
    by_user_shuffled = {row["UserName"]: row for row in shuffled.to_dicts()}
    for account, expected_row in by_user.items():
        got = by_user_shuffled[account]
        for column in INTRADAY_V3_FEATURES:
            assert got[column] == pytest.approx(expected_row[column], abs=1e-9), (account, column)

    # Bất biến: khung rỗng vẫn trả đúng schema, không crash
    empty = intraday_behavior_features(_synthetic_events().head(0))
    assert empty.height == 0
    assert list(empty.columns[2:]) == list(INTRADAY_V3_FEATURES)

