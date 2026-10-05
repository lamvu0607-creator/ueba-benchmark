"""
Unit tests for system_config.yaml and feature_schema.yaml.
"""

from pathlib import Path
import pytest
import yaml

from src.features.extractor import TEMPLATE_V4_FEATURES
from src.features.schema import FeatureSchema
from src.features.templates import candidate_by_name


def test_system_config_structure():
    config_path = Path("configs/system_config.yaml")
    assert config_path.is_file(), "File configs/system_config.yaml phải tồn tại"

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert "system" in cfg
    assert "paths" in cfg
    assert "features" in cfg
    assert "evaluation" in cfg

    paths = cfg["paths"]
    assert "raw_data_dir" in paths
    assert "interim_data_dir" in paths
    assert "features_data_dir" in paths
    assert "processed_data_dir" in paths


def test_feature_schema_contract():
    schema = FeatureSchema()
    assert schema.identity_keys == ["DomainName", "UserName", "day"]
    assert schema.label_key == "entity_type"

    core_feats = schema.core_features
    assert len(core_feats) == 39, (
        f"Hợp đồng Schema v4.0 phải có đúng 39 core features, hiện có {len(core_feats)}"
    )

    expected_core = [
        "log_total_logons",
        "failure_ratio",
        "failure_locked_out_share",
        "off_hours_ratio",
        "interarrival_dt_mean",
        "delta_t_cv",
        "same_second_share",
        "is_single_event",
        "interactive_ratio",
        "rare_logon_type_count_log",
        "ntlm_ratio",
        "log_distinct_hosts",
        "distinct_sources_count",
        "missing_source_ratio",
        "remote_logon_ratio",
        "custom_proc_share",
        # v3.0 nhóm 9 (trong ngày) — đã qua cổng |rho| >= 0,85
        "activity_peak_hour_sin",
        "activity_peak_hour_cos",
        "hour_entropy",
        "dst_host_entropy",
        # v3.0 nhóm 10 (lịch sử) — chỉ dùng dữ liệu <= t-1; 4 biến đã qua cổng kiểm định
        "new_source_count_7d",
        "new_host_count_7d",
        "days_since_last_activity",
        "volume_robust_z_7d",
    ] + list(TEMPLATE_V4_FEATURES)  # v4.0 nhóm 11 — 15 biến template đã qua phễu
    for feat in expected_core:
        assert feat in core_feats, f"Đặc trưng cốt lõi '{feat}' thiếu trong schema"

    # 6 đặc trưng đã ĐO và bác bỏ phải nằm ở mục `removed` (kèm lý do định lượng), KHÔNG ở core.
    removed = {entry["name"] for entry in schema.raw_schema.get("removed", [])}
    for rejected in (
        "max_failure_streak", "success_after_failure_ratio",
        "new_source_count", "new_host_count", "source_recency", "source_host_pair_novelty",
    ):
        assert rejected not in core_feats, f"'{rejected}' phải bị loại khỏi core (đã đo vượt ngưỡng)"
        assert rejected in removed, f"'{rejected}' phải được ghi vào mục `removed` kèm lý do"

    # v4.0: biến dự bị (qua 4 cổng, chưa vào core) và biến bị loại ở vòng 2 không được lọt vào core
    reserve = {entry["name"] for entry in schema.raw_schema.get("reserve", [])}
    assert len(reserve) == 11 and not reserve & set(core_feats)
    for name in ("count_fail_15m", "fanout_fail_source", "recency_host_hist", "count_night"):
        assert name in removed and name not in core_feats

    # mọi biến template trong core phải khớp đúng tổ hợp template khai báo trong schema
    defs = schema.feature_definitions
    for name in TEMPLATE_V4_FEATURES:
        c = candidate_by_name(name)
        t = defs[name]["template"]
        assert (t["measure"], t["object"], t["entity"], t["window"]) == (c.measure, c.obj, c.entity, c.window)
