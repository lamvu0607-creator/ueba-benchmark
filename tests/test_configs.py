"""
Unit tests for system_config.yaml and feature_schema.yaml.
"""

from pathlib import Path
import pytest
import yaml

from src.features.schema import FeatureSchema


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
    assert len(core_feats) == 20, (
        f"Hợp đồng Schema v3.0 phải có đúng 20 core features, hiện có {len(core_feats)}"
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
        # v3.0 (nhóm 9) — chỉ dùng trạng thái trong ngày, đã qua cổng |rho| >= 0,85
        "activity_peak_hour_sin",
        "activity_peak_hour_cos",
        "hour_entropy",
        "dst_host_entropy",
    ]
    for feat in expected_core:
        assert feat in core_feats, f"Đặc trưng cốt lõi '{feat}' thiếu trong schema"

    # 2 đặc trưng đã ĐO và bác bỏ phải nằm ở mục `removed` (kèm lý do định lượng), KHÔNG ở core.
    removed = {entry["name"] for entry in schema.raw_schema.get("removed", [])}
    for rejected in ("max_failure_streak", "success_after_failure_ratio"):
        assert rejected not in core_feats, f"'{rejected}' phải bị loại khỏi core (đã đo vượt ngưỡng rho)"
        assert rejected in removed, f"'{rejected}' phải được ghi vào mục `removed` kèm lý do"
