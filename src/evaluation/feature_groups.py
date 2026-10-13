"""
Nhóm đặc trưng THEO NGHĨA dùng cho SHAP/LIME và ablation (tuần 5).

Schema gộp 17 đặc trưng template vào một nhóm ``template_v4``; ở đây 41 đặc trưng core được chia lại thành 8 nhóm
theo hành vi chúng đo. Nguồn gốc (core_v3 / template_v4) vẫn đọc từ schema bằng ``schema_origin``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import yaml

__all__ = ["ABLATION_GROUPS", "FEATURE_GROUP", "LOW_SHAP_FEATURES", "schema_origin"]

ABLATION_GROUPS: Dict[str, List[str]] = {
    "volume": ["log_total_logons", "volume_robust_z_7d", "delta_mean_count_7d"],
    "failure": ["failure_ratio", "failure_locked_out_share", "delta_mean_share_fail_7d",
                "delta_mean_distinct_fail_host_7d", "novelty_fail_host_7d", "novelty_fail_failreason_7d"],
    "time": ["off_hours_ratio", "activity_peak_hour_sin", "activity_peak_hour_cos", "hour_entropy",
             "dist_shift_hour_7d", "jaccard_hour_7d", "peer_z_share_night", "delta_mean_share_night_7d"],
    "rhythm": ["interarrival_dt_mean", "delta_t_cv", "same_second_share", "is_single_event"],
    "logon_type": ["interactive_ratio", "rare_logon_type_count_log", "ntlm_ratio", "dist_shift_logontype_7d",
                   "jaccard_logontype_7d", "novelty_logontype_7d"],
    "fanout": ["log_distinct_hosts", "distinct_sources_count", "dst_host_entropy", "delta_mean_distinct_source_7d",
               "peer_z_distinct_source", "delta_mean_distinct_pair_7d"],
    "novelty": ["new_source_count_7d", "new_host_count_7d", "days_since_last_activity", "jaccard_source_7d",
                "novelty_share_pair_7d"],
    "context": ["missing_source_ratio", "remote_logon_ratio", "custom_proc_share"],
}
FEATURE_GROUP: Dict[str, str] = {f: g for g, fs in ABLATION_GROUPS.items() for f in fs}

# 10 đặc trưng có tỷ phần |SHAP| cao nhất (qua 3 mô hình, trên nạn nhân) thấp nhất — reports/week5/shap_lime.md.
LOW_SHAP_FEATURES: List[str] = [
    "is_single_event", "activity_peak_hour_sin", "log_distinct_hosts", "missing_source_ratio", "log_total_logons",
    "dst_host_entropy", "delta_t_cv", "volume_robust_z_7d", "peer_z_share_night", "same_second_share",
]


def schema_origin(path: Path | str) -> Dict[str, str]:
    """core_v3 (nhóm schema thường) hay template_v4 (sinh từ template ở vòng v4), cho mọi đặc trưng core."""
    schema = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return {f["name"]: "template_v4" if f.get("group") == "template_v4" else "core_v3"
            for f in schema["features"] if f.get("core")}
