"""
Unit tests cho Segmented Execution của benchmark (``segments`` trong ``configs/model_params.yaml``).
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
import yaml

from src.models.benchmark import ALL_SEGMENT, SegmentConfig, load_labels, run_model_benchmark, split_segments
from src.models.pipeline import AnomalyPipeline

PARAMS_PATH = Path(__file__).resolve().parents[1] / "configs" / "model_params.yaml"
FEATURES = ["f1", "f2", "f3"]
SEGMENTS_BLOCK = {
    "column": "entity_type",
    "enabled": ["Machine", "User"],
    "ignored": ["Admin", "Service", "System", "Other"],
}


def segmented_frame(seed: int = 0) -> pl.DataFrame:
    """
    10 ngày; Machine ở quanh 100, User quanh 0 (hai quần thể rất khác thang đo), cộng vài dòng
    Admin/Service. Mỗi phân khúc có một NULL ở train để kiểm tra median impute riêng.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for day in range(1, 11):
        for entity_type, n, loc in (("Machine", 40, 100.0), ("User", 30, 0.0), ("Admin", 2, 50.0), ("Service", 1, 7.0)):
            for i in range(n):
                values = rng.normal(loc, 1.0, len(FEATURES))
                rows.append(
                    {"DomainName": "DOM", "UserName": f"{entity_type}{i}", "day": day, "entity_type": entity_type,
                     **dict(zip(FEATURES, values.tolist()))}
                )
    df = pl.DataFrame(rows)
    df = df.with_columns(
        pl.when((pl.col("day") == 1) & (pl.col("UserName").is_in(["Machine0", "User0"])))
        .then(None)
        .otherwise(pl.col("f1"))
        .alias("f1")
    )
    return df


def write_inputs(tmp: Path, segments=SEGMENTS_BLOCK) -> tuple:
    data_path = tmp / "matrix.parquet"
    segmented_frame().write_parquet(data_path)
    params = yaml.safe_load(PARAMS_PATH.read_text(encoding="utf-8"))
    if segments is None:
        params.pop("segments", None)
    else:
        params["segments"] = segments
    params_path = tmp / "params.yaml"
    params_path.write_text(yaml.safe_dump(params, allow_unicode=True), encoding="utf-8")
    return data_path, params_path


def run(tmp: Path, **kwargs):
    tmp.mkdir(parents=True, exist_ok=True)
    data_path, params_path = write_inputs(tmp, kwargs.pop("segments", SEGMENTS_BLOCK))
    return run_model_benchmark(
        data_path=data_path,
        model_names=["isolation_forest", "zscore_baseline"],
        params_path=params_path,
        output_results_dir=tmp / "results",
        output_models_dir=tmp / "models",
        split_day=7,
        feature_names=FEATURES,
        experiment_log_path=tmp / "log.csv",
        system_config_path=None,
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# Cấu hình
# --------------------------------------------------------------------------- #
def test_segment_config_from_params():
    cfg = SegmentConfig.from_params({"segments": SEGMENTS_BLOCK})
    assert cfg.column == "entity_type" and cfg.enabled == ("Machine", "User")
    assert "Admin" in cfg.ignored
    assert SegmentConfig.from_params({}) is None
    assert SegmentConfig.from_params({"segments": {"enabled": []}}) is None

    with pytest.raises(ValueError, match="vừa nằm trong"):
        SegmentConfig.from_params({"segments": {"enabled": ["User"], "ignored": ["User"]}})
    with pytest.raises(ValueError, match="bị lặp"):
        SegmentConfig.from_params({"segments": {"enabled": ["User", "User"]}})


def test_repo_config_declares_machine_and_user_segments():
    cfg = SegmentConfig.from_params(yaml.safe_load(PARAMS_PATH.read_text(encoding="utf-8")))
    assert cfg is not None and cfg.enabled == ("Machine", "User")
    assert set(cfg.ignored) == {"Admin", "Service", "System", "Other"}


def test_split_segments_counts_and_bypass(caplog):
    df = pl.DataFrame({"entity_type": ["Machine"] * 5 + ["User"] * 3 + ["Admin", "Weird", None], "x": range(11)})
    cfg = SegmentConfig.from_params({"segments": SEGMENTS_BLOCK})
    with caplog.at_level(logging.WARNING, logger="ueba_benchmark.models.benchmark"):
        parts, stats = split_segments(df, cfg, context="train")

    assert {k: v.height for k, v in parts.items()} == {"Machine": 5, "User": 3}
    assert stats["n_bypassed"] == 3 and stats["bypassed"] == {"Admin": 1, "Weird": 1, "null": 1}
    assert stats["bypassed_pct"] == pytest.approx(100 * 3 / 11, abs=1e-3)
    assert stats["unknown_values"] == ["Weird", "null"]
    assert "Weird" in caplog.text  # giá trị lạ phải được cảnh báo, không lặng lẽ bỏ qua

    with pytest.raises(ValueError, match="không có dòng nào"):
        split_segments(df.filter(pl.col("entity_type") != "User"), cfg)
    with pytest.raises(ValueError, match="Thiếu cột phân khúc"):
        split_segments(df.drop("entity_type"), cfg)


# --------------------------------------------------------------------------- #
# Benchmark end-to-end
# --------------------------------------------------------------------------- #
def test_segmented_benchmark_isolates_populations(temp_artifact_dir):
    result = run(temp_artifact_dir)
    df = segmented_frame()
    test_df = df.filter(pl.col("day") > 7)

    # Chỉ Machine/User được chấm điểm; Admin/Service bị loại và được ghi lại.
    scores = result["scores"]
    assert set(scores["segment"].unique()) == {"Machine", "User"}
    assert set(scores["entity_type"].unique()) == {"Machine", "User"}
    assert scores.height == test_df.filter(pl.col("entity_type").is_in(["Machine", "User"])).height
    assert result["segments"]["train"]["bypassed"] == {"Admin": 14, "Service": 7}
    assert result["segments"]["test"]["n_bypassed"] == 9

    # Một dòng tóm tắt cho mỗi (segment, model), n_fit = số dòng train của phân khúc.
    summary = result["summary"].set_index(["segment", "model"])
    assert len(summary) == 4
    assert summary.loc[("Machine", "isolation_forest"), "n_fit"] == 7 * 40
    assert summary.loc[("User", "zscore_baseline"), "n_eval"] == 3 * 30

    # Imputer/scaler fit RIÊNG trên train của từng phân khúc (median Machine ≈ 100, User ≈ 0).
    machine = AnomalyPipeline.load(temp_artifact_dir / "models" / "Machine" / "isolation_forest.joblib")
    user = AnomalyPipeline.load(temp_artifact_dir / "models" / "User" / "isolation_forest.joblib")
    assert machine.n_train_rows_ == 280 and user.n_train_rows_ == 210
    assert machine.imputer_.statistics_[0] == pytest.approx(
        df.filter((pl.col("entity_type") == "Machine") & (pl.col("day") <= 7))["f1"].median()
    )
    assert machine.imputer_.statistics_[0] > 90 and abs(user.imputer_.statistics_[0]) < 10
    assert machine.scaler_.center_[1] > 90 and abs(user.scaler_.center_[1]) < 10

    # *_pct xếp hạng TRONG phân khúc: mỗi phân khúc có đủ dải [0, 1].
    for seg in ("Machine", "User"):
        pct = scores.filter(pl.col("segment") == seg)["isolation_forest_pct"]
        assert pct.min() == 0.0 and pct.max() == 1.0

    # Artifact có cột segment; manifest & log ghi đủ.
    results_dir = temp_artifact_dir / "results"
    overlap = pd.read_csv(results_dir / "model_topk_overlap.csv")
    assert list(overlap.columns[:2]) == ["segment", "model"] and len(overlap) == 4
    assert "segment" in pd.read_csv(results_dir / "benchmark_summary.csv").columns
    manifest = json.loads((results_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["segments"]["enabled"] == ["Machine", "User"]
    assert set(manifest["extra"]["alert_rate_drift_pp"]) == {
        "Machine/isolation_forest", "Machine/zscore_baseline", "User/isolation_forest", "User/zscore_baseline"
    }
    log = pd.read_csv(temp_artifact_dir / "log.csv")
    assert sorted(log["segment"]) == ["Machine", "Machine", "User", "User"]


def test_benchmark_without_segments_keeps_single_population(temp_artifact_dir):
    """Không có khối segments (hoặc use_segments=False) ⇒ một mô hình chung, segment = "all"."""
    result = run(temp_artifact_dir, segments=None)
    assert result["segments"] is None
    assert set(result["summary"]["segment"]) == {ALL_SEGMENT}
    assert result["scores"].height == segmented_frame().filter(pl.col("day") > 7).height
    assert (temp_artifact_dir / "models" / ALL_SEGMENT / "isolation_forest.joblib").is_file()

    forced = run(temp_artifact_dir / "forced", use_segments=False)
    assert forced["segments"] is None and set(forced["summary"]["segment"]) == {ALL_SEGMENT}


def test_benchmark_with_labels_reports_section_5_4_metrics(temp_artifact_dir):
    """File nhãn ⇒ benchmark_summary có PR-AUC/ROC-AUC/P@k/Recall tại ngân sách, tính trong từng phân khúc."""
    temp_artifact_dir.mkdir(parents=True, exist_ok=True)
    # 3 dòng User ở tập test (day > 7) là dị biệt; Machine không có nhãn dương ⇒ NaN; 1 dòng train bị cảnh báo.
    labels_path = temp_artifact_dir / "labels.csv"
    pd.DataFrame(
        {
            "DomainName": ["DOM"] * 4,
            "UserName": ["User1", "User2", "User3", "User1"],
            "day": [8, 9, 10, 3],
            "scenario": ["brute_force", "off_hours", "lateral_fanout", "brute_force"],
        }
    ).to_csv(labels_path, index=False)
    assert load_labels(labels_path)["label"].to_list() == [1, 1, 1, 1]

    result = run(temp_artifact_dir, labels_path=labels_path, precision_ks=(10, 50), daily_budgets=(1, 5))
    summary = result["summary"].set_index(["segment", "model"])
    for col in ("pr_auc", "roc_auc", "precision_at_10", "precision_at_50", "recall_at_budget",
                "recall_at_1_per_day", "recall_at_5_per_day", "alerts_per_day_mean"):
        assert col in summary.columns

    user = summary.loc[("User", "isolation_forest")]
    assert user["n_positive"] == 3
    assert 0.0 <= user["pr_auc"] <= 1.0 and 0.0 <= user["roc_auc"] <= 1.0
    assert user["recall_at_5_per_day"] >= user["recall_at_1_per_day"]
    assert np.isnan(summary.loc[("Machine", "isolation_forest"), "pr_auc"])
    assert summary.loc[("Machine", "isolation_forest"), "n_positive"] == 0

    assert int(result["scores"]["label"].sum()) == 3
    stats = result["manifest"]["extra"]["labels"]
    assert stats["n_positive_in_test"] == 3 and stats["n_positive_in_train"] == 1
    assert stats["n_positive_unmatched"] == 0

    on_disk = pd.read_csv(temp_artifact_dir / "results" / "benchmark_summary.csv")
    assert "pr_auc" in on_disk.columns


def test_benchmark_without_labels_has_no_labeled_columns(temp_artifact_dir):
    result = run(temp_artifact_dir)
    assert "pr_auc" not in result["summary"].columns
    assert "alerts_per_day_mean" in result["summary"].columns
    assert "label" not in result["scores"].columns
