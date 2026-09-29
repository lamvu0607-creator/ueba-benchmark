"""
Module Experiment Log - Ghi log thí nghiệm vào ``experiments/logs/experiment_log.csv``.

CẢNH BÁO CHỐNG MẤT DỮ LIỆU: log này từng nằm trong ``.gitignore`` nên 8 dòng kết quả của bản
benchmark trước (5 mô hình, 2 cấu hình ``train_samples`` = 5.000/10.000) suýt mất hoàn toàn —
nó là dấu vết duy nhất giúp phục hồi lại thiết kế đã mất (``model_factory`` + ~10 test).
Từ Tuần 3, file log và mọi artifact của run đều được COMMIT vào git.

Định dạng: **18 cột legacy giữ nguyên tên và thứ tự** (để dòng cũ và dòng mới nằm liền mạch trong
cùng một bảng), sau đó là các cột bổ sung của Tuần 3 (có tiền tố rõ ràng, không ghi đè cột cũ).
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.evaluation.metrics import score_quantiles

logger = logging.getLogger("ueba_benchmark.evaluation.experiment_log")

__all__ = [
    "COLUMNS",
    "LEGACY_COLUMNS",
    "LEGACY_MODEL_NAMES",
    "LEGACY_QUANTILE_PROBS",
    "NEW_COLUMNS",
    "append_experiment_log",
    "build_log_row",
    "legacy_model_name",
]

#: 18 cột của bản log cũ — KHÔNG đổi tên, KHÔNG đổi thứ tự.
LEGACY_COLUMNS = [
    "timestamp",
    "model_name",
    "train_samples",
    "train_partition_size",
    "test_samples",
    "fit_time_sec",
    "inference_time_sec",
    "contamination",
    "score_count",
    "score_mean",
    "score_std",
    "score_min",
    "score_p25",
    "score_median",
    "score_p75",
    "score_p95",
    "score_p99",
    "score_max",
]

#: Cột mới của Tuần 3 (đủ để tái lập và truy vết, số liệu chi tiết nằm ở benchmark_summary.csv).
NEW_COLUMNS = [
    "model_key",
    "seed",
    "split_strategy",
    "split_day",
    "n_features",
    "feature_set",
    "imputer",
    "scaler",
    "threshold",
    "alert_rate_pct",
    "sklearn_version",
    "git_commit",
]

COLUMNS = LEGACY_COLUMNS + NEW_COLUMNS

#: Tên CamelCase của log cũ <-> key canonical mới (giữ log cũ đọc được liền mạch).
LEGACY_MODEL_NAMES: Dict[str, str] = {
    "isolation_forest": "IsolationForest",
    "local_outlier_factor": "LocalOutlierFactor",
    "one_class_svm": "OneClassSVM",
    "zscore_baseline": "ZScoreBaseline",
    "rule_threshold_baseline": "RuleThresholdBaseline",
}

#: Phân vị của log cũ (đúng 7 giá trị: min, p25, median, p75, p95, p99, max).
LEGACY_QUANTILE_PROBS: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)

#: Khoá chống ghi trùng khi chạy lại cùng một cấu hình.
_DEDUPE_KEYS = ("model_key", "seed", "split_day", "train_samples", "test_samples", "git_commit")


def _canonical_name(name: str) -> str:
    """
    Đổi tên/alias thành key canonical.

    Với 5 mô hình chuẩn thì tra thẳng bảng (không cần import), chỉ tên/alias lạ mới gọi registry —
    nhờ vậy module log không tạo vòng import ``src.models <-> src.evaluation``.
    """
    key = str(name).strip().lower()
    if key in LEGACY_MODEL_NAMES:
        return key
    from src.models.registry import resolve_model  # import muộn để tránh vòng import

    return resolve_model(name).name


def legacy_model_name(name: str) -> str:
    """Đổi tên canonical/alias thành nhãn CamelCase dùng trong log cũ."""
    canonical = _canonical_name(name)
    return LEGACY_MODEL_NAMES.get(canonical, canonical)


def build_log_row(
    model_name: str,
    scores: Sequence[float] | np.ndarray,
    summary: Mapping[str, Any],
    split: Mapping[str, Any],
    *,
    seed: int = 42,
    feature_set: str = "core",
    git_commit: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """Dựng một dòng log: 19 cột legacy (tính từ điểm số thật) + các cột truy vết của Tuần 3."""
    scores_arr = np.asarray(scores, dtype=np.float64).ravel()
    if scores_arr.size == 0:
        raise ValueError("scores rỗng, không thể ghi log.")
    quantiles = score_quantiles(scores_arr, probs=LEGACY_QUANTILE_PROBS)

    canonical = _canonical_name(model_name)
    return {
        "timestamp": timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": LEGACY_MODEL_NAMES.get(canonical, canonical),
        "train_samples": int(summary.get("n_fit", scores_arr.size)),
        "train_partition_size": int(summary.get("n_train_partition", 0)),
        "test_samples": int(summary.get("n_eval", scores_arr.size)),
        "fit_time_sec": round(float(summary.get("fit_seconds", 0.0)), 4),
        "inference_time_sec": round(float(summary.get("score_seconds", 0.0)), 4),
        "contamination": float(summary.get("contamination", 0.05)),
        "score_count": float(scores_arr.size),
        "score_mean": round(float(scores_arr.mean()), 6),
        "score_std": round(float(scores_arr.std(ddof=1)), 6) if scores_arr.size > 1 else 0.0,
        "score_min": round(quantiles["p00"], 6),
        "score_p25": round(quantiles["p25"], 6),
        "score_median": round(quantiles["p50"], 6),
        "score_p75": round(quantiles["p75"], 6),
        "score_p95": round(quantiles["p95"], 6),
        "score_p99": round(quantiles["p99"], 6),
        "score_max": round(quantiles["p100"], 6),
        "model_key": canonical,
        "seed": int(seed),
        "split_strategy": str(split.get("strategy", "time")),
        "split_day": split.get("split_day"),
        "n_features": int(summary.get("n_features", 0)),
        "feature_set": feature_set,
        "imputer": summary.get("imputer"),
        "scaler": summary.get("scaler"),
        "threshold": float(summary.get("threshold", 0.0)),
        "alert_rate_pct": float(summary.get("alert_rate_pct", 0.0)),
        "sklearn_version": summary.get("sklearn_version"),
        "git_commit": git_commit,
    }


def _dedupe_key(record: Mapping[str, Any]) -> Tuple[str, ...]:
    """Chuẩn hoá một dòng (dict mới hoặc dòng đọc từ CSV) thành khoá so trùng."""
    return tuple(str(record.get(key, "") or "").strip() for key in _DEDUPE_KEYS)


def append_experiment_log(
    log_path: Path | str,
    rows: Sequence[Mapping[str, Any]],
    dedupe: bool = True,
) -> Path:
    """
    Ghi thêm các dòng vào log thí nghiệm (giữ nguyên các dòng cũ và thứ tự cột).

    ``dedupe=True`` bỏ qua những dòng trùng khoá ``(model_key, seed, split_day, train_samples,
    test_samples, git_commit)`` — nhờ vậy chạy lại cùng một cấu hình không làm phình log, mà vẫn
    giữ được lịch sử khi cấu hình/commit đổi.
    """
    target = Path(log_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    new_rows = [dict(row) for row in rows]

    if target.is_file():
        existing = pd.read_csv(target, dtype=str, keep_default_na=False)
        if dedupe:
            known = {_dedupe_key(record) for record in existing.to_dict("records")}
            kept = [row for row in new_rows if _dedupe_key(row) not in known]
            if len(kept) != len(new_rows):
                logger.info("Bỏ qua %d dòng đã có trong log (dedupe).", len(new_rows) - len(kept))
            new_rows = kept
        # Giữ mọi cột đã có trong file (kể cả cột do người khác thêm) và bổ sung cột mới ở cuối.
        columns = COLUMNS + [c for c in existing.columns if c not in COLUMNS]
        combined = pd.concat([existing, pd.DataFrame(new_rows, columns=columns)], ignore_index=True)
    else:
        columns = COLUMNS
        combined = pd.DataFrame(new_rows, columns=columns)

    combined = combined.reindex(columns=columns)
    combined.to_csv(target, index=False, encoding="utf-8")
    logger.info("Đã ghi %d dòng mới vào log thí nghiệm '%s' (tổng %d dòng).", len(new_rows), target, len(combined))
    return target

