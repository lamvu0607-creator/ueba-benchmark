"""
Module Evaluation - Chia tập theo thời gian và các chỉ số đánh giá label-free/label-aware.

Gói này **chưa từng tồn tại trong Git** (tài liệu cũ mô tả ``src/evaluation/*`` nhưng không có
file nào được commit). Nó được tạo lại ở Tuần 3 theo đúng thiết kế đã chốt:

  * ``split.py``          — chia train/test theo NGÀY (day <= split_day là train),
  * ``metrics.py``        — chỉ số label-free (alert rate, Precision@k, Recall@budget, top-K overlap,
                            phân vị điểm, thời gian chạy) + stub label-aware cho Tuần 4,
  * ``experiment_log.py`` — ghi log thí nghiệm tương thích 19 cột lịch sử,
  * ``manifest.py``       — run_manifest.json (seed, commit, hash dữ liệu, phiên bản thư viện).
"""

from src.evaluation.metrics import (
    DEFAULT_QUANTILES,
    alert_rate,
    average_precision,
    budget_flags,
    budget_size,
    pairwise_stability,
    precision_at_k,
    rank_correlation,
    recall_at_budget,
    roc_auc,
    score_quantiles,
    timed,
    top_k_indices,
    topk_overlap,
)
from src.evaluation.split import SplitInfo, resolve_split_day, time_split

__all__ = [
    "SplitInfo",
    "resolve_split_day",
    "time_split",
    "DEFAULT_QUANTILES",
    "alert_rate",
    "average_precision",
    "budget_flags",
    "budget_size",
    "pairwise_stability",
    "precision_at_k",
    "rank_correlation",
    "recall_at_budget",
    "roc_auc",
    "score_quantiles",
    "timed",
    "top_k_indices",
    "topk_overlap",
]
