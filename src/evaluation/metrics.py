"""
Module Metrics - Chỉ số đánh giá cho benchmark phát hiện dị biệt UEBA.

Tuần 3 chạy **label-free**: nhãn tấn công thật (``redteam.txt`` của LANL) thuộc phạm vi Tuần 4,
nên các chỉ số cần nhãn ở đây đã được cài đặt + kiểm thử nhưng **chưa** dùng trên dữ liệu thật.

Nhóm label-free (dùng ngay trong Tuần 3):
  * ``alert_rate``        — tỷ lệ cảnh báo của ``predict()`` (đối chiếu ngân sách ``contamination``),
  * ``budget_flags``      — ngân sách cảnh báo theo xếp hạng điểm (đúng ``budget_ratio`` theo thiết kế),
  * ``top_k_indices``     — K dòng điểm cao nhất, tie-break **ổn định** theo thứ tự dòng,
  * ``topk_overlap``      — mức trùng nhau của Top-K giữa 2 mô hình (so sánh liên mô hình),
  * ``score_quantiles``   — 11 phân vị của điểm để nhìn phân bố,
  * ``pairwise_stability``— độ ổn định đa seed (Spearman + overlap trung bình),
  * ``timed``             — đo thời gian fit/score tách bạch.

Nhóm label-aware (stub cho Tuần 4 — chữ ký hàm đã khoá):
  * ``precision_at_k``, ``recall_at_budget``, ``roc_auc``, ``average_precision``.

Ghi chú về tie-break: các mô hình như ``RuleThresholdBaseline`` có rất nhiều dòng đồng điểm.
``top_k_indices`` dùng sắp xếp **stable** trên điểm giảm dần nên trong nhóm đồng điểm thứ tự là
theo thứ tự dòng của dữ liệu — xác định và tái lập được, nhưng KHÔNG mang ý nghĩa "dòng này dị biệt
hơn dòng kia". Điều này được nêu rõ trong báo cáo thay vì che bằng tie-break ngầm.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, Sequence, Tuple

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

logger = logging.getLogger("ueba_benchmark.evaluation.metrics")

__all__ = [
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

#: 11 phân vị mặc định: p00, p10, ..., p100.
DEFAULT_QUANTILES: Tuple[float, ...] = tuple(i / 10 for i in range(11))


def _as_1d(values: Any, name: str = "values") -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} rỗng.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} chứa NaN/Inf.")
    return arr


def alert_rate(flags: Any) -> float:
    """Tỷ lệ cảnh báo (0..1) của mảng nhãn 0/1."""
    arr = np.asarray(flags).ravel()
    if arr.size == 0:
        raise ValueError("flags rỗng.")
    return float(np.count_nonzero(arr) / arr.size)


def budget_size(n_samples: int, budget_ratio: float) -> int:
    """Số lượng cảnh báo của ngân sách: ``round(budget_ratio * n)``, tối thiểu 1, tối đa ``n``."""
    if not 0.0 < float(budget_ratio) <= 1.0:
        raise ValueError(f"budget_ratio phải nằm trong (0, 1], nhận được {budget_ratio}.")
    if n_samples <= 0:
        raise ValueError(f"n_samples phải > 0, nhận được {n_samples}.")
    return int(min(max(round(float(budget_ratio) * n_samples), 1), n_samples))


def top_k_indices(scores: Any, k: int) -> np.ndarray:
    """
    Chỉ số của ``k`` dòng có điểm cao nhất (điểm CAO = DỊ BIỆT).

    Dùng sắp xếp stable nên nhóm đồng điểm giữ nguyên thứ tự dòng -> tái lập được 100%.
    """
    arr = _as_1d(scores, "scores")
    if k <= 0:
        raise ValueError(f"k phải > 0, nhận được {k}.")
    if k > arr.size:
        raise ValueError(f"k={k} lớn hơn số dòng {arr.size}.")
    order = np.argsort(-arr, kind="stable")
    return order[:k]


def budget_flags(scores: Any, budget_ratio: float = 0.05) -> np.ndarray:
    """Mảng 0/1 với đúng ``budget_size`` vị trí điểm cao nhất được bật."""
    arr = _as_1d(scores, "scores")
    flags = np.zeros(arr.size, dtype=np.int8)
    flags[top_k_indices(arr, budget_size(arr.size, budget_ratio))] = 1
    return flags


def topk_overlap(scores_a: Any, scores_b: Any, k: int) -> float:
    """Tỷ lệ trùng nhau của Top-K giữa hai mô hình: ``|A ∩ B| / k`` (1.0 = giống hệt)."""
    arr_a, arr_b = _as_1d(scores_a, "scores_a"), _as_1d(scores_b, "scores_b")
    if arr_a.size != arr_b.size:
        raise ValueError(f"Hai mảng điểm phải cùng độ dài: {arr_a.size} vs {arr_b.size}.")
    set_a = set(top_k_indices(arr_a, k).tolist())
    set_b = set(top_k_indices(arr_b, k).tolist())
    return float(len(set_a & set_b) / k)


def rank_correlation(scores_a: Any, scores_b: Any) -> float:
    """Hệ số tương quan hạng Spearman giữa hai mảng điểm (điểm bằng nhau hết -> 1.0)."""
    arr_a, arr_b = _as_1d(scores_a, "scores_a"), _as_1d(scores_b, "scores_b")
    if arr_a.size != arr_b.size:
        raise ValueError(f"Hai mảng điểm phải cùng độ dài: {arr_a.size} vs {arr_b.size}.")
    if np.all(arr_a == arr_a[0]) and np.all(arr_b == arr_b[0]):
        return 1.0
    rho = float(spearmanr(arr_a, arr_b).statistic)
    return 1.0 if np.isnan(rho) else rho


def score_quantiles(scores: Any, probs: Sequence[float] = DEFAULT_QUANTILES) -> Dict[str, float]:
    """11 phân vị của điểm: ``{"p00": ..., "p10": ..., ..., "p100": ...}``."""
    arr = _as_1d(scores, "scores")
    values = np.quantile(arr, list(probs))
    return {f"p{int(round(float(p) * 100)):02d}": float(v) for p, v in zip(probs, values)}


def pairwise_stability(score_runs: Sequence[Any], k: int) -> Dict[str, float]:
    """
    Độ ổn định đa seed: Spearman và Top-K overlap trung bình/bé nhất giữa mọi cặp lần chạy.

    ``score_runs`` là danh sách mảng điểm của cùng một mô hình với các seed khác nhau
    (LOF/OCSVM lấy mẫu con theo seed nên luôn cần đo mức dao động này).
    """
    if len(score_runs) < 2:
        raise ValueError("Cần ít nhất 2 lần chạy để đo độ ổn định.")

    rhos, overlaps = [], []
    for i in range(len(score_runs)):
        for j in range(i + 1, len(score_runs)):
            rhos.append(rank_correlation(score_runs[i], score_runs[j]))
            overlaps.append(topk_overlap(score_runs[i], score_runs[j], k))
    return {
        "n_runs": float(len(score_runs)),
        "spearman_mean": float(np.mean(rhos)),
        "spearman_min": float(np.min(rhos)),
        "topk_overlap_mean": float(np.mean(overlaps)),
        "topk_overlap_min": float(np.min(overlaps)),
    }


def timed(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Tuple[Any, float]:
    """Chạy ``func`` và trả ``(kết quả, số giây)`` — đo thời gian mà không trộn vào logic."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    return result, time.perf_counter() - start


# --------------------------------------------------------------------------- #
# Chỉ số cần NHÃN (đã cài đặt + test, dùng chính thức từ Tuần 4 với redteam.txt)
# --------------------------------------------------------------------------- #
def _as_labels(labels: Any, scores: np.ndarray) -> np.ndarray:
    arr = np.asarray(labels).ravel()
    if arr.size != scores.size:
        raise ValueError(f"labels ({arr.size}) và scores ({scores.size}) phải cùng độ dài.")
    unique = sorted(set(np.unique(arr).tolist()))
    if not set(unique) <= {0, 1}:
        raise ValueError(f"labels phải là 0/1, nhận được các giá trị {unique}.")
    if unique == [0]:
        raise ValueError("Tập đánh giá không có nhãn dương (1) -> không tính được chỉ số cần nhãn.")
    return arr.astype(np.int8)


def precision_at_k(labels: Any, scores: Any, k: int) -> float:
    """Tỷ lệ dòng đúng là dị biệt thật trong K dòng điểm cao nhất."""
    arr_s = _as_1d(scores, "scores")
    arr_l = _as_labels(labels, arr_s)
    return float(arr_l[top_k_indices(arr_s, k)].sum() / k)


def recall_at_budget(labels: Any, scores: Any, budget_ratio: float = 0.05) -> float:
    """Tỷ lệ dòng dị biệt thật được bắt trong ngân sách cảnh báo."""
    arr_s = _as_1d(scores, "scores")
    arr_l = _as_labels(labels, arr_s)
    n_pos = int(arr_l.sum())
    hits = int(arr_l[budget_flags(arr_s, budget_ratio) == 1].sum())
    return float(hits / n_pos)


def roc_auc(labels: Any, scores: Any) -> float:
    """AUC của ROC (xác suất điểm của dòng dị biệt cao hơn dòng bình thường)."""
    arr_s = _as_1d(scores, "scores")
    arr_l = _as_labels(labels, arr_s)
    return float(roc_auc_score(arr_l, arr_s))


def average_precision(labels: Any, scores: Any) -> float:
    """Average Precision (diện tích dưới đường Precision-Recall) — phù hợp dữ liệu mất cân bằng."""
    arr_s = _as_1d(scores, "scores")
    arr_l = _as_labels(labels, arr_s)
    return float(average_precision_score(arr_l, arr_s))
