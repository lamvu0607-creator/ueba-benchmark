"""
Module Metrics - Chỉ số đánh giá cho benchmark phát hiện dị biệt UEBA.

Benchmark chạy **label-free** khi không có nhãn; khi truyền file nhãn (``labels_path`` — bất thường
tổng hợp tiêm theo mục 5.2) thì nhóm chỉ số cần nhãn bên dưới được tính và ghi vào benchmark_summary.

Nhóm label-free (dùng ngay trong Tuần 3):
  * ``alert_rate``        — tỷ lệ cảnh báo của ``predict()`` (đối chiếu ngân sách ``contamination``),
  * ``budget_flags``      — ngân sách cảnh báo theo xếp hạng điểm (đúng ``budget_ratio`` theo thiết kế),
  * ``top_k_indices``     — K dòng điểm cao nhất, tie-break **ổn định** theo thứ tự dòng,
  * ``topk_overlap``      — mức trùng nhau của Top-K giữa 2 mô hình (so sánh liên mô hình),
  * ``score_quantiles``   — 11 phân vị của điểm để nhìn phân bố,
  * ``pairwise_stability``— độ ổn định đa seed (Spearman + overlap trung bình),
  * ``timed``             — đo thời gian fit/score tách bạch.

Nhóm label-aware — bộ chỉ số mục 5.4 của ``docs/Tong quan de tai ueba.md`` (nhãn = bất thường tiêm):
  * ``average_precision`` — PR-AUC, chỉ số CHÍNH (hợp dữ liệu mất cân bằng hơn ROC-AUC),
  * ``roc_auc``           — chỉ số phụ, để đối chiếu tài liệu tham khảo,
  * ``precision_at_k``    — tỷ lệ đúng trong K cảnh báo điểm cao nhất (K = 10, 50, 100),
  * ``recall_at_budget``  — recall khi chỉ được cảnh báo ``budget_ratio`` số dòng,
  * ``recall_at_daily_budget`` — recall khi analyst chỉ xử lý được N cảnh báo MỖI NGÀY,
  * ``labeled_metrics``   — gom cả bộ thành một dict (một dòng của ``benchmark_summary.csv``),
  * ``pr_auc_by_subset``  — PR-AUC theo từng kịch bản, DÙNG CHUNG một tập dòng âm,
  * ``operating_point``   — precision/recall của cờ theo ngưỡng train (ngân sách chung),
  * ``campaign_recall``   — tỷ lệ chiến dịch có >= 1 dòng bị gắn cờ.
  Tỷ lệ cảnh báo (``alert_rate``) và thời gian huấn luyện/suy luận (``timed``) là phần label-free của bộ.

Ghi chú về tie-break: các mô hình như ``RuleThresholdBaseline`` có rất nhiều dòng đồng điểm.
``top_k_indices`` dùng sắp xếp **stable** trên điểm giảm dần nên trong nhóm đồng điểm thứ tự là
theo thứ tự dòng của dữ liệu — xác định và tái lập được, nhưng KHÔNG mang ý nghĩa "dòng này dị biệt
hơn dòng kia". Điều này được nêu rõ trong báo cáo thay vì che bằng tie-break ngầm.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

logger = logging.getLogger("ueba_benchmark.evaluation.metrics")

__all__ = [
    "DEFAULT_DAILY_BUDGETS",
    "DEFAULT_PRECISION_KS",
    "DEFAULT_QUANTILES",
    "alert_rate",
    "average_precision",
    "budget_flags",
    "budget_size",
    "campaign_recall",
    "daily_budget_flags",
    "labeled_metric_columns",
    "labeled_metrics",
    "operating_point",
    "pairwise_stability",
    "pr_auc_by_subset",
    "precision_at_k",
    "rank_correlation",
    "recall_at_budget",
    "recall_at_daily_budget",
    "roc_auc",
    "score_quantiles",
    "timed",
    "top_k_indices",
    "topk_overlap",
]

#: 11 phân vị mặc định: p00, p10, ..., p100.
DEFAULT_QUANTILES: Tuple[float, ...] = tuple(i / 10 for i in range(11))

#: K mặc định cho Precision@k (mục 5.4: k = 10, 50, 100).
DEFAULT_PRECISION_KS: Tuple[int, ...] = (10, 50, 100)

#: Ngân sách mặc định (số cảnh báo / ngày) cho Recall tại ngân sách ngày.
DEFAULT_DAILY_BUDGETS: Tuple[int, ...] = (10, 50, 100)


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
# Chỉ số cần NHÃN — bộ chỉ số mục 5.4 (nhãn = bất thường tổng hợp của mục 5.2)
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


def daily_budget_flags(scores: Any, days: Any, alerts_per_day: int) -> np.ndarray:
    """
    Mảng 0/1: trong MỖI ngày bật ``alerts_per_day`` dòng điểm cao nhất (ngày ít dòng hơn thì bật hết).

    Mô phỏng analyst chỉ xử lý được N cảnh báo/ngày. Tie-break stable như ``top_k_indices``.
    """
    arr_s = _as_1d(scores, "scores")
    arr_d = np.asarray(days).ravel()
    if arr_d.size != arr_s.size:
        raise ValueError(f"days ({arr_d.size}) và scores ({arr_s.size}) phải cùng độ dài.")
    if alerts_per_day <= 0:
        raise ValueError(f"alerts_per_day phải > 0, nhận được {alerts_per_day}.")
    # Sắp theo (ngày, điểm giảm dần, thứ tự dòng) rồi lấy N dòng đầu mỗi ngày.
    order = np.lexsort((np.arange(arr_s.size), -arr_s, arr_d))
    sorted_days = arr_d[order]
    starts = np.r_[0, np.flatnonzero(sorted_days[1:] != sorted_days[:-1]) + 1]
    rank_in_day = np.arange(arr_s.size) - np.repeat(starts, np.diff(np.r_[starts, arr_s.size]))
    flags = np.zeros(arr_s.size, dtype=np.int8)
    flags[order[rank_in_day < alerts_per_day]] = 1
    return flags


def recall_at_daily_budget(labels: Any, scores: Any, days: Any, alerts_per_day: int) -> float:
    """Tỷ lệ dòng dị biệt thật bắt được khi mỗi ngày chỉ được cảnh báo ``alerts_per_day`` dòng."""
    arr_s = _as_1d(scores, "scores")
    arr_l = _as_labels(labels, arr_s)
    flags = daily_budget_flags(arr_s, days, alerts_per_day)
    return float(arr_l[flags == 1].sum() / arr_l.sum())


def labeled_metric_columns(
    ks: Sequence[int] = DEFAULT_PRECISION_KS,
    daily_budgets: Sequence[int] = DEFAULT_DAILY_BUDGETS,
) -> list:
    """Tên cột (đúng thứ tự) mà ``labeled_metrics`` trả về."""
    return (
        ["n_positive", "positive_rate_pct", "pr_auc", "roc_auc"]
        + [f"precision_at_{int(k)}" for k in ks]
        + ["recall_at_budget"]
        + [f"recall_at_{int(n)}_per_day" for n in daily_budgets]
    )


def labeled_metrics(
    labels: Any,
    scores: Any,
    days: Optional[Any] = None,
    ks: Sequence[int] = DEFAULT_PRECISION_KS,
    budget_ratio: float = 0.05,
    daily_budgets: Sequence[int] = DEFAULT_DAILY_BUDGETS,
) -> Dict[str, float]:
    """
    Bộ chỉ số cần nhãn của mục 5.4 cho MỘT mô hình trên MỘT tập đánh giá.

    Trả NaN (không raise) khi chỉ số không xác định được, để benchmark vẫn ghi đủ các dòng:
    tập không có nhãn dương ⇒ mọi chỉ số NaN; ``k`` > số dòng ⇒ ``precision_at_k`` NaN;
    không có ``days`` ⇒ ``recall_at_N_per_day`` NaN.
    """
    arr_s = _as_1d(scores, "scores")
    arr_l = np.asarray(labels).ravel()
    out: Dict[str, float] = {c: float("nan") for c in labeled_metric_columns(ks, daily_budgets)}
    if arr_l.size != arr_s.size:
        raise ValueError(f"labels ({arr_l.size}) và scores ({arr_s.size}) phải cùng độ dài.")
    n_pos = int(np.count_nonzero(arr_l))
    out["n_positive"] = float(n_pos)
    out["positive_rate_pct"] = 100.0 * n_pos / arr_s.size
    if n_pos == 0:
        return out

    out["pr_auc"] = average_precision(arr_l, arr_s)
    out["roc_auc"] = roc_auc(arr_l, arr_s) if n_pos < arr_s.size else float("nan")
    for k in ks:
        if int(k) <= arr_s.size:
            out[f"precision_at_{int(k)}"] = precision_at_k(arr_l, arr_s, int(k))
    out["recall_at_budget"] = recall_at_budget(arr_l, arr_s, budget_ratio)
    if days is not None:
        for n in daily_budgets:
            out[f"recall_at_{int(n)}_per_day"] = recall_at_daily_budget(arr_l, arr_s, days, int(n))
    return out


# --------------------------------------------------------------------------- #
# So sánh baseline ↔ ML trên nhãn tiêm: PR-AUC theo tập con, điểm vận hành, recall theo chiến dịch
# --------------------------------------------------------------------------- #
def pr_auc_by_subset(labels: Any, scores: Any, subsets: Any) -> Dict[str, Dict[str, float]]:
    """
    PR-AUC cho TOÀN BỘ dòng dương và cho từng tập con dòng dương (vd. theo ``scenario``),
    mọi ô dùng CHUNG một tập dòng âm (mọi dòng nhãn 0 của tập đánh giá).

    Vì sao dùng chung tập âm: PR-AUC phụ thuộc tỷ lệ dương tính; nếu mỗi kịch bản tự chọn tập âm
    riêng thì các ô không so sánh được. Với tập âm cố định, chênh lệch giữa hai ô chỉ đến từ cách
    phương pháp xếp hạng dòng dương của kịch bản đó so với CÙNG một đám đông bình thường.
    Mỗi ô kèm ``random_pr_auc`` = n_pos / (n_pos + n_neg) — mốc ngẫu nhiên của chính ô đó.
    """
    arr_s = _as_1d(scores, "scores")
    arr_l = np.asarray(labels).ravel().astype(np.int8)
    arr_g = np.asarray(subsets, dtype=object).ravel()
    if not (arr_l.size == arr_s.size == arr_g.size):
        raise ValueError(f"labels ({arr_l.size}), scores ({arr_s.size}), subsets ({arr_g.size}) phải cùng độ dài.")
    neg = arr_l == 0
    pos = arr_l == 1
    groups = {"__all__": pos}
    for value in sorted({str(v) for v in arr_g[pos] if v is not None}):
        groups[value] = pos & (arr_g.astype(str) == value)

    out: Dict[str, Dict[str, float]] = {}
    for name, mask in groups.items():
        n_pos, n_neg = int(mask.sum()), int(neg.sum())
        cell = {"n_pos": float(n_pos), "n_neg": float(n_neg), "pr_auc": float("nan"), "random_pr_auc": float("nan")}
        if n_pos and n_neg:
            keep = mask | neg
            cell["pr_auc"] = float(average_precision_score(arr_l[keep], arr_s[keep]))
            cell["random_pr_auc"] = n_pos / (n_pos + n_neg)
        out[name] = cell
    return out


def operating_point(labels: Any, flags: Any) -> Dict[str, float]:
    """Precision / recall của một tập cờ cố định (cờ do ngưỡng train quyết định, không do nhãn)."""
    arr_l = np.asarray(labels).ravel().astype(bool)
    arr_f = np.asarray(flags).ravel().astype(bool)
    if arr_l.size != arr_f.size:
        raise ValueError(f"labels ({arr_l.size}) và flags ({arr_f.size}) phải cùng độ dài.")
    tp = int((arr_l & arr_f).sum())
    n_alerts, n_pos = int(arr_f.sum()), int(arr_l.sum())
    return {
        "n_alerts": float(n_alerts),
        "alert_rate": float(n_alerts / arr_f.size) if arr_f.size else float("nan"),
        "true_positives": float(tp),
        "precision": float(tp / n_alerts) if n_alerts else float("nan"),
        "recall": float(tp / n_pos) if n_pos else float("nan"),
    }


def campaign_recall(labels: Any, flags: Any, campaign_ids: Any) -> Dict[str, float]:
    """
    Recall theo chiến dịch: một ``campaign_id`` được tính là PHÁT HIỆN nếu ít nhất một dòng dương
    của nó bị gắn cờ (analyst chỉ cần một cảnh báo để mở điều tra cả chiến dịch).
    Dòng dương thiếu campaign_id bị bỏ qua khỏi phép đếm này.
    """
    arr_l = np.asarray(labels).ravel().astype(bool)
    arr_f = np.asarray(flags).ravel().astype(bool)
    arr_c = np.asarray(campaign_ids, dtype=object).ravel()
    if not (arr_l.size == arr_f.size == arr_c.size):
        raise ValueError("labels, flags, campaign_ids phải cùng độ dài.")
    pos = arr_l & np.array([c is not None for c in arr_c], dtype=bool)
    campaigns = {str(c) for c in arr_c[pos]}
    detected = {str(c) for c in arr_c[pos & arr_f]}
    n = len(campaigns)
    return {
        "n_campaigns": float(n),
        "n_campaigns_detected": float(len(detected)),
        "campaign_recall": float(len(detected) / n) if n else float("nan"),
    }
