"""
Ngưỡng theo NGÂN SÁCH CẢNH BÁO — dùng chung cho mọi phương pháp (3 baseline và 3 mô hình ML).

Quy tắc duy nhất::

    threshold = quantile_{1 − budget_ratio}( điểm của CHÍNH phương pháp đó trên TRAIN )
    flag(x)   = score(x) > threshold

Vì sao như vậy:
  * Điểm của các phương pháp có thang đo khác nhau (|z|, ECDF, decision_function ...), nên một con
    số chung như "|z| >= 3" không so sánh được. Cùng một ngân sách (vd. 1% số dòng) thì so được.
  * Phân vị lấy trên TRAIN (không phải trên test) để ngưỡng là thứ có sẵn TRƯỚC khi nhìn ngày cần
    chấm — đúng vận hành thật, và không dùng bất kỳ thống kê nào của tập đánh giá.
  * So sánh CHẶT (``>``): điểm có nhiều giá trị trùng (luật, ECDF) không thể "mua" thêm cảnh báo
    vượt ngân sách nhờ đồng hạng — tỷ lệ cờ trên train luôn <= budget_ratio. Tỷ lệ thực tế được
    ghi lại (``train_alert_rate``) để người đọc thấy phương pháp nào bị thiếu cờ vì đồng hạng.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

import numpy as np

__all__ = ["BudgetThreshold", "fit_budget_threshold", "apply_threshold", "alerts_per_day"]


@dataclass(frozen=True)
class BudgetThreshold:
    """Ngưỡng đã fit cùng các con số để truy vết (ghi vào thresholds.json)."""

    threshold: float
    budget_ratio: float
    n_train: int
    train_alert_rate: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def fit_budget_threshold(train_scores: Any, budget_ratio: float) -> BudgetThreshold:
    """Ngưỡng = phân vị ``1 − budget_ratio`` (method="higher": một giá trị điểm có thật) của điểm train."""
    scores = np.asarray(train_scores, dtype=np.float64).ravel()
    if scores.size == 0:
        raise ValueError("train_scores rỗng.")
    if not np.all(np.isfinite(scores)):
        raise ValueError("train_scores chứa NaN/Inf.")
    if not 0.0 < float(budget_ratio) < 1.0:
        raise ValueError(f"budget_ratio phải nằm trong (0, 1), nhận được {budget_ratio}.")
    threshold = float(np.quantile(scores, 1.0 - float(budget_ratio), method="higher"))
    return BudgetThreshold(
        threshold=threshold,
        budget_ratio=float(budget_ratio),
        n_train=int(scores.size),
        train_alert_rate=float(np.mean(scores > threshold)),
    )


def apply_threshold(scores: Any, threshold: BudgetThreshold | float) -> np.ndarray:
    """Mảng bool: dòng có điểm CHẶT lớn hơn ngưỡng đã fit trên train."""
    value = threshold.threshold if isinstance(threshold, BudgetThreshold) else float(threshold)
    return np.asarray(scores, dtype=np.float64).ravel() > value


def alerts_per_day(flags: Any, days: Any) -> Dict[str, float]:
    """Quy đổi cờ ra số cảnh báo/ngày (mean, median, min, max) trên các ngày có trong ``days``."""
    arr_f = np.asarray(flags).ravel().astype(bool)
    arr_d = np.asarray(days).ravel()
    if arr_f.size != arr_d.size:
        raise ValueError(f"flags ({arr_f.size}) và days ({arr_d.size}) phải cùng độ dài.")
    unique_days, inverse = np.unique(arr_d, return_inverse=True)
    if unique_days.size == 0:
        return {"n_days": 0.0, "alerts_per_day_mean": float("nan"), "alerts_per_day_median": float("nan"),
                "alerts_per_day_min": float("nan"), "alerts_per_day_max": float("nan")}
    per_day = np.bincount(inverse, weights=arr_f.astype(np.float64), minlength=unique_days.size)
    return {
        "n_days": float(unique_days.size),
        "alerts_per_day_mean": float(per_day.mean()),
        "alerts_per_day_median": float(np.median(per_day)),
        "alerts_per_day_min": float(per_day.min()),
        "alerts_per_day_max": float(per_day.max()),
    }
