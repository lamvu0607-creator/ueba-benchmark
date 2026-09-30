"""
Module Detectors - Bọc 3 thuật toán phát hiện dị biệt của scikit-learn theo giao diện chung.

Cả 3 lớp đều:
  * kế thừa ``BaseAnomalyModel`` (API ``fit`` / ``score`` / ``predict`` / ``score_rank_pct``),
  * đổi dấu sklearn để thoả bất biến **CAO = DỊ BIỆT** (``-decision_function``),
  * khai báo ``estimator_class`` để tham số YAML sai key bị chặn ngay khi khởi tạo,
  * đánh dấu ``requires_scaling = True`` (đặc trưng UEBA lệch nặng: ``interarrival_dt_mean``
    max 86.400 giây nằm cạnh các ratio trong [0, 1]).

Ghi chú ngân sách lấy mẫu: LOF và One-Class SVM mặc định fit trên 20.000 dòng để giữ
thời gian/RAM hợp lý; Isolation Forest fit trên toàn bộ tập train. Cả ``n_fit`` và
``n_eval`` luôn được log tách bạch để diễn giải kết quả đúng.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from src.models.base import BaseAnomalyModel

logger = logging.getLogger("ueba_benchmark.models.detectors")

__all__ = [
    "IsolationForestDetector",
    "LocalOutlierFactorDetector",
    "OneClassSVMDetector",
]


def negated_decision_function(estimator: BaseEstimator, X: np.ndarray) -> np.ndarray:
    """
    Đảo dấu ``decision_function`` của sklearn: điểm cao => dị biệt.

    sklearn trả giá trị ÂM cho điểm càng bất thường; nếu dùng trực tiếp, bảng xếp hạng sẽ
    bị đảo. Mọi detector dùng chung hàm này để quy ước chỉ tồn tại ở một chỗ duy nhất.
    """
    return -np.asarray(estimator.decision_function(X), dtype=np.float64)


class IsolationForestDetector(BaseAnomalyModel):
    """
    Isolation Forest (tree-based) — cô lập quan sát bằng cây chia ngẫu nhiên.

    * ``requires_scaling = True``: khoảng chia cây phụ thuộc biên độ đặc trưng.
    * Không giới hạn ``max_train_samples`` mặc định: chấm điểm/onfit theo batch nên chịu được
      toàn bộ tập train (721.612 dòng ngày 1-42).
    """

    name = "isolation_forest"
    aliases = ("IsolationForest",)
    requires_scaling = True
    estimator_class = IsolationForest
    default_max_train_samples = None

    def _build_estimator(self) -> BaseEstimator:
        params: dict[str, Any] = dict(self.native_params)
        params.setdefault("contamination", self.contamination)
        params.setdefault("random_state", self.random_state)
        params.setdefault("n_jobs", -1)
        return IsolationForest(**params)

    def _anomaly_score(self, X: np.ndarray) -> np.ndarray:
        return negated_decision_function(self.estimator_, X)


class LocalOutlierFactorDetector(BaseAnomalyModel):
    """
    Local Outlier Factor (density-based, ``novelty=True``).

    ``novelty=True`` là **bắt buộc**: nếu không, sklearn chỉ cho ``fit_predict`` trên tập
    huấn luyện và không thể chấm điểm tập đánh giá (đúng yêu cầu fit-on-train/score-on-test).
    Mặc định fit trên mẫu con 20.000 dòng (decision_function của LOF là O(n_fit)).
    """

    name = "local_outlier_factor"
    aliases = ("LocalOutlierFactor",)
    requires_scaling = True
    estimator_class = LocalOutlierFactor
    default_max_train_samples = 20_000

    def _build_estimator(self) -> BaseEstimator:
        params: dict[str, Any] = dict(self.native_params)
        params.setdefault("contamination", self.contamination)
        params.setdefault("novelty", True)
        params.setdefault("n_jobs", -1)
        estimator = LocalOutlierFactor(**params)
        if not getattr(estimator, "novelty", False):
            raise ValueError(
                "LocalOutlierFactor bắt buộc 'novelty=True' để chấm điểm được tập đánh giá "
                "(thiếu nó thì chỉ dùng được fit_predict trên chính tập fit)."
            )
        return estimator

    def _anomaly_score(self, X: np.ndarray) -> np.ndarray:
        return negated_decision_function(self.estimator_, X)


class OneClassSVMDetector(BaseAnomalyModel):
    """
    One-Class SVM (kernel RBF) — học biên bao quanh dữ liệu bình thường.

    Lưu ý ``nu`` của sklearn chính là **ngân sách cảnh báo**: nếu người dùng truyền ``nu``
    trong tham số gốc thì nó được dùng làm ``contamination`` để ngưỡng phân vị và alert rate
    khớp nhau. Không dùng trực tiếp ``estimator.predict`` vì đo thực tế ``nu=0.05`` cho ~8%
    cảnh báo (ngưỡng offset nội bộ của sklearn khác phân vị mong muốn).
    """

    name = "one_class_svm"
    aliases = ("OneClassSVM",)
    requires_scaling = True
    estimator_class = OneClassSVM
    default_max_train_samples = 20_000

    def __init__(
        self,
        contamination: float = 0.05,
        random_state: int = 42,
        max_train_samples: int | None = None,
        **native_params: Any,
    ):
        # 'nu' của sklearn tương đương 'contamination' của giao diện chung -> đồng bộ 2 chiều.
        if "nu" in native_params:
            contamination = float(native_params["nu"])
        super().__init__(
            contamination=contamination,
            random_state=random_state,
            max_train_samples=max_train_samples,
            **native_params,
        )
        self.native_params.setdefault("nu", self.contamination)

    def _build_estimator(self) -> BaseEstimator:
        params: dict[str, Any] = dict(self.native_params)
        params.setdefault("kernel", "rbf")
        params.setdefault("gamma", "scale")
        params.setdefault("nu", self.contamination)
        params.setdefault("tol", 1e-3)
        return OneClassSVM(**params)

    def _anomaly_score(self, X: np.ndarray) -> np.ndarray:
        return negated_decision_function(self.estimator_, X)
