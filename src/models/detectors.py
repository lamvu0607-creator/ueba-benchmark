"""
Module Detectors - Bọc 3 thuật toán phát hiện dị biệt (giao diện PyOD) theo hợp đồng ``BaseAnomalyModel``.

==========================  ======================================================================
``IsolationForestDetector``  ``pyod.models.iforest.IForest`` (bọc sklearn IsolationForest)
``LocalOutlierFactorDetector``  ``HNSWLOF`` — LOF trên láng giềng xấp xỉ ``hnswlib``
``OneClassSVMDetector``      ``NystroemSGDOCSVM`` — OCSVM nhân RBF xấp xỉ (Nystroem + SGD)
==========================  ======================================================================

Cả 3 lớp đều:
  * kế thừa ``BaseAnomalyModel`` (API ``fit`` / ``score`` / ``predict`` / ``score_rank_pct``),
  * dùng NGUYÊN ``decision_function`` của PyOD — vốn đã là **CAO = DỊ BIỆT** — nên KHÔNG đảo dấu thêm,
  * truyền rõ ``contamination`` (mặc định PyOD là 0.1) và ``n_jobs`` (mặc định PyOD là 1),
  * khai báo ``estimator_class`` để tham số YAML sai key bị chặn ngay khi khởi tạo,
  * đánh dấu ``requires_scaling = True`` (đặc trưng UEBA lệch nặng: ``interarrival_dt_mean``
    max 86.400 giây nằm cạnh các ratio trong [0, 1]).

Ngưỡng cảnh báo vẫn do ``BaseAnomalyModel`` tính (phân vị trên tập fit); ``predict``/``threshold_``
của PyOD không được dùng. Cả 3 mặc định fit trên TOÀN BỘ tập train (``default_max_train_samples = None``).
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
from pyod import __version__ as PYOD_VERSION
from pyod.models.base import BaseDetector
from pyod.models.iforest import IForest

from src.models.base import BaseAnomalyModel
from src.models.pyod_detectors import HNSWLOF, NystroemSGDOCSVM

logger = logging.getLogger("ueba_benchmark.models.detectors")

__all__ = [
    "IsolationForestDetector",
    "LocalOutlierFactorDetector",
    "OneClassSVMDetector",
    "PYOD_VERSION",
]


class _PyODDetector(BaseAnomalyModel):
    """Phần dùng chung: dựng detector PyOD với contamination/random_state/n_jobs tường minh."""

    requires_scaling = True
    default_max_train_samples = None
    #: Có truyền ``n_jobs`` cho estimator không (OCSVM xấp xỉ không có tham số này).
    _uses_n_jobs = True

    def _build_estimator(self) -> BaseDetector:
        params: Dict[str, Any] = dict(self.native_params)
        params.setdefault("contamination", self.contamination)
        params.setdefault("random_state", self.random_state)
        if self._uses_n_jobs:
            params.setdefault("n_jobs", -1)
        return self.estimator_class(**params)

    def _anomaly_score(self, X: np.ndarray) -> np.ndarray:
        # PyOD: decision_function đã là CAO = DỊ BIỆT ⇒ không đảo dấu.
        return np.asarray(self.estimator_.decision_function(X), dtype=np.float64)

    def _fit_set_scores(self, X_fit: np.ndarray) -> np.ndarray:
        """
        Dùng ``decision_scores_`` của PyOD = điểm của tập fit tính lúc ``fit``.

        Với IForest/OCSVM nó trùng ``score(X_fit)``; với HNSWLOF nó là LOF **bỏ chính điểm đó**
        khỏi láng giềng (leave-self-out) — chấm lại bằng ``decision_function`` sẽ gặp lại chính
        điểm ở khoảng cách 0 và làm ngưỡng quá thấp (đo thật: alert rate test 17,9% thay vì ~5%).
        """
        scores = np.asarray(self.estimator_.decision_scores_, dtype=np.float64).ravel()
        if scores.shape[0] != X_fit.shape[0] or not np.all(np.isfinite(scores)):
            raise ValueError(f"[{self.name}] decision_scores_ của PyOD không hợp lệ cho tập fit.")
        return scores

    def get_metadata(self) -> Dict[str, Any]:
        meta = super().get_metadata()
        meta["pyod_version"] = PYOD_VERSION
        return meta


class IsolationForestDetector(_PyODDetector):
    """
    Isolation Forest (tree-based) — ``pyod.models.iforest.IForest``.

    PyOD bọc đúng ``sklearn.ensemble.IsolationForest`` và trả ``-decision_function`` của sklearn,
    nên điểm trùng tuyệt đối với bản sklearn cũ khi cùng dữ liệu fit và cùng ``random_state``.
    """

    name = "isolation_forest"
    aliases = ("IsolationForest",)
    estimator_class = IForest


class LocalOutlierFactorDetector(_PyODDetector):
    """
    Local Outlier Factor (density-based) trên láng giềng xấp xỉ HNSW — ``HNSWLOF``.

    Thay cho ``sklearn.neighbors.LocalOutlierFactor(novelty=True)`` vốn phải lấy mẫu con
    (truy vấn láng giềng chính xác O(n_fit) mỗi điểm); bản HNSW fit được toàn bộ tập train.
    Thống kê dòng trùng lặp được ghi vào ``get_metadata()["duplicate_stats"]``; mặc định
    ``dedup=True`` dựng index trên dòng duy nhất (số điểm ở ``get_metadata()["n_index_points"]``).
    """

    name = "local_outlier_factor"
    aliases = ("LocalOutlierFactor",)
    estimator_class = HNSWLOF

    def get_metadata(self) -> Dict[str, Any]:
        meta = super().get_metadata()
        if self.estimator_ is not None:
            meta["duplicate_stats"] = dict(getattr(self.estimator_, "duplicate_stats_", {}))
            meta["n_index_points"] = getattr(self.estimator_, "n_index_points_", None)
        return meta


class OneClassSVMDetector(_PyODDetector):
    """
    One-Class SVM nhân RBF **xấp xỉ** — ``NystroemSGDOCSVM`` (Nystroem → SGDOneClassSVM).

    Giữ key ``one_class_svm`` / alias ``OneClassSVM`` để registry, CLI và experiment_log không đổi.
    ``nu`` chính là **ngân sách cảnh báo**: nếu YAML có ``nu`` thì nó được dùng làm
    ``contamination`` (đồng bộ 2 chiều) để ngưỡng phân vị và alert rate khớp nhau.
    """

    name = "one_class_svm"
    aliases = ("OneClassSVM",)
    estimator_class = NystroemSGDOCSVM
    _uses_n_jobs = False

    def __init__(
        self,
        contamination: float = 0.05,
        random_state: int = 42,
        max_train_samples: int | None = None,
        **native_params: Any,
    ):
        # 'nu' tương đương 'contamination' của giao diện chung -> đồng bộ 2 chiều.
        if "nu" in native_params:
            contamination = float(native_params["nu"])
        super().__init__(
            contamination=contamination,
            random_state=random_state,
            max_train_samples=max_train_samples,
            **native_params,
        )
        self.native_params.setdefault("nu", self.contamination)

    def get_metadata(self) -> Dict[str, Any]:
        meta = super().get_metadata()
        if self.estimator_ is not None:
            meta["gamma_effective"] = float(self.estimator_.gamma_)
            meta["n_components_effective"] = int(self.estimator_.nystroem_.n_components)
            meta["chunked_fit"] = bool(self.estimator_.chunked_)
        return meta
