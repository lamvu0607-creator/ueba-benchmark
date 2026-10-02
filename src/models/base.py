"""
Module Base Model - Hợp đồng giao diện chung (contract) cho mọi mô hình phát hiện dị biệt UEBA.

Toàn bộ mô hình (3 thuật toán sklearn + 3 baseline label-free) đều kế thừa ``BaseAnomalyModel``
nên dùng chung đúng một API: ``fit`` / ``score`` / ``predict`` / ``score_rank_pct`` /
``get_metadata`` / ``save`` / ``load``. Ba bất biến bắt buộc:

1. ``score(X)`` trả điểm dị biệt theo quy ước **CAO = DỊ BIỆT** (sklearn vốn ngược dấu:
   ``-decision_function``). Quy ước này được khoá bằng test ``test_score_direction`` vì
   đây từng là nơi phát sinh lỗi đảo thứ hạng (bảng xếp hạng bị đảo).
2. ``predict(X)`` trả nhãn 0/1 theo ``threshold_`` được fit **trên chính tập huấn luyện**:
   ``threshold_ = quantile(score(X_fit), 1 - contamination)``. Nhờ vậy tỷ lệ cảnh báo (alert rate)
   của mọi mô hình đều xấp xỉ ``contamination`` và so sánh được với nhau, không phụ thuộc
   ngưỡng offset nội bộ của sklearn (đo thực tế: OneClassSVM ``nu=0.05`` nhưng ``predict()``
   của sklearn cho 8% cảnh báo).
3. ``get_metadata()`` trả đủ dấu vết để tái lập: tên mô hình, tham số, ngưỡng, ``n_fit``, phiên bản sklearn.
"""

from __future__ import annotations

import inspect
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Sequence, Tuple, Type

import joblib
import numpy as np
from sklearn import __version__ as SKLEARN_VERSION
from sklearn.base import BaseEstimator

logger = logging.getLogger("ueba_benchmark.models.base")

__all__ = [
    "BaseAnomalyModel",
    "SKLEARN_VERSION",
    "as_float_matrix",
    "validate_estimator_params",
]

#: Nhãn phiên bản payload khi lưu mô hình ra đĩa (để phát hiện file cũ/không tương thích).
MODEL_PAYLOAD_FORMAT = "ueba-anomaly-model/1"


def as_float_matrix(X: Any, context: str = "dữ liệu đầu vào") -> np.ndarray:
    """
    Chuẩn hoá đầu vào thành mảng ``float64`` 2 chiều ``(n_samples, n_features)``.

    Chấp nhận ``numpy.ndarray``, ``polars.DataFrame/Series``, ``pandas.DataFrame/Series``.
    Từ chối: mảng 1 chiều, mảng rỗng và mọi giá trị NaN/Inf — lỗi impute phải do
    ``AnomalyPipeline`` xử lý, không được âm thầm lọt vào mô hình.
    """
    if isinstance(X, np.ndarray):
        arr = X
    elif hasattr(X, "to_numpy"):
        arr = X.to_numpy()
    else:
        arr = np.asarray(X)

    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{context}: cần mảng 2 chiều (n_samples, n_features), nhận được {arr.ndim} chiều.")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{context}: mảng rỗng với shape={arr.shape}.")

    finite_mask = np.isfinite(arr)
    if not finite_mask.all():
        n_bad = int((~finite_mask).sum())
        raise ValueError(
            f"{context}: có {n_bad} giá trị NaN/Inf. Hãy để AnomalyPipeline impute trước khi chấm điểm "
            "(các NULL hợp lệ như failure_locked_out_share / interarrival_dt_mean / delta_t_cv "
            "KHÔNG được fill 0 một cách mù quáng)."
        )
    return np.ascontiguousarray(arr)


def validate_estimator_params(
    estimator_cls: Type[BaseEstimator],
    params: Dict[str, Any],
    context: str = "",
) -> Dict[str, Any]:
    """
    Kiểm tra tham số trước khi khởi tạo estimator sklearn: key lạ -> lỗi rõ ràng.

    Trước đây ``load_model_params`` đẩy thẳng mọi key trong YAML vào sklearn nên key sai/đổi tên
    chỉ lộ ra bằng ``TypeError`` khó đọc ở giữa pipeline.
    """
    signature = inspect.signature(estimator_cls.__init__)
    allowed = [
        name
        for name, param in signature.parameters.items()
        if name != "self" and param.kind is not inspect.Parameter.VAR_KEYWORD
    ]
    unknown = sorted(set(params) - set(allowed))
    if unknown:
        prefix = f" ({context})" if context else ""
        raise ValueError(
            f"Tham số không hợp lệ cho {estimator_cls.__name__}{prefix}: {unknown}. "
            f"Tham số được phép: {sorted(allowed)}."
        )
    return dict(params)


class BaseAnomalyModel(ABC):
    """
    Lớp cơ sở trừu tượng cho mọi mô hình phát hiện dị biệt của dự án.

    Lớp con bắt buộc:
      * khai báo ``name`` (key canonical snake_case dùng cho registry/CLI),
      * cài đặt ``_anomaly_score(X)`` với quy ước **CAO = DỊ BIỆT**.

    Lớp con override thêm khi cần:
      * ``estimator_class`` (estimator sklearn tương ứng -> bật whitelist tham số),
      * ``_fit_estimator`` (nếu không dùng estimator sklearn, ví dụ các baseline),
      * ``default_max_train_samples`` (mặc định ngân sách lấy mẫu để fit),
      * ``requires_scaling`` (đánh dấu mô hình cần scale đặc trưng, dùng cho tài liệu/log).
    """

    #: Key canonical (snake_case) — dùng cho registry, CLI ``--models`` và tên cột kết quả.
    name: ClassVar[str] = "base_model"
    #: Tên gọi cũ / CamelCase còn xuất hiện trong experiment_log.csv lịch sử để tương thích ngược.
    aliases: ClassVar[Tuple[str, ...]] = ()
    #: True nếu là baseline đơn giản (không phải thuật toán học máy).
    is_baseline: ClassVar[bool] = False
    #: True nếu mô hình chỉ chính xác khi đặc trưng đã được scale (LOF/OCSVM).
    requires_scaling: ClassVar[bool] = False
    #: Estimator sklearn tương ứng (None với baseline thuần thống kê/luật).
    estimator_class: ClassVar[Optional[Type[BaseEstimator]]] = None
    #: Ngân sách lấy mẫu mặc định khi fit (None = dùng toàn bộ tập train).
    default_max_train_samples: ClassVar[Optional[int]] = None

    def __init__(
        self,
        contamination: float = 0.05,
        random_state: int = 42,
        max_train_samples: Optional[int] = None,
        **native_params: Any,
    ):
        contamination = float(contamination)
        if not 0.0 < contamination < 1.0:
            raise ValueError(f"contamination phải nằm trong (0, 1), nhận được {contamination}.")

        if max_train_samples is None:
            resolved_limit: Optional[int] = self.default_max_train_samples
        else:
            resolved_limit = int(max_train_samples)
            if resolved_limit < 0:
                raise ValueError("max_train_samples phải >= 0 (0 nghĩa là không giới hạn).")
            resolved_limit = resolved_limit or None

        self.contamination = contamination
        self.random_state = random_state
        self.max_train_samples = resolved_limit
        if self.estimator_class is not None:
            self.native_params = validate_estimator_params(
                self.estimator_class, native_params, context=self.name
            )
        else:
            self.native_params = dict(native_params)

        self.estimator_: Optional[BaseEstimator] = None
        self.threshold_: Optional[float] = None
        self.fit_scores_: Optional[np.ndarray] = None
        self.subsample_indices_: Optional[np.ndarray] = None
        self.n_fit_: int = 0
        self.n_features_in_: int = 0
        self.feature_names_in_: Optional[Sequence[str]] = None
        self.is_fitted_: bool = False

    # ------------------------------------------------------------------ #
    # Tiện ích lớp
    # ------------------------------------------------------------------ #
    @classmethod
    def native_param_names(cls) -> Tuple[str, ...]:
        """Tên tham số gốc hợp lệ truyền xuống estimator sklearn (dùng cho test whitelist)."""
        if cls.estimator_class is None:
            return ()
        signature = inspect.signature(cls.estimator_class.__init__)
        return tuple(
            name
            for name, param in signature.parameters.items()
            if name != "self" and param.kind is not inspect.Parameter.VAR_KEYWORD
        )

    def _check_fitted(self, method: str) -> None:
        if not self.is_fitted_:
            raise RuntimeError(
                f"Mô hình '{self.name}' chưa được fit. Hãy gọi fit(X) hoặc dùng "
                f"AnomalyPipeline trước khi gọi {method}()."
            )

    # ------------------------------------------------------------------ #
    # Vòng đời mô hình (API công khai dùng chung cho cả 6 mô hình)
    # ------------------------------------------------------------------ #
    def fit(self, X: Any) -> "BaseAnomalyModel":
        """
        Fit mô hình trên **tập huấn luyện** (không bao giờ fit trên tập đánh giá).

        Nếu số dòng vượt ``max_train_samples`` thì lấy mẫu con ngẫu nhiên **có seed**
        (LOF/One-Class SVM mặc định 20.000 dòng; Isolation Forest dùng toàn bộ tập train).
        Ngưỡng cảnh báo ``threshold_`` được suy ra từ chính tập đã fit.
        """
        X_arr = as_float_matrix(X, context=f"[{self.name}] tập huấn luyện")
        X_fit, indices = self._subsample(X_arr)

        self.estimator_ = self._fit_estimator(X_fit)
        self.n_fit_ = int(X_fit.shape[0])
        self.n_features_in_ = int(X_fit.shape[1])
        self.subsample_indices_ = indices
        # Bật cờ trước khi chấm điểm: ngưỡng cảnh báo được suy ra từ chính tập đã fit.
        self.is_fitted_ = True

        self.fit_scores_ = self.score(X_fit)
        self.threshold_ = self._resolve_threshold(self.fit_scores_)

        logger.info(
            "[%s] fit xong: n_fit=%s / n_train=%s, %d đặc trưng, threshold=%.4f (mục tiêu %.2f%% cảnh báo)",
            self.name,
            f"{self.n_fit_:,}",
            f"{X_arr.shape[0]:,}",
            self.n_features_in_,
            self.threshold_,
            self.contamination * 100,
        )
        return self

    def score(self, X: Any) -> np.ndarray:
        """
        Điểm dị biệt cho từng dòng: **CAO = DỊ BIỆT** (bất biến số 1 của dự án).

        Giá trị là điểm thô theo đúng thuật toán (đã đổi dấu cho sklearn), không bị ép về [0, 1]
        để còn debug được "khoảng cách dị biệt". Muốn so sánh liên mô hình -> ``score_rank_pct``.
        """
        X_arr = as_float_matrix(X, context=f"[{self.name}] tập chấm điểm")
        self._check_fitted("score")
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"[{self.name}] sai số đặc trưng: mô hình được fit với {self.n_features_in_} đặc trưng "
                f"nhưng nhận được {X_arr.shape[1]}. Hãy dùng AnomalyPipeline để bảo đảm đúng tập đặc trưng."
            )

        scores = np.asarray(self._anomaly_score(X_arr), dtype=np.float64).ravel()
        if scores.shape[0] != X_arr.shape[0]:
            raise ValueError(
                f"[{self.name}] số điểm ({scores.shape[0]}) không khớp số dòng đầu vào ({X_arr.shape[0]})."
            )
        if not np.all(np.isfinite(scores)):
            raise ValueError(f"[{self.name}] điểm dị biệt chứa NaN/Inf.")
        return scores

    def predict(self, X: Any) -> np.ndarray:
        """
        Nhãn nhị phân (1 = dị biệt, 0 = bình thường) theo ``threshold_`` đã fit (bất biến số 2).

        Tỷ lệ nhãn 1 xấp xỉ ``contamination`` nên "alert rate" của cả 6 mô hình so sánh được trực tiếp.
        """
        self._check_fitted("predict")
        return (self.score(X) >= float(self.threshold_)).astype(np.int8)

    def score_rank_pct(self, X: Any, reference_scores: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Percentile hạng của điểm dị biệt trong [0, 1] so với một tập tham chiếu.

        ``score()`` giữ nguyên thang đo gốc (cần cho debug), còn hàm này phục vụ so sánh
        **liên mô hình** vì LOF/OCSVM/Isolation Forest có thang đo rất khác nhau. Benchmark
        truyền ``reference_scores`` = điểm của cả tập đánh giá để mọi mô hình được xếp hạng
        trên cùng một quần thể; mặc định (None) dùng ``fit_scores_``.
        """
        scores = self.score(X)
        if reference_scores is None:
            reference = self.fit_scores_
            if reference is None:
                raise RuntimeError(f"[{self.name}] chưa có điểm tham chiếu, hãy fit() trước.")
        else:
            reference = np.asarray(reference_scores, dtype=np.float64).ravel()
            if not np.all(np.isfinite(reference)):
                raise ValueError("reference_scores chứa NaN/Inf.")

        if reference.size == 0:
            raise ValueError("reference_scores rỗng, không thể tính percentile hạng.")
        if reference.size == 1:
            return np.ones_like(scores)

        sorted_ref = np.sort(reference)
        ranks = np.searchsorted(sorted_ref, scores, side="right")
        return np.clip((ranks - 1) / (sorted_ref.size - 1), 0.0, 1.0)

    def get_metadata(self) -> Dict[str, Any]:
        """Dấu vết tái lập của mô hình (đi kèm mọi artifact .joblib và run_manifest.json)."""
        return {
            "model_name": self.name,
            "model_class": type(self).__name__,
            "is_baseline": self.is_baseline,
            "requires_scaling": self.requires_scaling,
            "contamination": self.contamination,
            "random_state": self.random_state,
            "max_train_samples": self.max_train_samples,
            "native_params": dict(self.native_params),
            "n_fit": self.n_fit_,
            "n_features_in": self.n_features_in_,
            "feature_names_in": list(self.feature_names_in_) if self.feature_names_in_ else None,
            "threshold": self.threshold_,
            "is_fitted": self.is_fitted_,
            "sklearn_version": SKLEARN_VERSION,
        }

    def save(self, path: Path | str) -> Path:
        """Lưu mô hình (kèm metadata) ra file .joblib."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"format": MODEL_PAYLOAD_FORMAT, "model": self, "metadata": self.get_metadata()},
            target,
        )
        logger.info("Đã lưu mô hình '%s' tại '%s'.", self.name, target)
        return target

    @classmethod
    def load(cls, path: Path | str) -> "BaseAnomalyModel":
        """Nạp mô hình từ file .joblib, kiểm tra đúng định dạng và đúng lớp cơ sở."""
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"Không tìm thấy file mô hình: '{source}'.")

        payload = joblib.load(source)
        if not isinstance(payload, dict) or "model" not in payload:
            raise ValueError(
                f"File '{source}' không đúng định dạng {MODEL_PAYLOAD_FORMAT} "
                "(có thể là mô hình trần do phiên bản benchmark cũ lưu)."
            )
        model = payload["model"]
        if not isinstance(model, BaseAnomalyModel):
            raise TypeError(f"Đối tượng trong '{source}' không phải BaseAnomalyModel: {type(model)!r}.")
        return model

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted_ else "unfitted"
        return (
            f"{type(self).__name__}(name='{self.name}', contamination={self.contamination}, "
            f"n_fit={self.n_fit_}, threshold={self.threshold_}, {status})"
        )

    # ------------------------------------------------------------------ #
    # Phần lớp con cài đặt / override
    # ------------------------------------------------------------------ #
    def _resolve_threshold(self, fit_scores: np.ndarray) -> float:
        """
        Ngưỡng cảnh báo mặc định = phân vị ``(1 - contamination)`` trên **tập fit**.

        Nhờ vậy "alert rate" xấp xỉ ``contamination`` cho mọi mô hình và so sánh được.
        Lớp con có thể override để dùng ngưỡng tuyệt đối (ví dụ rule threshold).
        """
        return float(np.quantile(fit_scores, 1.0 - self.contamination))

    def _subsample(self, X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Lấy mẫu con có seed để fit; trả (X_fit, chỉ số dòng được chọn) hoặc (X, None) nếu dùng hết."""
        limit = self.max_train_samples
        if limit is None or X.shape[0] <= limit:
            return X, None

        rng = np.random.default_rng(self.random_state if self.random_state is not None else 42)
        indices = np.sort(rng.choice(X.shape[0], size=int(limit), replace=False))
        logger.info(
            "[%s] tập train %s dòng > ngân sách %s -> lấy mẫu con %s dòng (seed=%s).",
            self.name,
            f"{X.shape[0]:,}",
            f"{limit:,}",
            f"{len(indices):,}",
            self.random_state,
        )
        return X[indices], indices

    def _fit_estimator(self, X: np.ndarray) -> Optional[BaseEstimator]:
        """
        Dựng estimator sklearn từ ``native_params`` rồi fit.

        Các baseline thuần thống kê/luật override hàm này để tính statistic trên tập fit
        mà không cần estimator sklearn.
        """
        estimator = self._build_estimator()
        if estimator is not None:
            estimator.fit(X)
        return estimator

    def _build_estimator(self) -> Optional[BaseEstimator]:
        """Khởi tạo estimator sklearn tương ứng (mặc định None dành cho baseline)."""
        return None

    @abstractmethod
    def _anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        Điểm dị biệt thô do lớp con cài đặt, **CAO = DỊ BIỆT**.

        Với sklearn (định nghĩa ngược dấu): ``-estimator.decision_function(X)``
        hoặc ``-estimator.score_samples(X)``.
        """
        raise NotImplementedError




