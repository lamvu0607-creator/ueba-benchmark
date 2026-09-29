"""
Module Pipeline - Ghép ``Imputer -> Scaler -> Detector`` thành một đơn vị tái lập được.

Vì sao dự án cần pipeline chung (3 lỗi đã đo được ở bản benchmark cũ):

1. **Sai tập đặc trưng**: bản cũ lấy *mọi* cột số của parquet (18 cột, gồm cả biến thể thô
   trùng lặp như ``total_logons``/``distinct_hosts``/``rare_logon_type_count``) thay vì đúng
   **16 đặc trưng core** trong ``configs/feature_schema.yaml``.
2. **Fill NULL mù quáng**: ``fill_null(0.0)`` biến các NULL hợp lệ (``failure_locked_out_share``,
   ``interarrival_dt_mean``, ``delta_t_cv``) thành 0 — nghĩa là "0 giây giữa 2 sự kiện" và
   "không có thất bại nào", sai về mặt ngữ nghĩa. Nay dùng median **fit trên tập train**.
3. **Không scale**: ``interarrival_dt_mean`` (tới 86.400 giây) nằm cạnh các ratio trong [0, 1],
   khiến LOF/OCSVM bị chi phối bởi một đặc trưng. Nay fit scaler trên tập train rồi chỉ
   ``transform`` tập đánh giá (không rò rỉ phân phối).

Luồng bảo đảm không rò rỉ: ``imputer``/``scaler`` fit trên **toàn bộ train** (rẻ) rồi
``model.fit`` trên tập train (LOF/OCSVM tự lấy mẫu con có seed bên trong ``BaseAnomalyModel``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import joblib
import numpy as np
import polars as pl
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer, RobustScaler, StandardScaler

from src.features.schema import FeatureSchema
from src.models.base import BaseAnomalyModel

logger = logging.getLogger("ueba_benchmark.models.pipeline")

__all__ = [
    "AnomalyPipeline",
    "PIPELINE_PAYLOAD_FORMAT",
    "SCALER_REGISTRY",
    "IMPUTER_REGISTRY",
    "build_scaler",
    "build_imputer",
]

PIPELINE_PAYLOAD_FORMAT = "ueba-anomaly-pipeline/1"

#: ``name -> hàm khởi tạo`` để config YAML chỉ cần ghi chuỗi ngắn (robust/standard/quantile).
SCALER_REGISTRY: Dict[str, Callable[[int], Any]] = {
    "robust": lambda seed: RobustScaler(),
    "standard": lambda seed: StandardScaler(),
    "quantile": lambda seed: QuantileTransformer(
        n_quantiles=1000, output_distribution="normal", subsample=100_000, random_state=seed
    ),
}

IMPUTER_REGISTRY: Dict[str, Callable[[int], Any]] = {
    "median": lambda seed: SimpleImputer(strategy="median", keep_empty_features=True),
    "mean": lambda seed: SimpleImputer(strategy="mean", keep_empty_features=True),
    "zero": lambda seed: SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True),
}


def build_scaler(name: Optional[str], random_state: int = 42) -> Optional[Any]:
    """Khởi tạo scaler theo tên trong config; ``None``/``"none"`` nghĩa là không scale."""
    if name is None or str(name).lower() in {"none", "null"}:
        return None
    key = str(name).lower()
    if key not in SCALER_REGISTRY:
        raise ValueError(f"scaler không hợp lệ: {name!r}. Chỉ hỗ trợ {sorted(SCALER_REGISTRY)}.")
    return SCALER_REGISTRY[key](random_state)


def build_imputer(name: Optional[str] = "median") -> Any:
    """Khởi tạo imputer theo tên trong config (mặc định median — bền với outlier)."""
    key = str(name or "median").lower()
    if key not in IMPUTER_REGISTRY:
        raise ValueError(f"imputer không hợp lệ: {name!r}. Chỉ hỗ trợ {sorted(IMPUTER_REGISTRY)}.")
    return IMPUTER_REGISTRY[key](0)


class AnomalyPipeline:
    """
    Pipeline chuẩn: **16 đặc trưng core -> imputer -> scaler -> detector**.

    Ghi chú thiết kế:

    * ``scaler`` chỉ được áp dụng khi ``model.requires_scaling`` (Isolation Forest/LOF/OCSVM).
      Baseline luật dùng ngưỡng theo **đơn vị gốc** (``failure_ratio >= 0.5``); nếu scale,
      các ngưỡng đó mất ý nghĩa nên pipeline cố tình bỏ qua scaler cho baseline.
    * ``imputer``/``scaler`` fit trên toàn bộ tập train, ``model.fit`` nhận ma trận đã xử lý
      (LOF/OCSVM tự lấy mẫu con theo seed bên trong ``BaseAnomalyModel``).
    * Mọi bước đều có thể truy vết qua ``get_metadata()`` (tên cột, loại scaler/imputer, số NULL đã impute).
    """

    def __init__(
        self,
        model: BaseAnomalyModel,
        scaler: Optional[str] = "robust",
        imputer: str = "median",
        feature_names: Optional[Sequence[str]] = None,
        random_state: int = 42,
        schema_path: Optional[Path | str] = None,
    ):
        if not isinstance(model, BaseAnomalyModel):
            raise TypeError(f"model phải là BaseAnomalyModel, nhận được {type(model)!r}.")

        self.model = model
        self.random_state = int(random_state)
        self.feature_names: List[str] = (
            list(feature_names) if feature_names else list(FeatureSchema(schema_path).core_features)
        )
        if not self.feature_names:
            raise ValueError("Danh sách đặc trưng rỗng: kiểm tra 'core: true' trong configs/feature_schema.yaml.")

        self.imputer_ = build_imputer(imputer)
        self.scaler_ = build_scaler(scaler, self.random_state) if model.requires_scaling else None
        self.is_fitted_ = False
        self.n_train_rows_ = 0
        self.n_nulls_imputed_ = 0

    # ------------------------------------------------------------------ #
    # Trích đặc trưng & biến đổi
    # ------------------------------------------------------------------ #
    def _extract_features(self, data: Any, context: str) -> np.ndarray:
        """Lấy **đúng** các đặc trưng theo tên; thiếu cột -> lỗi rõ ràng (không tự bỏ qua)."""
        if isinstance(data, pl.DataFrame):
            missing = [c for c in self.feature_names if c not in data.columns]
            if missing:
                raise ValueError(f"{context}: dữ liệu thiếu {len(missing)} đặc trưng core: {missing}.")
            matrix = data.select([pl.col(c).cast(pl.Float64) for c in self.feature_names]).to_numpy()
        elif hasattr(data, "columns"):
            columns = list(data.columns)
            missing = [c for c in self.feature_names if c not in columns]
            if missing:
                raise ValueError(f"{context}: DataFrame thiếu {len(missing)} đặc trưng core: {missing}.")
            matrix = data[self.feature_names].to_numpy(dtype=np.float64)
        else:
            matrix = np.asarray(data, dtype=np.float64)
            if matrix.ndim != 2 or matrix.shape[1] != len(self.feature_names):
                raise TypeError(
                    f"{context}: cần polars/pandas DataFrame hoặc mảng 2 chiều "
                    f"(n, {len(self.feature_names)}) theo đúng thứ tự {self.feature_names}; "
                    f"nhận được shape={getattr(matrix, 'shape', None)}."
                )
        return np.ascontiguousarray(matrix, dtype=np.float64)

    def transform(self, data: Any, context: str = "tập dữ liệu") -> np.ndarray:
        """Impute + scale bằng tham số **đã học từ tập train** (chỉ transform, không fit lại)."""
        if not self.is_fitted_:
            raise RuntimeError("AnomalyPipeline chưa fit. Hãy gọi fit(df_train) trước khi transform/score.")
        X = self._extract_features(data, context)
        X = self.imputer_.transform(X)
        if self.scaler_ is not None:
            X = self.scaler_.transform(X)
        return np.ascontiguousarray(X, dtype=np.float64)

    def fit(self, data: Any, context: str = "tập train") -> "AnomalyPipeline":
        """
        Fit imputer + scaler + mô hình trên **chỉ tập train**.

        Tuyệt đối không truyền tập đánh giá vào đây: bản benchmark cũ fit và chấm điểm trên
        cùng một ma trận nên mọi con số đánh giá đều là rò rỉ dữ liệu.
        """
        X_raw = self._extract_features(data, context)
        self.n_train_rows_ = int(X_raw.shape[0])

        null_mask = np.isnan(X_raw)
        self.n_nulls_imputed_ = int(null_mask.sum())
        if self.n_nulls_imputed_:
            counts = {name: int(c) for name, c in zip(self.feature_names, null_mask.sum(axis=0)) if c}
            logger.info(
                "[%s] impute %s giá trị NULL bằng median fit trên tập train: %s",
                self.model.name,
                f"{self.n_nulls_imputed_:,}",
                counts,
            )

        X_imp = self.imputer_.fit_transform(X_raw)
        if self.scaler_ is not None:
            X_imp = self.scaler_.fit_transform(X_imp)
        X = np.ascontiguousarray(X_imp, dtype=np.float64)

        # RuleThresholdBaseline biên dịch luật theo TÊN đặc trưng -> phải gán trước khi fit.
        self.model.feature_names_in_ = list(self.feature_names)
        self.model.fit(X)
        self.is_fitted_ = True

        logger.info(
            "[%s] pipeline fit xong: %s dòng train x %d đặc trưng (imputer=%s, scaler=%s).",
            self.model.name,
            f"{self.n_train_rows_:,}",
            len(self.feature_names),
            type(self.imputer_).__name__,
            type(self.scaler_).__name__ if self.scaler_ is not None else "none",
        )
        return self

    # ------------------------------------------------------------------ #
    # Chấm điểm (uỷ quyền cho mô hình sau khi biến đổi dữ liệu)
    # ------------------------------------------------------------------ #
    def score(self, data: Any, context: str = "tập đánh giá") -> np.ndarray:
        """Điểm dị biệt (CAO = DỊ BIỆT) cho từng dòng của ``data``."""
        return self.model.score(self.transform(data, context))

    def predict(self, data: Any, context: str = "tập đánh giá") -> np.ndarray:
        """Nhãn 0/1 theo ``threshold_`` đã fit trên tập train."""
        return self.model.predict(self.transform(data, context))

    def score_rank_pct(self, data: Any, reference_scores: Optional[np.ndarray] = None) -> np.ndarray:
        """Percentile hạng [0, 1] của điểm — dùng để so sánh liên mô hình."""
        return self.model.score_rank_pct(self.transform(data), reference_scores=reference_scores)

    def get_metadata(self) -> Dict[str, Any]:
        """Metadata gộp của pipeline + mô hình (nguồn dữ liệu duy nhất cho run_manifest.json)."""
        return {
            "model": self.model.get_metadata(),
            "feature_names": list(self.feature_names),
            "n_features": len(self.feature_names),
            "imputer": type(self.imputer_).__name__,
            "scaler": type(self.scaler_).__name__ if self.scaler_ is not None else None,
            "n_train_rows": self.n_train_rows_,
            "n_nulls_imputed": self.n_nulls_imputed_,
            "is_fitted": self.is_fitted_,
        }

    def save(self, path: Path | str) -> Path:
        """Lưu cả pipeline (imputer + scaler + mô hình đã fit) ra một file .joblib."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"format": PIPELINE_PAYLOAD_FORMAT, "pipeline": self, "metadata": self.get_metadata()},
            target,
        )
        logger.info("Đã lưu pipeline '%s' tại '%s'.", self.model.name, target)
        return target

    @classmethod
    def load(cls, path: Path | str) -> "AnomalyPipeline":
        """Nạp pipeline đã lưu; chấm điểm lại không cần fit lần nữa."""
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"Không tìm thấy file pipeline: '{source}'.")

        payload = joblib.load(source)
        if not isinstance(payload, dict) or "pipeline" not in payload:
            raise ValueError(f"File '{source}' không đúng định dạng {PIPELINE_PAYLOAD_FORMAT}.")
        pipeline = payload["pipeline"]
        if not isinstance(pipeline, AnomalyPipeline):
            raise TypeError(f"Đối tượng trong '{source}' không phải AnomalyPipeline: {type(pipeline)!r}.")
        return pipeline

    def __repr__(self) -> str:
        return (
            f"AnomalyPipeline(model='{self.model.name}', n_features={len(self.feature_names)}, "
            f"imputer={type(self.imputer_).__name__}, "
            f"scaler={type(self.scaler_).__name__ if self.scaler_ is not None else 'none'}, "
            f"fitted={self.is_fitted_})"
        )


