"""
Module Baselines - Hai mốc so sánh label-free (không cần nhãn) cho benchmark UEBA.

1. ``ZScoreBaseline``  : bắt dòng lệch xa phân phối chuẩn của **từng đặc trưng** (thống kê đơn biến).
2. ``RuleThresholdBaseline`` : bắt dòng thoả các **luật nghiệp vụ tường minh** (tri thức chuyên gia).

Cả hai kế thừa ``BaseAnomalyModel`` nên dùng chung API ``fit`` / ``score`` / ``predict`` và
tuân thủ cùng 2 bất biến: điểm CAO = DỊ BIỆT, và ngưỡng cảnh báo lấy theo phân vị trên tập fit
(``threshold_mode: "quantile"``) để "alert rate" so sánh được với LOF/OCSVM/Isolation Forest.
Có thể đặt ``threshold_mode: "fixed"`` nếu muốn dùng ngưỡng tuyệt đối (z_threshold / min_rules).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.models.base import BaseAnomalyModel

logger = logging.getLogger("ueba_benchmark.models.baselines")

__all__ = ["ZScoreBaseline", "RuleThresholdBaseline", "RULE_OPERATORS"]

#: Các toán tử so sánh được phép trong luật.
RULE_OPERATORS = (">=", ">", "<=", "<", "==", "!=")


def _apply_rule(values: np.ndarray, op: str, threshold: float) -> np.ndarray:
    """Trả về mảng 0/1 cho một luật ``feature op value``; toán tử lạ -> lỗi rõ ràng."""
    if op == ">=":
        return (values >= threshold).astype(np.float64)
    if op == ">":
        return (values > threshold).astype(np.float64)
    if op == "<=":
        return (values <= threshold).astype(np.float64)
    if op == "<":
        return (values < threshold).astype(np.float64)
    if op == "==":
        return np.isclose(values, threshold).astype(np.float64)
    if op == "!=":
        return (~np.isclose(values, threshold)).astype(np.float64)
    raise ValueError(f"Toán tử luật không hợp lệ: {op!r}. Chỉ hỗ trợ {list(RULE_OPERATORS)}.")


class ZScoreBaseline(BaseAnomalyModel):
    """
    Baseline thống kê đơn biến: điểm = ``agg_j |(x_j - loc_j) / scale_j|``.

    Tham số (đọc từ ``configs/model_params.yaml`` -> ``zscore_baseline``):
      * ``method``: ``"robust"`` (median/MAD*1.4826 — mặc định, chống outlier) hoặc ``"classic"`` (mean/std),
      * ``agg``: ``"max"`` (mặc định, "có bất kỳ đặc trưng nào lệch mạnh") | ``"mean"`` | ``"p95"``,
      * ``z_threshold`` + ``threshold_mode: "fixed"`` nếu muốn ngưỡng tuyệt đối cổ điển (|z| >= 3).

    Độ tán xạ khi ``method="robust"`` dùng chuỗi fallback ``MAD -> IQR/1.349 -> std -> 1.0``: trên dữ liệu
    thật 7/16 đặc trưng có MAD = 0 (hơn nửa số dòng bằng đúng median, ví dụ các ratio vốn bằng 0),
    nên nếu cứ đặt scale = 1 thì đơn vị thô của một vài đặc trưng sẽ lấn át toàn bộ điểm.
    Số đặc trưng phải fallback luôn được ghi log để người đọc biết kết quả dựa trên gì.

    Không cần scale vì điểm tự chuẩn hoá theo từng đặc trưng (``requires_scaling = False``).
    """

    name = "zscore_baseline"
    aliases = ("ZScoreBaseline", "zscore")
    is_baseline = True
    estimator_class = None
    default_max_train_samples = None

    def _fit_estimator(self, X: np.ndarray) -> None:
        method = str(self.native_params.get("method", "robust"))
        if method == "robust":
            self.location_ = np.median(X, axis=0)
            # Chuỗi fallback cho độ tán xạ (đo trên dữ liệu thật: 7/16 đặc trưng có MAD = 0 vì
            # hơn nửa số dòng bằng đúng median — nếu đặt scale = 1 thì đơn vị thô (ví dụ
            # interarrival_dt_mean tới 86.297 giây) sẽ lấn át toàn bộ điểm z-score).
            mad = 1.4826 * np.median(np.abs(X - self.location_), axis=0)
            iqr = (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)) / 1.349
            std = X.std(axis=0)
            scale = np.where(mad > 0, mad, np.where(iqr > 0, iqr, std))
            n_mad_zero, n_degenerate = int((mad <= 0).sum()), int((scale <= 0).sum())
            if n_mad_zero:
                logger.info(
                    "[%s] có %d/%d đặc trưng MAD = 0 -> dùng IQR (rồi tới std) làm độ tán xạ thay thế.",
                    self.name,
                    n_mad_zero,
                    X.shape[1],
                )
            if n_degenerate:
                logger.warning(
                    "[%s] có %d đặc trưng hằng số hoàn toàn trên tập fit -> đặt scale = 1 (không đóng góp điểm).",
                    self.name,
                    n_degenerate,
                )
            scale = np.where(scale > 0, scale, 1.0)
        elif method == "classic":
            self.location_ = X.mean(axis=0)
            std = X.std(axis=0)
            n_degenerate = int((std <= 0).sum())
            if n_degenerate:
                logger.warning(
                    "[%s] có %d đặc trưng hằng số trên tập fit -> đặt scale = 1 để tránh chia 0.",
                    self.name,
                    n_degenerate,
                )
            scale = np.where(std > 0, std, 1.0)
        else:
            raise ValueError(
                f"method không hợp lệ cho ZScoreBaseline: {method!r}. Chỉ hỗ trợ 'robust' hoặc 'classic'."
            )

        self.scale_ = scale
        return None

    def _anomaly_score(self, X: np.ndarray) -> np.ndarray:
        z = np.abs((X - self.location_) / self.scale_)
        agg = str(self.native_params.get("agg", "max"))
        if agg == "max":
            return z.max(axis=1)
        if agg == "mean":
            return z.mean(axis=1)
        if agg == "p95":
            return np.percentile(z, 95.0, axis=1)
        raise ValueError(f"agg không hợp lệ cho ZScoreBaseline: {agg!r}. Chỉ hỗ trợ 'max', 'mean', 'p95'.")

    def _resolve_threshold(self, fit_scores: np.ndarray) -> float:
        if str(self.native_params.get("threshold_mode", "quantile")) == "fixed":
            return float(self.native_params.get("z_threshold", 3.0))
        return super()._resolve_threshold(fit_scores)


class RuleThresholdBaseline(BaseAnomalyModel):
    """
    Baseline luật nghiệp vụ: điểm = **tỷ lệ luật được thoả** trên mỗi dòng (0..1).

    Tham số (đọc từ ``configs/model_params.yaml`` -> ``rule_threshold_baseline``):
      * ``rules``: danh sách ``{feature, op, value, rationale}``; ``op`` ∈ ``>= > <= < == !=``;
      * ``threshold_mode: "fixed"`` + ``min_rules`` -> cảnh báo khi thoả >= ``min_rules`` luật;
      * ``threshold_mode: "quantile"`` (mặc định) -> vẫn dùng phân vị để alert rate ~ contamination,
        nhờ đó baseline luật so sánh được công bằng với các mô hình học máy.

    Luật làm việc theo **TÊN đặc trưng**: ``AnomalyPipeline`` tự gán ``feature_names_in_`` trước khi
    fit; nếu dùng trực tiếp thì phải truyền ``feature_names=[...]`` khi khởi tạo.

    Lưu ý về đồng điểm: điểm là tỷ lệ luật rời rạc nên có **rất nhiều dòng cùng điểm** (đặc biệt 0.0).
    Trong nhóm đồng điểm, thứ tự Top-K là tuỳ ý — tài liệu của benchmark ghi rõ điều này thay vì
    che bằng một tie-break ngầm khó diễn giải.
    """

    name = "rule_threshold_baseline"
    aliases = ("RuleThresholdBaseline",)
    is_baseline = True
    estimator_class = None
    default_max_train_samples = None

    def __init__(
        self,
        contamination: float = 0.05,
        random_state: int = 42,
        max_train_samples: Optional[int] = None,
        feature_names: Optional[Sequence[str]] = None,
        **native_params: Any,
    ):
        super().__init__(
            contamination=contamination,
            random_state=random_state,
            max_train_samples=max_train_samples,
            **native_params,
        )
        self.feature_names = list(feature_names) if feature_names else None
        self.compiled_rules_: List[Tuple[int, str, float, str]] = []

    def _resolved_feature_names(self) -> List[str]:
        names = self.feature_names or (list(self.feature_names_in_) if self.feature_names_in_ else None)
        if not names:
            raise ValueError(
                f"[{self.name}] cần TÊN đặc trưng: hãy fit qua AnomalyPipeline (tự gán feature_names_in_) "
                "hoặc truyền feature_names=[...] khi khởi tạo."
            )
        return names

    def _compiled_rules(self) -> List[Tuple[int, str, float, str]]:
        """Chuyển luật (tên đặc trưng) thành (chỉ số cột, toán tử, ngưỡng, lý do) và kiểm tra hợp lệ."""
        rules = self.native_params.get("rules") or []
        if not rules:
            raise ValueError(f"[{self.name}] thiếu danh sách 'rules' trong configs/model_params.yaml.")

        names = self._resolved_feature_names()
        compiled: List[Tuple[int, str, float, str]] = []
        for rule in rules:
            feature = rule.get("feature")
            op = rule.get("op")
            if feature not in names:
                raise ValueError(
                    f"[{self.name}] luật dùng đặc trưng '{feature}' không thuộc tập đặc trưng đang dùng: {names}."
                )
            if op not in RULE_OPERATORS:
                raise ValueError(
                    f"[{self.name}] toán tử '{op}' không hợp lệ cho luật trên '{feature}'. "
                    f"Chỉ hỗ trợ {list(RULE_OPERATORS)}."
                )
            compiled.append((names.index(feature), str(op), float(rule.get("value")), str(rule.get("rationale", ""))))
        return compiled

    def _fit_estimator(self, X: np.ndarray) -> None:
        # Luật là tri thức cố định -> chỉ cần biên dịch sang chỉ số cột một lần.
        self.compiled_rules_ = self._compiled_rules()
        logger.info(
            "[%s] đã nạp %d luật: %s",
            self.name,
            len(self.compiled_rules_),
            [f"{self._resolved_feature_names()[idx]} {op} {value:g}" for idx, op, value, _ in self.compiled_rules_],
        )
        return None

    def _anomaly_score(self, X: np.ndarray) -> np.ndarray:
        hits = np.zeros(X.shape[0], dtype=np.float64)
        for col_idx, op, value, _ in self.compiled_rules_:
            hits += _apply_rule(X[:, col_idx], op, value)
        return hits / float(len(self.compiled_rules_))

    def _resolve_threshold(self, fit_scores: np.ndarray) -> float:
        if str(self.native_params.get("threshold_mode", "quantile")) == "fixed":
            min_rules = float(self.native_params.get("min_rules", 1))
            return min_rules / float(len(self.compiled_rules_) or 1)
        return super()._resolve_threshold(fit_scores)

    def get_metadata(self) -> Dict[str, Any]:
        metadata = super().get_metadata()
        names = self.feature_names or (list(self.feature_names_in_) if self.feature_names_in_ else [])
        metadata["n_rules"] = len(self.compiled_rules_)
        metadata["rules"] = [
            {
                "feature": names[idx] if idx < len(names) else f"col_{idx}",
                "op": op,
                "value": value,
                "rationale": rationale,
            }
            for idx, op, value, rationale in self.compiled_rules_
        ]
        return metadata

