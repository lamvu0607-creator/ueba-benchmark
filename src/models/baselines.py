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
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.models.base import BaseAnomalyModel

logger = logging.getLogger("ueba_benchmark.models.baselines")

__all__ = ["RandomBaseline", "ZScoreBaseline", "RuleThresholdBaseline", "FailureCountBaseline", "RULE_OPERATORS"]

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
      * ``z_threshold`` + ``threshold_mode: "fixed"`` nếu muốn ngưỡng tuyệt đối cổ điển (|z| >= 3),
      * ``scale_fallback`` (khi ``method="robust"``) — độ tán xạ thay thế cho đặc trưng có MAD = 0:

        - ``"mad_floor"`` (mặc định): **z-score toàn cục** đúng như ``zscore_global`` của stage baselines
          (``src/baselines/zscore_baseline.py``): sàn = 1.4826 × phân vị ``mad_floor_quantile`` của các độ
          lệch KHÁC 0 trên tập fit (dung sai ``zero_tolerance``); đặc trưng hằng số đóng góp 0;
        - ``"iqr_std"``: chuỗi cũ ``MAD -> IQR/1.349 -> std -> 1.0`` (giữ để tái lập kết quả trước 2026-10-09).

    MAD = 0 xảy ra khi hơn nửa số dòng bằng đúng median (các ratio vốn bằng 0); đặt scale = 1 thì đơn vị
    thô của vài đặc trưng sẽ lấn át toàn bộ điểm. Số đặc trưng phải fallback luôn được ghi log.

    Không cần scale vì điểm tự chuẩn hoá theo từng đặc trưng (``requires_scaling = False``).
    """

    name = "zscore_baseline"
    aliases = ("ZScoreBaseline", "zscore")
    is_baseline = True
    estimator_class = None
    default_max_train_samples = None

    def _fit_estimator(self, X: np.ndarray) -> None:
        method = str(self.native_params.get("method", "robust"))
        fallback = str(self.native_params.get("scale_fallback", "mad_floor"))
        if method == "robust" and fallback == "mad_floor":
            from src.baselines.zscore_baseline import DEFAULT_ZERO_TOLERANCE, robust_location_scale

            q = float(self.native_params.get("mad_floor_quantile", 0.25))
            tol = float(self.native_params.get("zero_tolerance", DEFAULT_ZERO_TOLERANCE))
            self.location_, mad, scale = robust_location_scale(X, q, tol)
            logger.info(
                "[%s] z-score toàn cục: %d/%d đặc trưng MAD = 0 dùng sàn (q=%.3g); %d hằng số (đóng góp 0).",
                self.name, int(((mad <= 0) & np.isfinite(scale)).sum()), X.shape[1], q,
                int((~np.isfinite(scale)).sum()),
            )
        elif method == "robust" and fallback == "iqr_std":
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
        elif method == "robust":
            raise ValueError(
                f"scale_fallback không hợp lệ cho ZScoreBaseline: {fallback!r}. Chỉ hỗ trợ 'mad_floor' hoặc 'iqr_std'."
            )
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


class RandomBaseline(BaseAnomalyModel):
    """
    Baseline **"đoán mò"** (null hypothesis): điểm dị biệt là **ngẫu nhiên tất định theo nội dung dòng**.

    Vì sao cần baseline này:
      * Là **mốc dưới** cho MỌI chỉ số. Kỳ vọng lý thuyết: ``ROC-AUC ≈ 0,5``,
        ``Average Precision ≈ tỷ lệ dương tính``, ``Precision@K ≈ tỷ lệ dương tính``,
        và ``alert rate ≈ contamination`` (vì ngưỡng lấy theo phân vị trên tập fit).
      * Nếu mô hình "thật" không vượt được mốc này thì kết quả **không có nghĩa** —
        đúng loại kiểm tra mà kết luận Tuần 3 (không thể xếp hạng bằng label-free) cần có.

    Vì sao **băm theo NỘI DUNG dòng** thay vì gọi ``rng(n)`` mỗi lần chấm điểm:
      * cùng một dòng luôn nhận cùng điểm ⇒ tái lập được, chấm theo lô (batch) không đổi kết quả;
      * bất biến với **thứ tự dòng** (không phụ thuộc thứ tự/kích thước batch);
      * vẫn **không có quan hệ thống kê** với dữ liệu ⇒ đúng nghĩa "ngẫu nhiên".
    Cái giá phải trả: điểm là hàm của nội dung dòng nên **không** dùng để minh hoạ "cùng dữ liệu,
    hai lần gọi khác nhau" — muốn vậy phải đổi ``random_state`` (``seed``).

    Tham số (đọc từ ``configs/model_params.yaml`` -> ``random_baseline``):
      * ``random_state``: seed của phép băm (đổi seed ⇒ bộ điểm khác, alert rate vẫn ~ ``contamination``).

    Điểm nằm trong ``[0, 1)`` (phân phối đều) nên có thể đọc trực tiếp như "percentile ngẫu nhiên".
    """

    name = "random_baseline"
    aliases = ("RandomBaseline", "random")
    is_baseline = True
    estimator_class = None
    default_max_train_samples = None

    #: Hằng số trộn của splitmix64/FNV (chỉ để băm, không phải tham số người dùng).
    _MIX_A: ClassVar[int] = 0x9E3779B97F4A7C15
    _MIX_B: ClassVar[int] = 0xBF58476D1CE4E5B9
    _MIX_C: ClassVar[int] = 0x94D049BB133111EB
    _FNV_PRIME: ClassVar[int] = 0x100000001B3

    def _fit_estimator(self, X: np.ndarray) -> None:
        """Không học gì từ dữ liệu (đó chính là ý nghĩa "đoán mò"); chỉ ghi log để không hiểu nhầm."""
        logger.info(
            "[%s] baseline ngẫu nhiên: KHÔNG học từ dữ liệu (băm theo seed=%s, %d đặc trưng). "
            "Kỳ vọng: ROC-AUC ≈ 0,5 · AP ≈ tỷ lệ dương tính · alert rate ≈ %.2f%%.",
            self.name,
            self.random_state,
            int(X.shape[1]),
            self.contamination * 100,
        )
        return None

    def _anomaly_score(self, X: np.ndarray) -> np.ndarray:
        return self._hash_scores(X)

    def _hash_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Băm từng dòng thành số đều trong ``[0, 1)`` bằng splitmix64 trên chữ ký đã lượng tử hoá.

        Lượng tử hoá ``floor(x * 1e6)`` để hai giá trị "gần bằng nhau về mặt ngữ nghĩa" vẫn cho
        cùng điểm (chống nhiễu dấu phẩy động), và ``nan_to_num`` để NaN/Inf không làm hỏng phép băm
        (pipeline đã impute trước khi chấm điểm, đây chỉ là chốt an toàn).
        """
        payload = np.nan_to_num(
            np.asarray(X, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
        )
        quant = np.floor(payload * 1e6).astype(np.int64).astype(np.uint64)

        h = np.full(
            quant.shape[0], np.uint64(self.random_state) + np.uint64(self._MIX_A), dtype=np.uint64
        )
        for col in range(quant.shape[1]):
            h = (h ^ quant[:, col]) * np.uint64(self._FNV_PRIME)
        # Finalizer splitmix64: trộn bit để tránh tương quan giữa các dòng "gần giống nhau".
        h ^= h >> np.uint64(30)
        h *= np.uint64(self._MIX_B)
        h ^= h >> np.uint64(27)
        h *= np.uint64(self._MIX_C)
        h ^= h >> np.uint64(31)
        return (h >> np.uint64(11)).astype(np.float64) / float(1 << 53)

    def get_metadata(self) -> Dict[str, Any]:
        metadata = super().get_metadata()
        metadata["note"] = (
            "Baseline ngẫu nhiên tất định theo nội dung dòng (băm splitmix64 theo seed): "
            "mốc dưới cho mọi chỉ số — ROC-AUC kỳ vọng 0,5, AP/Precision@K kỳ vọng ≈ tỷ lệ dương tính."
        )
        return metadata


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


class FailureCountBaseline(BaseAnomalyModel):
    """
    Mốc "luật ngưỡng số lần thất bại" của đề cương (mục 4.1 #6, 5.3): điểm = **số lần đăng nhập thất bại
    trong ngày** của (tài khoản, ngày); cảnh báo khi điểm >= ``min_failures`` (ngưỡng CỐ ĐỊNH).

    Số lần thất bại không có sẵn trong bộ đặc trưng core nên được dựng lại chính xác (số nguyên) từ
    ``failure_ratio × expm1(log_total_logons)`` (``log_total_logons = log1p(total_logons)``).

    Tham số (``configs/model_params.yaml`` -> ``failure_count_baseline``):
      * ``min_failures``: ngưỡng cảnh báo (mặc định 5 = ngưỡng khoá tài khoản L ước lượng từ train),
      * ``threshold_mode``: ``"fixed"`` (mặc định, đúng đề cương) hoặc ``"quantile"`` (alert rate ~ contamination),
      * ``ratio_feature`` / ``total_log_feature``: tên cột (mặc định ``failure_ratio`` / ``log_total_logons``).

    Không cần scale (``requires_scaling = False``): điểm là số đếm thô.
    """

    name = "failure_count_baseline"
    aliases = ("FailureCountBaseline", "failure_count")
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
        self.columns_: Tuple[int, int] = (-1, -1)

    def _fit_estimator(self, X: np.ndarray) -> None:
        names = self.feature_names or (list(self.feature_names_in_) if self.feature_names_in_ else None)
        if not names:
            raise ValueError(
                f"[{self.name}] cần TÊN đặc trưng: hãy fit qua AnomalyPipeline hoặc truyền feature_names=[...]."
            )
        ratio = str(self.native_params.get("ratio_feature", "failure_ratio"))
        total = str(self.native_params.get("total_log_feature", "log_total_logons"))
        missing = [c for c in (ratio, total) if c not in names]
        if missing:
            raise ValueError(f"[{self.name}] thiếu đặc trưng {missing} trong tập đặc trưng đang dùng.")
        self.columns_ = (names.index(ratio), names.index(total))
        logger.info("[%s] điểm = số lần thất bại/ngày = round(%s × expm1(%s)); ngưỡng cố định >= %s.",
                    self.name, ratio, total, self.native_params.get("min_failures", 5))
        return None

    def _anomaly_score(self, X: np.ndarray) -> np.ndarray:
        r, t = self.columns_
        return np.round(np.clip(X[:, r], 0.0, None) * np.expm1(np.clip(X[:, t], 0.0, None)))

    def _resolve_threshold(self, fit_scores: np.ndarray) -> float:
        if str(self.native_params.get("threshold_mode", "fixed")) == "fixed":
            return float(self.native_params.get("min_failures", 5))
        return super()._resolve_threshold(fit_scores)
