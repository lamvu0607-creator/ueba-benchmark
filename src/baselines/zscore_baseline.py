"""
BASELINE 2 — Z-SCORE ĐƠN BIẾN ROBUST trên 39 đặc trưng core.

    z_j(x)  = (x_j − median_j) / (1.4826 · MAD_j)
    score(x) = max_j |z_j(x)|

Căn cứ từng lựa chọn:
  * **median / MAD thay vì mean / std**: train chứa sẵn các dòng bất thường thật (không có nhãn để
    loại). Mean/std bị chính các outlier đó kéo lệch (breakdown point 0%), median/MAD chịu được tới
    50% dữ liệu bẩn. Đặc trưng UEBA còn lệch phải rất mạnh (skewness tới >100), mean/std càng kém.
  * **1.4826**: = 1/Φ⁻¹(0.75), hằng số làm MAD là ước lượng nhất quán của σ khi dữ liệu chuẩn —
    nhờ đó |z| đọc được theo "số độ lệch chuẩn". Đây là hằng số toán học, KHÔNG phải tham số chọn.
  * **max thay vì tổng/trung bình**: logic "một chỉ số vượt là đủ đáng ngờ" của SOC — tấn công thường
    chỉ làm lệch 1–2 chiều (vd. failure_count) trong khi các chiều khác bình thường; lấy trung bình sẽ
    pha loãng tín hiệu đó bởi 37 chiều còn lại. max cũng không cần giả định độc lập giữa các đặc trưng.
  * **Sàn MAD**: MAD_j = 0 khi > 50% dòng bằng đúng median (ratio vốn bằng 0, ...). Chia cho 0 không
    xác định, đặt scale = 1 thì đơn vị thô (giây) lấn át. Sàn là phân vị nhỏ của các độ lệch KHÁC 0
    quan sát được trên train (xem configs/baselines.yaml → zscore.mad_floor) — một quy tắc viết sẵn,
    không phải con số gán tay. "Khác 0" hiểu theo dung sai số học ``zero_tolerance`` (1e-9): các đặc
    trưng delta_mean_* có độ lệch 1e-16 do làm tròn dấu phẩy động — không phải biến thiên thật.

Hai biến thể xuất hai điểm riêng:
  (a) ``GlobalRobustZScore``  — median/MAD trên toàn train (của phân khúc).
  (b) ``AccountRobustZScore`` — median/MAD trên lịch sử train của chính tài khoản; tài khoản có
      < ``min_train_days`` dòng train (hoặc chưa từng xuất hiện ở train) dùng thống kê toàn cục.

Cả hai chỉ đọc dữ liệu train không nhãn trong ``fit``; ``score`` chấm từng dòng độc lập bằng tham số
đã học (điểm của một dòng không phụ thuộc các dòng đánh giá khác).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import polars as pl

from src.baselines._common import ACCOUNT_KEYS, TrainMedianImputer, column_matrix, logger

__all__ = ["MAD_CONSISTENCY", "DEFAULT_ZERO_TOLERANCE", "GlobalRobustZScore", "AccountRobustZScore"]

#: 1 / Φ⁻¹(0.75): MAD × hằng số này ước lượng nhất quán σ của phân phối chuẩn.
MAD_CONSISTENCY = 1.4826


#: Dung sai số học mặc định: |độ lệch| <= giá trị này được coi là 0 (configs/baselines.yaml → zscore.zero_tolerance).
DEFAULT_ZERO_TOLERANCE = 1e-9


def _check_quantile(q: float) -> float:
    if not 0.0 <= float(q) <= 1.0:
        raise ValueError(f"mad_floor.quantile phải nằm trong [0, 1], nhận được {q}.")
    return float(q)


def _max_abs_z(X: np.ndarray, location: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """max_j |(x_j − loc_j) / scale_j|; scale = +inf (đặc trưng hằng số) cho đóng góp 0."""
    z = np.abs((X - location) / scale)
    return z.max(axis=1) if z.shape[1] else np.zeros(z.shape[0])


class GlobalRobustZScore:
    """(a) median/MAD toàn cục trên train; MAD_j = 0 -> sàn từ các độ lệch khác 0 của chính đặc trưng j."""

    name = "zscore_global"

    def __init__(
        self, features: Sequence[str], mad_floor_quantile: float = 0.01, zero_tolerance: float = DEFAULT_ZERO_TOLERANCE
    ) -> None:
        if not features:
            raise ValueError("Danh sách đặc trưng rỗng.")
        self.features: List[str] = list(features)
        self.mad_floor_quantile = _check_quantile(mad_floor_quantile)
        self.zero_tolerance = float(zero_tolerance)
        self.imputer_ = TrainMedianImputer()

    def fit(self, train: pl.DataFrame) -> "GlobalRobustZScore":
        X = column_matrix(train, self.features, f"[{self.name}] train")
        X = self.imputer_.fit(X, self.features).transform(X)
        self.location_ = np.median(X, axis=0)
        dev = np.abs(X - self.location_)
        dev[dev <= self.zero_tolerance] = 0.0  # nhiễu dấu phẩy động (vd. hiệu hai trung bình = 1e-16) là 0
        mad = MAD_CONSISTENCY * np.median(dev, axis=0)

        floor = np.full(len(self.features), np.inf)
        for j in range(len(self.features)):
            nonzero = dev[:, j][dev[:, j] > 0]
            if nonzero.size:
                floor[j] = MAD_CONSISTENCY * float(np.quantile(nonzero, self.mad_floor_quantile))
        self.mad_ = mad
        self.floor_ = floor
        self.scale_ = np.where(mad > 0, mad, floor)
        self.floored_features_ = [f for f, m, s in zip(self.features, mad, self.scale_) if m <= 0 and np.isfinite(s)]
        self.constant_features_ = [f for f, s in zip(self.features, self.scale_) if not np.isfinite(s)]
        self.n_train_ = int(X.shape[0])
        logger.info(
            "[%s] fit trên %s dòng train: %d/%d đặc trưng MAD = 0 dùng sàn (q=%.3g) %s; hằng số (loại): %s",
            self.name, f"{self.n_train_:,}", len(self.floored_features_), len(self.features),
            self.mad_floor_quantile, self.floored_features_, self.constant_features_,
        )
        return self

    def transform(self, df: pl.DataFrame) -> np.ndarray:
        return self.imputer_.transform(column_matrix(df, self.features, f"[{self.name}] score"))

    def score(self, df: pl.DataFrame) -> np.ndarray:
        if not hasattr(self, "scale_"):
            raise RuntimeError(f"[{self.name}] chưa fit.")
        return _max_abs_z(self.transform(df), self.location_, self.scale_)

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "n_train": self.n_train_,
            "mad_floor_quantile": self.mad_floor_quantile,
            "zero_tolerance": self.zero_tolerance,
            "floored_features": self.floored_features_,
            "constant_features": self.constant_features_,
            "location": dict(zip(self.features, map(float, self.location_))),
            "scale": dict(zip(self.features, map(float, self.scale_))),
        }


class AccountRobustZScore:
    """
    (b) median/MAD theo tài khoản trên lịch sử TRAIN của chính tài khoản đó.

    * Tài khoản có >= ``min_train_days`` dòng train: median/MAD riêng; MAD_{a,j} = 0 -> sàn_j =
      phân vị ``mad_floor_quantile`` của các MAD_{·,j} khác 0 của mọi tài khoản đủ ngày (không có
      MAD khác 0 nào -> dùng scale toàn cục của j).
    * Tài khoản ít ngày / không có ở train: cả dòng dùng median/scale toàn cục (``GlobalRobustZScore``).
    NULL được impute bằng median TOÀN CỤC của train trước khi tính (cùng ma trận với ML).
    """

    name = "zscore_account"

    def __init__(
        self,
        features: Sequence[str],
        mad_floor_quantile: float = 0.01,
        min_train_days: int = 7,
        account_keys: Sequence[str] = ACCOUNT_KEYS,
        zero_tolerance: float = DEFAULT_ZERO_TOLERANCE,
    ) -> None:
        if int(min_train_days) < 1:
            raise ValueError(f"min_train_days phải >= 1, nhận được {min_train_days}.")
        self.features: List[str] = list(features)
        self.mad_floor_quantile = _check_quantile(mad_floor_quantile)
        self.min_train_days = int(min_train_days)
        self.account_keys: List[str] = list(account_keys)
        self.zero_tolerance = float(zero_tolerance)
        self.global_ = GlobalRobustZScore(self.features, self.mad_floor_quantile, self.zero_tolerance)

    def _imputed_frame(self, df: pl.DataFrame) -> pl.DataFrame:
        X = self.global_.transform(df)
        return df.select(self.account_keys).with_columns(
            [pl.Series(f"__x{j}", X[:, j]) for j in range(len(self.features))]
        )

    def fit(self, train: pl.DataFrame) -> "AccountRobustZScore":
        self.global_.fit(train)
        n = len(self.features)
        xs = [f"__x{j}" for j in range(n)]
        frame = self._imputed_frame(train)

        days = frame.group_by(self.account_keys).len()
        eligible = days.filter(pl.col("len") >= self.min_train_days).select(self.account_keys)
        frame = frame.join(eligible, on=self.account_keys, how="semi")

        med = frame.group_by(self.account_keys).agg([pl.col(c).median().alias(f"__m{j}") for j, c in enumerate(xs)])
        mad = (
            frame.join(med, on=self.account_keys, how="left")
            .select(self.account_keys + [(pl.col(c) - pl.col(f"__m{j}")).abs().alias(f"__d{j}") for j, c in enumerate(xs)])
            .group_by(self.account_keys)
            .agg([(pl.col(f"__d{j}").median() * MAD_CONSISTENCY).alias(f"__mad{j}") for j in range(n)])
            .with_columns([
                pl.when(pl.col(f"__mad{j}") > self.zero_tolerance * MAD_CONSISTENCY).then(pl.col(f"__mad{j}")).otherwise(0.0)
                .alias(f"__mad{j}") for j in range(n)
            ])
        )
        stats = med.join(mad, on=self.account_keys, how="left")

        floor = np.array(self.global_.scale_, dtype=np.float64)
        for j in range(n):
            values = stats[f"__mad{j}"].to_numpy()
            nonzero = values[values > 0]
            if nonzero.size:
                floor[j] = float(np.quantile(nonzero, self.mad_floor_quantile))
        self.floor_ = floor
        self.stats_ = stats.select(
            self.account_keys
            + [pl.col(f"__m{j}") for j in range(n)]
            + [
                pl.when(pl.col(f"__mad{j}") > 0).then(pl.col(f"__mad{j}")).otherwise(pl.lit(floor[j])).alias(f"__s{j}")
                for j in range(n)
            ]
        )
        self.n_accounts_total_ = int(days.height)
        self.n_accounts_own_stats_ = int(self.stats_.height)
        logger.info(
            "[%s] %d/%d tài khoản có >= %d ngày train -> median/MAD riêng; còn lại dùng thống kê toàn cục.",
            self.name, self.n_accounts_own_stats_, self.n_accounts_total_, self.min_train_days,
        )
        return self

    def score(self, df: pl.DataFrame) -> np.ndarray:
        if not hasattr(self, "stats_"):
            raise RuntimeError(f"[{self.name}] chưa fit.")
        n = len(self.features)
        joined = self._imputed_frame(df).join(self.stats_, on=self.account_keys, how="left", maintain_order="left")
        X = joined.select([f"__x{j}" for j in range(n)]).to_numpy()
        loc = joined.select([f"__m{j}" for j in range(n)]).to_numpy().astype(np.float64)
        scale = joined.select([f"__s{j}" for j in range(n)]).to_numpy().astype(np.float64)
        fallback = np.isnan(loc[:, 0]) if n else np.zeros(len(X), dtype=bool)
        loc[fallback] = self.global_.location_
        scale[fallback] = self.global_.scale_
        self.last_fallback_rate_ = float(fallback.mean()) if fallback.size else 0.0
        return _max_abs_z(X, loc, scale)

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "min_train_days": self.min_train_days,
            "mad_floor_quantile": self.mad_floor_quantile,
            "n_accounts_train": self.n_accounts_total_,
            "n_accounts_own_stats": self.n_accounts_own_stats_,
            "account_floor": dict(zip(self.features, map(float, self.floor_))),
        }
