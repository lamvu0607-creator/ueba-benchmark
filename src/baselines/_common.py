"""Phần dùng chung của các baseline: lấy ma trận theo tên cột và impute NULL bằng median TRAIN."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np
import polars as pl

logger = logging.getLogger("ueba_benchmark.baselines")

#: Khoá một dòng của ma trận đặc trưng và khoá một tài khoản.
ROW_KEYS: List[str] = ["DomainName", "UserName", "day"]
ACCOUNT_KEYS: List[str] = ["DomainName", "UserName"]


def column_matrix(df: pl.DataFrame, columns: Sequence[str], context: str) -> np.ndarray:
    """Ma trận float64 đúng thứ tự ``columns``; thiếu cột -> lỗi rõ ràng (không tự bỏ qua)."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{context}: thiếu {len(missing)} cột: {missing}.")
    return df.select([pl.col(c).cast(pl.Float64) for c in columns]).to_numpy()


class TrainMedianImputer:
    """
    NULL -> median của cột trên TRAIN (giống ``SimpleImputer(strategy="median")`` trong pipeline ML).

    Lý do: baseline phải nhìn đúng ma trận mà ML nhìn, nếu không chênh lệch kết quả có thể đến từ
    cách xử lý NULL chứ không phải từ phương pháp. Cột toàn NULL trên train -> median = 0 (ghi log).
    """

    def __init__(self) -> None:
        self.medians_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, columns: Sequence[str]) -> "TrainMedianImputer":
        with np.errstate(all="ignore"):
            medians = np.nanmedian(X, axis=0) if X.shape[0] else np.full(X.shape[1], np.nan)
        empty = [c for c, m in zip(columns, medians) if np.isnan(m)]
        if empty:
            logger.warning("Cột toàn NULL trên train -> impute 0: %s", empty)
        self.medians_ = np.where(np.isnan(medians), 0.0, medians)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.medians_ is None:
            raise RuntimeError("TrainMedianImputer chưa fit.")
        return np.where(np.isnan(X), self.medians_, X)

    def as_dict(self, columns: Sequence[str]) -> Dict[str, float]:
        return {c: float(m) for c, m in zip(columns, self.medians_ if self.medians_ is not None else [])}
