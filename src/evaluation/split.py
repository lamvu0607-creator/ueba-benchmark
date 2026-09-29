"""
Module Split - Chia tập huấn luyện / đánh giá theo **THỜI GIAN** (time-based split).

Vì sao không dùng chia ngẫu nhiên: mục tiêu của UEBA là phát hiện hành vi bất thường trong
tương lai, nên tập đánh giá phải là các ngày **sau** tập huấn luyện. Chia ngẫu nhiên cùng ngày
sẽ khiến mô hình "nhìn thấy" chính hành vi của ngày cần chấm điểm (rò rỉ thời gian).

Bản benchmark cũ fit và chấm điểm trên **cùng một ma trận** (1.055.283 dòng) — mọi con số đánh giá
đều là rò rỉ dữ liệu. Module này thay thế bằng đúng một quy tắc::

    train = {day <= split_day}      test = {day > split_day}

Mặc định ``split_day = 42`` (train ngày 1-42: 721.612 dòng; test ngày 43-60: 333.671 dòng).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import polars as pl

logger = logging.getLogger("ueba_benchmark.evaluation.split")

__all__ = ["SplitInfo", "resolve_split_day", "time_split"]


@dataclass(frozen=True)
class SplitInfo:
    """Mô tả một lần chia tập theo thời gian (đủ chi tiết để tái lập và ghi vào manifest)."""

    day_col: str
    split_day: int
    min_day: int
    max_day: int
    n_train: int
    n_eval: int
    n_train_days: int
    n_eval_days: int
    strategy: str = "time"

    @property
    def n_total(self) -> int:
        return self.n_train + self.n_eval

    @property
    def eval_ratio(self) -> float:
        """Tỷ lệ dòng nằm ở tập đánh giá (giá trị THỰC sau khi chia, không phải tham số đặt trước)."""
        return self.n_eval / self.n_total if self.n_total else 0.0

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["n_total"] = self.n_total
        payload["eval_ratio"] = self.eval_ratio
        return payload


def resolve_split_day(df: pl.DataFrame, day_col: str = "day", test_split_ratio: float = 0.30) -> int:
    """
    Chọn ``split_day`` sao cho tỷ lệ dòng đánh giá gần ``test_split_ratio`` nhất.

    Chỉ được cắt tại **ranh giới ngày** (không cắt giữa ngày) để tập đánh giá là một khoảng
    thời gian liền mạch — đúng cách vận hành thật. Nếu nhiều ranh giới có cùng độ lệch thì chọn
    ranh giới sớm hơn (nhiều dữ liệu huấn luyện hơn).
    """
    if not 0.0 < float(test_split_ratio) < 1.0:
        raise ValueError(f"test_split_ratio phải nằm trong (0, 1), nhận được {test_split_ratio}.")
    if day_col not in df.columns:
        raise ValueError(f"Không tìm thấy cột ngày '{day_col}' trong dữ liệu: {df.columns}.")

    counts = {
        int(row[day_col]): int(row["len"])
        for row in df.group_by(day_col).len().sort(day_col).iter_rows(named=True)
    }
    if len(counts) < 2:
        raise ValueError("Cần ít nhất 2 ngày dữ liệu để chia tập theo thời gian.")

    total = sum(counts.values())
    target = float(test_split_ratio) * total

    best_day, best_gap = None, None
    for split_day in sorted(counts)[:-1]:  # luôn để lại ít nhất 1 ngày cho tập đánh giá
        n_eval = sum(v for k, v in counts.items() if k > split_day)
        gap = abs(n_eval - target)
        if best_gap is None or gap < best_gap:
            best_day, best_gap = split_day, gap
    return int(best_day)


def time_split(
    df: pl.DataFrame,
    day_col: str = "day",
    split_day: Optional[int] = None,
    test_split_ratio: float = 0.30,
) -> Tuple[pl.DataFrame, pl.DataFrame, SplitInfo]:
    """
    Chia ``df`` thành ``(train, test, info)`` theo ngày.

    ``split_day`` được ưu tiên nếu truyền vào (dùng cho tái lập thí nghiệm cũ); nếu không thì suy
    ra từ ``test_split_ratio`` qua ``resolve_split_day``.
    """
    if day_col not in df.columns:
        raise ValueError(f"Không tìm thấy cột ngày '{day_col}' trong dữ liệu: {df.columns}.")
    if df.height == 0:
        raise ValueError("Dữ liệu rỗng, không thể chia tập.")

    if split_day is None:
        split_day = resolve_split_day(df, day_col=day_col, test_split_ratio=test_split_ratio)

    train = df.filter(pl.col(day_col) <= split_day)
    test = df.filter(pl.col(day_col) > split_day)
    if train.height == 0 or test.height == 0:
        min_day, max_day = int(df[day_col].min()), int(df[day_col].max())
        raise ValueError(
            f"split_day={split_day} cho tập rỗng (train={train.height:,}, test={test.height:,}); "
            f"dữ liệu có day {min_day}..{max_day}."
        )

    info = SplitInfo(
        day_col=day_col,
        split_day=int(split_day),
        min_day=int(df[day_col].min()),
        max_day=int(df[day_col].max()),
        n_train=train.height,
        n_eval=test.height,
        n_train_days=train[day_col].n_unique(),
        n_eval_days=test[day_col].n_unique(),
    )
    logger.info(
        "Chia theo thời gian: train=%s dòng (day <= %s), test=%s dòng (day > %s), eval_ratio=%.2f%%.",
        f"{info.n_train:,}",
        info.split_day,
        f"{info.n_eval:,}",
        info.split_day,
        info.eval_ratio * 100,
    )
    return train, test, info
