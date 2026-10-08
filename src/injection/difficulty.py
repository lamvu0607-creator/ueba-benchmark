"""
Điểm nối (hook) ORACLE chấm độ khó — bước TÙY CHỌN chạy SAU khi đã tính lại ma trận đặc trưng
trên log của run (``python main.py --stage features --events-dir <run>/events_injected``).

**Trạng thái hiện tại: CHƯA cài oracle nào.** Chỉ có interface và :class:`NullOracle` (trả độ khó
NULL cho mọi mẫu) để đường ống chạy được đầu-cuối. Muốn thêm oracle thật:

  1. viết một lớp có ``name`` và ``score(features, labels) -> DataFrame`` đúng hợp đồng bên dưới;
  2. đăng ký vào ``ORACLES`` (hoặc truyền thẳng đối tượng vào :func:`score_difficulty`).

Hợp đồng của ``score``:
  * vào: ``features`` = ma trận processed của run (khoá ``DomainName, UserName, day`` + đặc trưng),
    ``labels`` = ``labels.parquet`` của run (chỉ dòng dương tính);
  * ra: một dòng cho MỖI dòng của ``labels`` — cột ``DomainName, UserName, day, difficulty``
    (Float64, NULL = chưa chấm). Quy ước: càng lớn càng khó phát hiện; thang đo do oracle tự ghi
    trong ``describe()``.
  * Oracle chỉ dùng để MÔ TẢ mẫu tiêm (phân tầng kết quả theo độ khó), không được dùng để chọn
    ngưỡng hay fit bất kỳ detector/baseline nào.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Union, runtime_checkable

import polars as pl

__all__ = [
    "DIFFICULTY_KEYS",
    "DifficultyOracle",
    "NullOracle",
    "ORACLES",
    "get_oracle",
    "validate_difficulty",
    "score_difficulty",
]

DIFFICULTY_KEYS = ["DomainName", "UserName", "day"]


@runtime_checkable
class DifficultyOracle(Protocol):
    """Interface oracle: nhận ma trận đặc trưng + nhãn, trả độ khó cho từng mẫu dương tính."""

    name: str

    def score(self, features: pl.DataFrame, labels: pl.DataFrame) -> pl.DataFrame:
        ...

    def describe(self) -> Dict[str, Any]:
        ...


class NullOracle:
    """PLACEHOLDER: không chấm gì, ``difficulty`` = NULL cho mọi mẫu. Thay bằng oracle thật sau."""

    name = "none"

    def score(self, features: pl.DataFrame, labels: pl.DataFrame) -> pl.DataFrame:
        return labels.select(DIFFICULTY_KEYS).with_columns(pl.lit(None, pl.Float64).alias("difficulty"))

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "placeholder": True, "scale": None}


ORACLES: Dict[str, type] = {"none": NullOracle}


def get_oracle(name: Optional[str]) -> DifficultyOracle:
    """Lấy oracle theo tên trong ``ORACLES`` (``None`` -> ``"none"``)."""
    key = name or "none"
    if key not in ORACLES:
        raise KeyError(f"Oracle '{key}' chưa được đăng ký. Có: {sorted(ORACLES)}.")
    return ORACLES[key]()


def validate_difficulty(out: pl.DataFrame, labels: pl.DataFrame) -> pl.DataFrame:
    """Kiểm tra đầu ra oracle: đủ cột, đúng một dòng cho mỗi khoá của ``labels``, không thừa khoá."""
    missing = [c for c in DIFFICULTY_KEYS + ["difficulty"] if c not in out.columns]
    if missing:
        raise ValueError(f"Đầu ra oracle thiếu cột {missing}.")
    out = out.select(
        pl.col("DomainName").cast(pl.String),
        pl.col("UserName").cast(pl.String),
        pl.col("day").cast(pl.Int64),
        pl.col("difficulty").cast(pl.Float64),
    )
    keys = labels.select(
        pl.col("DomainName").cast(pl.String), pl.col("UserName").cast(pl.String), pl.col("day").cast(pl.Int64)
    )
    if out.height != out.select(DIFFICULTY_KEYS).unique().height:
        raise ValueError("Đầu ra oracle có khoá lặp.")
    extra = out.join(keys, on=DIFFICULTY_KEYS, how="anti").height
    absent = keys.join(out, on=DIFFICULTY_KEYS, how="anti").height
    if extra or absent:
        raise ValueError(f"Đầu ra oracle lệch nhãn: {extra} khoá thừa, {absent} khoá thiếu.")
    return out


def score_difficulty(
    layout,
    oracle: Union[str, DifficultyOracle, None] = None,
    features_path: Optional[Path | str] = None,
) -> Path:
    """
    Chạy oracle trên một run (``layout`` là ``RunLayout``) -> ghi ``layout.difficulty_path``.

    ``features_path`` mặc định = ``<run>/processed/feature_matrix_processed.parquet``.
    """
    impl = get_oracle(oracle) if (oracle is None or isinstance(oracle, str)) else oracle
    feats_path = Path(features_path) if features_path else Path(layout.processed_dir) / "feature_matrix_processed.parquet"
    if not feats_path.is_file():
        raise FileNotFoundError(f"Chưa có ma trận của run '{feats_path}' — chạy stage features với --events-dir trước.")
    labels_path = Path(layout.labels_path)
    if not labels_path.is_file():
        raise FileNotFoundError(f"Chưa có nhãn '{labels_path}'.")
    labels = pl.read_parquet(labels_path)
    out = validate_difficulty(impl.score(pl.read_parquet(feats_path), labels), labels)
    dest = Path(layout.difficulty_path)
    out.write_parquet(dest)
    dest.with_suffix(".json").write_text(json.dumps(impl.describe(), ensure_ascii=False, indent=2), encoding="utf-8")
    return dest
