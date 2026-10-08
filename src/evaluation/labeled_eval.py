"""
Đánh giá có nhãn DÙNG CHUNG cho baseline và mô hình ML: nhận điểm đánh giá + ngưỡng train, không
quan tâm điểm đến từ phương pháp nào.

File nhãn (``labels.parquet`` do module nhãn cung cấp)::

    DomainName, UserName, day, is_anomaly (0/1), eval_exclude (bool), scenario, campaign_id

  * dòng của tập đánh giá không có trong file nhãn -> ``is_anomaly = 0``, ``eval_exclude = False``;
  * ``is_anomaly`` thiếu nhưng có ``label`` (layout cũ của src/injection) -> dùng ``label``;
  * ``eval_exclude``/``scenario``/``campaign_id`` thiếu -> False / NULL / NULL.

Dòng ``eval_exclude`` bị loại TRƯỚC mọi phép tính (PR-AUC, cờ, cảnh báo/ngày, chiến dịch).
Nhãn chỉ được đọc ở đây — module này không bao giờ được gọi trong ``fit`` của một baseline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import polars as pl

from src.evaluation.metrics import (
    DEFAULT_DAILY_BUDGETS,
    campaign_recall,
    operating_point,
    pr_auc_by_subset,
    recall_at_daily_budget,
)

__all__ = ["LABEL_KEYS", "load_eval_labels", "align_eval_labels", "evaluate_detector"]

LABEL_KEYS = ["DomainName", "UserName", "day"]


def load_eval_labels(path: Path | str, *, allow_implicit_positive: bool = False) -> pl.DataFrame:
    """Đọc và chuẩn hoá file nhãn về đúng 7 cột; trùng khoá -> lỗi (nhãn phải duy nhất theo dòng)."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Không tìm thấy file nhãn '{source}'.")
    df = pl.read_csv(source) if source.suffix.lower() == ".csv" else pl.read_parquet(source)
    missing = [c for c in LABEL_KEYS if c not in df.columns]
    if missing:
        raise ValueError(f"File nhãn '{source}' thiếu cột {missing}.")
    if "is_anomaly" not in df.columns:
        if "label" not in df.columns:
            if not allow_implicit_positive:
                raise ValueError(f"File nhãn '{source}' cần cột 'is_anomaly' (hoặc 'label').")
            df = df.with_columns(pl.lit(1, pl.Int8).alias("is_anomaly"))
        else:
            df = df.rename({"label": "is_anomaly"})
    defaults = {"eval_exclude": pl.lit(False), "scenario": pl.lit(None, pl.String), "campaign_id": pl.lit(None, pl.String)}
    df = df.with_columns([expr.alias(c) for c, expr in defaults.items() if c not in df.columns])
    df = df.select(
        pl.col("DomainName").cast(pl.String),
        pl.col("UserName").cast(pl.String),
        pl.col("day").cast(pl.Int64),
        pl.col("is_anomaly").cast(pl.Int8),
        pl.col("eval_exclude").cast(pl.Boolean).fill_null(False),
        pl.col("scenario").cast(pl.String),
        pl.col("campaign_id").cast(pl.String),
    )
    bad = sorted(set(df["is_anomaly"].drop_nulls().unique().to_list()) - {0, 1})
    if bad or df["is_anomaly"].null_count():
        raise ValueError(f"is_anomaly của '{source}' chỉ được chứa 0/1 (gặp {bad or 'NULL'}).")
    dup = df.height - df.select(LABEL_KEYS).unique().height
    if dup:
        raise ValueError(f"File nhãn '{source}' có {dup} khoá (DomainName, UserName, day) bị lặp.")
    return df


def align_eval_labels(eval_keys: pl.DataFrame, labels: pl.DataFrame) -> pl.DataFrame:
    """Nhãn khớp từng dòng của ``eval_keys`` theo đúng thứ tự (thiếu -> âm, không loại)."""
    keys = eval_keys.select(
        pl.col("DomainName").cast(pl.String), pl.col("UserName").cast(pl.String), pl.col("day").cast(pl.Int64)
    )
    out = keys.join(labels, on=LABEL_KEYS, how="left", maintain_order="left")
    if out.height != keys.height:
        raise ValueError("Nối nhãn làm đổi số dòng — kiểm tra khoá trùng.")
    return out.with_columns(pl.col("is_anomaly").fill_null(0), pl.col("eval_exclude").fill_null(False))


def evaluate_detector(
    scores: Any,
    threshold: float,
    aligned: pl.DataFrame,
    daily_budgets: Sequence[int] = DEFAULT_DAILY_BUDGETS,
    subset_col: str = "scenario",
) -> Dict[str, Any]:
    """
    Bộ chỉ số của MỘT phương pháp trên MỘT tập đánh giá (sau khi loại ``eval_exclude``).

    ``threshold`` là ngưỡng đã fit trên điểm TRAIN của chính phương pháp (thresholding.py); cờ = điểm > ngưỡng.
    Trả ``{"summary": {...}, "subsets": {tên: {...}}}``.
    """
    from src.baselines.thresholding import alerts_per_day

    arr_s = np.asarray(scores, dtype=np.float64).ravel()
    if arr_s.size != aligned.height:
        raise ValueError(f"scores ({arr_s.size}) và nhãn ({aligned.height}) phải cùng độ dài.")
    keep = ~aligned["eval_exclude"].to_numpy().astype(bool)
    s = arr_s[keep]
    lab = aligned["is_anomaly"].to_numpy()[keep].astype(np.int8)
    days = aligned["day"].to_numpy()[keep]
    flags = s > float(threshold)

    summary: Dict[str, Any] = {"n_rows": float(keep.size), "n_excluded": float((~keep).sum()), "n_eval": float(s.size)}
    summary.update(operating_point(lab, flags))
    summary.update(alerts_per_day(flags, days))
    summary.update(campaign_recall(lab, flags, aligned["campaign_id"].to_numpy()[keep]))
    for n in daily_budgets:
        summary[f"recall_at_{int(n)}_per_day"] = (
            recall_at_daily_budget(lab, s, days, int(n)) if lab.any() else float("nan")
        )
    subsets = pr_auc_by_subset(lab, s, aligned[subset_col].to_numpy()[keep])
    summary["pr_auc"] = subsets["__all__"]["pr_auc"]
    summary["random_pr_auc"] = subsets["__all__"]["random_pr_auc"]
    return {"summary": summary, "subsets": subsets}
