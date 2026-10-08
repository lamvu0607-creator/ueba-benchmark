"""
Sinh ``labels.parquet`` của một run từ ``injection_manifest.csv``.

Đầu ra đúng hợp đồng của ``src.evaluation.labeled_eval.load_eval_labels``::

    DomainName, UserName, day, is_anomaly (0/1), eval_exclude (bool), scenario, campaign_id

Mỗi dòng manifest là một lần tiêm vào MỘT khoá (tài khoản, ngày) -> một dòng nhãn dương tính. Dòng
nào của tập đánh giá không có trong file nhãn được ``align_eval_labels`` coi là âm tính, nên file chỉ
chứa dòng dương tính.

``DomainName`` được chuẩn hoá giống hệt extractor (lowercase + strip, null/rỗng -> "Unknown") để
khoá nhãn khớp khoá của ma trận đặc trưng dù manifest ghi giá trị thô hay đã chuẩn hoá.

Khi bật tiêm nhiều ngày sau này: manifest chỉ cần thêm cột ``days`` (danh sách ngày, phân tách
bằng ``;``) — mỗi ngày trong đó thành một dòng nhãn riêng; không có cột này thì dùng ``day``.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import polars as pl

__all__ = ["LABEL_COLUMNS", "MANIFEST_REQUIRED", "labels_from_manifest", "write_run_labels"]

LABEL_COLUMNS: List[str] = ["DomainName", "UserName", "day", "is_anomaly", "eval_exclude", "scenario", "campaign_id"]
MANIFEST_REQUIRED: List[str] = ["inj_id", "scenario", "DomainName", "UserName", "day"]


def _normalized_domain() -> pl.Expr:
    c = pl.col("DomainName").cast(pl.String)
    return (
        pl.when(c.is_null() | (c.str.strip_chars() == ""))
        .then(pl.lit("Unknown"))
        .otherwise(c.str.strip_chars().str.to_lowercase())
        .alias("DomainName")
    )


def labels_from_manifest(manifest: pl.DataFrame) -> pl.DataFrame:
    """Manifest (một dòng / lần tiêm) -> bảng nhãn dương tính, khoá duy nhất."""
    missing = [c for c in MANIFEST_REQUIRED if c not in manifest.columns]
    if missing:
        raise ValueError(f"Manifest thiếu cột {missing}.")
    df = manifest
    if "campaign_id" not in df.columns:
        df = df.with_columns(pl.lit(None, pl.String).alias("campaign_id"))
    if "days" in df.columns:
        df = (
            df.with_columns(
                pl.when(pl.col("days").is_null())
                .then(pl.col("day").cast(pl.String))
                .otherwise(pl.col("days").cast(pl.String))
                .str.split(";")
                .alias("_days")
            )
            .explode("_days", empty_as_null=True)
            .with_columns(pl.col("_days").str.strip_chars().cast(pl.Int64).alias("day"))
        )
    labels = df.select(
        _normalized_domain(),
        pl.col("UserName").cast(pl.String),
        pl.col("day").cast(pl.Int64),
        pl.lit(1, pl.Int8).alias("is_anomaly"),
        pl.lit(False).alias("eval_exclude"),
        pl.col("scenario").cast(pl.String),
        pl.col("campaign_id").cast(pl.String),
    )
    if labels["day"].null_count() or labels["UserName"].null_count():
        raise ValueError("Manifest có dòng thiếu UserName hoặc day.")
    dup = labels.height - labels.select(["DomainName", "UserName", "day"]).unique().height
    if dup:
        raise ValueError(
            f"{dup} khoá (DomainName, UserName, day) bị tiêm nhiều lần — mỗi tài khoản-ngày chỉ được một nhãn."
        )
    return labels.sort(["day", "DomainName", "UserName"])


def write_run_labels(layout) -> Path:
    """Đọc ``layout.manifest_path`` -> ghi ``layout.labels_path`` (``layout`` là ``RunLayout``)."""
    manifest_path = Path(layout.manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy manifest '{manifest_path}'.")
    manifest = pl.read_csv(manifest_path, infer_schema_length=None, schema_overrides={"campaign_id": pl.String})
    labels = labels_from_manifest(manifest)
    out = Path(layout.labels_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    labels.write_parquet(out)
    return out
