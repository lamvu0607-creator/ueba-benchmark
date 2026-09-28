"""
Feature Preprocessor & Normalizer.
Chuyển đổi ma trận đặc trưng thô (Tier 3: data/features/raw/) sang ma trận đã chuẩn hóa sẵn sàng cho huấn luyện (Tier 4: data/processed/).
Bám sát hợp đồng Feature Schema v2.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional
import numpy as np
import polars as pl

logger = logging.getLogger("ueba_benchmark.features.preprocessor")


def normalize_features(
    df_raw: pl.DataFrame,
    fill_null_strategy: str = "indicator",
) -> pl.DataFrame:
    """
    Áp dụng các biến đổi tiền xử lý theo Feature Schema v2:
    - log1p cho các cột volume và counts lệch nặng:
        log_total_logons = log1p(total_logons)
        log_distinct_hosts = log1p(distinct_hosts)
        rare_logon_type_count_log = log1p(rare_logon_type_count)
    - Xử lý NULL có kiểm soát:
        - failure_locked_out_share: fill 0 khi failure_count = 0
        - interarrival_dt_mean / delta_t_cv: fill 0 nếu dùng indicator flag is_single_event
    """
    df = df_raw.clone()

    # 1. Các biến đổi log1p
    df = df.with_columns([
        (pl.col("total_logons").cast(pl.Float64).log1p()).alias("log_total_logons"),
        (pl.col("distinct_hosts").cast(pl.Float64).log1p()).alias("log_distinct_hosts"),
        (pl.col("rare_logon_type_count").cast(pl.Float64).log1p()).alias("rare_logon_type_count_log"),
    ])

    # 2. Xử lý NULL hợp lệ
    if fill_null_strategy == "indicator":
        # Không có thất bại -> tỷ trọng lockout bằng 0
        df = df.with_columns(
            pl.col("failure_locked_out_share").fill_null(0.0).alias("failure_locked_out_share")
        )
        # 1 sự kiện -> dt_mean và dt_cv = 0 (đã có cờ is_single_event = 1 giải thích)
        df = df.with_columns([
            pl.col("interarrival_dt_mean").fill_null(0.0).alias("interarrival_dt_mean"),
            pl.col("delta_t_cv").fill_null(0.0).alias("delta_t_cv"),
        ])

    return df


def prepare_processed_dataset(
    raw_feature_path: Path | str,
    output_path: Path | str,
) -> pl.DataFrame:
    """
    Nạp feature_matrix_raw.parquet, chuẩn hóa theo schema v2 và ghi vào data/processed/.
    """
    raw_path = Path(raw_feature_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file ma trận đặc trưng thô: {raw_path}")

    logger.info(f"Đang nạp dữ liệu từ '{raw_path}'...")
    df_raw = pl.read_parquet(raw_path)

    logger.info("Đang thực hiện chuẩn hóa đặc trưng (Tier 4: Processed)...")
    df_processed = normalize_features(df_raw)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df_processed.write_parquet(out_file, compression="snappy")
    logger.info(f"Đã lưu ma trận chuẩn hóa tại '{out_file}' ({df_processed.height:,} dòng).")

    return df_processed
