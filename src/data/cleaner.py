"""
Module Data Cleaner - Làm sạch dữ liệu sự kiện Windows Event Logs 4624 & 4625.
Căn cứ kỹ thuật: Báo cáo khảo sát khuyết thiếu (Missing Patterns Day 16).

Các quyết định xử lý làm sạch:
1. Điền khuyết LogonTypeDescription theo ánh xạ 1-1 với LogonType (chuẩn Microsoft).
2. Xóa bỏ hoàn toàn các cột Subject* và Process* (tỷ lệ khuyết >92% - 100%, gây nhiễu).
3. Điền nhãn "Unknown" cho các trường cốt lõi bị khuyết: Source, LogonID, DomainName.
4. Lọc bỏ các bản ghi trùng lặp hoàn toàn (Exact Duplicates).
5. Hiệu chỉnh chuỗi thời gian cho hiện tượng lệch giờ mùa hè (DST - Daylight Saving Time) từ Day 42.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import polars as pl

logger = logging.getLogger("ueba_benchmark.data.cleaner")

# Bảng ánh xạ LogonType -> LogonTypeDescription chuẩn Microsoft Security Auditing
LOGON_TYPE_MAP: Dict[int, str] = {
    0: "System",
    2: "Interactive",
    3: "Network",
    4: "Batch",
    5: "Service",
    7: "Unlock",
    8: "NetworkCleartext",
    9: "NewCredentials",
    10: "RemoteInteractive",
    11: "CachedInteractive",
    12: "CachedRemoteInteractive",
    13: "CachedUnlock",
}

# Các cột cần giữ lại cho bài toán UEBA (đã loại bỏ Subject*, Process*, Status)
ESSENTIAL_COLUMNS: List[str] = [
    "Time",
    "EventID",
    "LogHost",
    "LogonType",
    "LogonTypeDescription",
    "UserName",
    "DomainName",
    "LogonID",
    "Source",
    "AuthenticationPackage",
    "FailureReason",
]


def clean_single_day(
    day: int,
    interim_dir: Path | str,
    output_dir: Path | str,
    adjust_dst: bool = True,
    dst_start_day: int = 42,
    dst_offset_seconds: int = -3600,
) -> Dict[str, Any]:
    """
    Làm sạch dữ liệu sự kiện của 1 ngày cụ thể (kết hợp cả 4624 và 4625).
    
    Args:
        day: Số thứ tự ngày (1, 2, ...).
        interim_dir: Thư mục chứa log interim (gồm event_4624/ và event_4625/).
        output_dir: Thư mục lưu file log sạch.
        adjust_dst: Bật/tắt hiệu chỉnh lệch giờ mùa hè (DST).
        dst_start_day: Ngày bắt đầu có DST (mặc định: Day 42).
        dst_offset_seconds: Độ lệch giây cần bù (mặc định: -3600 giây = -1 giờ).
        
    Returns:
        Dict chứa thống kê số liệu trước và sau khi làm sạch.
    """
    interim_path = Path(interim_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    path_4624 = interim_path / "event_4624" / f"event_4624_day-{day:02d}.parquet"
    path_4625 = interim_path / "event_4625" / f"event_4625_day-{day:02d}.parquet"

    has_4624 = path_4624.exists()
    has_4625 = path_4625.exists()

    if not has_4624 and not has_4625:
        logger.warning(f"[Day {day:02d}] Không tìm thấy file dữ liệu 4624 và 4625 tại '{interim_path}'.")
        return {"day": day, "status": "missing", "raw_rows": 0, "cleaned_rows": 0}

    t0 = time.time()
    scan_list = []

    # 1. Đọc lười và chọn lọc cột thiết yếu (loại bỏ Subject* và Process*)
    if has_4624:
        lf_24 = pl.scan_parquet(path_4624)
        cols_24_exist = [c for c in ESSENTIAL_COLUMNS if c in lf_24.collect_schema().names() and c != "FailureReason"]
        lf_24 = lf_24.select(cols_24_exist).with_columns(
            pl.lit(None, dtype=pl.String).alias("FailureReason")
        )
        scan_list.append(lf_24)

    if has_4625:
        lf_25 = pl.scan_parquet(path_4625)
        cols_25_exist = [c for c in ESSENTIAL_COLUMNS if c in lf_25.collect_schema().names()]
        lf_25 = lf_25.select(cols_25_exist)
        scan_list.append(lf_25)

    events_lazy = pl.concat(scan_list, how="diagonal")

    # 2. Lọc bỏ các dòng UserName không hợp lệ (null, rỗng, nan)
    events_lazy = events_lazy.filter(
        pl.col("UserName").is_not_null()
        & (pl.col("UserName").str.strip_chars() != "")
        & (pl.col("UserName").str.to_lowercase().is_in(["null", "none", "nan"]).not_())
    )

    # 3. Điền giá trị "Unknown" cho các trường cốt lõi bị khuyết
    events_lazy = events_lazy.with_columns([
        # DomainName
        pl.when(pl.col("DomainName").is_null() | (pl.col("DomainName").str.strip_chars() == ""))
        .then(pl.lit("Unknown"))
        .otherwise(pl.col("DomainName").str.strip_chars().str.to_lowercase())
        .alias("DomainName"),

        # LogonID
        pl.when(pl.col("LogonID").is_null() | (pl.col("LogonID").str.strip_chars() == ""))
        .then(pl.lit("Unknown"))
        .otherwise(pl.col("LogonID").str.strip_chars())
        .alias("LogonID"),

        # Source
        pl.when(pl.col("Source").is_null() | (pl.col("Source").str.strip_chars() == ""))
        .then(pl.lit("Unknown"))
        .otherwise(pl.col("Source").str.strip_chars())
        .alias("Source"),
    ])

    # 4. Ánh xạ LogonTypeDescription từ LogonType
    events_lazy = events_lazy.with_columns(
        pl.when(pl.col("LogonTypeDescription").is_null() | (pl.col("LogonTypeDescription").str.strip_chars() == ""))
        .then(
            pl.col("LogonType").replace_strict(LOGON_TYPE_MAP, default=pl.lit("Unknown"))
        )
        .otherwise(pl.col("LogonTypeDescription"))
        .alias("LogonTypeDescription")
    )

    # 5. Hiệu chỉnh lệch giờ mùa hè (DST) nếu có cấu hình và qua ngưỡng ngày
    if adjust_dst and day >= dst_start_day:
        events_lazy = events_lazy.with_columns(
            (pl.col("Time") + dst_offset_seconds).alias("Time")
        )

    # Thu thập dữ liệu vào DataFrame
    df = events_lazy.collect()
    raw_rows = df.height

    # 6. Loại bỏ bản ghi trùng lặp hoàn toàn (Exact Duplicates)
    df_clean = df.unique()
    cleaned_rows = df_clean.height
    duplicates_dropped = raw_rows - cleaned_rows

    # 7. Sắp xếp chuỗi thời gian cho tối ưu truy vấn
    df_clean = df_clean.sort(["Time", "DomainName", "UserName"])

    # 8. Lưu kết quả ra file Parquet sạch
    out_file = out_path / f"cleaned_day-{day:02d}.parquet"
    df_clean.write_parquet(out_file, compression="snappy")

    duration = time.time() - t0
    file_size_mb = out_file.stat().st_size / (1024 * 1024)

    logger.info(
        f"[Day {day:02d}] Hoàn tất làm sạch trong {duration:.2f}s | "
        f"Gốc: {raw_rows:,} dòng -> Sạch: {cleaned_rows:,} dòng "
        f"(Đã xóa {duplicates_dropped:,} dòng trùng) | Dung lượng: {file_size_mb:.2f} MB"
    )

    return {
        "day": day,
        "status": "success",
        "raw_rows": raw_rows,
        "cleaned_rows": cleaned_rows,
        "duplicates_dropped": duplicates_dropped,
        "file_size_mb": file_size_mb,
        "output_path": str(out_file),
        "duration_seconds": duration,
    }


def clean_dataset(
    start_day: int,
    end_day: int,
    interim_dir: Path | str,
    output_dir: Path | str,
    adjust_dst: bool = True,
) -> List[Dict[str, Any]]:
    """
    Làm sạch toàn bộ chuỗi ngày từ start_day đến end_day.
    """
    logger.info(f"=== BẮT ĐẦU CHẶNG 1: LÀM SẠCH DỮ LIỆU (DAY {start_day:02d} -> DAY {end_day:02d}) ===")
    results = []
    total_raw = 0
    total_cleaned = 0
    total_dup = 0
    t_start = time.time()

    for d in range(start_day, end_day + 1):
        res = clean_single_day(
            day=d,
            interim_dir=interim_dir,
            output_dir=output_dir,
            adjust_dst=adjust_dst,
        )
        results.append(res)
        if res.get("status") == "success":
            total_raw += res["raw_rows"]
            total_cleaned += res["cleaned_rows"]
            total_dup += res["duplicates_dropped"]

    total_time = time.time() - t_start
    logger.info("=== HOÀN TẤT CHẶNG 1: LÀM SẠCH DỮ LIỆU ===")
    logger.info(
        f"Tổng cộng: {len(results)} ngày | Tổng log gốc: {total_raw:,} | "
        f"Tổng log sạch: {total_cleaned:,} | Đã loại trùng: {total_dup:,} | "
        f"Tổng thời gian: {total_time:.2f}s"
    )
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
    parser = argparse.ArgumentParser(description="Chương trình làm sạch Windows Event Logs theo chuẩn PDF")
    parser.add_argument("--start-day", type=int, default=1, help="Ngày bắt đầu (mặc định: 1)")
    parser.add_argument("--end-day", type=int, default=3, help="Ngày kết thúc (mặc định: 3)")
    parser.add_argument("--interim-dir", type=str, default="data/interim", help="Thư mục log interim")
    parser.add_argument("--output-dir", type=str, default="data/cleaned", help="Thư mục xuất log sạch")
    parser.add_argument("--no-dst", action="store_true", help="Tắt hiệu chỉnh DST")

    args = parser.parse_args()
    clean_dataset(
        start_day=args.start_day,
        end_day=args.end_day,
        interim_dir=Path(args.interim_dir),
        output_dir=Path(args.output_dir),
        adjust_dst=not args.no_dst,
    )
