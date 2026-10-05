"""
⚠️ FILE LƯU TRỮ — KHÔNG DÙNG ĐỂ CHẠY ⚠️
Đây là bản v1.0 (trước khi sửa các bất thường của bộ đặc trưng), chỉ giữ để đối chiếu.
Bản dùng để chạy là: extract_account_day_matrix.py (bản 2.0 — tự dò dữ liệu/config, không cần tham số).
Nếu chạy file này, kết quả sẽ quay lại schema cũ (30 đặc trưng, có 13 cặp |rho| >= 0.85, 2 đặc trưng chết).
Xem lý do và danh sách khác biệt: reports/archive/bao_cao_sua_bo_dac_trung.md §4–§5.

extract_account_day_matrix.py  (v1.0 — bản lưu trữ)

Pipeline tổng hợp log thô (Event 4624 & 4625) thành Ma trận Đặc trưng (Tài khoản × Ngày).
Trích xuất >= 15 đặc trưng an ninh mạng cho từng cặp (UserName, Day) phục vụ mô hình UEBA.

Đầu vào:
  - data/interim/event_4624/event_4624_day-{day:02d}.parquet
  - data/interim/event_4625/event_4625_day-{day:02d}.parquet

Đầu ra:
  - data/features/daily/user_features_day-{day:02d}.parquet (Từng ngày)
  - data/features/account_day_matrix.parquet (Ma trận tổng hợp Tài khoản x Ngày)
  - data/features/pivot_account_day_logons.parquet (Ma trận 2D Tài khoản x Ngày của volume)
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys
import time

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import polars as pl


def extract_features_single_day(day: int, interim_dir: Path, output_daily_dir: Path | None = None) -> pl.DataFrame:
    """
    Trích xuất vector đặc trưng cho tất cả tài khoản trong một ngày cụ thể (UserName x day).
    Trích xuất >= 15 đặc trưng hành vi bảo mật.
    """
    path_4624 = interim_dir / "event_4624" / f"event_4624_day-{day:02d}.parquet"
    path_4625 = interim_dir / "event_4625" / f"event_4625_day-{day:02d}.parquet"

    if not path_4624.exists() or not path_4625.exists():
        print(f"[CẢNH BÁO] Không tìm thấy dữ liệu cho Day {day:02d}, bỏ qua.")
        return pl.DataFrame()

    cols_4624 = [
        "Time", "EventID", "UserName", "LogHost", "LogonType", 
        "AuthenticationPackage", "ProcessID", "ProcessName", "Source", "LogonID", "DomainName"
    ]
    cols_4625 = [
        "Time", "EventID", "UserName", "LogHost", "LogonType", 
        "AuthenticationPackage", "Status", "ProcessID", "ProcessName", "Source", "LogonID", "DomainName"
    ]

    # 1. Quét lười (Lazy Evaluation)
    lf_4624 = (
        pl.scan_parquet(path_4624)
        .select(cols_4624)
        .with_columns(pl.lit(None, dtype=pl.String).alias("Status"))
    )
    lf_4625 = pl.scan_parquet(path_4625).select(cols_4625)

    events = pl.concat([lf_4624, lf_4625], how="diagonal")

    # 2. Làm sạch trường UserName
    events = events.filter(
        pl.col("UserName").is_not_null()
        & (pl.col("UserName").str.strip_chars() != "")
        & (pl.col("UserName").str.to_lowercase().is_in(["null", "none", "nan"]).not_())
    )

    # 3. Tính giờ trong ngày [0 - 23]
    events = events.with_columns(
        ((pl.col("Time") % 86400) // 3600).cast(pl.Int32).alias("hour")
    )

    # 4. Sắp xếp để tính khoảng cách liên tiếp delta_t
    df_events = events.sort(["UserName", "Time"]).collect()

    df_events = df_events.with_columns(
        pl.col("Time").diff().over("UserName").alias("delta_t")
    )

    # 5. Tổng hợp đặc trưng theo UserName cho Ngày này
    df_day = df_events.group_by("UserName").agg([
        # Nhóm 1: Volume & Tần suất
        pl.len().alias("total_logons"),
        (pl.col("EventID") == 4625).sum().alias("failure_count"),

        # Nhóm 2: Thời gian & Nhịp sinh học
        (((pl.col("hour") >= 22) | (pl.col("hour") <= 5)).sum()).alias("off_hours_count"),
        (((pl.col("hour") >= 8) & (pl.col("hour") <= 17)).sum()).alias("work_hours_count"),
        pl.col("delta_t").mean().alias("interarrival_dt_mean"),
        pl.col("delta_t").std().alias("interarrival_dt_std"),
        ((pl.col("delta_t").is_not_null()) & (pl.col("delta_t") <= 2)).sum().alias("burst_logon_count"),

        # Nhóm 3: Vector truy cập (LogonType)
        (pl.col("LogonType") == 3).sum().alias("type3_network_count"),
        (pl.col("LogonType") == 2).sum().alias("type2_interactive_count"),
        (pl.col("LogonType") == 10).sum().alias("type10_rdp_count"),
        (pl.col("LogonType") == 5).sum().alias("type5_service_count"),
        (pl.col("LogonType") == 4).sum().alias("type4_batch_count"),

        # Nhóm 4: Bảo mật & Mã lỗi xác thực
        pl.col("AuthenticationPackage").str.to_lowercase().str.contains("ntlm").fill_null(False).sum().alias("ntlm_count"),
        pl.col("Status").str.to_lowercase().str.contains("c000006a").fill_null(False).sum().alias("wrong_password_count"),
        pl.col("Status").str.to_lowercase().str.contains("c0000064").fill_null(False).sum().alias("unknown_user_count"),

        # Nhóm 5: Bậc mạng Fan-out
        pl.col("LogHost").n_unique().alias("distinct_hosts_count"),

        # Nhóm 6: Ngữ cảnh Nguồn (Source) [MỚI]
        ((pl.col("Source").is_null()) | (pl.col("Source").str.strip_chars() == "")).sum().alias("missing_source_count"),
        ((pl.col("Source").is_not_null()) & (pl.col("Source") != pl.col("LogHost"))).sum().alias("remote_logon_count"),
        (pl.col("Source") == pl.col("LogHost")).sum().alias("local_logon_count"),
        pl.col("Source").drop_nulls().n_unique().alias("distinct_sources_count"),

        # Nhóm 7: Ngữ cảnh Tiến trình (Process) [MỚI]
        pl.col("ProcessName").is_not_null().sum().alias("proc_info_count"),
        pl.col("ProcessName").str.to_lowercase().is_in(["services.exe", "lsass.exe", "winlogon.exe", "svchost.exe"]).sum().alias("system_proc_count"),
        pl.col("ProcessName").str.to_lowercase().str.starts_with("proc").fill_null(False).sum().alias("custom_proc_count"),

        # Nhóm 8: Phiên LUID & Tên miền (LogonID & DomainName) [MỚI]
        (pl.col("LogonID") == "0x3e7").sum().alias("system_logon_id_count"),
        pl.col("DomainName").n_unique().alias("domain_count"),
        pl.col("DomainName").str.to_lowercase().str.starts_with("comp").fill_null(False).sum().alias("local_domain_count")
    ]).with_columns([
        # Khóa chiều Ngày (Day coordinate)
        pl.lit(day).cast(pl.Int32).alias("day"),

        # Ratios cũ
        (pl.col("off_hours_count") / pl.col("total_logons")).alias("off_hours_ratio"),
        (pl.col("work_hours_count") / pl.col("total_logons")).alias("work_hours_ratio"),
        (pl.col("failure_count") / pl.col("total_logons")).alias("failure_ratio"),

        (pl.col("type3_network_count") / pl.col("total_logons")).alias("network_ratio"),
        (pl.col("type2_interactive_count") / pl.col("total_logons")).alias("interactive_ratio"),
        (pl.col("type10_rdp_count") / pl.col("total_logons")).alias("rdp_ratio"),
        (pl.col("type5_service_count") / pl.col("total_logons")).alias("service_ratio"),
        (pl.col("type4_batch_count") / pl.col("total_logons")).alias("batch_ratio"),
        (pl.col("ntlm_count") / pl.col("total_logons")).alias("ntlm_ratio"),

        # Điền missing theo nghiệp vụ
        pl.col("interarrival_dt_mean").fill_null(86400.0),
        pl.col("interarrival_dt_std").fill_null(0.0),

        # Phân loại thực thể
        pl.when(pl.col("UserName").str.ends_with("$")).then(pl.lit("Machine"))
        .when(pl.col("UserName").str.starts_with("User")).then(pl.lit("User"))
        .otherwise(pl.lit("Service"))
        .alias("entity_type"),

        # Log transform
        pl.col("total_logons").log1p().alias("log_total_logons"),
        pl.col("distinct_hosts_count").log1p().alias("log_distinct_hosts"),

        # Cờ nhị phân (UInt8) & Tỷ lệ ngữ cảnh mới [MỚI]
        (pl.col("missing_source_count") > 0).cast(pl.UInt8).alias("has_missing_source"),
        (pl.col("missing_source_count") / pl.col("total_logons")).alias("missing_source_ratio"),
        (pl.col("remote_logon_count") > 0).cast(pl.UInt8).alias("has_remote_logon"),
        (pl.col("remote_logon_count") / pl.col("total_logons")).alias("remote_logon_ratio"),
        (pl.col("local_logon_count") > 0).cast(pl.UInt8).alias("has_local_logon"),

        (pl.col("proc_info_count") > 0).cast(pl.UInt8).alias("has_process_info"),
        (pl.col("system_proc_count") > 0).cast(pl.UInt8).alias("has_system_process"),
        (pl.col("custom_proc_count") > 0).cast(pl.UInt8).alias("has_custom_proc"),

        (pl.col("system_logon_id_count") > 0).cast(pl.UInt8).alias("has_system_logon_id"),

        (pl.col("domain_count") > 1).cast(pl.UInt8).alias("has_multi_domains"),
        (pl.col("local_domain_count") > 0).cast(pl.UInt8).alias("has_local_domain")
    ])

    # Sắp xếp lại thứ tự cột cho rõ ràng: [UserName, day, entity_type, ... features] (35 cột)
    ordered_cols = [
        "UserName",
        "day",
        "entity_type",
        "total_logons",
        "failure_count",
        "failure_ratio",
        "off_hours_count",
        "off_hours_ratio",
        "work_hours_ratio",
        "interarrival_dt_mean",
        "interarrival_dt_std",
        "burst_logon_count",
        "network_ratio",
        "interactive_ratio",
        "rdp_ratio",
        "service_ratio",
        "batch_ratio",
        "ntlm_ratio",
        "wrong_password_count",
        "unknown_user_count",
        "distinct_hosts_count",
        "log_total_logons",
        "log_distinct_hosts",
        # 12 đặc trưng ngữ cảnh nhị phân & tỷ lệ mới
        "has_missing_source",
        "missing_source_ratio",
        "has_remote_logon",
        "remote_logon_ratio",
        "has_local_logon",
        "distinct_sources_count",
        "has_process_info",
        "has_system_process",
        "has_custom_proc",
        "has_system_logon_id",
        "has_multi_domains",
        "has_local_domain"
    ]
    df_day = df_day.select(ordered_cols)

    # Lưu file lẻ từng ngày nếu có chỉ định
    if output_daily_dir is not None:
        daily_file = output_daily_dir / f"user_features_day-{day:02d}.parquet"
        df_day.write_parquet(daily_file, compression="snappy")

    return df_day


def build_account_day_matrix(
    start_day: int = 1,
    end_day: int = 3,
    interim_dir: Path = Path("data/interim"),
    output_dir: Path = Path("data/features")
) -> pl.DataFrame:
    """
    Hàm thực thi pipeline trích xuất ma trận (Tài khoản x Ngày) xuyên suốt từ start_day đến end_day.
    """
    t_start = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    daily_dir = output_dir / "daily"
    daily_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PIPELINE TỔNG HỢP LOG THÔ THÀNH MA TRẬN (TÀI KHOẢN × NGÀY)")
    print(f"Phạm vi: Day {start_day:02d} đến Day {end_day:02d}")
    print(f"Thư mục nguồn: {interim_dir.resolve()}")
    print(f"Thư mục đích:   {output_dir.resolve()}")
    print("=" * 80)

    daily_dfs: list[pl.DataFrame] = []

    for d in range(start_day, end_day + 1):
        t0 = time.time()
        print(f"\n[+] Đang xử lý Day {d:02d}...", end=" ", flush=True)
        df_d = extract_features_single_day(d, interim_dir, daily_dir)
        if df_d.height > 0:
            daily_dfs.append(df_d)
            print(f"Hoàn thành! {df_d.height:,} tài khoản ({time.time() - t0:.2f}s)")
        else:
            print("Không có dữ liệu!")

    if not daily_dfs:
        print("[LỖI] Không có dữ liệu ngày nào được xử lý thành công.")
        return pl.DataFrame()

    # Ghép toàn bộ các ngày thành Ma trận (Tài khoản x Ngày)
    t_concat = time.time()
    matrix = pl.concat(daily_dfs, how="vertical")
    print(f"\n[+] Đã ghép nối toàn bộ {len(daily_dfs)} ngày thành Ma trận (Tài khoản × Ngày) trong {time.time() - t_concat:.2f}s")
    print(f"    - Tổng số bản ghi (Tài khoản × Ngày): {matrix.height:,}")
    print(f"    - Tổng số đặc trưng trích xuất:       {matrix.width - 2} đặc trưng (ngoài 2 khóa UserName và day)")

    # 1. Lưu Ma trận chuẩn (Panel Data / Long format)
    matrix_file = output_dir / "account_day_matrix.parquet"
    matrix.write_parquet(matrix_file, compression="snappy")
    print(f"[✓] Đã lưu Ma trận (Tài khoản × Ngày) vào: {matrix_file.resolve()}")
    print(f"    Dung lượng file: {matrix_file.stat().st_size / (1024 * 1024):.2f} MB")

    # 2. Tạo bổ sung Ma trận 2D trực diện (Pivot Table: Rows=UserName, Cols=Day) cho volume
    try:
        pivot_df = matrix.pivot(
            values="total_logons",
            index="UserName",
            on="day"
        ).fill_null(0)
        # Đổi tên cột ngày cho rõ ràng: day_1, day_2...
        rename_map = {col: f"day_{int(col):02d}" for col in pivot_df.columns if col != "UserName"}
        pivot_df = pivot_df.rename(rename_map)

        pivot_file = output_dir / "pivot_account_day_logons.parquet"
        pivot_df.write_parquet(pivot_file, compression="snappy")
        print(f"[✓] Đã tạo Ma trận 2D Pivot (Accounts × Days Volume) tại: {pivot_file.resolve()}")
    except Exception as e:
        print(f"[!] Bỏ qua tạo Pivot: {e}")

    print(f"\n[HOÀN TẤT] Toàn bộ pipeline chạy xong trong: {time.time() - t_start:.2f}s")
    return matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline tổng hợp log thô thành Ma trận (Tài khoản × Ngày)")
    parser.add_argument("--start-day", type=int, default=1, help="Ngày bắt đầu (mặc định: 1)")
    parser.add_argument("--end-day", type=int, default=3, help="Ngày kết thúc (mặc định: 3)")
    parser.add_argument("--interim-dir", type=str, default="data/interim", help="Thư mục interim")
    parser.add_argument("--output-dir", type=str, default="data/features", help="Thư mục lưu ma trận")

    args = parser.parse_args()

    build_account_day_matrix(
        start_day=args.start_day,
        end_day=args.end_day,
        interim_dir=Path(args.interim_dir),
        output_dir=Path(args.output_dir)
    )


    