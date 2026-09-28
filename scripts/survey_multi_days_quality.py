"""CLI Script khảo sát chất lượng dữ liệu log Windows đa ngày bằng Polars.

Cách sử dụng:
    # 1. Chạy khảo sát 2 ngày kiểm chứng:
    python scripts/survey_multi_days_quality.py --days day-01,day-02

    # 2. Chạy khảo sát toàn bộ 60 ngày (mặc định) và lưu CSV, biểu đồ:
    python scripts/survey_multi_days_quality.py

    # 3. Chạy 10 ngày đầu tiên:
    python scripts/survey_multi_days_quality.py --max-days 10
"""

import argparse
from pathlib import Path
import sys

# Đảm bảo UTF-8 trên Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Thêm PROJECT_ROOT vào sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.quality_survey import (
    load_yaml_config,
    survey_multiple_days,
    plot_survey_trends,
    extract_day_number,
)


def print_survey_table(df):
    """In bảng tóm tắt kết quả khảo sát ra console trực quan."""
    headers = [
        "Ngày",
        "Event 4624",
        "Fail (%)",
        "Users",
        "Hosts",
        "Máy (%)",
        "Ngoài giờ(%)",
        "Exact Dup",
        "Session Dup",
        "Bursty Dup(%)",
        "Lỗi Null",
    ]
    col_widths = [8, 13, 9, 8, 8, 8, 13, 11, 12, 14, 9]

    header_str = " | ".join(f"{h:<{w}}" for h, w in zip(headers, col_widths))
    sep_str = "-+-".join("-" * w for w in col_widths)
    print("\n" + header_str)
    print(sep_str)

    for row in df.iter_rows(named=True):
        c4624_str = f"{row['count_4624']:,}"
        users_str = f"{row['distinct_users']:,}"
        hosts_str = f"{row['distinct_hosts']:,}"
        fail_str = f"{row['failure_rate_pct']:.2f}"
        mach_str = f"{row['machine_account_pct']:.1f}"
        off_str = f"{row['off_hours_pct']:.1f}"
        exact_str = f"{row.get('exact_dups', 0):,}"
        session_str = f"{row.get('session_dups', 0):,}"
        bursty_str = f"{row.get('bursty_dup_pct', 0.0):.1f}%"
        null_str = str(row['null_critical_count'])

        row_str = " | ".join([
            f"{row['day']:<{col_widths[0]}}",
            f"{c4624_str:<{col_widths[1]}}",
            f"{fail_str:<{col_widths[2]}}",
            f"{users_str:<{col_widths[3]}}",
            f"{hosts_str:<{col_widths[4]}}",
            f"{mach_str:<{col_widths[5]}}",
            f"{off_str:<{col_widths[6]}}",
            f"{exact_str:<{col_widths[7]}}",
            f"{session_str:<{col_widths[8]}}",
            f"{bursty_str:<{col_widths[9]}}",
            f"{null_str:<{col_widths[10]}}",
        ])
        print(row_str)
    print(sep_str)


def main():
    parser = argparse.ArgumentParser(
        description="Khảo sát vĩ mô chất lượng dữ liệu Parquet Interim đa ngày (Polars)"
    )
    parser.add_argument(
        "--interim-dir",
        type=str,
        default="data/interim",
        help="Thư mục interim chứa event_4624 và event_4625 (mặc định: data/interim)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/system_config.yaml",
        help="Đường dẫn file cấu hình hệ thống (mặc định: configs/system_config.yaml)",
    )
    parser.add_argument(
        "--days",
        type=str,
        default="all",
        help="Danh sách ngày cần khảo sát (ví dụ: 'day-01,day-02' hoặc 'all')",
    )
    parser.add_argument(
        "--max-days",
        type=int,
        default=None,
        help="Giới hạn số ngày tối đa khảo sát (tiện cho test nhanh)",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="reports/quality_survey_summary.csv",
        help="Đường dẫn lưu file CSV tổng hợp (mặc định: reports/quality_survey_summary.csv)",
    )
    parser.add_argument(
        "--output-plot",
        type=str,
        default="reports/figures/multi_day_trend.png",
        help="Đường dẫn lưu file biểu đồ xu hướng (mặc định: reports/figures/multi_day_trend.png)",
    )

    args = parser.parse_args()
    interim_path = Path(args.interim_dir)
    config = load_yaml_config(args.config)

    print("=" * 85)
    print("UEBA BENCHMARK - KHẢO SÁT CHẤT LƯỢNG DỮ LIỆU ĐA NGÀY (POLARS ENGINE)")
    print(f"Thư mục Interim:  {interim_path}")
    print(f"File Cấu hình:    {args.config}")
    print("=" * 85)

    # Thu thập danh sách các ngày có sẵn trong thư mục interim
    event_4624_dir = interim_path / "event_4624"
    if not event_4624_dir.exists():
        print(f"Lỗi: Không tìm thấy thư mục '{event_4624_dir}'.")
        return

    available_files = list(event_4624_dir.glob("*.parquet"))
    import re
    available_days = []
    for f in available_files:
        m = re.search(r"day[-_]?(\d+)", f.name, re.IGNORECASE)
        if m:
            available_days.append(f"day-{int(m.group(1)):02d}")
    available_days = sorted(list(set(available_days)), key=extract_day_number)

    if not available_days:
        print("Lỗi: Không tìm thấy file parquet nào trong thư mục interim.")
        return

    # Xác định danh sách ngày cần chạy
    if args.days.strip().lower() == "all":
        target_days = available_days
    else:
        target_days = [d.strip() for d in args.days.split(",") if d.strip()]

    if args.max_days is not None:
        target_days = target_days[: args.max_days]

    print(f"\n--> Bắt đầu khảo sát trên {len(target_days)} ngày...")
    summary_df = survey_multiple_days(
        interim_dir=interim_path,
        days_list=target_days,
        config=config,
    )

    # Hiển thị bảng ra console
    print_survey_table(summary_df)

    # Xuất file CSV
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.write_csv(out_csv)
    print(f"\n--> [CSV] Đã xuất bảng tổng hợp tại: {out_csv}")

    # Vẽ và xuất biểu đồ
    out_plot = Path(args.output_plot)
    plot_survey_trends(
        summary_df,
        output_path=out_plot,
        title=f"Khảo sát Xu hướng & Nhịp điệu {len(target_days)} Ngày (Event 4624 & 4625)",
    )
    print(f"--> [PLOT] Đã xuất biểu đồ xu hướng tại: {out_plot}")

    print("\n" + "=" * 85)
    print("HOÀN TẤT KHẢO SÁT CHẤT LƯỢNG ĐA NGÀY!")
    print("=" * 85)


if __name__ == "__main__":
    main()
