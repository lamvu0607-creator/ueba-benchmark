"""CLI Script kiểm định chất lượng và đối soát dữ liệu Parquet (Interim).

Cách sử dụng:
    # 1. Chạy quét nhanh toàn bộ 120 file parquet (kiểm tra schema & corruption):
    python scripts/validate_interim.py --mode fast

    # 2. Kiểm tra chất lượng dữ liệu chi tiết (null rates, dải giá trị) cho ngày chỉ định:
    python scripts/validate_interim.py --mode quality --days day-01,day-02

    # 3. Đối soát số lượng record giữa file thô bz2 và parquet:
    python scripts/validate_interim.py --mode reconcile --days day-01

    # 4. Chạy toàn diện và xuất báo cáo markdown:
    python scripts/validate_interim.py --mode all --days day-01 --report reports/validation_day01.md
"""

import argparse
from pathlib import Path
import sys

# Thêm PROJECT_ROOT vào sys.path để import an toàn
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.validate_interim import (
    scan_all_interim_integrity,
    check_data_quality,
    reconcile_raw_vs_interim,
    generate_markdown_report,
)


def print_table(headers: list, rows: list, col_widths: list = None):
    """In bảng định dạng console trực quan."""
    if not col_widths:
        col_widths = [len(h) for h in headers]
        for row in rows:
            for idx, cell in enumerate(row):
                col_widths[idx] = max(col_widths[idx], len(str(cell)))

    header_str = " | ".join(f"{h:<{w}}" for h, w in zip(headers, col_widths))
    separator = "-+-".join("-" * w for w in col_widths)
    print(header_str)
    print(separator)
    for row in rows:
        row_str = " | ".join(f"{str(cell):<{w}}" for cell, w in zip(row, col_widths))
        print(row_str)


def parse_days_argument(days_arg: str, available_days: list) -> list:
    """Xử lý tham số days (all hoặc danh sách cách nhau bởi dấu phẩy)."""
    if days_arg.strip().lower() == "all":
        return sorted(available_days)
    selected = [d.strip() for d in days_arg.split(",") if d.strip()]
    return selected


def main():
    parser = argparse.ArgumentParser(
        description="Kiểm định chất lượng và đối soát dữ liệu Parquet Interim"
    )
    parser.add_argument(
        "--interim-dir",
        type=str,
        default="data/interim",
        help="Thư mục interim chứa các thư mục event_* (mặc định: data/interim)",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw",
        help="Thư mục raw chứa các file .bz2 (mặc định: data/raw)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["fast", "quality", "reconcile", "all"],
        default="fast",
        help="Chế độ kiểm định: fast (nhanh), quality (chất lượng), reconcile (đối soát), all (toàn bộ)",
    )
    parser.add_argument(
        "--days",
        type=str,
        default="all",
        help="Danh sách các ngày cần kiểm tra (ví dụ: 'day-01' hoặc 'day-01,day-02' hoặc 'all')",
    )
    parser.add_argument(
        "--target-events",
        type=str,
        default="4624,4625",
        help="Danh sách EventID cần kiểm tra, cách nhau dấu phẩy (mặc định: 4624,4625)",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Đường dẫn file markdown xuất báo cáo kết quả (tuỳ chọn)",
    )

    args = parser.parse_args()
    interim_path = Path(args.interim_dir)
    raw_path = Path(args.raw_dir)
    target_event_ids = {int(e.strip()) for e in args.target_events.split(",") if e.strip()}

    print("=" * 80)
    print("UEBA BENCHMARK - HỆ THỐNG KIỂM ĐỊNH & ĐỐI SOÁT DỮ LIỆU INTERIM")
    print(f"Thư mục Interim: {interim_path}")
    print(f"Chế độ chạy:     {args.mode.upper()}")
    print("=" * 80)

    integrity_res = None
    quality_results = []
    reconcile_results = []

    # BƯỚC 1: Quét nhanh tính toàn vẹn (Luôn chạy vì rất nhanh < 1s)
    print("\n--> [TẦNG 1] Đang quét kiểm tra Metadata, Schema & File Integrity...")
    integrity_res = scan_all_interim_integrity(interim_path, target_event_ids)
    print(f"    - Tổng số file Parquet:  {integrity_res['total_files']}")
    print(f"    - Số file hợp lệ (Pass): {integrity_res['valid_files']}")
    print(f"    - Số file lỗi (Fail):    {integrity_res['corrupted_files']}")
    print(f"    - Dung lượng tổng:       {integrity_res['total_size_bytes'] / (1024**3):.2f} GB")
    for eid in sorted(target_event_ids):
        print(f"    - Tổng dòng Event {eid}:  {integrity_res['total_rows'].get(eid, 0):,}")

    if integrity_res["corrupted_files"] > 0:
        print("\n❌ CẢNH BÁO: Phát hiện file bị hỏng hoặc sai schema:")
        for tag, detail in integrity_res["file_details"].items():
            if not detail["is_valid"]:
                print(f"    File: {detail['filename']} -> Lỗi: {detail.get('error')}")

    available_days = integrity_res["days_found"]
    selected_days = parse_days_argument(args.days, available_days)

    # BƯỚC 2: Kiểm định Chất lượng Dữ liệu (Quality Mode hoặc All)
    if args.mode in ["quality", "all"]:
        print(f"\n--> [TẦNG 2] Đang phân tích chất lượng dữ liệu trên {len(selected_days)} ngày...")
        for day in selected_days:
            for eid in sorted(target_event_ids):
                p_file = interim_path / f"event_{eid}" / f"event_{eid}_{day}.parquet"
                if not p_file.exists():
                    continue
                q_res = check_data_quality(p_file, eid)
                quality_results.append(q_res)

                # In tóm tắt ra console
                issues = q_res["sanity_issues"]
                status = "✅ PASS" if not issues else f"⚠️ CẢNH BÁO ({len(issues)} vấn đề)"
                print(f"    [{day}] Event {eid} ({q_res['num_rows']:,} dòng): {status}")
                for iss in issues:
                    print(f"        -> {iss}")

    # BƯỚC 3: Đối soát số lượng Raw vs Interim (Reconcile Mode hoặc All)
    if args.mode in ["reconcile", "all"]:
        print(f"\n--> [TẦNG 3] Đang đối soát số lượng Raw bz2 vs Parquet trên {len(selected_days)} ngày...")
        reconcile_rows = []
        for day in selected_days:
            # Tìm file raw tương ứng
            raw_candidates = list(raw_path.glob(f"*{day}*.bz2"))
            if not raw_candidates:
                print(f"    ⚠️ Không tìm thấy file raw cho ngày '{day}' trong {raw_path}")
                continue
            raw_file = raw_candidates[0]
            print(f"    Đang quét đối soát '{raw_file.name}'...", end="", flush=True)
            rec_res = reconcile_raw_vs_interim(raw_file, interim_path, target_event_ids)
            reconcile_results.append(rec_res)

            status_str = "✅ KHỚP" if rec_res["is_reconciled"] else "❌ LỆCH"
            print(f" Xong! [{status_str}]")

            for eid in sorted(target_event_ids):
                reconcile_rows.append([
                    day,
                    eid,
                    f"{rec_res['raw_counts'][eid]:,}",
                    f"{rec_res['parquet_counts'][eid]:,}",
                    f"{rec_res['deltas'][eid]:+,}",
                    "✅ 100%" if rec_res["deltas"][eid] == 0 else "❌ LỆCH",
                ])

        if reconcile_rows:
            print("\nBẢNG ĐỐI SOÁT CHI TIẾT:")
            print_table(
                headers=["Ngày", "EventID", "Dòng Raw (.bz2)", "Dòng Parquet", "Chênh lệch (Δ)", "Trạng thái"],
                rows=reconcile_rows,
            )

    # BƯỚC 4: Xuất báo cáo nếu được yêu cầu
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_content = generate_markdown_report(
            integrity_res=integrity_res,
            quality_results=quality_results,
            reconcile_results=reconcile_results,
        )
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        print(f"\n--> [BÁO CÁO] Đã xuất báo cáo chi tiết thành công tại: {report_path}")

    print("\n" + "=" * 80)
    print("HOÀN TẤT KIỂM ĐỊNH!")
    print("=" * 80)


if __name__ == "__main__":
    main()
