"""CLI Script thực thi chuyển đổi dữ liệu thô bz2 sang Parquet cho Event 4624 & 4625.

Cách sử dụng:
    python scripts/convert_raw_to_interim.py
    python scripts/convert_raw_to_interim.py --max-files 1   # Test chạy 1 ngày
    python scripts/convert_raw_to_interim.py --overwrite     # Ghi đè file cũ
"""

import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# Thêm thư mục gốc vào PYTHONPATH để import src không bị lỗi
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.raw_to_interim import convert_all_raw_to_interim


def main():
    parser = argparse.ArgumentParser(
        description="Chuyển đổi dữ liệu raw bz2 sang interim Parquet (Event 4624 & 4625)"
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw",
        help="Thư mục chứa các file .bz2 thô (mặc định: data/raw)",
    )
    parser.add_argument(
        "--interim-dir",
        type=str,
        default="data/interim",
        help="Thư mục lưu file parquet kết quả (mặc định: data/interim)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50000,
        help="Kích thước batch ghi đệm Parquet để kiểm soát RAM (mặc định: 50,000)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Số lượng file tối đa cần xử lý (hữu ích khi test thử nghiệm)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ghi đè nếu file parquet đã tồn tại",
    )
    parser.add_argument(
        "--step-pct",
        type=int,
        default=10,
        help="Khoảng cách %% in log tiến độ (mặc định: 10%%)",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=4,
        help="Số tiến trình CPU xử lý song song (mặc định: 4)",
    )

    args = parser.parse_args()

    convert_all_raw_to_interim(
        raw_dir=args.raw_dir,
        interim_dir=args.interim_dir,
        target_event_ids={4624, 4625},
        batch_size=args.batch_size,
        overwrite=args.overwrite,
        progress_step_pct=args.step_pct,
        max_files=args.max_files,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
