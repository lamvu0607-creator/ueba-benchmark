"""
Chẩn đoán: thống kê từng đặc trưng core trên **tập train** (day <= split_day).

Chạy: ``python scripts/diagnostics/feature_variance_check.py [split_day]``

Mục đích:
  * phát hiện **đặc trưng hằng số / gần hằng số** trên tập train — chúng không mang tín hiệu cho
    mô hình dựa trên khoảng cách và làm Z-score baseline chỉ còn phụ thuộc vài đặc trưng;
  * đếm NULL hợp lệ theo schema (imputer phải dùng median, KHÔNG được fill 0);
  * in khoảng giá trị để phát hiện giá trị vô lý (ví dụ tỷ lệ ngoài [0, 1]).

Warnings của ``SimpleImputer``/``RobustScaler`` về "đặc trưng hằng số" cũng xuất phát từ bảng này.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.features.schema import FeatureSchema  # noqa: E402

DATA_PATH = Path("data/processed/feature_matrix_processed.parquet")
DEFAULT_SPLIT_DAY = 42

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # pragma: no cover - chỉ ảnh hưởng hiển thị console
        pass


def main() -> None:
    split_day = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SPLIT_DAY
    if not DATA_PATH.is_file():
        raise SystemExit(f"Không tìm thấy '{DATA_PATH}'. Hãy chạy stage features trước.")

    features = list(FeatureSchema().core_features)
    # Đọc cột day ở dạng Int32 để so sánh (to_numpy của UInt8 sẽ thành uint8, so sánh vẫn đúng).
    train = pl.read_parquet(DATA_PATH).filter(pl.col("day") <= split_day)
    print(f"Tập train: {train.height:,} dòng (day <= {split_day}), {len(features)} đặc trưng core")
    print()
    print(f"{'feature':<30} {'nulls':>9} {'n_unique':>9} {'min':>12} {'max':>12} {'std':>12}  ghi chú")

    constant = []
    for name in features:
        column = train[name].cast(pl.Float64, strict=False)
        n_null = int(column.null_count())
        filled = column.drop_nulls()
        n_unique = int(filled.n_unique())
        if n_unique <= 1:
            constant.append(name)
        note = "HẰNG SỐ" if n_unique <= 1 else ("gần hằng số" if n_unique <= 3 else "")
        print(
            f"{name:<30} {n_null:>9,} {n_unique:>9,} "
            f"{float(filled.min()):>12.4f} {float(filled.max()):>12.4f} "
            f"{float(filled.std()):>12.4f}  {note}"
        )

    print()
    print(f"Đặc trưng hằng số trên tập train ({len(constant)}): {constant}")
    print("Lưu ý: Z-score baseline sẽ đặt scale = 1 cho các đặc trưng này (không đóng góp điểm).")


if __name__ == "__main__":
    main()
