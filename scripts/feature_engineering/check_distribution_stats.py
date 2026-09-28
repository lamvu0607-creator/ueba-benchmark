"""
check_distribution_stats.py

Kiểm tra thang phân vị, thống kê mô tả (Mean, Std, Min, Max) và độ lệch phân phối 
(Skewness, Kurtosis) cho 4 đặc trưng đếm và thời gian trọng yếu:
1. distinct_sources_count
2. total_logons
3. interarrival_dt_mean
4. delta_t_cv

Đầu vào:
  - data/features/account_day_matrix.parquet

Đầu ra:
  - docs/feature_engineering/tables/distribution_stats_targeted.csv
  - artifacts/distribution_stats_targeted.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List

# Đảm bảo in tiếng Việt chuẩn UTF-8 trên Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats

# Cấu hình đường dẫn thư mục
BASE_DIR = Path(__file__).resolve().parent.parent.parent
REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = REPO_ROOT / "artifacts"
OUTPUT_TAB_DIR = BASE_DIR / "docs" / "feature_engineering" / "tables"

OUTPUT_TAB_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_FEATURES = [
    "distinct_sources_count",
    "total_logons",
    "interarrival_dt_mean",
    "delta_t_cv",
]


def profile_feature_distribution(name: str, s_raw: pd.Series) -> Dict[str, Any]:
    """Tính toán chi tiết các chỉ số thống kê mô tả và độ lệch của 1 đặc trưng."""
    total_records = len(s_raw)
    null_count = int(s_raw.isna().sum())
    null_pct = (null_count / total_records) * 100 if total_records > 0 else 0.0

    # Lọc bỏ NaN để tính toán thống kê (với interarrival_dt_mean và delta_t_cv)
    s_valid = s_raw.dropna().astype(float)
    valid_count = len(s_valid)

    if valid_count == 0:
        return {
            "Feature": name,
            "Valid_N": 0,
            "Null_%": round(null_pct, 2),
            "Mean": np.nan,
            "Std": np.nan,
            "Min": np.nan,
            "P25": np.nan,
            "P50_Median": np.nan,
            "P75": np.nan,
            "P90": np.nan,
            "P95": np.nan,
            "P99": np.nan,
            "Max": np.nan,
            "Skewness": np.nan,
            "Kurtosis": np.nan,
            "Skew_log1p": np.nan,
            "Reduction_%": 0.0,
            "Shape_Assessment": "Toàn bộ là NaN",
        }

    # 1. Thống kê mô tả trung tâm và phân tán
    mean_val = float(s_valid.mean())
    std_val = float(s_valid.std())

    # 2. Thang phân vị chuẩn
    min_val = float(s_valid.min())
    p25 = float(np.percentile(s_valid, 25))
    median_val = float(np.percentile(s_valid, 50))
    p75 = float(np.percentile(s_valid, 75))
    p90 = float(np.percentile(s_valid, 90))
    p95 = float(np.percentile(s_valid, 95))
    p99 = float(np.percentile(s_valid, 99))
    max_val = float(s_valid.max())

    # 3. Chỉ số hình thái phân phối
    skew_val = float(stats.skew(s_valid, bias=False))
    kurt_val = float(stats.kurtosis(s_valid, bias=False))

    # 4. Đánh giá thử nghiệm biến đổi log(1 + x) nếu miền giá trị không âm
    skew_log1p = np.nan
    reduction_pct = 0.0
    if min_val >= 0:
        s_log = np.log1p(s_valid)
        skew_log1p = float(stats.skew(s_log, bias=False))
        if abs(skew_val) > 1e-4:
            reduction_pct = ((abs(skew_val) - abs(skew_log1p)) / abs(skew_val)) * 100

    # 5. Nhận định hình thái nghiệp vụ
    if abs(skew_val) >= 10.0:
        shape_assessment = "Lệch dương cực đoan (Heavy-tailed / Extreme Outliers)"
    elif abs(skew_val) >= 3.0:
        shape_assessment = "Lệch phải mạnh (Đuôi dài, cần nén log/RobustScaler)"
    elif abs(skew_val) >= 1.0:
        shape_assessment = "Lệch vừa phải"
    else:
        shape_assessment = "Gần đối xứng / Chuẩn"

    return {
        "Feature": name,
        "Valid_N": valid_count,
        "Null_%": round(null_pct, 2),
        "Mean": round(mean_val, 2),
        "Std": round(std_val, 2),
        "Min": round(min_val, 2),
        "P25": round(p25, 2),
        "P50_Median": round(median_val, 2),
        "P75": round(p75, 2),
        "P90": round(p90, 2),
        "P95": round(p95, 2),
        "P99": round(p99, 2),
        "Max": round(max_val, 2),
        "Skewness": round(skew_val, 2),
        "Kurtosis": round(kurt_val, 2),
        "Skew_log1p": round(skew_log1p, 2) if not np.isnan(skew_log1p) else np.nan,
        "Reduction_%": round(reduction_pct, 1) if not np.isnan(reduction_pct) else 0.0,
        "Shape_Assessment": shape_assessment,
    }


def main():
    parser = argparse.ArgumentParser(description="Kiểm tra phân vị và độ lệch phân phối cho 4 đặc trưng trọng yếu")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=BASE_DIR / "data" / "features" / "raw" / "feature_matrix_raw.parquet",
        help="Đường dẫn file ma trận đặc trưng THÔ (feature_matrix_raw.parquet)",
    )
    args = parser.parse_args()

    if not args.data_path.exists():
        print(f"[ERROR] Không tìm thấy file dữ liệu tại: {args.data_path.resolve()}")
        sys.exit(1)

    print(f"[INFO] Nạp dữ liệu từ: {args.data_path.resolve()}")
    df_raw = pl.read_parquet(args.data_path).to_pandas()
    print(f"[INFO] Tổng số dòng: {len(df_raw):,} bản ghi (Account × Day)\n")

    # Kiểm tra cột có trong dữ liệu
    available_cols = [c for c in TARGET_FEATURES if c in df_raw.columns]
    missing_cols = set(TARGET_FEATURES) - set(available_cols)
    if missing_cols:
        print(f"[WARN] Các cột không có trong bảng đặc trưng: {missing_cols}")

    if not available_cols:
        print("[ERROR] Không có cột nào trong 4 đặc trưng mục tiêu tồn tại trong file dữ liệu.")
        return

    records: List[Dict[str, Any]] = []
    for col in available_cols:
        res = profile_feature_distribution(col, df_raw[col])
        records.append(res)

    df_summary = pd.DataFrame(records)

    # In Bảng 1: Thang Phân Vị & Cực Trị (Min, Mean, Median, P90, P99, Max)
    print("=" * 115)
    print("BẢNG 1: THỐNG KÊ MÔ TẢ & THANG PHÂN VỊ (PERCENTILE LADDER)")
    print("=" * 115)
    cols_percentiles = ["Feature", "Valid_N", "Null_%", "Min", "P25", "P50_Median", "P75", "P90", "P95", "P99", "Max"]
    print(df_summary[cols_percentiles].to_string(index=False))
    print("-" * 115)

    # In Bảng 2: Chỉ số Độ Lệch & Đánh giá Hiệu Quả Log1p
    print("\n" + "=" * 115)
    print("BẢNG 2: ĐO LƯỜNG ĐỘ LỆCH (SKEWNESS / KURTOSIS) & KHẢ NĂNG NÉN LOG")
    print("=" * 115)
    cols_shape = ["Feature", "Mean", "Std", "Skewness", "Kurtosis", "Skew_log1p", "Reduction_%", "Shape_Assessment"]
    print(df_summary[cols_shape].to_string(index=False))
    print("=" * 115 + "\n")

    # In Nhận xét chi tiết cho từng biến phục vụ thuyết trình/báo cáo
    print(">>> TÓM TẮT ĐÁNH GIÁ CHẤT LƯỢNG:")
    for _, row in df_summary.iterrows():
        feat = row["Feature"]
        mean_p50_gap = row["Mean"] / row["P50_Median"] if row["P50_Median"] > 0 else 0
        p99_max_gap = row["Max"] / row["P99"] if row["P99"] > 0 else 0
        print(f"• [{feat}]:")
        print(f"   - Tỷ lệ NaN: {row['Null_%']}% | Trung bình: {row['Mean']:,} vs Median: {row['P50_Median']:,} (Gấp {mean_p50_gap:.1f} lần)")
        print(f"   - P99: {row['P99']:,} vs Cực đại (Max): {row['Max']:,} (Max gấp {p99_max_gap:.1f} lần P99)")
        print(f"   - Skewness thô: {row['Skewness']} -> Sau log(1+x): {row['Skew_log1p']} (Giảm lệch: {row['Reduction_%']}%)")
        print(f"   - Đánh giá: {row['Shape_Assessment']}\n")

    # Lưu kết quả ra CSV
    out_csv = OUTPUT_TAB_DIR / "distribution_stats_targeted.csv"
    artifact_csv = ARTIFACT_DIR / "distribution_stats_targeted.csv"
    df_summary.to_csv(out_csv, index=False, float_format="%.2f")
    df_summary.to_csv(artifact_csv, index=False, float_format="%.2f")

    print(f"[✓] Đã lưu bảng thống kê tại: {out_csv.resolve()}")
    print(f"[✓] Đã lưu bản sao lưu tại: {artifact_csv.resolve()}")


if __name__ == "__main__":
    main()