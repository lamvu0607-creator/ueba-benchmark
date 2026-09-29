"""
run_all_feature_plots.py

Script điều phối thực thi toàn bộ pipeline trực quan hóa đặc trưng Tuần 2:
1. plot_feature_correlation.py          (Ma trận tương quan Spearman & Đa cộng tuyến)
2. plot_feature_distribution.py        (Phân phối đặc trưng đếm trước/sau nén log)
3. plot_sanity_check_human_vs_machine.py (Kiểm định nghiệp vụ Người dùng vs Tài khoản Máy)
4. plot_temporal_stability.py           (Tính ổn định chu kỳ: nhịp tuần, cuối tuần & nghỉ sinh học)
5. check_multicollinearity.py           (Soát đa cộng tuyến: Spearman/Pearson + VIF)
6. check_normalization_effect.py        (Kiểm tra hậu chuẩn hóa log1p trên toàn bộ đặc trưng)

Đầu vào:
  - data/features/raw/feature_matrix_raw.parquet   (ma trận THÔ — chưa chuẩn hóa)

Đầu ra:
  - Đồng bộ toàn bộ Figures và Tables vào docs/feature_engineering/ và artifacts/
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time
from typing import Dict, List

# Thiết lập UTF-8 trên Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ==============================================================================
# 1. THIẾT LẬP ĐƯỜNG DẪN DỰ ÁN
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent.parent
REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "features" / "raw" / "feature_matrix_raw.parquet"  # ma trận THÔ (bước 1)

OUTPUT_FIG_DIR = BASE_DIR / "docs" / "feature_engineering" / "figures"
OUTPUT_TAB_DIR = BASE_DIR / "docs" / "feature_engineering" / "tables"
ARTIFACT_DIR = REPO_ROOT / "artifacts"

# Danh sách 4 script trực quan hóa thành phần và mục tiêu kiểm tra kết quả
TASKS_CONFIG = [
    {
        "id": "1_correlation",
        "name": "Spearman Correlation & Multicollinearity",
        "script": "plot_feature_correlation.py",
        "expected_artifacts": [
            OUTPUT_FIG_DIR / "spearman_correlation_matrix.png",
            OUTPUT_TAB_DIR / "correlation_spearman.csv",
            OUTPUT_TAB_DIR / "high_multicollinearity_pairs.csv",
        ],
    },
    {
        "id": "2_distribution",
        "name": "Count Features Distribution (Log Scaling)",
        "script": "plot_feature_distribution.py",
        "expected_artifacts": [
            OUTPUT_TAB_DIR / "distribution_skewness_comparison.csv",
        ],
    },
    {
        "id": "3_sanity_check",
        "name": "Domain Sanity Check (Human vs Machine)",
        "script": "plot_sanity_check_human_vs_machine.py",
        "expected_artifacts": [
            OUTPUT_FIG_DIR / "sanity_check_human_vs_machine.png",
            OUTPUT_FIG_DIR / "sanity_check_axis_temporal_off_hours_ratio.png",
            OUTPUT_FIG_DIR / "sanity_check_axis_logon_mechanism_interactive_ratio.png",
            OUTPUT_FIG_DIR / "sanity_check_axis_spatial_fanout_distinct_hosts.png",
            OUTPUT_FIG_DIR / "sanity_check_human_vs_machine_boxplot_grid.png",
            OUTPUT_TAB_DIR / "sanity_check_human_vs_machine.csv",
            OUTPUT_TAB_DIR / "sanity_check_human_vs_machine_axes.csv",
        ],
    },
    {
        "id": "4_temporal_stability",
        "name": "Temporal Stationarity (Weekly Cycle & Rest Rhythm)",
        "script": "plot_temporal_stability.py",
        "expected_artifacts": [
            OUTPUT_FIG_DIR / "temporal_stability_daily.png",
            OUTPUT_FIG_DIR / "temporal_weekly_cycle.png",
            # 2 artifact dưới đây cần data/interim (event_4624/, event_4625/);
            # nếu chạy với --skip-hourly thì tác vụ này sẽ báo WARNING (thiếu file).
            OUTPUT_FIG_DIR / "temporal_hourly_rhythm.png",
            OUTPUT_TAB_DIR / "daily_activity_summary.csv",
            OUTPUT_TAB_DIR / "weekly_cycle_summary.csv",
            OUTPUT_TAB_DIR / "hourly_rhythm_summary.csv",
            OUTPUT_TAB_DIR / "weekday_vs_weekend_features.csv",
            OUTPUT_TAB_DIR / "temporal_stationarity_metrics.csv",
        ],
    },
    {
        "id": "5_multicollinearity",
        "name": "Multicollinearity Audit (Spearman/Pearson/VIF)",
        "script": "check_multicollinearity.py",
        "expected_artifacts": [
            OUTPUT_TAB_DIR / "multicollinearity_vif.csv",
        ],
    },
    {
        "id": "6_normalization_check",
        "name": "Post-Normalization Audit (log1p effect on all features)",
        "script": "check_normalization_effect.py",
        "expected_artifacts": [
            OUTPUT_TAB_DIR / "normalization_effect_all_features.csv",
            OUTPUT_TAB_DIR / "normalization_alternative_transforms.csv",
            OUTPUT_TAB_DIR / "normalization_shape_zoom_vs_full.csv",
        ],
    },
]


def run_single_script(
    script_path: Path, data_path: Path, extra_args: List[str] | None = None
) -> Tuple[bool, float, str]:
    """Thực thi một script Python độc lập và đo đạc thời gian."""
    cmd = [sys.executable, str(script_path)]

    # Bổ sung tham số --data-path nếu script hỗ trợ argparse
    if "--data-path" not in (extra_args or []):
        cmd.extend(["--data-path", str(data_path)])
    if extra_args:
        cmd.extend(extra_args)

    t0 = time.time()
    try:
        # Chạy trực tiếp và stream output ra màn hình console
        process = subprocess.run(cmd, check=True)
        elapsed = time.time() - t0
        return (process.returncode == 0), elapsed, ""
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - t0
        return False, elapsed, f"Mã lỗi tiến trình: {e.returncode}"
    except Exception as e:
        elapsed = time.time() - t0
        return False, elapsed, str(e)


def main():
    parser = argparse.ArgumentParser(
        description="Chạy tự động toàn bộ 4 script trực quan hóa ma trận đặc trưng UEBA."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DATA_PATH,
        help="Đường dẫn đến file ma trận account_day_matrix.parquet",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Dừng ngay lập tức nếu một script gặp lỗi",
    )
    args = parser.parse_args()

    # Kiểm tra sự tồn tại của file dữ liệu đầu vào
    if not args.data_path.exists():
        print(f"[ERROR] Không tìm thấy dữ liệu ma trận tại: {args.data_path.resolve()}")
        print("Vui lòng chạy 'python main.py --stage features' "
              "(hoặc scripts/feature_engineering/extract_account_day_matrix.py — bản 2.0) trước khi vẽ đồ thị.")
        sys.exit(1)

    print("=" * 85)
    print("KHỞI ĐỘNG PIPELINE TỰ ĐỘNG XUẤT BIỂU ĐỒ & BÁO CÁO ĐẶC TRƯNG UEBA")
    print(f"Dữ liệu đầu vào: {args.data_path.resolve()}")
    print(f"Tổng số tác vụ:  {len(TASKS_CONFIG)} scripts")
    print("=" * 85)

    pipeline_start = time.time()
    task_results: List[Dict] = []

    for idx, task in enumerate(TASKS_CONFIG, start=1):
        script_file = SCRIPTS_DIR / task["script"]
        print(f"\n[{idx}/{len(TASKS_CONFIG)}] BẮT ĐẦU: {task['name']}")
        print(f"      File thực thi: {script_file.name}")
        print("-" * 85)

        if not script_file.exists():
            print(f"[LỖI] Không tìm thấy file script tại: {script_file.resolve()}")
            task_results.append({
                "Task": task["name"],
                "Script": task["script"],
                "Status": "FAILED",
                "Duration": 0.0,
                "Note": "File not found",
            })
            if args.stop_on_error:
                break
            continue

        # Thực thi script
        success, duration, error_msg = run_single_script(script_file, args.data_path)

        # Kiểm tra sự xuất hiện của các file sản phẩm
        missing_artifacts = [
            p.name for p in task["expected_artifacts"] if not p.exists()
        ]

        if success and not missing_artifacts:
            status = "SUCCESS"
            note = "Đầy đủ figures & tables"
        elif success and missing_artifacts:
            status = "WARNING"
            note = f"Thiếu file: {', '.join(missing_artifacts)}"
        else:
            status = "FAILED"
            note = error_msg

        task_results.append({
            "Task": task["name"],
            "Script": task["script"],
            "Status": status,
            "Duration": round(duration, 2),
            "Note": note,
        })

        if status == "FAILED" and args.stop_on_error:
            print(f"[DỪNG KHẨN CẤP] Tác vụ {task['script']} gặp lỗi.")
            break

    total_pipeline_time = time.time() - pipeline_start

    # ==========================================================================
    # BÁO CÁO TỔNG KẾT
    # ==========================================================================
    print("\n" + "=" * 85)
    print("BẢNG TỔNG KẾT KẾT QUẢ THỰC THI TOÀN BỘ SCRIPTS VẼ BIỂU ĐỒ:")
    print("=" * 85)
    header = f"{'STT':<4} | {'Tên tác vụ':<38} | {'Trạng thái':<10} | {'Thời gian':<10} | {'Ghi chú'}"
    print(header)
    print("-" * 85)

    for i, res in enumerate(task_results, start=1):
        status_color = res["Status"]
        print(
            f"{i:<4} | {res['Task']:<38} | {status_color:<10} | {res['Duration']:>7.2f}s | {res['Note']}"
        )

    print("-" * 85)
    print(f"Tổng thời gian hoàn thành toàn bộ pipeline: {total_pipeline_time:.2f} giây")
    print(f"Thư mục lưu trữ biểu đồ: {OUTPUT_FIG_DIR.resolve()}")
    print(f"Thư mục lưu trữ bảng:    {OUTPUT_TAB_DIR.resolve()}")
    print(f"Thư mục artifacts:       {ARTIFACT_DIR.resolve()}")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    main()