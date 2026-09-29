"""
UEBA Benchmark Orchestrator - Main Entrypoint
File điều phối thực thi pipeline từ A -> Z.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List
import yaml

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
)
logger = logging.getLogger("ueba_benchmark")


def load_yaml(path: Path | str) -> Dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        logger.warning(f"Không tìm thấy file cấu hình YAML: '{p}'.")
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def run_pipeline(args):
    # 1. Load cấu hình hệ thống
    sys_cfg = load_yaml(args.config)
    paths = sys_cfg.get("paths", {})

    raw_dir = Path(paths.get("raw_data_dir", "data/raw"))
    interim_dir = Path(paths.get("interim_data_dir", "data/interim"))
    features_dir = Path(paths.get("features_data_dir", "data/features"))
    processed_dir = Path(paths.get("processed_data_dir", "data/processed"))

    logger.info("=== UEBA BENCHMARK PIPELINE ===")
    logger.info(f"Cấu hình hệ thống: {args.config}")
    logger.info(f"Giai đoạn thực thi: {args.stage}")

    # Stage 1: Clean & Ingest (raw/interim -> cleaned)
    if args.stage in ["all", "clean"]:
        cleaned_dir = Path(paths.get("cleaned_data_dir", "data/cleaned"))
        end_day_clean = args.end_day if args.end_day is not None else 3
        logger.info(f"--> [Stage 1: Clean] Bắt đầu làm sạch Windows Event Logs (Day {args.start_day:02d} -> Day {end_day_clean:02d})...")
        from src.data.cleaner import clean_dataset
        clean_dataset(
            start_day=args.start_day,
            end_day=end_day_clean,
            interim_dir=interim_dir,
            output_dir=cleaned_dir,
            adjust_dst=True,
        )
        logger.info("--> [Stage 1: Clean] Hoàn tất làm sạch và chuẩn hóa log.")

    if args.stage == "clean":
        return

    # Stage 2: Feature Engineering (cleaned/interim -> features -> processed)
    if args.stage in ["all", "features"]:
        logger.info("--> [Stage 2: Features] Trích xuất ma trận đặc trưng hành vi (Tài khoản × Ngày)...")
        from src.features import (
            build_account_day_matrix,
            prepare_processed_dataset,
            FeatureSchema,
        )

        raw_feature_out = features_dir / "raw"
        df_raw = build_account_day_matrix(
            start_day=args.start_day,
            end_day=args.end_day,
            interim_dir=interim_dir,
            output_dir=raw_feature_out,
            config=sys_cfg,
        )

        if df_raw.height > 0:
            # Tiền xử lý / chuẩn hóa sang Tier 4 (Processed)
            processed_file = processed_dir / "feature_matrix_processed.parquet"
            raw_matrix_file = raw_feature_out / "feature_matrix_raw.parquet"
            df_processed = prepare_processed_dataset(
                raw_feature_path=raw_matrix_file,
                output_path=processed_file,
            )

            # Kiểm tra hợp đồng Schema
            try:
                schema = FeatureSchema()
                val_res = schema.validate_features(df_processed)
                if val_res["is_valid"]:
                    logger.info("--> [Stage 2: Features] Xác thực Schema thành công: 16 đặc trưng cốt lõi hợp lệ.")
                else:
                    logger.warning(f"--> [Stage 2: Features] Cảnh báo kiểm định Schema: {val_res}")
            except Exception as e:
                logger.warning(f"Không thể kiểm tra schema: {e}")

            logger.info("--> [Stage 2: Features] Hoàn tất trích xuất và chuẩn hóa ma trận đặc trưng.")
        else:
            logger.warning("Không có dữ liệu đặc trưng nào được tạo. Vui lòng kiểm tra lại dữ liệu.")
            if args.stage == "features":
                return

    if args.stage == "features":
        return

    # Stage 3: Model Benchmark & Leaderboard
    if args.stage in ["all", "benchmark"]:
        logger.info("--> [Stage 3: Benchmark] Bắt đầu đánh giá các mô hình UEBA...")
        selected_models = args.models or ["isolation_forest", "local_outlier_factor", "one_class_svm"]
        logger.info(f"Mô hình được chọn: {selected_models}")

        processed_matrix = processed_dir / "feature_matrix_processed.parquet"
        if not processed_matrix.exists():
            logger.warning(
                f"Chưa tìm thấy ma trận đặc trưng tại '{processed_matrix}'. "
                "Vui lòng chạy: python main.py --stage features trước."
            )
            return

        from src.models.benchmark import run_model_benchmark
        run_model_benchmark(
            data_path=processed_matrix,
            output_models_dir=paths.get("models_dir", "experiments/models"),
            output_results_dir=paths.get("results_dir", "experiments/results"),
            model_names=selected_models,
            params_path=args.model_params,
        )
        logger.info("--> [Stage 3: Benchmark] Hoàn tất huấn luyện và xuất bảng xếp hạng mô hình.")


def main():
    parser = argparse.ArgumentParser(description="UEBA Anomaly Detection Benchmark Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/system_config.yaml",
        help="Path to system configuration YAML file",
    )
    parser.add_argument(
        "--model-params",
        type=str,
        default="configs/model_params.yaml",
        help="Path to model parameters YAML file",
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["all", "clean", "features", "benchmark"],
        default="all",
        help="Pipeline stage to execute (all, clean, features, benchmark)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["isolation_forest", "local_outlier_factor", "one_class_svm"],
        help="List of anomaly detection models to benchmark",
    )
    parser.add_argument(
        "--start-day",
        type=int,
        default=1,
        help="Starting day for feature extraction (default: 1)",
    )
    parser.add_argument(
        "--end-day",
        type=int,
        default=None,
        help="Ending day for feature extraction (default: last available day in interim)",
    )

    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()

