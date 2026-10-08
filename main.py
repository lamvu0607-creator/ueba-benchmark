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
        processed_out = processed_dir
        if args.events_dir:
            # Log đã tiêm -> ma trận ghi vào thư mục của run, KHÔNG đè ma trận gốc trong data/.
            from src.injection.layout import RunLayout

            run_layout = RunLayout.from_events_dir(args.events_dir)
            raw_feature_out, processed_out = run_layout.features_raw_dir, run_layout.processed_dir
            logger.info("--> [Stage 2: Features] Dùng log đè '%s' -> ghi vào run '%s'.", args.events_dir, run_layout.root)
        df_raw = build_account_day_matrix(
            start_day=args.start_day,
            end_day=args.end_day,
            interim_dir=interim_dir,
            output_dir=raw_feature_out,
            config=sys_cfg,
            events_dir=args.events_dir,
        )

        if df_raw.height > 0:
            # Tiền xử lý / chuẩn hóa sang Tier 4 (Processed)
            processed_file = processed_out / "feature_matrix_processed.parquet"
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
                    logger.info(
                        "--> [Stage 2: Features] Xác thực Schema thành công: %d đặc trưng cốt lõi hợp lệ "
                        "(%d dòng × %d cột).",
                        len(schema.core_features),
                        val_res["row_count"],
                        val_res["column_count"],
                    )
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

    # Stage 3: Model Benchmark & Leaderboard (label-free, chia theo THỜI GIAN)
    if args.stage in ["all", "benchmark"]:
        from src.models.benchmark import run_model_benchmark
        from src.models.registry import DEFAULT_MODEL_NAMES

        evaluation_cfg = sys_cfg.get("evaluation", {}) or {}
        system_cfg = sys_cfg.get("system", {}) or {}

        selected_models = args.models or list(DEFAULT_MODEL_NAMES)
        split_day = args.split_day if args.split_day is not None else evaluation_cfg.get("split_day", 42)
        split_ratio = (
            args.split_ratio if args.split_ratio is not None else evaluation_cfg.get("test_split_ratio", 0.30)
        )
        seed = args.seed if args.seed is not None else system_cfg.get("random_seed", 42)
        k = args.k if args.k is not None else evaluation_cfg.get("k_metric", 20)
        budget_ratio = (
            args.budget_ratio if args.budget_ratio is not None else evaluation_cfg.get("budget_ratio", 0.05)
        )
        stability_seeds = evaluation_cfg.get("seeds") or [seed]
        labels_path = args.labels if args.labels is not None else evaluation_cfg.get("labels_path")

        logger.info(
            "--> [Stage 3: Benchmark] %d mô hình | train = day <= %s | seed=%s | K=%s | ngân sách=%.1f%%",
            len(selected_models),
            split_day,
            seed,
            k,
            float(budget_ratio) * 100,
        )

        processed_matrix = processed_dir / "feature_matrix_processed.parquet"
        if not processed_matrix.is_file():
            # Hard-fail thay vì cảnh báo rồi thoát: chạy benchmark trên dữ liệu không tồn tại là lỗi.
            raise FileNotFoundError(
                f"Chưa có ma trận đặc trưng '{processed_matrix}'. Hãy chạy trước: "
                "python main.py --stage features"
            )

        result = run_model_benchmark(
            data_path=processed_matrix,
            model_names=selected_models,
            params_path=args.model_params,
            output_models_dir=paths.get("models_dir", "experiments/models"),
            output_results_dir=paths.get("results_dir", "experiments/results"),
            split_day=None if split_day is not None and int(split_day) < 0 else split_day,
            test_split_ratio=float(split_ratio),
            seed=int(seed),
            k=int(k),
            budget_ratio=float(budget_ratio),
            contamination=evaluation_cfg.get("default_contamination", 0.05),
            stability_seeds=list(stability_seeds),
            experiment_log_path=paths.get("experiment_log", "experiments/logs/experiment_log.csv"),
            system_config_path=args.config,
            use_segments=not args.no_segments,
            labels_path=labels_path,
            precision_ks=evaluation_cfg.get("precision_ks") or [10, 50, 100],
            daily_budgets=evaluation_cfg.get("daily_budgets") or [10, 50, 100],
        )

        logger.info("Số mô hình đã chạy: %d; chia tập: %s", len(result["summary"]), result["split"])
        logger.info("Artifact đã ghi:")
        for name, path in result["artifacts"].items():
            logger.info("  - %s: %s", name, path)
        logger.info("--> [Stage 3: Benchmark] Hoàn tất huấn luyện, chấm điểm và xuất bảng xếp hạng.")

    # Stage baselines (chỉ chạy khi gọi tường minh): random / z-score robust / luật ECDF, ngưỡng từ train.
    if args.stage == "baselines":
        from src.baselines.runner import run_baselines

        evaluation_cfg = sys_cfg.get("evaluation", {}) or {}
        result = run_baselines(
            system_config_path=args.config,
            baselines_config_path=args.baselines_config,
            params_path=args.model_params,
            data_path=processed_dir / "feature_matrix_processed.parquet",
            labels_path=args.labels if args.labels is not None else evaluation_cfg.get("labels_path"),
            injected_events_dir=args.events_dir,
            use_segments=False if args.no_segments else None,
        )
        logger.info("--> [Stage Baselines] Điểm: %s | ước lượng L: %s", result["scores_path"], result["lockout"])


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
        choices=["all", "clean", "features", "benchmark", "baselines"],
        default="all",
        help="Pipeline stage to execute (all, clean, features, benchmark; 'baselines' runs only when named)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="List of anomaly detection models to benchmark (default: all 5 models of the registry)",
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
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for the benchmark run (default: system.random_seed in system_config.yaml)",
    )
    parser.add_argument(
        "--split-day",
        type=int,
        default=None,
        help="train = day <= SPLIT_DAY; -1 to derive it from --split-ratio instead",
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=None,
        help="Test ratio used when --split-day is -1 (default: evaluation.test_split_ratio)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="K for Precision@K / top-K overlap (default: evaluation.k_metric)",
    )
    parser.add_argument(
        "--budget-ratio",
        type=float,
        default=None,
        help="Alert budget ratio for budget-based metrics (default: evaluation.budget_ratio)",
    )

    parser.add_argument(
        "--no-segments",
        action="store_true",
        help="Ignore the 'segments' block of model_params.yaml and fit one model on all rows",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="Label file (DomainName, UserName, day[, label, scenario]) enabling the section 5.4 metrics "
        "(default: evaluation.labels_path)",
    )

    parser.add_argument(
        "--baselines-config",
        type=str,
        default="configs/baselines.yaml",
        help="Baseline configuration YAML (stage 'baselines')",
    )
    parser.add_argument(
        "--events-dir",
        type=str,
        default=None,
        help="Overlay events directory (events_injected/ of a run): its days replace the original logs in "
        "stages 'features' (output goes to the run directory) and 'baselines'",
    )

    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()

