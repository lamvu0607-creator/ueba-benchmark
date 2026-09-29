"""
Module Model Benchmark - Huấn luyện và chấm điểm dị biệt các mô hình UEBA Baseline.
Mô hình hỗ trợ:
1. Isolation Forest (Tree-based)
2. Local Outlier Factor (Density-based)
3. One-Class SVM (Kernel Support Vector Machine)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import joblib
import numpy as np
import pandas as pd
import polars as pl
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
import yaml

logger = logging.getLogger("ueba_benchmark.models.benchmark")

DEFAULT_MODEL_PARAMS: Dict[str, Any] = {
    "isolation_forest": {
        "n_estimators": 150,
        "max_samples": "auto",
        "contamination": 0.05,
        "random_state": 42,
        "n_jobs": -1,
    },
    "local_outlier_factor": {
        "n_neighbors": 20,
        "contamination": 0.05,
        "novelty": True,
        "n_jobs": -1,
    },
    "one_class_svm": {
        "kernel": "rbf",
        "gamma": "scale",
        "nu": 0.05,
    },
}

NON_FEATURE_COLS = ["DomainName", "UserName", "day", "entity_type", "total_logons", "user_key"]


def load_model_params(params_path: Optional[Path | str] = None) -> Dict[str, Any]:
    """Tải tham số mô hình từ file cấu hình YAML."""
    if not params_path:
        return DEFAULT_MODEL_PARAMS
    p = Path(params_path)
    if not p.is_file():
        logger.warning(f"Không tìm thấy file cấu hình mô hình '{p}', sử dụng tham số mặc định.")
        return DEFAULT_MODEL_PARAMS
    with open(p, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f) or {}
    merged = dict(DEFAULT_MODEL_PARAMS)
    merged.update(params)
    return merged


def train_and_evaluate_model(
    model_name: str,
    params: Dict[str, Any],
    X_train: np.ndarray,
    df_eval: pl.DataFrame,
    feature_names: List[str],
) -> Dict[str, Any]:
    """
    Huấn luyện và tính điểm dị biệt cho 1 mô hình cụ thể.
    """
    logger.info(f"--> Bắt đầu huấn luyện mô hình: '{model_name.upper()}' với {len(feature_names)} đặc trưng...")
    t0 = time.time()

    if model_name == "isolation_forest":
        model = IsolationForest(**params)
    elif model_name == "local_outlier_factor":
        model = LocalOutlierFactor(**params)
    elif model_name == "one_class_svm":
        model = OneClassSVM(**params)
    else:
        raise ValueError(f"Mô hình không được hỗ trợ: '{model_name}'.")

    # Huấn luyện
    model.fit(X_train)
    fit_duration = time.time() - t0

    # Dự đoán (-1: Dị biệt, 1: Bình thường)
    preds = model.predict(X_train)
    is_anomaly = (preds == -1).astype(int)

    # Điểm dị biệt (Anomaly Score: càng cao càng bất thường)
    # scikit-learn decision_function càng thấp càng bất thường -> đổi dấu -
    raw_scores = model.decision_function(X_train)
    anomaly_scores = -raw_scores

    n_samples = len(X_train)
    n_anomalies = int(is_anomaly.sum())
    anomaly_rate = (n_anomalies / n_samples) * 100

    logger.info(
        f"[✓] {model_name.upper()} hoàn tất trong {fit_duration:.2f}s | "
        f"Phát hiện: {n_anomalies:,} / {n_samples:,} thực thể dị biệt ({anomaly_rate:.2f}%)"
    )

    return {
        "model_name": model_name,
        "model": model,
        "fit_duration": fit_duration,
        "n_samples": n_samples,
        "n_anomalies": n_anomalies,
        "anomaly_rate": anomaly_rate,
        "preds": preds,
        "is_anomaly": is_anomaly,
        "anomaly_scores": anomaly_scores,
    }


def run_model_benchmark(
    data_path: Path | str,
    output_models_dir: Path | str = "experiments/models",
    output_results_dir: Path | str = "experiments/results",
    model_names: Optional[List[str]] = None,
    params_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """
    Thực thi benchmark các mô hình phát hiện dị biệt.
    """
    logger.info("=== BẮT ĐẦU CHẶNG 3: BENCHMARK CÁC MÔ HÌNH DỊ BIỆT ===")
    d_path = Path(data_path)
    if not d_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file ma trận đặc trưng tại '{d_path}'.")

    models_dir = Path(output_models_dir)
    results_dir = Path(output_results_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    target_models = model_names or ["isolation_forest", "local_outlier_factor", "one_class_svm"]
    all_params = load_model_params(params_path)

    # Đọc dữ liệu ma trận đặc trưng
    df = pl.read_parquet(d_path)
    logger.info(f"Đã nạp ma trận đặc trưng: {df.height:,} dòng × {df.width} cột từ '{d_path}'.")

    # Xác định các cột đặc trưng số
    feature_cols = [
        col for col in df.columns
        if col not in NON_FEATURE_COLS and df[col].dtype in [
            pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.UInt32, pl.UInt8
        ]
    ]
    logger.info(f"Số lượng đặc trưng đưa vào huấn luyện ({len(feature_cols)}): {feature_cols}")

    # Chuyển sang NumPy array và fill null an toàn
    X = df.select(feature_cols).fill_null(0.0).to_numpy()

    # Bảng kết quả tổng hợp
    summary_rows = []
    df_scores = df.select([c for c in ["DomainName", "UserName", "day", "entity_type"] if c in df.columns])

    for m_name in target_models:
        m_params = all_params.get(m_name, {})
        res = train_and_evaluate_model(
            model_name=m_name,
            params=m_params,
            X_train=X,
            df_eval=df,
            feature_names=feature_cols,
        )

        # Lưu file mô hình đã huấn luyện (.joblib)
        model_file = models_dir / f"{m_name}.joblib"
        joblib.dump(res["model"], model_file)
        logger.info(f"Đã lưu mô hình tại '{model_file}'.")

        # Thêm kết quả vào summary
        summary_rows.append({
            "model": m_name,
            "samples": res["n_samples"],
            "anomalies": res["n_anomalies"],
            "anomaly_rate_pct": round(res["anomaly_rate"], 2),
            "train_time_sec": round(res["fit_duration"], 2),
            "model_path": str(model_file),
        })

        # Thêm cột điểm số vào DataFrame kết quả
        df_scores = df_scores.with_columns([
            pl.Series(name=f"{m_name}_anomaly", values=res["is_anomaly"]),
            pl.Series(name=f"{m_name}_score", values=res["anomaly_scores"].round(4)),
        ])

    # Xuất bảng tổng hợp kết quả Benchmark
    df_summary = pd.DataFrame(summary_rows)
    summary_csv = results_dir / "benchmark_summary.csv"
    df_summary.to_csv(summary_csv, index=False, encoding="utf-8")
    logger.info(f"[✓] Đã xuất bảng tổng kết Leaderboard tại '{summary_csv}'.")

    # Xuất chi tiết điểm dị biệt của các tài khoản
    scores_file = results_dir / "anomaly_scores.parquet"
    df_scores.write_parquet(scores_file, compression="snappy")
    logger.info(f"[✓] Đã xuất chi tiết điểm dị biệt của {df_scores.height:,} tài khoản tại '{scores_file}'.")

    logger.info("=== HOÀN TẤT CHẶNG 3: BENCHMARK MÔ HÌNH ===")
    return {
        "summary": df_summary,
        "scores_path": str(scores_file),
        "summary_path": str(summary_csv),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
    parser = argparse.ArgumentParser(description="Chương trình Benchmark các mô hình phát hiện dị biệt UEBA")
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/feature_matrix_processed.parquet",
        help="Đường dẫn đến file ma trận đặc trưng",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["isolation_forest", "local_outlier_factor", "one_class_svm"],
        help="Danh sách các mô hình cần chạy",
    )
    parser.add_argument(
        "--model-params",
        type=str,
        default="configs/model_params.yaml",
        help="Đường dẫn đến file tham số mô hình YAML",
    )
    args = parser.parse_args()

    run_model_benchmark(
        data_path=args.data,
        model_names=args.models,
        params_path=args.model_params,
    )
