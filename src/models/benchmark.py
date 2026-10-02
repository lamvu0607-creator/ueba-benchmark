"""
Module Benchmark - Điều phối benchmark phát hiện dị biệt **label-free** cho 6 mô hình.

Luồng chuẩn (Tuần 3)::

    đọc parquet -> chia theo THỜI GIAN (train = day <= 42) -> với mỗi mô hình:
        16 đặc trưng core -> imputer(median, fit train) -> scaler(robust, fit train) -> fit
        -> chấm điểm tập TEST -> alert rate / ngân sách / Top-K / phân vị điểm / thời gian
    -> benchmark_summary.csv + anomaly_scores.parquet + model_topk_overlap.csv + stability

Ba lỗi của bản benchmark cũ đã được sửa ở đây:

1. **Rò rỉ dữ liệu**: cũ fit và chấm điểm trên cùng một ma trận (1.055.283 dòng) — mọi con số
   đánh giá đều vô nghĩa. Nay ``train`` và ``test`` là hai phân đoạn thời gian tách biệt, và
   ``imputer``/``scaler`` chỉ được fit trên train.
2. **Sai tập đặc trưng**: cũ dùng mọi cột số (18 cột, lẫn biến thể thô trùng lặp). Nay dùng
   đúng 16 đặc trưng core theo ``configs/feature_schema.yaml``.
3. **Ngưỡng cảnh báo mù**: cũ dùng ``predict()`` của sklearn (OCSVM ``nu=0.05`` cho ~8% cảnh báo).
   Nay ``threshold_ = quantile(score_train, 1 - contamination)``; alert rate trên tập test được
   **báo cáo trung thực** (có thể trôi — xem ``scripts/diagnostics/alert_rate_calibration.py``)
   song song với chỉ số theo **ngân sách** (xếp hạng trên chính tập đánh giá) để so sánh công bằng.
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import polars as pl

from src.evaluation.metrics import (
    alert_rate,
    budget_size,
    pairwise_stability,
    score_quantiles,
    timed,
    topk_overlap,
)
from src.evaluation.experiment_log import append_experiment_log, build_log_row
from src.evaluation.manifest import build_manifest, git_state, write_manifest
from src.evaluation.split import SplitInfo, time_split
from src.models.base import SKLEARN_VERSION
from src.models.registry import (
    DEFAULT_MODEL_NAMES,
    available_models,
    create_model,
    create_pipeline,
    load_params,
    resolve_model,
)

logger = logging.getLogger("ueba_benchmark.models.benchmark")

DEFAULT_DATA_PATH = "data/processed/feature_matrix_processed.parquet"
DEFAULT_PARAMS_PATH = "configs/model_params.yaml"
DEFAULT_SYSTEM_CONFIG_PATH = "configs/system_config.yaml"
DEFAULT_RESULTS_DIR = "experiments/results"
DEFAULT_MODELS_DIR = "experiments/models"
DEFAULT_EXPERIMENT_LOG_PATH = "experiments/logs/experiment_log.csv"
DEFAULT_STABILITY_SAMPLE_SIZE = 50_000

__all__ = [
    "DEFAULT_MODEL_PARAMS",
    "NON_FEATURE_COLS",
    "SUMMARY_COLUMNS",
    "load_feature_matrix",
    "load_model_params",
    "run_model_benchmark",
    "run_single_model",
    "train_and_evaluate_model",
]

#: Giữ lại bảng tham số mặc định cũ để mã/test cũ vẫn chạy được (registry là nguồn duy nhất mới).
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

#: Cột danh tính/không phải đặc trưng — giữ để tài liệu hoá lý do loại bỏ khỏi ma trận mô hình.
NON_FEATURE_COLS = ["DomainName", "UserName", "day", "entity_type", "total_logons", "user_key"]

#: Các phân vị của điểm ghi vào ``benchmark_summary.csv`` (gồm p25/p95/p99 để đối chiếu với log cũ).
SUMMARY_QUANTILE_PROBS: Tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)

#: Thứ tự cột của ``benchmark_summary.csv`` (một dòng cho mỗi mô hình).
SUMMARY_COLUMNS = [
    "model",
    "is_baseline",
    "n_train_partition",
    "n_fit",
    "n_eval",
    "n_features",
    "imputer",
    "scaler",
    "contamination",
    "threshold",
    "alert_rate_pct",
    "alert_rate_drift_pp",
    "budget_ratio",
    "budget_n",
    "topk_k",
    "fit_seconds",
    "score_seconds",
    "seed",
    "sklearn_version",
]


# --------------------------------------------------------------------------- #
# Nạp dữ liệu & cấu hình
# --------------------------------------------------------------------------- #
def load_feature_matrix(data_path: Path | str = DEFAULT_DATA_PATH) -> pl.DataFrame:
    """Đọc ma trận đặc trưng processed; thiếu file -> lỗi rõ ràng thay vì chạy tiếp với dữ liệu rỗng."""
    path = Path(data_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy ma trận đặc trưng tại '{path}'. Hãy chạy trước: python main.py --stage features"
        )
    df = pl.read_parquet(path)
    if df.height == 0:
        raise ValueError(f"Ma trận '{path}' rỗng.")
    if "day" not in df.columns:
        raise ValueError(f"Ma trận '{path}' thiếu cột 'day' — bắt buộc cho time-based split.")

    logger.info(
        "Đã nạp '%s': %s dòng x %d cột, day %s..%s.",
        path,
        f"{df.height:,}",
        df.width,
        int(df["day"].min()),
        int(df["day"].max()),
    )
    return df


def load_model_params(params_path: Optional[Path | str] = DEFAULT_PARAMS_PATH) -> Dict[str, Any]:
    """Tương thích ngược: đọc tham số mô hình (uỷ quyền cho ``registry.load_params``)."""
    return load_params(params_path)


def train_and_evaluate_model(
    model_name: str,
    params: Optional[Dict[str, Any]] = None,
    X_train: Optional[np.ndarray] = None,
    df_eval: Optional[pl.DataFrame] = None,
    feature_names: Optional[Sequence[str]] = None,
    batch_size: int = 100_000,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    API CŨ — fit và chấm điểm trên **cùng một** ma trận; giữ lại chỉ để tương thích ngược.

    .. deprecated::
       Hàm này tái hiện đúng hành vi rò rỉ dữ liệu của bản benchmark cũ (nên mọi con số của nó
       không dùng được cho báo cáo). Benchmark thật dùng ``run_model_benchmark`` với time-based split.

    Trả về đúng các key mà mã cũ mong đợi: ``model``, ``model_name``, ``n_samples``, ``n_anomalies``,
    ``anomaly_rate``, ``fit_duration``, ``anomaly_scores``, ``is_anomaly``, ``threshold``.
    """
    warnings.warn(
        "train_and_evaluate_model() fit và chấm điểm trên cùng tập dữ liệu (rò rỉ dữ liệu). "
        "Hãy dùng run_model_benchmark() với time-based split cho mọi kết quả báo cáo.",
        DeprecationWarning,
        stacklevel=2,
    )
    if X_train is None:
        raise ValueError("Thiếu 'X_train' (ma trận numpy đã dựng sẵn).")

    model_block = dict(params or DEFAULT_MODEL_PARAMS.get(model_name, {}))
    model = create_model(model_name, params={model_name: model_block}, random_state=random_state)
    matrix = np.asarray(X_train, dtype=np.float64)

    _, fit_duration = timed(model.fit, matrix)
    if feature_names is not None:
        model.feature_names_in_ = list(feature_names)

    anomaly_scores = model.score(matrix)
    is_anomaly = model.predict(matrix).astype(int)
    n_samples = int(matrix.shape[0])

    return {
        "model": model,
        "model_name": model.name,
        "n_samples": n_samples,
        "n_anomalies": int(is_anomaly.sum()),
        "anomaly_rate": float(is_anomaly.sum() / n_samples) if n_samples else 0.0,
        "fit_duration": float(fit_duration),
        "anomaly_scores": anomaly_scores,
        "is_anomaly": is_anomaly,
        "threshold": model.threshold_,
    }


# --------------------------------------------------------------------------- #
# Chạy một mô hình trên time-based split
# --------------------------------------------------------------------------- #
def run_single_model(
    name: str,
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    params: Optional[Dict[str, Any]] = None,
    seed: int = 42,
    k: int = 20,
    budget_ratio: float = 0.05,
    contamination: Optional[float] = None,
    feature_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Fit 1 mô hình trên ``train_df`` rồi chấm điểm ``test_df`` (không bao giờ chấm điểm tập train).

    Trả dict: ``pipeline``, ``scores`` (điểm thô), ``rank_pct`` (percentile trên chính tập test,
    dùng để so sánh liên mô hình), ``labels`` (0/1 theo ngưỡng fit trên train), ``meta`` (metadata
    pipeline) và ``summary`` (một dòng cho ``benchmark_summary.csv``).
    """
    pipeline = create_pipeline(
        name,
        params=params,
        feature_names=feature_names,
        contamination=contamination,
        random_state=seed,
    )

    _, fit_seconds = timed(pipeline.fit, train_df)
    scores, score_seconds = timed(pipeline.score, test_df)
    labels, _ = timed(pipeline.predict, test_df)
    rank_pct = pipeline.score_rank_pct(test_df, reference_scores=scores)

    flags = labels.astype(int)
    metadata = pipeline.get_metadata()
    summary: Dict[str, Any] = {
        "model": name,
        "is_baseline": pipeline.model.is_baseline,
        "n_train_partition": int(train_df.height),
        "n_fit": int(metadata["model"]["n_fit"]),
        "n_eval": int(test_df.height),
        "n_features": int(metadata["n_features"]),
        "imputer": metadata["imputer"],
        "scaler": metadata["scaler"],
        "contamination": float(pipeline.model.contamination),
        "threshold": float(pipeline.model.threshold_),
        "alert_rate_pct": round(alert_rate(flags) * 100, 4),
        # Độ trôi so với ngân sách cảnh báo (điểm phần trăm): dương = cảnh báo nhiều hơn ngân sách.
        # Đây là hiện tượng THẬT khi phân phối tập test khác tập train — được báo cáo, không che.
        "alert_rate_drift_pp": round(alert_rate(flags) * 100 - budget_ratio * 100, 4),
        "budget_ratio": float(budget_ratio),
        "budget_n": budget_size(test_df.height, budget_ratio),
        "topk_k": int(k),
        "fit_seconds": round(fit_seconds, 3),
        "score_seconds": round(score_seconds, 3),
        "seed": int(seed),
        "sklearn_version": SKLEARN_VERSION,
    }
    summary.update(
        {f"score_{key}": value for key, value in score_quantiles(scores, probs=SUMMARY_QUANTILE_PROBS).items()}
    )

    logger.info(
        "[%s] n_fit=%s (tập train %s dòng), n_eval=%s, alert_rate=%.2f%% (ngân sách %.1f%%), "
        "threshold=%.6g, fit=%.1fs, score=%.1fs",
        name,
        f"{summary['n_fit']:,}",
        f"{summary['n_train_partition']:,}",
        f"{summary['n_eval']:,}",
        summary["alert_rate_pct"],
        budget_ratio * 100,
        summary["threshold"],
        fit_seconds,
        score_seconds,
    )

    return {
        "pipeline": pipeline,
        "scores": scores,
        "rank_pct": rank_pct,
        "labels": flags,
        "meta": metadata,
        "summary": summary,
    }


def _stability_scores(
    name: str,
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    params: Optional[Dict[str, Any]],
    seeds: Sequence[int],
    contamination: Optional[float],
    feature_names: Optional[Sequence[str]],
    sample_size: int,
    k: int,
) -> Tuple[Optional[Dict[str, float]], int]:
    """
    Đo độ ổn định đa seed: chạy lại mô hình với từng seed trên **mẫu con của tập test**.

    Dùng mẫu con để giữ thời gian chạy hợp lý (LOF/OCSVM chấm điểm rất nặng); kích thước mẫu được
    ghi lại trong artifact để người đọc biết con số ổn định được đo trên bao nhiêu dòng.
    """
    if len(seeds) < 2:
        return None, 0

    n_rows = test_df.height
    if sample_size >= n_rows:
        sampled = test_df
    else:
        rng = np.random.default_rng(seeds[0])
        indices = np.sort(rng.choice(n_rows, size=sample_size, replace=False))
        sampled = test_df[indices.tolist()]

    score_runs = []
    for seed in seeds:
        pipeline = create_pipeline(
            name, params=params, feature_names=feature_names, contamination=contamination, random_state=int(seed)
        ).fit(train_df)
        score_runs.append(pipeline.score(sampled))
    return pairwise_stability(score_runs, k=min(k, sampled.height)), sampled.height


# --------------------------------------------------------------------------- #
# Điều phối toàn bộ benchmark
# --------------------------------------------------------------------------- #
def run_model_benchmark(
    data_path: Path | str = DEFAULT_DATA_PATH,
    model_names: Optional[Sequence[str]] = None,
    params_path: Optional[Path | str] = DEFAULT_PARAMS_PATH,
    output_results_dir: Path | str = DEFAULT_RESULTS_DIR,
    output_models_dir: Optional[Path | str] = DEFAULT_MODELS_DIR,
    split_day: Optional[int] = None,
    test_split_ratio: float = 0.30,
    seed: int = 42,
    k: int = 20,
    budget_ratio: float = 0.05,
    contamination: Optional[float] = None,
    feature_names: Optional[Sequence[str]] = None,
    stability_seeds: Optional[Sequence[int]] = None,
    stability_sample_size: int = DEFAULT_STABILITY_SAMPLE_SIZE,
    save_models: bool = True,
    experiment_log_path: Optional[Path | str] = DEFAULT_EXPERIMENT_LOG_PATH,
    system_config_path: Optional[Path | str] = DEFAULT_SYSTEM_CONFIG_PATH,
    feature_set: str = "core",
    write_run_manifest: bool = True,
) -> Dict[str, Any]:
    """
    Chạy benchmark label-free: chia theo thời gian, chạy các mô hình, xuất artifact.

    Không có bước nào cho mô hình nhìn thấy tập đánh giá trong lúc fit. Kết quả ghi ra:
    ``benchmark_summary.csv``, ``anomaly_scores.parquet``, ``model_topk_overlap.csv``,
    ``model_stability.csv``.
    """
    params = load_params(params_path)
    names = [str(n) for n in (model_names or available_models())]
    for name in names:
        resolve_model(name)  # fail fast nếu tên sai, kèm danh sách hợp lệ

    results_dir = Path(output_results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== BENCHMARK LABEL-FREE (Tuần 3): %s mô hình, seed=%s, ngân sách=%.1f%% ===", len(names), seed, budget_ratio * 100)
    df = load_feature_matrix(data_path)
    train_df, test_df, split_info = time_split(
        df, day_col="day", split_day=split_day, test_split_ratio=test_split_ratio
    )

    identity_cols = [c for c in ("DomainName", "UserName", "day", "entity_type") if c in test_df.columns]
    scores_frame = test_df.select(identity_cols)
    model_results: Dict[str, Dict[str, Any]] = {}
    summary_rows: List[Dict[str, Any]] = []

    for name in names:
        result = run_single_model(
            name,
            train_df,
            test_df,
            params=params,
            seed=seed,
            k=k,
            budget_ratio=budget_ratio,
            contamination=contamination,
            feature_names=feature_names,
        )
        model_results[name] = result
        summary_rows.append(result["summary"])

        if feature_names is None:
            feature_names = list(result["meta"]["feature_names"])

        scores_frame = scores_frame.with_columns(
            [
                pl.Series(name=f"{name}_score", values=np.round(result["scores"], 6)),
                pl.Series(name=f"{name}_pct", values=np.round(result["rank_pct"], 6)),
                pl.Series(name=f"{name}_anomaly", values=result["labels"].astype(np.int8)),
            ]
        )
        if save_models and output_models_dir:
            result["pipeline"].save(Path(output_models_dir) / f"{name}.joblib")

    # Mức trùng nhau của Top-K giữa các mô hình: chỉ số label-free nói mô hình có "đồng thuận" không.
    overlap_matrix = pd.DataFrame(np.zeros((len(names), len(names))), index=names, columns=names)
    for model_a in names:
        for model_b in names:
            overlap_matrix.loc[model_a, model_b] = topk_overlap(
                model_results[model_a]["scores"], model_results[model_b]["scores"], k
            )

    # Độ ổn định đa seed (đặc biệt cần cho LOF/OCSVM vốn lấy mẫu con để fit).
    seeds = list(stability_seeds) if stability_seeds else [seed]
    stability_rows: List[Dict[str, Any]] = []
    for name in names:
        stats, n_sampled = _stability_scores(
            name, train_df, test_df, params, seeds, contamination, feature_names, stability_sample_size, k
        )
        if stats is None:
            continue
        stability_rows.append(
            {"model": name, "n_seeds": len(seeds), "stability_sample_size": n_sampled, "topk_k": k, **stats}
        )

    summary = pd.DataFrame(summary_rows)
    quantile_cols = [c for c in summary.columns if c.startswith("score_p")]
    summary = summary[SUMMARY_COLUMNS + quantile_cols]
    stability = pd.DataFrame(stability_rows)

    artifacts = _write_artifacts(results_dir, summary, scores_frame, overlap_matrix, stability, split_info)

    # Log thí nghiệm (tương thích 18 cột cũ) + manifest tái lập — artifact phải nằm trong git.
    git_commit = git_state().get("commit")
    split_payload = split_info.to_dict()
    if experiment_log_path:
        log_rows = [
            build_log_row(
                name,
                model_results[name]["scores"],
                model_results[name]["summary"],
                split_payload,
                seed=seed,
                feature_set=feature_set,
                git_commit=git_commit,
            )
            for name in names
        ]
        artifacts["experiment_log"] = str(append_experiment_log(experiment_log_path, log_rows))
    else:
        artifacts["experiment_log"] = None

    manifest: Optional[Dict[str, Any]] = None
    if write_run_manifest:
        manifest = build_manifest(
            data_path=data_path,
            params_path=params_path,
            config_path=system_config_path,
            artifacts=artifacts,
            split=split_payload,
            models=[model_results[name]["meta"] for name in names],
            seed=seed,
            k=k,
            budget_ratio=budget_ratio,
            feature_names=feature_names,
            feature_set=feature_set,
            extra={
                "alert_rate_drift_pp": {
                    name: model_results[name]["summary"]["alert_rate_drift_pp"] for name in names
                },
                "stability_seeds": list(seeds),
                "stability_sample_size": int(stability_sample_size),
            },
        )
        artifacts["run_manifest"] = str(write_manifest(manifest, results_dir / "run_manifest.json"))

    logger.info("Bảng xếp hạng (sắp theo p99 điểm — label-free, KHÔNG phải precision thật):")
    for _, row in summary.sort_values("score_p99", ascending=False).iterrows():
        logger.info(
            "  %-24s n_fit=%-9s alert_rate=%5.2f%%  p99=%9.4f  fit=%6.2fs score=%6.2fs",
            row["model"],
            f"{int(row['n_fit']):,}",
            row["alert_rate_pct"],
            row["score_p99"],
            row["fit_seconds"],
            row["score_seconds"],
        )

    return {
        "split": split_info.to_dict(),
        "summary": summary,
        "scores": scores_frame,
        "overlap": overlap_matrix,
        "stability": stability,
        "models": model_results,
        "artifacts": artifacts,
        "manifest": manifest,
        "feature_names": list(feature_names or []),
    }


# --------------------------------------------------------------------------- #
# Ghi artifact
# --------------------------------------------------------------------------- #
def _write_artifacts(
    results_dir: Path,
    summary: pd.DataFrame,
    scores_frame: pl.DataFrame,
    overlap_matrix: pd.DataFrame,
    stability: pd.DataFrame,
    split_info: SplitInfo,
) -> Dict[str, Optional[str]]:
    """Ghi 4 artifact chính; trả về dict đường dẫn để ``run_manifest.json`` tham chiếu."""
    summary_path = results_dir / "benchmark_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8")
    logger.info("Đã xuất bảng xếp hạng: '%s'.", summary_path)

    scores_path = results_dir / "anomaly_scores.parquet"
    scores_frame.write_parquet(scores_path, compression="snappy")
    logger.info("Đã xuất điểm dị biệt của %s dòng: '%s'.", f"{scores_frame.height:,}", scores_path)

    overlap_path = results_dir / "model_topk_overlap.csv"
    overlap_matrix.to_csv(overlap_path, encoding="utf-8")
    logger.info("Đã xuất ma trận trùng nhau Top-K: '%s'.", overlap_path)

    split_path = results_dir / "split_info.json"
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split_info.to_dict(), f, ensure_ascii=False, indent=2)

    stability_path: Optional[str] = None
    if not stability.empty:
        stability_file = results_dir / "model_stability.csv"
        stability.to_csv(stability_file, index=False, encoding="utf-8")
        logger.info("Đã xuất độ ổn định đa seed: '%s'.", stability_file)
        stability_path = str(stability_file)

    return {
        "benchmark_summary": str(summary_path),
        "anomaly_scores": str(scores_path),
        "model_topk_overlap": str(overlap_path),
        "model_stability": stability_path,
        "split_info": str(split_path),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark các mô hình phát hiện dị biệt UEBA (label-free, chia theo thời gian)"
    )
    parser.add_argument("--data", type=str, default=DEFAULT_DATA_PATH, help="Ma trận đặc trưng parquet")
    parser.add_argument("--models", nargs="+", default=None, help=f"Mặc định: {DEFAULT_MODEL_NAMES}")
    parser.add_argument("--model-params", type=str, default=DEFAULT_PARAMS_PATH, help="File tham số YAML")
    parser.add_argument("--results-dir", type=str, default=DEFAULT_RESULTS_DIR, help="Thư mục ghi artifact")
    parser.add_argument("--models-dir", type=str, default=DEFAULT_MODELS_DIR, help="Thư mục lưu pipeline .joblib")
    parser.add_argument("--split-day", type=int, default=42, help="train = day <= split_day (mặc định 42)")
    parser.add_argument("--split-ratio", type=float, default=0.30, help="Chỉ dùng khi --split-day = -1")
    parser.add_argument("--seed", type=int, default=42, help="Seed của lần chạy chính")
    parser.add_argument("--k", type=int, default=20, help="K cho Top-K / overlap")
    parser.add_argument("--budget-ratio", type=float, default=0.05, help="Ngân sách cảnh báo")
    parser.add_argument("--stability-seeds", nargs="*", type=int, default=None, help="Seed cho kiểm tra ổn định")
    parser.add_argument(
        "--experiment-log", type=str, default=DEFAULT_EXPERIMENT_LOG_PATH, help="CSV log thí nghiệm"
    )
    parser.add_argument(
        "--system-config", type=str, default=DEFAULT_SYSTEM_CONFIG_PATH, help="File cấu hình hệ thống"
    )
    parser.add_argument("--no-run-manifest", action="store_true", help="Không ghi run_manifest.json")
    parser.add_argument("--no-save-models", action="store_true", help="Không lưu pipeline .joblib")
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
    args = _build_arg_parser().parse_args()
    run_model_benchmark(
        data_path=args.data,
        model_names=args.models,
        params_path=args.model_params,
        output_results_dir=args.results_dir,
        output_models_dir=args.models_dir,
        split_day=None if args.split_day is not None and args.split_day < 0 else args.split_day,
        test_split_ratio=args.split_ratio,
        seed=args.seed,
        k=args.k,
        budget_ratio=args.budget_ratio,
        stability_seeds=args.stability_seeds,
        save_models=not args.no_save_models,
        experiment_log_path=args.experiment_log,
        system_config_path=args.system_config,
        write_run_manifest=not args.no_run_manifest,
    )




