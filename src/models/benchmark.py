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

**Phân khúc (Segmented Execution):** khi ``model_params.yaml`` có khối ``segments``, dữ liệu được tách
theo ``segments.column`` (mặc định ``entity_type``) SAU khi chia thời gian. Mỗi phân khúc trong
``enabled`` (Machine, User) chạy toàn bộ luồng trên **chỉ dữ liệu của nó**: imputer/scaler/mô hình fit
riêng trên train của phân khúc, ngưỡng và ``*_pct`` tính trong phân khúc. Các giá trị còn lại
(``ignored``: Admin/Service/System/Other, hoặc giá trị lạ) bị loại khỏi benchmark ML và được log rõ
số dòng / %. Mọi artifact có cột ``segment`` (``"all"`` khi không phân khúc).

**Bộ chỉ số mục 5.4 (cần nhãn):** khi truyền ``labels_path`` (parquet/csv có ``DomainName``, ``UserName``,
``day`` và tuỳ chọn ``label`` 0/1, ``scenario`` — ví dụ ``InjectionPlan.labels()`` của bất thường tiêm),
mỗi dòng của ``benchmark_summary.csv`` có thêm PR-AUC (chỉ số chính), ROC-AUC, Precision@k (10/50/100),
Recall tại ngân sách (theo tỷ lệ và theo N cảnh báo/ngày). Tỷ lệ cảnh báo, số cảnh báo/ngày và thời gian
fit/score luôn được ghi (không cần nhãn). Chỉ số tính TRONG từng phân khúc; phân khúc không có nhãn
dương ghi NaN.
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import polars as pl

from src.evaluation.metrics import (
    DEFAULT_DAILY_BUDGETS,
    DEFAULT_PRECISION_KS,
    alert_rate,
    budget_size,
    labeled_metric_columns,
    labeled_metrics,
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
    "ALL_SEGMENT",
    "DEFAULT_MODEL_PARAMS",
    "NON_FEATURE_COLS",
    "SUMMARY_COLUMNS",
    "SegmentConfig",
    "attach_labels",
    "load_feature_matrix",
    "load_labels",
    "load_model_params",
    "run_model_benchmark",
    "run_single_model",
    "split_segments",
    "train_and_evaluate_model",
]

#: Nhãn phân khúc khi KHÔNG phân khúc (một mô hình chung cho mọi dòng).
ALL_SEGMENT = "all"

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
        "M": 32,
        "ef_construction": 200,
        "ef": 200,
        "contamination": 0.05,
        "n_jobs": -1,
    },
    "one_class_svm": {
        "n_components": 300,
        "gamma": "scale",
        "nu": 0.05,
        "max_iter": 1000,
        "tol": 0.001,
        "learning_rate": "optimal",
    },
}

#: Cột danh tính/không phải đặc trưng — giữ để tài liệu hoá lý do loại bỏ khỏi ma trận mô hình.
NON_FEATURE_COLS = ["DomainName", "UserName", "day", "entity_type", "total_logons", "user_key"]

#: Các phân vị của điểm ghi vào ``benchmark_summary.csv`` (gồm p25/p95/p99 để đối chiếu với log cũ).
SUMMARY_QUANTILE_PROBS: Tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)

#: Thứ tự cột của ``benchmark_summary.csv`` (một dòng cho mỗi mô hình).
SUMMARY_COLUMNS = [
    "model",
    "segment",
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
    "alerts_per_day_mean",
    "budget_ratio",
    "budget_n",
    "topk_k",
    "fit_seconds",
    "score_seconds",
    "seed",
    "sklearn_version",
]


# --------------------------------------------------------------------------- #
# Phân khúc (segment)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SegmentConfig:
    """Cấu hình phân khúc — khối ``segments`` của ``configs/model_params.yaml``."""

    column: str
    enabled: Tuple[str, ...]
    ignored: Tuple[str, ...] = ()

    @classmethod
    def from_params(cls, params: Optional[Dict[str, Any]]) -> Optional["SegmentConfig"]:
        """Đọc khối ``segments``; trả None nếu không có hoặc ``enabled`` rỗng (= không phân khúc)."""
        block = (params or {}).get("segments")
        if not block:
            return None
        if not isinstance(block, dict):
            raise ValueError("Khối 'segments' phải là mapping {column, enabled, ignored}.")
        enabled = tuple(str(v) for v in (block.get("enabled") or []))
        if not enabled:
            return None
        ignored = tuple(str(v) for v in (block.get("ignored") or []))
        repeated = sorted({v for v in enabled if enabled.count(v) > 1})
        if repeated:
            raise ValueError(f"segments.enabled bị lặp: {repeated}.")
        both = sorted(set(enabled) & set(ignored))
        if both:
            raise ValueError(f"Giá trị vừa nằm trong segments.enabled vừa trong segments.ignored: {both}.")
        return cls(column=str(block.get("column") or "entity_type"), enabled=enabled, ignored=ignored)

    def to_dict(self) -> Dict[str, Any]:
        return {"column": self.column, "enabled": list(self.enabled), "ignored": list(self.ignored)}


def split_segments(
    df: pl.DataFrame, config: SegmentConfig, context: str = "dữ liệu"
) -> Tuple[Dict[str, pl.DataFrame], Dict[str, Any]]:
    """
    Tách ``df`` thành một DataFrame cho mỗi phân khúc ``config.enabled``; mọi dòng khác bị loại.

    Dòng bị loại gồm ``config.ignored`` (bỏ qua có chủ đích) và giá trị không khai báo / NULL
    (vẫn bị loại nhưng có cảnh báo). Trả ``(parts, stats)``; ``stats`` đi vào ``run_manifest.json``.
    """
    if config.column not in df.columns:
        raise ValueError(f"Thiếu cột phân khúc '{config.column}' trong {context}.")
    n_rows = df.height
    counts = {
        ("null" if value is None else str(value)): int(count)
        for value, count in df.group_by(config.column).len().iter_rows()
    }

    parts: Dict[str, pl.DataFrame] = {}
    for segment in config.enabled:
        part = df.filter(pl.col(config.column) == segment)
        if part.height == 0:
            raise ValueError(f"Phân khúc '{segment}' không có dòng nào trong {context}.")
        parts[segment] = part
        logger.info(
            "[segment] %s — %s: %s dòng (%.2f%%).", context, segment, f"{part.height:,}", 100 * part.height / n_rows
        )

    bypassed = {value: count for value, count in sorted(counts.items()) if value not in config.enabled}
    n_bypassed = sum(bypassed.values())
    unknown = [value for value in bypassed if value not in config.ignored]
    logger.info(
        "[segment] %s — LOẠI khỏi benchmark ML: %s dòng (%.2f%%) %s",
        context,
        f"{n_bypassed:,}",
        100 * n_bypassed / n_rows if n_rows else 0.0,
        bypassed,
    )
    if unknown:
        logger.warning(
            "[segment] %s — giá trị '%s' không nằm trong segments.enabled/ignored: %s (bị loại).",
            context,
            config.column,
            {value: bypassed[value] for value in unknown},
        )

    stats = {
        "n_rows": n_rows,
        "segments": {segment: part.height for segment, part in parts.items()},
        "bypassed": bypassed,
        "n_bypassed": n_bypassed,
        "bypassed_pct": round(100 * n_bypassed / n_rows, 4) if n_rows else 0.0,
        "unknown_values": unknown,
    }
    return parts, stats


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


#: Khoá nối nhãn với ma trận đặc trưng (một dòng = một tài khoản × một ngày).
LABEL_KEYS = ["DomainName", "UserName", "day"]


def load_labels(labels_path: Path | str) -> pl.DataFrame:
    """
    Đọc file nhãn (parquet hoặc csv): mỗi dòng là một (tài khoản, ngày) DỊ BIỆT.

    Cột bắt buộc ``DomainName``, ``UserName``, ``day``; ``label`` (0/1) tuỳ chọn — thiếu thì mọi dòng
    được coi là 1; ``scenario`` tuỳ chọn (loại bất thường, giữ lại để phân tích theo kịch bản).
    """
    path = Path(labels_path)
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file nhãn '{path}'.")
    df = pl.read_csv(path) if path.suffix.lower() == ".csv" else pl.read_parquet(path)
    missing = [c for c in LABEL_KEYS if c not in df.columns]
    if missing:
        raise ValueError(f"File nhãn '{path}' thiếu cột {missing}.")
    if "label" not in df.columns:
        df = df.with_columns(pl.lit(1).alias("label"))
    df = df.with_columns(pl.col("day").cast(pl.Int64), pl.col("label").cast(pl.Int8))
    bad = sorted(set(df["label"].unique().to_list()) - {0, 1})
    if bad:
        raise ValueError(f"Cột 'label' của '{path}' chỉ được chứa 0/1, gặp {bad}.")
    keep = LABEL_KEYS + ["label"] + (["scenario"] if "scenario" in df.columns else [])
    df = df.select(keep).unique(subset=LABEL_KEYS, keep="first", maintain_order=True)
    logger.info("Đã nạp nhãn '%s': %s dòng (%s dương).", path, f"{df.height:,}", f"{int(df['label'].sum()):,}")
    return df


def attach_labels(eval_df: pl.DataFrame, labels: pl.DataFrame) -> np.ndarray:
    """Mảng 0/1 khớp từng dòng của ``eval_df`` (dòng không có trong ``labels`` = 0)."""
    joined = eval_df.select(LABEL_KEYS).with_columns(pl.col("day").cast(pl.Int64)).join(
        labels.select(LABEL_KEYS + ["label"]), on=LABEL_KEYS, how="left", maintain_order="left"
    )
    return joined["label"].fill_null(0).to_numpy().astype(np.int8)


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
    segment: str = ALL_SEGMENT,
    eval_labels: Optional[np.ndarray] = None,
    precision_ks: Sequence[int] = DEFAULT_PRECISION_KS,
    daily_budgets: Sequence[int] = DEFAULT_DAILY_BUDGETS,
) -> Dict[str, Any]:
    """
    Fit 1 mô hình trên ``train_df`` rồi chấm điểm ``test_df`` (không bao giờ chấm điểm tập train).

    ``eval_labels`` (0/1, khớp từng dòng ``test_df``) bật bộ chỉ số cần nhãn của mục 5.4
    (xem ``labeled_metrics``); ``None`` ⇒ chỉ ghi chỉ số label-free.

    ``segment`` chỉ là nhãn ghi vào ``summary``: lọc dữ liệu theo phân khúc là việc của người gọi
    (``run_model_benchmark``), nên imputer/scaler/mô hình ở đây fit trên đúng ``train_df`` được truyền vào.

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
    test_days = test_df["day"].to_numpy() if "day" in test_df.columns else None
    n_test_days = int(np.unique(test_days).size) if test_days is not None else 1
    summary: Dict[str, Any] = {
        "model": name,
        "segment": segment,
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
        # Gánh nặng vận hành: số cảnh báo trung bình mỗi ngày của tập đánh giá (theo ngưỡng fit trên train).
        "alerts_per_day_mean": round(int(flags.sum()) / n_test_days, 2),
        "budget_ratio": float(budget_ratio),
        "budget_n": budget_size(test_df.height, budget_ratio),
        "topk_k": int(k),
        "fit_seconds": round(fit_seconds, 3),
        "score_seconds": round(score_seconds, 3),
        "seed": int(seed),
        "sklearn_version": SKLEARN_VERSION,
    }
    if eval_labels is not None:
        summary.update(
            labeled_metrics(
                eval_labels,
                scores,
                days=test_days,
                ks=precision_ks,
                budget_ratio=budget_ratio,
                daily_budgets=daily_budgets,
            )
        )
    summary.update(
        {f"score_{key}": value for key, value in score_quantiles(scores, probs=SUMMARY_QUANTILE_PROBS).items()}
    )

    logger.info(
        "[%s/%s] n_fit=%s (tập train %s dòng), n_eval=%s, alert_rate=%.2f%% (ngân sách %.1f%%), "
        "threshold=%.6g, fit=%.1fs, score=%.1fs",
        segment,
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
    if eval_labels is not None:
        logger.info(
            "[%s/%s] nhãn dương=%d | PR-AUC=%.4f ROC-AUC=%.4f %s",
            segment,
            name,
            int(summary["n_positive"]),
            summary["pr_auc"],
            summary["roc_auc"],
            " ".join(f"P@{int(k)}={summary[f'precision_at_{int(k)}']:.3f}" for k in precision_ks),
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
    use_segments: bool = True,
    labels_path: Optional[Path | str] = None,
    precision_ks: Sequence[int] = DEFAULT_PRECISION_KS,
    daily_budgets: Sequence[int] = DEFAULT_DAILY_BUDGETS,
) -> Dict[str, Any]:
    """
    Chạy benchmark label-free: chia theo thời gian, tách phân khúc, chạy các mô hình, xuất artifact.

    Không có bước nào cho mô hình nhìn thấy tập đánh giá trong lúc fit. Kết quả ghi ra:
    ``benchmark_summary.csv``, ``anomaly_scores.parquet``, ``model_topk_overlap.csv``,
    ``model_stability.csv`` — mỗi file có cột ``segment``.

    ``use_segments=False`` bỏ qua khối ``segments`` của YAML (một mô hình chung, ``segment = "all"``).
    Mô hình được lưu ở ``<output_models_dir>/<segment>/<model>.joblib`` (``all/`` khi không phân khúc).

    ``labels_path`` (tuỳ chọn) bật bộ chỉ số cần nhãn mục 5.4: PR-AUC, ROC-AUC, Precision@``precision_ks``,
    Recall@``budget_ratio`` và Recall@N cảnh báo/ngày (``daily_budgets``).
    """
    params = load_params(params_path)
    names = [str(n) for n in (model_names or available_models())]
    for name in names:
        resolve_model(name)  # fail fast nếu tên sai, kèm danh sách hợp lệ
    segment_config = SegmentConfig.from_params(params) if use_segments else None

    results_dir = Path(output_results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "=== BENCHMARK LABEL-FREE: %s mô hình, seed=%s, ngân sách=%.1f%%, phân khúc=%s ===",
        len(names),
        seed,
        budget_ratio * 100,
        list(segment_config.enabled) if segment_config else ALL_SEGMENT,
    )
    df = load_feature_matrix(data_path)
    train_df, test_df, split_info = time_split(
        df, day_col="day", split_day=split_day, test_split_ratio=test_split_ratio
    )
    labels_df = load_labels(labels_path) if labels_path else None
    label_stats: Optional[Dict[str, Any]] = None
    if labels_df is not None:
        positives = labels_df.filter(pl.col("label") == 1)
        n_in_test = int(attach_labels(test_df, positives).sum())
        n_in_train = int(attach_labels(train_df, positives).sum())
        label_stats = {
            "labels_path": str(labels_path),
            "n_positive_rows": positives.height,
            "n_positive_in_test": n_in_test,
            "n_positive_in_train": n_in_train,
            "n_positive_unmatched": positives.height - n_in_test - n_in_train,
        }
        if n_in_train:
            logger.warning(
                "%d dòng nhãn dương rơi vào tập TRAIN — không được đánh giá và có thể đã làm bẩn dữ liệu huấn luyện.",
                n_in_train,
            )
        if label_stats["n_positive_unmatched"]:
            logger.warning(
                "%d dòng nhãn dương không khớp dòng nào của ma trận (sai khoá DomainName/UserName/day?).",
                label_stats["n_positive_unmatched"],
            )
        logger.info("Nhãn dương trong tập test: %d / %s dòng.", n_in_test, f"{test_df.height:,}")

    # Chia thời gian TRƯỚC, tách phân khúc SAU ⇒ mọi phân khúc dùng chung ranh giới ngày.
    segment_stats: Optional[Dict[str, Any]] = None
    if segment_config is not None:
        train_parts, train_stats = split_segments(train_df, segment_config, context="train")
        test_parts, test_stats = split_segments(test_df, segment_config, context="test")
        populations = [(seg, train_parts[seg], test_parts[seg]) for seg in segment_config.enabled]
        segment_stats = {**segment_config.to_dict(), "train": train_stats, "test": test_stats}
    else:
        populations = [(ALL_SEGMENT, train_df, test_df)]

    seeds = list(stability_seeds) if stability_seeds else [seed]
    model_results: Dict[str, Dict[str, Dict[str, Any]]] = {}
    summary_rows: List[Dict[str, Any]] = []
    score_frames: List[pl.DataFrame] = []
    overlap_frames: List[pd.DataFrame] = []
    stability_rows: List[Dict[str, Any]] = []

    for segment, seg_train, seg_test in populations:
        models_dir: Optional[Path] = None
        if save_models and output_models_dir:
            models_dir = Path(output_models_dir) / segment
        run = _run_population(
            segment,
            seg_train,
            seg_test,
            names,
            params=params,
            seed=seed,
            k=k,
            budget_ratio=budget_ratio,
            contamination=contamination,
            feature_names=feature_names,
            stability_seeds=seeds,
            stability_sample_size=stability_sample_size,
            models_dir=models_dir,
            labels_df=labels_df,
            precision_ks=precision_ks,
            daily_budgets=daily_budgets,
        )
        feature_names = run["feature_names"]
        model_results[segment] = run["results"]
        summary_rows.extend(run["summary_rows"])
        score_frames.append(run["scores"])
        overlap_frames.append(run["overlap"])
        stability_rows.extend(run["stability_rows"])

    summary = pd.DataFrame(summary_rows)
    quantile_cols = [c for c in summary.columns if c.startswith("score_p")]
    metric_cols = labeled_metric_columns(precision_ks, daily_budgets) if labels_df is not None else []
    summary = summary[SUMMARY_COLUMNS + metric_cols + quantile_cols]
    scores_frame = pl.concat(score_frames, how="vertical")
    overlap = pd.concat(overlap_frames, ignore_index=True)
    stability = pd.DataFrame(stability_rows)

    artifacts = _write_artifacts(results_dir, summary, scores_frame, overlap, stability, split_info)

    # Log thí nghiệm (tương thích 18 cột cũ) + manifest tái lập — artifact phải nằm trong git.
    git_commit = git_state().get("commit")
    split_payload = split_info.to_dict()
    run_keys = [(segment, name) for segment, _, _ in populations for name in names]
    if experiment_log_path:
        log_rows = [
            build_log_row(
                name,
                model_results[segment][name]["scores"],
                model_results[segment][name]["summary"],
                split_payload,
                seed=seed,
                feature_set=feature_set,
                git_commit=git_commit,
                segment=segment,
            )
            for segment, name in run_keys
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
            models=[{"segment": segment, **model_results[segment][name]["meta"]} for segment, name in run_keys],
            seed=seed,
            k=k,
            budget_ratio=budget_ratio,
            feature_names=feature_names,
            feature_set=feature_set,
            extra={
                "segments": segment_stats,
                "alert_rate_drift_pp": {
                    f"{segment}/{name}": model_results[segment][name]["summary"]["alert_rate_drift_pp"]
                    for segment, name in run_keys
                },
                "labels": label_stats,
                "precision_ks": [int(k) for k in precision_ks] if labels_df is not None else None,
                "daily_budgets": [int(n) for n in daily_budgets] if labels_df is not None else None,
                "stability_seeds": list(seeds),
                "stability_sample_size": int(stability_sample_size),
            },
        )
        artifacts["run_manifest"] = str(write_manifest(manifest, results_dir / "run_manifest.json"))

    if labels_df is not None:
        logger.info("Bảng xếp hạng theo PR-AUC (chỉ số chính mục 5.4 — nhãn tổng hợp, CẬN TRÊN lạc quan):")
        for _, row in summary.sort_values(["segment", "pr_auc"], ascending=[True, False]).iterrows():
            logger.info(
                "  %-8s %-24s PR-AUC=%.4f ROC-AUC=%.4f R@budget=%.3f alert_rate=%5.2f%%",
                row["segment"],
                row["model"],
                row["pr_auc"],
                row["roc_auc"],
                row["recall_at_budget"],
                row["alert_rate_pct"],
            )
    logger.info("Bảng xếp hạng (sắp theo p99 điểm — label-free, KHÔNG phải precision thật):")
    for _, row in summary.sort_values(["segment", "score_p99"], ascending=[True, False]).iterrows():
        logger.info(
            "  %-8s %-24s n_fit=%-9s alert_rate=%5.2f%%  p99=%9.4f  fit=%6.2fs score=%6.2fs",
            row["segment"],
            row["model"],
            f"{int(row['n_fit']):,}",
            row["alert_rate_pct"],
            row["score_p99"],
            row["fit_seconds"],
            row["score_seconds"],
        )

    return {
        "split": split_info.to_dict(),
        "segments": segment_stats,
        "summary": summary,
        "scores": scores_frame,
        "overlap": overlap,
        "stability": stability,
        "models": model_results,
        "artifacts": artifacts,
        "manifest": manifest,
        "feature_names": list(feature_names or []),
    }


def _run_population(
    segment: str,
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    names: Sequence[str],
    *,
    params: Optional[Dict[str, Any]],
    seed: int,
    k: int,
    budget_ratio: float,
    contamination: Optional[float],
    feature_names: Optional[Sequence[str]],
    stability_seeds: Sequence[int],
    stability_sample_size: int,
    models_dir: Optional[Path],
    labels_df: Optional[pl.DataFrame] = None,
    precision_ks: Sequence[int] = DEFAULT_PRECISION_KS,
    daily_budgets: Sequence[int] = DEFAULT_DAILY_BUDGETS,
) -> Dict[str, Any]:
    """
    Chạy mọi mô hình trên MỘT quần thể (một phân khúc, hoặc toàn bộ dữ liệu khi ``segment = "all"``).

    Mỗi mô hình có pipeline riêng ⇒ imputer/scaler fit trên ``train_df`` của chính quần thể này.
    ``*_pct`` và Top-K overlap tính trong quần thể: điểm của các phân khúc khác nhau không cùng thang đo.
    """
    identity_cols = [c for c in ("DomainName", "UserName", "day", "entity_type") if c in test_df.columns]
    scores_frame = test_df.select(identity_cols).with_columns(pl.lit(segment).alias("segment"))
    eval_labels = attach_labels(test_df, labels_df) if labels_df is not None else None
    if eval_labels is not None:
        scores_frame = scores_frame.with_columns(pl.Series(name="label", values=eval_labels))
    results: Dict[str, Dict[str, Any]] = {}
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
            segment=segment,
            eval_labels=eval_labels,
            precision_ks=precision_ks,
            daily_budgets=daily_budgets,
        )
        results[name] = result
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
        if models_dir is not None:
            result["pipeline"].save(models_dir / f"{name}.joblib")

    # Mức trùng nhau của Top-K giữa các mô hình: chỉ số label-free nói mô hình có "đồng thuận" không.
    overlap = pd.DataFrame(
        [[topk_overlap(results[a]["scores"], results[b]["scores"], k) for b in names] for a in names],
        columns=list(names),
    )
    overlap.insert(0, "model", list(names))
    overlap.insert(0, "segment", segment)

    # Độ ổn định đa seed: chạy lại pipeline với từng seed trên mẫu con của tập test của quần thể.
    stability_rows: List[Dict[str, Any]] = []
    for name in names:
        stats, n_sampled = _stability_scores(
            name, train_df, test_df, params, stability_seeds, contamination, feature_names, stability_sample_size, k
        )
        if stats is None:
            continue
        stability_rows.append(
            {
                "model": name,
                "segment": segment,
                "n_seeds": len(stability_seeds),
                "stability_sample_size": n_sampled,
                "topk_k": k,
                **stats,
            }
        )

    return {
        "results": results,
        "summary_rows": summary_rows,
        "scores": scores_frame,
        "overlap": overlap,
        "stability_rows": stability_rows,
        "feature_names": feature_names,
    }


# --------------------------------------------------------------------------- #
# Ghi artifact
# --------------------------------------------------------------------------- #
def _write_artifacts(
    results_dir: Path,
    summary: pd.DataFrame,
    scores_frame: pl.DataFrame,
    overlap: pd.DataFrame,
    stability: pd.DataFrame,
    split_info: SplitInfo,
) -> Dict[str, Optional[str]]:
    """
    Ghi 4 artifact chính; trả về dict đường dẫn để ``run_manifest.json`` tham chiếu.

    ``model_topk_overlap.csv``: một ma trận mô hình × mô hình cho mỗi phân khúc, xếp chồng theo cột
    ``segment`` (cột ``model`` là hàng của ma trận).
    """
    summary_path = results_dir / "benchmark_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8")
    logger.info("Đã xuất bảng xếp hạng: '%s'.", summary_path)

    scores_path = results_dir / "anomaly_scores.parquet"
    scores_frame.write_parquet(scores_path, compression="snappy")
    logger.info("Đã xuất điểm dị biệt của %s dòng: '%s'.", f"{scores_frame.height:,}", scores_path)

    overlap_path = results_dir / "model_topk_overlap.csv"
    overlap.to_csv(overlap_path, index=False, encoding="utf-8")
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
    parser.add_argument(
        "--no-segments", action="store_true", help="Bỏ qua khối 'segments' của YAML: một mô hình chung cho mọi dòng"
    )
    parser.add_argument(
        "--labels", type=str, default=None,
        help="File nhãn (DomainName, UserName, day[, label, scenario]) — bật bộ chỉ số mục 5.4",
    )
    parser.add_argument(
        "--precision-ks", nargs="+", type=int, default=list(DEFAULT_PRECISION_KS), help="K cho Precision@k"
    )
    parser.add_argument(
        "--daily-budgets", nargs="+", type=int, default=list(DEFAULT_DAILY_BUDGETS),
        help="Số cảnh báo/ngày cho Recall tại ngân sách ngày",
    )
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
        use_segments=not args.no_segments,
        labels_path=args.labels,
        precision_ks=args.precision_ks,
        daily_budgets=args.daily_budgets,
    )




